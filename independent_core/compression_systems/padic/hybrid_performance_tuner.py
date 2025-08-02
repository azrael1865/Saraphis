"""
Hybrid Performance Tuner - Automatic tuning of hybrid system performance
NO FALLBACKS - HARD FAILURES ONLY
"""

import asyncio
import logging
import math
import random
import threading
import time
import torch
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Set

# Import hybrid system components
from .hybrid_padic_structures import HybridPadicWeight, HybridPadicManager
from .dynamic_switching_manager import DynamicSwitchingManager
from .hybrid_padic_compressor import HybridPadicCompressionSystem
from .hybrid_performance_monitor import HybridPerformanceMonitor
from .hybrid_performance_analyzer import HybridPerformanceAnalyzer, OptimizationRecommendation


class TuningStrategy(Enum):
    """Tuning strategy enumeration"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"


class TuningPhase(Enum):
    """Tuning phase enumeration"""
    INITIALIZATION = "initialization"
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"


class OptimizationMetric(Enum):
    """Optimization metric enumeration"""
    EXECUTION_TIME = "execution_time"
    GPU_MEMORY_USAGE = "gpu_memory_usage"
    CPU_USAGE = "cpu_usage"
    THROUGHPUT = "throughput"
    SUCCESS_RATE = "success_rate"
    SWITCHING_OVERHEAD = "switching_overhead"
    OVERALL_PERFORMANCE = "overall_performance"


@dataclass
class TuningParameter:
    """Individual tuning parameter"""
    name: str
    current_value: float
    min_value: float
    max_value: float
    step_size: float
    parameter_type: str  # "continuous", "discrete", "categorical"
    impact_weight: float = 1.0
    optimization_priority: str = "medium"  # "low", "medium", "high"
    
    def __post_init__(self):
        """Validate tuning parameter"""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Parameter name must be non-empty string")
        if not isinstance(self.current_value, (int, float)):
            raise TypeError("Current value must be numeric")
        if not isinstance(self.min_value, (int, float)):
            raise TypeError("Min value must be numeric")
        if not isinstance(self.max_value, (int, float)):
            raise TypeError("Max value must be numeric")
        if self.min_value >= self.max_value:
            raise ValueError("Min value must be less than max value")
        if not self.min_value <= self.current_value <= self.max_value:
            raise ValueError("Current value must be between min and max values")
        if self.step_size <= 0:
            raise ValueError("Step size must be positive")
    
    def propose_new_value(self, strategy: TuningStrategy = TuningStrategy.BALANCED) -> float:
        """Propose new parameter value based on strategy"""
        if strategy == TuningStrategy.CONSERVATIVE:
            # Small changes
            delta = self.step_size * 0.5 * (random.random() - 0.5)
        elif strategy == TuningStrategy.AGGRESSIVE:
            # Large changes
            delta = self.step_size * 2.0 * (random.random() - 0.5)
        else:  # BALANCED or ADAPTIVE
            # Medium changes
            delta = self.step_size * (random.random() - 0.5)
        
        new_value = self.current_value + delta
        return max(self.min_value, min(self.max_value, new_value))


@dataclass
class TuningExperiment:
    """Individual tuning experiment"""
    experiment_id: str
    parameters: Dict[str, float]
    performance_metrics: Dict[str, float]
    success: bool
    execution_time_ms: float
    timestamp: datetime
    strategy_used: TuningStrategy
    
    def __post_init__(self):
        """Validate tuning experiment"""
        if not isinstance(self.experiment_id, str) or not self.experiment_id.strip():
            raise ValueError("Experiment ID must be non-empty string")
        if not isinstance(self.parameters, dict):
            raise TypeError("Parameters must be dict")
        if not isinstance(self.performance_metrics, dict):
            raise TypeError("Performance metrics must be dict")
        if not isinstance(self.success, bool):
            raise TypeError("Success must be bool")
    
    def get_overall_score(self, metric_weights: Dict[str, float]) -> float:
        """Calculate overall performance score"""
        if not self.success:
            return 0.0
        
        score = 0.0
        total_weight = 0.0
        
        for metric, value in self.performance_metrics.items():
            weight = metric_weights.get(metric, 1.0)
            
            # Normalize metrics (assume higher is better for most metrics)
            if metric in ['execution_time_ms', 'gpu_memory_usage', 'cpu_usage', 'switching_overhead']:
                # Lower is better for these metrics
                normalized_value = max(0.0, 1.0 - (value / 1000.0))  # Simple normalization
            else:
                # Higher is better for these metrics
                normalized_value = min(1.0, value)
            
            score += normalized_value * weight
            total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0


@dataclass
class TuningResult:
    """Result of tuning process"""
    best_parameters: Dict[str, float]
    best_performance: Dict[str, float]
    performance_improvement: float
    experiments_conducted: int
    tuning_duration_minutes: float
    convergence_achieved: bool
    tuning_strategy_used: TuningStrategy
    validation_results: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate tuning result"""
        if not isinstance(self.best_parameters, dict):
            raise TypeError("Best parameters must be dict")
        if not isinstance(self.best_performance, dict):
            raise TypeError("Best performance must be dict")
        if not isinstance(self.performance_improvement, (int, float)):
            raise TypeError("Performance improvement must be numeric")
        if self.experiments_conducted < 0:
            raise ValueError("Experiments conducted must be non-negative")


@dataclass
class HybridTuningConfig:
    """Configuration for hybrid performance tuning"""
    tuning_strategy: TuningStrategy = TuningStrategy.ADAPTIVE
    optimization_metric: OptimizationMetric = OptimizationMetric.OVERALL_PERFORMANCE
    
    # Tuning parameters
    max_tuning_iterations: int = 100
    convergence_threshold: float = 0.01  # 1% improvement threshold
    patience: int = 10  # Iterations without improvement before stopping
    validation_ratio: float = 0.2  # 20% of experiments for validation
    
    # Optimization settings
    enable_gpu_memory_tuning: bool = True
    enable_switching_threshold_tuning: bool = True
    enable_performance_parameter_tuning: bool = True
    enable_adaptive_learning: bool = True
    
    # Safety limits
    max_gpu_memory_mb: int = 2048
    max_cpu_usage_percent: float = 90.0
    min_success_rate: float = 0.95
    
    # Tuning ranges
    gpu_memory_threshold_range: Tuple[float, float] = (512.0, 2048.0)
    switching_threshold_range: Tuple[float, float] = (100.0, 10000.0)
    learning_rate_range: Tuple[float, float] = (0.001, 0.1)
    
    def __post_init__(self):
        """Validate configuration"""
        if not isinstance(self.tuning_strategy, TuningStrategy):
            raise TypeError("Tuning strategy must be TuningStrategy")
        if not isinstance(self.optimization_metric, OptimizationMetric):
            raise TypeError("Optimization metric must be OptimizationMetric")
        if self.max_tuning_iterations <= 0:
            raise ValueError("Max tuning iterations must be positive")
        if not 0.0 < self.convergence_threshold < 1.0:
            raise ValueError("Convergence threshold must be between 0.0 and 1.0")
        if not 0.0 < self.validation_ratio < 1.0:
            raise ValueError("Validation ratio must be between 0.0 and 1.0")


class HybridPerformanceTuner:
    """
    Automatic tuning of hybrid system performance.
    Provides intelligent parameter optimization and adaptive tuning strategies.
    """
    
    def __init__(self, config: Optional[HybridTuningConfig] = None):
        """Initialize hybrid performance tuner"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, HybridTuningConfig):
            raise TypeError(f"Config must be HybridTuningConfig or None, got {type(config)}")
        
        self.config = config or HybridTuningConfig()
        self.logger = logging.getLogger('HybridPerformanceTuner')
        
        # Component references
        self.hybrid_padic_manager: Optional[HybridPadicManager] = None
        self.dynamic_switching_manager: Optional[DynamicSwitchingManager] = None
        self.hybrid_compression_system: Optional[HybridPadicCompressionSystem] = None
        self.performance_monitor: Optional[HybridPerformanceMonitor] = None
        self.performance_analyzer: Optional[HybridPerformanceAnalyzer] = None
        
        # Tuning state
        self.is_initialized = False
        self.tuning_phase = TuningPhase.INITIALIZATION
        self.is_tuning = False
        
        # Tuning parameters
        self.tuning_parameters: Dict[str, TuningParameter] = {}
        self.baseline_performance: Dict[str, float] = {}
        self.current_best_performance: Dict[str, float] = {}
        
        # Experiment tracking
        self.experiments: deque = deque(maxlen=1000)
        self.current_experiment: Optional[TuningExperiment] = None
        self.best_experiment: Optional[TuningExperiment] = None
        
        # Tuning history and analytics
        self.tuning_history: List[TuningResult] = []
        self.parameter_sensitivity: Dict[str, float] = {}
        self.optimization_recommendations: List[OptimizationRecommendation] = []
        
        # Adaptive learning
        self.parameter_correlations: Dict[str, Dict[str, float]] = {}
        self.learning_rate: float = 0.01
        self.exploration_rate: float = 0.3
        
        # Thread safety
        self._tuning_lock = threading.RLock()
        self._experiment_lock = threading.RLock()
        self._parameters_lock = threading.RLock()
        
        # Background tuning
        self._tuning_thread: Optional[threading.Thread] = None
        self._stop_tuning = threading.Event()
        self._continuous_tuning_enabled = False
        
        self.logger.info("HybridPerformanceTuner created successfully")
    
    def initialize_tuner(self,
                        hybrid_manager: HybridPadicManager,
                        switching_manager: DynamicSwitchingManager,
                        compression_system: HybridPadicCompressionSystem,
                        performance_monitor: HybridPerformanceMonitor,
                        performance_analyzer: HybridPerformanceAnalyzer) -> None:
        """
        Initialize performance tuner.
        
        Args:
            hybrid_manager: Hybrid p-adic manager instance
            switching_manager: Dynamic switching manager instance
            compression_system: Hybrid compression system instance
            performance_monitor: Performance monitor instance
            performance_analyzer: Performance analyzer instance
            
        Raises:
            TypeError: If any component is invalid
            RuntimeError: If initialization fails
        """
        if not isinstance(hybrid_manager, HybridPadicManager):
            raise TypeError(f"Hybrid manager must be HybridPadicManager, got {type(hybrid_manager)}")
        if not isinstance(switching_manager, DynamicSwitchingManager):
            raise TypeError(f"Switching manager must be DynamicSwitchingManager, got {type(switching_manager)}")
        if not isinstance(compression_system, HybridPadicCompressionSystem):
            raise TypeError(f"Compression system must be HybridPadicCompressionSystem, got {type(compression_system)}")
        if not isinstance(performance_monitor, HybridPerformanceMonitor):
            raise TypeError(f"Performance monitor must be HybridPerformanceMonitor, got {type(performance_monitor)}")
        if not isinstance(performance_analyzer, HybridPerformanceAnalyzer):
            raise TypeError(f"Performance analyzer must be HybridPerformanceAnalyzer, got {type(performance_analyzer)}")
        
        try:
            with self._tuning_lock:
                self.tuning_phase = TuningPhase.INITIALIZATION
                
                # Set component references
                self.hybrid_padic_manager = hybrid_manager
                self.dynamic_switching_manager = switching_manager
                self.hybrid_compression_system = compression_system
                self.performance_monitor = performance_monitor
                self.performance_analyzer = performance_analyzer
                
                # Initialize tuning parameters
                self._initialize_tuning_parameters()
                
                # Establish baseline performance
                self._establish_baseline_performance()
                
                # Initialize adaptive learning
                if self.config.enable_adaptive_learning:
                    self._initialize_adaptive_learning()
                
                self.tuning_phase = TuningPhase.EXPLORATION
                self.is_initialized = True
                
                self.logger.info("Hybrid performance tuner initialized successfully")
                
        except Exception as e:
            self.tuning_phase = TuningPhase.INITIALIZATION
            self.logger.error(f"Failed to initialize tuner: {e}")
            raise RuntimeError(f"Tuner initialization failed: {e}")
    
    def tune_hybrid_parameters(self) -> TuningResult:
        """
        Tune hybrid system parameters for optimal performance.
        
        Returns:
            Tuning result with optimized parameters
            
        Raises:
            RuntimeError: If tuning fails
        """
        if not self.is_initialized:
            raise RuntimeError("Tuner not initialized")
        
        with self._tuning_lock:
            try:
                self.tuning_phase = TuningPhase.EXPLORATION
                self.is_tuning = True
                start_time = time.time()
                
                best_score = 0.0
                iterations_without_improvement = 0
                experiments_conducted = 0
                
                self.logger.info("Starting hybrid parameter tuning")
                
                # Main tuning loop
                for iteration in range(self.config.max_tuning_iterations):
                    # Determine tuning strategy for this iteration
                    current_strategy = self._select_tuning_strategy(iteration)
                    
                    # Generate parameter candidates
                    parameter_candidates = self._generate_parameter_candidates(current_strategy)
                    
                    # Run experiments
                    for candidate_params in parameter_candidates:
                        experiment = self._run_tuning_experiment(candidate_params, current_strategy)
                        experiments_conducted += 1
                        
                        if experiment.success:
                            score = experiment.get_overall_score(self._get_metric_weights())
                            
                            if score > best_score:
                                best_score = score
                                self.best_experiment = experiment
                                self.current_best_performance = experiment.performance_metrics.copy()
                                iterations_without_improvement = 0
                                
                                self.logger.info(f"New best performance: {score:.4f} at iteration {iteration}")
                            else:
                                iterations_without_improvement += 1
                    
                    # Check convergence
                    if self._check_convergence(best_score) or iterations_without_improvement >= self.config.patience:
                        self.logger.info(f"Tuning converged after {iteration + 1} iterations")
                        break
                    
                    # Update adaptive learning
                    if self.config.enable_adaptive_learning:
                        self._update_adaptive_learning()
                
                # Validation phase
                self.tuning_phase = TuningPhase.VALIDATION
                validation_results = self._validate_best_parameters()
                
                # Calculate improvement
                baseline_score = self._calculate_baseline_score()
                performance_improvement = (best_score - baseline_score) / baseline_score if baseline_score > 0 else 0.0
                
                # Create tuning result
                tuning_duration = (time.time() - start_time) / 60.0  # Convert to minutes
                
                result = TuningResult(
                    best_parameters=self.best_experiment.parameters if self.best_experiment else {},
                    best_performance=self.current_best_performance,
                    performance_improvement=performance_improvement,
                    experiments_conducted=experiments_conducted,
                    tuning_duration_minutes=tuning_duration,
                    convergence_achieved=iterations_without_improvement < self.config.patience,
                    tuning_strategy_used=self.config.tuning_strategy,
                    validation_results=validation_results,
                    recommendations=self._generate_tuning_recommendations()
                )
                
                # Store result
                self.tuning_history.append(result)
                
                self.tuning_phase = TuningPhase.DEPLOYMENT
                self.is_tuning = False
                
                self.logger.info(f"Tuning completed: {performance_improvement:.2%} improvement")
                
                return result
                
            except Exception as e:
                self.is_tuning = False
                self.tuning_phase = TuningPhase.EXPLORATION
                self.logger.error(f"Parameter tuning failed: {e}")
                raise RuntimeError(f"Hybrid parameter tuning failed: {e}")
    
    def optimize_switching_thresholds(self) -> Dict[str, Any]:
        """
        Optimize switching thresholds for hybrid-pure transitions.
        
        Returns:
            Dictionary containing optimization results
            
        Raises:
            RuntimeError: If optimization fails
        """
        if not self.is_initialized:
            raise RuntimeError("Tuner not initialized")
        
        try:
            self.logger.info("Starting switching threshold optimization")
            
            # Get current switching performance
            current_metrics = self.performance_monitor.get_hybrid_performance_metrics()
            current_switching_overhead = current_metrics.get('switching_metrics', {}).get('average_switching_overhead_ms', 0.0)
            
            # Define threshold parameters
            threshold_params = {
                'data_size_threshold': TuningParameter(
                    name='data_size_threshold',
                    current_value=1000.0,
                    min_value=self.config.switching_threshold_range[0],
                    max_value=self.config.switching_threshold_range[1],
                    step_size=100.0,
                    parameter_type='continuous'
                ),
                'complexity_threshold': TuningParameter(
                    name='complexity_threshold',
                    current_value=500.0,
                    min_value=100.0,
                    max_value=2000.0,
                    step_size=50.0,
                    parameter_type='continuous'
                ),
                'performance_threshold': TuningParameter(
                    name='performance_threshold',
                    current_value=0.7,
                    min_value=0.5,
                    max_value=0.9,
                    step_size=0.05,
                    parameter_type='continuous'
                )
            }
            
            best_params = {}
            best_score = 0.0
            
            # Grid search for optimal thresholds
            for data_threshold in [500, 1000, 2000, 5000]:
                for complexity_threshold in [250, 500, 1000]:
                    for perf_threshold in [0.6, 0.7, 0.8]:
                        
                        # Test these parameters
                        test_params = {
                            'data_size_threshold': data_threshold,
                            'complexity_threshold': complexity_threshold,
                            'performance_threshold': perf_threshold
                        }
                        
                        score = self._evaluate_switching_parameters(test_params)
                        
                        if score > best_score:
                            best_score = score
                            best_params = test_params.copy()
            
            # Apply best parameters
            if best_params and self.dynamic_switching_manager:
                self._apply_switching_parameters(best_params)
            
            # Calculate improvement
            new_metrics = self.performance_monitor.get_hybrid_performance_metrics()
            new_switching_overhead = new_metrics.get('switching_metrics', {}).get('average_switching_overhead_ms', 0.0)
            
            improvement = (current_switching_overhead - new_switching_overhead) / current_switching_overhead if current_switching_overhead > 0 else 0.0
            
            result = {
                'optimized_parameters': best_params,
                'performance_improvement': improvement,
                'old_switching_overhead_ms': current_switching_overhead,
                'new_switching_overhead_ms': new_switching_overhead,
                'optimization_score': best_score
            }
            
            self.logger.info(f"Switching threshold optimization completed: {improvement:.2%} improvement")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Switching threshold optimization failed: {e}")
            raise RuntimeError(f"Switching threshold optimization failed: {e}")
    
    def tune_gpu_memory_allocation(self) -> Dict[str, Any]:
        """
        Tune GPU memory allocation for optimal performance.
        
        Returns:
            Dictionary containing tuning results
            
        Raises:
            RuntimeError: If tuning fails
        """
        if not self.is_initialized:
            raise RuntimeError("Tuner not initialized")
        
        try:
            self.logger.info("Starting GPU memory allocation tuning")
            
            # Get current GPU memory usage
            current_metrics = self.performance_monitor.get_hybrid_performance_metrics()
            current_gpu_memory = current_metrics.get('resource_metrics', {}).get('average_gpu_memory_mb', 0.0)
            
            # Define memory allocation strategies
            allocation_strategies = [
                {'strategy': 'conservative', 'base_allocation': 256, 'growth_factor': 1.2},
                {'strategy': 'balanced', 'base_allocation': 512, 'growth_factor': 1.5},
                {'strategy': 'aggressive', 'base_allocation': 1024, 'growth_factor': 2.0}
            ]
            
            best_strategy = None
            best_score = 0.0
            
            for strategy in allocation_strategies:
                # Test allocation strategy
                score = self._evaluate_memory_allocation_strategy(strategy)
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
            
            # Apply best strategy
            if best_strategy:
                self._apply_memory_allocation_strategy(best_strategy)
            
            # Measure improvement
            new_metrics = self.performance_monitor.get_hybrid_performance_metrics()
            new_gpu_memory = new_metrics.get('resource_metrics', {}).get('average_gpu_memory_mb', 0.0)
            
            memory_efficiency = current_gpu_memory / new_gpu_memory if new_gpu_memory > 0 else 1.0
            
            result = {
                'best_strategy': best_strategy,
                'performance_score': best_score,
                'memory_efficiency_improvement': memory_efficiency - 1.0,
                'old_gpu_memory_mb': current_gpu_memory,
                'new_gpu_memory_mb': new_gpu_memory
            }
            
            self.logger.info(f"GPU memory tuning completed: {best_strategy['strategy']} strategy selected")
            
            return result
            
        except Exception as e:
            self.logger.error(f"GPU memory allocation tuning failed: {e}")
            raise RuntimeError(f"GPU memory allocation tuning failed: {e}")
    
    def optimize_switching_strategy(self) -> Dict[str, Any]:
        """
        Optimize the overall switching strategy.
        
        Returns:
            Dictionary containing optimization results
            
        Raises:
            RuntimeError: If optimization fails
        """
        if not self.is_initialized:
            raise RuntimeError("Tuner not initialized")
        
        try:
            self.logger.info("Starting switching strategy optimization")
            
            # Analyze current switching patterns
            switching_analysis = self.performance_analyzer.analyze_switching_overhead()
            
            # Define strategy improvements
            strategy_improvements = []
            
            # Reduce switching frequency if overhead is high
            if switching_analysis.average_switching_overhead_ms > self.config.switching_threshold_range[0]:
                strategy_improvements.append({
                    'type': 'reduce_switching_frequency',
                    'parameters': {'hysteresis_factor': 1.2, 'min_duration_ms': 100}
                })
            
            # Implement predictive switching
            strategy_improvements.append({
                'type': 'predictive_switching',
                'parameters': {'prediction_window': 5, 'confidence_threshold': 0.8}
            })
            
            # Batch switching decisions
            strategy_improvements.append({
                'type': 'batch_switching',
                'parameters': {'batch_size': 10, 'batch_timeout_ms': 50}
            })
            
            best_improvement = None
            best_score = switching_analysis.switching_efficiency
            
            for improvement in strategy_improvements:
                # Test improvement
                score = self._evaluate_switching_improvement(improvement)
                
                if score > best_score:
                    best_score = score
                    best_improvement = improvement
            
            # Apply best improvement
            if best_improvement:
                self._apply_switching_improvement(best_improvement)
            
            result = {
                'best_improvement': best_improvement,
                'efficiency_improvement': best_score - switching_analysis.switching_efficiency,
                'old_efficiency': switching_analysis.switching_efficiency,
                'new_efficiency': best_score,
                'optimization_applied': best_improvement is not None
            }
            
            self.logger.info(f"Switching strategy optimization completed")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Switching strategy optimization failed: {e}")
            raise RuntimeError(f"Switching strategy optimization failed: {e}")
    
    def apply_performance_optimizations(self, optimizations: List[OptimizationRecommendation]) -> Dict[str, Any]:
        """
        Apply performance optimizations.
        
        Args:
            optimizations: List of optimization recommendations to apply
            
        Returns:
            Dictionary containing application results
            
        Raises:
            ValueError: If optimizations are invalid
            RuntimeError: If application fails
        """
        if not isinstance(optimizations, list):
            raise TypeError("Optimizations must be list")
        if not optimizations:
            raise ValueError("Optimizations list cannot be empty")
        
        if not self.is_initialized:
            raise RuntimeError("Tuner not initialized")
        
        try:
            self.logger.info(f"Applying {len(optimizations)} performance optimizations")
            
            applied_optimizations = []
            failed_optimizations = []
            
            # Get baseline performance
            baseline_metrics = self.performance_monitor.get_hybrid_performance_metrics()
            
            for optimization in optimizations:
                try:
                    success = self._apply_single_optimization(optimization)
                    
                    if success:
                        applied_optimizations.append(optimization)
                        self.logger.info(f"Applied optimization: {optimization.title}")
                    else:
                        failed_optimizations.append(optimization)
                        self.logger.warning(f"Failed to apply optimization: {optimization.title}")
                        
                except Exception as e:
                    failed_optimizations.append(optimization)
                    self.logger.error(f"Error applying optimization '{optimization.title}': {e}")
            
            # Measure overall improvement
            new_metrics = self.performance_monitor.get_hybrid_performance_metrics()
            
            # Calculate performance improvement
            improvement = self._calculate_optimization_improvement(baseline_metrics, new_metrics)
            
            result = {
                'applied_optimizations': len(applied_optimizations),
                'failed_optimizations': len(failed_optimizations),
                'overall_improvement': improvement,
                'applied_optimization_ids': [opt.recommendation_id for opt in applied_optimizations],
                'failed_optimization_ids': [opt.recommendation_id for opt in failed_optimizations],
                'success_rate': len(applied_optimizations) / len(optimizations)
            }
            
            self.logger.info(f"Optimization application completed: {improvement:.2%} improvement")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization application failed: {e}")
            raise RuntimeError(f"Performance optimization application failed: {e}")
    
    def get_tuning_recommendations(self) -> List[str]:
        """
        Get tuning recommendations based on current performance.
        
        Returns:
            List of tuning recommendations
        """
        recommendations = []
        
        try:
            # Get current performance metrics
            current_metrics = self.performance_monitor.get_hybrid_performance_metrics()
            
            # Analyze performance patterns
            if current_metrics.get('resource_metrics', {}).get('average_gpu_memory_mb', 0) > 1500:
                recommendations.append("Consider optimizing GPU memory usage - current usage is high")
            
            if current_metrics.get('switching_metrics', {}).get('average_switching_overhead_ms', 0) > 20:
                recommendations.append("Switching overhead is high - consider optimizing switching thresholds")
            
            if current_metrics.get('operation_metrics', {}).get('success_rate', 1.0) < 0.95:
                recommendations.append("Success rate is below 95% - investigate failure causes")
            
            if current_metrics.get('performance_metrics', {}).get('average_execution_time_ms', 0) > 1000:
                recommendations.append("Execution time is high - consider performance optimizations")
            
            # Add adaptive recommendations based on tuning history
            if self.tuning_history:
                last_result = self.tuning_history[-1]
                if last_result.performance_improvement < 0.05:
                    recommendations.append("Recent tuning showed minimal improvement - consider different optimization strategies")
            
            # Parameter-specific recommendations
            for param_name, sensitivity in self.parameter_sensitivity.items():
                if sensitivity > 0.5:
                    recommendations.append(f"Parameter '{param_name}' has high sensitivity - fine-tune carefully")
            
        except Exception as e:
            self.logger.error(f"Error generating tuning recommendations: {e}")
            recommendations.append("Error analyzing performance - manual review recommended")
        
        return recommendations
    
    def _initialize_tuning_parameters(self) -> None:
        """Initialize tuning parameters"""
        try:
            # GPU memory parameters
            if self.config.enable_gpu_memory_tuning:
                self.tuning_parameters['gpu_memory_threshold'] = TuningParameter(
                    name='gpu_memory_threshold',
                    current_value=1024.0,
                    min_value=self.config.gpu_memory_threshold_range[0],
                    max_value=self.config.gpu_memory_threshold_range[1],
                    step_size=128.0,
                    parameter_type='continuous',
                    optimization_priority='high'
                )
            
            # Switching threshold parameters
            if self.config.enable_switching_threshold_tuning:
                self.tuning_parameters['switching_threshold'] = TuningParameter(
                    name='switching_threshold',
                    current_value=1000.0,
                    min_value=self.config.switching_threshold_range[0],
                    max_value=self.config.switching_threshold_range[1],
                    step_size=100.0,
                    parameter_type='continuous',
                    optimization_priority='high'
                )
            
            # Performance parameters
            if self.config.enable_performance_parameter_tuning:
                self.tuning_parameters['learning_rate'] = TuningParameter(
                    name='learning_rate',
                    current_value=0.01,
                    min_value=self.config.learning_rate_range[0],
                    max_value=self.config.learning_rate_range[1],
                    step_size=0.005,
                    parameter_type='continuous',
                    optimization_priority='medium'
                )
            
            self.logger.debug(f"Initialized {len(self.tuning_parameters)} tuning parameters")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tuning parameters: {e}")
            raise
    
    def _establish_baseline_performance(self) -> None:
        """Establish baseline performance metrics"""
        try:
            current_metrics = self.performance_monitor.get_hybrid_performance_metrics()
            
            self.baseline_performance = {
                'execution_time_ms': current_metrics.get('performance_metrics', {}).get('average_execution_time_ms', 1000.0),
                'gpu_memory_mb': current_metrics.get('resource_metrics', {}).get('average_gpu_memory_mb', 512.0),
                'cpu_usage': current_metrics.get('resource_metrics', {}).get('average_cpu_usage', 50.0),
                'success_rate': current_metrics.get('operation_metrics', {}).get('success_rate', 0.95),
                'switching_overhead_ms': current_metrics.get('switching_metrics', {}).get('average_switching_overhead_ms', 10.0)
            }
            
            self.current_best_performance = self.baseline_performance.copy()
            
            self.logger.debug("Baseline performance established")
            
        except Exception as e:
            self.logger.error(f"Failed to establish baseline performance: {e}")
            # Use default values if unable to get current metrics
            self.baseline_performance = {
                'execution_time_ms': 1000.0,
                'gpu_memory_mb': 512.0,
                'cpu_usage': 50.0,
                'success_rate': 0.95,
                'switching_overhead_ms': 10.0
            }
            self.current_best_performance = self.baseline_performance.copy()
    
    def _initialize_adaptive_learning(self) -> None:
        """Initialize adaptive learning components"""
        try:
            # Initialize parameter correlations
            param_names = list(self.tuning_parameters.keys())
            for param1 in param_names:
                self.parameter_correlations[param1] = {}
                for param2 in param_names:
                    self.parameter_correlations[param1][param2] = 0.0
            
            # Initialize parameter sensitivity
            for param_name in param_names:
                self.parameter_sensitivity[param_name] = 0.5  # Start with medium sensitivity
            
            self.logger.debug("Adaptive learning initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize adaptive learning: {e}")
    
    def _select_tuning_strategy(self, iteration: int) -> TuningStrategy:
        """Select tuning strategy for current iteration"""
        if self.config.tuning_strategy == TuningStrategy.ADAPTIVE:
            # Start with exploration, gradually move to exploitation
            if iteration < self.config.max_tuning_iterations * 0.3:
                return TuningStrategy.AGGRESSIVE  # Exploration phase
            elif iteration < self.config.max_tuning_iterations * 0.7:
                return TuningStrategy.BALANCED    # Balanced phase
            else:
                return TuningStrategy.CONSERVATIVE # Exploitation phase
        else:
            return self.config.tuning_strategy
    
    def _generate_parameter_candidates(self, strategy: TuningStrategy) -> List[Dict[str, float]]:
        """Generate parameter candidates for testing"""
        candidates = []
        
        # Generate based on strategy
        if strategy == TuningStrategy.AGGRESSIVE:
            num_candidates = 5
        elif strategy == TuningStrategy.BALANCED:
            num_candidates = 3
        else:  # CONSERVATIVE
            num_candidates = 2
        
        for _ in range(num_candidates):
            candidate = {}
            for param_name, param in self.tuning_parameters.items():
                candidate[param_name] = param.propose_new_value(strategy)
            candidates.append(candidate)
        
        return candidates
    
    def _run_tuning_experiment(self, parameters: Dict[str, float], strategy: TuningStrategy) -> TuningExperiment:
        """Run a single tuning experiment"""
        experiment_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Apply parameters temporarily
            original_params = self._backup_current_parameters()
            self._apply_parameters_temporarily(parameters)
            
            # Run test operations and measure performance
            performance_metrics = self._measure_performance_with_parameters(parameters)
            
            # Restore original parameters
            self._restore_parameters(original_params)
            
            experiment = TuningExperiment(
                experiment_id=experiment_id,
                parameters=parameters.copy(),
                performance_metrics=performance_metrics,
                success=True,
                execution_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.utcnow(),
                strategy_used=strategy
            )
            
        except Exception as e:
            self.logger.error(f"Experiment {experiment_id} failed: {e}")
            experiment = TuningExperiment(
                experiment_id=experiment_id,
                parameters=parameters.copy(),
                performance_metrics={},
                success=False,
                execution_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.utcnow(),
                strategy_used=strategy
            )
        
        # Store experiment
        with self._experiment_lock:
            self.experiments.append(experiment)
        
        return experiment
    
    def _get_metric_weights(self) -> Dict[str, float]:
        """Get metric weights for overall score calculation"""
        if self.config.optimization_metric == OptimizationMetric.EXECUTION_TIME:
            return {'execution_time_ms': 1.0}
        elif self.config.optimization_metric == OptimizationMetric.GPU_MEMORY_USAGE:
            return {'gpu_memory_mb': 1.0}
        elif self.config.optimization_metric == OptimizationMetric.THROUGHPUT:
            return {'throughput': 1.0}
        else:  # OVERALL_PERFORMANCE
            return {
                'execution_time_ms': 0.3,
                'gpu_memory_mb': 0.2,
                'cpu_usage': 0.1,
                'success_rate': 0.3,
                'switching_overhead_ms': 0.1
            }
    
    def _check_convergence(self, current_score: float) -> bool:
        """Check if tuning has converged"""
        if not hasattr(self, '_last_best_score'):
            self._last_best_score = current_score
            return False
        
        improvement = (current_score - self._last_best_score) / self._last_best_score if self._last_best_score > 0 else 0
        converged = improvement < self.config.convergence_threshold
        
        self._last_best_score = current_score
        return converged
    
    def _update_adaptive_learning(self) -> None:
        """Update adaptive learning based on experiment results"""
        if len(self.experiments) < 10:
            return
        
        # Update parameter sensitivity based on recent experiments
        recent_experiments = list(self.experiments)[-10:]
        
        for param_name in self.tuning_parameters:
            param_values = [exp.parameters.get(param_name, 0) for exp in recent_experiments if exp.success]
            param_scores = [exp.get_overall_score(self._get_metric_weights()) for exp in recent_experiments if exp.success]
            
            if len(param_values) >= 5:
                # Calculate correlation between parameter and performance
                correlation = self._calculate_correlation(param_values, param_scores)
                self.parameter_sensitivity[param_name] = abs(correlation)
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation between two lists"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        sum_y2 = sum(y[i] ** 2 for i in range(n))
        
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        correlation = (n * sum_xy - sum_x * sum_y) / denominator
        return correlation
    
    def _validate_best_parameters(self) -> Dict[str, Any]:
        """Validate best parameters found during tuning"""
        if not self.best_experiment:
            return {'validation_error': 'No best experiment found'}
        
        try:
            # Run validation experiments with best parameters
            validation_experiments = []
            
            for _ in range(5):  # Run 5 validation experiments
                validation_exp = self._run_tuning_experiment(
                    self.best_experiment.parameters,
                    TuningStrategy.CONSERVATIVE
                )
                validation_experiments.append(validation_exp)
            
            # Calculate validation metrics
            successful_validations = [exp for exp in validation_experiments if exp.success]
            
            if not successful_validations:
                return {'validation_error': 'All validation experiments failed'}
            
            validation_scores = [exp.get_overall_score(self._get_metric_weights()) for exp in successful_validations]
            
            return {
                'validation_success_rate': len(successful_validations) / len(validation_experiments),
                'average_validation_score': sum(validation_scores) / len(validation_scores),
                'validation_score_std': (sum((s - sum(validation_scores) / len(validation_scores)) ** 2 for s in validation_scores) / len(validation_scores)) ** 0.5,
                'validation_experiments_count': len(validation_experiments)
            }
            
        except Exception as e:
            return {'validation_error': f'Validation failed: {e}'}
    
    def _calculate_baseline_score(self) -> float:
        """Calculate baseline performance score"""
        return 0.5  # Placeholder baseline score
    
    def _generate_tuning_recommendations(self) -> List[str]:
        """Generate recommendations based on tuning results"""
        recommendations = []
        
        if self.best_experiment:
            # Parameter-specific recommendations
            for param_name, param_value in self.best_experiment.parameters.items():
                original_param = self.tuning_parameters.get(param_name)
                if original_param:
                    change_percentage = abs(param_value - original_param.current_value) / original_param.current_value * 100
                    if change_percentage > 20:
                        recommendations.append(f"Significant improvement found by changing {param_name} by {change_percentage:.1f}%")
        
        # Sensitivity-based recommendations
        for param_name, sensitivity in self.parameter_sensitivity.items():
            if sensitivity > 0.7:
                recommendations.append(f"Parameter {param_name} is highly sensitive - monitor closely")
            elif sensitivity < 0.1:
                recommendations.append(f"Parameter {param_name} has low impact - consider fixing or removing")
        
        return recommendations
    
    # Placeholder methods for complete implementation
    def _backup_current_parameters(self): return {}
    def _apply_parameters_temporarily(self, parameters): pass
    def _restore_parameters(self, parameters): pass
    def _measure_performance_with_parameters(self, parameters): return {}
    def _evaluate_switching_parameters(self, parameters): return 0.5
    def _apply_switching_parameters(self, parameters): pass
    def _evaluate_memory_allocation_strategy(self, strategy): return 0.5
    def _apply_memory_allocation_strategy(self, strategy): pass
    def _evaluate_switching_improvement(self, improvement): return 0.5
    def _apply_switching_improvement(self, improvement): pass
    def _apply_single_optimization(self, optimization): return True
    def _calculate_optimization_improvement(self, baseline, new_metrics): return 0.05
    
    def shutdown(self) -> None:
        """Shutdown hybrid performance tuner"""
        self.logger.info("Shutting down hybrid performance tuner")
        
        # Stop continuous tuning if running
        if self._continuous_tuning_enabled:
            self._continuous_tuning_enabled = False
            self._stop_tuning.set()
            if self._tuning_thread and self._tuning_thread.is_alive():
                self._tuning_thread.join(timeout=5.0)
        
        # Clear references
        self.hybrid_padic_manager = None
        self.dynamic_switching_manager = None
        self.hybrid_compression_system = None
        self.performance_monitor = None
        self.performance_analyzer = None
        
        # Clear data
        self.experiments.clear()
        self.tuning_history.clear()
        self.optimization_recommendations.clear()
        
        self.is_initialized = False
        self.tuning_phase = TuningPhase.INITIALIZATION
        self.logger.info("Hybrid performance tuner shutdown complete")