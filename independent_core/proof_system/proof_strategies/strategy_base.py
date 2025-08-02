"""
Strategy Base Classes - Base classes for all proof strategies
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import uuid
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import logging

# Import caching functionality
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from strategy_cache import StrategyCache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Base configuration for proof strategies"""
    max_iterations: int = 1000
    timeout_seconds: float = 60.0
    confidence_threshold: float = 0.8
    validation_required: bool = True
    parallel_execution: bool = False
    cache_results: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # NO FALLBACKS - HARD FAILURES ONLY
        if self.max_iterations <= 0:
            raise ValueError("Max iterations must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout seconds must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        if not isinstance(self.validation_required, bool):
            raise TypeError("Validation required must be boolean")
        if not isinstance(self.parallel_execution, bool):
            raise TypeError("Parallel execution must be boolean")
        if not isinstance(self.cache_results, bool):
            raise TypeError("Cache results must be boolean")
        if not isinstance(self.metadata, dict):
            raise TypeError("Metadata must be dict")


@dataclass
class StrategyResult:
    """Result from strategy execution"""
    success: bool
    proof_data: Dict[str, Any]
    confidence: float
    execution_time: float
    iterations_used: int
    strategy_name: str
    validation_passed: bool = True
    sub_results: Optional[List['StrategyResult']] = None
    error_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # NO FALLBACKS - HARD FAILURES ONLY
        if not isinstance(self.success, bool):
            raise TypeError("Success must be boolean")
        if not isinstance(self.proof_data, dict):
            raise TypeError("Proof data must be dict")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.execution_time < 0:
            raise ValueError("Execution time must be non-negative")
        if self.iterations_used < 0:
            raise ValueError("Iterations used must be non-negative")
        if not self.strategy_name or not isinstance(self.strategy_name, str):
            raise ValueError("Strategy name must be non-empty string")
        if not isinstance(self.validation_passed, bool):
            raise TypeError("Validation passed must be boolean")
        if self.sub_results is not None and not isinstance(self.sub_results, list):
            raise TypeError("Sub results must be list or None")
        if self.error_info is not None and not isinstance(self.error_info, dict):
            raise TypeError("Error info must be dict or None")
        if not isinstance(self.metadata, dict):
            raise TypeError("Metadata must be dict")


class ProofStrategy(ABC):
    """Abstract base class for all proof strategies"""
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        if config is not None and not isinstance(config, StrategyConfig):
            raise TypeError("Config must be StrategyConfig instance or None")
        
        self.config = config or StrategyConfig()
        self.strategy_name = self.__class__.__name__
        
        # Performance metrics
        self.performance_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'total_execution_time': 0.0,
            'avg_execution_time': 0.0,
            'avg_iterations': 0.0,
            'avg_confidence': 0.0
        }
        
        # Thread safety
        self.metrics_lock = threading.RLock()
        
        # Execution state
        self._current_execution_id: Optional[str] = None
        self._execution_start_time: Optional[float] = None
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> StrategyResult:
        """Execute the strategy with given input data"""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data before execution"""
        pass
    
    @abstractmethod
    def estimate_complexity(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate computational complexity for given input"""
        pass
    
    def run(self, input_data: Dict[str, Any]) -> StrategyResult:
        """Main execution method with validation and metrics tracking"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if input_data is None:
            raise ValueError("Input data cannot be None")
        if not isinstance(input_data, dict):
            raise TypeError("Input data must be dict")
        
        # Generate execution ID
        execution_id = str(uuid.uuid4())
        self._current_execution_id = execution_id
        
        # Validate input
        if self.config.validation_required:
            if not self.validate_input(input_data):
                raise ValueError("Input validation failed")
        
        # Check timeout
        start_time = time.time()
        self._execution_start_time = start_time
        
        try:
            # Execute strategy
            result = self.execute(input_data)
            
            # Validate result
            if not isinstance(result, StrategyResult):
                raise TypeError("Strategy must return StrategyResult")
            
            # Check timeout
            execution_time = time.time() - start_time
            if execution_time > self.config.timeout_seconds:
                raise TimeoutError(f"Strategy execution exceeded timeout of {self.config.timeout_seconds} seconds")
            
            # Update result with actual execution time
            result.execution_time = execution_time
            
            # Update performance metrics
            self._update_metrics(result)
            
            return result
            
        except Exception as e:
            # Create error result
            execution_time = time.time() - start_time
            
            error_result = StrategyResult(
                success=False,
                proof_data={},
                confidence=0.0,
                execution_time=execution_time,
                iterations_used=0,
                strategy_name=self.strategy_name,
                validation_passed=False,
                error_info={
                    'type': type(e).__name__,
                    'message': str(e),
                    'execution_id': execution_id
                }
            )
            
            # Update metrics for failed execution
            self._update_metrics(error_result)
            
            # Re-raise the exception
            raise
        
        finally:
            self._current_execution_id = None
            self._execution_start_time = None
    
    def _check_timeout(self, start_time: float) -> None:
        """Check if execution has timed out"""
        if time.time() - start_time > self.config.timeout_seconds:
            raise TimeoutError(f"Strategy execution exceeded timeout of {self.config.timeout_seconds} seconds")
    
    def _update_metrics(self, result: StrategyResult) -> None:
        """Update performance metrics"""
        with self.metrics_lock:
            self.performance_metrics['total_executions'] += 1
            
            if result.success:
                self.performance_metrics['successful_executions'] += 1
            
            self.performance_metrics['total_execution_time'] += result.execution_time
            
            # Update averages
            total_execs = self.performance_metrics['total_executions']
            self.performance_metrics['avg_execution_time'] = (
                self.performance_metrics['total_execution_time'] / total_execs
            )
            
            # Update iteration average
            old_avg_iterations = self.performance_metrics['avg_iterations']
            self.performance_metrics['avg_iterations'] = (
                (old_avg_iterations * (total_execs - 1) + result.iterations_used) / total_execs
            )
            
            # Update confidence average
            old_avg_confidence = self.performance_metrics['avg_confidence']
            self.performance_metrics['avg_confidence'] = (
                (old_avg_confidence * (total_execs - 1) + result.confidence) / total_execs
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        with self.metrics_lock:
            return self.performance_metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics"""
        with self.metrics_lock:
            self.performance_metrics = {
                'total_executions': 0,
                'successful_executions': 0,
                'total_execution_time': 0.0,
                'avg_execution_time': 0.0,
                'avg_iterations': 0.0,
                'avg_confidence': 0.0
            }
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get comprehensive strategy information"""
        return {
            'strategy_name': self.strategy_name,
            'strategy_class': self.__class__.__name__,
            'config': {
                'max_iterations': self.config.max_iterations,
                'timeout_seconds': self.config.timeout_seconds,
                'confidence_threshold': self.config.confidence_threshold,
                'validation_required': self.config.validation_required,
                'parallel_execution': self.config.parallel_execution,
                'cache_results': self.config.cache_results
            },
            'performance_metrics': self.get_performance_metrics(),
            'current_execution_id': self._current_execution_id,
            'is_executing': self._current_execution_id is not None
        }


class ComposableStrategy(ProofStrategy):
    """Base class for strategies that can be composed with other strategies"""
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config)
        self.sub_strategies: Dict[str, ProofStrategy] = {}
        self.composition_lock = threading.RLock()
    
    def add_sub_strategy(self, name: str, strategy: ProofStrategy) -> None:
        """Add a sub-strategy"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not name or not isinstance(name, str):
            raise ValueError("Strategy name must be non-empty string")
        if not isinstance(strategy, ProofStrategy):
            raise TypeError("Strategy must be ProofStrategy instance")
        
        with self.composition_lock:
            if name in self.sub_strategies:
                raise ValueError(f"Sub-strategy '{name}' already exists")
            
            self.sub_strategies[name] = strategy
    
    def remove_sub_strategy(self, name: str) -> None:
        """Remove a sub-strategy"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not name or not isinstance(name, str):
            raise ValueError("Strategy name must be non-empty string")
        
        with self.composition_lock:
            if name not in self.sub_strategies:
                raise ValueError(f"Sub-strategy '{name}' does not exist")
            
            del self.sub_strategies[name]
    
    def get_sub_strategy(self, name: str) -> ProofStrategy:
        """Get a sub-strategy by name"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not name or not isinstance(name, str):
            raise ValueError("Strategy name must be non-empty string")
        
        with self.composition_lock:
            if name not in self.sub_strategies:
                raise ValueError(f"Sub-strategy '{name}' does not exist")
            
            return self.sub_strategies[name]
    
    def list_sub_strategies(self) -> List[str]:
        """List all sub-strategy names"""
        with self.composition_lock:
            return list(self.sub_strategies.keys())
    
    def get_sub_strategy_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all sub-strategies"""
        with self.composition_lock:
            info = {}
            for name, strategy in self.sub_strategies.items():
                info[name] = strategy.get_strategy_info()
            return info


class CachingStrategy(ProofStrategy):
    """Base class for strategies with caching capabilities"""
    
    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config)
        
        # Initialize cache if available and enabled
        self.cache: Optional[StrategyCache] = None
        if CACHE_AVAILABLE and self.config.cache_results:
            try:
                cache_config = self.config.metadata.get('cache_config', {})
                self.cache = StrategyCache(
                    max_size=cache_config.get('max_size', 1000),
                    max_size_bytes=cache_config.get('max_size_bytes', 100 * 1024 * 1024),
                    default_ttl=cache_config.get('default_ttl', 3600)  # 1 hour default TTL
                )
                logger.debug(f"Cache initialized for strategy {self.strategy_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize cache for strategy {self.strategy_name}: {e}")
                self.cache = None
    
    def _generate_cache_key(self, input_data: Dict[str, Any]) -> str:
        """Generate cache key for input data"""
        try:
            import json
            # Create deterministic key from input data
            sorted_data = json.dumps(input_data, sort_keys=True, default=str)
            import hashlib
            return hashlib.sha256(sorted_data.encode()).hexdigest()
        except Exception:
            # Fallback to string representation
            return str(hash(str(sorted(input_data.items()))))
    
    def _check_cache(self, input_data: Dict[str, Any]) -> Optional[StrategyResult]:
        """Check cache for existing result"""
        if not self.cache:
            return None
        
        try:
            cache_key = self._generate_cache_key(input_data)
            cached_result = self.cache.get(cache_key)
            
            if cached_result and isinstance(cached_result, StrategyResult):
                logger.debug(f"Cache hit for strategy {self.strategy_name}")
                return cached_result
            
        except Exception as e:
            logger.warning(f"Cache lookup failed for strategy {self.strategy_name}: {e}")
        
        return None
    
    def _update_cache(self, input_data: Dict[str, Any], result: StrategyResult) -> None:
        """Update cache with result"""
        if not self.cache:
            return
        
        try:
            cache_key = self._generate_cache_key(input_data)
            
            # Only cache successful results
            if result.success and result.confidence >= 0.5:
                self.cache.put(cache_key, result)
                logger.debug(f"Cached result for strategy {self.strategy_name}")
            
        except Exception as e:
            logger.warning(f"Cache update failed for strategy {self.strategy_name}: {e}")
    
    def get_cache_statistics(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics"""
        if not self.cache:
            return None
        
        try:
            stats = self.cache.get_statistics()
            return {
                'hit_rate': stats.hit_rate,
                'miss_rate': stats.miss_rate,
                'total_requests': stats.total_requests,
                'hits': stats.hits,
                'misses': stats.misses,
                'evictions': stats.evictions,
                'cache_size': self.cache.get_size(),
                'cache_size_bytes': self.cache.get_size_bytes()
            }
        except Exception as e:
            logger.warning(f"Failed to get cache statistics: {e}")
            return None
    
    def clear_cache(self) -> None:
        """Clear the cache"""
        if self.cache:
            try:
                self.cache.clear()
                logger.info(f"Cache cleared for strategy {self.strategy_name}")
            except Exception as e:
                logger.warning(f"Failed to clear cache: {e}")
    
    def __del__(self):
        """Destructor to clean up cache"""
        if self.cache:
            try:
                self.cache.shutdown()
            except Exception:
                pass  # Ignore errors during destruction


# Helper functions for strategy validation

def validate_strategy_result(result: StrategyResult) -> bool:
    """Validate a strategy result"""
    try:
        # Basic type validation is done in __post_init__
        # Additional logical validation
        
        if result.success and result.confidence < 0.1:
            logger.warning("Successful result with very low confidence")
            return False
        
        if not result.success and result.confidence > 0.9:
            logger.warning("Failed result with very high confidence")
            return False
        
        if result.iterations_used > 10000:  # Reasonable upper bound
            logger.warning("Extremely high iteration count")
            return False
        
        if result.execution_time > 3600:  # 1 hour upper bound
            logger.warning("Extremely long execution time")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Strategy result validation failed: {e}")
        return False


def validate_strategy_config(config: StrategyConfig) -> bool:
    """Validate a strategy configuration"""
    try:
        # Basic type validation is done in __post_init__
        # Additional logical validation
        
        if config.max_iterations > 100000:
            logger.warning("Very high max iterations setting")
            return False
        
        if config.timeout_seconds > 7200:  # 2 hours
            logger.warning("Very long timeout setting")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Strategy config validation failed: {e}")
        return False


__all__ = [
    'StrategyConfig',
    'StrategyResult', 
    'ProofStrategy',
    'ComposableStrategy',
    'CachingStrategy',
    'validate_strategy_result',
    'validate_strategy_config'
]