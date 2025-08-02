"""
Proof Strategies Module - Strategy registry and factory for proof strategies
NO FALLBACKS - HARD FAILURES ONLY
"""

import threading
from typing import Dict, Type, Any, Optional, List, Callable
import logging

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Central registry for all proof strategies"""
    
    def __init__(self):
        self._strategies: Dict[str, Type] = {}
        self._strategy_configs: Dict[str, Dict[str, Any]] = {}
        self._strategy_metadata: Dict[str, Dict[str, Any]] = {}
        self._registry_lock = threading.RLock()
    
    def register_strategy(self, name: str, strategy_class: Type, 
                         default_config: Optional[Dict[str, Any]] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a strategy class"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not name or not isinstance(name, str):
            raise ValueError("Strategy name must be non-empty string")
        if not isinstance(strategy_class, type):
            raise TypeError("Strategy class must be a class type")
        
        with self._registry_lock:
            if name in self._strategies:
                raise ValueError(f"Strategy '{name}' is already registered")
            
            self._strategies[name] = strategy_class
            self._strategy_configs[name] = default_config or {}
            self._strategy_metadata[name] = metadata or {}
            
            logger.info(f"Registered strategy: {name}")
    
    def unregister_strategy(self, name: str) -> None:
        """Unregister a strategy"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not name or not isinstance(name, str):
            raise ValueError("Strategy name must be non-empty string")
        
        with self._registry_lock:
            if name not in self._strategies:
                raise ValueError(f"Strategy '{name}' is not registered")
            
            del self._strategies[name]
            del self._strategy_configs[name]
            del self._strategy_metadata[name]
            
            logger.info(f"Unregistered strategy: {name}")
    
    def get_strategy_class(self, name: str) -> Type:
        """Get strategy class by name"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not name or not isinstance(name, str):
            raise ValueError("Strategy name must be non-empty string")
        
        with self._registry_lock:
            if name not in self._strategies:
                raise ValueError(f"Strategy '{name}' is not registered")
            
            return self._strategies[name]
    
    def get_strategy_config(self, name: str) -> Dict[str, Any]:
        """Get default config for strategy"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not name or not isinstance(name, str):
            raise ValueError("Strategy name must be non-empty string")
        
        with self._registry_lock:
            if name not in self._strategies:
                raise ValueError(f"Strategy '{name}' is not registered")
            
            return self._strategy_configs[name].copy()
    
    def get_strategy_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for strategy"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not name or not isinstance(name, str):
            raise ValueError("Strategy name must be non-empty string")
        
        with self._registry_lock:
            if name not in self._strategies:
                raise ValueError(f"Strategy '{name}' is not registered")
            
            return self._strategy_metadata[name].copy()
    
    def list_strategies(self) -> List[str]:
        """List all registered strategy names"""
        with self._registry_lock:
            return list(self._strategies.keys())
    
    def is_registered(self, name: str) -> bool:
        """Check if strategy is registered"""
        if not name or not isinstance(name, str):
            return False
        
        with self._registry_lock:
            return name in self._strategies
    
    def get_registry_info(self) -> Dict[str, Any]:
        """Get comprehensive registry information"""
        with self._registry_lock:
            info = {}
            for name in self._strategies:
                info[name] = {
                    'class': self._strategies[name].__name__,
                    'module': self._strategies[name].__module__,
                    'config_keys': list(self._strategy_configs[name].keys()),
                    'metadata': self._strategy_metadata[name].copy()
                }
            
            return {
                'strategies': info,
                'total_count': len(self._strategies),
                'strategy_names': list(self._strategies.keys())
            }


class StrategyFactory:
    """Factory for creating strategy instances"""
    
    def __init__(self, registry: Optional[StrategyRegistry] = None):
        self.registry = registry or StrategyRegistry()
        self._creation_hooks: Dict[str, List[Callable]] = {}
        self._factory_lock = threading.RLock()
    
    def create_strategy(self, name: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Create strategy instance"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not name or not isinstance(name, str):
            raise ValueError("Strategy name must be non-empty string")
        
        with self._factory_lock:
            # Get strategy class
            strategy_class = self.registry.get_strategy_class(name)
            
            # Merge configs
            default_config = self.registry.get_strategy_config(name)
            if config:
                if not isinstance(config, dict):
                    raise TypeError("Config must be dict")
                merged_config = default_config.copy()
                merged_config.update(config)
            else:
                merged_config = default_config
            
            # Create instance
            try:
                if merged_config:
                    instance = strategy_class(merged_config)
                else:
                    instance = strategy_class()
            except Exception as e:
                raise RuntimeError(f"Failed to create strategy '{name}': {e}")
            
            # Execute creation hooks
            for hook in self._creation_hooks.get(name, []):
                try:
                    hook(instance, merged_config)
                except Exception as e:
                    logger.error(f"Creation hook failed for strategy '{name}': {e}")
                    raise RuntimeError(f"Creation hook failed for strategy '{name}': {e}")
            
            logger.debug(f"Created strategy instance: {name}")
            return instance
    
    def add_creation_hook(self, strategy_name: str, hook: Callable) -> None:
        """Add a hook to be called after strategy creation"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not strategy_name or not isinstance(strategy_name, str):
            raise ValueError("Strategy name must be non-empty string")
        if not callable(hook):
            raise TypeError("Hook must be callable")
        
        with self._factory_lock:
            if strategy_name not in self._creation_hooks:
                self._creation_hooks[strategy_name] = []
            
            self._creation_hooks[strategy_name].append(hook)
    
    def remove_creation_hook(self, strategy_name: str, hook: Callable) -> None:
        """Remove a creation hook"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not strategy_name or not isinstance(strategy_name, str):
            raise ValueError("Strategy name must be non-empty string")
        if not callable(hook):
            raise TypeError("Hook must be callable")
        
        with self._factory_lock:
            if strategy_name in self._creation_hooks:
                try:
                    self._creation_hooks[strategy_name].remove(hook)
                    if not self._creation_hooks[strategy_name]:
                        del self._creation_hooks[strategy_name]
                except ValueError:
                    raise ValueError(f"Hook not found for strategy '{strategy_name}'")


# Global registry and factory instances
_global_registry = StrategyRegistry()
_global_factory = StrategyFactory(_global_registry)


def register_strategy(name: str, strategy_class: Type, 
                     default_config: Optional[Dict[str, Any]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
    """Register a strategy globally"""
    _global_registry.register_strategy(name, strategy_class, default_config, metadata)


def unregister_strategy(name: str) -> None:
    """Unregister a strategy globally"""
    _global_registry.unregister_strategy(name)


def create_strategy(name: str, config: Optional[Dict[str, Any]] = None) -> Any:
    """Create strategy instance globally"""
    return _global_factory.create_strategy(name, config)


def list_strategies() -> List[str]:
    """List all registered strategies globally"""
    return _global_registry.list_strategies()


def is_strategy_registered(name: str) -> bool:
    """Check if strategy is registered globally"""
    return _global_registry.is_registered(name)


def get_registry() -> StrategyRegistry:
    """Get global registry"""
    return _global_registry


def get_factory() -> StrategyFactory:
    """Get global factory"""
    return _global_factory


def get_strategy_info(name: str) -> Dict[str, Any]:
    """Get comprehensive strategy information"""
    if not is_strategy_registered(name):
        raise ValueError(f"Strategy '{name}' is not registered")
    
    return {
        'name': name,
        'class': _global_registry.get_strategy_class(name).__name__,
        'module': _global_registry.get_strategy_class(name).__module__,
        'default_config': _global_registry.get_strategy_config(name),
        'metadata': _global_registry.get_strategy_metadata(name)
    }


# Auto-register built-in strategies when module is imported
def _register_builtin_strategies():
    """Register built-in strategies"""
    try:
        # Import and register adaptive strategy
        from .adaptive_strategy import AdaptiveStrategy
        register_strategy(
            'adaptive',
            AdaptiveStrategy,
            default_config={
                'learning_rate': 0.1,
                'exploration_rate': 0.2,
                'adaptation_window': 100,
                'min_confidence_improvement': 0.05,
                'strategy_selection_method': 'epsilon_greedy',
                'performance_memory': 1000
            },
            metadata={
                'description': 'Adaptive proof strategy with reinforcement learning',
                'approaches': ['aggressive', 'conservative', 'balanced', 'exploratory'],
                'selection_methods': ['epsilon_greedy', 'ucb', 'thompson'],
                'version': '1.0.0'
            }
        )
        
        # Import and register ensemble strategy
        from .ensemble_strategy import EnsembleStrategy
        register_strategy(
            'ensemble',
            EnsembleStrategy,
            default_config={
                'voting_method': 'weighted',
                'min_agreement': 0.6,
                'weight_update_rate': 0.1,
                'parallel_limit': 4,
                'timeout_per_strategy': 30.0,
                'confidence_aggregation': 'weighted_mean',
                'require_quorum': True,
                'quorum_size': 0.5,
                'boosting_enabled': True,
                'boosting_rounds': 3
            },
            metadata={
                'description': 'Ensemble strategy combining multiple proof strategies',
                'voting_methods': ['weighted', 'majority', 'consensus', 'adaptive'],
                'aggregation_methods': ['weighted_mean', 'max', 'median'],
                'version': '1.0.0'
            }
        )
        
        logger.info("Built-in strategies registered successfully")
        
    except ImportError as e:
        logger.warning(f"Failed to register built-in strategies: {e}")
    except Exception as e:
        logger.error(f"Error registering built-in strategies: {e}")


# Register built-in strategies on module import
_register_builtin_strategies()


__all__ = [
    'StrategyRegistry',
    'StrategyFactory',
    'register_strategy',
    'unregister_strategy',
    'create_strategy',
    'list_strategies',
    'is_strategy_registered',
    'get_registry',
    'get_factory',
    'get_strategy_info'
]