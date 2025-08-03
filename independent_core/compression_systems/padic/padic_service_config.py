"""
P-Adic Service Configuration Module

Provides unified configuration management for P-Adic services with validation,
history tracking, and change notification capabilities.

Integrates with existing Saraphis independent core P-adic compression system.
"""

import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from collections import deque
from pathlib import Path
import logging

# Import existing P-adic configuration types for compatibility
try:
    from .padic_integration import PadicIntegrationConfig
    from .padic_advanced import HenselLiftingConfig, ClusteringConfig, GPUDecompressionConfig
except ImportError:
    # Handle direct script execution
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from padic_integration import PadicIntegrationConfig
    from padic_advanced import HenselLiftingConfig, ClusteringConfig, GPUDecompressionConfig

# Configure logging
logger = logging.getLogger(__name__)


class PadicServiceConfiguration:
    """
    Unified configuration management for P-Adic services.
    
    Provides centralized configuration for all P-adic compression components
    while maintaining compatibility with existing configuration classes.
    """
    
    def __init__(self, config_path: Optional[Path] = None,
                 default_config: Optional[Dict[str, Any]] = None):
        """
        Initialize P-Adic service configuration.
        
        Args:
            config_path: Optional path to configuration file
            default_config: Optional default configuration
        """
        self.config_path = config_path
        self.config = default_config or self._get_default_config()
        self.config_history = deque(maxlen=100)
        self.config_schema = self._get_config_schema()
        
        # Load configuration from file if provided
        if config_path and config_path.exists():
            self.load_configuration()
        
        # Validate initial configuration
        self.validate_configuration()
        
        # Configuration change callbacks
        self.change_callbacks = []
        
        logger.info("P-Adic service configuration initialized")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> bool:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
            
        Returns:
            True if successful
        """
        try:
            # Save current state to history
            self._save_to_history()
            
            # Set value
            keys = key.split('.')
            target = self.config
            
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            
            old_value = target.get(keys[-1])
            target[keys[-1]] = value
            
            # Validate new configuration
            if not self.validate_configuration():
                # Rollback on validation failure
                target[keys[-1]] = old_value
                return False
            
            # Notify callbacks
            self._notify_change(key, old_value, value)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set configuration {key}: {str(e)}")
            return False
    
    def update(self, updates: Dict[str, Any]) -> bool:
        """
        Update multiple configuration values.
        
        Args:
            updates: Dictionary of updates
            
        Returns:
            True if all updates successful
        """
        # Save current state
        self._save_to_history()
        original_config = self.config.copy()
        
        try:
            # Apply updates
            self._deep_update(self.config, updates)
            
            # Validate
            if not self.validate_configuration():
                self.config = original_config
                return False
            
            # Notify callbacks for all changes
            for key, value in self._flatten_dict(updates).items():
                old_value = self._get_nested_value(original_config, key)
                if old_value != value:
                    self._notify_change(key, old_value, value)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {str(e)}")
            self.config = original_config
            return False
    
    def validate_configuration(self) -> bool:
        """
        Validate current configuration against schema.
        
        Returns:
            True if valid
        """
        try:
            return self._validate_against_schema(self.config, self.config_schema)
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False
    
    def load_configuration(self, path: Optional[Path] = None) -> bool:
        """
        Load configuration from file.
        
        Args:
            path: Optional path override
            
        Returns:
            True if successful
        """
        load_path = path or self.config_path
        if not load_path or not load_path.exists():
            logger.error(f"Configuration file not found: {load_path}")
            return False
        
        try:
            with open(load_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Validate loaded configuration
            if self._validate_against_schema(loaded_config, self.config_schema):
                self._save_to_history()
                self.config = loaded_config
                logger.info(f"Configuration loaded from {load_path}")
                return True
            else:
                logger.error("Loaded configuration failed validation")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            return False
    
    def save_configuration(self, path: Optional[Path] = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            path: Optional path override
            
        Returns:
            True if successful
        """
        save_path = path or self.config_path
        if not save_path:
            logger.error("No configuration path specified")
            return False
        
        try:
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            return False
    
    def get_history(self, count: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get configuration history.
        
        Args:
            count: Optional number of entries to retrieve
            
        Returns:
            List of historical configurations
        """
        if count:
            return list(self.config_history)[-count:]
        return list(self.config_history)
    
    def rollback(self, steps: int = 1) -> bool:
        """
        Rollback configuration to previous state.
        
        Args:
            steps: Number of steps to rollback
            
        Returns:
            True if successful
        """
        if steps > len(self.config_history):
            logger.error(f"Cannot rollback {steps} steps, only {len(self.config_history)} available")
            return False
        
        try:
            # Get target configuration
            target_config = self.config_history[-steps]
            
            # Validate before applying
            if self._validate_against_schema(target_config['config'], self.config_schema):
                self.config = target_config['config'].copy()
                logger.info(f"Configuration rolled back {steps} steps")
                return True
            else:
                logger.error("Rollback target configuration failed validation")
                return False
                
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")
            return False
    
    def register_change_callback(self, callback: Callable) -> None:
        """
        Register callback for configuration changes.
        
        Args:
            callback: Callback function(key, old_value, new_value)
        """
        self.change_callbacks.append(callback)
    
    def export_configuration(self) -> Dict[str, Any]:
        """
        Export current configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()
    
    def import_configuration(self, config: Dict[str, Any]) -> bool:
        """
        Import configuration.
        
        Args:
            config: Configuration to import
            
        Returns:
            True if successful
        """
        if self._validate_against_schema(config, self.config_schema):
            self._save_to_history()
            self.config = config.copy()
            return True
        return False
    
    def get_integration_config(self) -> PadicIntegrationConfig:
        """
        Get PadicIntegrationConfig object from current configuration.
        
        Returns:
            PadicIntegrationConfig instance
        """
        integration_cfg = self.get('integration', {})
        return PadicIntegrationConfig(
            prime=integration_cfg.get('default_p', 2999),
            base_precision=integration_cfg.get('default_precision', 100),
            adaptive_precision=integration_cfg.get('adaptive_precision', True),
            precision_min=integration_cfg.get('precision_min', 50),
            precision_max=integration_cfg.get('precision_max', 500),
            precision_step=integration_cfg.get('precision_step', 50),
            gac_compression_threshold=integration_cfg.get('gac_compression_threshold', 0.001),
            gac_batch_size=self.get('service.max_batch_size', 32),
            gac_async_processing=integration_cfg.get('gac_async_processing', True),
            brain_priority=integration_cfg.get('brain_priority', 10),
            brain_auto_register=integration_cfg.get('brain_auto_register', True),
            brain_memory_limit=integration_cfg.get('brain_memory_limit', 4 * 1024 * 1024 * 1024)
        )
    
    def get_hensel_config(self) -> HenselLiftingConfig:
        """
        Get HenselLiftingConfig object from current configuration.
        
        Returns:
            HenselLiftingConfig instance
        """
        hensel_cfg = self.get('padic.hensel_lifting', {})
        return HenselLiftingConfig(
            max_iterations=hensel_cfg.get('max_iterations', 50),
            convergence_tolerance=hensel_cfg.get('convergence_tolerance', 1e-12),
            damping_factor=hensel_cfg.get('damping_factor', 0.8),
            adaptive_damping=hensel_cfg.get('adaptive_damping', True),
            min_damping=hensel_cfg.get('min_damping', 0.1),
            max_damping=hensel_cfg.get('max_damping', 1.0),
            precision_schedule=hensel_cfg.get('precision_schedule'),
            enable_validation=hensel_cfg.get('enable_validation', True)
        )
    
    def get_clustering_config(self) -> ClusteringConfig:
        """
        Get ClusteringConfig object from current configuration.
        
        Returns:
            ClusteringConfig instance
        """
        cluster_cfg = self.get('padic.clustering', {})
        return ClusteringConfig(
            max_clusters=cluster_cfg.get('max_clusters', 100),
            min_cluster_size=cluster_cfg.get('min_cluster_size', 10),
            distance_threshold=cluster_cfg.get('distance_threshold', 0.1),
            adaptive_clustering=cluster_cfg.get('adaptive_clustering', True),
            rebalance_threshold=cluster_cfg.get('rebalance_threshold', 0.2),
            max_iterations=cluster_cfg.get('max_iterations', 100),
            convergence_tolerance=cluster_cfg.get('convergence_tolerance', 1e-6),
            enable_gpu_acceleration=cluster_cfg.get('enable_gpu_acceleration', True)
        )
    
    def get_gpu_config(self) -> GPUDecompressionConfig:
        """
        Get GPUDecompressionConfig object from current configuration.
        
        Returns:
            GPUDecompressionConfig instance
        """
        gpu_cfg = self.get('padic.gpu_decompression', {})
        return GPUDecompressionConfig(
            device_id=gpu_cfg.get('device_id', 0),
            memory_fraction=gpu_cfg.get('memory_fraction', 0.8),
            batch_size=gpu_cfg.get('batch_size', 32),
            enable_tensor_cores=gpu_cfg.get('enable_tensor_cores', True),
            stream_count=gpu_cfg.get('stream_count', 4),
            enable_memory_pooling=gpu_cfg.get('enable_memory_pooling', True),
            max_workspace_size=gpu_cfg.get('max_workspace_size', 1024 * 1024 * 1024),
            enable_mixed_precision=gpu_cfg.get('enable_mixed_precision', True)
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration tailored for Saraphis P-adic system."""
        return {
            'padic': {
                'default_p': 2999,
                'default_precision': 100,
                'max_precision': 500,
                'compression_ratio': 0.1,
                'enable_gpu': True,
                'cache_size': 1000,
                'hensel_lifting': {
                    'max_iterations': 50,
                    'convergence_tolerance': 1e-12,
                    'damping_factor': 0.8,
                    'adaptive_damping': True,
                    'min_damping': 0.1,
                    'max_damping': 1.0,
                    'enable_validation': True
                },
                'clustering': {
                    'max_clusters': 100,
                    'min_cluster_size': 10,
                    'distance_threshold': 0.1,
                    'adaptive_clustering': True,
                    'rebalance_threshold': 0.2,
                    'max_iterations': 100,
                    'convergence_tolerance': 1e-6,
                    'enable_gpu_acceleration': True
                },
                'gpu_decompression': {
                    'device_id': 0,
                    'memory_fraction': 0.85,          # Safe for 16GB card
                    'batch_size': 5000,               # Safe for 16GB VRAM
                    'enable_tensor_cores': True,
                    'stream_count': 16,               # Reduced for 16GB
                    'enable_memory_pooling': True,
                    'max_workspace_size': 2147483648, # 2GB workspace
                                    'enable_mixed_precision': True,
                'prefetch_factor': 2,             # Reduced prefetch
                'persistent_workers': True        # Keep workers alive
            },
            'cpu_decompression': {
                'batch_size': 50000,              # Massive CPU batches
                'num_workers': 16,                # Use all CPUs (hardcoded for safety)
                'chunk_multiplier': 10,           # Process multiple chunks
                'use_multiprocessing': True,      # Enable multiprocessing
                'cache_size_mb': 8192,            # 8GB cache
                'huge_pages': True,               # Use huge pages
                'numa_aware': True                # NUMA optimization
            },
            'memory_management': {
                'aggressive_gc': False,           # Disable GC during burst
                'huge_pages': True,               # Use huge pages
                'numa_aware': True,               # NUMA optimization
                'pinned_memory': True,            # Pinned memory transfers
                'memory_pool_count': 2,           # 2 pools for 16GB card
                'emergency_workers': 100          # Emergency worker pool
            }
            },
            'service': {
                'max_batch_size': 10000,              # 10x increase
                'request_timeout': 30,
                'enable_authentication': False,
                'enable_rate_limiting': False,
                'max_requests_per_minute': 600,        # 10x increase
                'enable_response_compression': True,
                'max_concurrent_requests': 1000,       # 10x increase
                'enable_caching': True,
                'cache_size_mb': 8192                  # 8x increase
            },
            'orchestration': {
                'max_queue_size': 500000,              # 10x increase
                'max_workers': 500,                    # 10x increase
                'load_balancing_strategy': 'least_loaded',
                'cache_ttl': 300,
                'history_size': 10000                  # 10x increase
            },
            'monitoring': {
                'health_check_interval': 30,
                'stale_service_timeout': 300,
                'metrics_aggregation_interval': 60,
                'time_series_window': 3600,
                'enable_metrics_collection': True,
                'metrics_buffer_size': 10000
            },
            'integration': {
                'brain_core_enabled': True,
                'training_manager_enabled': True,
                'gpu_memory_enabled': True,
                'gradient_compression_ratio': 0.05,
                'adaptive_precision': True,
                'precision_min': 50,
                'precision_max': 500,
                'precision_step': 50,
                'gac_compression_threshold': 0.001,
                'gac_async_processing': True,
                'brain_priority': 10,
                'brain_auto_register': True,
                'brain_memory_limit': 4294967296
            }
        }
    
    def _get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema with Saraphis-specific validations."""
        return {
            'type': 'object',
            'properties': {
                'padic': {
                    'type': 'object',
                    'properties': {
                        'default_p': {'type': 'integer', 'minimum': 2, 'maximum': 9973},
                        'default_precision': {'type': 'integer', 'minimum': 1, 'maximum': 500},
                        'max_precision': {'type': 'integer', 'minimum': 1, 'maximum': 1000},
                        'compression_ratio': {'type': 'number', 'minimum': 0.001, 'maximum': 0.999},
                        'enable_gpu': {'type': 'boolean'},
                        'cache_size': {'type': 'integer', 'minimum': 0},
                        'hensel_lifting': {
                            'type': 'object',
                            'properties': {
                                'max_iterations': {'type': 'integer', 'minimum': 1, 'maximum': 1000},
                                'convergence_tolerance': {'type': 'number', 'minimum': 1e-15, 'maximum': 1e-6},
                                'damping_factor': {'type': 'number', 'minimum': 0.01, 'maximum': 1.0},
                                'adaptive_damping': {'type': 'boolean'},
                                'min_damping': {'type': 'number', 'minimum': 0.01, 'maximum': 1.0},
                                'max_damping': {'type': 'number', 'minimum': 0.01, 'maximum': 1.0},
                                'enable_validation': {'type': 'boolean'}
                            }
                        },
                        'clustering': {
                            'type': 'object',
                            'properties': {
                                'max_clusters': {'type': 'integer', 'minimum': 1, 'maximum': 10000},
                                'min_cluster_size': {'type': 'integer', 'minimum': 1, 'maximum': 1000},
                                'distance_threshold': {'type': 'number', 'minimum': 0.001, 'maximum': 1.0},
                                'adaptive_clustering': {'type': 'boolean'},
                                'rebalance_threshold': {'type': 'number', 'minimum': 0.01, 'maximum': 1.0},
                                'max_iterations': {'type': 'integer', 'minimum': 1, 'maximum': 10000},
                                'convergence_tolerance': {'type': 'number', 'minimum': 1e-15, 'maximum': 1e-3},
                                'enable_gpu_acceleration': {'type': 'boolean'}
                            }
                        },
                        'gpu_decompression': {
                            'type': 'object',
                            'properties': {
                                'device_id': {'type': 'integer', 'minimum': 0, 'maximum': 16},
                                'memory_fraction': {'type': 'number', 'minimum': 0.1, 'maximum': 1.0},
                                'batch_size': {'type': 'integer', 'minimum': 1, 'maximum': 1024},
                                'enable_tensor_cores': {'type': 'boolean'},
                                'stream_count': {'type': 'integer', 'minimum': 1, 'maximum': 32},
                                'enable_memory_pooling': {'type': 'boolean'},
                                'max_workspace_size': {'type': 'integer', 'minimum': 1048576},
                                'enable_mixed_precision': {'type': 'boolean'}
                            }
                        }
                    }
                },
                'service': {
                    'type': 'object',
                    'properties': {
                        'max_batch_size': {'type': 'integer', 'minimum': 1, 'maximum': 1024},
                        'request_timeout': {'type': 'number', 'minimum': 0},
                        'enable_authentication': {'type': 'boolean'},
                        'enable_rate_limiting': {'type': 'boolean'},
                        'max_requests_per_minute': {'type': 'integer', 'minimum': 1},
                        'enable_response_compression': {'type': 'boolean'},
                        'max_concurrent_requests': {'type': 'integer', 'minimum': 1, 'maximum': 1000},
                        'enable_caching': {'type': 'boolean'},
                        'cache_size_mb': {'type': 'integer', 'minimum': 1}
                    }
                },
                'orchestration': {
                    'type': 'object',
                    'properties': {
                        'max_queue_size': {'type': 'integer', 'minimum': 1},
                        'max_workers': {'type': 'integer', 'minimum': 1, 'maximum': 100},
                        'load_balancing_strategy': {'type': 'string', 'enum': ['round_robin', 'least_loaded', 'random']},
                        'cache_ttl': {'type': 'integer', 'minimum': 0},
                        'history_size': {'type': 'integer', 'minimum': 0}
                    }
                },
                'monitoring': {
                    'type': 'object',
                    'properties': {
                        'health_check_interval': {'type': 'number', 'minimum': 1},
                        'stale_service_timeout': {'type': 'number', 'minimum': 1},
                        'metrics_aggregation_interval': {'type': 'number', 'minimum': 1},
                        'time_series_window': {'type': 'integer', 'minimum': 60},
                        'enable_metrics_collection': {'type': 'boolean'},
                        'metrics_buffer_size': {'type': 'integer', 'minimum': 1}
                    }
                },
                'integration': {
                    'type': 'object',
                    'properties': {
                        'brain_core_enabled': {'type': 'boolean'},
                        'training_manager_enabled': {'type': 'boolean'},
                        'gpu_memory_enabled': {'type': 'boolean'},
                        'gradient_compression_ratio': {'type': 'number', 'minimum': 0.001, 'maximum': 0.999},
                        'adaptive_precision': {'type': 'boolean'},
                        'precision_min': {'type': 'integer', 'minimum': 1, 'maximum': 1000},
                        'precision_max': {'type': 'integer', 'minimum': 1, 'maximum': 1000},
                        'precision_step': {'type': 'integer', 'minimum': 1, 'maximum': 100},
                        'gac_compression_threshold': {'type': 'number', 'minimum': 0.0001, 'maximum': 0.1},
                        'gac_async_processing': {'type': 'boolean'},
                        'brain_priority': {'type': 'integer', 'minimum': 1, 'maximum': 100},
                        'brain_auto_register': {'type': 'boolean'},
                        'brain_memory_limit': {'type': 'integer', 'minimum': 1048576}
                    }
                }
            }
        }
    
    def _validate_against_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate data against schema."""
        if schema.get('type') == 'object':
            if not isinstance(data, dict):
                return False
            
            properties = schema.get('properties', {})
            for key, prop_schema in properties.items():
                if key in data:
                    if not self._validate_property(data[key], prop_schema):
                        logger.error(f"Validation failed for property: {key}")
                        return False
        
        return True
    
    def _validate_property(self, value: Any, prop_schema: Dict[str, Any]) -> bool:
        """Validate a single property against its schema."""
        prop_type = prop_schema.get('type')
        
        # Type validation
        if prop_type == 'integer':
            if not isinstance(value, int):
                return False
            # Range validation
            if 'minimum' in prop_schema and value < prop_schema['minimum']:
                return False
            if 'maximum' in prop_schema and value > prop_schema['maximum']:
                return False
                
        elif prop_type == 'number':
            if not isinstance(value, (int, float)):
                return False
            # Range validation
            if 'minimum' in prop_schema and value < prop_schema['minimum']:
                return False
            if 'maximum' in prop_schema and value > prop_schema['maximum']:
                return False
                
        elif prop_type == 'boolean':
            if not isinstance(value, bool):
                return False
                
        elif prop_type == 'string':
            if not isinstance(value, str):
                return False
            # Enum validation
            if 'enum' in prop_schema and value not in prop_schema['enum']:
                return False
                
        elif prop_type == 'object':
            return self._validate_against_schema(value, prop_schema)
        
        return True
    
    def _save_to_history(self) -> None:
        """Save current configuration to history."""
        self.config_history.append({
            'config': self.config.copy(),
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def _notify_change(self, key: str, old_value: Any, new_value: Any) -> None:
        """Notify callbacks of configuration change."""
        for callback in self.change_callbacks:
            try:
                callback(key, old_value, new_value)
            except Exception as e:
                logger.error(f"Configuration change callback failed: {str(e)}")
    
    def _deep_update(self, target: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Deep update dictionary."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _get_nested_value(self, d: Dict[str, Any], key: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = key.split('.')
        value = d
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        return value


# Module initialization
logger.info("P-Adic service configuration module loaded successfully")