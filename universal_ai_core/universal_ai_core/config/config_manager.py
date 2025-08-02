#!/usr/bin/env python3
"""
Universal Configuration Manager
===============================

This module provides comprehensive configuration management capabilities for the Universal AI Core system.
Adapted from Saraphis configuration patterns, providing enterprise-grade configuration management with
validation, persistence, hot-reloading, and environment-based configuration.

Features:
- Hierarchical configuration with type safety and validation
- Multiple configuration sources (JSON, environment variables, programmatic)
- Hot-reloading and runtime configuration updates
- Configuration inheritance and merging
- Version management and migration
- Thread-safe configuration access
- Configuration caching and optimization
- Comprehensive error handling and recovery
"""

import logging
import json
import os
import time
import hashlib
import threading
import copy
import weakref
from typing import Dict, List, Any, Optional, Type, Union, Callable, Set
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime, timedelta
from pathlib import Path
from abc import ABC, abstractmethod
import inspect
from collections import defaultdict

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Configuration-related errors"""
    pass


class ConfigurationValidationError(ConfigurationError):
    """Configuration validation errors"""
    pass


class ConfigurationNotFoundError(ConfigurationError):
    """Configuration not found errors"""
    pass


@dataclass
class ConfigMetadata:
    """Metadata for configuration tracking"""
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    source: str = "default"
    hash: str = ""
    environment: str = "development"
    
    def update_modified(self):
        """Update modification timestamp"""
        self.modified_at = datetime.utcnow()


class ConfigurationValidator:
    """Configuration validation engine"""
    
    def __init__(self):
        self.validators: Dict[str, List[Callable]] = defaultdict(list)
        self.cross_validators: List[Callable] = []
    
    def add_field_validator(self, field_name: str, validator: Callable):
        """Add validator for specific field"""
        self.validators[field_name].append(validator)
    
    def add_cross_validator(self, validator: Callable):
        """Add cross-field validator"""
        self.cross_validators.append(validator)
    
    def validate_field(self, field_name: str, value: Any, config_obj: Any = None) -> bool:
        """Validate specific field"""
        for validator in self.validators[field_name]:
            try:
                if not validator(value, config_obj):
                    return False
            except Exception as e:
                logger.error(f"Validation error for field {field_name}: {e}")
                return False
        return True
    
    def validate_config(self, config_obj: Any) -> bool:
        """Validate entire configuration object"""
        # Validate individual fields
        for field in fields(config_obj):
            field_value = getattr(config_obj, field.name)
            if not self.validate_field(field.name, field_value, config_obj):
                return False
        
        # Validate cross-field dependencies
        for validator in self.cross_validators:
            try:
                if not validator(config_obj):
                    return False
            except Exception as e:
                logger.error(f"Cross-validation error: {e}")
                return False
        
        return True


class ConfigurationWatcher:
    """Configuration file watcher for hot-reloading"""
    
    def __init__(self, callback: Callable):
        self.callback = callback
        self.watched_files: Dict[str, float] = {}
        self.watching = False
        self.watch_thread = None
        self._lock = threading.Lock()
    
    def add_file(self, filepath: str):
        """Add file to watch list"""
        if os.path.exists(filepath):
            self.watched_files[filepath] = os.path.getmtime(filepath)
    
    def start_watching(self, interval: float = 1.0):
        """Start watching for file changes"""
        if self.watching:
            return
        
        self.watching = True
        self.watch_thread = threading.Thread(
            target=self._watch_loop, 
            args=(interval,), 
            daemon=True
        )
        self.watch_thread.start()
    
    def stop_watching(self):
        """Stop watching for file changes"""
        self.watching = False
        if self.watch_thread:
            self.watch_thread.join(timeout=2.0)
    
    def _watch_loop(self, interval: float):
        """Main watch loop"""
        while self.watching:
            try:
                with self._lock:
                    for filepath, last_mtime in list(self.watched_files.items()):
                        if os.path.exists(filepath):
                            current_mtime = os.path.getmtime(filepath)
                            if current_mtime > last_mtime:
                                logger.info(f"Configuration file changed: {filepath}")
                                self.watched_files[filepath] = current_mtime
                                try:
                                    self.callback(filepath)
                                except Exception as e:
                                    logger.error(f"Error in config reload callback: {e}")
                        else:
                            logger.warning(f"Watched file no longer exists: {filepath}")
                            del self.watched_files[filepath]
                
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in configuration watcher: {e}")
                time.sleep(interval)


class BaseConfiguration(ABC):
    """Abstract base class for all configuration objects"""
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        self.validate()
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate configuration"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if hasattr(value, 'to_dict'):
                result[field.name] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                result[field.name] = [
                    item.to_dict() if hasattr(item, 'to_dict') else item
                    for item in value
                ]
            elif isinstance(value, dict):
                result[field.name] = {
                    k: v.to_dict() if hasattr(v, 'to_dict') else v
                    for k, v in value.items()
                }
            else:
                result[field.name] = value
        return result
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """Convert configuration to JSON"""
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=2, default=str)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
            logger.info(f"Configuration saved to {filepath}")
        
        return json_str
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create configuration from dictionary"""
        # Get field information for the class
        class_fields = {f.name: f for f in fields(cls)}
        filtered_dict = {}
        
        for key, value in config_dict.items():
            if key in class_fields:
                field_info = class_fields[key]
                
                # Handle nested configuration objects
                if hasattr(field_info.type, 'from_dict'):
                    filtered_dict[key] = field_info.type.from_dict(value)
                elif hasattr(field_info.type, '__origin__'):  # Handle typing generics
                    filtered_dict[key] = value
                else:
                    filtered_dict[key] = value
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        return cls(**filtered_dict)
    
    @classmethod
    def from_json(cls, filepath: str):
        """Load configuration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except FileNotFoundError:
            raise ConfigurationNotFoundError(f"Configuration file not found: {filepath}")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration with values from dictionary"""
        def update_nested(obj, updates_dict):
            for key, value in updates_dict.items():
                if hasattr(obj, key):
                    attr = getattr(obj, key)
                    if hasattr(attr, '__dict__') and isinstance(value, dict):
                        update_nested(attr, value)
                    else:
                        try:
                            setattr(obj, key, value)
                        except Exception as e:
                            logger.error(f"Failed to update {key}: {e}")
                else:
                    logger.warning(f"Unknown configuration key: {key}")
        
        update_nested(self, updates)
        self.validate()
    
    def get_hash(self) -> str:
        """Get unique hash of the configuration"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def copy(self):
        """Create deep copy of configuration"""
        return copy.deepcopy(self)
    
    def merge(self, other: 'BaseConfiguration') -> 'BaseConfiguration':
        """Merge with another configuration"""
        if not isinstance(other, self.__class__):
            raise ConfigurationError(f"Cannot merge {type(other)} with {type(self)}")
        
        merged_dict = self.to_dict()
        other_dict = other.to_dict()
        
        def deep_merge(base: dict, update: dict) -> dict:
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base
        
        merged = deep_merge(merged_dict, other_dict)
        return self.__class__.from_dict(merged)


@dataclass
class EnvironmentConfig(BaseConfiguration):
    """Environment-specific configuration"""
    environment: str = "development"
    debug_mode: bool = True
    log_level: str = "INFO"
    enable_monitoring: bool = False
    max_memory_gb: float = 8.0
    cache_size_mb: int = 512
    worker_threads: int = 4
    
    def validate(self) -> bool:
        """Validate environment configuration"""
        if self.environment not in ["development", "staging", "production"]:
            raise ConfigurationValidationError(f"Invalid environment: {self.environment}")
        
        if self.max_memory_gb <= 0:
            raise ConfigurationValidationError("max_memory_gb must be positive")
        
        if self.cache_size_mb <= 0:
            raise ConfigurationValidationError("cache_size_mb must be positive")
        
        if self.worker_threads <= 0:
            raise ConfigurationValidationError("worker_threads must be positive")
        
        return True


@dataclass
class SecurityConfig(BaseConfiguration):
    """Security configuration"""
    enable_encryption: bool = True
    require_authentication: bool = True
    session_timeout_minutes: int = 30
    max_login_attempts: int = 3
    encryption_algorithm: str = "AES-256"
    hash_algorithm: str = "SHA-256"
    secure_headers: bool = True
    
    def validate(self) -> bool:
        """Validate security configuration"""
        if self.session_timeout_minutes <= 0:
            raise ConfigurationValidationError("session_timeout_minutes must be positive")
        
        if self.max_login_attempts <= 0:
            raise ConfigurationValidationError("max_login_attempts must be positive")
        
        valid_algorithms = ["AES-256", "AES-128", "ChaCha20"]
        if self.encryption_algorithm not in valid_algorithms:
            raise ConfigurationValidationError(f"Invalid encryption algorithm: {self.encryption_algorithm}")
        
        return True


@dataclass
class PerformanceConfig(BaseConfiguration):
    """Performance configuration"""
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    batch_size: int = 32
    parallel_processing: bool = True
    memory_limit_mb: int = 1024
    request_timeout_seconds: int = 30
    retry_attempts: int = 3
    
    def validate(self) -> bool:
        """Validate performance configuration"""
        if self.cache_ttl_seconds < 0:
            raise ConfigurationValidationError("cache_ttl_seconds cannot be negative")
        
        if self.batch_size <= 0:
            raise ConfigurationValidationError("batch_size must be positive")
        
        if self.memory_limit_mb <= 0:
            raise ConfigurationValidationError("memory_limit_mb must be positive")
        
        if self.request_timeout_seconds <= 0:
            raise ConfigurationValidationError("request_timeout_seconds must be positive")
        
        return True


@dataclass
class UniversalConfiguration(BaseConfiguration):
    """Universal AI Core configuration"""
    environment_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    security_config: SecurityConfig = field(default_factory=SecurityConfig)
    performance_config: PerformanceConfig = field(default_factory=PerformanceConfig)
    metadata: ConfigMetadata = field(default_factory=ConfigMetadata)
    plugin_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate universal configuration"""
        # All nested configurations will validate themselves via __post_init__
        return True
    
    def get_plugin_config(self, plugin_name: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get configuration for specific plugin"""
        return self.plugin_configs.get(plugin_name, default or {})
    
    def set_plugin_config(self, plugin_name: str, config: Dict[str, Any]):
        """Set configuration for specific plugin"""
        self.plugin_configs[plugin_name] = config
        self.metadata.update_modified()
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment_config.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment_config.environment == "development"


class ConfigurationManager:
    """
    Universal Configuration Manager
    
    Provides comprehensive configuration management with validation,
    persistence, hot-reloading, and environment-based configuration.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self._config: Optional[UniversalConfiguration] = None
        self._config_lock = threading.RLock()
        self._observers: List[weakref.ReferenceType] = []
        self._watcher: Optional[ConfigurationWatcher] = None
        self._validator = ConfigurationValidator()
        
        # Setup default validators
        self._setup_default_validators()
        
        # Load configuration
        self._load_configuration()
    
    def _setup_default_validators(self):
        """Setup default configuration validators"""
        # Environment validators
        self._validator.add_field_validator(
            "environment",
            lambda value, config: value in ["development", "staging", "production"]
        )
        
        # Memory validators
        self._validator.add_field_validator(
            "max_memory_gb",
            lambda value, config: isinstance(value, (int, float)) and value > 0
        )
        
        # Cross-field validators
        self._validator.add_cross_validator(
            lambda config: not (config.environment_config.environment == "production" and config.environment_config.debug_mode)
        )
    
    def _load_configuration(self):
        """Load configuration from various sources"""
        # Try to load from file first
        config_file = self.config_dir / "config.json"
        
        if config_file.exists():
            try:
                self._config = UniversalConfiguration.from_json(str(config_file))
                logger.info(f"Configuration loaded from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load configuration from file: {e}")
                self._config = self._create_default_config()
        else:
            # Try environment variables
            env_config = self._load_from_environment()
            if env_config:
                self._config = env_config
                logger.info("Configuration loaded from environment variables")
            else:
                # Fall back to defaults
                self._config = self._create_default_config()
                logger.info("Using default configuration")
        
        # Setup file watcher for hot-reloading
        if config_file.exists():
            self._setup_watcher(str(config_file))
    
    def _create_default_config(self) -> UniversalConfiguration:
        """Create default configuration"""
        return UniversalConfiguration()
    
    def _load_from_environment(self, prefix: str = "UNIVERSAL_AI_") -> Optional[UniversalConfiguration]:
        """Load configuration from environment variables"""
        try:
            config = UniversalConfiguration()
            
            env_mappings = {
                "ENVIRONMENT": ("environment_config.environment", str),
                "DEBUG": ("environment_config.debug_mode", lambda x: x.lower() in ('true', '1', 'yes', 'on')),
                "LOG_LEVEL": ("environment_config.log_level", str),
                "MAX_MEMORY_GB": ("environment_config.max_memory_gb", float),
                "CACHE_SIZE_MB": ("environment_config.cache_size_mb", int),
                "WORKER_THREADS": ("environment_config.worker_threads", int),
                "ENABLE_ENCRYPTION": ("security_config.enable_encryption", lambda x: x.lower() in ('true', '1', 'yes', 'on')),
                "SESSION_TIMEOUT": ("security_config.session_timeout_minutes", int),
                "BATCH_SIZE": ("performance_config.batch_size", int),
                "MEMORY_LIMIT_MB": ("performance_config.memory_limit_mb", int),
            }
            
            found_any = False
            for env_suffix, (attr_path, type_func) in env_mappings.items():
                env_var = f"{prefix}{env_suffix}"
                env_value = os.getenv(env_var)
                
                if env_value is not None:
                    try:
                        value = type_func(env_value)
                        
                        # Set nested attribute
                        obj = config
                        attrs = attr_path.split('.')
                        for attr in attrs[:-1]:
                            obj = getattr(obj, attr)
                        setattr(obj, attrs[-1], value)
                        found_any = True
                        
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Failed to set {attr_path} from {env_var}: {e}")
            
            return config if found_any else None
            
        except Exception as e:
            logger.error(f"Error loading configuration from environment: {e}")
            return None
    
    def _setup_watcher(self, config_file: str):
        """Setup configuration file watcher"""
        if self._watcher:
            self._watcher.stop_watching()
        
        self._watcher = ConfigurationWatcher(self._on_config_file_changed)
        self._watcher.add_file(config_file)
        self._watcher.start_watching()
    
    def _on_config_file_changed(self, filepath: str):
        """Handle configuration file changes"""
        try:
            new_config = UniversalConfiguration.from_json(filepath)
            
            with self._config_lock:
                old_config = self._config
                self._config = new_config
            
            logger.info("Configuration reloaded from file")
            self._notify_observers(old_config, new_config)
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
    
    def _notify_observers(self, old_config: UniversalConfiguration, new_config: UniversalConfiguration):
        """Notify configuration change observers"""
        # Clean up dead references
        self._observers = [ref for ref in self._observers if ref() is not None]
        
        for observer_ref in self._observers:
            observer = observer_ref()
            if observer and hasattr(observer, 'on_config_changed'):
                try:
                    observer.on_config_changed(old_config, new_config)
                except Exception as e:
                    logger.error(f"Error notifying observer: {e}")
    
    def get_config(self) -> UniversalConfiguration:
        """Get current configuration"""
        with self._config_lock:
            return self._config.copy()
    
    def update_config(self, updates: Dict[str, Any], persist: bool = True) -> bool:
        """Update configuration with new values"""
        try:
            with self._config_lock:
                old_config = self._config.copy()
                self._config.update_from_dict(updates)
                
                # Validate updated configuration
                if not self._validator.validate_config(self._config):
                    self._config = old_config  # Rollback
                    return False
                
                self._config.metadata.update_modified()
                
                if persist:
                    self.save_config()
                
                self._notify_observers(old_config, self._config)
                return True
                
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False
    
    def save_config(self, filepath: Optional[str] = None) -> bool:
        """Save configuration to file"""
        try:
            if not filepath:
                filepath = str(self.config_dir / "config.json")
            
            with self._config_lock:
                self._config.to_json(filepath)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def add_observer(self, observer) -> None:
        """Add configuration change observer"""
        self._observers.append(weakref.ref(observer))
    
    def remove_observer(self, observer) -> None:
        """Remove configuration change observer"""
        self._observers = [ref for ref in self._observers if ref() != observer]
    
    def create_environment_config(self, environment: str) -> UniversalConfiguration:
        """Create configuration for specific environment"""
        base_config = self._create_default_config()
        
        if environment == "production":
            base_config.environment_config.environment = "production"
            base_config.environment_config.debug_mode = False
            base_config.environment_config.log_level = "WARNING"
            base_config.environment_config.enable_monitoring = True
            base_config.security_config.require_authentication = True
            base_config.performance_config.parallel_processing = True
            
        elif environment == "staging":
            base_config.environment_config.environment = "staging"
            base_config.environment_config.debug_mode = False
            base_config.environment_config.log_level = "INFO"
            base_config.environment_config.enable_monitoring = True
            
        elif environment == "development":
            base_config.environment_config.environment = "development"
            base_config.environment_config.debug_mode = True
            base_config.environment_config.log_level = "DEBUG"
            base_config.security_config.require_authentication = False
        
        return base_config
    
    def validate_current_config(self) -> bool:
        """Validate current configuration"""
        with self._config_lock:
            return self._validator.validate_config(self._config)
    
    def get_config_hash(self) -> str:
        """Get hash of current configuration"""
        with self._config_lock:
            return self._config.get_hash()
    
    def backup_config(self, backup_dir: Optional[str] = None) -> str:
        """Create backup of current configuration"""
        if not backup_dir:
            backup_dir = self.config_dir / "backups"
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path / f"config_backup_{timestamp}.json"
        
        if self.save_config(str(backup_file)):
            logger.info(f"Configuration backed up to {backup_file}")
            return str(backup_file)
        else:
            raise ConfigurationError("Failed to create configuration backup")
    
    def shutdown(self):
        """Shutdown configuration manager"""
        if self._watcher:
            self._watcher.stop_watching()
        
        # Clear observers
        self._observers.clear()
        
        logger.info("Configuration manager shutdown complete")


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None
_manager_lock = threading.Lock()


def get_config_manager(config_dir: Optional[str] = None) -> ConfigurationManager:
    """Get global configuration manager instance"""
    global _config_manager
    
    if _config_manager is None:
        with _manager_lock:
            if _config_manager is None:
                _config_manager = ConfigurationManager(config_dir)
    
    return _config_manager


def get_config() -> UniversalConfiguration:
    """Get current configuration"""
    return get_config_manager().get_config()


def update_config(updates: Dict[str, Any], persist: bool = True) -> bool:
    """Update global configuration"""
    return get_config_manager().update_config(updates, persist)


# Export public API
__all__ = [
    'ConfigurationManager', 'UniversalConfiguration', 'EnvironmentConfig',
    'SecurityConfig', 'PerformanceConfig', 'BaseConfiguration',
    'ConfigurationError', 'ConfigurationValidationError', 'ConfigurationNotFoundError',
    'get_config_manager', 'get_config', 'update_config'
]