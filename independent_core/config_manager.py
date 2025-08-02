"""
Basic Configuration Manager for Independent Core
Minimal but functional configuration management system
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ConfigManagerSettings:
    """Configuration manager settings"""
    config_dir: str = "./config"
    environment: str = "dev"
    auto_create_dirs: bool = True
    default_format: str = "json"

class ConfigManager:
    """
    Basic configuration manager for independent core
    Provides essential configuration loading and management
    """
    
    def __init__(self, settings: Optional[ConfigManagerSettings] = None):
        """
        Initialize configuration manager
        
        Args:
            settings: Configuration manager settings
        """
        self.settings = settings or ConfigManagerSettings()
        self.config_dir = Path(self.settings.config_dir)
        self.environment = self.settings.environment
        
        # Configuration storage
        self.configs: Dict[str, Dict[str, Any]] = {}
        
        # Create config directory if needed
        if self.settings.auto_create_dirs:
            self._ensure_config_directories()
        
        logger.info(f"ConfigManager initialized for environment: {self.environment}")
    
    def _ensure_config_directories(self):
        """Ensure configuration directories exist"""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Create environment-specific directories
            env_dirs = ['dev', 'test', 'staging', 'prod']
            for env in env_dirs:
                (self.config_dir / env).mkdir(exist_ok=True)
            
            logger.debug("Configuration directories created")
        except Exception as e:
            logger.error(f"Failed to create config directories: {e}")
    
    def load_config(self, config_name: str, required: bool = False) -> Optional[Dict[str, Any]]:
        """
        Load configuration from file
        
        Args:
            config_name: Name of configuration to load
            required: Whether configuration is required
            
        Returns:
            Configuration dictionary or None
        """
        try:
            # Try environment-specific config first
            config_file = self.config_dir / self.environment / f"{config_name}.json"
            if not config_file.exists():
                # Fallback to base config
                config_file = self.config_dir / f"{config_name}.json"
            
            if not config_file.exists():
                if required:
                    raise FileNotFoundError(f"Required configuration not found: {config_name}")
                logger.warning(f"Configuration file not found: {config_file}")
                return None
            
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Apply environment variable overrides
            config_data = self._apply_env_overrides(config_name, config_data)
            
            # Store in cache
            self.configs[config_name] = config_data
            
            logger.info(f"Loaded configuration: {config_name}")
            return config_data
            
        except Exception as e:
            logger.error(f"Failed to load configuration {config_name}: {e}")
            if required:
                raise
            return None
    
    def save_config(self, config_name: str, config_data: Dict[str, Any], 
                   to_environment: bool = True) -> bool:
        """
        Save configuration to file
        
        Args:
            config_name: Name of configuration
            config_data: Configuration data to save
            to_environment: Save to environment-specific directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if to_environment:
                config_file = self.config_dir / self.environment / f"{config_name}.json"
            else:
                config_file = self.config_dir / f"{config_name}.json"
            
            # Ensure directory exists
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Update cache
            self.configs[config_name] = config_data
            
            logger.info(f"Saved configuration: {config_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration {config_name}: {e}")
            return False
    
    def get_config(self, config_name: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get configuration from cache or load if not cached
        
        Args:
            config_name: Name of configuration
            default: Default value if config not found
            
        Returns:
            Configuration dictionary
        """
        if config_name in self.configs:
            return self.configs[config_name].copy()
        
        config = self.load_config(config_name)
        return config if config is not None else (default or {})
    
    def update_config(self, config_name: str, updates: Dict[str, Any], 
                     save_immediately: bool = True) -> bool:
        """
        Update configuration with new values
        
        Args:
            config_name: Name of configuration
            updates: Updates to apply
            save_immediately: Whether to save changes immediately
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current config
            current_config = self.get_config(config_name)
            
            # Apply updates
            self._deep_update(current_config, updates)
            
            # Update cache
            self.configs[config_name] = current_config
            
            # Save if requested
            if save_immediately:
                return self.save_config(config_name, current_config)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration {config_name}: {e}")
            return False
    
    def get_setting(self, config_name: str, setting_path: str, 
                   default: Any = None) -> Any:
        """
        Get specific setting from configuration
        
        Args:
            config_name: Name of configuration
            setting_path: Dot-separated path to setting (e.g., "database.host")
            default: Default value if setting not found
            
        Returns:
            Setting value or default
        """
        config = self.get_config(config_name)
        if not config:
            return default
        
        try:
            value = config
            for key in setting_path.split('.'):
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_setting(self, config_name: str, setting_path: str, value: Any,
                   save_immediately: bool = True) -> bool:
        """
        Set specific setting in configuration
        
        Args:
            config_name: Name of configuration
            setting_path: Dot-separated path to setting
            value: Value to set
            save_immediately: Whether to save changes immediately
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config = self.get_config(config_name)
            
            # Navigate to parent of target setting
            keys = setting_path.split('.')
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the value
            current[keys[-1]] = value
            
            # Update cache
            self.configs[config_name] = config
            
            # Save if requested
            if save_immediately:
                return self.save_config(config_name, config)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set setting {setting_path} in {config_name}: {e}")
            return False
    
    def list_configs(self) -> list:
        """
        List all available configurations
        
        Returns:
            List of configuration names
        """
        config_files = []
        
        # Check environment directory
        env_dir = self.config_dir / self.environment
        if env_dir.exists():
            config_files.extend([f.stem for f in env_dir.glob("*.json")])
        
        # Check base directory
        if self.config_dir.exists():
            config_files.extend([f.stem for f in self.config_dir.glob("*.json")])
        
        return list(set(config_files))  # Remove duplicates
    
    def reload_config(self, config_name: str) -> bool:
        """
        Reload configuration from file
        
        Args:
            config_name: Name of configuration to reload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove from cache
            if config_name in self.configs:
                del self.configs[config_name]
            
            # Load fresh from file
            config = self.load_config(config_name)
            return config is not None
            
        except Exception as e:
            logger.error(f"Failed to reload configuration {config_name}: {e}")
            return False
    
    def _apply_env_overrides(self, config_name: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        result = config_data.copy()
        
        # Environment variable format: CONFIG_{CONFIG_NAME}_{SETTING_PATH}
        prefix = f"CONFIG_{config_name.upper()}_"
        
        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                setting_path = env_var[len(prefix):].lower().replace('_', '.')
                self._set_nested_value(result, setting_path, self._parse_env_value(value))
        
        return result
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any):
        """Set nested dictionary value using dot path"""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type"""
        # Try JSON parsing first
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Try boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Deep update dictionary with another dictionary"""
        for key, value in update_dict.items():
            if (key in base_dict and 
                isinstance(base_dict[key], dict) and 
                isinstance(value, dict)):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get_environment(self) -> str:
        """Get current environment"""
        return self.environment
    
    def set_environment(self, environment: str):
        """Set current environment"""
        self.environment = environment
        # Clear cache to force reload with new environment
        self.configs.clear()
        logger.info(f"Environment changed to: {environment}")
    
    def create_default_config(self, config_name: str, template: Dict[str, Any]) -> bool:
        """
        Create default configuration file if it doesn't exist
        
        Args:
            config_name: Name of configuration
            template: Default configuration template
            
        Returns:
            True if created or exists, False on error
        """
        try:
            config_file = self.config_dir / self.environment / f"{config_name}.json"
            
            if config_file.exists():
                logger.debug(f"Configuration already exists: {config_name}")
                return True
            
            return self.save_config(config_name, template)
            
        except Exception as e:
            logger.error(f"Failed to create default config {config_name}: {e}")
            return False

def create_config_manager(config_dir: str = "./config", 
                         environment: str = None) -> ConfigManager:
    """
    Create configuration manager with defaults
    
    Args:
        config_dir: Configuration directory
        environment: Environment name
        
    Returns:
        Configured ConfigManager instance
    """
    if environment is None:
        environment = os.getenv('ENVIRONMENT', 'dev')
    
    settings = ConfigManagerSettings(
        config_dir=config_dir,
        environment=environment,
        auto_create_dirs=True
    )
    
    return ConfigManager(settings)

# Default configuration templates
DEFAULT_CONFIGS = {
    'brain_system': {
        'base_path': './brain_data',
        'enable_persistence': True,
        'enable_monitoring': True,
        'max_domains': 10,
        'max_memory_gb': 4.0,
        'log_level': 'INFO'
    },
    'accuracy_tracking': {
        'check_interval_seconds': 30,
        'alert_thresholds': {
            'response_time_ms': 5000,
            'error_rate': 0.05,
            'cpu_usage': 80.0,
            'memory_usage': 85.0
        },
        'enable_alerts': True,
        'max_alerts': 1000
    }
}

if __name__ == "__main__":
    # Example usage
    config_manager = create_config_manager()
    
    # Create default configs
    for name, template in DEFAULT_CONFIGS.items():
        config_manager.create_default_config(name, template)
    
    # Test loading
    brain_config = config_manager.get_config('brain_system')
    print(f"Brain config loaded: {bool(brain_config)}")
    
    # Test setting
    config_manager.set_setting('brain_system', 'log_level', 'DEBUG')
    log_level = config_manager.get_setting('brain_system', 'log_level')
    print(f"Log level: {log_level}")
    
    print("Basic config manager ready for use")