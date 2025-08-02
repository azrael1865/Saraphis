"""
Configuration Manager for Financial Fraud Detection Domain
Enhanced configuration management with comprehensive validation,
multi-environment support, validation, security, and hot-reload capabilities.
Integrated with enhanced configuration management system.
"""

import json
import os
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from cryptography.fernet import Fernet
from jsonschema import validate, ValidationError
import yaml
import copy

# Import enhanced configuration components
from enhanced_config_manager import (
    EnhancedConfigManager, ValidationLevel, ConfigValidationError,
    ConfigLoadError, ConfigSecurityError, ConfigCompatibilityError
)
from enhanced_domain_config import (
    DomainConfig, FinancialFraudConfig, EnvironmentType, SecurityLevel
)

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ConfigMetadata:
    """Configuration metadata tracking"""
    version: str
    created_at: str
    modified_at: str
    environment: str
    checksum: str
    backup_count: int = 0

class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration hot-reload"""
    
    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager
        self.debounce_timer = None
        self.debounce_delay = 1.0  # 1 second debounce
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        if file_path.suffix in ['.json', '.yaml', '.yml'] and not file_path.name.startswith('.'):
            self._debounced_reload(file_path)
    
    def _debounced_reload(self, file_path: Path):
        """Debounced configuration reload"""
        if self.debounce_timer:
            self.debounce_timer.cancel()
        
        self.debounce_timer = threading.Timer(
            self.debounce_delay,
            lambda: self.config_manager._handle_file_change(file_path)
        )
        self.debounce_timer.start()

class ConfigManager:
    """
    Enhanced configuration manager for fraud detection domain
    
    Features:
    - Multi-environment support (dev, test, staging, prod)
    - Environment variable overrides
    - Enhanced configuration validation with comprehensive error handling
    - Encryption for sensitive data
    - Hot-reload capabilities
    - Configuration versioning and backup
    - Thread-safe operations
    - Integration with enhanced validation system
    """
    
    def __init__(self, 
                 config_dir: Optional[str] = None,
                 environment: Optional[str] = None,
                 enable_hot_reload: bool = True,
                 enable_encryption: bool = True,
                 validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """
        Initialize enhanced configuration manager
        
        Args:
            config_dir: Directory containing configuration files
            environment: Current environment (dev/test/staging/prod)
            enable_hot_reload: Enable automatic config reloading on file changes
            enable_encryption: Enable encryption for sensitive configuration data
            validation_level: Level of validation to apply (MINIMAL, STANDARD, STRICT, PARANOID)
        """
        self.config_dir = Path(config_dir) if config_dir else Path("./config")
        self.environment = environment or os.getenv('FRAUD_DETECTION_ENV', 'dev')
        self.enable_hot_reload = enable_hot_reload
        self.enable_encryption = enable_encryption
        self.validation_level = validation_level
        
        # Initialize enhanced configuration manager
        self.enhanced_manager = EnhancedConfigManager(
            config_dir=str(self.config_dir),
            environment=self.environment,
            validation_level=validation_level,
            enable_encryption=enable_encryption,
            enable_recovery=True
        )
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Configuration storage
        self.configs: Dict[str, Any] = {}
        self.metadata: Dict[str, ConfigMetadata] = {}
        self.schemas: Dict[str, Dict] = {}
        self.change_callbacks: List[Callable[[str, Dict], None]] = []
        
        # Encryption
        self._cipher_suite = None
        if self.enable_encryption:
            self._initialize_encryption()
        
        # File watching
        self.observer = None
        self.file_handler = None
        
        # Create config directory structure
        self._ensure_config_structure()
        
        # Load environment-specific configurations
        self._load_environment_configs()
        
        # Start file watching if enabled
        if self.enable_hot_reload:
            self._start_file_watching()
        
        logger.info(f"ConfigManager initialized for environment: {self.environment}")
    
    def _initialize_encryption(self):
        """Initialize encryption for sensitive configuration data"""
        key_file = self.config_dir / ".encryption_key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            key_file.parent.mkdir(parents=True, exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            # Secure the key file
            os.chmod(key_file, 0o600)
        
        self._cipher_suite = Fernet(key)
        logger.debug("Encryption initialized")
    
    def _ensure_config_structure(self):
        """Ensure proper configuration directory structure"""
        directories = [
            self.config_dir,
            self.config_dir / "environments",
            self.config_dir / "schemas",
            self.config_dir / "backups",
            self.config_dir / "templates"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create environment-specific directories
        environments = ['dev', 'test', 'staging', 'prod']
        for env in environments:
            (self.config_dir / "environments" / env).mkdir(parents=True, exist_ok=True)
    
    def _load_environment_configs(self):
        """Load configurations for current environment"""
        env_dir = self.config_dir / "environments" / self.environment
        
        if not env_dir.exists():
            logger.warning(f"Environment directory not found: {env_dir}")
            return
        
        # Load all configuration files in environment directory
        for config_file in env_dir.glob("*.{json,yaml,yml}"):
            config_name = config_file.stem
            try:
                self.load_config(config_name, from_environment=True)
            except Exception as e:
                logger.error(f"Failed to load config {config_name}: {e}")
    
    def _start_file_watching(self):
        """Start file system watching for hot-reload"""
        if self.observer:
            return
        
        self.file_handler = ConfigFileHandler(self)
        self.observer = Observer()
        self.observer.schedule(
            self.file_handler,
            str(self.config_dir),
            recursive=True
        )
        self.observer.start()
        logger.debug("File watching started for hot-reload")
    
    def _stop_file_watching(self):
        """Stop file system watching"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            self.file_handler = None
            logger.debug("File watching stopped")
    
    def _handle_file_change(self, file_path: Path):
        """Handle configuration file changes"""
        try:
            relative_path = file_path.relative_to(self.config_dir)
            
            # Determine if this is an environment-specific config
            if relative_path.parts[0] == "environments":
                if len(relative_path.parts) >= 3 and relative_path.parts[1] == self.environment:
                    config_name = relative_path.parts[2].split('.')[0]
                    self.reload_config(config_name)
                    logger.info(f"Hot-reloaded config: {config_name}")
            else:
                # Handle base configuration files
                config_name = file_path.stem
                self.reload_config(config_name)
                logger.info(f"Hot-reloaded config: {config_name}")
                
        except Exception as e:
            logger.error(f"Error handling file change for {file_path}: {e}")
    
    def add_schema(self, config_name: str, schema: Dict[str, Any]) -> None:
        """
        Add validation schema for a configuration
        
        Args:
            config_name: Name of the configuration
            schema: JSON schema for validation
        """
        with self._lock:
            self.schemas[config_name] = schema
            
            # Save schema to file
            schema_file = self.config_dir / "schemas" / f"{config_name}.json"
            with open(schema_file, 'w') as f:
                json.dump(schema, f, indent=2)
            
            logger.debug(f"Added schema for config: {config_name}")
    
    def load_config(self, 
                   config_name: str, 
                   from_environment: bool = True,
                   apply_env_overrides: bool = True) -> Dict[str, Any]:
        """
        Load configuration from file with enhanced validation
        
        Args:
            config_name: Name of the configuration to load
            from_environment: Load from environment-specific directory
            apply_env_overrides: Apply environment variable overrides
            
        Returns:
            Loaded configuration dictionary
        """
        with self._lock:
            try:
                # Try enhanced manager first
                config_data = self.enhanced_manager.load_config(
                    config_name, 
                    required=False, 
                    use_defaults=True
                )
                
                if config_data:
                    # Store in legacy format for backward compatibility
                    self.configs[config_name] = config_data
                    
                    # Create/update metadata
                    self.metadata[config_name] = ConfigMetadata(
                        version=self._generate_version(),
                        created_at=datetime.now().isoformat(),
                        modified_at=datetime.now().isoformat(),
                        environment=self.environment,
                        checksum=self._calculate_checksum(config_data)
                    )
                    
                    logger.info(f"Loaded configuration with enhanced validation: {config_name}")
                    return copy.deepcopy(config_data)
                
                # Fallback to legacy loading
                return self._legacy_load_config(config_name, from_environment, apply_env_overrides)
                
            except ConfigLoadError as e:
                logger.error(f"Failed to load configuration {config_name}: {e}")
                raise
            except Exception as e:
                logger.warning(f"Enhanced loading failed for {config_name}, trying legacy: {e}")
                return self._legacy_load_config(config_name, from_environment, apply_env_overrides)
    
    def _legacy_load_config(self, 
                           config_name: str, 
                           from_environment: bool = True,
                           apply_env_overrides: bool = True) -> Dict[str, Any]:
        """Legacy configuration loading method"""
        # Determine config file path
        if from_environment:
            config_file = self.config_dir / "environments" / self.environment / f"{config_name}.json"
            if not config_file.exists():
                # Fallback to YAML
                config_file = self.config_dir / "environments" / self.environment / f"{config_name}.yaml"
        else:
            config_file = self.config_dir / f"{config_name}.json"
            if not config_file.exists():
                config_file = self.config_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        # Load configuration data
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix == '.json':
                    config_data = json.load(f)
                else:
                    config_data = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to parse configuration file {config_file}: {e}")
        
        # Decrypt sensitive fields if encryption is enabled
        if self.enable_encryption:
            config_data = self._decrypt_sensitive_fields(config_data)
        
        # Apply environment variable overrides
        if apply_env_overrides:
            config_data = self._apply_env_overrides(config_name, config_data)
        
        # Validate against schema if available
        if config_name in self.schemas:
            self._validate_config(config_name, config_data)
        
        # Store configuration
        self.configs[config_name] = config_data
        
        # Create/update metadata
        self.metadata[config_name] = ConfigMetadata(
            version=self._generate_version(),
            created_at=datetime.now().isoformat(),
            modified_at=datetime.now().isoformat(),
            environment=self.environment,
            checksum=self._calculate_checksum(config_data)
        )
        
        logger.info(f"Loaded configuration (legacy): {config_name}")
        return copy.deepcopy(config_data)
    
    def save_config(self, 
                   config_name: str, 
                   config_data: Dict[str, Any],
                   to_environment: bool = True,
                   create_backup: bool = True) -> bool:
        """
        Save configuration to file with enhanced validation
        
        Args:
            config_name: Name of the configuration
            config_data: Configuration data to save
            to_environment: Save to environment-specific directory
            create_backup: Create backup before saving
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                # Use enhanced manager for validation and saving
                result = self.enhanced_manager.save_config(config_name, config_data)
                
                if result:
                    # Update legacy storage for backward compatibility
                    self.configs[config_name] = copy.deepcopy(config_data)
                    
                    # Update metadata
                    if config_name in self.metadata:
                        self.metadata[config_name].modified_at = datetime.now().isoformat()
                        self.metadata[config_name].checksum = self._calculate_checksum(config_data)
                    else:
                        self.metadata[config_name] = ConfigMetadata(
                            version=self._generate_version(),
                            created_at=datetime.now().isoformat(),
                            modified_at=datetime.now().isoformat(),
                            environment=self.environment,
                            checksum=self._calculate_checksum(config_data)
                        )
                    
                    # Notify change callbacks
                    self._notify_change_callbacks(config_name, config_data)
                    
                    logger.info(f"Saved configuration with enhanced validation: {config_name}")
                
                return result
                
            except ConfigValidationError as e:
                logger.error(f"Configuration validation failed for {config_name}: {e}")
                return False
            except ConfigSecurityError as e:
                logger.error(f"Security validation failed for {config_name}: {e}")
                return False
            except Exception as e:
                logger.error(f"Failed to save configuration {config_name}: {e}")
                return False
    
    def get_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Get loaded configuration
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Configuration dictionary or None if not found
        """
        with self._lock:
            return copy.deepcopy(self.configs.get(config_name))
    
    def update_config(self, 
                     config_name: str, 
                     updates: Dict[str, Any],
                     save_immediately: bool = True) -> bool:
        """
        Update configuration with new values
        
        Args:
            config_name: Name of the configuration
            updates: Dictionary of updates to apply
            save_immediately: Whether to save changes immediately
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if config_name not in self.configs:
                logger.error(f"Configuration not found: {config_name}")
                return False
            
            try:
                # Apply updates
                updated_config = copy.deepcopy(self.configs[config_name])
                self._deep_update(updated_config, updates)
                
                # Validate updated configuration
                if config_name in self.schemas:
                    self._validate_config(config_name, updated_config)
                
                # Update in-memory configuration
                self.configs[config_name] = updated_config
                
                # Save if requested
                if save_immediately:
                    return self.save_config(config_name, updated_config)
                
                # Update metadata
                if config_name in self.metadata:
                    self.metadata[config_name].modified_at = datetime.now().isoformat()
                    self.metadata[config_name].checksum = self._calculate_checksum(updated_config)
                
                # Notify change callbacks
                self._notify_change_callbacks(config_name, updated_config)
                
                logger.info(f"Updated configuration: {config_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update configuration {config_name}: {e}")
                return False
    
    def reload_config(self, config_name: str) -> bool:
        """
        Reload configuration from file
        
        Args:
            config_name: Name of the configuration to reload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            old_config = self.configs.get(config_name)
            new_config = self.load_config(config_name)
            
            # Check if configuration actually changed
            if old_config and self._calculate_checksum(old_config) == self._calculate_checksum(new_config):
                logger.debug(f"Configuration unchanged, skipping reload: {config_name}")
                return True
            
            # Notify change callbacks
            self._notify_change_callbacks(config_name, new_config)
            
            logger.info(f"Reloaded configuration: {config_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload configuration {config_name}: {e}")
            return False
    
    def validate_config(self, config_name: str) -> bool:
        """
        Validate configuration against its schema
        
        Args:
            config_name: Name of the configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        with self._lock:
            if config_name not in self.configs:
                logger.error(f"Configuration not found: {config_name}")
                return False
            
            if config_name not in self.schemas:
                logger.warning(f"No schema available for configuration: {config_name}")
                return True
            
            try:
                self._validate_config(config_name, self.configs[config_name])
                return True
            except ValidationError as e:
                logger.error(f"Configuration validation failed for {config_name}: {e}")
                return False
    
    def add_change_callback(self, callback: Callable[[str, Dict], None]) -> None:
        """
        Add callback for configuration changes
        
        Args:
            callback: Function to call when configuration changes
        """
        self.change_callbacks.append(callback)
        logger.debug("Added configuration change callback")
    
    def remove_change_callback(self, callback: Callable[[str, Dict], None]) -> None:
        """
        Remove configuration change callback
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
            logger.debug("Removed configuration change callback")
    
    def get_config_metadata(self, config_name: str) -> Optional[ConfigMetadata]:
        """
        Get metadata for configuration
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Configuration metadata or None if not found
        """
        return self.metadata.get(config_name)
    
    def list_configs(self) -> List[str]:
        """
        List all loaded configurations
        
        Returns:
            List of configuration names
        """
        return list(self.configs.keys())
    
    def get_environment(self) -> str:
        """
        Get current environment
        
        Returns:
            Current environment name
        """
        return self.environment
    
    def set_environment(self, environment: str) -> None:
        """
        Set current environment and reload configurations
        
        Args:
            environment: New environment name
        """
        if environment == self.environment:
            return
        
        old_environment = self.environment
        self.environment = environment
        
        try:
            # Clear current configurations
            self.configs.clear()
            self.metadata.clear()
            
            # Load configurations for new environment
            self._load_environment_configs()
            
            logger.info(f"Switched environment from {old_environment} to {environment}")
            
        except Exception as e:
            # Rollback on error
            self.environment = old_environment
            self._load_environment_configs()
            logger.error(f"Failed to switch environment to {environment}: {e}")
            raise
    
    def create_config_template(self, config_name: str, template_data: Dict[str, Any]) -> bool:
        """
        Create configuration template
        
        Args:
            config_name: Name of the configuration template
            template_data: Template data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            template_file = self.config_dir / "templates" / f"{config_name}.json"
            template_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(template_file, 'w') as f:
                json.dump(template_data, f, indent=2)
            
            logger.info(f"Created configuration template: {config_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create configuration template {config_name}: {e}")
            return False
    
    def _validate_config(self, config_name: str, config_data: Dict[str, Any]) -> None:
        """Validate configuration data against schema"""
        if config_name not in self.schemas:
            return
        
        validate(instance=config_data, schema=self.schemas[config_name])
    
    def _apply_env_overrides(self, config_name: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration"""
        # Environment variable format: FRAUD_CONFIG_{CONFIG_NAME}_{KEY_PATH}
        prefix = f"FRAUD_CONFIG_{config_name.upper()}_"
        
        result = copy.deepcopy(config_data)
        
        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                key_path = env_var[len(prefix):].lower().split('_')
                self._set_nested_value(result, key_path, self._parse_env_value(value))
        
        return result
    
    def _set_nested_value(self, data: Dict, key_path: List[str], value: Any) -> None:
        """Set nested dictionary value using key path"""
        current = data
        for key in key_path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[key_path[-1]] = value
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type"""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Try boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _encrypt_sensitive_fields(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields in configuration data"""
        if not self._cipher_suite:
            return config_data
        
        result = copy.deepcopy(config_data)
        sensitive_keys = ['password', 'secret', 'key', 'token', 'credential']
        
        def encrypt_recursive(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                        if isinstance(value, str) and not value.startswith('encrypted:'):
                            encrypted_value = self._cipher_suite.encrypt(value.encode()).decode()
                            data[key] = f"encrypted:{encrypted_value}"
                    elif isinstance(value, (dict, list)):
                        encrypt_recursive(value)
            elif isinstance(data, list):
                for item in data:
                    encrypt_recursive(item)
        
        encrypt_recursive(result)
        return result
    
    def _decrypt_sensitive_fields(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive fields in configuration data"""
        if not self._cipher_suite:
            return config_data
        
        result = copy.deepcopy(config_data)
        
        def decrypt_recursive(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str) and value.startswith('encrypted:'):
                        try:
                            encrypted_value = value[10:]  # Remove 'encrypted:' prefix
                            decrypted_value = self._cipher_suite.decrypt(encrypted_value.encode()).decode()
                            data[key] = decrypted_value
                        except Exception as e:
                            logger.error(f"Failed to decrypt field {key}: {e}")
                    elif isinstance(value, (dict, list)):
                        decrypt_recursive(value)
            elif isinstance(data, list):
                for item in data:
                    decrypt_recursive(item)
        
        decrypt_recursive(result)
        return result
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep update dictionary with another dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for configuration data"""
        import hashlib
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _generate_version(self) -> str:
        """Generate version string"""
        return f"v{int(time.time())}"
    
    def _create_backup(self, config_name: str) -> None:
        """Create backup of configuration"""
        if config_name not in self.configs:
            return
        
        backup_dir = self.config_dir / "backups" / config_name
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{config_name}_{timestamp}.json"
        
        with open(backup_file, 'w') as f:
            json.dump(self.configs[config_name], f, indent=2)
        
        # Update backup count in metadata
        if config_name in self.metadata:
            self.metadata[config_name].backup_count += 1
        
        logger.debug(f"Created backup for config {config_name}: {backup_file}")
    
    def _notify_change_callbacks(self, config_name: str, config_data: Dict[str, Any]) -> None:
        """Notify all registered change callbacks"""
        for callback in self.change_callbacks:
            try:
                callback(config_name, config_data)
            except Exception as e:
                logger.error(f"Error in configuration change callback: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self._stop_file_watching()
        logger.info("ConfigManager cleanup completed")

def create_enhanced_fraud_config_manager(config_dir: str = "./config", 
                                        environment: str = "dev",
                                        validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ConfigManager:
    """
    Create enhanced configuration manager with fraud detection defaults
    
    Args:
        config_dir: Configuration directory
        environment: Environment name (dev/test/staging/prod)
        validation_level: Validation level to apply
        
    Returns:
        Configured ConfigManager instance
    """
    config_manager = ConfigManager(
        config_dir=config_dir,
        environment=environment,
        enable_hot_reload=True,
        enable_encryption=True,
        validation_level=validation_level
    )
    
    # Add enhanced fraud detection schema
    fraud_schema = {
        "type": "object",
        "properties": {
            "model_threshold": {"type": "number", "minimum": 0, "maximum": 1},
            "batch_size": {"type": "integer", "minimum": 1},
            "alert_threshold": {"type": "number", "minimum": 0, "maximum": 1},
            "database": {
                "type": "object",
                "properties": {
                    "host": {"type": "string"},
                    "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                    "database": {"type": "string"},
                    "username": {"type": "string"},
                    "password": {"type": "string", "minLength": 8}
                },
                "required": ["host", "port", "database", "username"]
            },
            "security": {
                "type": "object",
                "properties": {
                    "encryption_enabled": {"type": "boolean"},
                    "security_level": {"type": "string", "enum": ["low", "standard", "high", "critical"]},
                    "data_retention_days": {"type": "integer", "minimum": 1}
                }
            }
        },
        "required": ["model_threshold", "batch_size"]
    }
    
    config_manager.add_schema("fraud_detection", fraud_schema)
    
    return config_manager


if __name__ == "__main__":
    # Example usage with enhanced validation
    config_manager = create_enhanced_fraud_config_manager(
        config_dir="./config",
        environment="dev",
        validation_level=ValidationLevel.STRICT
    )
    
    print("Enhanced configuration manager initialized and ready for use")
    print(f"Validation level: {config_manager.validation_level.value}")
    print(f"Environment: {config_manager.environment}")
    
    # Test enhanced domain configuration
    try:
        fraud_config = FinancialFraudConfig(
            config_path="./config/fraud_config.json",
            environment="dev",
            strict_mode=False
        )
        print("Enhanced fraud configuration loaded successfully")
    except Exception as e:
        print(f"Note: Enhanced fraud config not available: {e}")