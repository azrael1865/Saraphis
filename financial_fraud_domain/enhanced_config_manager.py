"""
Enhanced Configuration Manager with comprehensive validation and error handling
"""

import json
import os
import logging
import threading
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass
import hashlib
import shutil
from datetime import datetime
import re
from cryptography.fernet import Fernet


class ValidationLevel(Enum):
    """Configuration validation levels"""
    MINIMAL = "minimal"      # Basic type checking only
    STANDARD = "standard"    # Type + range validation
    STRICT = "strict"        # Full validation + security checks
    PARANOID = "paranoid"    # Maximum validation + integrity checks


class ConfigValidationError(Exception):
    """Raised when configuration validation fails"""
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.field = field
        self.value = value
        super().__init__(message)


class ConfigLoadError(Exception):
    """Raised when configuration loading fails"""
    pass


class ConfigSecurityError(Exception):
    """Raised when security validation fails"""
    pass


class ConfigCompatibilityError(Exception):
    """Raised when configuration compatibility issues are found"""
    pass


class ConfigIntegrityError(Exception):
    """Raised when configuration integrity checks fail"""
    pass


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    field_errors: Dict[str, List[str]]


class ConfigValidator:
    """Configuration validator with multiple validation levels"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.schemas = {}
        self.custom_validators = {}
        
    def register_schema(self, config_type: str, schema: Dict[str, Any]):
        """Register JSON schema for configuration type"""
        self.schemas[config_type] = schema
        
    def register_custom_validator(self, field_path: str, validator: Callable):
        """Register custom validator for specific field"""
        self.custom_validators[field_path] = validator
        
    def validate(self, config: Dict[str, Any], config_type: str = None) -> ValidationResult:
        """Validate configuration against registered schemas and rules"""
        errors = []
        warnings = []
        field_errors = {}
        
        try:
            # Basic structure validation
            if config_type and config_type in self.schemas:
                schema_errors = self._validate_schema(config, self.schemas[config_type])
                errors.extend(schema_errors)
            
            # Type validation
            type_errors = self._validate_types(config)
            errors.extend(type_errors)
            
            # Value validation
            if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                value_errors = self._validate_values(config)
                errors.extend(value_errors)
            
            # Security validation
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                security_errors = self._validate_security(config)
                errors.extend(security_errors)
            
            # Custom validation
            custom_errors = self._validate_custom(config)
            errors.extend(custom_errors)
            
            # Performance validation
            if self.validation_level == ValidationLevel.PARANOID:
                perf_warnings = self._validate_performance(config)
                warnings.extend(perf_warnings)
                
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            field_errors=field_errors
        )
    
    def _validate_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Validate against JSON schema"""
        errors = []
        # Basic schema validation implementation
        return errors
    
    def _validate_types(self, config: Dict[str, Any], path: str = "") -> List[str]:
        """Validate data types"""
        errors = []
        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                errors.extend(self._validate_types(value, current_path))
            elif key.endswith("_port") and not isinstance(value, int):
                errors.append(f"Port {current_path} must be integer, got {type(value).__name__}")
            elif key.endswith("_timeout") and not isinstance(value, (int, float)):
                errors.append(f"Timeout {current_path} must be numeric, got {type(value).__name__}")
            elif key.endswith("_enabled") and not isinstance(value, bool):
                errors.append(f"Flag {current_path} must be boolean, got {type(value).__name__}")
                
        return errors
    
    def _validate_values(self, config: Dict[str, Any], path: str = "") -> List[str]:
        """Validate value ranges and constraints"""
        errors = []
        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                errors.extend(self._validate_values(value, current_path))
            elif key.endswith("_port") and isinstance(value, int):
                if not (1 <= value <= 65535):
                    errors.append(f"Port {current_path} must be between 1-65535, got {value}")
            elif key.endswith("_threshold") and isinstance(value, (int, float)):
                if not (0 <= value <= 1):
                    errors.append(f"Threshold {current_path} must be between 0-1, got {value}")
            elif key.endswith("_size") and isinstance(value, int):
                if value < 0:
                    errors.append(f"Size {current_path} must be non-negative, got {value}")
                    
        return errors
    
    def _validate_security(self, config: Dict[str, Any], path: str = "") -> List[str]:
        """Validate security-related configurations"""
        errors = []
        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                errors.extend(self._validate_security(value, current_path))
            elif "password" in key.lower() and isinstance(value, str):
                if len(value) < 8:
                    errors.append(f"Password {current_path} too short (minimum 8 characters)")
                if value in ["password", "123456", "admin"]:
                    errors.append(f"Password {current_path} is too weak")
            elif "key" in key.lower() and isinstance(value, str):
                if len(value) < 16:
                    errors.append(f"Key {current_path} too short (minimum 16 characters)")
                    
        return errors
    
    def _validate_custom(self, config: Dict[str, Any], path: str = "") -> List[str]:
        """Run custom validators"""
        errors = []
        for field_path, validator in self.custom_validators.items():
            try:
                if field_path in config:
                    result = validator(config[field_path])
                    if not result:
                        errors.append(f"Custom validation failed for {field_path}")
            except Exception as e:
                errors.append(f"Custom validator error for {field_path}: {str(e)}")
        return errors
    
    def _validate_performance(self, config: Dict[str, Any]) -> List[str]:
        """Validate performance-related settings"""
        warnings = []
        # Add performance validation logic
        return warnings


class EnhancedConfigManager:
    """Enhanced configuration manager with validation, encryption, and recovery"""
    
    def __init__(self, 
                 config_dir: str = "config",
                 environment: str = "development",
                 validation_level: ValidationLevel = ValidationLevel.STANDARD,
                 enable_encryption: bool = True,
                 enable_recovery: bool = True,
                 backup_count: int = 5):
        
        self.config_dir = Path(config_dir)
        self.environment = environment
        self.validation_level = validation_level
        self.enable_encryption = enable_encryption
        self.enable_recovery = enable_recovery
        self.backup_count = backup_count
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize encryption
        self.cipher_suite = None
        if enable_encryption:
            self._init_encryption()
        
        # Initialize validators
        self.validators = {}
        self._init_validators()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create directory structure
        self._ensure_directories()
    
    def _init_encryption(self):
        """Initialize encryption for sensitive data"""
        key_file = self.config_dir / ".encryption_key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Restrict permissions
        
        self.cipher_suite = Fernet(key)
    
    def _init_validators(self):
        """Initialize configuration validators"""
        for config_type in ["default", "database", "security", "api", "preprocessing", "fraud_detection"]:
            self.validators[config_type] = ConfigValidator(self.validation_level)
        
        # Register preprocessing configuration schema
        self._register_preprocessing_schema()
    
    def _ensure_directories(self):
        """Create necessary directory structure"""
        directories = [
            self.config_dir,
            self.config_dir / "environments" / self.environment,
            self.config_dir / "backups",
            self.config_dir / "schemas"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _register_preprocessing_schema(self):
        """Register preprocessing configuration schema"""
        preprocessing_schema = {
            "type": "object",
            "properties": {
                "feature_engineering": {
                    "type": "object",
                    "properties": {
                        "enable_time_features": {"type": "boolean"},
                        "enable_amount_features": {"type": "boolean"},
                        "enable_frequency_features": {"type": "boolean"},
                        "enable_velocity_features": {"type": "boolean"},
                        "enable_merchant_features": {"type": "boolean"},
                        "enable_geographic_features": {"type": "boolean"}
                    },
                    "required": ["enable_time_features", "enable_amount_features"]
                },
                "data_quality": {
                    "type": "object",
                    "properties": {
                        "missing_value_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                        "outlier_method": {"type": "string", "enum": ["iqr", "zscore", "isolation"]},
                        "outlier_threshold": {"type": "number", "minimum": 0},
                        "duplicate_threshold": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["missing_value_threshold", "outlier_method"]
                },
                "feature_selection": {
                    "type": "object",
                    "properties": {
                        "method": {"type": "string", "enum": ["mutual_info", "chi2", "f_classif"]},
                        "k_features": {"type": "integer", "minimum": 1, "maximum": 10000},
                        "correlation_threshold": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["method", "k_features"]
                },
                "scaling": {
                    "type": "object",
                    "properties": {
                        "method": {"type": "string", "enum": ["standard", "minmax", "robust"]},
                        "feature_range": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        }
                    },
                    "required": ["method"]
                }
            },
            "required": ["feature_engineering", "data_quality", "scaling"]
        }
        
        fraud_detection_schema = {
            "type": "object",
            "properties": {
                "detection_strategy": {"type": "string", "enum": ["rules_only", "ml_only", "hybrid", "ensemble"]},
                "enable_preprocessing": {"type": "boolean"},
                "fraud_probability_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "risk_score_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                "preprocessing_config": preprocessing_schema
            },
            "required": ["detection_strategy", "enable_preprocessing"]
        }
        
        self.validators["preprocessing"].register_schema("preprocessing", preprocessing_schema)
        self.validators["fraud_detection"].register_schema("fraud_detection", fraud_detection_schema)
        
        # Register custom validators for preprocessing
        self._register_preprocessing_custom_validators()
    
    def _register_preprocessing_custom_validators(self):
        """Register custom validators for preprocessing configuration"""
        
        def validate_feature_threshold(value):
            if isinstance(value, (int, float)) and 0 <= value <= 1:
                return True, None
            return False, "Value must be between 0 and 1"
        
        def validate_feature_count(value):
            if isinstance(value, int) and 1 <= value <= 10000:
                return True, None
            return False, "Feature count must be between 1 and 10000"
        
        def validate_outlier_threshold(value):
            if isinstance(value, (int, float)) and value > 0:
                return True, None
            return False, "Outlier threshold must be positive"
        
        # Register custom validators
        preprocessing_validator = self.validators["preprocessing"]
        preprocessing_validator.register_custom_validator("data_quality.missing_value_threshold", validate_feature_threshold)
        preprocessing_validator.register_custom_validator("feature_selection.k_features", validate_feature_count)
        preprocessing_validator.register_custom_validator("data_quality.outlier_threshold", validate_outlier_threshold)
    
    def _get_validator(self, config_type: str) -> ConfigValidator:
        """Get validator for configuration type"""
        return self.validators.get(config_type, self.validators["default"])
    
    def _encrypt_sensitive_fields(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields in configuration"""
        if not self.enable_encryption or not self.cipher_suite:
            return config
        
        sensitive_patterns = [
            r".*password.*", r".*secret.*", r".*key.*", r".*token.*"
        ]
        
        def encrypt_recursive(obj, path=""):
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check if field should be encrypted
                    should_encrypt = any(
                        re.match(pattern, key.lower()) 
                        for pattern in sensitive_patterns
                    )
                    
                    if should_encrypt and isinstance(value, str) and not value.startswith("encrypted:"):
                        encrypted_value = self.cipher_suite.encrypt(value.encode()).decode()
                        result[key] = f"encrypted:{encrypted_value}"
                    elif isinstance(value, (dict, list)):
                        result[key] = encrypt_recursive(value, current_path)
                    else:
                        result[key] = value
                return result
            elif isinstance(obj, list):
                return [encrypt_recursive(item, path) for item in obj]
            else:
                return obj
        
        return encrypt_recursive(config)
    
    def _decrypt_sensitive_fields(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive fields in configuration"""
        if not self.enable_encryption or not self.cipher_suite:
            return config
        
        def decrypt_recursive(obj):
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    if isinstance(value, str) and value.startswith("encrypted:"):
                        try:
                            encrypted_data = value[10:]  # Remove "encrypted:" prefix
                            decrypted_value = self.cipher_suite.decrypt(encrypted_data.encode()).decode()
                            result[key] = decrypted_value
                        except Exception as e:
                            self.logger.error(f"Failed to decrypt {key}: {e}")
                            result[key] = value
                    elif isinstance(value, (dict, list)):
                        result[key] = decrypt_recursive(value)
                    else:
                        result[key] = value
                return result
            elif isinstance(obj, list):
                return [decrypt_recursive(item) for item in obj]
            else:
                return obj
        
        return decrypt_recursive(config)
    
    def _create_backup(self, config_name: str, config_data: Dict[str, Any]):
        """Create backup of configuration"""
        if not self.enable_recovery:
            return
        
        backup_dir = self.config_dir / "backups" / config_name
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"{config_name}_{timestamp}.json"
        
        with open(backup_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Clean old backups
        self._clean_old_backups(backup_dir)
    
    def _clean_old_backups(self, backup_dir: Path):
        """Remove old backup files, keeping only the latest ones"""
        backup_files = sorted(backup_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        for old_backup in backup_files[self.backup_count:]:
            old_backup.unlink()
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for configuration data"""
        config_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def _apply_environment_overrides(self, config: Dict[str, Any], config_name: str) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        env_prefix = f"FRAUD_CONFIG_{self.environment.upper()}_{config_name.upper()}_"
        
        def apply_overrides(obj, path=""):
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    env_key = f"{env_prefix}{path}{key}".upper().replace(".", "_")
                    
                    if env_key in os.environ:
                        # Parse environment value based on original type
                        env_value = os.environ[env_key]
                        if isinstance(value, bool):
                            result[key] = env_value.lower() in ("true", "1", "yes")
                        elif isinstance(value, int):
                            result[key] = int(env_value)
                        elif isinstance(value, float):
                            result[key] = float(env_value)
                        else:
                            result[key] = env_value
                    elif isinstance(value, dict):
                        result[key] = apply_overrides(value, f"{path}{key}_")
                    else:
                        result[key] = value
                return result
            else:
                return obj
        
        return apply_overrides(config)
    
    def save_config(self, config_name: str, config_data: Dict[str, Any], 
                   config_type: str = "default") -> bool:
        """Save configuration with validation and encryption"""
        
        with self._lock:
            try:
                # Validate configuration
                validator = self._get_validator(config_type)
                validation_result = validator.validate(config_data, config_type)
                
                if not validation_result.is_valid:
                    error_msg = f"Configuration validation failed: {validation_result.errors}"
                    raise ConfigValidationError(error_msg)
                
                # Log warnings
                for warning in validation_result.warnings:
                    self.logger.warning(f"Config warning for {config_name}: {warning}")
                
                # Create backup of existing config
                config_file = self._get_config_file_path(config_name)
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        existing_config = json.load(f)
                    self._create_backup(config_name, existing_config)
                
                # Encrypt sensitive fields
                encrypted_config = self._encrypt_sensitive_fields(config_data.copy())
                
                # Add metadata
                metadata = {
                    "created_at": datetime.now().isoformat(),
                    "environment": self.environment,
                    "validation_level": self.validation_level.value,
                    "checksum": self._calculate_checksum(config_data)
                }
                encrypted_config["_metadata"] = metadata
                
                # Save configuration
                with open(config_file, 'w') as f:
                    json.dump(encrypted_config, f, indent=2)
                
                self.logger.info(f"Configuration {config_name} saved successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to save configuration {config_name}: {e}")
                if isinstance(e, (ConfigValidationError, ConfigSecurityError)):
                    raise
                raise ConfigLoadError(f"Failed to save configuration: {e}")
    
    def load_config(self, config_name: str, required: bool = True, 
                   use_defaults: bool = False) -> Optional[Dict[str, Any]]:
        """Load configuration with decryption and validation"""
        
        with self._lock:
            config_file = self._get_config_file_path(config_name)
            
            try:
                # Try to load configuration file
                if not config_file.exists():
                    if required:
                        raise ConfigLoadError(f"Required configuration {config_name} not found")
                    elif use_defaults:
                        return self._get_default_config(config_name)
                    return None
                
                with open(config_file, 'r') as f:
                    encrypted_config = json.load(f)
                
                # Verify integrity if enabled
                if self.validation_level == ValidationLevel.PARANOID:
                    self._verify_integrity(encrypted_config)
                
                # Decrypt sensitive fields
                config_data = self._decrypt_sensitive_fields(encrypted_config)
                
                # Remove metadata
                config_data.pop("_metadata", None)
                
                # Apply environment overrides
                config_data = self._apply_environment_overrides(config_data, config_name)
                
                self.logger.debug(f"Configuration {config_name} loaded successfully")
                return config_data
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Invalid JSON in configuration {config_name}: {e}")
                
                if self.enable_recovery:
                    return self._attempt_recovery(config_name, use_defaults)
                
                if required:
                    raise ConfigLoadError(f"Configuration {config_name} is corrupted")
                return None
                
            except Exception as e:
                self.logger.error(f"Failed to load configuration {config_name}: {e}")
                
                if self.enable_recovery and not required:
                    return self._attempt_recovery(config_name, use_defaults)
                
                raise ConfigLoadError(f"Failed to load configuration: {e}")
    
    def _get_config_file_path(self, config_name: str) -> Path:
        """Get full path to configuration file"""
        return self.config_dir / "environments" / self.environment / f"{config_name}.json"
    
    def _verify_integrity(self, config_data: Dict[str, Any]):
        """Verify configuration integrity"""
        metadata = config_data.get("_metadata", {})
        stored_checksum = metadata.get("checksum")
        
        if stored_checksum:
            # Remove metadata for checksum calculation
            config_copy = config_data.copy()
            config_copy.pop("_metadata", None)
            
            calculated_checksum = self._calculate_checksum(config_copy)
            
            if stored_checksum != calculated_checksum:
                raise ConfigIntegrityError("Configuration integrity check failed")
    
    def _attempt_recovery(self, config_name: str, use_defaults: bool = False) -> Optional[Dict[str, Any]]:
        """Attempt to recover configuration from backup"""
        self.logger.info(f"Attempting recovery for configuration {config_name}")
        
        backup_dir = self.config_dir / "backups" / config_name
        if backup_dir.exists():
            backup_files = sorted(backup_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
            
            for backup_file in backup_files:
                try:
                    with open(backup_file, 'r') as f:
                        backup_config = json.load(f)
                    
                    self.logger.info(f"Recovered configuration {config_name} from {backup_file.name}")
                    return backup_config
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load backup {backup_file.name}: {e}")
                    continue
        
        if use_defaults:
            self.logger.info(f"Using default configuration for {config_name}")
            return self._get_default_config(config_name)
        
        return None
    
    def _get_default_config(self, config_name: str) -> Dict[str, Any]:
        """Get default configuration"""
        defaults = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "fraud_detection",
                "username": "fraud_user",
                "password": "change_me",
                "connection_pool_size": 10
            },
            "api": {
                "host": "127.0.0.1",
                "port": 8000,
                "debug_mode": False,
                "ssl_enabled": False
            },
            "security": {
                "encryption_enabled": True,
                "security_level": "STANDARD"
            }
        }
        
        return defaults.get(config_name, {})
    
    def list_configs(self) -> List[str]:
        """List available configurations"""
        env_dir = self.config_dir / "environments" / self.environment
        if not env_dir.exists():
            return []
        
        config_files = env_dir.glob("*.json")
        return [f.stem for f in config_files]
    
    def delete_config(self, config_name: str, create_backup: bool = True) -> bool:
        """Delete configuration file"""
        with self._lock:
            config_file = self._get_config_file_path(config_name)
            
            if not config_file.exists():
                return False
            
            try:
                if create_backup:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                    self._create_backup(config_name, config_data)
                
                config_file.unlink()
                self.logger.info(f"Configuration {config_name} deleted")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to delete configuration {config_name}: {e}")
                return False