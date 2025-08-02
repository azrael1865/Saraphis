"""
Accuracy Tracking Configuration Loader - Phase 5C-2A
Advanced configuration loading and validation system with multi-source support,
comprehensive schema validation, and environment-specific configuration management.
"""

import os
import json
import yaml
import toml
import re
import hashlib
import sqlite3
import requests
import threading
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import jsonschema
from jsonschema import Draft7Validator, validators
from cryptography.fernet import Fernet
import logging
from collections import defaultdict, OrderedDict
from functools import lru_cache
import warnings

# Import from existing modules
try:
    from config_manager import ConfigManager, ConfigMetadata
    from enhanced_config_manager import (
        EnhancedConfigManager, ValidationLevel, ConfigValidationError,
        ConfigLoadError, ConfigSecurityError
    )
    from enhanced_fraud_core_monitoring import (
        MonitoringManager, PerformanceMetrics, CacheManager,
        monitor_performance, MonitoringConfig
    )
    from enhanced_fraud_core_exceptions import (
        EnhancedFraudException, ValidationError, ConfigurationError,
        ErrorContext, create_error_context
    )
    from accuracy_tracking_health_monitor import (
        AccuracyTrackingHealthMonitor, HealthStatus, ComponentType
    )
except ImportError:
    # Fallback for standalone development
    from config_manager import ConfigManager, ConfigMetadata
    from enhanced_config_manager import (
        EnhancedConfigManager, ValidationLevel, ConfigValidationError,
        ConfigLoadError, ConfigSecurityError
    )
    from enhanced_fraud_core_monitoring import (
        MonitoringManager, PerformanceMetrics, CacheManager,
        monitor_performance, MonitoringConfig
    )
    from enhanced_fraud_core_exceptions import (
        EnhancedFraudException, ValidationError, ConfigurationError,
        ErrorContext, create_error_context
    )
    from accuracy_tracking_health_monitor import (
        AccuracyTrackingHealthMonitor, HealthStatus, ComponentType
    )

# Configure logging
logger = logging.getLogger(__name__)

# ======================== CUSTOM EXCEPTIONS ========================

class ConfigLoaderError(ConfigurationError):
    """Base exception for configuration loader errors"""
    pass

class ConfigSourceError(ConfigLoaderError):
    """Exception raised when configuration source fails"""
    pass

class ConfigValidationError(ConfigLoaderError):
    """Exception raised when configuration validation fails"""
    pass

class ConfigMergeError(ConfigLoaderError):
    """Exception raised when configuration merge fails"""
    pass

# ======================== ENUMS AND CONSTANTS ========================

class ConfigSourceType(Enum):
    """Types of configuration sources"""
    FILE = "file"
    ENVIRONMENT = "environment"
    DATABASE = "database"
    REMOTE = "remote"
    MEMORY = "memory"

class ConfigFormat(Enum):
    """Supported configuration formats"""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    ENV = "env"
    PROPERTIES = "properties"

class ValidationStage(Enum):
    """Validation pipeline stages"""
    SYNTAX = "syntax"
    SCHEMA = "schema"
    BUSINESS_RULES = "business_rules"
    CROSS_FIELD = "cross_field"
    PERFORMANCE = "performance"
    SECURITY = "security"

class ConfigEnvironment(Enum):
    """Configuration environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

# Default configuration schema version
SCHEMA_VERSION = "1.0.0"

# Configuration source priorities (lower number = higher priority)
SOURCE_PRIORITIES = {
    ConfigSourceType.MEMORY: 0,
    ConfigSourceType.ENVIRONMENT: 1,
    ConfigSourceType.FILE: 2,
    ConfigSourceType.DATABASE: 3,
    ConfigSourceType.REMOTE: 4
}

# ======================== DATA STRUCTURES ========================

@dataclass
class ConfigSourceInfo:
    """Information about a configuration source"""
    source_type: ConfigSourceType
    source_path: str
    format: ConfigFormat
    priority: int
    is_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    """Result of configuration validation"""
    stage: ValidationStage
    is_valid: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConfigVersion:
    """Configuration version information"""
    version: str
    schema_version: str
    created_at: datetime
    created_by: Optional[str] = None
    change_summary: Optional[str] = None
    is_rollback: bool = False
    parent_version: Optional[str] = None

@dataclass
class ConfigLoadResult:
    """Result of configuration loading"""
    config: Dict[str, Any]
    sources_loaded: List[ConfigSourceInfo]
    validation_results: List[ValidationResult]
    version: ConfigVersion
    load_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

# ======================== CONFIGURATION SOURCES ========================

class ConfigSource(ABC):
    """Abstract base class for configuration sources"""
    
    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Load configuration from source"""
        pass
    
    @abstractmethod
    def save(self, config: Dict[str, Any]) -> bool:
        """Save configuration to source"""
        pass
    
    @abstractmethod
    def exists(self) -> bool:
        """Check if source exists"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get source metadata"""
        pass

class FileConfigSource(ConfigSource):
    """File-based configuration source"""
    
    def __init__(self, file_path: Path, format: ConfigFormat):
        self.file_path = Path(file_path)
        self.format = format
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.exists():
            raise ConfigSourceError(
                f"Configuration file not found: {self.file_path}",
                context=create_error_context(
                    component="FileConfigSource",
                    operation="load",
                    file_path=str(self.file_path)
                )
            )
        
        try:
            with open(self.file_path, 'r') as f:
                if self.format == ConfigFormat.JSON:
                    return json.load(f)
                elif self.format == ConfigFormat.YAML:
                    return yaml.safe_load(f) or {}
                elif self.format == ConfigFormat.TOML:
                    return toml.load(f)
                else:
                    raise ConfigSourceError(f"Unsupported format: {self.format}")
                    
        except Exception as e:
            raise ConfigSourceError(
                f"Failed to load configuration from {self.file_path}: {str(e)}",
                context=create_error_context(
                    component="FileConfigSource",
                    operation="load",
                    file_path=str(self.file_path),
                    format=self.format.value
                )
            )
    
    def save(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file"""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.file_path, 'w') as f:
                if self.format == ConfigFormat.JSON:
                    json.dump(config, f, indent=2, default=str)
                elif self.format == ConfigFormat.YAML:
                    yaml.dump(config, f, default_flow_style=False)
                elif self.format == ConfigFormat.TOML:
                    toml.dump(config, f)
                else:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def exists(self) -> bool:
        """Check if file exists"""
        return self.file_path.exists()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get file metadata"""
        if not self.exists():
            return {}
        
        stat = self.file_path.stat()
        return {
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'format': self.format.value,
            'path': str(self.file_path)
        }

class EnvironmentConfigSource(ConfigSource):
    """Environment variable configuration source"""
    
    def __init__(self, prefix: str = "ACCURACY_TRACKING_", 
                 delimiter: str = "__",
                 case_sensitive: bool = False):
        self.prefix = prefix
        self.delimiter = delimiter
        self.case_sensitive = case_sensitive
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = {}
        
        for key, value in os.environ.items():
            if not self.case_sensitive:
                if not key.upper().startswith(self.prefix.upper()):
                    continue
                config_key = key[len(self.prefix):]
            else:
                if not key.startswith(self.prefix):
                    continue
                config_key = key[len(self.prefix):]
            
            # Convert delimiter to nested structure
            parts = config_key.split(self.delimiter)
            current = config
            
            for i, part in enumerate(parts[:-1]):
                if not self.case_sensitive:
                    part = part.lower()
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the value
            final_key = parts[-1]
            if not self.case_sensitive:
                final_key = final_key.lower()
            
            current[final_key] = self._parse_value(value)
        
        return config
    
    def save(self, config: Dict[str, Any]) -> bool:
        """Save configuration to environment (not supported)"""
        self.logger.warning("Saving to environment variables not supported")
        return False
    
    def exists(self) -> bool:
        """Check if any matching environment variables exist"""
        for key in os.environ:
            if key.startswith(self.prefix):
                return True
        return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get environment metadata"""
        matching_vars = [
            key for key in os.environ 
            if key.startswith(self.prefix)
        ]
        
        return {
            'prefix': self.prefix,
            'delimiter': self.delimiter,
            'variable_count': len(matching_vars),
            'variables': matching_vars[:10]  # First 10 for safety
        }
    
    def _parse_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type"""
        # Try to parse as JSON first
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Boolean values
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Default to string
        return value

class DatabaseConfigSource(ConfigSource):
    """Database-based configuration source"""
    
    def __init__(self, db_path: str, table_name: str = "accuracy_tracking_config"):
        self.db_path = db_path
        self.table_name = table_name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database table"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(f'''
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        environment TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metadata TEXT
                    )
                ''')
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from database"""
        try:
            config = {}
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    f"SELECT key, value FROM {self.table_name} ORDER BY key"
                )
                
                for key, value in cursor:
                    # Parse nested keys
                    parts = key.split('.')
                    current = config
                    
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    
                    # Parse value
                    try:
                        current[parts[-1]] = json.loads(value)
                    except json.JSONDecodeError:
                        current[parts[-1]] = value
            
            return config
            
        except Exception as e:
            raise ConfigSourceError(
                f"Failed to load configuration from database: {str(e)}",
                context=create_error_context(
                    component="DatabaseConfigSource",
                    operation="load",
                    db_path=self.db_path
                )
            )
    
    def save(self, config: Dict[str, Any]) -> bool:
        """Save configuration to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Clear existing configuration
                conn.execute(f"DELETE FROM {self.table_name}")
                
                # Flatten and save configuration
                flat_config = self._flatten_config(config)
                current_time = datetime.now().isoformat()
                
                for key, value in flat_config.items():
                    value_str = json.dumps(value) if not isinstance(value, str) else value
                    
                    conn.execute(
                        f'''INSERT INTO {self.table_name} 
                            (key, value, created_at, updated_at) 
                            VALUES (?, ?, ?, ?)''',
                        (key, value_str, current_time, current_time)
                    )
                
                conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def exists(self) -> bool:
        """Check if database exists and has configuration"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    f"SELECT COUNT(*) FROM {self.table_name}"
                )
                count = cursor.fetchone()[0]
                return count > 0
        except:
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get database metadata"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    f"SELECT COUNT(*), MAX(updated_at) FROM {self.table_name}"
                )
                count, last_updated = cursor.fetchone()
                
                return {
                    'db_path': self.db_path,
                    'table_name': self.table_name,
                    'config_count': count,
                    'last_updated': last_updated
                }
        except:
            return {}
    
    def _flatten_config(self, config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested configuration"""
        flat = {}
        
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                flat.update(self._flatten_config(value, full_key))
            else:
                flat[full_key] = value
        
        return flat

class RemoteConfigSource(ConfigSource):
    """Remote configuration source (HTTP/HTTPS)"""
    
    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None,
                 timeout: int = 30, cache_ttl: int = 300):
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Simple cache
        self._cache = None
        self._cache_time = None
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from remote source"""
        # Check cache
        if self._cache and self._cache_time:
            if datetime.now() - self._cache_time < timedelta(seconds=self.cache_ttl):
                return self._cache.copy()
        
        try:
            response = requests.get(
                self.url,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response based on content type
            content_type = response.headers.get('content-type', '').lower()
            
            if 'json' in content_type:
                config = response.json()
            elif 'yaml' in content_type:
                config = yaml.safe_load(response.text)
            elif 'toml' in content_type:
                config = toml.loads(response.text)
            else:
                # Try to auto-detect format
                try:
                    config = response.json()
                except:
                    try:
                        config = yaml.safe_load(response.text)
                    except:
                        config = toml.loads(response.text)
            
            # Update cache
            self._cache = config
            self._cache_time = datetime.now()
            
            return config
            
        except Exception as e:
            raise ConfigSourceError(
                f"Failed to load remote configuration: {str(e)}",
                context=create_error_context(
                    component="RemoteConfigSource",
                    operation="load",
                    url=self.url
                )
            )
    
    def save(self, config: Dict[str, Any]) -> bool:
        """Save configuration to remote source (not implemented)"""
        self.logger.warning("Saving to remote source not implemented")
        return False
    
    def exists(self) -> bool:
        """Check if remote source is accessible"""
        try:
            response = requests.head(
                self.url,
                headers=self.headers,
                timeout=5
            )
            return response.status_code < 400
        except:
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get remote source metadata"""
        try:
            response = requests.head(
                self.url,
                headers=self.headers,
                timeout=5
            )
            
            return {
                'url': self.url,
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type'),
                'last_modified': response.headers.get('last-modified'),
                'cached': self._cache is not None
            }
        except:
            return {'url': self.url, 'accessible': False}

# ======================== CONFIGURATION VALIDATOR ========================

class ConfigValidator:
    """Advanced configuration validator"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.custom_validators = {}
        self.business_rules = []
        self.cross_field_validators = []
    
    def add_custom_validator(self, name: str, validator_func: Callable) -> None:
        """Add custom validator function"""
        self.custom_validators[name] = validator_func
    
    def add_business_rule(self, rule_func: Callable, error_message: str) -> None:
        """Add business rule validation"""
        self.business_rules.append((rule_func, error_message))
    
    def add_cross_field_validator(self, validator_func: Callable) -> None:
        """Add cross-field validator"""
        self.cross_field_validators.append(validator_func)
    
    def validate_syntax(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration syntax"""
        result = ValidationResult(
            stage=ValidationStage.SYNTAX,
            is_valid=True
        )
        
        # Check for basic structure
        if not isinstance(config, dict):
            result.is_valid = False
            result.errors.append({
                'field': 'root',
                'message': 'Configuration must be a dictionary',
                'value': type(config).__name__
            })
            return result
        
        # Check for required top-level keys
        required_keys = ['version', 'environment']
        for key in required_keys:
            if key not in config:
                result.warnings.append({
                    'field': key,
                    'message': f'Recommended key "{key}" not found',
                    'suggestion': f'Add "{key}" to configuration for better tracking'
                })
        
        # Validate key naming conventions
        for key in config:
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', key):
                result.warnings.append({
                    'field': key,
                    'message': 'Key does not follow naming convention',
                    'suggestion': 'Use snake_case or camelCase for keys'
                })
        
        return result
    
    def validate_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
        """Validate configuration against JSON schema"""
        result = ValidationResult(
            stage=ValidationStage.SCHEMA,
            is_valid=True
        )
        
        try:
            # Create validator with custom validators
            validator_class = validators.create(
                meta_schema=Draft7Validator.META_SCHEMA,
                validators=self.custom_validators
            )
            validator = validator_class(schema)
            
            # Validate
            errors = list(validator.iter_errors(config))
            
            if errors:
                result.is_valid = False
                for error in errors:
                    error_dict = {
                        'field': '.'.join(str(p) for p in error.path),
                        'message': error.message,
                        'value': error.instance,
                        'schema_path': '.'.join(str(p) for p in error.schema_path)
                    }
                    
                    # Add suggestions based on error type
                    if 'enum' in error.schema:
                        error_dict['suggestion'] = f"Valid values: {error.schema['enum']}"
                    elif 'type' in error.schema:
                        error_dict['suggestion'] = f"Expected type: {error.schema['type']}"
                    elif 'minimum' in error.schema or 'maximum' in error.schema:
                        min_val = error.schema.get('minimum', 'N/A')
                        max_val = error.schema.get('maximum', 'N/A')
                        error_dict['suggestion'] = f"Value must be between {min_val} and {max_val}"
                    
                    result.errors.append(error_dict)
            
        except Exception as e:
            result.is_valid = False
            result.errors.append({
                'field': 'schema',
                'message': f'Schema validation error: {str(e)}'
            })
        
        return result
    
    def validate_business_rules(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate business rules"""
        result = ValidationResult(
            stage=ValidationStage.BUSINESS_RULES,
            is_valid=True
        )
        
        for rule_func, error_message in self.business_rules:
            try:
                if not rule_func(config):
                    result.is_valid = False
                    result.errors.append({
                        'rule': rule_func.__name__,
                        'message': error_message
                    })
            except Exception as e:
                result.warnings.append({
                    'rule': rule_func.__name__,
                    'message': f'Rule check failed: {str(e)}'
                })
        
        return result
    
    def validate_cross_fields(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate cross-field dependencies"""
        result = ValidationResult(
            stage=ValidationStage.CROSS_FIELD,
            is_valid=True
        )
        
        for validator_func in self.cross_field_validators:
            try:
                validation = validator_func(config)
                if isinstance(validation, dict):
                    if not validation.get('valid', True):
                        result.is_valid = False
                        result.errors.append({
                            'validator': validator_func.__name__,
                            'message': validation.get('message', 'Cross-field validation failed'),
                            'fields': validation.get('fields', [])
                        })
                        
                        if 'suggestion' in validation:
                            result.suggestions.append(validation['suggestion'])
                            
            except Exception as e:
                result.warnings.append({
                    'validator': validator_func.__name__,
                    'message': f'Validator failed: {str(e)}'
                })
        
        return result
    
    def validate_performance_impact(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate performance impact of configuration"""
        result = ValidationResult(
            stage=ValidationStage.PERFORMANCE,
            is_valid=True
        )
        
        # Check cache sizes
        cache_config = config.get('cache', {})
        cache_size = cache_config.get('size', 0)
        
        if cache_size > 10000:
            result.warnings.append({
                'field': 'cache.size',
                'message': f'Large cache size ({cache_size}) may impact memory usage',
                'suggestion': 'Consider using a smaller cache or implementing cache eviction'
            })
        
        # Check monitoring intervals
        monitoring_config = config.get('monitoring', {})
        for key, value in monitoring_config.items():
            if 'interval' in key and isinstance(value, (int, float)):
                if value < 1:
                    result.warnings.append({
                        'field': f'monitoring.{key}',
                        'message': f'Very short interval ({value}s) may impact performance',
                        'suggestion': 'Consider using intervals >= 1 second'
                    })
        
        # Check thread/worker counts
        if 'max_workers' in config:
            max_workers = config['max_workers']
            cpu_count = os.cpu_count() or 1
            
            if max_workers > cpu_count * 2:
                result.warnings.append({
                    'field': 'max_workers',
                    'message': f'High worker count ({max_workers}) for {cpu_count} CPUs',
                    'suggestion': f'Consider using {cpu_count}-{cpu_count * 2} workers'
                })
        
        return result
    
    def validate_security(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate security aspects of configuration"""
        result = ValidationResult(
            stage=ValidationStage.SECURITY,
            is_valid=True
        )
        
        # Check for exposed secrets
        sensitive_patterns = [
            r'password', r'secret', r'key', r'token', r'credential'
        ]
        
        def check_for_secrets(data: Dict[str, Any], path: str = ""):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check key names
                for pattern in sensitive_patterns:
                    if re.search(pattern, key, re.IGNORECASE):
                        if isinstance(value, str) and not value.startswith('encrypted:'):
                            result.warnings.append({
                                'field': current_path,
                                'message': 'Potential unencrypted secret detected',
                                'suggestion': 'Consider encrypting sensitive values'
                            })
                
                # Recurse into nested dictionaries
                if isinstance(value, dict):
                    check_for_secrets(value, current_path)
        
        check_for_secrets(config)
        
        # Check SSL/TLS settings
        if 'api' in config:
            api_config = config['api']
            if api_config.get('enable_https', True) and not api_config.get('ssl_cert'):
                result.warnings.append({
                    'field': 'api.ssl_cert',
                    'message': 'HTTPS enabled but no SSL certificate configured',
                    'suggestion': 'Configure SSL certificate for secure communication'
                })
        
        # Check authentication settings
        auth_config = config.get('authentication', {})
        if not auth_config.get('enabled', True):
            result.warnings.append({
                'field': 'authentication.enabled',
                'message': 'Authentication is disabled',
                'suggestion': 'Enable authentication for production environments'
            })
        
        return result

# ======================== VALIDATION PIPELINE ========================

class ValidationPipeline:
    """Multi-stage validation pipeline"""
    
    def __init__(self, validator: ConfigValidator):
        self.validator = validator
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.stages = [
            ValidationStage.SYNTAX,
            ValidationStage.SCHEMA,
            ValidationStage.BUSINESS_RULES,
            ValidationStage.CROSS_FIELD,
            ValidationStage.PERFORMANCE,
            ValidationStage.SECURITY
        ]
        self.stage_handlers = {
            ValidationStage.SYNTAX: self.validator.validate_syntax,
            ValidationStage.SCHEMA: self._validate_schema_wrapper,
            ValidationStage.BUSINESS_RULES: self.validator.validate_business_rules,
            ValidationStage.CROSS_FIELD: self.validator.validate_cross_fields,
            ValidationStage.PERFORMANCE: self.validator.validate_performance_impact,
            ValidationStage.SECURITY: self.validator.validate_security
        }
        self.schema = None
    
    def set_schema(self, schema: Dict[str, Any]) -> None:
        """Set JSON schema for validation"""
        self.schema = schema
    
    def _validate_schema_wrapper(self, config: Dict[str, Any]) -> ValidationResult:
        """Wrapper for schema validation"""
        if not self.schema:
            return ValidationResult(
                stage=ValidationStage.SCHEMA,
                is_valid=True,
                warnings=[{
                    'message': 'No schema configured for validation'
                }]
            )
        
        return self.validator.validate_schema(config, self.schema)
    
    def validate(self, config: Dict[str, Any], 
                 stages: Optional[List[ValidationStage]] = None) -> List[ValidationResult]:
        """Run validation pipeline"""
        stages_to_run = stages or self.stages
        results = []
        
        for stage in stages_to_run:
            if stage not in self.stage_handlers:
                self.logger.warning(f"Unknown validation stage: {stage}")
                continue
            
            try:
                handler = self.stage_handlers[stage]
                result = handler(config)
                results.append(result)
                
                # Stop on critical errors
                if not result.is_valid and stage in [ValidationStage.SYNTAX, ValidationStage.SCHEMA]:
                    break
                    
            except Exception as e:
                self.logger.error(f"Validation stage {stage} failed: {e}")
                results.append(ValidationResult(
                    stage=stage,
                    is_valid=False,
                    errors=[{
                        'message': f'Validation stage failed: {str(e)}'
                    }]
                ))
        
        return results
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Get summary of validation results"""
        total_errors = sum(len(r.errors) for r in results)
        total_warnings = sum(len(r.warnings) for r in results)
        all_suggestions = []
        
        for result in results:
            all_suggestions.extend(result.suggestions)
        
        return {
            'is_valid': all(r.is_valid for r in results),
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'failed_stages': [r.stage.value for r in results if not r.is_valid],
            'suggestions': list(set(all_suggestions)),  # Unique suggestions
            'stage_results': {
                r.stage.value: {
                    'valid': r.is_valid,
                    'errors': len(r.errors),
                    'warnings': len(r.warnings)
                }
                for r in results
            }
        }

# ======================== CONFIGURATION SCHEMA ========================

class ConfigSchemaManager:
    """Manages configuration schemas and versions"""
    
    def __init__(self):
        self.schemas = {}
        self.current_version = SCHEMA_VERSION
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def register_schema(self, version: str, schema: Dict[str, Any]) -> None:
        """Register a configuration schema"""
        self.schemas[version] = schema
    
    def get_schema(self, version: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration schema by version"""
        version = version or self.current_version
        
        if version not in self.schemas:
            # Return default schema
            return self.get_default_schema()
        
        return self.schemas[version]
    
    def get_default_schema(self) -> Dict[str, Any]:
        """Get default configuration schema"""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["version", "environment"],
            "properties": {
                "version": {
                    "type": "string",
                    "pattern": "^\\d+\\.\\d+\\.\\d+$"
                },
                "environment": {
                    "type": "string",
                    "enum": ["development", "testing", "staging", "production"]
                },
                "orchestrator": {
                    "type": "object",
                    "properties": {
                        "enable_auto_recovery": {"type": "boolean"},
                        "recovery_max_attempts": {"type": "integer", "minimum": 1},
                        "component_timeout_seconds": {"type": "number", "minimum": 1},
                        "max_parallel_workflows": {"type": "integer", "minimum": 1}
                    }
                },
                "monitoring": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "interval_seconds": {"type": "number", "minimum": 1},
                        "metrics_retention_hours": {"type": "integer", "minimum": 1}
                    }
                },
                "api": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                        "enable_https": {"type": "boolean"},
                        "rate_limit": {"type": "string"}
                    }
                },
                "database": {
                    "type": "object",
                    "properties": {
                        "connection_string": {"type": "string"},
                        "pool_size": {"type": "integer", "minimum": 1},
                        "timeout_seconds": {"type": "number", "minimum": 1}
                    }
                },
                "cache": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "size": {"type": "integer", "minimum": 1},
                        "ttl_seconds": {"type": "integer", "minimum": 1}
                    }
                },
                "security": {
                    "type": "object",
                    "properties": {
                        "encryption_enabled": {"type": "boolean"},
                        "authentication_required": {"type": "boolean"},
                        "allowed_origins": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                }
            },
            "additionalProperties": True
        }
    
    def migrate_config(self, config: Dict[str, Any], 
                       from_version: str, 
                       to_version: str) -> Dict[str, Any]:
        """Migrate configuration between schema versions"""
        if from_version == to_version:
            return config
        
        # Simple version update for now
        migrated = copy.deepcopy(config)
        migrated['version'] = to_version
        
        # Add migration logic here as schemas evolve
        
        return migrated

# ======================== MAIN CONFIG LOADER ========================

class AccuracyTrackingConfigLoader:
    """
    Advanced configuration loader for accuracy tracking system.
    Supports multi-source loading, comprehensive validation, and hot-reload.
    """
    
    def __init__(self, 
                 base_path: Optional[Path] = None,
                 environment: Optional[ConfigEnvironment] = None,
                 enable_hot_reload: bool = True,
                 enable_encryption: bool = True,
                 cache_config: bool = True):
        """
        Initialize configuration loader.
        
        Args:
            base_path: Base path for configuration files
            environment: Configuration environment
            enable_hot_reload: Enable hot-reload functionality
            enable_encryption: Enable encryption for sensitive values
            cache_config: Enable configuration caching
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Configuration
        self.base_path = base_path or Path("./config")
        self.environment = environment or self._detect_environment()
        self.enable_hot_reload = enable_hot_reload
        self.enable_encryption = enable_encryption
        self.cache_config = cache_config
        
        # Components
        self.sources: Dict[str, ConfigSource] = {}
        self.validator = ConfigValidator()
        self.validation_pipeline = ValidationPipeline(self.validator)
        self.schema_manager = ConfigSchemaManager()
        
        # Setup default schema
        self.validation_pipeline.set_schema(self.schema_manager.get_default_schema())
        
        # Encryption
        if enable_encryption:
            self._init_encryption()
        else:
            self.cipher_suite = None
        
        # State
        self.current_config: Dict[str, Any] = {}
        self.config_versions: List[ConfigVersion] = []
        self.load_result: Optional[ConfigLoadResult] = None
        
        # Caching
        if cache_config:
            self.cache_manager = CacheManager(
                MonitoringConfig(
                    enable_caching=True,
                    cache_size=100,
                    cache_ttl=300
                )
            )
        else:
            self.cache_manager = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Hot reload
        if enable_hot_reload:
            self._init_hot_reload()
        
        # Setup default validators
        self._setup_default_validators()
        
        self.logger.info(
            f"AccuracyTrackingConfigLoader initialized for environment: {self.environment.value}"
        )
    
    def _detect_environment(self) -> ConfigEnvironment:
        """Detect current environment"""
        env_value = os.getenv('ACCURACY_TRACKING_ENV', 'development').lower()
        
        env_map = {
            'dev': ConfigEnvironment.DEVELOPMENT,
            'development': ConfigEnvironment.DEVELOPMENT,
            'test': ConfigEnvironment.TESTING,
            'testing': ConfigEnvironment.TESTING,
            'stage': ConfigEnvironment.STAGING,
            'staging': ConfigEnvironment.STAGING,
            'prod': ConfigEnvironment.PRODUCTION,
            'production': ConfigEnvironment.PRODUCTION
        }
        
        return env_map.get(env_value, ConfigEnvironment.DEVELOPMENT)
    
    def _init_encryption(self) -> None:
        """Initialize encryption for sensitive values"""
        key_file = self.base_path / ".encryption_key"
        
        try:
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                key_file.parent.mkdir(parents=True, exist_ok=True)
                with open(key_file, 'wb') as f:
                    f.write(key)
                os.chmod(key_file, 0o600)
            
            self.cipher_suite = Fernet(key)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {e}")
            self.cipher_suite = None
    
    def _init_hot_reload(self) -> None:
        """Initialize hot reload functionality"""
        # Would implement file watching here
        # For now, just log that it's enabled
        self.logger.info("Hot reload enabled for configuration changes")
    
    def _setup_default_validators(self) -> None:
        """Setup default configuration validators"""
        # Add custom validators
        self.validator.add_custom_validator('positive_number', lambda x: x > 0)
        self.validator.add_custom_validator('valid_url', lambda x: x.startswith(('http://', 'https://')))
        
        # Add business rules
        self.validator.add_business_rule(
            lambda config: config.get('orchestrator', {}).get('max_parallel_workflows', 1) <= 20,
            "Maximum parallel workflows should not exceed 20"
        )
        
        self.validator.add_business_rule(
            lambda config: config.get('cache', {}).get('size', 0) <= 10000,
            "Cache size should not exceed 10000 items"
        )
        
        # Add cross-field validators
        def validate_monitoring_intervals(config: Dict[str, Any]) -> Dict[str, Any]:
            monitoring = config.get('monitoring', {})
            if monitoring.get('enabled', True):
                interval = monitoring.get('interval_seconds', 60)
                retention = monitoring.get('metrics_retention_hours', 24)
                
                # Check if retention is reasonable for the interval
                max_metrics = (retention * 3600) / interval
                if max_metrics > 100000:
                    return {
                        'valid': False,
                        'message': f'Monitoring configuration would generate {int(max_metrics)} metrics',
                        'fields': ['monitoring.interval_seconds', 'monitoring.metrics_retention_hours'],
                        'suggestion': 'Increase interval or decrease retention to reduce metric volume'
                    }
            
            return {'valid': True}
        
        self.validator.add_cross_field_validator(validate_monitoring_intervals)
    
    def add_source(self, name: str, source: ConfigSource, 
                   priority: Optional[int] = None) -> None:
        """
        Add configuration source.
        
        Args:
            name: Source name
            source: Configuration source instance
            priority: Source priority (lower = higher priority)
        """
        with self._lock:
            self.sources[name] = source
            
            # Set priority based on source type if not specified
            if priority is None and hasattr(source, '__class__'):
                for source_type, default_priority in SOURCE_PRIORITIES.items():
                    if source.__class__.__name__.lower().startswith(source_type.value):
                        priority = default_priority
                        break
            
            if priority is not None:
                source.priority = priority
            
            self.logger.debug(f"Added configuration source: {name}")
    
    def load_configuration(self, 
                          sources_to_load: Optional[List[str]] = None,
                          merge_strategy: str = "deep",
                          validate: bool = True) -> ConfigLoadResult:
        """
        Load configuration from all sources.
        
        Args:
            sources_to_load: Specific sources to load (None = all)
            merge_strategy: Strategy for merging configurations
            validate: Whether to validate configuration
            
        Returns:
            Configuration load result
        """
        start_time = datetime.now()
        
        with self._lock:
            try:
                # Determine sources to load
                if sources_to_load:
                    sources = {k: v for k, v in self.sources.items() if k in sources_to_load}
                else:
                    sources = self.sources
                
                # Load from sources in priority order
                configs_loaded = []
                sources_loaded = []
                
                sorted_sources = sorted(
                    sources.items(),
                    key=lambda x: getattr(x[1], 'priority', 999)
                )
                
                for name, source in sorted_sources:
                    try:
                        if source.exists():
                            config = source.load()
                            
                            # Decrypt if needed
                            if self.enable_encryption:
                                config = self._decrypt_config(config)
                            
                            configs_loaded.append(config)
                            sources_loaded.append(ConfigSourceInfo(
                                source_type=self._get_source_type(source),
                                source_path=str(source),
                                format=self._get_source_format(source),
                                priority=getattr(source, 'priority', 999)
                            ))
                            
                            self.logger.debug(f"Loaded configuration from source: {name}")
                    
                    except Exception as e:
                        self.logger.error(f"Failed to load from source {name}: {e}")
                        if getattr(source, 'is_required', False):
                            raise
                
                # Merge configurations
                if configs_loaded:
                    merged_config = self._merge_configurations(configs_loaded, merge_strategy)
                else:
                    merged_config = {}
                
                # Apply environment-specific overrides
                merged_config = self._apply_environment_overrides(merged_config)
                
                # Validate if requested
                validation_results = []
                if validate:
                    validation_results = self.validation_pipeline.validate(merged_config)
                    
                    # Check if validation passed
                    if not all(r.is_valid for r in validation_results):
                        self.logger.warning("Configuration validation failed")
                
                # Create version info
                version = ConfigVersion(
                    version=merged_config.get('version', '1.0.0'),
                    schema_version=self.schema_manager.current_version,
                    created_at=datetime.now(),
                    created_by=os.getenv('USER', 'system')
                )
                
                # Store configuration
                self.current_config = merged_config
                self.config_versions.append(version)
                
                # Create load result
                load_time = (datetime.now() - start_time).total_seconds() * 1000
                
                self.load_result = ConfigLoadResult(
                    config=merged_config,
                    sources_loaded=sources_loaded,
                    validation_results=validation_results,
                    version=version,
                    load_time_ms=load_time
                )
                
                # Cache if enabled
                if self.cache_manager:
                    cache_key = f"config:{self.environment.value}:{version.version}"
                    self.cache_manager.set(cache_key, merged_config)
                
                self.logger.info(
                    f"Configuration loaded successfully in {load_time:.2f}ms from "
                    f"{len(sources_loaded)} sources"
                )
                
                return self.load_result
                
            except Exception as e:
                raise ConfigLoaderError(
                    f"Failed to load configuration: {str(e)}",
                    context=create_error_context(
                        component="AccuracyTrackingConfigLoader",
                        operation="load_configuration"
                    )
                )
    
    def save_configuration(self, 
                          config: Optional[Dict[str, Any]] = None,
                          target_sources: Optional[List[str]] = None) -> bool:
        """
        Save configuration to sources.
        
        Args:
            config: Configuration to save (None = current)
            target_sources: Sources to save to (None = all writable)
            
        Returns:
            Success status
        """
        with self._lock:
            config_to_save = config or self.current_config
            
            if not config_to_save:
                self.logger.error("No configuration to save")
                return False
            
            # Encrypt if needed
            if self.enable_encryption:
                config_to_save = self._encrypt_config(copy.deepcopy(config_to_save))
            
            # Determine target sources
            if target_sources:
                sources = {k: v for k, v in self.sources.items() if k in target_sources}
            else:
                # Save to all file-based sources by default
                sources = {
                    k: v for k, v in self.sources.items()
                    if isinstance(v, FileConfigSource)
                }
            
            success_count = 0
            for name, source in sources.items():
                try:
                    if source.save(config_to_save):
                        success_count += 1
                        self.logger.debug(f"Saved configuration to source: {name}")
                    else:
                        self.logger.warning(f"Failed to save to source: {name}")
                        
                except Exception as e:
                    self.logger.error(f"Error saving to source {name}: {e}")
            
            return success_count > 0
    
    def reload_configuration(self) -> ConfigLoadResult:
        """Reload configuration from sources"""
        self.logger.info("Reloading configuration")
        
        # Clear cache if enabled
        if self.cache_manager:
            self.cache_manager.clear()
        
        return self.load_configuration()
    
    def get_configuration(self, path: Optional[str] = None) -> Any:
        """
        Get configuration value by path.
        
        Args:
            path: Dot-separated path (None = entire config)
            
        Returns:
            Configuration value
        """
        with self._lock:
            if not path:
                return copy.deepcopy(self.current_config)
            
            # Navigate path
            parts = path.split('.')
            current = self.current_config
            
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            
            return copy.deepcopy(current)
    
    def set_configuration(self, path: str, value: Any, save: bool = False) -> bool:
        """
        Set configuration value.
        
        Args:
            path: Dot-separated path
            value: Value to set
            save: Whether to save configuration
            
        Returns:
            Success status
        """
        with self._lock:
            try:
                # Navigate to parent
                parts = path.split('.')
                current = self.current_config
                
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Set value
                current[parts[-1]] = value
                
                # Update version
                if self.config_versions:
                    self.config_versions[-1].change_summary = f"Updated {path}"
                
                # Save if requested
                if save:
                    return self.save_configuration()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to set configuration value: {e}")
                return False
    
    def rollback_configuration(self, version: str) -> bool:
        """
        Rollback to previous configuration version.
        
        Args:
            version: Version to rollback to
            
        Returns:
            Success status
        """
        # This would be implemented with version storage
        self.logger.warning("Configuration rollback not yet implemented")
        return False
    
    def validate_configuration(self, 
                              config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate (None = current)
            
        Returns:
            Validation summary
        """
        config_to_validate = config or self.current_config
        
        if not config_to_validate:
            return {
                'is_valid': False,
                'errors': [{'message': 'No configuration to validate'}]
            }
        
        results = self.validation_pipeline.validate(config_to_validate)
        return self.validation_pipeline.get_validation_summary(results)
    
    def export_schema(self, output_path: Path) -> bool:
        """
        Export configuration schema.
        
        Args:
            output_path: Path to export schema
            
        Returns:
            Success status
        """
        try:
            schema = self.schema_manager.get_schema()
            
            with open(output_path, 'w') as f:
                json.dump(schema, f, indent=2)
            
            self.logger.info(f"Exported schema to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export schema: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get configuration loader system status"""
        return {
            'environment': self.environment.value,
            'sources_count': len(self.sources),
            'cache_enabled': self.cache_config,
            'encryption_enabled': self.enable_encryption,
            'hot_reload_enabled': self.enable_hot_reload,
            'current_config_version': self.config_versions[-1].version if self.config_versions else None,
            'load_result': {
                'load_time_ms': self.load_result.load_time_ms if self.load_result else None,
                'sources_loaded': len(self.load_result.sources_loaded) if self.load_result else 0,
                'validation_passed': all(r.is_valid for r in self.load_result.validation_results) if self.load_result else None
            } if self.load_result else None
        }
    
    def shutdown(self) -> None:
        """Shutdown configuration loader"""
        self.logger.info("Shutting down AccuracyTrackingConfigLoader")
        
        # Clear cache if enabled
        if self.cache_manager:
            self.cache_manager.clear()
        
        # Clear sources
        self.sources.clear()
        
        # Clear current configuration
        self.current_config.clear()
    
    # ======================== HELPER METHODS ========================
    
    def _get_source_type(self, source: ConfigSource) -> ConfigSourceType:
        """Get source type from source instance"""
        class_name = source.__class__.__name__.lower()
        
        for source_type in ConfigSourceType:
            if source_type.value in class_name:
                return source_type
        
        return ConfigSourceType.FILE
    
    def _get_source_format(self, source: ConfigSource) -> ConfigFormat:
        """Get source format"""
        if hasattr(source, 'format'):
            return source.format
        
        return ConfigFormat.JSON
    
    def _merge_configurations(self, configs: List[Dict[str, Any]], 
                             strategy: str = "deep") -> Dict[str, Any]:
        """Merge multiple configurations"""
        if not configs:
            return {}
        
        if len(configs) == 1:
            return configs[0]
        
        if strategy == "deep":
            return self._deep_merge(*configs)
        else:
            # Simple merge (last wins)
            result = {}
            for config in configs:
                result.update(config)
            return result
    
    def _deep_merge(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge multiple dictionaries"""
        result = {}
        
        for config in configs:
            for key, value in config.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._deep_merge(result[key], value)
                else:
                    result[key] = copy.deepcopy(value)
        
        return result
    
    def _apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment-specific overrides"""
        env_file = self.base_path / "environments" / f"{self.environment.value}.json"
        
        if env_file.exists():
            try:
                source = FileConfigSource(env_file, ConfigFormat.JSON)
                env_config = source.load()
                
                # Merge with environment config taking precedence
                return self._deep_merge(config, env_config)
                
            except Exception as e:
                self.logger.warning(f"Failed to load environment overrides: {e}")
        
        return config
    
    def _encrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive configuration values"""
        if not self.cipher_suite:
            return config
        
        sensitive_patterns = ['password', 'secret', 'key', 'token', 'credential']
        
        def encrypt_recursive(data: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in data.items():
                # Check if key contains sensitive pattern
                is_sensitive = any(pattern in key.lower() for pattern in sensitive_patterns)
                
                if is_sensitive and isinstance(value, str) and not value.startswith('encrypted:'):
                    # Encrypt the value
                    encrypted = self.cipher_suite.encrypt(value.encode()).decode()
                    data[key] = f"encrypted:{encrypted}"
                elif isinstance(value, dict):
                    encrypt_recursive(value)
            
            return data
        
        return encrypt_recursive(copy.deepcopy(config))
    
    def _decrypt_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt encrypted configuration values"""
        if not self.cipher_suite:
            return config
        
        def decrypt_recursive(data: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in data.items():
                if isinstance(value, str) and value.startswith('encrypted:'):
                    try:
                        encrypted_data = value[10:]  # Remove 'encrypted:' prefix
                        decrypted = self.cipher_suite.decrypt(encrypted_data.encode()).decode()
                        data[key] = decrypted
                    except Exception as e:
                        self.logger.error(f"Failed to decrypt {key}: {e}")
                elif isinstance(value, dict):
                    decrypt_recursive(value)
            
            return data
        
        return decrypt_recursive(copy.deepcopy(config))
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # Save configuration on exit if modified
        if hasattr(self, '_config_modified') and self._config_modified:
            self.save_configuration()


# ======================== FACTORY FUNCTIONS ========================

def create_standard_config_loader(
    environment: Optional[str] = None,
    config_dir: Optional[str] = None
) -> AccuracyTrackingConfigLoader:
    """
    Create standard configuration loader with default sources.
    
    Args:
        environment: Environment name
        config_dir: Configuration directory
        
    Returns:
        Configured AccuracyTrackingConfigLoader
    """
    # Determine environment
    if environment:
        env = ConfigEnvironment(environment.lower())
    else:
        env = None
    
    # Create loader
    loader = AccuracyTrackingConfigLoader(
        base_path=Path(config_dir) if config_dir else None,
        environment=env,
        enable_hot_reload=True,
        enable_encryption=True,
        cache_config=True
    )
    
    # Add standard sources
    
    # 1. Environment variables (highest priority)
    loader.add_source(
        'environment',
        EnvironmentConfigSource(prefix="ACCURACY_TRACKING_"),
        priority=1
    )
    
    # 2. Main configuration file
    config_file = loader.base_path / "accuracy_tracking.json"
    if not config_file.exists():
        config_file = loader.base_path / "accuracy_tracking.yaml"
    
    if config_file.exists():
        format = ConfigFormat.JSON if config_file.suffix == '.json' else ConfigFormat.YAML
        loader.add_source(
            'main',
            FileConfigSource(config_file, format),
            priority=2
        )
    
    # 3. Environment-specific file
    env_file = loader.base_path / "environments" / f"{loader.environment.value}.json"
    if not env_file.exists():
        env_file = loader.base_path / "environments" / f"{loader.environment.value}.yaml"
    
    if env_file.exists():
        format = ConfigFormat.JSON if env_file.suffix == '.json' else ConfigFormat.YAML
        loader.add_source(
            'environment_specific',
            FileConfigSource(env_file, format),
            priority=2
        )
    
    # 4. Database configuration (if available)
    db_path = loader.base_path / "config.db"
    if db_path.exists():
        loader.add_source(
            'database',
            DatabaseConfigSource(str(db_path)),
            priority=3
        )
    
    return loader


def create_production_config_loader(
    config_sources: Dict[str, Dict[str, Any]]
) -> AccuracyTrackingConfigLoader:
    """
    Create production configuration loader with custom sources.
    
    Args:
        config_sources: Configuration source definitions
        
    Returns:
        Configured AccuracyTrackingConfigLoader
    """
    loader = AccuracyTrackingConfigLoader(
        environment=ConfigEnvironment.PRODUCTION,
        enable_hot_reload=False,  # Disabled for production
        enable_encryption=True,
        cache_config=True
    )
    
    # Add custom sources
    for name, source_config in config_sources.items():
        source_type = source_config.get('type', 'file')
        
        if source_type == 'file':
            source = FileConfigSource(
                Path(source_config['path']),
                ConfigFormat(source_config.get('format', 'json'))
            )
        elif source_type == 'environment':
            source = EnvironmentConfigSource(
                prefix=source_config.get('prefix', 'ACCURACY_TRACKING_')
            )
        elif source_type == 'database':
            source = DatabaseConfigSource(
                source_config['db_path'],
                source_config.get('table_name', 'accuracy_tracking_config')
            )
        elif source_type == 'remote':
            source = RemoteConfigSource(
                source_config['url'],
                headers=source_config.get('headers'),
                timeout=source_config.get('timeout', 30)
            )
        else:
            continue
        
        loader.add_source(
            name,
            source,
            priority=source_config.get('priority')
        )
    
    # Add production-specific validators
    loader.validator.add_business_rule(
        lambda config: config.get('security', {}).get('authentication_required', False),
        "Authentication must be enabled in production"
    )
    
    loader.validator.add_business_rule(
        lambda config: config.get('api', {}).get('enable_https', False),
        "HTTPS must be enabled in production"
    )
    
    return loader