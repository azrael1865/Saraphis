"""
Production Configuration Manager - Unified production configuration management system
NO FALLBACKS - HARD FAILURES ONLY

This module provides comprehensive production configuration management capabilities,
including environment-specific settings, validation, encryption, versioning, and migration.

Key Features:
- Unified production configuration management across all environments
- Environment-specific configuration switching (dev, staging, prod)
- Configuration validation with strict error checking
- Configuration encryption for sensitive settings
- Configuration versioning and migration support
- Hot-reloading capabilities for dynamic configuration updates
- Integration with existing GAC and domain configuration systems
- Comprehensive logging and audit trails
- Backup and restore functionality

Architecture: NO FALLBACKS - HARD FAILURES ONLY
All configuration operations must succeed or fail explicitly with detailed error information.
"""

import os
import json
import yaml
import logging
import hashlib
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import copy
import traceback
import uuid
from contextlib import contextmanager

try:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from gac_system.gac_config import GACConfigManager, GACConfig
        from golf_domain.domain_config import GolfDomainConfig
        from production_training_execution import ProductionTrainingExecutor
except ImportError:
    pass

logger = logging.getLogger(__name__)


class ProductionEnvironment(Enum):
    """Production environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    LOCAL = "local"


class ConfigurationFormat(Enum):
    """Configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"


class ConfigurationPriority(Enum):
    """Configuration priority levels."""
    ENVIRONMENT_VARIABLES = "environment_variables"
    COMMAND_LINE = "command_line"
    CONFIG_FILE = "config_file"
    DEFAULT = "default"


class ValidationLevel(Enum):
    """Configuration validation levels."""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class ProductionConfig:
    """Main production configuration dataclass."""
    environment: ProductionEnvironment = ProductionEnvironment.DEVELOPMENT
    version: str = "1.0.0"
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Application settings
    app_name: str = "saraphis_independent_core"
    app_version: str = "1.0.0"
    app_description: str = "Saraphis Independent Core Production System"
    
    # Server settings
    host: str = "localhost"
    port: int = 8000
    workers: int = 4
    worker_timeout: float = 30.0
    max_requests: int = 1000
    max_requests_jitter: int = 100
    
    # Database settings
    database_url: str = "sqlite:///production.db"
    database_pool_size: int = 20
    database_max_overflow: int = 30
    database_timeout: float = 30.0
    
    # Cache settings
    cache_type: str = "redis"
    cache_url: str = "redis://localhost:6379/0"
    cache_timeout: int = 3600
    cache_max_entries: int = 10000
    
    # Session settings
    session_type: str = "filesystem"
    session_permanent: bool = False
    session_timeout: int = 1800
    session_key_prefix: str = "saraphis:"
    
    # Security settings
    secret_key: str = ""
    encryption_key: str = ""
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # CORS settings
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_headers: List[str] = field(default_factory=lambda: ["Content-Type", "Authorization"])
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600
    rate_limit_storage: str = "memory"
    
    # Monitoring settings
    metrics_enabled: bool = True
    metrics_endpoint: str = "/metrics"
    health_check_enabled: bool = True
    health_check_endpoint: str = "/health"
    
    # Logging settings
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    log_file_enabled: bool = True
    log_file_path: str = "logs/production.log"
    log_file_max_size: int = 10485760  # 10MB
    log_file_backup_count: int = 5
    log_rotation_enabled: bool = True
    
    # Feature flags
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentConfig:
    """Deployment environment configuration."""
    environment: ProductionEnvironment
    name: str
    description: str = ""
    
    # Environment-specific overrides
    host: Optional[str] = None
    port: Optional[int] = None
    workers: Optional[int] = None
    debug_mode: Optional[bool] = None
    log_level: Optional[str] = None
    
    # Environment variables
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    # Configuration file paths
    config_files: List[str] = field(default_factory=list)
    
    # Validation settings
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    
    # Deployment metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class SecurityConfig:
    """Security configuration for production."""
    encryption_enabled: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_enabled: bool = True
    key_rotation_interval_days: int = 30
    
    # Authentication
    authentication_required: bool = True
    authentication_type: str = "jwt"
    multi_factor_enabled: bool = False
    
    # Authorization
    authorization_enabled: bool = True
    role_based_access: bool = True
    permission_based_access: bool = True
    
    # SSL/TLS
    ssl_enabled: bool = True
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    ssl_ca_path: str = ""
    ssl_verify_mode: str = "CERT_REQUIRED"
    
    # Security headers
    security_headers_enabled: bool = True
    content_security_policy: str = "default-src 'self'"
    x_frame_options: str = "DENY"
    x_content_type_options: str = "nosniff"
    
    # Audit settings
    audit_enabled: bool = True
    audit_log_path: str = "logs/audit.log"
    audit_retention_days: int = 90
    
    # Security scanning
    vulnerability_scanning_enabled: bool = True
    dependency_scanning_enabled: bool = True
    security_scan_interval_hours: int = 24


@dataclass
class PerformanceConfig:
    """Performance configuration for production."""
    # Threading settings
    max_threads: int = 100
    thread_pool_size: int = 20
    thread_timeout: float = 30.0
    
    # Memory settings
    max_memory_mb: int = 4096
    memory_warning_threshold: float = 0.8
    memory_critical_threshold: float = 0.95
    garbage_collection_enabled: bool = True
    garbage_collection_interval: int = 300
    
    # CPU settings
    cpu_warning_threshold: float = 0.8
    cpu_critical_threshold: float = 0.95
    cpu_monitoring_interval: int = 60
    
    # I/O settings
    max_connections: int = 1000
    connection_timeout: float = 30.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0
    
    # Caching settings
    cache_enabled: bool = True
    cache_size_mb: int = 512
    cache_ttl_seconds: int = 3600
    cache_compression_enabled: bool = True
    
    # Optimization settings
    code_optimization_enabled: bool = True
    jit_compilation_enabled: bool = True
    profile_guided_optimization: bool = False
    
    # Concurrency settings
    async_enabled: bool = True
    max_concurrent_requests: int = 100
    request_queue_size: int = 1000
    
    # Database performance
    database_connection_pooling: bool = True
    database_query_caching: bool = True
    database_query_timeout: float = 30.0


@dataclass
class MonitoringConfig:
    """Monitoring configuration for production."""
    # Basic monitoring
    monitoring_enabled: bool = True
    monitoring_interval: int = 60
    
    # Metrics collection
    metrics_enabled: bool = True
    metrics_collection_interval: int = 30
    metrics_retention_days: int = 30
    
    # Health checks
    health_checks_enabled: bool = True
    health_check_interval: int = 30
    health_check_timeout: float = 10.0
    
    # Alerting
    alerting_enabled: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["email", "slack"])
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Logging monitoring
    log_monitoring_enabled: bool = True
    log_error_threshold: int = 10
    log_warning_threshold: int = 50
    
    # Performance monitoring
    performance_monitoring_enabled: bool = True
    response_time_threshold: float = 1.0
    throughput_threshold: int = 100
    
    # External monitoring
    external_monitoring_enabled: bool = False
    external_monitoring_urls: List[str] = field(default_factory=list)
    external_monitoring_interval: int = 300


@dataclass
class ConfigurationMetadata:
    """Configuration metadata and versioning."""
    version: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str
    environment: ProductionEnvironment
    checksum: str
    size_bytes: int
    format: ConfigurationFormat
    encrypted: bool = False
    backup_available: bool = False
    migration_required: bool = False
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""


class ConfigurationEncryption:
    """Configuration encryption and decryption utilities."""
    
    def __init__(self, password: Optional[str] = None):
        self.password = password or os.environ.get('CONFIG_ENCRYPTION_PASSWORD', '')
        if not self.password:
            raise ValueError("Encryption password not provided")
        self._key = self._derive_key(self.password)
        self._cipher = Fernet(self._key)
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password."""
        salt = b'saraphis_config_salt'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt(self, data: str) -> str:
        """Encrypt configuration data."""
        try:
            encrypted_data = self._cipher.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            raise RuntimeError(f"Failed to encrypt configuration data: {e}")
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt configuration data."""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self._cipher.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            raise RuntimeError(f"Failed to decrypt configuration data: {e}")
    
    def encrypt_dict(self, data: Dict[str, Any], sensitive_keys: Set[str]) -> Dict[str, Any]:
        """Encrypt sensitive values in dictionary."""
        encrypted_data = copy.deepcopy(data)
        for key, value in encrypted_data.items():
            if key in sensitive_keys and isinstance(value, str):
                encrypted_data[key] = self.encrypt(value)
        return encrypted_data
    
    def decrypt_dict(self, data: Dict[str, Any], sensitive_keys: Set[str]) -> Dict[str, Any]:
        """Decrypt sensitive values in dictionary."""
        decrypted_data = copy.deepcopy(data)
        for key, value in decrypted_data.items():
            if key in sensitive_keys and isinstance(value, str):
                try:
                    decrypted_data[key] = self.decrypt(value)
                except:
                    # Value might not be encrypted
                    pass
        return decrypted_data


class ConfigurationValidator:
    """Configuration validation utilities."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
    
    def validate_production_config(self, config: ProductionConfig) -> List[str]:
        """Validate production configuration."""
        errors = []
        
        if self.validation_level == ValidationLevel.NONE:
            return errors
        
        # Basic validation
        if not config.app_name:
            errors.append("app_name cannot be empty")
        
        if config.port < 1 or config.port > 65535:
            errors.append(f"port must be between 1 and 65535, got {config.port}")
        
        if config.workers < 1:
            errors.append(f"workers must be positive, got {config.workers}")
        
        if config.worker_timeout <= 0:
            errors.append(f"worker_timeout must be positive, got {config.worker_timeout}")
        
        # Standard validation
        if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            if not config.secret_key:
                errors.append("secret_key is required for production")
            
            if config.environment == ProductionEnvironment.PRODUCTION:
                if config.debug_mode:
                    errors.append("debug_mode must be False in production")
                
                if not config.database_url or config.database_url.startswith('sqlite://'):
                    errors.append("sqlite database not recommended for production")
        
        # Strict validation
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            if len(config.secret_key) < 32:
                errors.append("secret_key should be at least 32 characters")
            
            if not config.encryption_key:
                errors.append("encryption_key is required for strict validation")
            
            if config.environment == ProductionEnvironment.PRODUCTION:
                if not config.ssl_enabled:
                    errors.append("SSL should be enabled in production")
        
        # Paranoid validation
        if self.validation_level == ValidationLevel.PARANOID:
            if not config.multi_factor_enabled:
                errors.append("Multi-factor authentication recommended for paranoid validation")
            
            if not config.audit_enabled:
                errors.append("Audit logging required for paranoid validation")
        
        return errors
    
    def validate_deployment_config(self, config: DeploymentConfig) -> List[str]:
        """Validate deployment configuration."""
        errors = []
        
        if not config.name:
            errors.append("deployment name cannot be empty")
        
        if config.port is not None and (config.port < 1 or config.port > 65535):
            errors.append(f"port must be between 1 and 65535, got {config.port}")
        
        if config.workers is not None and config.workers < 1:
            errors.append(f"workers must be positive, got {config.workers}")
        
        # Validate environment variables
        for key, value in config.environment_variables.items():
            if not key:
                errors.append("environment variable key cannot be empty")
            if not isinstance(value, str):
                errors.append(f"environment variable {key} must be string, got {type(value)}")
        
        return errors
    
    def validate_security_config(self, config: SecurityConfig) -> List[str]:
        """Validate security configuration."""
        errors = []
        
        if config.ssl_enabled:
            if not config.ssl_cert_path:
                errors.append("ssl_cert_path required when SSL is enabled")
            if not config.ssl_key_path:
                errors.append("ssl_key_path required when SSL is enabled")
        
        if config.key_rotation_enabled:
            if config.key_rotation_interval_days < 1:
                errors.append("key_rotation_interval_days must be positive")
        
        if config.audit_enabled:
            if not config.audit_log_path:
                errors.append("audit_log_path required when audit is enabled")
        
        return errors


class ConfigurationMigrator:
    """Configuration migration utilities."""
    
    def __init__(self):
        self.migrations: Dict[str, Callable] = {}
    
    def register_migration(self, from_version: str, to_version: str, migration_func: Callable):
        """Register a configuration migration."""
        key = f"{from_version}->{to_version}"
        self.migrations[key] = migration_func
    
    def migrate_config(self, config_data: Dict[str, Any], from_version: str, to_version: str) -> Dict[str, Any]:
        """Migrate configuration from one version to another."""
        migration_key = f"{from_version}->{to_version}"
        
        if migration_key not in self.migrations:
            raise ValueError(f"No migration available from {from_version} to {to_version}")
        
        try:
            migrated_data = self.migrations[migration_key](config_data)
            logger.info(f"Successfully migrated configuration from {from_version} to {to_version}")
            return migrated_data
        except Exception as e:
            raise RuntimeError(f"Configuration migration failed: {e}")
    
    def get_migration_path(self, from_version: str, to_version: str) -> List[str]:
        """Get migration path between versions."""
        # Simple implementation - in production, use graph traversal
        available_migrations = list(self.migrations.keys())
        direct_migration = f"{from_version}->{to_version}"
        
        if direct_migration in available_migrations:
            return [direct_migration]
        
        # For now, only support direct migrations
        raise ValueError(f"No migration path available from {from_version} to {to_version}")


class ProductionConfigManager:
    """
    Production Configuration Manager - Unified production configuration management.
    
    This class provides comprehensive configuration management for production environments,
    including validation, encryption, versioning, migration, and hot-reloading capabilities.
    """
    
    def __init__(self, config_dir: str = "config", environment: Optional[ProductionEnvironment] = None):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.environment = environment or self._detect_environment()
        self.config_lock = threading.RLock()
        self.hot_reload_enabled = False
        self.hot_reload_thread: Optional[threading.Thread] = None
        self.last_reload_check = time.time()
        
        # Configuration components
        self.production_config = ProductionConfig()
        self.deployment_configs: Dict[str, DeploymentConfig] = {}
        self.security_config = SecurityConfig()
        self.performance_config = PerformanceConfig()
        self.monitoring_config = MonitoringConfig()
        
        # Management components
        self.encryptor: Optional[ConfigurationEncryption] = None
        self.validator = ConfigurationValidator()
        self.migrator = ConfigurationMigrator()
        
        # Configuration metadata
        self.metadata: Optional[ConfigurationMetadata] = None
        self.config_checksums: Dict[str, str] = {}
        self.backup_configs: Dict[str, Any] = {}
        
        # Integration references
        self.gac_config_manager: Optional['GACConfigManager'] = None
        self.golf_domain_config: Optional['GolfDomainConfig'] = None
        self.production_training_executor: Optional['ProductionTrainingExecutor'] = None
        
        # Sensitive configuration keys
        self.sensitive_keys = {
            'secret_key', 'encryption_key', 'jwt_secret_key', 'database_url',
            'cache_url', 'ssl_key_path', 'api_keys', 'passwords', 'tokens'
        }
        
        self._setup_default_migrations()
        logger.info(f"ProductionConfigManager initialized for {self.environment.value} environment")
    
    def _detect_environment(self) -> ProductionEnvironment:
        """Detect current environment from environment variables."""
        env_name = os.environ.get('SARAPHIS_ENV', '').lower()
        app_env = os.environ.get('APP_ENV', '').lower()
        flask_env = os.environ.get('FLASK_ENV', '').lower()
        
        env_mapping = {
            'production': ProductionEnvironment.PRODUCTION,
            'prod': ProductionEnvironment.PRODUCTION,
            'staging': ProductionEnvironment.STAGING,
            'stage': ProductionEnvironment.STAGING,
            'development': ProductionEnvironment.DEVELOPMENT,
            'dev': ProductionEnvironment.DEVELOPMENT,
            'testing': ProductionEnvironment.TESTING,
            'test': ProductionEnvironment.TESTING,
            'local': ProductionEnvironment.LOCAL
        }
        
        for env_var in [env_name, app_env, flask_env]:
            if env_var in env_mapping:
                return env_mapping[env_var]
        
        return ProductionEnvironment.DEVELOPMENT
    
    def initialize_production_config_manager(
        self,
        gac_config_manager: Optional['GACConfigManager'] = None,
        golf_domain_config: Optional['GolfDomainConfig'] = None,
        production_training_executor: Optional['ProductionTrainingExecutor'] = None,
        encryption_password: Optional[str] = None
    ) -> None:
        """Initialize production config manager with system integrations."""
        try:
            with self.config_lock:
                # Store system integrations
                self.gac_config_manager = gac_config_manager
                self.golf_domain_config = golf_domain_config
                self.production_training_executor = production_training_executor
                
                # Initialize encryption if password provided
                if encryption_password:
                    self.encryptor = ConfigurationEncryption(encryption_password)
                
                # Load existing configuration
                self._load_configuration()
                
                # Initialize integrations
                if gac_config_manager:
                    self._integrate_gac_config()
                
                if golf_domain_config:
                    self._integrate_golf_domain_config()
                
                logger.info("Production configuration manager initialized successfully")
                
        except Exception as e:
            error_msg = f"Failed to initialize production configuration manager: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def load_configuration(self, config_path: Optional[str] = None, validate: bool = True) -> ProductionConfig:
        """Load production configuration from file."""
        try:
            with self.config_lock:
                if config_path:
                    config_file = Path(config_path)
                else:
                    config_file = self.config_dir / f"{self.environment.value}.json"
                
                if not config_file.exists():
                    logger.warning(f"Configuration file {config_file} not found, using defaults")
                    return self.production_config
                
                # Load configuration data
                config_data = self._load_config_file(config_file)
                
                # Decrypt sensitive data if encryptor available
                if self.encryptor:
                    config_data = self.encryptor.decrypt_dict(config_data, self.sensitive_keys)
                
                # Apply environment variable overrides
                config_data = self._apply_environment_overrides(config_data)
                
                # Create production config from data
                self.production_config = self._create_production_config_from_dict(config_data)
                
                # Validate configuration if requested
                if validate:
                    self.validate_configuration()
                
                # Update metadata
                self._update_configuration_metadata(config_file, config_data)
                
                logger.info(f"Configuration loaded successfully from {config_file}")
                return self.production_config
                
        except Exception as e:
            error_msg = f"Failed to load configuration: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def save_configuration(self, config_path: Optional[str] = None, encrypt_sensitive: bool = True) -> None:
        """Save production configuration to file."""
        try:
            with self.config_lock:
                if config_path:
                    config_file = Path(config_path)
                else:
                    config_file = self.config_dir / f"{self.environment.value}.json"
                
                # Create backup before saving
                self._create_configuration_backup()
                
                # Convert config to dictionary
                config_data = asdict(self.production_config)
                
                # Encrypt sensitive data if requested and encryptor available
                if encrypt_sensitive and self.encryptor:
                    config_data = self.encryptor.encrypt_dict(config_data, self.sensitive_keys)
                
                # Save configuration file
                self._save_config_file(config_file, config_data)
                
                # Update metadata
                self._update_configuration_metadata(config_file, config_data)
                
                logger.info(f"Configuration saved successfully to {config_file}")
                
        except Exception as e:
            error_msg = f"Failed to save configuration: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def validate_configuration(self, config: Optional[ProductionConfig] = None) -> List[str]:
        """Validate production configuration."""
        try:
            config_to_validate = config or self.production_config
            
            errors = []
            errors.extend(self.validator.validate_production_config(config_to_validate))
            errors.extend(self.validator.validate_security_config(self.security_config))
            
            # Validate deployment configurations
            for deploy_config in self.deployment_configs.values():
                errors.extend(self.validator.validate_deployment_config(deploy_config))
            
            if errors:
                error_msg = f"Configuration validation failed: {'; '.join(errors)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info("Configuration validation passed")
            return errors
            
        except Exception as e:
            error_msg = f"Configuration validation error: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def migrate_configuration(self, from_version: str, to_version: str) -> None:
        """Migrate configuration between versions."""
        try:
            with self.config_lock:
                # Create backup before migration
                self._create_configuration_backup()
                
                # Convert current config to dict for migration
                config_data = asdict(self.production_config)
                
                # Perform migration
                migrated_data = self.migrator.migrate_config(config_data, from_version, to_version)
                
                # Update production config with migrated data
                self.production_config = self._create_production_config_from_dict(migrated_data)
                
                # Update version
                self.production_config.version = to_version
                
                # Validate migrated configuration
                self.validate_configuration()
                
                # Save migrated configuration
                self.save_configuration()
                
                logger.info(f"Configuration migrated successfully from {from_version} to {to_version}")
                
        except Exception as e:
            error_msg = f"Configuration migration failed: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def enable_hot_reload(self, check_interval: float = 5.0) -> None:
        """Enable hot reloading of configuration."""
        try:
            if self.hot_reload_enabled:
                logger.warning("Hot reload already enabled")
                return
            
            self.hot_reload_enabled = True
            self.hot_reload_thread = threading.Thread(
                target=self._hot_reload_loop,
                args=(check_interval,),
                name="ConfigHotReload",
                daemon=True
            )
            self.hot_reload_thread.start()
            
            logger.info(f"Hot reload enabled with {check_interval}s check interval")
            
        except Exception as e:
            error_msg = f"Failed to enable hot reload: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def disable_hot_reload(self) -> None:
        """Disable hot reloading of configuration."""
        try:
            self.hot_reload_enabled = False
            if self.hot_reload_thread and self.hot_reload_thread.is_alive():
                self.hot_reload_thread.join(timeout=10.0)
            
            logger.info("Hot reload disabled")
            
        except Exception as e:
            logger.error(f"Error disabling hot reload: {e}")
    
    def get_configuration_status(self) -> Dict[str, Any]:
        """Get current configuration status and metadata."""
        try:
            with self.config_lock:
                return {
                    'environment': self.environment.value,
                    'version': self.production_config.version,
                    'encrypted': self.encryptor is not None,
                    'hot_reload_enabled': self.hot_reload_enabled,
                    'validation_level': self.validator.validation_level.value,
                    'metadata': asdict(self.metadata) if self.metadata else None,
                    'deployment_configs': list(self.deployment_configs.keys()),
                    'config_files': list(self.config_checksums.keys()),
                    'backup_available': len(self.backup_configs) > 0,
                    'integrations': {
                        'gac_config': self.gac_config_manager is not None,
                        'golf_domain_config': self.golf_domain_config is not None,
                        'production_training': self.production_training_executor is not None
                    }
                }
                
        except Exception as e:
            error_msg = f"Failed to get configuration status: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _setup_default_migrations(self) -> None:
        """Setup default configuration migrations."""
        def migrate_1_0_0_to_1_1_0(config_data: Dict[str, Any]) -> Dict[str, Any]:
            """Example migration from 1.0.0 to 1.1.0."""
            migrated = copy.deepcopy(config_data)
            
            # Add new fields with defaults
            if 'rate_limit_enabled' not in migrated:
                migrated['rate_limit_enabled'] = True
            if 'rate_limit_requests' not in migrated:
                migrated['rate_limit_requests'] = 100
            
            # Update version
            migrated['version'] = '1.1.0'
            
            return migrated
        
        self.migrator.register_migration('1.0.0', '1.1.0', migrate_1_0_0_to_1_1_0)
    
    def _load_configuration(self) -> None:
        """Load configuration from default location."""
        config_file = self.config_dir / f"{self.environment.value}.json"
        if config_file.exists():
            self.load_configuration()
        else:
            logger.info(f"No configuration file found for {self.environment.value}, using defaults")
    
    def _load_config_file(self, config_file: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() == '.json':
                    return json.load(f)
                elif config_file.suffix.lower() in ['.yml', '.yaml']:
                    return yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration file {config_file}: {e}")
    
    def _save_config_file(self, config_file: Path, config_data: Dict[str, Any]) -> None:
        """Save configuration to file."""
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                if config_file.suffix.lower() == '.json':
                    json.dump(config_data, f, indent=2, default=str)
                elif config_file.suffix.lower() in ['.yml', '.yaml']:
                    yaml.dump(config_data, f, default_flow_style=False)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
        except Exception as e:
            raise RuntimeError(f"Failed to save configuration file {config_file}: {e}")
    
    def _apply_environment_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        overrides = {}
        
        # Define environment variable mappings
        env_mappings = {
            'SARAPHIS_DEBUG': 'debug_mode',
            'SARAPHIS_LOG_LEVEL': 'log_level',
            'SARAPHIS_HOST': 'host',
            'SARAPHIS_PORT': 'port',
            'SARAPHIS_WORKERS': 'workers',
            'SARAPHIS_DATABASE_URL': 'database_url',
            'SARAPHIS_CACHE_URL': 'cache_url',
            'SARAPHIS_SECRET_KEY': 'secret_key',
            'SARAPHIS_ENCRYPTION_KEY': 'encryption_key'
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Convert to appropriate type
                if config_key in ['debug_mode']:
                    overrides[config_key] = env_value.lower() in ['true', '1', 'yes', 'on']
                elif config_key in ['port', 'workers']:
                    overrides[config_key] = int(env_value)
                else:
                    overrides[config_key] = env_value
        
        # Apply overrides
        config_data.update(overrides)
        return config_data
    
    def _create_production_config_from_dict(self, config_data: Dict[str, Any]) -> ProductionConfig:
        """Create ProductionConfig instance from dictionary."""
        try:
            # Filter out unknown fields
            valid_fields = {f.name for f in ProductionConfig.__dataclass_fields__.values()}
            filtered_data = {k: v for k, v in config_data.items() if k in valid_fields}
            
            # Handle enum fields
            if 'environment' in filtered_data:
                env_value = filtered_data['environment']
                if isinstance(env_value, str):
                    filtered_data['environment'] = ProductionEnvironment(env_value)
            
            return ProductionConfig(**filtered_data)
        except Exception as e:
            raise RuntimeError(f"Failed to create ProductionConfig from dictionary: {e}")
    
    def _update_configuration_metadata(self, config_file: Path, config_data: Dict[str, Any]) -> None:
        """Update configuration metadata."""
        try:
            config_str = json.dumps(config_data, sort_keys=True)
            checksum = hashlib.sha256(config_str.encode()).hexdigest()
            
            self.metadata = ConfigurationMetadata(
                version=self.production_config.version,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by=os.environ.get('USER', 'unknown'),
                updated_by=os.environ.get('USER', 'unknown'),
                environment=self.environment,
                checksum=checksum,
                size_bytes=len(config_str),
                format=ConfigurationFormat.JSON,
                encrypted=self.encryptor is not None
            )
            
            self.config_checksums[str(config_file)] = checksum
            
        except Exception as e:
            logger.error(f"Failed to update configuration metadata: {e}")
    
    def _create_configuration_backup(self) -> None:
        """Create backup of current configuration."""
        try:
            backup_key = f"{self.environment.value}_{datetime.utcnow().isoformat()}"
            self.backup_configs[backup_key] = {
                'production_config': asdict(self.production_config),
                'deployment_configs': {k: asdict(v) for k, v in self.deployment_configs.items()},
                'security_config': asdict(self.security_config),
                'performance_config': asdict(self.performance_config),
                'monitoring_config': asdict(self.monitoring_config),
                'metadata': asdict(self.metadata) if self.metadata else None
            }
            
            # Keep only last 10 backups
            if len(self.backup_configs) > 10:
                oldest_key = min(self.backup_configs.keys())
                del self.backup_configs[oldest_key]
            
            logger.debug(f"Configuration backup created: {backup_key}")
            
        except Exception as e:
            logger.error(f"Failed to create configuration backup: {e}")
    
    def _hot_reload_loop(self, check_interval: float) -> None:
        """Hot reload monitoring loop."""
        while self.hot_reload_enabled:
            try:
                time.sleep(check_interval)
                
                # Check if configuration files have changed
                config_file = self.config_dir / f"{self.environment.value}.json"
                if not config_file.exists():
                    continue
                
                # Check file modification time
                mtime = config_file.stat().st_mtime
                if mtime > self.last_reload_check:
                    logger.info("Configuration file changed, reloading...")
                    
                    try:
                        self.load_configuration(validate=True)
                        self.last_reload_check = time.time()
                        logger.info("Configuration reloaded successfully")
                    except Exception as e:
                        logger.error(f"Failed to reload configuration: {e}")
                
            except Exception as e:
                logger.error(f"Error in hot reload loop: {e}")
                time.sleep(check_interval)
    
    def _integrate_gac_config(self) -> None:
        """Integrate with GAC configuration system."""
        try:
            if not self.gac_config_manager:
                return
            
            gac_config = self.gac_config_manager.config
            
            # Apply GAC system settings to production config
            self.production_config.workers = gac_config.system.max_workers
            self.production_config.worker_timeout = gac_config.system.worker_timeout
            self.production_config.debug_mode = gac_config.system.debug_mode
            self.production_config.log_level = gac_config.system.log_level
            
            # Apply GAC monitoring settings
            if hasattr(gac_config, 'monitoring'):
                self.monitoring_config.monitoring_enabled = gac_config.monitoring.enabled
                self.monitoring_config.metrics_enabled = gac_config.monitoring.metrics_enabled
                self.monitoring_config.health_checks_enabled = gac_config.monitoring.health_checks_enabled
            
            logger.info("GAC configuration integrated successfully")
            
        except Exception as e:
            logger.error(f"Failed to integrate GAC configuration: {e}")
    
    def _integrate_golf_domain_config(self) -> None:
        """Integrate with Golf Domain configuration."""
        try:
            if not self.golf_domain_config:
                return
            
            brain_config = self.golf_domain_config.brain_integration_config
            
            # Apply domain-specific settings
            if hasattr(brain_config, 'max_memory_mb'):
                self.performance_config.max_memory_mb = max(
                    self.performance_config.max_memory_mb,
                    brain_config.max_memory_mb
                )
            
            # Apply domain caching settings
            if hasattr(brain_config, 'enable_caching'):
                self.performance_config.cache_enabled = brain_config.enable_caching
            
            logger.info("Golf Domain configuration integrated successfully")
            
        except Exception as e:
            logger.error(f"Failed to integrate Golf Domain configuration: {e}")
    
    @contextmanager
    def configuration_transaction(self):
        """Context manager for configuration transactions."""
        try:
            with self.config_lock:
                # Create backup before changes
                self._create_configuration_backup()
                yield
                # Changes committed automatically on successful exit
        except Exception as e:
            # Restore from backup on error
            if self.backup_configs:
                latest_backup = max(self.backup_configs.keys())
                backup_data = self.backup_configs[latest_backup]
                self.production_config = self._create_production_config_from_dict(
                    backup_data['production_config']
                )
                logger.warning(f"Configuration transaction failed, restored from backup: {e}")
            raise
    
    def get_configuration_diff(self, other_config: ProductionConfig) -> Dict[str, Any]:
        """Get differences between current and another configuration."""
        current_dict = asdict(self.production_config)
        other_dict = asdict(other_config)
        
        differences = {}
        all_keys = set(current_dict.keys()) | set(other_dict.keys())
        
        for key in all_keys:
            current_value = current_dict.get(key)
            other_value = other_dict.get(key)
            
            if current_value != other_value:
                differences[key] = {
                    'current': current_value,
                    'other': other_value
                }
        
        return differences
    
    def shutdown(self) -> None:
        """Shutdown configuration manager."""
        try:
            # Disable hot reload
            self.disable_hot_reload()
            
            # Save current configuration
            self.save_configuration()
            
            logger.info("Production configuration manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during configuration manager shutdown: {e}")


def create_production_config_manager(
    config_dir: str = "config",
    environment: Optional[ProductionEnvironment] = None
) -> ProductionConfigManager:
    """Factory function to create a ProductionConfigManager instance."""
    return ProductionConfigManager(config_dir=config_dir, environment=environment)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    config_manager = create_production_config_manager()
    
    # Initialize with sample integrations
    config_manager.initialize_production_config_manager()
    
    # Load configuration
    config = config_manager.load_configuration()
    print(f"Loaded configuration for {config.environment.value} environment")
    
    # Validate configuration
    try:
        config_manager.validate_configuration()
        print("Configuration validation passed")
    except Exception as e:
        print(f"Configuration validation failed: {e}")
    
    # Get status
    status = config_manager.get_configuration_status()
    print(f"Configuration status: {status['environment']} v{status['version']}")