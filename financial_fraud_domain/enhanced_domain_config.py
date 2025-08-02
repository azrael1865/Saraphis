"""
Enhanced Domain Configuration with comprehensive validation and error handling
"""

import os
import re
import ipaddress
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path


class EnvironmentType(Enum):
    """Environment types for configuration"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class SecurityLevel(Enum):
    """Security levels for configuration"""
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationMixin:
    """Base validation functionality for configuration classes"""
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration and return (is_valid, errors)"""
        errors = []
        
        # Run all validation methods
        for method_name in dir(self):
            if method_name.startswith('_validate_') and callable(getattr(self, method_name)):
                try:
                    method_errors = getattr(self, method_name)()
                    if method_errors:
                        errors.extend(method_errors)
                except Exception as e:
                    errors.append(f"Validation error in {method_name}: {str(e)}")
        
        return len(errors) == 0, errors
    
    def _validate_required_fields(self) -> List[str]:
        """Validate that required fields are present and not empty"""
        errors = []
        required_fields = getattr(self, '_required_fields', [])
        
        for field_name in required_fields:
            if hasattr(self, field_name):
                value = getattr(self, field_name)
                if value is None or (isinstance(value, str) and not value.strip()):
                    errors.append(f"Required field '{field_name}' is missing or empty")
            else:
                errors.append(f"Required field '{field_name}' is not defined")
        
        return errors


class SecurityValidationMixin:
    """Security-specific validation functionality"""
    
    def _validate_security_requirements(self) -> List[str]:
        """Validate security-related requirements"""
        errors = []
        
        # Check for hardcoded credentials
        errors.extend(self._check_hardcoded_credentials())
        
        # Check for weak passwords
        errors.extend(self._check_password_strength())
        
        # Check for insecure configurations
        errors.extend(self._check_insecure_configs())
        
        return errors
    
    def _check_hardcoded_credentials(self) -> List[str]:
        """Check for hardcoded credentials in configuration"""
        errors = []
        dangerous_patterns = [
            r'password.*=.*["\'].*["\']',
            r'secret.*=.*["\'].*["\']',
            r'key.*=.*["\'].*["\']',
            r'token.*=.*["\'].*["\']'
        ]
        
        config_str = str(self.__dict__)
        for pattern in dangerous_patterns:
            if re.search(pattern, config_str, re.IGNORECASE):
                errors.append("Potential hardcoded credentials detected")
                break
        
        return errors
    
    def _check_password_strength(self) -> List[str]:
        """Check password strength requirements"""
        errors = []
        
        for attr_name in dir(self):
            if 'password' in attr_name.lower():
                password = getattr(self, attr_name, None)
                if isinstance(password, str):
                    if len(password) < 8:
                        errors.append(f"Password '{attr_name}' is too short (minimum 8 characters)")
                    if not re.search(r'[A-Z]', password):
                        errors.append(f"Password '{attr_name}' must contain uppercase letters")
                    if not re.search(r'[a-z]', password):
                        errors.append(f"Password '{attr_name}' must contain lowercase letters")
                    if not re.search(r'\d', password):
                        errors.append(f"Password '{attr_name}' must contain digits")
        
        return errors
    
    def _check_insecure_configs(self) -> List[str]:
        """Check for insecure configuration settings"""
        errors = []
        
        # Check for debug mode in production
        if hasattr(self, 'environment') and self.environment == EnvironmentType.PRODUCTION:
            if hasattr(self, 'debug_mode') and getattr(self, 'debug_mode', False):
                errors.append("Debug mode should be disabled in production")
        
        return errors


@dataclass
class DatabaseConfig(ValidationMixin):
    """Database configuration with validation"""
    host: str = "localhost"
    port: int = 5432
    database: str = "fraud_detection"
    username: str = "fraud_user"
    password: str = ""
    connection_pool_size: int = 10
    connection_pool_max_overflow: int = 20
    connection_timeout: int = 30
    query_timeout: int = 300
    ssl_enabled: bool = False
    ssl_cert_path: str = ""
    backup_enabled: bool = True
    backup_schedule: str = "0 2 * * *"  # Daily at 2 AM
    
    _required_fields = ['host', 'port', 'database', 'username']
    
    def _validate_port(self) -> List[str]:
        """Validate database port"""
        errors = []
        if not (1 <= self.port <= 65535):
            errors.append(f"Database port must be between 1-65535, got {self.port}")
        return errors
    
    def _validate_connection_pool(self) -> List[str]:
        """Validate connection pool settings"""
        errors = []
        if self.connection_pool_size <= 0:
            errors.append("Connection pool size must be positive")
        if self.connection_pool_max_overflow < 0:
            errors.append("Connection pool max overflow must be non-negative")
        if self.connection_pool_size + self.connection_pool_max_overflow > 1000:
            errors.append("Total connection pool size too large (>1000), may cause performance issues")
        return errors
    
    def _validate_timeouts(self) -> List[str]:
        """Validate timeout settings"""
        errors = []
        if self.connection_timeout <= 0:
            errors.append("Connection timeout must be positive")
        if self.query_timeout <= 0:
            errors.append("Query timeout must be positive")
        if self.query_timeout > 3600:
            errors.append("Query timeout >1 hour may cause performance issues")
        return errors
    
    def _validate_ssl_config(self) -> List[str]:
        """Validate SSL configuration"""
        errors = []
        if self.ssl_enabled and not self.ssl_cert_path:
            errors.append("SSL certificate path required when SSL is enabled")
        if self.ssl_cert_path and not Path(self.ssl_cert_path).exists():
            errors.append(f"SSL certificate file not found: {self.ssl_cert_path}")
        return errors
    
    def _validate_backup_schedule(self) -> List[str]:
        """Validate backup schedule (cron format)"""
        errors = []
        if self.backup_enabled:
            # Basic cron validation
            parts = self.backup_schedule.split()
            if len(parts) != 5:
                errors.append("Backup schedule must be in cron format (5 fields)")
        return errors


@dataclass
class RedisConfig(ValidationMixin):
    """Redis configuration with validation"""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: str = ""
    connection_pool_size: int = 50
    connection_timeout: int = 10
    socket_timeout: int = 10
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    
    def _validate_port(self) -> List[str]:
        """Validate Redis port"""
        errors = []
        if not (1 <= self.port <= 65535):
            errors.append(f"Redis port must be between 1-65535, got {self.port}")
        return errors
    
    def _validate_database(self) -> List[str]:
        """Validate Redis database number"""
        errors = []
        if not (0 <= self.database <= 15):
            errors.append(f"Redis database must be between 0-15, got {self.database}")
        return errors
    
    def _validate_connection_settings(self) -> List[str]:
        """Validate connection settings"""
        errors = []
        if self.connection_pool_size <= 0:
            errors.append("Redis connection pool size must be positive")
        if self.connection_timeout <= 0:
            errors.append("Redis connection timeout must be positive")
        if self.socket_timeout <= 0:
            errors.append("Redis socket timeout must be positive")
        return errors


@dataclass
class MLModelConfig(ValidationMixin):
    """ML Model configuration with validation"""
    model_path: str = ""
    model_threshold: float = 0.5
    batch_size: int = 32
    max_features: int = 10000
    auto_retrain: bool = False
    retrain_interval_hours: int = 24
    validation_split: float = 0.2
    test_split: float = 0.1
    ensemble_enabled: bool = False
    ensemble_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])
    feature_importance_threshold: float = 0.01
    
    def _validate_threshold(self) -> List[str]:
        """Validate model threshold"""
        errors = []
        if not (0 <= self.model_threshold <= 1):
            errors.append(f"Model threshold must be between 0-1, got {self.model_threshold}")
        return errors
    
    def _validate_batch_size(self) -> List[str]:
        """Validate batch size"""
        errors = []
        if self.batch_size <= 0:
            errors.append("Batch size must be positive")
        if self.batch_size > 10000:
            errors.append("Batch size >10000 may cause memory issues")
        return errors
    
    def _validate_data_splits(self) -> List[str]:
        """Validate data split ratios"""
        errors = []
        if not (0 < self.validation_split < 1):
            errors.append(f"Validation split must be between 0-1, got {self.validation_split}")
        if not (0 < self.test_split < 1):
            errors.append(f"Test split must be between 0-1, got {self.test_split}")
        if self.validation_split + self.test_split >= 1:
            errors.append("Validation + test splits must be < 1")
        return errors
    
    def _validate_ensemble_config(self) -> List[str]:
        """Validate ensemble configuration"""
        errors = []
        if self.ensemble_enabled:
            if not self.ensemble_weights:
                errors.append("Ensemble weights required when ensemble is enabled")
            elif abs(sum(self.ensemble_weights) - 1.0) > 0.01:
                errors.append("Ensemble weights must sum to 1.0")
        return errors
    
    def _validate_model_path(self) -> List[str]:
        """Validate model path"""
        errors = []
        if self.auto_retrain and not self.model_path:
            errors.append("Model path required when auto-retrain is enabled")
        return errors


@dataclass
class SecurityConfig(ValidationMixin, SecurityValidationMixin):
    """Security configuration with validation"""
    encryption_enabled: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_days: int = 90
    security_level: SecurityLevel = SecurityLevel.STANDARD
    data_retention_days: int = 365
    audit_logging_enabled: bool = True
    failed_login_threshold: int = 5
    session_timeout_minutes: int = 30
    allowed_ip_ranges: List[str] = field(default_factory=list)
    blocked_ip_ranges: List[str] = field(default_factory=list)
    rate_limiting_enabled: bool = True
    rate_limit_requests_per_minute: int = 100
    
    def _validate_encryption_settings(self) -> List[str]:
        """Validate encryption settings"""
        errors = []
        valid_algorithms = ["AES-256-GCM", "AES-256-CBC", "ChaCha20-Poly1305"]
        if self.encryption_algorithm not in valid_algorithms:
            errors.append(f"Invalid encryption algorithm: {self.encryption_algorithm}")
        
        if self.key_rotation_days <= 0:
            errors.append("Key rotation period must be positive")
        elif self.key_rotation_days > 365:
            errors.append("Key rotation period >365 days may be insecure")
        
        return errors
    
    def _validate_data_retention(self) -> List[str]:
        """Validate data retention settings"""
        errors = []
        if self.data_retention_days <= 0:
            errors.append("Data retention period must be positive")
        
        # Security level requirements
        if self.security_level == SecurityLevel.HIGH and self.data_retention_days < 90:
            errors.append("HIGH security level requires minimum 90 days data retention")
        elif self.security_level == SecurityLevel.CRITICAL and self.data_retention_days < 180:
            errors.append("CRITICAL security level requires minimum 180 days data retention")
        
        return errors
    
    def _validate_session_settings(self) -> List[str]:
        """Validate session settings"""
        errors = []
        if self.session_timeout_minutes <= 0:
            errors.append("Session timeout must be positive")
        elif self.session_timeout_minutes > 480:  # 8 hours
            errors.append("Session timeout >8 hours may be insecure")
        
        if self.failed_login_threshold <= 0:
            errors.append("Failed login threshold must be positive")
        
        return errors
    
    def _validate_ip_ranges(self) -> List[str]:
        """Validate IP address ranges"""
        errors = []
        
        # Validate allowed IP ranges
        for ip_range in self.allowed_ip_ranges:
            try:
                ipaddress.ip_network(ip_range, strict=False)
            except ValueError:
                errors.append(f"Invalid IP range in allowed list: {ip_range}")
        
        # Validate blocked IP ranges
        for ip_range in self.blocked_ip_ranges:
            try:
                ipaddress.ip_network(ip_range, strict=False)
            except ValueError:
                errors.append(f"Invalid IP range in blocked list: {ip_range}")
        
        # Check for conflicts between allowed and blocked ranges
        for allowed in self.allowed_ip_ranges:
            for blocked in self.blocked_ip_ranges:
                try:
                    allowed_net = ipaddress.ip_network(allowed, strict=False)
                    blocked_net = ipaddress.ip_network(blocked, strict=False)
                    if allowed_net.overlaps(blocked_net):
                        errors.append(f"IP range conflict: {allowed} overlaps with blocked {blocked}")
                except ValueError:
                    continue  # Already caught above
        
        return errors
    
    def _validate_rate_limiting(self) -> List[str]:
        """Validate rate limiting settings"""
        errors = []
        if self.rate_limiting_enabled and self.rate_limit_requests_per_minute <= 0:
            errors.append("Rate limit must be positive when enabled")
        return errors


@dataclass
class APIConfig(ValidationMixin):
    """API configuration with validation"""
    host: str = "127.0.0.1"
    port: int = 8000
    debug_mode: bool = False
    ssl_enabled: bool = False
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    request_timeout: int = 30
    worker_processes: int = 1
    
    def _validate_port(self) -> List[str]:
        """Validate API port"""
        errors = []
        if not (1 <= self.port <= 65535):
            errors.append(f"API port must be between 1-65535, got {self.port}")
        return errors
    
    def _validate_ssl_config(self) -> List[str]:
        """Validate SSL configuration"""
        errors = []
        if self.ssl_enabled:
            if not self.ssl_cert_path:
                errors.append("SSL certificate path required when SSL is enabled")
            if not self.ssl_key_path:
                errors.append("SSL key path required when SSL is enabled")
            
            if self.ssl_cert_path and not Path(self.ssl_cert_path).exists():
                errors.append(f"SSL certificate file not found: {self.ssl_cert_path}")
            if self.ssl_key_path and not Path(self.ssl_key_path).exists():
                errors.append(f"SSL key file not found: {self.ssl_key_path}")
        
        return errors
    
    def _validate_cors_config(self) -> List[str]:
        """Validate CORS configuration"""
        errors = []
        if self.cors_enabled and "*" in self.cors_origins and len(self.cors_origins) > 1:
            errors.append("CORS origins: wildcard '*' should be the only origin when used")
        return errors
    
    def _validate_request_settings(self) -> List[str]:
        """Validate request settings"""
        errors = []
        if self.max_request_size <= 0:
            errors.append("Max request size must be positive")
        elif self.max_request_size > 1024 * 1024 * 1024:  # 1GB
            errors.append("Max request size >1GB may cause performance issues")
        
        if self.request_timeout <= 0:
            errors.append("Request timeout must be positive")
        elif self.request_timeout > 300:  # 5 minutes
            errors.append("Request timeout >5 minutes may cause performance issues")
        
        return errors
    
    def _validate_worker_config(self) -> List[str]:
        """Validate worker configuration"""
        errors = []
        if self.worker_processes <= 0:
            errors.append("Worker processes must be positive")
        elif self.worker_processes > 16:
            errors.append("Worker processes >16 may cause resource contention")
        return errors


@dataclass
class DomainConfig(ValidationMixin, SecurityValidationMixin):
    """Main domain configuration with comprehensive validation"""
    
    # Environment settings
    environment: EnvironmentType = EnvironmentType.DEVELOPMENT
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    ml_model: MLModelConfig = field(default_factory=MLModelConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    # Domain-specific settings
    model_threshold: float = 0.5
    alert_threshold: float = 0.8
    real_time_processing: bool = False
    batch_processing_interval: int = 300  # 5 minutes
    
    def _validate_thresholds(self) -> List[str]:
        """Validate threshold values"""
        errors = []
        if not (0 <= self.model_threshold <= 1):
            errors.append(f"Model threshold must be between 0-1, got {self.model_threshold}")
        if not (0 <= self.alert_threshold <= 1):
            errors.append(f"Alert threshold must be between 0-1, got {self.alert_threshold}")
        if self.alert_threshold <= self.model_threshold:
            errors.append("Alert threshold should be higher than model threshold")
        return errors
    
    def _validate_processing_config(self) -> List[str]:
        """Validate processing configuration"""
        errors = []
        if self.real_time_processing and not self.redis.host:
            errors.append("Real-time processing requires Redis configuration")
        
        if self.batch_processing_interval <= 0:
            errors.append("Batch processing interval must be positive")
        
        return errors
    
    def _validate_environment_requirements(self) -> List[str]:
        """Validate environment-specific requirements"""
        errors = []
        
        if self.environment == EnvironmentType.PRODUCTION:
            # Production requirements
            if self.api.debug_mode:
                errors.append("Debug mode must be disabled in production")
            if not self.api.ssl_enabled:
                errors.append("SSL must be enabled in production")
            if not self.security.encryption_enabled:
                errors.append("Encryption must be enabled in production")
            if self.security.security_level == SecurityLevel.LOW:
                errors.append("Production requires HIGH or CRITICAL security level")
        
        elif self.environment == EnvironmentType.DEVELOPMENT:
            # Development warnings (not errors)
            pass
        
        return errors
    
    def _validate_component_compatibility(self) -> List[str]:
        """Validate compatibility between components"""
        errors = []
        
        # Check ML model and database compatibility
        if self.ml_model.auto_retrain and not self.database.backup_enabled:
            errors.append("Auto-retrain requires database backups to be enabled")
        
        # Check security and API compatibility
        if self.security.security_level == SecurityLevel.CRITICAL and not self.api.ssl_enabled:
            errors.append("CRITICAL security level requires SSL to be enabled")
        
        return errors


class FinancialFraudConfig:
    """Financial fraud domain configuration manager"""
    
    def __init__(self, 
                 config_path: str = "config/fraud_config.json",
                 environment: str = "development",
                 strict_mode: bool = True):
        
        self.config_path = Path(config_path)
        self.environment = EnvironmentType(environment)
        self.strict_mode = strict_mode
        
        # Load configuration
        self.config = self._load_or_create_config()
    
    def _load_or_create_config(self) -> DomainConfig:
        """Load existing configuration or create default"""
        if self.config_path.exists():
            try:
                import json
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Create config object from data
                config = self._dict_to_config(config_data)
                
                # Validate if in strict mode
                if self.strict_mode:
                    is_valid, errors = config.validate()
                    if not is_valid:
                        raise ValueError(f"Configuration validation failed: {errors}")
                
                return config
                
            except Exception as e:
                if self.strict_mode:
                    raise e
                # Fall back to default configuration
                return DomainConfig(environment=self.environment)
        else:
            # Create default configuration
            return DomainConfig(environment=self.environment)
    
    def _dict_to_config(self, data: Dict[str, Any]) -> DomainConfig:
        """Convert dictionary to configuration object"""
        # This is a simplified conversion - in practice, you'd want more robust handling
        config = DomainConfig(environment=self.environment)
        
        # Update fields from data
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def save(self) -> bool:
        """Save current configuration to file"""
        try:
            import json
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert config to dictionary
            config_dict = self._config_to_dict(self.config)
            
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            return True
        except Exception:
            return False
    
    def _config_to_dict(self, config: DomainConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary"""
        # Simplified conversion
        return {
            'environment': config.environment.value,
            'model_threshold': config.model_threshold,
            'alert_threshold': config.alert_threshold,
            'real_time_processing': config.real_time_processing,
            'batch_processing_interval': config.batch_processing_interval
        }
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate current configuration"""
        return self.config.validate()
    
    def get_config(self) -> DomainConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values"""
        try:
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # Validate if in strict mode
            if self.strict_mode:
                is_valid, errors = self.config.validate()
                if not is_valid:
                    raise ValueError(f"Configuration validation failed: {errors}")
            
            return True
        except Exception:
            return False