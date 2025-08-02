"""
Financial Fraud Detection Domain Configuration
Comprehensive configuration definitions and management for the fraud detection domain
with support for all domain components, ML models, security settings, and validation rules.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
from enhanced_ml_framework import ModelType

# Configure logging
logger = logging.getLogger(__name__)

class EnvironmentType(Enum):
    """Environment types for configuration"""
    DEVELOPMENT = "dev"
    TESTING = "test"
    STAGING = "staging"
    PRODUCTION = "prod"

class SecurityLevel(Enum):
    """Security levels for different components"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = "localhost"
    port: int = 5432
    database: str = "fraud_detection"
    username: str = "fraud_user"
    password: str = ""  # Will be encrypted
    ssl_mode: str = "require"
    connection_pool_size: int = 20
    connection_pool_max_overflow: int = 50
    connection_timeout: int = 30
    query_timeout: int = 60
    backup_enabled: bool = True
    backup_schedule: str = "0 2 * * *"  # Daily at 2 AM
    retention_days: int = 30

@dataclass
class RedisConfig:
    """Redis configuration for caching and real-time data"""
    host: str = "localhost"
    port: int = 6379
    password: str = ""  # Will be encrypted
    database: int = 0
    ssl_enabled: bool = False
    connection_pool_size: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    health_check_interval: int = 30
    max_connections: int = 100

@dataclass
class IEEEDatasetConfig:
    """IEEE fraud detection dataset configuration"""
    enabled: bool = True
    data_dir: str = "training_data/ieee-fraud-detection"
    use_cache: bool = True
    cache_dir: str = "training_data/ieee-fraud-detection/processed_cache"
    validation_split: float = 0.2
    missing_value_fill: float = -999
    feature_dtype: str = "float32"
    enable_validation: bool = True
    batch_processing: bool = True
    batch_size: int = 10000
    memory_limit_gb: float = 4.0
    
    # Feature processing options
    categorical_encoding: str = "label"  # label, onehot, target
    feature_selection_enabled: bool = True
    feature_selection_method: str = "variance"  # variance, correlation, mutual_info
    feature_scaling_enabled: bool = True
    feature_scaling_method: str = "standard"  # standard, minmax, robust
    
    # Data quality settings
    max_missing_rate: float = 0.9  # Drop features with >90% missing
    min_variance_threshold: float = 1e-6
    correlation_threshold: float = 0.95  # Drop highly correlated features
    
    # Brain integration settings
    brain_integration_enabled: bool = True
    notify_brain_on_load: bool = True
    send_quality_reports: bool = True

@dataclass
class MLModelConfig:
    """Machine learning model configuration"""
    model_type: ModelType = ModelType.ENSEMBLE
    model_path: str = "./models/fraud_detection"
    threshold: float = 0.85
    confidence_threshold: float = 0.9
    batch_size: int = 1000
    max_features: int = 1000
    training_data_path: str = "./data/training"
    validation_split: float = 0.2
    test_split: float = 0.1
    feature_scaling: bool = True
    feature_selection: bool = True
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    ensemble_weights: Dict[str, float] = field(default_factory=dict)
    retrain_schedule: str = "0 3 * * 0"  # Weekly on Sunday at 3 AM
    auto_retrain: bool = True
    performance_threshold: float = 0.95
    drift_detection_enabled: bool = True
    
    # IEEE dataset integration
    use_ieee_dataset: bool = True
    ieee_config: IEEEDatasetConfig = field(default_factory=IEEEDatasetConfig)
    drift_threshold: float = 0.1

@dataclass
class RealTimeProcessingConfig:
    """Real-time processing configuration"""
    enabled: bool = True
    max_concurrent_streams: int = 100
    buffer_size: int = 10000
    batch_timeout_ms: int = 1000
    processing_timeout_ms: int = 5000
    retry_attempts: int = 3
    backpressure_threshold: int = 8000
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 10
    circuit_breaker_recovery_timeout: int = 30
    stream_partitions: int = 8
    consumer_group: str = "fraud_detection_processors"

@dataclass
class AlertConfig:
    """Alert and notification configuration"""
    enabled: bool = True
    email_alerts: bool = True
    sms_alerts: bool = False
    slack_alerts: bool = True
    webhook_alerts: bool = True
    alert_aggregation_window: int = 300  # 5 minutes
    severity_thresholds: Dict[AlertSeverity, float] = field(default_factory=lambda: {
        AlertSeverity.LOW: 0.7,
        AlertSeverity.MEDIUM: 0.8,
        AlertSeverity.HIGH: 0.9,
        AlertSeverity.CRITICAL: 0.95
    })
    escalation_rules: Dict[str, Any] = field(default_factory=dict)
    notification_templates: Dict[str, str] = field(default_factory=dict)
    rate_limiting: Dict[str, int] = field(default_factory=lambda: {
        "max_alerts_per_minute": 100,
        "max_alerts_per_hour": 1000
    })

@dataclass
class SecurityConfig:
    """Security and compliance configuration"""
    security_level: SecurityLevel = SecurityLevel.HIGH
    encryption_enabled: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_days: int = 90
    audit_logging_enabled: bool = True
    pii_detection_enabled: bool = True
    pii_masking_enabled: bool = True
    data_retention_days: int = 2555  # 7 years for compliance
    access_control_enabled: bool = True
    role_based_access: bool = True
    api_rate_limiting: Dict[str, int] = field(default_factory=lambda: {
        "requests_per_minute": 1000,
        "requests_per_hour": 10000
    })
    allowed_ip_ranges: List[str] = field(default_factory=list)
    blocked_ip_ranges: List[str] = field(default_factory=list)
    session_timeout_minutes: int = 30
    password_policy: Dict[str, Any] = field(default_factory=lambda: {
        "min_length": 12,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_numbers": True,
        "require_special_chars": True,
        "password_history": 5
    })

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enabled: bool = True
    metrics_collection_interval: int = 60  # seconds
    health_check_interval: int = 30
    performance_monitoring: bool = True
    error_tracking: bool = True
    trace_sampling_rate: float = 0.1
    log_level: str = "INFO"
    structured_logging: bool = True
    metric_retention_days: int = 90
    alert_on_high_latency: bool = True
    latency_threshold_ms: int = 1000
    alert_on_high_error_rate: bool = True
    error_rate_threshold: float = 0.05
    custom_metrics: List[str] = field(default_factory=list)
    external_monitoring_endpoints: List[str] = field(default_factory=list)

@dataclass
class DataProcessingConfig:
    """Data processing pipeline configuration"""
    input_validation_enabled: bool = True
    data_cleansing_enabled: bool = True
    feature_engineering_enabled: bool = True
    anomaly_detection_enabled: bool = True
    batch_processing_enabled: bool = True
    stream_processing_enabled: bool = True
    parallel_processing: bool = True
    max_worker_threads: int = 8
    chunk_size: int = 10000
    memory_limit_gb: int = 8
    temp_storage_path: str = "./temp"
    output_format: str = "parquet"
    compression_enabled: bool = True
    compression_algorithm: str = "snappy"
    data_validation_rules: Dict[str, Any] = field(default_factory=dict)

@dataclass
class APIConfig:
    """API configuration settings"""
    host: str = "0.0.0.0"
    port: int = 8080
    debug_mode: bool = False
    ssl_enabled: bool = True
    ssl_cert_path: str = "./certs/server.crt"
    ssl_key_path: str = "./certs/server.key"
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    request_timeout: int = 30
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    rate_limiting_enabled: bool = True
    authentication_required: bool = True
    authorization_enabled: bool = True
    api_versioning: str = "v1"
    swagger_enabled: bool = True
    health_check_endpoint: str = "/health"
    metrics_endpoint: str = "/metrics"

@dataclass
class IntegrationConfig:
    """External system integration configuration"""
    kafka_enabled: bool = True
    kafka_brokers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    kafka_topics: Dict[str, str] = field(default_factory=lambda: {
        "transactions": "fraud.transactions",
        "alerts": "fraud.alerts",
        "models": "fraud.models"
    })
    elasticsearch_enabled: bool = True
    elasticsearch_hosts: List[str] = field(default_factory=lambda: ["localhost:9200"])
    elasticsearch_indices: Dict[str, str] = field(default_factory=lambda: {
        "transactions": "fraud-transactions",
        "alerts": "fraud-alerts",
        "audit": "fraud-audit"
    })
    external_apis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    webhook_endpoints: List[str] = field(default_factory=list)
    notification_services: Dict[str, Dict[str, str]] = field(default_factory=dict)

@dataclass
class DomainConfig:
    """Core domain configuration structure"""
    # Basic domain settings
    enabled: bool = True
    auto_start: bool = True
    environment: EnvironmentType = EnvironmentType.DEVELOPMENT
    debug_mode: bool = False
    
    # Performance settings
    max_concurrent_tasks: int = 100
    task_timeout: int = 300
    retry_attempts: int = 3
    batch_size: int = 1000
    
    # Fraud-specific settings
    model_threshold: float = 0.85
    real_time_processing: bool = True
    alert_threshold: float = 0.9
    auto_block_threshold: float = 0.95
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    ml_model: MLModelConfig = field(default_factory=MLModelConfig)
    real_time: RealTimeProcessingConfig = field(default_factory=RealTimeProcessingConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    data_processing: DataProcessingConfig = field(default_factory=DataProcessingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    integrations: IntegrationConfig = field(default_factory=IntegrationConfig)
    
    # IEEE dataset configuration
    ieee_dataset: IEEEDatasetConfig = field(default_factory=IEEEDatasetConfig)
    
    # Additional configuration
    metadata: Dict[str, Any] = field(default_factory=dict)
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    feature_flags: Dict[str, bool] = field(default_factory=lambda: {
        'enable_ieee_dataset': True,
        'enable_brain_integration': True,
        'enable_gac_system': True,
        'enable_advanced_preprocessing': True,
        'enable_real_time_monitoring': True
    })

class FinancialFraudConfig:
    """
    Financial fraud detection domain configuration manager
    
    Provides comprehensive configuration management for all aspects of the fraud detection domain
    including ML models, data processing, security, monitoring, and integrations.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 environment: Optional[str] = None,
                 auto_load: bool = True):
        """
        Initialize fraud detection configuration
        
        Args:
            config_path: Path to configuration file
            environment: Target environment (dev/test/staging/prod)
            auto_load: Whether to automatically load configuration on initialization
        """
        self.config_path = Path(config_path) if config_path else Path("./config/fraud_detection.json")
        self.environment = EnvironmentType(environment) if environment else EnvironmentType.DEVELOPMENT
        self.config = DomainConfig()
        self.config.environment = self.environment
        
        # Configuration validation schemas
        self._schemas = self._initialize_schemas()
        
        # Load configuration if requested
        if auto_load:
            try:
                self.load_config()
            except FileNotFoundError:
                logger.warning(f"Configuration file not found: {self.config_path}")
                logger.info("Using default configuration")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                logger.info("Using default configuration")
        
        # Apply environment-specific overrides
        self._apply_environment_overrides()
        
        logger.info(f"FinancialFraudConfig initialized for environment: {self.environment.value}")
    
    def _initialize_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Initialize validation schemas for configuration components"""
        return {
            "database": {
                "type": "object",
                "properties": {
                    "host": {"type": "string"},
                    "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                    "database": {"type": "string"},
                    "connection_pool_size": {"type": "integer", "minimum": 1},
                    "connection_timeout": {"type": "integer", "minimum": 1}
                },
                "required": ["host", "port", "database"]
            },
            "ml_model": {
                "type": "object",
                "properties": {
                    "threshold": {"type": "number", "minimum": 0, "maximum": 1},
                    "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                    "batch_size": {"type": "integer", "minimum": 1},
                    "validation_split": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["threshold", "batch_size"]
            },
            "security": {
                "type": "object",
                "properties": {
                    "encryption_enabled": {"type": "boolean"},
                    "audit_logging_enabled": {"type": "boolean"},
                    "data_retention_days": {"type": "integer", "minimum": 1},
                    "session_timeout_minutes": {"type": "integer", "minimum": 1}
                },
                "required": ["encryption_enabled", "audit_logging_enabled"]
            },
            "api": {
                "type": "object",
                "properties": {
                    "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                    "ssl_enabled": {"type": "boolean"},
                    "request_timeout": {"type": "integer", "minimum": 1},
                    "max_request_size": {"type": "integer", "minimum": 1}
                },
                "required": ["port"]
            }
        }
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides"""
        if self.environment == EnvironmentType.DEVELOPMENT:
            self.config.debug_mode = True
            self.config.api.debug_mode = True
            self.config.security.security_level = SecurityLevel.MEDIUM
            self.config.monitoring.log_level = "DEBUG"
            self.config.ml_model.auto_retrain = False
            
        elif self.environment == EnvironmentType.TESTING:
            self.config.debug_mode = False
            self.config.api.debug_mode = False
            self.config.security.security_level = SecurityLevel.MEDIUM
            self.config.monitoring.log_level = "INFO"
            self.config.database.database = "fraud_detection_test"
            self.config.redis.database = 1
            
        elif self.environment == EnvironmentType.STAGING:
            self.config.debug_mode = False
            self.config.api.debug_mode = False
            self.config.security.security_level = SecurityLevel.HIGH
            self.config.monitoring.log_level = "INFO"
            self.config.api.ssl_enabled = True
            self.config.security.encryption_enabled = True
            
        elif self.environment == EnvironmentType.PRODUCTION:
            self.config.debug_mode = False
            self.config.api.debug_mode = False
            self.config.security.security_level = SecurityLevel.CRITICAL
            self.config.monitoring.log_level = "WARNING"
            self.config.api.ssl_enabled = True
            self.config.security.encryption_enabled = True
            self.config.security.audit_logging_enabled = True
            self.config.monitoring.trace_sampling_rate = 0.01  # Lower sampling in prod
    
    def load_config(self) -> bool:
        """
        Load configuration from file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Convert dictionaries to dataclass instances
            self._load_from_dict(config_data)
            
            logger.info(f"Configuration loaded from: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def save_config(self, create_backup: bool = True) -> bool:
        """
        Save configuration to file
        
        Args:
            create_backup: Whether to create a backup of existing config
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create backup if requested
            if create_backup and self.config_path.exists():
                self._create_backup()
            
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dictionary and save
            config_dict = self._to_dict()
            
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """
        Validate current configuration
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Basic validation
            if self.config.model_threshold < 0 or self.config.model_threshold > 1:
                errors.append("model_threshold must be between 0 and 1")
            
            if self.config.alert_threshold < 0 or self.config.alert_threshold > 1:
                errors.append("alert_threshold must be between 0 and 1")
            
            if self.config.auto_block_threshold < 0 or self.config.auto_block_threshold > 1:
                errors.append("auto_block_threshold must be between 0 and 1")
            
            if self.config.task_timeout <= 0:
                errors.append("task_timeout must be positive")
            
            if self.config.batch_size <= 0:
                errors.append("batch_size must be positive")
            
            # Validate component configurations
            errors.extend(self._validate_database_config())
            errors.extend(self._validate_ml_config())
            errors.extend(self._validate_security_config())
            errors.extend(self._validate_api_config())
            errors.extend(self._validate_ieee_config())
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
            return False, errors
    
    def _validate_database_config(self) -> List[str]:
        """Validate database configuration"""
        errors = []
        db_config = self.config.database
        
        if not db_config.host:
            errors.append("Database host is required")
        
        if db_config.port <= 0 or db_config.port > 65535:
            errors.append("Database port must be between 1 and 65535")
        
        if not db_config.database:
            errors.append("Database name is required")
        
        if db_config.connection_pool_size <= 0:
            errors.append("Database connection pool size must be positive")
        
        return errors
    
    def _validate_ml_config(self) -> List[str]:
        """Validate ML model configuration"""
        errors = []
        ml_config = self.config.ml_model
        
        if ml_config.threshold < 0 or ml_config.threshold > 1:
            errors.append("ML model threshold must be between 0 and 1")
        
        if ml_config.confidence_threshold < 0 or ml_config.confidence_threshold > 1:
            errors.append("ML model confidence threshold must be between 0 and 1")
        
        if ml_config.batch_size <= 0:
            errors.append("ML model batch size must be positive")
        
        if ml_config.validation_split < 0 or ml_config.validation_split > 1:
            errors.append("ML model validation split must be between 0 and 1")
        
        return errors
    
    def _validate_security_config(self) -> List[str]:
        """Validate security configuration"""
        errors = []
        sec_config = self.config.security
        
        if sec_config.data_retention_days <= 0:
            errors.append("Data retention days must be positive")
        
        if sec_config.session_timeout_minutes <= 0:
            errors.append("Session timeout must be positive")
        
        if sec_config.key_rotation_days <= 0:
            errors.append("Key rotation days must be positive")
        
        return errors
    
    def _validate_api_config(self) -> List[str]:
        """Validate API configuration"""
        errors = []
        api_config = self.config.api
        
        if api_config.port <= 0 or api_config.port > 65535:
            errors.append("API port must be between 1 and 65535")
        
        if api_config.request_timeout <= 0:
            errors.append("API request timeout must be positive")
        
        if api_config.max_request_size <= 0:
            errors.append("API max request size must be positive")
        
        return errors
    
    def _validate_ieee_config(self) -> List[str]:
        """Validate IEEE dataset configuration"""
        errors = []
        ieee_config = self.config.ieee_dataset
        
        if ieee_config.validation_split < 0 or ieee_config.validation_split > 1:
            errors.append("IEEE validation split must be between 0 and 1")
        
        if ieee_config.batch_size <= 0:
            errors.append("IEEE batch size must be positive")
        
        if ieee_config.memory_limit_gb <= 0:
            errors.append("IEEE memory limit must be positive")
        
        if ieee_config.max_missing_rate < 0 or ieee_config.max_missing_rate > 1:
            errors.append("IEEE max missing rate must be between 0 and 1")
        
        if ieee_config.correlation_threshold < 0 or ieee_config.correlation_threshold > 1:
            errors.append("IEEE correlation threshold must be between 0 and 1")
        
        # Validate categorical encoding method
        valid_encodings = ["label", "onehot", "target"]
        if ieee_config.categorical_encoding not in valid_encodings:
            errors.append(f"IEEE categorical encoding must be one of: {valid_encodings}")
        
        # Validate feature selection method
        valid_selection_methods = ["variance", "correlation", "mutual_info"]
        if ieee_config.feature_selection_method not in valid_selection_methods:
            errors.append(f"IEEE feature selection method must be one of: {valid_selection_methods}")
        
        # Validate feature scaling method
        valid_scaling_methods = ["standard", "minmax", "robust"]
        if ieee_config.feature_scaling_method not in valid_scaling_methods:
            errors.append(f"IEEE feature scaling method must be one of: {valid_scaling_methods}")
        
        return errors
    
    def get_config(self) -> DomainConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary of configuration updates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Apply updates using deep merge
            self._deep_update_config(self.config, updates)
            
            # Validate updated configuration
            is_valid, errors = self.validate_config()
            if not is_valid:
                logger.error(f"Configuration validation failed: {errors}")
                return False
            
            logger.info("Configuration updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        return {
            "environment": self.environment.value,
            "debug_mode": self.config.debug_mode,
            "security_level": self.config.security.security_level.value,
            "log_level": self.config.monitoring.log_level
        }
    
    def create_config_template(self, template_name: str = "default") -> Dict[str, Any]:
        """
        Create configuration template
        
        Args:
            template_name: Name of the template
            
        Returns:
            Configuration template dictionary
        """
        template = self._to_dict()
        template["_template_info"] = {
            "name": template_name,
            "created_at": datetime.now().isoformat(),
            "version": "1.0"
        }
        return template
    
    def load_from_template(self, template_data: Dict[str, Any]) -> bool:
        """
        Load configuration from template
        
        Args:
            template_data: Template data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove template metadata if present
            if "_template_info" in template_data:
                del template_data["_template_info"]
            
            self._load_from_dict(template_data)
            
            # Validate loaded configuration
            is_valid, errors = self.validate_config()
            if not is_valid:
                logger.error(f"Template validation failed: {errors}")
                return False
            
            logger.info("Configuration loaded from template")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load from template: {e}")
            return False
    
    def _load_from_dict(self, config_data: Dict[str, Any]):
        """Load configuration from dictionary"""
        # Load basic settings
        if "enabled" in config_data:
            self.config.enabled = config_data["enabled"]
        if "auto_start" in config_data:
            self.config.auto_start = config_data["auto_start"]
        if "debug_mode" in config_data:
            self.config.debug_mode = config_data["debug_mode"]
        if "model_threshold" in config_data:
            self.config.model_threshold = config_data["model_threshold"]
        if "alert_threshold" in config_data:
            self.config.alert_threshold = config_data["alert_threshold"]
        
        # Load component configurations
        if "database" in config_data:
            self._load_database_config(config_data["database"])
        if "redis" in config_data:
            self._load_redis_config(config_data["redis"])
        if "ml_model" in config_data:
            self._load_ml_config(config_data["ml_model"])
        if "security" in config_data:
            self._load_security_config(config_data["security"])
        if "api" in config_data:
            self._load_api_config(config_data["api"])
        if "ieee_dataset" in config_data:
            self._load_ieee_config(config_data["ieee_dataset"])
        
        # Load additional settings
        if "metadata" in config_data:
            self.config.metadata = config_data["metadata"]
        if "custom_settings" in config_data:
            self.config.custom_settings = config_data["custom_settings"]
        if "feature_flags" in config_data:
            self.config.feature_flags = config_data["feature_flags"]
    
    def _load_database_config(self, data: Dict[str, Any]):
        """Load database configuration from dictionary"""
        for key, value in data.items():
            if hasattr(self.config.database, key):
                setattr(self.config.database, key, value)
    
    def _load_redis_config(self, data: Dict[str, Any]):
        """Load Redis configuration from dictionary"""
        for key, value in data.items():
            if hasattr(self.config.redis, key):
                setattr(self.config.redis, key, value)
    
    def _load_ml_config(self, data: Dict[str, Any]):
        """Load ML model configuration from dictionary"""
        for key, value in data.items():
            if hasattr(self.config.ml_model, key):
                if key == "model_type" and isinstance(value, str):
                    setattr(self.config.ml_model, key, ModelType(value))
                else:
                    setattr(self.config.ml_model, key, value)
    
    def _load_security_config(self, data: Dict[str, Any]):
        """Load security configuration from dictionary"""
        for key, value in data.items():
            if hasattr(self.config.security, key):
                if key == "security_level" and isinstance(value, str):
                    setattr(self.config.security, key, SecurityLevel(value))
                else:
                    setattr(self.config.security, key, value)
    
    def _load_api_config(self, data: Dict[str, Any]):
        """Load API configuration from dictionary"""
        for key, value in data.items():
            if hasattr(self.config.api, key):
                setattr(self.config.api, key, value)
    
    def _load_ieee_config(self, data: Dict[str, Any]):
        """Load IEEE dataset configuration from dictionary"""
        for key, value in data.items():
            if hasattr(self.config.ieee_dataset, key):
                setattr(self.config.ieee_dataset, key, value)
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self.config)
    
    def _deep_update_config(self, config_obj: Any, updates: Dict[str, Any]):
        """Deep update configuration object with new values"""
        for key, value in updates.items():
            if hasattr(config_obj, key):
                current_value = getattr(config_obj, key)
                if isinstance(current_value, dict) and isinstance(value, dict):
                    current_value.update(value)
                elif hasattr(current_value, '__dict__') and isinstance(value, dict):
                    self._deep_update_config(current_value, value)
                else:
                    setattr(config_obj, key, value)
    
    def _create_backup(self):
        """Create backup of current configuration"""
        backup_dir = self.config_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"fraud_detection_{timestamp}.json"
        
        with open(backup_path, 'w') as f:
            json.dump(self._to_dict(), f, indent=2, default=str)
        
        logger.debug(f"Configuration backup created: {backup_path}")

if __name__ == "__main__":
    # Example usage
    fraud_config = FinancialFraudConfig(
        config_path="./config/fraud_detection.json",
        environment="dev",
        auto_load=False
    )
    
    # Customize configuration
    fraud_config.config.model_threshold = 0.9
    fraud_config.config.ml_model.batch_size = 2000
    fraud_config.config.alerts.enabled = True
    
    # Validate configuration
    is_valid, errors = fraud_config.validate_config()
    if is_valid:
        print("Configuration is valid")
        
        # Save configuration
        fraud_config.save_config()
        print("Configuration saved successfully")
    else:
        print(f"Configuration validation failed: {errors}")
    
    # Create template
    template = fraud_config.create_config_template("production_template")
    print("Configuration template created")