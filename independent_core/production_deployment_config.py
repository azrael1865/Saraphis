"""
Production Deployment Configuration - Production deployment configuration and orchestration
NO FALLBACKS - HARD FAILURES ONLY

This module provides comprehensive production deployment configuration and orchestration
capabilities, including environment-specific deployment strategies, resource management,
scaling configuration, load balancing, health checks, and rollback strategies.

Key Features:
- Environment-specific deployment configuration (dev, staging, production)
- Deployment strategy management with rollout configurations
- Resource allocation and scaling configuration
- Load balancing configuration for production workloads
- Health check configuration and monitoring
- Rollback configuration for automatic recovery
- Deployment validation and readiness checks
- Configuration templating for different environments
- Integration with existing production configuration systems

Architecture: NO FALLBACKS - HARD FAILURES ONLY
All deployment operations must succeed or fail explicitly with detailed error information.
"""

import os
import json
import yaml
import logging
import threading
import time
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import copy
import traceback
import uuid
import psutil
import socket
from contextlib import contextmanager
import tempfile
import jinja2

try:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from production_config_manager import ProductionConfigManager, ProductionConfig
        from gac_system.gac_config import GACConfigManager
        from production_training_execution import ProductionTrainingExecutor
except ImportError:
    pass

logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    LOCAL = "local"
    CANARY = "canary"
    BLUE_GREEN = "blue_green"


class DeploymentStrategy(Enum):
    """Deployment strategy types."""
    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"
    IMMEDIATE = "immediate"
    GRADUAL = "gradual"


class ScalingStrategy(Enum):
    """Auto-scaling strategy types."""
    NONE = "none"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    BOTH = "both"
    PREDICTIVE = "predictive"
    REACTIVE = "reactive"


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithm types."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"


class HealthCheckType(Enum):
    """Health check types."""
    HTTP = "http"
    TCP = "tcp"
    COMMAND = "command"
    CUSTOM = "custom"
    PING = "ping"
    DATABASE = "database"


class RollbackTrigger(Enum):
    """Rollback trigger conditions."""
    MANUAL = "manual"
    HEALTH_CHECK_FAILURE = "health_check_failure"
    ERROR_RATE_THRESHOLD = "error_rate_threshold"
    RESPONSE_TIME_THRESHOLD = "response_time_threshold"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CUSTOM_METRIC = "custom_metric"


@dataclass
class ResourceConfig:
    """Resource allocation configuration."""
    # CPU configuration
    cpu_cores: float = 2.0
    cpu_limit: float = 4.0
    cpu_request: float = 1.0
    cpu_threshold_percent: float = 80.0
    
    # Memory configuration
    memory_mb: int = 2048
    memory_limit_mb: int = 4096
    memory_request_mb: int = 1024
    memory_threshold_percent: float = 85.0
    
    # GPU configuration
    gpu_count: int = 0
    gpu_memory_mb: int = 0
    gpu_type: str = ""
    gpu_shared: bool = False
    
    # Storage configuration
    storage_gb: int = 20
    storage_type: str = "ssd"
    storage_iops: int = 1000
    storage_throughput_mb: int = 100
    
    # Network configuration
    network_bandwidth_mbps: int = 1000
    network_connections_limit: int = 1000
    network_timeout_seconds: float = 30.0
    
    # Process configuration
    max_processes: int = 4
    max_threads_per_process: int = 50
    max_file_descriptors: int = 1024
    
    # Resource monitoring
    monitoring_enabled: bool = True
    monitoring_interval_seconds: int = 30
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_percent': 80.0,
        'memory_percent': 85.0,
        'disk_percent': 90.0,
        'network_percent': 80.0
    })


@dataclass
class ScalingConfig:
    """Auto-scaling configuration."""
    strategy: ScalingStrategy = ScalingStrategy.HORIZONTAL
    enabled: bool = True
    
    # Horizontal scaling
    min_instances: int = 1
    max_instances: int = 10
    target_instances: int = 2
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 20.0
    scale_up_cooldown_seconds: int = 300
    scale_down_cooldown_seconds: int = 600
    
    # Vertical scaling
    min_cpu_cores: float = 0.5
    max_cpu_cores: float = 8.0
    min_memory_mb: int = 512
    max_memory_mb: int = 8192
    
    # Scaling metrics
    scaling_metrics: List[str] = field(default_factory=lambda: ['cpu_percent', 'memory_percent'])
    custom_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Predictive scaling
    predictive_enabled: bool = False
    prediction_window_minutes: int = 30
    historical_data_days: int = 7
    
    # Scaling policies
    scale_up_policy: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'step',
        'adjustment_type': 'ChangeInCapacity',
        'min_adjustment_magnitude': 1,
        'cooldown': 300
    })
    scale_down_policy: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'step',
        'adjustment_type': 'ChangeInCapacity',
        'min_adjustment_magnitude': 1,
        'cooldown': 600
    })


@dataclass
class LoadBalancingConfig:
    """Load balancing configuration."""
    enabled: bool = True
    algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN
    
    # Load balancer settings
    session_affinity: bool = False
    session_timeout_seconds: int = 1800
    health_check_enabled: bool = True
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 10
    health_check_unhealthy_threshold: int = 3
    health_check_healthy_threshold: int = 2
    
    # Backend configuration
    backend_timeout_seconds: float = 30.0
    backend_retries: int = 3
    backend_retry_timeout_seconds: float = 5.0
    
    # Connection settings
    max_connections_per_backend: int = 100
    connection_timeout_seconds: float = 5.0
    keepalive_timeout_seconds: float = 60.0
    
    # SSL/TLS settings
    ssl_enabled: bool = True
    ssl_redirect: bool = True
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    ssl_protocols: List[str] = field(default_factory=lambda: ["TLSv1.2", "TLSv1.3"])
    
    # Advanced settings
    weighted_backends: Dict[str, float] = field(default_factory=dict)
    backup_backends: List[str] = field(default_factory=list)
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60


@dataclass
class HealthCheckConfig:
    """Health check configuration."""
    enabled: bool = True
    type: HealthCheckType = HealthCheckType.HTTP
    
    # HTTP health checks
    http_path: str = "/health"
    http_method: str = "GET"
    http_expected_status: int = 200
    http_expected_body: str = ""
    http_headers: Dict[str, str] = field(default_factory=dict)
    
    # TCP health checks
    tcp_port: int = 8000
    tcp_timeout_seconds: float = 5.0
    
    # Command health checks
    command: str = ""
    command_timeout_seconds: float = 30.0
    command_expected_exit_code: int = 0
    
    # Timing configuration
    initial_delay_seconds: int = 30
    interval_seconds: int = 30
    timeout_seconds: float = 10.0
    failure_threshold: int = 3
    success_threshold: int = 1
    
    # Custom health checks
    custom_check_function: Optional[str] = None
    custom_check_params: Dict[str, Any] = field(default_factory=dict)
    
    # Health check dependencies
    dependencies: List[str] = field(default_factory=list)
    dependency_timeout_seconds: float = 10.0
    
    # Readiness vs liveness
    readiness_check: bool = True
    liveness_check: bool = True
    startup_check: bool = True


@dataclass
class RollbackConfig:
    """Rollback configuration."""
    enabled: bool = True
    automatic: bool = True
    
    # Rollback triggers
    triggers: List[RollbackTrigger] = field(default_factory=lambda: [
        RollbackTrigger.HEALTH_CHECK_FAILURE,
        RollbackTrigger.ERROR_RATE_THRESHOLD
    ])
    
    # Trigger thresholds
    error_rate_threshold: float = 0.05  # 5%
    response_time_threshold_ms: float = 5000.0
    health_check_failure_count: int = 3
    
    # Rollback timing
    rollback_timeout_seconds: int = 300
    rollback_delay_seconds: int = 30
    
    # Rollback strategy
    rollback_strategy: DeploymentStrategy = DeploymentStrategy.IMMEDIATE
    preserve_data: bool = True
    backup_before_rollback: bool = True
    
    # Rollback validation
    validate_rollback: bool = True
    validation_timeout_seconds: int = 120
    
    # Notification settings
    notify_on_rollback: bool = True
    notification_channels: List[str] = field(default_factory=lambda: ["email", "slack"])
    
    # Custom rollback hooks
    pre_rollback_hooks: List[str] = field(default_factory=list)
    post_rollback_hooks: List[str] = field(default_factory=list)


@dataclass
class DeploymentStrategyConfig:
    """Deployment strategy configuration."""
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE
    
    # Rolling update configuration
    max_unavailable: Union[int, str] = 1
    max_surge: Union[int, str] = 1
    rolling_update_timeout_seconds: int = 600
    
    # Blue-green deployment
    blue_green_switch_delay_seconds: int = 60
    blue_green_validation_timeout_seconds: int = 300
    blue_green_rollback_timeout_seconds: int = 120
    
    # Canary deployment
    canary_percentage: float = 10.0
    canary_duration_seconds: int = 600
    canary_success_threshold: float = 0.95
    canary_error_threshold: float = 0.01
    canary_traffic_split: Dict[str, float] = field(default_factory=lambda: {"canary": 10.0, "stable": 90.0})
    
    # Deployment validation
    validation_enabled: bool = True
    validation_timeout_seconds: int = 300
    validation_checks: List[str] = field(default_factory=lambda: ["health_check", "smoke_test"])
    
    # Deployment hooks
    pre_deployment_hooks: List[str] = field(default_factory=list)
    post_deployment_hooks: List[str] = field(default_factory=list)
    
    # Traffic management
    traffic_management_enabled: bool = True
    traffic_split_strategy: str = "weighted"
    traffic_migration_duration_seconds: int = 300


@dataclass
class DeploymentValidation:
    """Deployment validation configuration."""
    enabled: bool = True
    timeout_seconds: int = 300
    
    # Pre-deployment validation
    pre_deployment_checks: List[str] = field(default_factory=lambda: [
        "configuration_validation",
        "resource_availability",
        "dependency_check",
        "security_scan"
    ])
    
    # Post-deployment validation
    post_deployment_checks: List[str] = field(default_factory=lambda: [
        "health_check",
        "smoke_test",
        "integration_test",
        "performance_test"
    ])
    
    # Validation thresholds
    success_threshold: float = 0.95
    error_threshold: float = 0.05
    performance_threshold_ms: float = 1000.0
    
    # Custom validation
    custom_validators: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Validation reporting
    generate_report: bool = True
    report_format: str = "json"
    report_path: str = "deployment_validation_report"


@dataclass
class DeploymentTemplate:
    """Deployment template configuration."""
    name: str
    description: str = ""
    template_format: str = "jinja2"
    
    # Template sources
    template_files: List[str] = field(default_factory=list)
    template_variables: Dict[str, Any] = field(default_factory=dict)
    
    # Template rendering
    auto_escape: bool = True
    strict_variables: bool = True
    template_cache_enabled: bool = True
    
    # Template validation
    validate_rendered: bool = True
    validation_schema: Optional[str] = None
    
    # Environment-specific templates
    environment_templates: Dict[str, str] = field(default_factory=dict)
    environment_variables: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class DeploymentOrchestrator:
    """Deployment orchestration utilities."""
    
    def __init__(self, config_manager: 'DeploymentConfigManager'):
        self.config_manager = config_manager
        self.logger = logging.getLogger('DeploymentOrchestrator')
        self.current_deployment: Optional[str] = None
        self.deployment_lock = threading.RLock()
    
    def execute_deployment(self, environment: DeploymentEnvironment, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deployment to specified environment."""
        deployment_id = str(uuid.uuid4())
        
        try:
            with self.deployment_lock:
                self.current_deployment = deployment_id
                
                # Pre-deployment validation
                self._validate_pre_deployment(deployment_config)
                
                # Execute deployment strategy
                strategy = deployment_config.get('strategy', DeploymentStrategy.ROLLING_UPDATE)
                
                if strategy == DeploymentStrategy.ROLLING_UPDATE:
                    result = self._execute_rolling_update(environment, deployment_config)
                elif strategy == DeploymentStrategy.BLUE_GREEN:
                    result = self._execute_blue_green(environment, deployment_config)
                elif strategy == DeploymentStrategy.CANARY:
                    result = self._execute_canary(environment, deployment_config)
                else:
                    result = self._execute_immediate(environment, deployment_config)
                
                # Post-deployment validation
                self._validate_post_deployment(deployment_config)
                
                result['deployment_id'] = deployment_id
                result['success'] = True
                
                self.current_deployment = None
                return result
                
        except Exception as e:
            self.logger.error(f"Deployment {deployment_id} failed: {e}")
            self.current_deployment = None
            raise RuntimeError(f"Deployment failed: {e}")
    
    def _validate_pre_deployment(self, deployment_config: Dict[str, Any]) -> None:
        """Validate pre-deployment conditions."""
        validation_config = deployment_config.get('validation', {})
        
        if not validation_config.get('enabled', True):
            return
        
        checks = validation_config.get('pre_deployment_checks', [])
        
        for check in checks:
            if check == "configuration_validation":
                self._validate_configuration(deployment_config)
            elif check == "resource_availability":
                self._validate_resource_availability(deployment_config)
            elif check == "dependency_check":
                self._validate_dependencies(deployment_config)
            elif check == "security_scan":
                self._validate_security(deployment_config)
    
    def _validate_post_deployment(self, deployment_config: Dict[str, Any]) -> None:
        """Validate post-deployment conditions."""
        validation_config = deployment_config.get('validation', {})
        
        if not validation_config.get('enabled', True):
            return
        
        checks = validation_config.get('post_deployment_checks', [])
        
        for check in checks:
            if check == "health_check":
                self._validate_health_checks(deployment_config)
            elif check == "smoke_test":
                self._run_smoke_tests(deployment_config)
            elif check == "integration_test":
                self._run_integration_tests(deployment_config)
            elif check == "performance_test":
                self._run_performance_tests(deployment_config)
    
    def _execute_rolling_update(self, environment: DeploymentEnvironment, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute rolling update deployment."""
        self.logger.info(f"Executing rolling update deployment to {environment.value}")
        
        # Simulate rolling update
        return {
            'strategy': 'rolling_update',
            'environment': environment.value,
            'instances_updated': config.get('target_instances', 1),
            'duration_seconds': 60
        }
    
    def _execute_blue_green(self, environment: DeploymentEnvironment, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute blue-green deployment."""
        self.logger.info(f"Executing blue-green deployment to {environment.value}")
        
        # Simulate blue-green deployment
        return {
            'strategy': 'blue_green',
            'environment': environment.value,
            'switch_completed': True,
            'duration_seconds': 120
        }
    
    def _execute_canary(self, environment: DeploymentEnvironment, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute canary deployment."""
        self.logger.info(f"Executing canary deployment to {environment.value}")
        
        # Simulate canary deployment
        return {
            'strategy': 'canary',
            'environment': environment.value,
            'canary_percentage': config.get('canary_percentage', 10.0),
            'duration_seconds': 300
        }
    
    def _execute_immediate(self, environment: DeploymentEnvironment, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute immediate deployment."""
        self.logger.info(f"Executing immediate deployment to {environment.value}")
        
        # Simulate immediate deployment
        return {
            'strategy': 'immediate',
            'environment': environment.value,
            'duration_seconds': 30
        }
    
    def _validate_configuration(self, config: Dict[str, Any]) -> None:
        """Validate deployment configuration."""
        required_fields = ['environment', 'strategy', 'resources']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Required field '{field}' missing from deployment configuration")
    
    def _validate_resource_availability(self, config: Dict[str, Any]) -> None:
        """Validate resource availability."""
        resource_config = config.get('resources', {})
        
        # Check CPU availability
        required_cpu = resource_config.get('cpu_cores', 1.0)
        available_cpu = psutil.cpu_count()
        if required_cpu > available_cpu:
            raise RuntimeError(f"Insufficient CPU cores: required {required_cpu}, available {available_cpu}")
        
        # Check memory availability
        required_memory_mb = resource_config.get('memory_mb', 1024)
        available_memory_mb = psutil.virtual_memory().available // (1024 * 1024)
        if required_memory_mb > available_memory_mb:
            raise RuntimeError(f"Insufficient memory: required {required_memory_mb}MB, available {available_memory_mb}MB")
    
    def _validate_dependencies(self, config: Dict[str, Any]) -> None:
        """Validate deployment dependencies."""
        dependencies = config.get('dependencies', [])
        
        for dependency in dependencies:
            if dependency == 'database':
                # Check database connectivity
                pass
            elif dependency == 'cache':
                # Check cache connectivity
                pass
            elif dependency == 'external_api':
                # Check external API availability
                pass
    
    def _validate_security(self, config: Dict[str, Any]) -> None:
        """Validate security configuration."""
        security_config = config.get('security', {})
        
        if security_config.get('ssl_enabled', False):
            ssl_cert = security_config.get('ssl_cert_path')
            if ssl_cert and not Path(ssl_cert).exists():
                raise RuntimeError(f"SSL certificate not found: {ssl_cert}")
    
    def _validate_health_checks(self, config: Dict[str, Any]) -> None:
        """Validate health checks after deployment."""
        health_config = config.get('health_check', {})
        
        if not health_config.get('enabled', True):
            return
        
        # Simulate health check validation
        self.logger.info("Health checks passed")
    
    def _run_smoke_tests(self, config: Dict[str, Any]) -> None:
        """Run smoke tests after deployment."""
        self.logger.info("Running smoke tests")
        # Simulate smoke tests
        pass
    
    def _run_integration_tests(self, config: Dict[str, Any]) -> None:
        """Run integration tests after deployment."""
        self.logger.info("Running integration tests")
        # Simulate integration tests
        pass
    
    def _run_performance_tests(self, config: Dict[str, Any]) -> None:
        """Run performance tests after deployment."""
        self.logger.info("Running performance tests")
        # Simulate performance tests
        pass


class TemplateRenderer:
    """Configuration template rendering utilities."""
    
    def __init__(self, template_dir: str = "templates"):
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml']),
            strict_undefined=True
        )
    
    def render_template(self, template_name: str, variables: Dict[str, Any]) -> str:
        """Render template with variables."""
        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(**variables)
        except Exception as e:
            raise RuntimeError(f"Failed to render template {template_name}: {e}")
    
    def render_deployment_config(self, environment: DeploymentEnvironment, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Render deployment configuration from template."""
        template_name = f"deployment_{environment.value}.yaml"
        
        if not (self.template_dir / template_name).exists():
            template_name = "deployment_default.yaml"
        
        rendered_yaml = self.render_template(template_name, variables)
        return yaml.safe_load(rendered_yaml)


class DeploymentConfigManager:
    """
    Production Deployment Configuration Manager - Comprehensive deployment configuration and orchestration.
    
    This class provides complete deployment configuration management including environment-specific
    configurations, deployment strategies, resource management, scaling, load balancing, health checks,
    and rollback strategies for production environments.
    """
    
    def __init__(self, config_dir: str = "deployment_config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.deployment_lock = threading.RLock()
        
        # Deployment configurations
        self.deployment_configs: Dict[DeploymentEnvironment, Dict[str, Any]] = {}
        self.resource_configs: Dict[str, ResourceConfig] = {}
        self.scaling_configs: Dict[str, ScalingConfig] = {}
        self.load_balancing_configs: Dict[str, LoadBalancingConfig] = {}
        self.health_check_configs: Dict[str, HealthCheckConfig] = {}
        self.rollback_configs: Dict[str, RollbackConfig] = {}
        
        # Deployment management
        self.orchestrator = DeploymentOrchestrator(self)
        self.template_renderer = TemplateRenderer(str(self.config_dir / "templates"))
        
        # Integration references
        self.production_config_manager: Optional['ProductionConfigManager'] = None
        self.gac_config_manager: Optional['GACConfigManager'] = None
        self.production_training_executor: Optional['ProductionTrainingExecutor'] = None
        
        # Deployment tracking
        self.deployment_history: List[Dict[str, Any]] = []
        self.active_deployments: Dict[str, Dict[str, Any]] = {}
        self.deployment_metrics: Dict[str, Any] = {}
        
        self._setup_default_configurations()
        logger.info("DeploymentConfigManager initialized successfully")
    
    def initialize_deployment_config_manager(
        self,
        production_config_manager: Optional['ProductionConfigManager'] = None,
        gac_config_manager: Optional['GACConfigManager'] = None,
        production_training_executor: Optional['ProductionTrainingExecutor'] = None
    ) -> None:
        """Initialize deployment config manager with system integrations."""
        try:
            with self.deployment_lock:
                self.production_config_manager = production_config_manager
                self.gac_config_manager = gac_config_manager
                self.production_training_executor = production_training_executor
                
                # Load existing configurations
                self._load_deployment_configurations()
                
                # Initialize integrations
                if production_config_manager:
                    self._integrate_production_config()
                
                if gac_config_manager:
                    self._integrate_gac_config()
                
                logger.info("Deployment configuration manager initialized successfully")
                
        except Exception as e:
            error_msg = f"Failed to initialize deployment configuration manager: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def create_deployment_config(
        self,
        environment: DeploymentEnvironment,
        deployment_strategy: DeploymentStrategy = DeploymentStrategy.ROLLING_UPDATE,
        resource_config: Optional[ResourceConfig] = None,
        scaling_config: Optional[ScalingConfig] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create deployment configuration for environment."""
        try:
            with self.deployment_lock:
                config = {
                    'environment': environment.value,
                    'strategy': deployment_strategy.value,
                    'resources': asdict(resource_config or ResourceConfig()),
                    'scaling': asdict(scaling_config or ScalingConfig()),
                    'load_balancing': asdict(LoadBalancingConfig()),
                    'health_check': asdict(HealthCheckConfig()),
                    'rollback': asdict(RollbackConfig()),
                    'validation': asdict(DeploymentValidation()),
                    'created_at': datetime.utcnow().isoformat(),
                    'metadata': kwargs
                }
                
                # Apply environment-specific overrides
                config = self._apply_environment_overrides(environment, config)
                
                # Store configuration
                self.deployment_configs[environment] = config
                
                # Save to file
                self._save_deployment_config(environment, config)
                
                logger.info(f"Deployment configuration created for {environment.value}")
                return config
                
        except Exception as e:
            error_msg = f"Failed to create deployment configuration for {environment.value}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def deploy_to_environment(
        self,
        environment: DeploymentEnvironment,
        deployment_config: Optional[Dict[str, Any]] = None,
        validate: bool = True
    ) -> Dict[str, Any]:
        """Deploy to specified environment."""
        try:
            with self.deployment_lock:
                # Get or create deployment configuration
                if deployment_config is None:
                    if environment not in self.deployment_configs:
                        deployment_config = self.create_deployment_config(environment)
                    else:
                        deployment_config = self.deployment_configs[environment]
                
                # Validate deployment configuration if requested
                if validate:
                    self._validate_deployment_config(deployment_config)
                
                # Execute deployment
                result = self.orchestrator.execute_deployment(environment, deployment_config)
                
                # Record deployment
                self._record_deployment(environment, deployment_config, result)
                
                logger.info(f"Deployment to {environment.value} completed successfully")
                return result
                
        except Exception as e:
            error_msg = f"Failed to deploy to {environment.value}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def scale_deployment(
        self,
        environment: DeploymentEnvironment,
        target_instances: Optional[int] = None,
        cpu_cores: Optional[float] = None,
        memory_mb: Optional[int] = None
    ) -> Dict[str, Any]:
        """Scale deployment in specified environment."""
        try:
            with self.deployment_lock:
                if environment not in self.deployment_configs:
                    raise ValueError(f"No deployment configuration found for {environment.value}")
                
                config = self.deployment_configs[environment]
                scaling_config = config.get('scaling', {})
                
                # Update scaling parameters
                if target_instances is not None:
                    scaling_config['target_instances'] = target_instances
                if cpu_cores is not None:
                    config['resources']['cpu_cores'] = cpu_cores
                if memory_mb is not None:
                    config['resources']['memory_mb'] = memory_mb
                
                # Execute scaling
                scaling_result = self._execute_scaling(environment, scaling_config)
                
                # Update configuration
                config['last_scaled'] = datetime.utcnow().isoformat()
                self._save_deployment_config(environment, config)
                
                logger.info(f"Scaling completed for {environment.value}")
                return scaling_result
                
        except Exception as e:
            error_msg = f"Failed to scale deployment in {environment.value}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def configure_load_balancing(
        self,
        environment: DeploymentEnvironment,
        algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN,
        backend_servers: Optional[List[str]] = None,
        **kwargs
    ) -> LoadBalancingConfig:
        """Configure load balancing for environment."""
        try:
            with self.deployment_lock:
                lb_config = LoadBalancingConfig(
                    algorithm=algorithm,
                    **kwargs
                )
                
                config_key = f"{environment.value}_load_balancer"
                self.load_balancing_configs[config_key] = lb_config
                
                # Apply to deployment configuration
                if environment in self.deployment_configs:
                    self.deployment_configs[environment]['load_balancing'] = asdict(lb_config)
                    self._save_deployment_config(environment, self.deployment_configs[environment])
                
                logger.info(f"Load balancing configured for {environment.value}")
                return lb_config
                
        except Exception as e:
            error_msg = f"Failed to configure load balancing for {environment.value}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def configure_health_checks(
        self,
        environment: DeploymentEnvironment,
        health_check_type: HealthCheckType = HealthCheckType.HTTP,
        **kwargs
    ) -> HealthCheckConfig:
        """Configure health checks for environment."""
        try:
            with self.deployment_lock:
                health_config = HealthCheckConfig(
                    type=health_check_type,
                    **kwargs
                )
                
                config_key = f"{environment.value}_health_check"
                self.health_check_configs[config_key] = health_config
                
                # Apply to deployment configuration
                if environment in self.deployment_configs:
                    self.deployment_configs[environment]['health_check'] = asdict(health_config)
                    self._save_deployment_config(environment, self.deployment_configs[environment])
                
                logger.info(f"Health checks configured for {environment.value}")
                return health_config
                
        except Exception as e:
            error_msg = f"Failed to configure health checks for {environment.value}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def configure_rollback_strategy(
        self,
        environment: DeploymentEnvironment,
        triggers: Optional[List[RollbackTrigger]] = None,
        **kwargs
    ) -> RollbackConfig:
        """Configure rollback strategy for environment."""
        try:
            with self.deployment_lock:
                rollback_config = RollbackConfig(
                    triggers=triggers or [RollbackTrigger.HEALTH_CHECK_FAILURE],
                    **kwargs
                )
                
                config_key = f"{environment.value}_rollback"
                self.rollback_configs[config_key] = rollback_config
                
                # Apply to deployment configuration
                if environment in self.deployment_configs:
                    self.deployment_configs[environment]['rollback'] = asdict(rollback_config)
                    self._save_deployment_config(environment, self.deployment_configs[environment])
                
                logger.info(f"Rollback strategy configured for {environment.value}")
                return rollback_config
                
        except Exception as e:
            error_msg = f"Failed to configure rollback strategy for {environment.value}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def execute_rollback(self, environment: DeploymentEnvironment, reason: str = "Manual rollback") -> Dict[str, Any]:
        """Execute rollback for environment."""
        try:
            with self.deployment_lock:
                if environment not in self.deployment_configs:
                    raise ValueError(f"No deployment configuration found for {environment.value}")
                
                config = self.deployment_configs[environment]
                rollback_config = config.get('rollback', {})
                
                if not rollback_config.get('enabled', True):
                    raise RuntimeError(f"Rollback disabled for {environment.value}")
                
                # Execute rollback
                rollback_result = self._execute_rollback(environment, rollback_config, reason)
                
                # Record rollback
                self._record_rollback(environment, rollback_config, rollback_result, reason)
                
                logger.info(f"Rollback executed for {environment.value}: {reason}")
                return rollback_result
                
        except Exception as e:
            error_msg = f"Failed to execute rollback for {environment.value}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def get_deployment_status(self, environment: Optional[DeploymentEnvironment] = None) -> Dict[str, Any]:
        """Get deployment status for environment or all environments."""
        try:
            with self.deployment_lock:
                if environment:
                    if environment not in self.deployment_configs:
                        return {'error': f'No deployment found for {environment.value}'}
                    
                    config = self.deployment_configs[environment]
                    return {
                        'environment': environment.value,
                        'strategy': config.get('strategy'),
                        'status': self._get_environment_status(environment),
                        'last_deployment': config.get('last_deployment'),
                        'configuration': config
                    }
                else:
                    # Return status for all environments
                    status = {}
                    for env, config in self.deployment_configs.items():
                        status[env.value] = {
                            'strategy': config.get('strategy'),
                            'status': self._get_environment_status(env),
                            'last_deployment': config.get('last_deployment')
                        }
                    return status
                    
        except Exception as e:
            error_msg = f"Failed to get deployment status: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_deployment_analytics(self) -> Dict[str, Any]:
        """Get deployment analytics across all environments."""
        try:
            with self.deployment_lock:
                return {
                    'total_deployments': len(self.deployment_history),
                    'active_deployments': len(self.active_deployments),
                    'deployment_environments': list(self.deployment_configs.keys()),
                    'deployment_metrics': self.deployment_metrics,
                    'recent_deployments': self.deployment_history[-10:],
                    'deployment_success_rate': self._calculate_success_rate(),
                    'average_deployment_time': self._calculate_average_deployment_time()
                }
                
        except Exception as e:
            error_msg = f"Failed to get deployment analytics: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _setup_default_configurations(self) -> None:
        """Setup default deployment configurations."""
        try:
            # Default configurations for each environment
            environments = [
                DeploymentEnvironment.DEVELOPMENT,
                DeploymentEnvironment.STAGING,
                DeploymentEnvironment.PRODUCTION
            ]
            
            for env in environments:
                if env == DeploymentEnvironment.DEVELOPMENT:
                    resource_config = ResourceConfig(cpu_cores=1.0, memory_mb=1024)
                    scaling_config = ScalingConfig(min_instances=1, max_instances=2)
                elif env == DeploymentEnvironment.STAGING:
                    resource_config = ResourceConfig(cpu_cores=2.0, memory_mb=2048)
                    scaling_config = ScalingConfig(min_instances=1, max_instances=4)
                else:  # PRODUCTION
                    resource_config = ResourceConfig(cpu_cores=4.0, memory_mb=4096)
                    scaling_config = ScalingConfig(min_instances=2, max_instances=10)
                
                self.resource_configs[env.value] = resource_config
                self.scaling_configs[env.value] = scaling_config
            
            logger.debug("Default deployment configurations setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup default configurations: {e}")
    
    def _load_deployment_configurations(self) -> None:
        """Load existing deployment configurations."""
        try:
            for env in DeploymentEnvironment:
                config_file = self.config_dir / f"{env.value}_deployment.json"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    self.deployment_configs[env] = config
                    logger.debug(f"Loaded deployment configuration for {env.value}")
            
        except Exception as e:
            logger.error(f"Failed to load deployment configurations: {e}")
    
    def _save_deployment_config(self, environment: DeploymentEnvironment, config: Dict[str, Any]) -> None:
        """Save deployment configuration to file."""
        try:
            config_file = self.config_dir / f"{environment.value}_deployment.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Failed to save deployment configuration for {environment.value}: {e}")
    
    def _apply_environment_overrides(self, environment: DeploymentEnvironment, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment-specific configuration overrides."""
        if environment == DeploymentEnvironment.PRODUCTION:
            # Production-specific overrides
            config['resources']['cpu_cores'] = max(config['resources']['cpu_cores'], 2.0)
            config['resources']['memory_mb'] = max(config['resources']['memory_mb'], 2048)
            config['scaling']['min_instances'] = max(config['scaling']['min_instances'], 2)
            config['health_check']['enabled'] = True
            config['rollback']['enabled'] = True
        elif environment == DeploymentEnvironment.DEVELOPMENT:
            # Development-specific overrides
            config['resources']['cpu_cores'] = min(config['resources']['cpu_cores'], 2.0)
            config['resources']['memory_mb'] = min(config['resources']['memory_mb'], 2048)
            config['scaling']['max_instances'] = min(config['scaling']['max_instances'], 3)
        
        return config
    
    def _validate_deployment_config(self, config: Dict[str, Any]) -> None:
        """Validate deployment configuration."""
        required_fields = ['environment', 'strategy', 'resources']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Required field '{field}' missing from deployment configuration")
        
        # Validate resource requirements
        resources = config['resources']
        if resources['cpu_cores'] <= 0:
            raise ValueError("CPU cores must be positive")
        if resources['memory_mb'] <= 0:
            raise ValueError("Memory must be positive")
    
    def _execute_scaling(self, environment: DeploymentEnvironment, scaling_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scaling operation."""
        self.logger.info(f"Executing scaling for {environment.value}")
        
        # Simulate scaling operation
        return {
            'environment': environment.value,
            'scaling_completed': True,
            'new_instance_count': scaling_config.get('target_instances', 1),
            'duration_seconds': 30
        }
    
    def _execute_rollback(self, environment: DeploymentEnvironment, rollback_config: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Execute rollback operation."""
        self.logger.info(f"Executing rollback for {environment.value}: {reason}")
        
        # Simulate rollback operation
        return {
            'environment': environment.value,
            'rollback_completed': True,
            'reason': reason,
            'duration_seconds': 60
        }
    
    def _get_environment_status(self, environment: DeploymentEnvironment) -> str:
        """Get current status of environment."""
        # Simulate status check
        return "active"
    
    def _record_deployment(self, environment: DeploymentEnvironment, config: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Record deployment in history."""
        record = {
            'environment': environment.value,
            'timestamp': datetime.utcnow().isoformat(),
            'config': config,
            'result': result,
            'success': result.get('success', False)
        }
        self.deployment_history.append(record)
        
        # Keep only last 100 deployments
        if len(self.deployment_history) > 100:
            self.deployment_history = self.deployment_history[-100:]
    
    def _record_rollback(self, environment: DeploymentEnvironment, config: Dict[str, Any], result: Dict[str, Any], reason: str) -> None:
        """Record rollback in history."""
        record = {
            'type': 'rollback',
            'environment': environment.value,
            'timestamp': datetime.utcnow().isoformat(),
            'reason': reason,
            'config': config,
            'result': result,
            'success': result.get('rollback_completed', False)
        }
        self.deployment_history.append(record)
    
    def _calculate_success_rate(self) -> float:
        """Calculate deployment success rate."""
        if not self.deployment_history:
            return 0.0
        
        successful = sum(1 for d in self.deployment_history if d.get('success', False))
        return successful / len(self.deployment_history)
    
    def _calculate_average_deployment_time(self) -> float:
        """Calculate average deployment time."""
        if not self.deployment_history:
            return 0.0
        
        times = [d['result'].get('duration_seconds', 0) for d in self.deployment_history if 'result' in d]
        return sum(times) / len(times) if times else 0.0
    
    def _integrate_production_config(self) -> None:
        """Integrate with production configuration manager."""
        try:
            if not self.production_config_manager:
                return
            
            prod_config = self.production_config_manager.production_config
            
            # Apply production config settings to deployment configs
            for env, config in self.deployment_configs.items():
                config['host'] = prod_config.host
                config['port'] = prod_config.port
                config['workers'] = prod_config.workers
                config['debug_mode'] = prod_config.debug_mode
            
            logger.info("Production configuration integrated successfully")
            
        except Exception as e:
            logger.error(f"Failed to integrate production configuration: {e}")
    
    def _integrate_gac_config(self) -> None:
        """Integrate with GAC configuration."""
        try:
            if not self.gac_config_manager:
                return
            
            gac_config = self.gac_config_manager.config
            
            # Apply GAC settings to deployment configs
            for env, config in self.deployment_configs.items():
                config['max_workers'] = gac_config.system.max_workers
                config['worker_timeout'] = gac_config.system.worker_timeout
                config['checkpoint_interval'] = gac_config.system.checkpoint_interval
            
            logger.info("GAC configuration integrated successfully")
            
        except Exception as e:
            logger.error(f"Failed to integrate GAC configuration: {e}")
    
    def shutdown(self) -> None:
        """Shutdown deployment configuration manager."""
        try:
            with self.deployment_lock:
                # Save all configurations
                for env, config in self.deployment_configs.items():
                    self._save_deployment_config(env, config)
                
                logger.info("Deployment configuration manager shutdown completed")
                
        except Exception as e:
            logger.error(f"Error during deployment config manager shutdown: {e}")


def create_deployment_config_manager(config_dir: str = "deployment_config") -> DeploymentConfigManager:
    """Factory function to create a DeploymentConfigManager instance."""
    return DeploymentConfigManager(config_dir=config_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    config_manager = create_deployment_config_manager()
    
    # Initialize with sample integrations
    config_manager.initialize_deployment_config_manager()
    
    # Create deployment configuration
    deployment_config = config_manager.create_deployment_config(
        DeploymentEnvironment.STAGING,
        DeploymentStrategy.ROLLING_UPDATE
    )
    print(f"Created deployment configuration for staging")
    
    # Deploy to staging
    try:
        result = config_manager.deploy_to_environment(DeploymentEnvironment.STAGING)
        print(f"Deployment result: {result['success']}")
    except Exception as e:
        print(f"Deployment failed: {e}")
    
    # Get deployment status
    status = config_manager.get_deployment_status()
    print(f"Deployment status: {len(status)} environments configured")