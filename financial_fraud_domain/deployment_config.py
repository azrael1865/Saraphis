"""
Deployment Configuration for Financial Fraud Detection
Comprehensive deployment configuration and management for production environments
"""

import os
import json
import yaml
import logging
import socket
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import threading
import time
import psutil
import requests
from concurrent.futures import ThreadPoolExecutor, Future

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    DISASTER_RECOVERY = "disaster_recovery"


class DeploymentStrategy(Enum):
    """Deployment strategy types"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"


class DeploymentStatus(Enum):
    """Deployment status types"""
    PENDING = "pending"
    VALIDATING = "validating"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class ServiceType(Enum):
    """Service types in the deployment"""
    API = "api"
    WORKER = "worker"
    DATABASE = "database"
    CACHE = "cache"
    QUEUE = "queue"
    MONITORING = "monitoring"


@dataclass
class ServiceConfig:
    """Configuration for a single service"""
    name: str
    type: ServiceType
    version: str
    replicas: int = 1
    cpu_limit: str = "1000m"  # millicores
    memory_limit: str = "1Gi"
    cpu_request: str = "500m"
    memory_request: str = "512Mi"
    environment_vars: Dict[str, str] = field(default_factory=dict)
    ports: List[int] = field(default_factory=list)
    health_check_path: str = "/health"
    health_check_interval: int = 30  # seconds
    startup_timeout: int = 300  # seconds
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['type'] = self.type.value
        return data


@dataclass
class DatabaseConfig:
    """Database configuration"""
    type: str = "postgresql"  # postgresql, mysql, mongodb
    host: str = "localhost"
    port: int = 5432
    database: str = "fraud_detection"
    username: str = "fraud_user"
    password: str = ""  # Should be encrypted
    ssl_enabled: bool = True
    connection_pool_size: int = 20
    max_connections: int = 100
    backup_enabled: bool = True
    backup_schedule: str = "0 2 * * *"  # Daily at 2 AM
    retention_days: int = 30


@dataclass
class SecurityConfig:
    """Security configuration"""
    ssl_enabled: bool = True
    ssl_cert_path: str = "/etc/ssl/certs/fraud_detection.crt"
    ssl_key_path: str = "/etc/ssl/private/fraud_detection.key"
    api_key_enabled: bool = True
    api_key_header: str = "X-API-Key"
    rate_limiting_enabled: bool = True
    rate_limit_requests: int = 1000
    rate_limit_window: int = 3600  # seconds
    ip_whitelist: List[str] = field(default_factory=list)
    ip_blacklist: List[str] = field(default_factory=list)
    encryption_key: str = ""  # Should be in secure storage
    audit_logging_enabled: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    enabled: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    log_format: str = "json"
    apm_enabled: bool = True
    apm_service_name: str = "fraud-detection"
    apm_server_url: str = "http://localhost:8200"
    alerting_enabled: bool = True
    alert_channels: List[str] = field(default_factory=lambda: ["email", "slack"])
    health_check_interval: int = 30  # seconds
    custom_metrics: Dict[str, str] = field(default_factory=dict)


@dataclass
class DeploymentMetrics:
    """Deployment performance metrics"""
    deployment_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    status: DeploymentStatus = DeploymentStatus.PENDING
    services_deployed: int = 0
    services_failed: int = 0
    validation_time_seconds: float = 0.0
    deployment_time_seconds: float = 0.0
    rollback_time_seconds: Optional[float] = None
    error_count: int = 0
    warnings: List[str] = field(default_factory=list)
    
    def calculate_duration(self):
        """Calculate deployment duration"""
        if self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()


@dataclass
class DeploymentResult:
    """Result of a deployment operation"""
    success: bool
    deployment_id: str
    environment: Environment
    strategy: DeploymentStrategy
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    deployed_services: List[str]
    failed_services: List[str]
    metrics: DeploymentMetrics
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class DeploymentValidator:
    """Validates deployment configurations"""
    
    @staticmethod
    def validate_service_config(config: ServiceConfig) -> Tuple[bool, List[str]]:
        """Validate service configuration"""
        errors = []
        
        # Validate required fields
        if not config.name:
            errors.append("Service name is required")
        
        if not config.version:
            errors.append("Service version is required")
        
        if config.replicas < 1:
            errors.append("Replicas must be at least 1")
        
        # Validate resource limits
        if not DeploymentValidator._validate_resource_string(config.cpu_limit):
            errors.append(f"Invalid CPU limit: {config.cpu_limit}")
        
        if not DeploymentValidator._validate_resource_string(config.memory_limit):
            errors.append(f"Invalid memory limit: {config.memory_limit}")
        
        # Validate ports
        for port in config.ports:
            if not 1 <= port <= 65535:
                errors.append(f"Invalid port number: {port}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _validate_resource_string(resource: str) -> bool:
        """Validate Kubernetes-style resource string"""
        import re
        # Matches patterns like: 100m, 1000Mi, 2Gi, 1, 0.5
        pattern = r'^(\d+\.?\d*)(m|Mi|Gi|Ki|Ti)?$'
        return bool(re.match(pattern, resource))
    
    @staticmethod
    def validate_database_config(config: DatabaseConfig) -> Tuple[bool, List[str]]:
        """Validate database configuration"""
        errors = []
        
        if config.type not in ["postgresql", "mysql", "mongodb"]:
            errors.append(f"Unsupported database type: {config.type}")
        
        if not config.host:
            errors.append("Database host is required")
        
        if not 1 <= config.port <= 65535:
            errors.append(f"Invalid database port: {config.port}")
        
        if config.connection_pool_size > config.max_connections:
            errors.append("Connection pool size cannot exceed max connections")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_security_config(config: SecurityConfig) -> Tuple[bool, List[str]]:
        """Validate security configuration"""
        errors = []
        
        if config.ssl_enabled:
            if not os.path.exists(config.ssl_cert_path):
                errors.append(f"SSL certificate not found: {config.ssl_cert_path}")
            
            if not os.path.exists(config.ssl_key_path):
                errors.append(f"SSL key not found: {config.ssl_key_path}")
        
        if config.rate_limiting_enabled:
            if config.rate_limit_requests < 1:
                errors.append("Rate limit requests must be positive")
            
            if config.rate_limit_window < 1:
                errors.append("Rate limit window must be positive")
        
        return len(errors) == 0, errors


class DeploymentStrategyBase(ABC):
    """Abstract base class for deployment strategies"""
    
    @abstractmethod
    def deploy(self, services: List[ServiceConfig], environment: Environment) -> DeploymentResult:
        """Execute deployment strategy"""
        pass
    
    @abstractmethod
    def rollback(self, deployment_id: str) -> bool:
        """Rollback deployment"""
        pass


class BlueGreenStrategy(DeploymentStrategyBase):
    """Blue-Green deployment strategy"""
    
    def __init__(self):
        self.active_environment = "blue"
        self.inactive_environment = "green"
        
    def deploy(self, services: List[ServiceConfig], environment: Environment) -> DeploymentResult:
        """Deploy using blue-green strategy"""
        deployment_id = self._generate_deployment_id()
        start_time = datetime.now()
        metrics = DeploymentMetrics(deployment_id=deployment_id, start_time=start_time)
        
        try:
            # Deploy to inactive environment
            logger.info(f"Deploying to {self.inactive_environment} environment")
            deployed_services = []
            failed_services = []
            
            for service in services:
                try:
                    self._deploy_service(service, self.inactive_environment)
                    deployed_services.append(service.name)
                except Exception as e:
                    logger.error(f"Failed to deploy {service.name}: {e}")
                    failed_services.append(service.name)
                    metrics.services_failed += 1
            
            metrics.services_deployed = len(deployed_services)
            
            # Switch traffic to new environment
            if not failed_services:
                logger.info(f"Switching traffic from {self.active_environment} to {self.inactive_environment}")
                self._switch_traffic()
                self.active_environment, self.inactive_environment = self.inactive_environment, self.active_environment
                metrics.status = DeploymentStatus.DEPLOYED
            else:
                metrics.status = DeploymentStatus.FAILED
            
            end_time = datetime.now()
            metrics.end_time = end_time
            metrics.calculate_duration()
            
            return DeploymentResult(
                success=not failed_services,
                deployment_id=deployment_id,
                environment=environment,
                strategy=DeploymentStrategy.BLUE_GREEN,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=metrics.duration_seconds,
                deployed_services=deployed_services,
                failed_services=failed_services,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Blue-green deployment failed: {e}")
            metrics.status = DeploymentStatus.FAILED
            raise
    
    def rollback(self, deployment_id: str) -> bool:
        """Rollback to previous environment"""
        try:
            logger.info(f"Rolling back deployment {deployment_id}")
            self._switch_traffic()
            self.active_environment, self.inactive_environment = self.inactive_environment, self.active_environment
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def _deploy_service(self, service: ServiceConfig, environment: str):
        """Deploy a single service"""
        # Simulated deployment - replace with actual deployment logic
        logger.info(f"Deploying {service.name} v{service.version} to {environment}")
        time.sleep(0.5)  # Simulate deployment time
    
    def _switch_traffic(self):
        """Switch traffic between environments"""
        # Simulated traffic switch - replace with actual logic
        logger.info("Switching traffic between environments")
        time.sleep(1)  # Simulate switch time
    
    def _generate_deployment_id(self) -> str:
        """Generate unique deployment ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = hashlib.md5(os.urandom(16)).hexdigest()[:6]
        return f"deploy-{timestamp}-{random_suffix}"


class CanaryStrategy(DeploymentStrategyBase):
    """Canary deployment strategy"""
    
    def __init__(self, initial_percentage: int = 10, increment: int = 10):
        self.initial_percentage = initial_percentage
        self.increment = increment
        self.current_percentage = 0
        
    def deploy(self, services: List[ServiceConfig], environment: Environment) -> DeploymentResult:
        """Deploy using canary strategy"""
        deployment_id = self._generate_deployment_id()
        start_time = datetime.now()
        metrics = DeploymentMetrics(deployment_id=deployment_id, start_time=start_time)
        
        try:
            deployed_services = []
            failed_services = []
            
            # Start with initial percentage
            self.current_percentage = self.initial_percentage
            
            while self.current_percentage <= 100:
                logger.info(f"Deploying canary at {self.current_percentage}% traffic")
                
                for service in services:
                    try:
                        self._deploy_canary(service, self.current_percentage)
                        if service.name not in deployed_services:
                            deployed_services.append(service.name)
                    except Exception as e:
                        logger.error(f"Failed to deploy canary for {service.name}: {e}")
                        failed_services.append(service.name)
                        metrics.services_failed += 1
                        break
                
                if failed_services:
                    break
                
                # Monitor canary health
                if not self._check_canary_health():
                    logger.error("Canary health check failed")
                    metrics.status = DeploymentStatus.FAILED
                    break
                
                # Increment traffic
                self.current_percentage = min(100, self.current_percentage + self.increment)
                time.sleep(2)  # Wait between increments
            
            metrics.services_deployed = len(deployed_services)
            
            if not failed_services and self.current_percentage >= 100:
                metrics.status = DeploymentStatus.DEPLOYED
            else:
                metrics.status = DeploymentStatus.FAILED
            
            end_time = datetime.now()
            metrics.end_time = end_time
            metrics.calculate_duration()
            
            return DeploymentResult(
                success=not failed_services and self.current_percentage >= 100,
                deployment_id=deployment_id,
                environment=environment,
                strategy=DeploymentStrategy.CANARY,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=metrics.duration_seconds,
                deployed_services=deployed_services,
                failed_services=failed_services,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            metrics.status = DeploymentStatus.FAILED
            raise
    
    def rollback(self, deployment_id: str) -> bool:
        """Rollback canary deployment"""
        try:
            logger.info(f"Rolling back canary deployment {deployment_id}")
            self.current_percentage = 0
            # Implement actual rollback logic
            return True
        except Exception as e:
            logger.error(f"Canary rollback failed: {e}")
            return False
    
    def _deploy_canary(self, service: ServiceConfig, percentage: int):
        """Deploy canary instance"""
        logger.info(f"Deploying canary {service.name} at {percentage}% traffic")
        time.sleep(0.3)  # Simulate deployment
    
    def _check_canary_health(self) -> bool:
        """Check canary instance health"""
        # Simulated health check - replace with actual monitoring
        logger.info("Checking canary health metrics")
        time.sleep(0.5)
        return True  # Simulate healthy canary
    
    def _generate_deployment_id(self) -> str:
        """Generate unique deployment ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = hashlib.md5(os.urandom(16)).hexdigest()[:6]
        return f"canary-{timestamp}-{random_suffix}"


class FinancialDeploymentConfig:
    """
    Comprehensive deployment configuration for Financial Fraud Detection.
    Manages deployment strategies, environments, and configurations.
    """
    
    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        """Initialize deployment configuration"""
        self.environment = environment
        self.services: List[ServiceConfig] = []
        self.database_config = DatabaseConfig()
        self.security_config = SecurityConfig()
        self.monitoring_config = MonitoringConfig()
        self.deployment_history: List[DeploymentResult] = []
        self.current_deployment: Optional[DeploymentResult] = None
        self.config_file_path = Path("deployment_config.yaml")
        self.lock = threading.Lock()
        
        # Initialize deployment strategies
        self.strategies = {
            DeploymentStrategy.BLUE_GREEN: BlueGreenStrategy(),
            DeploymentStrategy.CANARY: CanaryStrategy(),
        }
        
        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = {
            'deployment_duration': [],
            'validation_duration': [],
            'rollback_duration': []
        }
        
        logger.info(f"FinancialDeploymentConfig initialized for {environment.value}")
    
    def configure_deployment(self, config: Dict[str, Any]) -> bool:
        """Configure deployment from dictionary"""
        try:
            with self.lock:
                # Configure services
                if 'services' in config:
                    self.services = []
                    for service_config in config['services']:
                        service = ServiceConfig(
                            name=service_config['name'],
                            type=ServiceType(service_config['type']),
                            version=service_config['version'],
                            replicas=service_config.get('replicas', 1),
                            cpu_limit=service_config.get('cpu_limit', '1000m'),
                            memory_limit=service_config.get('memory_limit', '1Gi'),
                            environment_vars=service_config.get('environment_vars', {}),
                            ports=service_config.get('ports', []),
                            dependencies=service_config.get('dependencies', [])
                        )
                        self.services.append(service)
                
                # Configure database
                if 'database' in config:
                    db_config = config['database']
                    self.database_config = DatabaseConfig(**db_config)
                
                # Configure security
                if 'security' in config:
                    sec_config = config['security']
                    self.security_config = SecurityConfig(**sec_config)
                
                # Configure monitoring
                if 'monitoring' in config:
                    mon_config = config['monitoring']
                    self.monitoring_config = MonitoringConfig(**mon_config)
                
                logger.info("Deployment configuration updated successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to configure deployment: {e}")
            return False
    
    def validate_deployment(self) -> Tuple[bool, List[str]]:
        """Validate deployment configuration"""
        start_time = time.time()
        all_errors = []
        
        try:
            logger.info("Starting deployment validation")
            
            # Validate services
            for service in self.services:
                valid, errors = DeploymentValidator.validate_service_config(service)
                if not valid:
                    all_errors.extend([f"Service {service.name}: {error}" for error in errors])
            
            # Validate database
            valid, errors = DeploymentValidator.validate_database_config(self.database_config)
            if not valid:
                all_errors.extend([f"Database: {error}" for error in errors])
            
            # Validate security
            valid, errors = DeploymentValidator.validate_security_config(self.security_config)
            if not valid:
                all_errors.extend([f"Security: {error}" for error in errors])
            
            # Check service dependencies
            service_names = {service.name for service in self.services}
            for service in self.services:
                for dep in service.dependencies:
                    if dep not in service_names:
                        all_errors.append(f"Service {service.name} depends on unknown service: {dep}")
            
            # Check system requirements
            if not self._check_system_requirements():
                all_errors.append("System requirements not met")
            
            validation_duration = time.time() - start_time
            self.performance_metrics['validation_duration'].append(validation_duration)
            
            if all_errors:
                logger.error(f"Deployment validation failed with {len(all_errors)} errors")
                for error in all_errors:
                    logger.error(f"  - {error}")
            else:
                logger.info(f"Deployment validation passed in {validation_duration:.2f}s")
            
            return len(all_errors) == 0, all_errors
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            all_errors.append(f"Validation exception: {str(e)}")
            return False, all_errors
    
    def deploy(self, strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN) -> DeploymentResult:
        """Deploy the application using specified strategy"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting deployment with {strategy.value} strategy")
            
            # Validate before deployment
            valid, errors = self.validate_deployment()
            if not valid:
                raise ValueError(f"Validation failed: {errors}")
            
            # Get deployment strategy
            if strategy not in self.strategies:
                raise ValueError(f"Unsupported deployment strategy: {strategy}")
            
            deployer = self.strategies[strategy]
            
            # Execute deployment
            result = deployer.deploy(self.services, self.environment)
            
            # Track deployment
            with self.lock:
                self.deployment_history.append(result)
                if result.success:
                    self.current_deployment = result
            
            # Track performance
            deployment_duration = time.time() - start_time
            self.performance_metrics['deployment_duration'].append(deployment_duration)
            
            # Log result
            if result.success:
                logger.info(f"Deployment {result.deployment_id} completed successfully in {deployment_duration:.2f}s")
            else:
                logger.error(f"Deployment {result.deployment_id} failed after {deployment_duration:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            
            # Create failure result
            end_time = datetime.now()
            metrics = DeploymentMetrics(
                deployment_id=f"failed-{int(time.time())}",
                start_time=datetime.now() - timedelta(seconds=time.time() - start_time),
                end_time=end_time,
                status=DeploymentStatus.FAILED
            )
            
            return DeploymentResult(
                success=False,
                deployment_id=metrics.deployment_id,
                environment=self.environment,
                strategy=strategy,
                start_time=metrics.start_time,
                end_time=end_time,
                duration_seconds=time.time() - start_time,
                deployed_services=[],
                failed_services=[s.name for s in self.services],
                metrics=metrics,
                errors=[str(e)]
            )
    
    def rollback(self, deployment_id: Optional[str] = None) -> bool:
        """Rollback to previous deployment"""
        start_time = time.time()
        
        try:
            if not deployment_id and self.current_deployment:
                deployment_id = self.current_deployment.deployment_id
            
            if not deployment_id:
                logger.error("No deployment to rollback")
                return False
            
            logger.info(f"Starting rollback of deployment {deployment_id}")
            
            # Find deployment in history
            deployment = None
            for dep in self.deployment_history:
                if dep.deployment_id == deployment_id:
                    deployment = dep
                    break
            
            if not deployment:
                logger.error(f"Deployment {deployment_id} not found")
                return False
            
            # Execute rollback
            strategy = self.strategies.get(deployment.strategy)
            if not strategy:
                logger.error(f"Strategy {deployment.strategy} not available for rollback")
                return False
            
            success = strategy.rollback(deployment_id)
            
            # Track rollback duration
            rollback_duration = time.time() - start_time
            self.performance_metrics['rollback_duration'].append(rollback_duration)
            
            if success:
                logger.info(f"Rollback completed successfully in {rollback_duration:.2f}s")
            else:
                logger.error(f"Rollback failed after {rollback_duration:.2f}s")
            
            return success
            
        except Exception as e:
            logger.error(f"Rollback error: {e}")
            return False
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        base_config = {
            "environment": self.environment.value,
            "debug": self.environment in [Environment.DEVELOPMENT, Environment.TESTING],
            "services": [service.to_dict() for service in self.services],
            "database": asdict(self.database_config),
            "security": asdict(self.security_config),
            "monitoring": asdict(self.monitoring_config)
        }
        
        # Environment-specific overrides
        if self.environment == Environment.PRODUCTION:
            base_config.update({
                "replicas_multiplier": 3,
                "auto_scaling_enabled": True,
                "backup_enabled": True,
                "monitoring_enhanced": True
            })
        elif self.environment == Environment.STAGING:
            base_config.update({
                "replicas_multiplier": 2,
                "auto_scaling_enabled": True,
                "backup_enabled": True,
                "monitoring_enhanced": False
            })
        elif self.environment == Environment.DEVELOPMENT:
            base_config.update({
                "replicas_multiplier": 1,
                "auto_scaling_enabled": False,
                "backup_enabled": False,
                "monitoring_enhanced": False
            })
        
        return base_config
    
    def save_configuration(self, file_path: Optional[str] = None) -> bool:
        """Save deployment configuration to file"""
        try:
            path = Path(file_path) if file_path else self.config_file_path
            
            config = {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                "environment": self.environment.value,
                "services": [service.to_dict() for service in self.services],
                "database": asdict(self.database_config),
                "security": asdict(self.security_config),
                "monitoring": asdict(self.monitoring_config),
                "deployment_history": [
                    {
                        "deployment_id": result.deployment_id,
                        "success": result.success,
                        "environment": result.environment.value,
                        "strategy": result.strategy.value,
                        "start_time": result.start_time.isoformat(),
                        "end_time": result.end_time.isoformat(),
                        "duration_seconds": result.duration_seconds,
                        "deployed_services": result.deployed_services,
                        "failed_services": result.failed_services
                    }
                    for result in self.deployment_history[-10:]  # Keep last 10 deployments
                ]
            }
            
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Configuration saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load_configuration(self, file_path: Optional[str] = None) -> bool:
        """Load deployment configuration from file"""
        try:
            path = Path(file_path) if file_path else self.config_file_path
            
            if not path.exists():
                logger.error(f"Configuration file not found: {path}")
                return False
            
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load configuration
            self.environment = Environment(config.get('environment', 'development'))
            
            # Configure using loaded data
            return self.configure_deployment(config)
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def get_deployment_status(self, deployment_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of a deployment"""
        if deployment_id:
            # Find specific deployment
            for deployment in self.deployment_history:
                if deployment.deployment_id == deployment_id:
                    return self._deployment_to_status_dict(deployment)
            return {"error": f"Deployment {deployment_id} not found"}
        
        # Return current deployment status
        if self.current_deployment:
            return self._deployment_to_status_dict(self.current_deployment)
        
        return {"status": "No active deployment"}
    
    def _deployment_to_status_dict(self, deployment: DeploymentResult) -> Dict[str, Any]:
        """Convert deployment result to status dictionary"""
        return {
            "deployment_id": deployment.deployment_id,
            "environment": deployment.environment.value,
            "strategy": deployment.strategy.value,
            "status": deployment.metrics.status.value,
            "success": deployment.success,
            "start_time": deployment.start_time.isoformat(),
            "end_time": deployment.end_time.isoformat(),
            "duration_seconds": deployment.duration_seconds,
            "services": {
                "deployed": deployment.deployed_services,
                "failed": deployment.failed_services,
                "total": len(deployment.deployed_services) + len(deployment.failed_services)
            },
            "metrics": {
                "validation_time": deployment.metrics.validation_time_seconds,
                "deployment_time": deployment.metrics.deployment_time_seconds,
                "error_count": deployment.metrics.error_count
            }
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get deployment performance report"""
        import statistics
        
        report = {}
        
        for metric_name, values in self.performance_metrics.items():
            if values:
                report[metric_name] = {
                    "count": len(values),
                    "average": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0
                }
        
        # Add deployment success rate
        if self.deployment_history:
            successful = sum(1 for d in self.deployment_history if d.success)
            report["success_rate"] = {
                "percentage": (successful / len(self.deployment_history)) * 100,
                "successful": successful,
                "failed": len(self.deployment_history) - successful,
                "total": len(self.deployment_history)
            }
        
        return report
    
    def _check_system_requirements(self) -> bool:
        """Check if system meets deployment requirements"""
        try:
            # Check CPU
            cpu_count = psutil.cpu_count()
            if cpu_count < 2:
                logger.warning(f"Low CPU count: {cpu_count}")
                return False
            
            # Check memory
            memory = psutil.virtual_memory()
            if memory.available < 2 * 1024 * 1024 * 1024:  # 2GB
                logger.warning(f"Low available memory: {memory.available / (1024**3):.2f} GB")
                return False
            
            # Check disk space
            disk = psutil.disk_usage('/')
            if disk.free < 10 * 1024 * 1024 * 1024:  # 10GB
                logger.warning(f"Low disk space: {disk.free / (1024**3):.2f} GB")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check system requirements: {e}")
            return False
    
    def prepare_deployment(self) -> bool:
        """Prepare for deployment"""
        try:
            logger.info("Preparing deployment environment")
            
            # Validate configuration
            valid, errors = self.validate_deployment()
            if not valid:
                logger.error(f"Validation failed: {errors}")
                return False
            
            # Check system requirements
            if not self._check_system_requirements():
                logger.error("System requirements not met")
                return False
            
            # Create necessary directories
            dirs_to_create = [
                Path("logs"),
                Path("data"),
                Path("backups"),
                Path("configs")
            ]
            
            for directory in dirs_to_create:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize monitoring
            if self.monitoring_config.enabled:
                logger.info("Initializing monitoring")
                # Monitoring initialization would go here
            
            # Set up security
            if self.security_config.ssl_enabled:
                logger.info("Setting up SSL/TLS")
                # SSL setup would go here
            
            logger.info("Deployment preparation completed")
            return True
            
        except Exception as e:
            logger.error(f"Deployment preparation failed: {e}")
            return False


# Example services for demonstration
def create_example_services() -> List[ServiceConfig]:
    """Create example service configurations"""
    return [
        ServiceConfig(
            name="fraud-api",
            type=ServiceType.API,
            version="1.0.0",
            replicas=3,
            cpu_limit="2000m",
            memory_limit="2Gi",
            ports=[8080, 9090],
            environment_vars={
                "LOG_LEVEL": "INFO",
                "DATABASE_URL": "postgresql://localhost:5432/fraud_db"
            }
        ),
        ServiceConfig(
            name="fraud-worker",
            type=ServiceType.WORKER,
            version="1.0.0",
            replicas=2,
            cpu_limit="1500m",
            memory_limit="1Gi",
            dependencies=["fraud-api"]
        ),
        ServiceConfig(
            name="fraud-cache",
            type=ServiceType.CACHE,
            version="1.0.0",
            replicas=1,
            cpu_limit="500m",
            memory_limit="512Mi",
            ports=[6379]
        )
    ]


if __name__ == "__main__":
    # Example usage of the deployment configuration
    
    # Initialize deployment config
    config = FinancialDeploymentConfig(Environment.STAGING)
    
    # Configure services
    config.services = create_example_services()
    
    # Configure database
    config.database_config = DatabaseConfig(
        type="postgresql",
        host="fraud-db.example.com",
        port=5432,
        database="fraud_detection",
        username="fraud_user",
        ssl_enabled=True,
        backup_enabled=True
    )
    
    # Configure security
    config.security_config = SecurityConfig(
        ssl_enabled=True,
        api_key_enabled=True,
        rate_limiting_enabled=True,
        rate_limit_requests=1000,
        rate_limit_window=3600,
        audit_logging_enabled=True
    )
    
    # Validate deployment
    print("\nValidating deployment configuration...")
    valid, errors = config.validate_deployment()
    if valid:
        print("✓ Deployment validation passed")
    else:
        print("✗ Deployment validation failed:")
        for error in errors:
            print(f"  - {error}")
    
    # Save configuration
    print("\nSaving deployment configuration...")
    if config.save_configuration("example_deployment.yaml"):
        print("✓ Configuration saved successfully")
    
    # Prepare deployment
    print("\nPreparing deployment...")
    if config.prepare_deployment():
        print("✓ Deployment preparation completed")
    
    # Simulate deployment
    print("\nSimulating deployment...")
    result = config.deploy(strategy=DeploymentStrategy.BLUE_GREEN)
    
    print(f"\nDeployment Result:")
    print(f"  Deployment ID: {result.deployment_id}")
    print(f"  Success: {result.success}")
    print(f"  Duration: {result.duration_seconds:.2f} seconds")
    print(f"  Deployed Services: {', '.join(result.deployed_services)}")
    
    if result.failed_services:
        print(f"  Failed Services: {', '.join(result.failed_services)}")
    
    # Get deployment status
    print("\nDeployment Status:")
    status = config.get_deployment_status()
    print(json.dumps(status, indent=2))
    
    # Get performance report
    print("\nPerformance Report:")
    report = config.get_performance_report()
    print(json.dumps(report, indent=2))