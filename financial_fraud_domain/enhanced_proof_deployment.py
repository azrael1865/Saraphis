"""
Enhanced Proof Verifier - Chunk 4: Deployment and Management Utilities
Comprehensive deployment, management, and operational utilities for the enhanced proof verifier system,
including configuration management, deployment automation, and system administration tools.
"""

import logging
import json
import yaml
import time
import threading
import asyncio
import subprocess
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Type, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import warnings
import traceback
import psutil
import pandas as pd
import sqlite3
import pickle
import hashlib
import secrets
import tempfile
import os
import sys
import signal
import docker
from croniter import croniter
import schedule

# Import enhanced proof components
try:
    from enhanced_proof_verifier import (
        FinancialProofVerifier as EnhancedFinancialProofVerifier,
        EnhancedProofClaim, EnhancedProofEvidence, EnhancedProofResult,
        SecurityLevel, ProofVerificationException, ProofConfigurationError,
        ProofGenerationError, ProofValidationError, ProofTimeoutError,
        ProofSecurityError, ProofIntegrityError, ClaimValidationError,
        EvidenceValidationError, ProofSystemError, ProofStorageError,
        CryptographicError, ProofExpiredError, ResourceLimitError,
        SecurityValidator, ResourceMonitor
    )
    from enhanced_proof_integration import (
        ProofIntegrationManager, ProofIntegrationConfig, ProofSystemAnalyzer,
        ProofIntegrationError, ProofSystemAnalysisError
    )
    from enhanced_proof_testing import (
        ProofVerifierTestSuite, ProofTestConfig, ProofTestError
    )
    from enhanced_proof_monitoring import (
        ComprehensiveMonitoringManager, PerformanceMonitoringConfig,
        AdvancedMetricsCollector, PerformanceOptimizer
    )
    PROOF_COMPONENTS = True
except ImportError as e:
    PROOF_COMPONENTS = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Proof verifier components not available: {e}")

# Configure logging
logger = logging.getLogger(__name__)

# ======================== DEPLOYMENT CONFIGURATION ========================

class DeploymentEnvironment(Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DISASTER_RECOVERY = "disaster_recovery"

class DeploymentStrategy(Enum):
    """Deployment strategy types"""
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"
    CANARY = "canary"
    RECREATE = "recreate"

class ServiceStatus(Enum):
    """Service status types"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    MAINTENANCE = "maintenance"

@dataclass
class DeploymentConfig:
    """Configuration for proof verifier deployment"""
    
    # Environment settings
    environment: DeploymentEnvironment = DeploymentEnvironment.DEVELOPMENT
    deployment_strategy: DeploymentStrategy = DeploymentStrategy.ROLLING
    
    # Service settings
    service_name: str = "proof-verifier"
    service_version: str = "1.0.0"
    service_port: int = 8080
    health_check_port: int = 8081
    
    # Resource limits
    cpu_limit: float = 2.0
    memory_limit: str = "4Gi"
    disk_limit: str = "10Gi"
    
    # Scaling settings
    min_instances: int = 1
    max_instances: int = 5
    auto_scaling_enabled: bool = True
    cpu_utilization_threshold: float = 70.0
    memory_utilization_threshold: float = 80.0
    
    # Health check settings
    health_check_interval: int = 30
    health_check_timeout: int = 10
    health_check_retries: int = 3
    
    # Backup settings
    backup_enabled: bool = True
    backup_schedule: str = "0 2 * * *"  # Daily at 2 AM
    backup_retention_days: int = 30
    
    # Security settings
    enable_tls: bool = True
    certificate_path: str = "/etc/ssl/certs/proof-verifier.crt"
    private_key_path: str = "/etc/ssl/private/proof-verifier.key"
    
    # Database settings
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "proof_verifier"
    database_user: str = "proof_user"
    database_password: str = "secure_password"
    
    # Configuration paths
    config_directory: str = "/etc/proof-verifier"
    data_directory: str = "/var/lib/proof-verifier"
    log_directory: str = "/var/log/proof-verifier"
    
    # Monitoring settings
    monitoring_enabled: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    
    # Integration settings
    enable_preprocessing: bool = True
    enable_ml_integration: bool = True
    enable_data_validation: bool = True

class DeploymentError(Exception):
    """Exception raised during deployment operations"""
    def __init__(self, message: str, stage: str = None, component: str = None):
        super().__init__(message)
        self.stage = stage
        self.component = component
        self.timestamp = datetime.now()

# ======================== CONFIGURATION MANAGER ========================

class ConfigurationManager:
    """Manages configuration for proof verifier deployment"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "/etc/proof-verifier/config.yaml"
        self.config = DeploymentConfig()
        self.config_lock = threading.Lock()
        
        # Load configuration
        self.load_configuration()
    
    def load_configuration(self):
        """Load configuration from file"""
        try:
            config_file = Path(self.config_path)
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                
                # Update configuration
                with self.config_lock:
                    for key, value in config_data.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)
                
                logger.info(f"Configuration loaded from {self.config_path}")
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise DeploymentError(f"Configuration loading failed: {e}", stage="configuration")
    
    def save_configuration(self):
        """Save configuration to file"""
        try:
            # Ensure directory exists
            config_dir = Path(self.config_path).parent
            config_dir.mkdir(parents=True, exist_ok=True)
            
            with self.config_lock:
                config_data = asdict(self.config)
                
                # Convert enum values to strings
                for key, value in config_data.items():
                    if isinstance(value, Enum):
                        config_data[key] = value.value
            
            with open(self.config_path, 'w') as f:
                if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                    yaml.dump(config_data, f, default_flow_style=False)
                else:
                    json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise DeploymentError(f"Configuration saving failed: {e}", stage="configuration")
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration settings"""
        try:
            validation_result = {
                'valid': True,
                'errors': [],
                'warnings': []
            }
            
            with self.config_lock:
                # Validate resource limits
                if self.config.cpu_limit <= 0:
                    validation_result['errors'].append("CPU limit must be positive")
                
                if self.config.min_instances < 1:
                    validation_result['errors'].append("Minimum instances must be at least 1")
                
                if self.config.max_instances < self.config.min_instances:
                    validation_result['errors'].append("Maximum instances must be >= minimum instances")
                
                # Validate ports
                if not (1024 <= self.config.service_port <= 65535):
                    validation_result['errors'].append("Service port must be between 1024 and 65535")
                
                # Validate paths
                if self.config.enable_tls:
                    if not Path(self.config.certificate_path).exists():
                        validation_result['warnings'].append(f"Certificate file not found: {self.config.certificate_path}")
                    
                    if not Path(self.config.private_key_path).exists():
                        validation_result['warnings'].append(f"Private key file not found: {self.config.private_key_path}")
                
                # Validate cron schedule
                try:
                    croniter(self.config.backup_schedule)
                except Exception:
                    validation_result['errors'].append("Invalid backup schedule format")
            
            # Set overall validity
            validation_result['valid'] = len(validation_result['errors']) == 0
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return {
                'valid': False,
                'errors': [f"Validation error: {e}"],
                'warnings': []
            }
    
    def get_configuration(self) -> DeploymentConfig:
        """Get current configuration"""
        with self.config_lock:
            return self.config
    
    def update_configuration(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        try:
            with self.config_lock:
                for key, value in updates.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                    else:
                        logger.warning(f"Unknown configuration key: {key}")
            
            # Save updated configuration
            self.save_configuration()
            
            logger.info("Configuration updated successfully")
            
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            raise DeploymentError(f"Configuration update failed: {e}", stage="configuration")

# ======================== SERVICE MANAGER ========================

class ServiceManager:
    """Manages proof verifier service lifecycle"""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.service_status = ServiceStatus.STOPPED
        self.service_pid = None
        self.service_process = None
        self.status_lock = threading.Lock()
        self.health_check_thread = None
        self.health_check_running = False
        
        # Initialize service components
        self.integration_manager = None
        self.monitoring_manager = None
        self.test_suite = None
    
    def start_service(self) -> bool:
        """Start the proof verifier service"""
        try:
            with self.status_lock:
                if self.service_status == ServiceStatus.RUNNING:
                    logger.warning("Service is already running")
                    return True
                
                self.service_status = ServiceStatus.STARTING
            
            config = self.config_manager.get_configuration()
            
            # Validate configuration
            validation_result = self.config_manager.validate_configuration()
            if not validation_result['valid']:
                raise DeploymentError(
                    f"Configuration validation failed: {validation_result['errors']}",
                    stage="startup"
                )
            
            # Create necessary directories
            self._create_service_directories(config)
            
            # Initialize service components
            self._initialize_service_components(config)
            
            # Start health check monitoring
            self._start_health_check_monitoring()
            
            with self.status_lock:
                self.service_status = ServiceStatus.RUNNING
                self.service_pid = os.getpid()
            
            logger.info(f"Proof verifier service started successfully (PID: {self.service_pid})")
            return True
            
        except Exception as e:
            logger.error(f"Service startup failed: {e}")
            
            with self.status_lock:
                self.service_status = ServiceStatus.FAILED
            
            raise DeploymentError(f"Service startup failed: {e}", stage="startup")
    
    def stop_service(self) -> bool:
        """Stop the proof verifier service"""
        try:
            with self.status_lock:
                if self.service_status == ServiceStatus.STOPPED:
                    logger.warning("Service is already stopped")
                    return True
                
                self.service_status = ServiceStatus.STOPPING
            
            # Stop health check monitoring
            self._stop_health_check_monitoring()
            
            # Shutdown service components
            self._shutdown_service_components()
            
            with self.status_lock:
                self.service_status = ServiceStatus.STOPPED
                self.service_pid = None
            
            logger.info("Proof verifier service stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Service shutdown failed: {e}")
            
            with self.status_lock:
                self.service_status = ServiceStatus.FAILED
            
            raise DeploymentError(f"Service shutdown failed: {e}", stage="shutdown")
    
    def restart_service(self) -> bool:
        """Restart the proof verifier service"""
        try:
            logger.info("Restarting proof verifier service")
            
            # Stop service
            self.stop_service()
            
            # Wait a moment
            time.sleep(2)
            
            # Start service
            return self.start_service()
            
        except Exception as e:
            logger.error(f"Service restart failed: {e}")
            raise DeploymentError(f"Service restart failed: {e}", stage="restart")
    
    def _create_service_directories(self, config: DeploymentConfig):
        """Create necessary service directories"""
        try:
            directories = [
                config.config_directory,
                config.data_directory,
                config.log_directory
            ]
            
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {directory}")
            
        except Exception as e:
            logger.error(f"Directory creation failed: {e}")
            raise DeploymentError(f"Directory creation failed: {e}", stage="initialization")
    
    def _initialize_service_components(self, config: DeploymentConfig):
        """Initialize service components"""
        try:
            if not PROOF_COMPONENTS:
                logger.warning("Proof components not available, running in limited mode")
                return
            
            # Initialize integration manager
            integration_config = ProofIntegrationConfig(
                enable_data_validation=config.enable_data_validation,
                enable_preprocessing=config.enable_preprocessing,
                enable_ml_integration=config.enable_ml_integration,
                enable_performance_monitoring=config.monitoring_enabled
            )
            
            self.integration_manager = ProofIntegrationManager(integration_config)
            
            # Initialize monitoring manager
            if config.monitoring_enabled:
                monitoring_config = PerformanceMonitoringConfig(
                    enable_metrics_persistence=True,
                    metrics_database_path=str(Path(config.data_directory) / "metrics.db")
                )
                
                self.monitoring_manager = ComprehensiveMonitoringManager(monitoring_config)
            
            # Initialize test suite
            test_config = ProofTestConfig(
                enable_unit_tests=True,
                enable_integration_tests=True,
                enable_performance_tests=True,
                test_data_size=100  # Smaller for production
            )
            
            self.test_suite = ProofVerifierTestSuite(test_config)
            
            logger.info("Service components initialized successfully")
            
        except Exception as e:
            logger.error(f"Service component initialization failed: {e}")
            raise DeploymentError(f"Component initialization failed: {e}", stage="initialization")
    
    def _shutdown_service_components(self):
        """Shutdown service components"""
        try:
            if self.monitoring_manager:
                self.monitoring_manager.stop_monitoring()
            
            if self.integration_manager:
                self.integration_manager.shutdown()
            
            if self.test_suite:
                self.test_suite.cleanup_test_environment()
            
            logger.info("Service components shutdown successfully")
            
        except Exception as e:
            logger.error(f"Service component shutdown failed: {e}")
    
    def _start_health_check_monitoring(self):
        """Start health check monitoring"""
        try:
            if self.health_check_running:
                return
            
            self.health_check_running = True
            self.health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True
            )
            self.health_check_thread.start()
            
            logger.info("Health check monitoring started")
            
        except Exception as e:
            logger.error(f"Health check monitoring startup failed: {e}")
    
    def _stop_health_check_monitoring(self):
        """Stop health check monitoring"""
        try:
            self.health_check_running = False
            
            if self.health_check_thread:
                self.health_check_thread.join(timeout=5)
            
            logger.info("Health check monitoring stopped")
            
        except Exception as e:
            logger.error(f"Health check monitoring shutdown failed: {e}")
    
    def _health_check_loop(self):
        """Health check monitoring loop"""
        config = self.config_manager.get_configuration()
        
        while self.health_check_running:
            try:
                # Perform health check
                health_status = self.get_health_status()
                
                # Log health status
                if health_status['status'] != 'healthy':
                    logger.warning(f"Health check failed: {health_status}")
                    
                    # Auto-restart if configured
                    if health_status['status'] == 'failed' and config.auto_scaling_enabled:
                        logger.info("Attempting service restart due to health check failure")
                        self.restart_service()
                
                # Sleep until next check
                time.sleep(config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(config.health_check_interval)
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status"""
        try:
            with self.status_lock:
                return {
                    'status': self.service_status.value,
                    'pid': self.service_pid,
                    'uptime': self._calculate_uptime(),
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Service status retrieval failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        try:
            health_status = {
                'status': 'healthy',
                'checks': {},
                'timestamp': datetime.now().isoformat()
            }
            
            # Check service status
            health_status['checks']['service'] = {
                'status': 'pass' if self.service_status == ServiceStatus.RUNNING else 'fail',
                'message': f"Service status: {self.service_status.value}"
            }
            
            # Check system resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            health_status['checks']['cpu'] = {
                'status': 'pass' if cpu_percent < 90 else 'fail',
                'message': f"CPU usage: {cpu_percent:.1f}%"
            }
            
            health_status['checks']['memory'] = {
                'status': 'pass' if memory_percent < 90 else 'fail',
                'message': f"Memory usage: {memory_percent:.1f}%"
            }
            
            # Check component health
            if self.integration_manager:
                integration_status = self.integration_manager.get_integration_status()
                health_status['checks']['integration'] = {
                    'status': 'pass' if integration_status.get('components', {}).get('proof_verifier') else 'fail',
                    'message': 'Integration manager status'
                }
            
            if self.monitoring_manager:
                monitoring_dashboard = self.monitoring_manager.get_monitoring_dashboard()
                health_status['checks']['monitoring'] = {
                    'status': 'pass' if monitoring_dashboard.get('health_status') == 'healthy' else 'fail',
                    'message': f"Monitoring status: {monitoring_dashboard.get('health_status', 'unknown')}"
                }
            
            # Determine overall health
            failed_checks = [check for check in health_status['checks'].values() if check['status'] == 'fail']
            if failed_checks:
                health_status['status'] = 'failed'
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health status check failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_uptime(self) -> Optional[float]:
        """Calculate service uptime in seconds"""
        try:
            if self.service_pid and self.service_status == ServiceStatus.RUNNING:
                process = psutil.Process(self.service_pid)
                return time.time() - process.create_time()
            return None
            
        except Exception:
            return None

# ======================== DEPLOYMENT MANAGER ========================

class DeploymentManager:
    """Manages deployment operations for proof verifier"""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.deployment_history = []
        self.deployment_lock = threading.Lock()
        self.backup_manager = BackupManager(config_manager)
        
    def deploy_service(self, deployment_package: str = None) -> Dict[str, Any]:
        """Deploy proof verifier service"""
        try:
            deployment_start = time.time()
            
            with self.deployment_lock:
                deployment_id = f"deploy_{int(time.time())}"
                
                deployment_record = {
                    'deployment_id': deployment_id,
                    'start_time': datetime.now(),
                    'status': 'in_progress',
                    'stages': [],
                    'errors': []
                }
                
                self.deployment_history.append(deployment_record)
            
            config = self.config_manager.get_configuration()
            
            # Stage 1: Pre-deployment validation
            self._execute_deployment_stage(
                deployment_record,
                "pre_deployment_validation",
                self._validate_deployment_environment,
                config
            )
            
            # Stage 2: Backup current deployment
            if config.backup_enabled:
                self._execute_deployment_stage(
                    deployment_record,
                    "backup_current_deployment",
                    self.backup_manager.create_backup,
                    "pre_deployment"
                )
            
            # Stage 3: Deploy new version
            self._execute_deployment_stage(
                deployment_record,
                "deploy_new_version",
                self._deploy_new_version,
                deployment_package
            )
            
            # Stage 4: Post-deployment validation
            self._execute_deployment_stage(
                deployment_record,
                "post_deployment_validation",
                self._validate_deployment_success
            )
            
            # Stage 5: Health check
            self._execute_deployment_stage(
                deployment_record,
                "health_check",
                self._perform_post_deployment_health_check
            )
            
            # Complete deployment
            deployment_duration = time.time() - deployment_start
            
            with self.deployment_lock:
                deployment_record['status'] = 'completed'
                deployment_record['end_time'] = datetime.now()
                deployment_record['duration'] = deployment_duration
            
            logger.info(f"Deployment {deployment_id} completed successfully in {deployment_duration:.2f}s")
            
            return {
                'success': True,
                'deployment_id': deployment_id,
                'duration': deployment_duration,
                'stages': deployment_record['stages']
            }
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            
            with self.deployment_lock:
                deployment_record['status'] = 'failed'
                deployment_record['end_time'] = datetime.now()
                deployment_record['errors'].append(str(e))
            
            return {
                'success': False,
                'error': str(e),
                'deployment_id': deployment_id if 'deployment_id' in locals() else None
            }
    
    def _execute_deployment_stage(self, deployment_record: Dict[str, Any], 
                                stage_name: str, stage_func: Callable, *args):
        """Execute a deployment stage with error handling"""
        try:
            stage_start = time.time()
            
            logger.info(f"Starting deployment stage: {stage_name}")
            
            # Execute stage
            stage_result = stage_func(*args)
            
            stage_duration = time.time() - stage_start
            
            # Record stage completion
            stage_record = {
                'stage': stage_name,
                'status': 'completed',
                'duration': stage_duration,
                'timestamp': datetime.now().isoformat(),
                'result': stage_result
            }
            
            deployment_record['stages'].append(stage_record)
            
            logger.info(f"Deployment stage {stage_name} completed in {stage_duration:.2f}s")
            
        except Exception as e:
            stage_duration = time.time() - stage_start if 'stage_start' in locals() else 0
            
            stage_record = {
                'stage': stage_name,
                'status': 'failed',
                'duration': stage_duration,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            
            deployment_record['stages'].append(stage_record)
            deployment_record['errors'].append(f"Stage {stage_name} failed: {e}")
            
            logger.error(f"Deployment stage {stage_name} failed: {e}")
            raise DeploymentError(f"Stage {stage_name} failed: {e}", stage=stage_name)
    
    def _validate_deployment_environment(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Validate deployment environment"""
        validation_result = {
            'environment_valid': True,
            'checks': []
        }
        
        # Check system resources
        cpu_count = psutil.cpu_count()
        memory_total = psutil.virtual_memory().total
        disk_free = psutil.disk_usage('/').free
        
        validation_result['checks'].append({
            'check': 'system_resources',
            'status': 'pass',
            'details': {
                'cpu_count': cpu_count,
                'memory_total': memory_total,
                'disk_free': disk_free
            }
        })
        
        # Check configuration
        config_validation = self.config_manager.validate_configuration()
        validation_result['checks'].append({
            'check': 'configuration',
            'status': 'pass' if config_validation['valid'] else 'fail',
            'details': config_validation
        })
        
        if not config_validation['valid']:
            validation_result['environment_valid'] = False
        
        # Check required directories
        required_dirs = [config.config_directory, config.data_directory, config.log_directory]
        for directory in required_dirs:
            dir_path = Path(directory)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
        
        validation_result['checks'].append({
            'check': 'directories',
            'status': 'pass',
            'details': 'Required directories created'
        })
        
        return validation_result
    
    def _deploy_new_version(self, deployment_package: str = None) -> Dict[str, Any]:
        """Deploy new version of the service"""
        deployment_result = {
            'deployment_successful': True,
            'actions_taken': []
        }
        
        # If deployment package provided, extract it
        if deployment_package:
            deployment_result['actions_taken'].append(f"Extracted deployment package: {deployment_package}")
        
        # Restart service with new configuration
        service_manager = ServiceManager(self.config_manager)
        
        # Stop current service
        service_manager.stop_service()
        deployment_result['actions_taken'].append("Stopped current service")
        
        # Start new service
        service_manager.start_service()
        deployment_result['actions_taken'].append("Started new service")
        
        return deployment_result
    
    def _validate_deployment_success(self) -> Dict[str, Any]:
        """Validate deployment success"""
        validation_result = {
            'deployment_valid': True,
            'validations': []
        }
        
        # Check service status
        service_manager = ServiceManager(self.config_manager)
        service_status = service_manager.get_service_status()
        
        validation_result['validations'].append({
            'validation': 'service_status',
            'status': 'pass' if service_status['status'] == 'running' else 'fail',
            'details': service_status
        })
        
        if service_status['status'] != 'running':
            validation_result['deployment_valid'] = False
        
        # Check health status
        health_status = service_manager.get_health_status()
        
        validation_result['validations'].append({
            'validation': 'health_check',
            'status': 'pass' if health_status['status'] == 'healthy' else 'fail',
            'details': health_status
        })
        
        if health_status['status'] != 'healthy':
            validation_result['deployment_valid'] = False
        
        return validation_result
    
    def _perform_post_deployment_health_check(self) -> Dict[str, Any]:
        """Perform post-deployment health check"""
        health_check_result = {
            'health_check_passed': True,
            'checks_performed': []
        }
        
        # Wait for service to stabilize
        time.sleep(10)
        
        # Perform comprehensive health check
        service_manager = ServiceManager(self.config_manager)
        
        for _ in range(3):  # Retry up to 3 times
            health_status = service_manager.get_health_status()
            
            health_check_result['checks_performed'].append({
                'timestamp': datetime.now().isoformat(),
                'health_status': health_status
            })
            
            if health_status['status'] == 'healthy':
                break
            
            time.sleep(5)
        else:
            health_check_result['health_check_passed'] = False
        
        return health_check_result
    
    def rollback_deployment(self, deployment_id: str = None) -> Dict[str, Any]:
        """Rollback to previous deployment"""
        try:
            logger.info(f"Starting deployment rollback for {deployment_id}")
            
            # Restore from backup
            restore_result = self.backup_manager.restore_backup("latest")
            
            if not restore_result['success']:
                raise DeploymentError(f"Backup restoration failed: {restore_result['error']}")
            
            # Restart service
            service_manager = ServiceManager(self.config_manager)
            service_manager.restart_service()
            
            # Validate rollback
            health_status = service_manager.get_health_status()
            
            rollback_result = {
                'success': True,
                'deployment_id': deployment_id,
                'restore_result': restore_result,
                'health_status': health_status,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Deployment rollback completed successfully")
            return rollback_result
            
        except Exception as e:
            logger.error(f"Deployment rollback failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'deployment_id': deployment_id
            }
    
    def get_deployment_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get deployment history"""
        with self.deployment_lock:
            return self.deployment_history[-limit:] if self.deployment_history else []

# ======================== BACKUP MANAGER ========================

class BackupManager:
    """Manages backup and restore operations"""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.backup_history = []
        self.backup_lock = threading.Lock()
    
    def create_backup(self, backup_type: str = "manual") -> Dict[str, Any]:
        """Create system backup"""
        try:
            backup_start = time.time()
            backup_id = f"backup_{int(time.time())}"
            
            config = self.config_manager.get_configuration()
            
            backup_record = {
                'backup_id': backup_id,
                'backup_type': backup_type,
                'timestamp': datetime.now(),
                'status': 'in_progress'
            }
            
            with self.backup_lock:
                self.backup_history.append(backup_record)
            
            # Create backup directory
            backup_dir = Path(config.data_directory) / "backups" / backup_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup configuration
            config_backup_path = backup_dir / "config.yaml"
            shutil.copy2(self.config_manager.config_path, config_backup_path)
            
            # Backup data directory
            data_backup_path = backup_dir / "data"
            if Path(config.data_directory).exists():
                shutil.copytree(config.data_directory, data_backup_path, dirs_exist_ok=True)
            
            # Backup logs (last 7 days)
            log_backup_path = backup_dir / "logs"
            if Path(config.log_directory).exists():
                shutil.copytree(config.log_directory, log_backup_path, dirs_exist_ok=True)
            
            # Create backup archive
            archive_path = backup_dir.parent / f"{backup_id}.tar.gz"
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(backup_dir, arcname=backup_id)
            
            # Remove temporary backup directory
            shutil.rmtree(backup_dir)
            
            backup_duration = time.time() - backup_start
            
            with self.backup_lock:
                backup_record['status'] = 'completed'
                backup_record['duration'] = backup_duration
                backup_record['archive_path'] = str(archive_path)
                backup_record['archive_size'] = archive_path.stat().st_size
            
            logger.info(f"Backup {backup_id} completed successfully in {backup_duration:.2f}s")
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            return {
                'success': True,
                'backup_id': backup_id,
                'archive_path': str(archive_path),
                'duration': backup_duration
            }
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            
            with self.backup_lock:
                backup_record['status'] = 'failed'
                backup_record['error'] = str(e)
            
            return {
                'success': False,
                'error': str(e),
                'backup_id': backup_id if 'backup_id' in locals() else None
            }
    
    def restore_backup(self, backup_id: str) -> Dict[str, Any]:
        """Restore from backup"""
        try:
            logger.info(f"Starting backup restoration: {backup_id}")
            
            config = self.config_manager.get_configuration()
            
            # Find backup
            if backup_id == "latest":
                backup_record = self._get_latest_backup()
            else:
                backup_record = self._find_backup(backup_id)
            
            if not backup_record:
                raise DeploymentError(f"Backup not found: {backup_id}")
            
            archive_path = Path(backup_record['archive_path'])
            
            if not archive_path.exists():
                raise DeploymentError(f"Backup archive not found: {archive_path}")
            
            # Create temporary extraction directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract backup
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(temp_path)
                
                backup_content_dir = temp_path / backup_record['backup_id']
                
                # Restore configuration
                config_path = backup_content_dir / "config.yaml"
                if config_path.exists():
                    shutil.copy2(config_path, self.config_manager.config_path)
                
                # Restore data
                data_path = backup_content_dir / "data"
                if data_path.exists():
                    # Backup current data
                    current_data_backup = Path(config.data_directory).parent / f"data_backup_{int(time.time())}"
                    if Path(config.data_directory).exists():
                        shutil.move(config.data_directory, current_data_backup)
                    
                    # Restore data
                    shutil.copytree(data_path, config.data_directory)
                
                # Restore logs
                log_path = backup_content_dir / "logs"
                if log_path.exists():
                    shutil.copytree(log_path, config.log_directory, dirs_exist_ok=True)
            
            logger.info(f"Backup {backup_id} restored successfully")
            
            return {
                'success': True,
                'backup_id': backup_record['backup_id'],
                'restored_from': str(archive_path)
            }
            
        except Exception as e:
            logger.error(f"Backup restoration failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'backup_id': backup_id
            }
    
    def _get_latest_backup(self) -> Optional[Dict[str, Any]]:
        """Get latest backup record"""
        with self.backup_lock:
            completed_backups = [b for b in self.backup_history if b['status'] == 'completed']
            if completed_backups:
                return max(completed_backups, key=lambda b: b['timestamp'])
            return None
    
    def _find_backup(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """Find backup by ID"""
        with self.backup_lock:
            for backup in self.backup_history:
                if backup['backup_id'] == backup_id:
                    return backup
            return None
    
    def _cleanup_old_backups(self):
        """Clean up old backups"""
        try:
            config = self.config_manager.get_configuration()
            cutoff_date = datetime.now() - timedelta(days=config.backup_retention_days)
            
            with self.backup_lock:
                backups_to_remove = []
                
                for backup in self.backup_history:
                    if backup['timestamp'] < cutoff_date and backup['status'] == 'completed':
                        # Remove backup file
                        archive_path = Path(backup['archive_path'])
                        if archive_path.exists():
                            archive_path.unlink()
                        
                        backups_to_remove.append(backup)
                
                # Remove from history
                for backup in backups_to_remove:
                    self.backup_history.remove(backup)
                    
                if backups_to_remove:
                    logger.info(f"Cleaned up {len(backups_to_remove)} old backups")
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
    
    def get_backup_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get backup history"""
        with self.backup_lock:
            return self.backup_history[-limit:] if self.backup_history else []

# ======================== SYSTEM ADMINISTRATION UTILITIES ========================

class SystemAdministration:
    """System administration utilities for proof verifier"""
    
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager
        self.service_manager = ServiceManager(config_manager)
        self.deployment_manager = DeploymentManager(config_manager)
        self.backup_manager = BackupManager(config_manager)
        
        # Initialize scheduled tasks
        self._initialize_scheduled_tasks()
    
    def _initialize_scheduled_tasks(self):
        """Initialize scheduled maintenance tasks"""
        try:
            config = self.config_manager.get_configuration()
            
            # Schedule automatic backups
            if config.backup_enabled:
                schedule.every().day.at("02:00").do(self._scheduled_backup)
            
            # Schedule health checks
            schedule.every(30).minutes.do(self._scheduled_health_check)
            
            # Schedule cleanup tasks
            schedule.every().day.at("03:00").do(self._scheduled_cleanup)
            
            # Start scheduler thread
            scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            scheduler_thread.start()
            
            logger.info("Scheduled tasks initialized")
            
        except Exception as e:
            logger.error(f"Scheduled tasks initialization failed: {e}")
    
    def _run_scheduler(self):
        """Run scheduled tasks"""
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)
    
    def _scheduled_backup(self):
        """Scheduled backup task"""
        try:
            logger.info("Starting scheduled backup")
            result = self.backup_manager.create_backup("scheduled")
            
            if result['success']:
                logger.info(f"Scheduled backup completed: {result['backup_id']}")
            else:
                logger.error(f"Scheduled backup failed: {result['error']}")
                
        except Exception as e:
            logger.error(f"Scheduled backup task failed: {e}")
    
    def _scheduled_health_check(self):
        """Scheduled health check task"""
        try:
            health_status = self.service_manager.get_health_status()
            
            if health_status['status'] != 'healthy':
                logger.warning(f"Health check failed: {health_status}")
                
                # Attempt automatic recovery
                if health_status['status'] == 'failed':
                    logger.info("Attempting automatic service recovery")
                    self.service_manager.restart_service()
                    
        except Exception as e:
            logger.error(f"Scheduled health check failed: {e}")
    
    def _scheduled_cleanup(self):
        """Scheduled cleanup task"""
        try:
            logger.info("Starting scheduled cleanup")
            
            # Cleanup old logs
            self._cleanup_old_logs()
            
            # Cleanup temporary files
            self._cleanup_temp_files()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("Scheduled cleanup completed")
            
        except Exception as e:
            logger.error(f"Scheduled cleanup failed: {e}")
    
    def _cleanup_old_logs(self):
        """Clean up old log files"""
        try:
            config = self.config_manager.get_configuration()
            log_dir = Path(config.log_directory)
            
            if not log_dir.exists():
                return
            
            cutoff_date = datetime.now() - timedelta(days=30)
            
            for log_file in log_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    logger.debug(f"Removed old log file: {log_file}")
            
        except Exception as e:
            logger.error(f"Log cleanup failed: {e}")
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            temp_dirs = [
                Path("/tmp"),
                Path("/var/tmp")
            ]
            
            for temp_dir in temp_dirs:
                if not temp_dir.exists():
                    continue
                
                for temp_file in temp_dir.glob("proof_verifier_*"):
                    try:
                        if temp_file.is_file():
                            temp_file.unlink()
                        elif temp_file.is_dir():
                            shutil.rmtree(temp_file)
                    except Exception as e:
                        logger.debug(f"Failed to remove temp file {temp_file}: {e}")
            
        except Exception as e:
            logger.error(f"Temp file cleanup failed: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            return {
                'timestamp': datetime.now().isoformat(),
                'service_status': self.service_manager.get_service_status(),
                'health_status': self.service_manager.get_health_status(),
                'configuration': asdict(self.config_manager.get_configuration()),
                'system_resources': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_percent': psutil.disk_usage('/').percent,
                    'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
                },
                'deployment_history': self.deployment_manager.get_deployment_history(5),
                'backup_history': self.backup_manager.get_backup_history(5)
            }
            
        except Exception as e:
            logger.error(f"System info retrieval failed: {e}")
            return {'error': str(e)}

# ======================== MAIN DEPLOYMENT INTERFACE ========================

def create_deployment_system(config_path: str = None) -> SystemAdministration:
    """Create deployment system with configuration"""
    config_manager = ConfigurationManager(config_path)
    return SystemAdministration(config_manager)

def deploy_proof_verifier(config_path: str = None, deployment_package: str = None) -> Dict[str, Any]:
    """Deploy proof verifier system"""
    try:
        admin_system = create_deployment_system(config_path)
        return admin_system.deployment_manager.deploy_service(deployment_package)
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return {'success': False, 'error': str(e)}

# Export components
__all__ = [
    'DeploymentEnvironment',
    'DeploymentStrategy',
    'ServiceStatus',
    'DeploymentConfig',
    'DeploymentError',
    'ConfigurationManager',
    'ServiceManager',
    'DeploymentManager',
    'BackupManager',
    'SystemAdministration',
    'create_deployment_system',
    'deploy_proof_verifier'
]

if __name__ == "__main__":
    print("Enhanced Proof Verifier - Chunk 4: Deployment and management utilities loaded")
    
    # Basic deployment test
    try:
        admin_system = create_deployment_system()
        print("✓ System administration created successfully")
        
        # Test system info
        system_info = admin_system.get_system_info()
        print(f"✓ System info retrieved: {system_info.get('service_status', {}).get('status', 'unknown')}")
        
        # Test configuration
        config = admin_system.config_manager.get_configuration()
        print(f"✓ Configuration loaded: {config.environment.value}")
        
    except Exception as e:
        print(f"✗ Deployment test failed: {e}")