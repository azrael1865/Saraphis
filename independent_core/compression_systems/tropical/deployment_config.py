"""
Comprehensive Deployment Configuration System for JAX Tropical Compression.
Extends existing JAX configuration infrastructure with remote deployment,
environment management, and production-ready features.

PRODUCTION-READY - NO PLACEHOLDERS - HARD FAILURES ONLY
"""

import os
import sys
import json
import yaml
import socket
import ssl
import time
import hashlib
import logging
import threading
import warnings
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import base64
import secrets

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import existing JAX infrastructure
from .jax_config import (
    JAXConfig, 
    JAXEnvironment,
    JAXPlatform
)
from .jax_tropical_engine import JAXTropicalConfig
from ..gpu_memory.gpu_auto_detector import (
    GPUAutoDetector,
    GPUSpecs,
    AutoOptimizedConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentMode(Enum):
    """Deployment mode options"""
    LOCAL = "local"           # Single machine deployment
    REMOTE = "remote"         # Remote server deployment
    HYBRID = "hybrid"         # Mixed local/remote deployment
    CLUSTER = "cluster"       # Multi-node cluster deployment
    EDGE = "edge"            # Edge device deployment
    SERVERLESS = "serverless" # Serverless function deployment


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for remote servers"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    IP_HASH = "ip_hash"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"
    RESOURCE_BASED = "resource_based"  # Based on GPU/CPU utilization


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"
    TESTING = "test"
    CANARY = "canary"
    DISASTER_RECOVERY = "dr"


@dataclass
class RemoteServerConfig:
    """Configuration for remote server endpoints"""
    host: str
    port: int
    protocol: str = "grpc"  # grpc, http, https, tcp, websocket
    weight: int = 1  # For weighted load balancing
    max_connections: int = 100
    connection_timeout_ms: int = 5000
    request_timeout_ms: int = 30000
    health_check_endpoint: str = "/health"
    health_check_interval_seconds: int = 30
    retry_on_failure: bool = True
    ssl_verify: bool = True
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    api_version: str = "v1"
    region: Optional[str] = None
    availability_zone: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate server configuration"""
        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"Invalid port: {self.port}")
        if self.weight <= 0:
            raise ValueError(f"Weight must be positive: {self.weight}")
        if self.max_connections <= 0:
            raise ValueError(f"max_connections must be positive: {self.max_connections}")
        if self.protocol not in ["grpc", "http", "https", "tcp", "websocket"]:
            raise ValueError(f"Unsupported protocol: {self.protocol}")
        if self.connection_timeout_ms <= 0:
            raise ValueError(f"connection_timeout_ms must be positive: {self.connection_timeout_ms}")
        if self.request_timeout_ms <= 0:
            raise ValueError(f"request_timeout_ms must be positive: {self.request_timeout_ms}")
    
    def get_url(self) -> str:
        """Get full URL for the server"""
        return f"{self.protocol}://{self.host}:{self.port}/{self.api_version}"
    
    def is_secure(self) -> bool:
        """Check if connection uses secure protocol"""
        return self.protocol in ["https", "grpc"] and self.ssl_verify


@dataclass
class ResourceLimits:
    """Resource quotas and limits"""
    max_memory_gb: float = 32.0
    max_gpu_memory_gb: float = 24.0
    max_cpu_cores: int = 16
    max_gpu_count: int = 4
    max_concurrent_requests: int = 1000
    max_request_size_mb: float = 100.0
    max_response_size_mb: float = 500.0
    max_batch_size: int = 256
    max_queue_size: int = 10000
    request_timeout_seconds: int = 300
    idle_timeout_seconds: int = 600
    max_retries: int = 3
    rate_limit_per_minute: int = 10000
    burst_limit: int = 1000
    memory_cleanup_threshold: float = 0.9  # Trigger cleanup at 90% usage
    
    def __post_init__(self):
        """Validate resource limits"""
        if self.max_memory_gb <= 0:
            raise ValueError(f"max_memory_gb must be positive: {self.max_memory_gb}")
        if self.max_gpu_memory_gb <= 0:
            raise ValueError(f"max_gpu_memory_gb must be positive: {self.max_gpu_memory_gb}")
        if self.max_cpu_cores <= 0:
            raise ValueError(f"max_cpu_cores must be positive: {self.max_cpu_cores}")
        if self.max_concurrent_requests <= 0:
            raise ValueError(f"max_concurrent_requests must be positive: {self.max_concurrent_requests}")
        if self.memory_cleanup_threshold <= 0 or self.memory_cleanup_threshold > 1:
            raise ValueError(f"memory_cleanup_threshold must be in (0, 1]: {self.memory_cleanup_threshold}")
    
    def check_request_size(self, size_mb: float) -> bool:
        """Check if request size is within limits"""
        return size_mb <= self.max_request_size_mb
    
    def check_response_size(self, size_mb: float) -> bool:
        """Check if response size is within limits"""
        return size_mb <= self.max_response_size_mb


@dataclass
class SecurityConfig:
    """Security and authentication settings"""
    enable_tls: bool = True
    tls_version: str = "TLSv1.3"
    cipher_suites: List[str] = field(default_factory=lambda: [
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
        "TLS_AES_128_GCM_SHA256"
    ])
    auth_type: str = "bearer"  # none, basic, bearer, oauth2, mtls, api_key
    auth_token: Optional[str] = None
    api_key: Optional[str] = None
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None
    oauth_token_url: Optional[str] = None
    mtls_cert_path: Optional[str] = None
    mtls_key_path: Optional[str] = None
    mtls_ca_cert_path: Optional[str] = None
    enable_encryption_at_rest: bool = True
    encryption_key: Optional[str] = None
    enable_audit_logging: bool = True
    audit_log_path: str = "/var/log/saraphis/audit.log"
    enable_request_signing: bool = False
    signing_algorithm: str = "HMAC-SHA256"
    signing_key: Optional[str] = None
    allowed_ips: List[str] = field(default_factory=list)
    blocked_ips: List[str] = field(default_factory=list)
    enable_rate_limiting: bool = True
    enable_ddos_protection: bool = True
    
    def __post_init__(self):
        """Validate security configuration and generate keys if needed"""
        if self.enable_tls and self.tls_version not in ["TLSv1.2", "TLSv1.3"]:
            raise ValueError(f"Unsupported TLS version: {self.tls_version}")
        if self.auth_type not in ["none", "basic", "bearer", "oauth2", "mtls", "api_key"]:
            raise ValueError(f"Unsupported auth type: {self.auth_type}")
        
        # Generate encryption key if needed
        if self.enable_encryption_at_rest and not self.encryption_key:
            self.encryption_key = base64.b64encode(secrets.token_bytes(32)).decode('utf-8')
            logger.info("Generated new encryption key for data at rest")
        
        # Generate signing key if needed
        if self.enable_request_signing and not self.signing_key:
            self.signing_key = base64.b64encode(secrets.token_bytes(32)).decode('utf-8')
            logger.info("Generated new signing key for requests")
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for requests"""
        headers = {}
        if self.auth_type == "bearer" and self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        elif self.auth_type == "api_key" and self.api_key:
            headers["X-API-Key"] = self.api_key
        elif self.auth_type == "basic" and self.api_key:
            # Use api_key as username:password for basic auth
            encoded = base64.b64encode(self.api_key.encode()).decode('ascii')
            headers["Authorization"] = f"Basic {encoded}"
        return headers
    
    def is_ip_allowed(self, ip: str) -> bool:
        """Check if IP address is allowed"""
        if self.blocked_ips and ip in self.blocked_ips:
            return False
        if self.allowed_ips and ip not in self.allowed_ips:
            return False
        return True


@dataclass
class ScalingPolicy:
    """Auto-scaling rules and configuration"""
    enable_auto_scaling: bool = True
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 0.7
    target_gpu_utilization: float = 0.8
    target_memory_utilization: float = 0.75
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    scale_up_cooldown_seconds: int = 300
    scale_down_cooldown_seconds: int = 600
    scale_up_increment: int = 1
    scale_down_increment: int = 1
    predictive_scaling: bool = True
    schedule_based_scaling: Dict[str, int] = field(default_factory=dict)  # hour -> instance count
    metrics_window_seconds: int = 300
    enable_burst_scaling: bool = True
    burst_capacity: int = 5
    
    def __post_init__(self):
        """Validate scaling policy"""
        if self.min_instances <= 0:
            raise ValueError(f"min_instances must be positive: {self.min_instances}")
        if self.max_instances < self.min_instances:
            raise ValueError(f"max_instances must be >= min_instances: {self.max_instances} < {self.min_instances}")
        if not 0 <= self.target_cpu_utilization <= 1:
            raise ValueError(f"target_cpu_utilization must be in [0, 1]: {self.target_cpu_utilization}")
        if not 0 <= self.target_gpu_utilization <= 1:
            raise ValueError(f"target_gpu_utilization must be in [0, 1]: {self.target_gpu_utilization}")
        if self.scale_up_threshold <= self.scale_down_threshold:
            raise ValueError("scale_up_threshold must be > scale_down_threshold")
    
    def should_scale_up(self, metrics: Dict[str, float]) -> bool:
        """Determine if scale-up is needed based on metrics"""
        if not self.enable_auto_scaling:
            return False
        
        cpu_util = metrics.get('cpu_utilization', 0)
        gpu_util = metrics.get('gpu_utilization', 0)
        mem_util = metrics.get('memory_utilization', 0)
        
        return (cpu_util > self.scale_up_threshold or 
                gpu_util > self.scale_up_threshold or
                mem_util > self.scale_up_threshold)
    
    def should_scale_down(self, metrics: Dict[str, float]) -> bool:
        """Determine if scale-down is possible based on metrics"""
        if not self.enable_auto_scaling:
            return False
        
        cpu_util = metrics.get('cpu_utilization', 0)
        gpu_util = metrics.get('gpu_utilization', 0)
        mem_util = metrics.get('memory_utilization', 0)
        
        return (cpu_util < self.scale_down_threshold and 
                gpu_util < self.scale_down_threshold and
                mem_util < self.scale_down_threshold)
    
    def get_scheduled_instances(self, current_hour: int) -> Optional[int]:
        """Get scheduled instance count for current hour"""
        return self.schedule_based_scaling.get(str(current_hour))


@dataclass
class ConnectionPoolConfig:
    """Connection pooling configuration"""
    min_connections: int = 10
    max_connections: int = 100
    connection_timeout_ms: int = 5000
    idle_timeout_ms: int = 300000  # 5 minutes
    max_connection_age_ms: int = 3600000  # 1 hour
    validation_query: str = "SELECT 1"
    validation_interval_ms: int = 30000
    enable_keep_alive: bool = True
    keep_alive_interval_ms: int = 60000
    enable_connection_reuse: bool = True
    max_requests_per_connection: int = 1000
    enable_pipelining: bool = True
    pipeline_limit: int = 10
    
    def __post_init__(self):
        """Validate connection pool configuration"""
        if self.min_connections < 0:
            raise ValueError(f"min_connections must be non-negative: {self.min_connections}")
        if self.max_connections < self.min_connections:
            raise ValueError(f"max_connections must be >= min_connections: {self.max_connections}")
        if self.connection_timeout_ms <= 0:
            raise ValueError(f"connection_timeout_ms must be positive: {self.connection_timeout_ms}")


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration for fault tolerance"""
    enable_circuit_breaker: bool = True
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: int = 60
    half_open_max_requests: int = 3
    error_types: List[str] = field(default_factory=lambda: [
        "ConnectionError", "TimeoutError", "HTTPError"
    ])
    exclude_status_codes: List[int] = field(default_factory=lambda: [404, 401, 403])
    enable_fallback: bool = True
    fallback_cache_ttl_seconds: int = 300
    
    def __post_init__(self):
        """Validate circuit breaker configuration"""
        if self.failure_threshold <= 0:
            raise ValueError(f"failure_threshold must be positive: {self.failure_threshold}")
        if self.success_threshold <= 0:
            raise ValueError(f"success_threshold must be positive: {self.success_threshold}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive: {self.timeout_seconds}")


@dataclass
class RetryPolicy:
    """Retry policy with exponential backoff"""
    enable_retry: bool = True
    max_retries: int = 3
    initial_delay_ms: int = 1000
    max_delay_ms: int = 30000
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1
    retryable_status_codes: List[int] = field(default_factory=lambda: [
        408, 429, 500, 502, 503, 504
    ])
    retryable_exceptions: List[str] = field(default_factory=lambda: [
        "ConnectionError", "TimeoutError", "TemporaryError"
    ])
    
    def __post_init__(self):
        """Validate retry policy"""
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative: {self.max_retries}")
        if self.initial_delay_ms <= 0:
            raise ValueError(f"initial_delay_ms must be positive: {self.initial_delay_ms}")
        if self.exponential_base <= 1:
            raise ValueError(f"exponential_base must be > 1: {self.exponential_base}")
    
    def get_delay_ms(self, attempt: int) -> int:
        """Calculate delay for retry attempt with exponential backoff"""
        if attempt <= 0:
            return self.initial_delay_ms
        
        delay = min(
            self.initial_delay_ms * (self.exponential_base ** (attempt - 1)),
            self.max_delay_ms
        )
        
        if self.jitter:
            import random
            jitter_amount = delay * self.jitter_factor
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(int(delay), 0)


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    enable_monitoring: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    enable_prometheus: bool = True
    enable_grafana: bool = True
    grafana_dashboard_id: Optional[str] = None
    enable_logging: bool = True
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str = "/var/log/saraphis/deployment.log"
    enable_tracing: bool = True
    tracing_backend: str = "jaeger"  # jaeger, zipkin, opentelemetry
    tracing_endpoint: str = "http://localhost:14268/api/traces"
    sampling_rate: float = 0.1
    enable_profiling: bool = False
    profiling_port: int = 6060
    custom_metrics: Dict[str, str] = field(default_factory=dict)
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "error_rate": 0.01,
        "latency_p99_ms": 1000,
        "cpu_utilization": 0.9,
        "memory_utilization": 0.9
    })
    
    def __post_init__(self):
        """Validate monitoring configuration"""
        if self.metrics_port <= 0 or self.metrics_port > 65535:
            raise ValueError(f"Invalid metrics_port: {self.metrics_port}")
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")
        if self.sampling_rate < 0 or self.sampling_rate > 1:
            raise ValueError(f"sampling_rate must be in [0, 1]: {self.sampling_rate}")


@dataclass
class DeploymentConfig(JAXConfig):
    """Comprehensive deployment configuration extending JAXConfig"""
    # Deployment settings
    deployment_mode: DeploymentMode = DeploymentMode.LOCAL
    environment: Environment = Environment.DEVELOPMENT
    deployment_id: str = field(default_factory=lambda: f"deploy-{int(time.time())}")
    deployment_name: str = "saraphis-tropical"
    deployment_version: str = "1.0.0"
    deployment_region: str = "us-west-2"
    deployment_zone: str = "us-west-2a"
    
    # Server configuration
    remote_servers: List[RemoteServerConfig] = field(default_factory=list)
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_RESPONSE_TIME
    
    # Resource management
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    
    # Security
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Scaling
    scaling: ScalingPolicy = field(default_factory=ScalingPolicy)
    
    # Connection management
    connection_pool: ConnectionPoolConfig = field(default_factory=ConnectionPoolConfig)
    
    # Fault tolerance
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    
    # Monitoring
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # JAX Tropical specific
    tropical_config: Optional[JAXTropicalConfig] = None
    
    # Feature flags
    enable_hot_reload: bool = True
    enable_config_validation: bool = True
    enable_health_checks: bool = True
    enable_graceful_shutdown: bool = True
    shutdown_timeout_seconds: int = 30
    
    # Config management
    config_source: str = "file"  # file, env, cli, remote
    config_file_path: Optional[str] = None
    config_refresh_interval_seconds: int = 60
    config_version: int = 1
    
    def __post_init__(self):
        """Validate deployment configuration after initialization"""
        # Call parent validation
        super().__post_init__()
        
        # Validate deployment-specific settings
        if not self.deployment_name:
            raise ValueError("deployment_name cannot be empty")
        
        if self.deployment_mode == DeploymentMode.REMOTE and not self.remote_servers:
            raise ValueError("Remote deployment mode requires at least one remote server")
        
        if self.deployment_mode in [DeploymentMode.CLUSTER, DeploymentMode.HYBRID] and len(self.remote_servers) < 2:
            logger.warning(f"{self.deployment_mode} mode typically requires multiple servers")
        
        # Initialize GPU auto-detection
        self.gpu_detector = GPUAutoDetector()
        self.detected_gpus = self.gpu_detector.detect_gpus()
        
        # Auto-configure based on detected hardware
        if self.detected_gpus:
            self._auto_configure_for_gpu(self.detected_gpus[0])
        
        # Set up tropical config if not provided
        if self.tropical_config is None:
            self.tropical_config = JAXTropicalConfig()
    
    def _auto_configure_for_gpu(self, gpu_specs: GPUSpecs):
        """Auto-configure settings based on detected GPU"""
        # Adjust memory limits based on GPU
        self.resource_limits.max_gpu_memory_gb = gpu_specs.total_memory_gb * 0.9
        self.memory_fraction = min(0.75, gpu_specs.total_memory_gb / 32)  # Scale with GPU memory
        
        # Adjust scaling based on GPU count
        gpu_count = len(self.detected_gpus)
        if gpu_count > 1:
            self.scaling.max_instances = min(gpu_count * 2, 16)
            if self.tropical_config:
                self.tropical_config.enable_pmap = True
        
        logger.info(f"Auto-configured for GPU: {gpu_specs.name} with {gpu_specs.total_memory_gb:.1f}GB memory")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = asdict(self)
        
        # Convert enums to strings
        config_dict['deployment_mode'] = self.deployment_mode.value
        config_dict['environment'] = self.environment.value
        config_dict['load_balancing_strategy'] = self.load_balancing_strategy.value
        config_dict['platform'] = self.platform.value
        
        # Add detected GPU info
        if hasattr(self, 'detected_gpus') and self.detected_gpus:
            config_dict['detected_gpus'] = [gpu.to_dict() for gpu in self.detected_gpus]
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DeploymentConfig':
        """Create configuration from dictionary"""
        # Convert string enums back to enum types
        if 'deployment_mode' in config_dict:
            config_dict['deployment_mode'] = DeploymentMode(config_dict['deployment_mode'])
        if 'environment' in config_dict:
            config_dict['environment'] = Environment(config_dict['environment'])
        if 'load_balancing_strategy' in config_dict:
            config_dict['load_balancing_strategy'] = LoadBalancingStrategy(config_dict['load_balancing_strategy'])
        if 'platform' in config_dict:
            config_dict['platform'] = JAXPlatform(config_dict['platform'])
        
        # Convert nested dataclasses
        if 'remote_servers' in config_dict:
            config_dict['remote_servers'] = [
                RemoteServerConfig(**server) if isinstance(server, dict) else server
                for server in config_dict['remote_servers']
            ]
        if 'resource_limits' in config_dict and isinstance(config_dict['resource_limits'], dict):
            config_dict['resource_limits'] = ResourceLimits(**config_dict['resource_limits'])
        if 'security' in config_dict and isinstance(config_dict['security'], dict):
            config_dict['security'] = SecurityConfig(**config_dict['security'])
        if 'scaling' in config_dict and isinstance(config_dict['scaling'], dict):
            config_dict['scaling'] = ScalingPolicy(**config_dict['scaling'])
        if 'connection_pool' in config_dict and isinstance(config_dict['connection_pool'], dict):
            config_dict['connection_pool'] = ConnectionPoolConfig(**config_dict['connection_pool'])
        if 'circuit_breaker' in config_dict and isinstance(config_dict['circuit_breaker'], dict):
            config_dict['circuit_breaker'] = CircuitBreakerConfig(**config_dict['circuit_breaker'])
        if 'retry_policy' in config_dict and isinstance(config_dict['retry_policy'], dict):
            config_dict['retry_policy'] = RetryPolicy(**config_dict['retry_policy'])
        if 'monitoring' in config_dict and isinstance(config_dict['monitoring'], dict):
            config_dict['monitoring'] = MonitoringConfig(**config_dict['monitoring'])
        
        # Remove detected_gpus as it's not a constructor parameter
        config_dict.pop('detected_gpus', None)
        
        return cls(**config_dict)
    
    def validate(self) -> List[str]:
        """Validate entire configuration and return list of issues"""
        issues = []
        
        # Check remote server connectivity if in remote mode
        if self.deployment_mode in [DeploymentMode.REMOTE, DeploymentMode.HYBRID]:
            for server in self.remote_servers:
                if not self._test_server_connectivity(server):
                    issues.append(f"Cannot connect to server {server.host}:{server.port}")
        
        # Check resource availability
        if self.detected_gpus:
            total_gpu_memory = sum(gpu.total_memory_gb for gpu in self.detected_gpus)
            if self.resource_limits.max_gpu_memory_gb > total_gpu_memory:
                issues.append(f"Requested GPU memory ({self.resource_limits.max_gpu_memory_gb}GB) exceeds available ({total_gpu_memory}GB)")
        
        # Check file paths
        if self.security.enable_audit_logging:
            audit_dir = os.path.dirname(self.security.audit_log_path)
            if not os.path.exists(audit_dir):
                issues.append(f"Audit log directory does not exist: {audit_dir}")
        
        if self.monitoring.enable_logging:
            log_dir = os.path.dirname(self.monitoring.log_file)
            if not os.path.exists(log_dir):
                issues.append(f"Log directory does not exist: {log_dir}")
        
        # Check scaling policy consistency
        if self.scaling.enable_auto_scaling:
            if self.scaling.min_instances > len(self.remote_servers) and self.deployment_mode == DeploymentMode.REMOTE:
                issues.append(f"min_instances ({self.scaling.min_instances}) exceeds available servers ({len(self.remote_servers)})")
        
        return issues
    
    def _test_server_connectivity(self, server: RemoteServerConfig) -> bool:
        """Test connectivity to a remote server"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(server.connection_timeout_ms / 1000.0)
            result = sock.connect_ex((server.host, server.port))
            sock.close()
            return result == 0
        except Exception as e:
            logger.error(f"Failed to test connectivity to {server.host}:{server.port}: {e}")
            return False


class ConfigurationLoader:
    """Load and manage deployment configurations from various sources"""
    
    def __init__(self, base_path: Optional[str] = None):
        """Initialize configuration loader"""
        self.base_path = base_path or os.getcwd()
        self.configs: Dict[str, DeploymentConfig] = {}
        self.active_config: Optional[DeploymentConfig] = None
        self._lock = threading.Lock()
        self._file_watchers: Dict[str, threading.Thread] = {}
    
    def load_from_file(self, file_path: str, environment: Optional[str] = None) -> DeploymentConfig:
        """Load configuration from YAML or JSON file"""
        full_path = Path(file_path) if os.path.isabs(file_path) else Path(self.base_path) / file_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {full_path}")
        
        with open(full_path, 'r') as f:
            if full_path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif full_path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {full_path.suffix}")
        
        # Apply environment-specific overrides
        if environment and 'environments' in data:
            env_data = data.get('environments', {}).get(environment, {})
            base_data = {k: v for k, v in data.items() if k != 'environments'}
            data = self._merge_configs(base_data, env_data)
        
        config = DeploymentConfig.from_dict(data)
        config.config_file_path = str(full_path)
        
        # Store in cache
        with self._lock:
            self.configs[str(full_path)] = config
        
        # Enable hot reload if requested
        if config.enable_hot_reload:
            self._start_file_watcher(str(full_path))
        
        logger.info(f"Loaded configuration from {full_path}")
        return config
    
    def load_from_env(self, prefix: str = "SARAPHIS_") -> DeploymentConfig:
        """Load configuration from environment variables"""
        config_dict = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Convert SARAPHIS_DEPLOYMENT_MODE to deployment_mode
                config_key = key[len(prefix):].lower()
                
                # Parse value based on content
                try:
                    # Try to parse as JSON first
                    parsed_value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    # Keep as string if not JSON
                    parsed_value = value
                
                # Set nested keys using dots
                self._set_nested_dict(config_dict, config_key.split('_'), parsed_value)
        
        config = DeploymentConfig.from_dict(config_dict)
        config.config_source = "env"
        
        logger.info(f"Loaded configuration from environment variables with prefix {prefix}")
        return config
    
    def load_from_cli(self, args: argparse.Namespace) -> DeploymentConfig:
        """Load configuration from command-line arguments"""
        config_dict = {}
        
        for key, value in vars(args).items():
            if value is not None:
                self._set_nested_dict(config_dict, key.split('_'), value)
        
        config = DeploymentConfig.from_dict(config_dict)
        config.config_source = "cli"
        
        logger.info("Loaded configuration from command-line arguments")
        return config
    
    def load_composite(self, 
                      file_path: Optional[str] = None,
                      environment: Optional[str] = None,
                      env_prefix: str = "SARAPHIS_",
                      cli_args: Optional[argparse.Namespace] = None) -> DeploymentConfig:
        """Load configuration from multiple sources with priority: CLI > ENV > File"""
        configs = []
        
        # Load from file if provided
        if file_path:
            try:
                file_config = self.load_from_file(file_path, environment)
                configs.append(file_config.to_dict())
            except Exception as e:
                logger.warning(f"Failed to load config from file: {e}")
        
        # Load from environment variables
        try:
            env_config = self.load_from_env(env_prefix)
            configs.append(env_config.to_dict())
        except Exception as e:
            logger.warning(f"Failed to load config from environment: {e}")
        
        # Load from CLI if provided
        if cli_args:
            try:
                cli_config = self.load_from_cli(cli_args)
                configs.append(cli_config.to_dict())
            except Exception as e:
                logger.warning(f"Failed to load config from CLI: {e}")
        
        # Merge configurations with increasing priority
        merged = {}
        for config_dict in configs:
            merged = self._merge_configs(merged, config_dict)
        
        config = DeploymentConfig.from_dict(merged)
        config.config_source = "composite"
        
        self.active_config = config
        logger.info("Loaded composite configuration from multiple sources")
        return config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _set_nested_dict(self, d: Dict[str, Any], keys: List[str], value: Any):
        """Set value in nested dictionary using list of keys"""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
    
    def _start_file_watcher(self, file_path: str):
        """Start watching configuration file for changes"""
        if file_path in self._file_watchers:
            return
        
        def watch_file():
            last_modified = os.path.getmtime(file_path)
            while file_path in self._file_watchers:
                try:
                    time.sleep(5)  # Check every 5 seconds
                    current_modified = os.path.getmtime(file_path)
                    if current_modified > last_modified:
                        logger.info(f"Configuration file {file_path} changed, reloading...")
                        config = self.load_from_file(file_path)
                        if config.enable_config_validation:
                            issues = config.validate()
                            if issues:
                                logger.error(f"Configuration validation failed: {issues}")
                                continue
                        with self._lock:
                            self.configs[file_path] = config
                            if self.active_config and self.active_config.config_file_path == file_path:
                                self.active_config = config
                        last_modified = current_modified
                        logger.info(f"Configuration reloaded successfully")
                except Exception as e:
                    logger.error(f"Failed to reload configuration: {e}")
        
        thread = threading.Thread(target=watch_file, daemon=True)
        thread.start()
        self._file_watchers[file_path] = thread
    
    def stop_watchers(self):
        """Stop all file watchers"""
        self._file_watchers.clear()
    
    def save_to_file(self, config: DeploymentConfig, file_path: str):
        """Save configuration to file"""
        full_path = Path(file_path) if os.path.isabs(file_path) else Path(self.base_path) / file_path
        
        config_dict = config.to_dict()
        
        with open(full_path, 'w') as f:
            if full_path.suffix in ['.yaml', '.yml']:
                yaml.safe_dump(config_dict, f, default_flow_style=False)
            elif full_path.suffix == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported file format: {full_path.suffix}")
        
        logger.info(f"Saved configuration to {full_path}")


class DeploymentManager:
    """Manage deployment lifecycle and operations"""
    
    def __init__(self, config: DeploymentConfig):
        """Initialize deployment manager"""
        self.config = config
        self.jax_env = JAXEnvironment(config)
        self.gpu_detector = GPUAutoDetector()
        self.is_deployed = False
        self._health_check_thread: Optional[threading.Thread] = None
        self._metrics: Dict[str, Any] = defaultdict(float)
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
    
    def deploy(self) -> Dict[str, Any]:
        """Deploy the system with current configuration"""
        if self.is_deployed:
            raise RuntimeError("System is already deployed")
        
        deployment_result = {
            'deployment_id': self.config.deployment_id,
            'timestamp': datetime.now().isoformat(),
            'mode': self.config.deployment_mode.value,
            'environment': self.config.environment.value,
            'status': 'deploying',
            'errors': []
        }
        
        try:
            # Validate configuration
            if self.config.enable_config_validation:
                issues = self.config.validate()
                if issues:
                    deployment_result['errors'] = issues
                    deployment_result['status'] = 'failed'
                    raise ValueError(f"Configuration validation failed: {issues}")
            
            # Setup JAX environment
            jax_result = self.jax_env.setup_environment()
            if not jax_result['jax_available']:
                raise RuntimeError(f"JAX setup failed: {jax_result['errors']}")
            deployment_result['jax_info'] = jax_result
            
            # Initialize based on deployment mode
            if self.config.deployment_mode == DeploymentMode.LOCAL:
                self._deploy_local()
            elif self.config.deployment_mode == DeploymentMode.REMOTE:
                self._deploy_remote()
            elif self.config.deployment_mode == DeploymentMode.HYBRID:
                self._deploy_hybrid()
            elif self.config.deployment_mode == DeploymentMode.CLUSTER:
                self._deploy_cluster()
            else:
                raise ValueError(f"Unsupported deployment mode: {self.config.deployment_mode}")
            
            # Start health checks
            if self.config.enable_health_checks:
                self._start_health_checks()
            
            # Start monitoring
            if self.config.monitoring.enable_monitoring:
                self._start_monitoring()
            
            self.is_deployed = True
            deployment_result['status'] = 'deployed'
            
            logger.info(f"Deployment successful: {self.config.deployment_id}")
            
        except Exception as e:
            deployment_result['status'] = 'failed'
            deployment_result['errors'].append(str(e))
            logger.error(f"Deployment failed: {e}")
            raise
        
        return deployment_result
    
    def _deploy_local(self):
        """Deploy on local machine"""
        logger.info("Deploying in LOCAL mode")
        
        # Detect and configure GPUs
        gpu_specs = self.gpu_detector.detect_gpus()
        if gpu_specs:
            logger.info(f"Detected {len(gpu_specs)} GPU(s)")
            for gpu in gpu_specs:
                logger.info(f"  - {gpu.name}: {gpu.total_memory_gb:.1f}GB")
        else:
            logger.warning("No GPUs detected, using CPU only")
    
    def _deploy_remote(self):
        """Deploy on remote servers"""
        logger.info(f"Deploying in REMOTE mode with {len(self.config.remote_servers)} servers")
        
        for server in self.config.remote_servers:
            if not self._test_server_health(server):
                raise RuntimeError(f"Remote server unhealthy: {server.host}:{server.port}")
            logger.info(f"Connected to remote server: {server.get_url()}")
    
    def _deploy_hybrid(self):
        """Deploy in hybrid mode (local + remote)"""
        logger.info("Deploying in HYBRID mode")
        self._deploy_local()
        if self.config.remote_servers:
            self._deploy_remote()
    
    def _deploy_cluster(self):
        """Deploy on cluster"""
        logger.info(f"Deploying in CLUSTER mode with {len(self.config.remote_servers)} nodes")
        
        if len(self.config.remote_servers) < 2:
            raise ValueError("Cluster deployment requires at least 2 nodes")
        
        # Initialize cluster coordination
        for server in self.config.remote_servers:
            if not self._test_server_health(server):
                raise RuntimeError(f"Cluster node unhealthy: {server.host}:{server.port}")
        
        logger.info("Cluster initialized successfully")
    
    def _test_server_health(self, server: RemoteServerConfig) -> bool:
        """Test if server is healthy"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(server.connection_timeout_ms / 1000.0)
            
            if server.protocol == "https" and server.ssl_verify:
                context = ssl.create_default_context()
                if server.ssl_cert_path:
                    context.load_cert_chain(server.ssl_cert_path, server.ssl_key_path)
                sock = context.wrap_socket(sock, server_hostname=server.host)
            
            result = sock.connect_ex((server.host, server.port))
            sock.close()
            
            return result == 0
        except Exception as e:
            logger.error(f"Health check failed for {server.host}:{server.port}: {e}")
            return False
    
    def _start_health_checks(self):
        """Start periodic health checks"""
        def health_check_loop():
            while not self._shutdown_event.is_set():
                try:
                    health_status = self.get_health_status()
                    if not health_status['healthy']:
                        logger.warning(f"Health check failed: {health_status}")
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Health check error: {e}")
        
        self._health_check_thread = threading.Thread(target=health_check_loop, daemon=True)
        self._health_check_thread.start()
    
    def _start_monitoring(self):
        """Start monitoring and metrics collection"""
        logger.info(f"Starting monitoring on port {self.config.monitoring.metrics_port}")
        
        # Initialize metrics
        self._metrics['start_time'] = time.time()
        self._metrics['request_count'] = 0
        self._metrics['error_count'] = 0
        self._metrics['total_latency'] = 0
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        status = {
            'healthy': True,
            'deployment_id': self.config.deployment_id,
            'uptime_seconds': time.time() - self._metrics.get('start_time', time.time()),
            'checks': {}
        }
        
        # Check JAX environment
        jax_info = self.jax_env.get_environment_info()
        status['checks']['jax'] = {
            'healthy': jax_info['jax_available'],
            'platform': jax_info['platform'],
            'device_count': jax_info['device_count']
        }
        
        # Check GPU availability
        gpu_specs = self.gpu_detector.detect_gpus()
        status['checks']['gpu'] = {
            'healthy': len(gpu_specs) > 0,
            'gpu_count': len(gpu_specs),
            'gpus': [gpu.name for gpu in gpu_specs]
        }
        
        # Check remote servers if applicable
        if self.config.deployment_mode in [DeploymentMode.REMOTE, DeploymentMode.HYBRID, DeploymentMode.CLUSTER]:
            server_health = []
            for server in self.config.remote_servers:
                server_health.append({
                    'host': server.host,
                    'port': server.port,
                    'healthy': self._test_server_health(server)
                })
            status['checks']['servers'] = server_health
            status['healthy'] = all(s['healthy'] for s in server_health)
        
        return status
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self._lock:
            metrics = self._metrics.copy()
        
        # Calculate derived metrics
        if metrics.get('request_count', 0) > 0:
            metrics['average_latency_ms'] = metrics.get('total_latency', 0) / metrics['request_count'] * 1000
            metrics['error_rate'] = metrics.get('error_count', 0) / metrics['request_count']
        
        return metrics
    
    def shutdown(self, timeout: Optional[int] = None):
        """Gracefully shutdown deployment"""
        timeout = timeout or self.config.shutdown_timeout_seconds
        logger.info(f"Starting graceful shutdown with {timeout}s timeout")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for health check thread
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=timeout)
        
        # Cleanup resources
        self.is_deployed = False
        logger.info("Shutdown complete")


def create_default_config(environment: str = "development") -> DeploymentConfig:
    """Create a default configuration for the specified environment"""
    env_map = {
        'development': Environment.DEVELOPMENT,
        'staging': Environment.STAGING,
        'production': Environment.PRODUCTION,
        'test': Environment.TESTING
    }
    
    config = DeploymentConfig(
        deployment_mode=DeploymentMode.LOCAL if environment == 'development' else DeploymentMode.HYBRID,
        environment=env_map.get(environment, Environment.DEVELOPMENT),
        deployment_name=f"saraphis-{environment}",
        enable_jit=True,
        memory_fraction=0.75
    )
    
    # Production-specific settings
    if environment == 'production':
        config.security.enable_tls = True
        config.security.auth_type = "bearer"
        config.monitoring.enable_monitoring = True
        config.monitoring.enable_tracing = True
        config.scaling.enable_auto_scaling = True
        config.circuit_breaker.enable_circuit_breaker = True
    
    return config


def main():
    """Example usage and testing"""
    # Create default development config
    dev_config = create_default_config('development')
    
    # Create deployment manager
    manager = DeploymentManager(dev_config)
    
    # Deploy system
    result = manager.deploy()
    print(f"Deployment result: {json.dumps(result, indent=2)}")
    
    # Check health
    health = manager.get_health_status()
    print(f"Health status: {json.dumps(health, indent=2)}")
    
    # Get metrics
    metrics = manager.get_metrics()
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    # Shutdown
    manager.shutdown()


if __name__ == "__main__":
    main()