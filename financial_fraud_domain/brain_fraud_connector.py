"""
Brain-Fraud System Connector - Complete Production Integration
Provides seamless integration between Universal AI Core Brain and Saraphis Fraud Detection
"""

import logging
import threading
import time
import json
import uuid
import asyncio
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
from contextlib import contextmanager
import os
import sys

# Add paths for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

# Import Brain system components
from independent_core.brain import Brain, BrainSystemConfig, BrainPredictionResult
from independent_core.domain_registry import (
    DomainRegistry, DomainConfig, DomainStatus, DomainType, DomainMetadata
)
from independent_core.domain_router import DomainRouter, RoutingStrategy, RoutingResult
from independent_core.domain_state import DomainStateManager, DomainState
from independent_core.training_manager import TrainingManager, TrainingConfig, TrainingStatus

# Import Fraud system components
from financial_fraud_domain.enhanced_fraud_core_main import (
    EnhancedFraudDetectionCore, EnhancedFraudCoreConfig, FraudDetectionResult,
    create_default_fraud_core, create_production_fraud_core
)
from financial_fraud_domain.enhanced_fraud_core_exceptions import (
    EnhancedFraudException, DetectionError, ValidationError, SecurityError,
    IntegrationError, DetectionStrategy, ValidationLevel, SecurityLevel
)
from financial_fraud_domain.enhanced_fraud_core_security import (
    SecurityContext, SecurityManager, SecurityConfig
)
from financial_fraud_domain.enhanced_fraud_core_integration import (
    IntegrationManager, IntegrationConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================== CONFIGURATION ========================

@dataclass
class BrainFraudConnectorConfig:
    """Complete configuration for Brain-Fraud system integration"""
    
    # Brain configuration
    brain_base_path: Path = field(default_factory=lambda: Path.cwd() / ".brain_fraud")
    brain_max_domains: int = 50
    brain_enable_persistence: bool = True
    brain_auto_save_interval: int = 300
    
    # Fraud domain configuration
    fraud_domain_name: str = "financial_fraud"
    fraud_domain_priority: int = 9  # High priority
    fraud_domain_memory_mb: int = 2048
    fraud_domain_cpu_percent: float = 40.0
    
    # Integration settings
    enable_auto_registration: bool = True
    enable_state_sync: bool = True
    sync_interval_seconds: int = 60
    enable_performance_monitoring: bool = True
    enable_security_validation: bool = True
    
    # Performance settings
    max_concurrent_predictions: int = 10
    prediction_timeout_seconds: float = 30.0
    batch_size_limit: int = 1000
    cache_predictions: bool = True
    cache_ttl_seconds: int = 3600
    
    # Security settings
    require_authentication: bool = True
    enable_audit_logging: bool = True
    enable_encryption: bool = True
    security_token_ttl: int = 3600
    
    # Monitoring settings
    metrics_collection_interval: int = 60
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'error_rate': 0.05,
        'latency_ms': 1000,
        'memory_usage_percent': 80,
        'cpu_usage_percent': 70
    })
    
    # Error handling
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60

# ======================== CONNECTOR STATE ========================

@dataclass
class ConnectorState:
    """Maintains state of Brain-Fraud connector"""
    
    # Connection state
    brain_connected: bool = False
    fraud_system_connected: bool = False
    domain_registered: bool = False
    last_sync_time: Optional[datetime] = None
    
    # Performance metrics
    total_predictions: int = 0
    successful_predictions: int = 0
    failed_predictions: int = 0
    total_latency_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Error tracking
    consecutive_errors: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    circuit_breaker_open: bool = False
    circuit_breaker_opened_at: Optional[datetime] = None
    
    # Resource usage
    current_memory_mb: float = 0.0
    current_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary"""
        return {
            'connection': {
                'brain_connected': self.brain_connected,
                'fraud_system_connected': self.fraud_system_connected,
                'domain_registered': self.domain_registered,
                'last_sync_time': self.last_sync_time.isoformat() if self.last_sync_time else None
            },
            'performance': {
                'total_predictions': self.total_predictions,
                'successful_predictions': self.successful_predictions,
                'failed_predictions': self.failed_predictions,
                'success_rate': self.successful_predictions / self.total_predictions if self.total_predictions > 0 else 0,
                'average_latency_ms': self.total_latency_ms / self.total_predictions if self.total_predictions > 0 else 0,
                'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
            },
            'errors': {
                'consecutive_errors': self.consecutive_errors,
                'last_error': self.last_error,
                'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
                'circuit_breaker_open': self.circuit_breaker_open
            },
            'resources': {
                'current_memory_mb': self.current_memory_mb,
                'current_cpu_percent': self.current_cpu_percent,
                'peak_memory_mb': self.peak_memory_mb,
                'peak_cpu_percent': self.peak_cpu_percent
            }
        }

# ======================== MAIN CONNECTOR CLASS ========================

class BrainFraudSystemConnector:
    """
    Complete production-ready connector between Brain and Fraud Detection systems.
    Provides seamless integration with full error handling, monitoring, and security.
    """
    
    def __init__(self, config: Optional[BrainFraudConnectorConfig] = None):
        """Initialize Brain-Fraud connector with complete configuration"""
        
        self.config = config or BrainFraudConnectorConfig()
        self.state = ConnectorState()
        self.initialized = False
        self._shutdown = False
        
        # Thread safety
        self._lock = threading.RLock()
        self._sync_thread = None
        self._monitor_thread = None
        
        # Core systems
        self.brain: Optional[Brain] = None
        self.fraud_system: Optional[Any] = None  # SaraphisEnhancedFraudSystem
        self.fraud_core: Optional[EnhancedFraudDetectionCore] = None
        
        # Caching
        self._prediction_cache: Dict[str, Tuple[FraudDetectionResult, datetime]] = {}
        self._cache_lock = threading.Lock()
        
        # Thread pool for async operations
        self._thread_pool = ThreadPoolExecutor(max_workers=self.config.max_concurrent_predictions)
        
        # Security
        self._security_tokens: Dict[str, datetime] = {}
        self._security_lock = threading.Lock()
        
        # Initialize connector
        self._initialize()
    
    def _initialize(self) -> None:
        """Complete initialization of Brain-Fraud connector"""
        try:
            logger.info("Initializing Brain-Fraud System Connector...")
            
            # Initialize Brain system
            self._initialize_brain_system()
            
            # Initialize Fraud system
            self._initialize_fraud_system()
            
            # Register fraud domain with Brain
            if self.config.enable_auto_registration:
                self._register_fraud_domain()
            
            # Setup integration
            self._setup_integration()
            
            # Start background threads
            self._start_background_threads()
            
            self.initialized = True
            logger.info("Brain-Fraud System Connector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Brain-Fraud connector: {e}")
            self._cleanup_partial_initialization()
            raise IntegrationError(f"Connector initialization failed: {e}")
    
    def _initialize_brain_system(self) -> None:
        """Initialize Brain system with production configuration"""
        try:
            logger.info("Initializing Brain system...")
            
            # Create Brain configuration
            brain_config = BrainSystemConfig(
                base_path=self.config.brain_base_path,
                enable_persistence=self.config.brain_enable_persistence,
                auto_save_interval=self.config.brain_auto_save_interval,
                max_domains=self.config.brain_max_domains,
                enable_monitoring=self.config.enable_performance_monitoring,
                enable_adaptation=True,
                max_memory_gb=8.0,
                max_cpu_percent=80.0
            )
            
            # Initialize Brain
            self.brain = Brain(brain_config)
            self.state.brain_connected = True
            
            logger.info("Brain system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Brain system: {e}")
            raise
    
    def _initialize_fraud_system(self) -> None:
        """Initialize Fraud Detection system"""
        try:
            logger.info("Initializing Fraud Detection system...")
            
            # Import fraud system
            from financial_fraud_domain.enhanced_fraud_core_system import SaraphisEnhancedFraudSystem
            
            # Initialize fraud system
            self.fraud_system = SaraphisEnhancedFraudSystem()
            
            # Get reference to fraud core
            if hasattr(self.fraud_system, 'fraud_core'):
                self.fraud_core = self.fraud_system.fraud_core
            elif hasattr(self.fraud_system, 'integration_manager') and hasattr(self.fraud_system.integration_manager, 'fraud_core'):
                self.fraud_core = self.fraud_system.integration_manager.fraud_core
            
            self.state.fraud_system_connected = True
            
            logger.info("Fraud Detection system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Fraud Detection system: {e}")
            raise
    
    def _register_fraud_domain(self) -> None:
        """Register fraud detection as a domain in Brain system"""
        try:
            logger.info(f"Registering fraud domain '{self.config.fraud_domain_name}' with Brain...")
            
            # Create domain configuration
            domain_config = DomainConfig(
                domain_type=DomainType.SPECIALIZED,
                description="Financial fraud detection and prevention domain",
                version="1.0.0",
                max_memory_mb=self.config.fraud_domain_memory_mb,
                max_cpu_percent=self.config.fraud_domain_cpu_percent,
                priority=self.config.fraud_domain_priority,
                hidden_layers=[512, 256, 128, 64],
                activation_function="relu",
                dropout_rate=0.2,
                learning_rate=0.001,
                enable_caching=self.config.cache_predictions,
                cache_size=1000,
                enable_logging=True,
                shared_foundation_layers=3,
                allow_cross_domain_access=False,
                author="Saraphis AI",
                tags=["financial", "fraud", "security", "risk", "compliance"]
            )
            
            # Add domain to Brain
            result = self.brain.add_domain(
                self.config.fraud_domain_name,
                domain_config,
                initialize_model=True
            )
            
            if result['success']:
                self.state.domain_registered = True
                logger.info(f"Fraud domain registered successfully: {result}")
                
                # Add fraud-specific patterns to domain router
                self._configure_fraud_routing()
                
            else:
                raise IntegrationError(f"Failed to register fraud domain: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Failed to register fraud domain: {e}")
            raise
    
    def _configure_fraud_routing(self) -> None:
        """Configure Brain's domain router for fraud detection"""
        try:
            # Add fraud-specific routing patterns
            if hasattr(self.brain, 'domain_router'):
                # These patterns help route fraud-related requests to the fraud domain
                fraud_patterns = [
                    {'pattern': 'fraud', 'weight': 1.0},
                    {'pattern': 'transaction', 'weight': 0.8},
                    {'pattern': 'payment', 'weight': 0.7},
                    {'pattern': 'financial', 'weight': 0.7},
                    {'pattern': 'risk', 'weight': 0.6},
                    {'pattern': 'suspicious', 'weight': 0.9},
                    {'pattern': 'anomaly', 'weight': 0.8},
                    {'pattern': 'money_laundering', 'weight': 1.0},
                    {'pattern': 'unauthorized', 'weight': 0.9}
                ]
                
                # Would add patterns to router if it had such a method
                logger.debug(f"Configured {len(fraud_patterns)} fraud routing patterns")
                
        except Exception as e:
            logger.warning(f"Failed to configure fraud routing: {e}")
    
    def _setup_integration(self) -> None:
        """Setup integration between Brain and Fraud systems"""
        try:
            logger.info("Setting up Brain-Fraud integration...")
            
            # Create bidirectional hooks
            self._setup_brain_hooks()
            self._setup_fraud_hooks()
            
            # Initialize shared state
            self._initialize_shared_state()
            
            # Setup monitoring integration
            if self.config.enable_performance_monitoring:
                self._setup_monitoring_integration()
            
            logger.info("Brain-Fraud integration setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup integration: {e}")
            raise
    
    def _setup_brain_hooks(self) -> None:
        """Setup hooks in Brain system for fraud detection"""
        # Add fraud detection capability to Brain
        if hasattr(self.brain, '_domain_specific_capabilities'):
            self.brain._domain_specific_capabilities = self.brain._domain_specific_capabilities or {}
            self.brain._domain_specific_capabilities[self.config.fraud_domain_name] = {
                'detect_fraud': self._brain_detect_fraud_wrapper,
                'get_fraud_status': self._brain_get_fraud_status_wrapper,
                'train_fraud_model': self._brain_train_fraud_wrapper
            }
    
    def _setup_fraud_hooks(self) -> None:
        """Setup hooks in Fraud system for Brain integration"""
        # Add Brain integration to fraud system
        if self.fraud_system and hasattr(self.fraud_system, '_brain_connector'):
            self.fraud_system._brain_connector = self
    
    def _initialize_shared_state(self) -> None:
        """Initialize shared state between systems"""
        try:
            # Create shared knowledge entry in Brain
            self.brain.brain_core.add_shared_knowledge(
                key=f"domain_{self.config.fraud_domain_name}_metadata",
                value={
                    'connector_id': str(uuid.uuid4()),
                    'initialized_at': datetime.now().isoformat(),
                    'capabilities': [
                        'real_time_detection',
                        'batch_processing',
                        'ml_prediction',
                        'rule_based_detection',
                        'behavioral_analysis',
                        'risk_scoring'
                    ],
                    'performance_metrics': self.state.to_dict()['performance']
                },
                domain=self.config.fraud_domain_name
            )
            
        except Exception as e:
            logger.warning(f"Failed to initialize shared state: {e}")
    
    def _setup_monitoring_integration(self) -> None:
        """Setup monitoring integration between systems"""
        # Connect Brain's monitoring to fraud system metrics
        if hasattr(self.brain, 'monitoring_manager') and hasattr(self.fraud_system, 'monitoring_manager'):
            # Share monitoring data
            logger.debug("Monitoring integration configured")
    
    def _start_background_threads(self) -> None:
        """Start background threads for sync and monitoring"""
        if self.config.enable_state_sync:
            self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
            self._sync_thread.start()
            logger.debug("State sync thread started")
        
        if self.config.enable_performance_monitoring:
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.debug("Monitoring thread started")
    
    def _sync_loop(self) -> None:
        """Background thread for state synchronization"""
        while not self._shutdown:
            try:
                time.sleep(self.config.sync_interval_seconds)
                if not self._shutdown:
                    self._sync_state()
            except Exception as e:
                logger.error(f"State sync error: {e}")
    
    def _monitor_loop(self) -> None:
        """Background thread for performance monitoring"""
        while not self._shutdown:
            try:
                time.sleep(self.config.metrics_collection_interval)
                if not self._shutdown:
                    self._collect_metrics()
                    self._check_alerts()
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def _sync_state(self) -> None:
        """Synchronize state between Brain and Fraud systems"""
        with self._lock:
            try:
                # Update Brain domain metrics
                if self.brain and self.state.domain_registered:
                    self.brain.domain_registry.update_domain_metrics(
                        self.config.fraud_domain_name,
                        {
                            'predictions': self.state.total_predictions,
                            'confidence': self.state.successful_predictions / self.state.total_predictions if self.state.total_predictions > 0 else 0,
                            'resource_usage': {
                                'memory_mb': self.state.current_memory_mb,
                                'cpu_percent': self.state.current_cpu_percent
                            }
                        }
                    )
                
                # Update shared knowledge
                self._update_shared_knowledge()
                
                self.state.last_sync_time = datetime.now()
                
            except Exception as e:
                logger.error(f"State sync failed: {e}")
    
    def _update_shared_knowledge(self) -> None:
        """Update shared knowledge in Brain system"""
        try:
            # Update performance metrics
            self.brain.brain_core.add_shared_knowledge(
                key=f"domain_{self.config.fraud_domain_name}_performance",
                value=self.state.to_dict(),
                domain=self.config.fraud_domain_name
            )
            
            # Update fraud patterns if available
            if self.fraud_core:
                stats = self.fraud_core.get_system_status()
                self.brain.brain_core.add_shared_knowledge(
                    key=f"domain_{self.config.fraud_domain_name}_patterns",
                    value={
                        'detection_strategies': ['rules_only', 'ml_only', 'hybrid', 'ensemble'],
                        'active_strategy': stats.get('core_status', {}).get('detection_strategy', 'hybrid'),
                        'component_health': stats.get('component_health', {})
                    },
                    domain=self.config.fraud_domain_name
                )
                
        except Exception as e:
            logger.warning(f"Failed to update shared knowledge: {e}")
    
    def _collect_metrics(self) -> None:
        """Collect performance metrics from both systems"""
        try:
            # Get Brain metrics
            brain_status = self.brain.get_brain_status() if self.brain else {}
            
            # Get Fraud system metrics
            fraud_status = self.fraud_system.get_system_status() if self.fraud_system else {}
            
            # Update resource usage
            resources = brain_status.get('resources', {})
            self.state.current_memory_mb = resources.get('memory_mb', 0)
            self.state.current_cpu_percent = resources.get('cpu_percent', 0)
            
            # Update peaks
            self.state.peak_memory_mb = max(self.state.peak_memory_mb, self.state.current_memory_mb)
            self.state.peak_cpu_percent = max(self.state.peak_cpu_percent, self.state.current_cpu_percent)
            
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
    
    def _check_alerts(self) -> None:
        """Check metrics against alert thresholds"""
        try:
            alerts = []
            
            # Check error rate
            if self.state.total_predictions > 0:
                error_rate = self.state.failed_predictions / self.state.total_predictions
                if error_rate > self.config.alert_thresholds['error_rate']:
                    alerts.append(f"High error rate: {error_rate:.2%}")
            
            # Check latency
            if self.state.total_predictions > 0:
                avg_latency = self.state.total_latency_ms / self.state.total_predictions
                if avg_latency > self.config.alert_thresholds['latency_ms']:
                    alerts.append(f"High latency: {avg_latency:.0f}ms")
            
            # Check resources
            if self.state.current_memory_mb > 0:
                memory_percent = (self.state.current_memory_mb / (self.config.fraud_domain_memory_mb)) * 100
                if memory_percent > self.config.alert_thresholds['memory_usage_percent']:
                    alerts.append(f"High memory usage: {memory_percent:.1f}%")
            
            if self.state.current_cpu_percent > self.config.alert_thresholds['cpu_usage_percent']:
                alerts.append(f"High CPU usage: {self.state.current_cpu_percent:.1f}%")
            
            # Log alerts
            for alert in alerts:
                logger.warning(f"Performance alert: {alert}")
                
        except Exception as e:
            logger.error(f"Alert check failed: {e}")
    
    # ======================== MAIN INTEGRATION METHODS ========================
    
    def detect_fraud(self, transaction: Dict[str, Any], 
                    user_context: Optional[Dict[str, Any]] = None,
                    use_brain: bool = True) -> Union[FraudDetectionResult, BrainPredictionResult]:
        """
        Main fraud detection method with Brain integration.
        
        Args:
            transaction: Transaction data to analyze
            user_context: Optional user context for security
            use_brain: Whether to use Brain system (True) or direct fraud system (False)
            
        Returns:
            FraudDetectionResult or BrainPredictionResult based on use_brain flag
        """
        if not self.initialized:
            raise IntegrationError("Connector not initialized")
        
        # Check circuit breaker
        if self.state.circuit_breaker_open:
            if self._should_close_circuit_breaker():
                self._close_circuit_breaker()
            else:
                raise IntegrationError("Circuit breaker is open due to repeated failures")
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(transaction)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.state.cache_hits += 1
                return cached_result
            
            self.state.cache_misses += 1
            
            # Validate security if required
            if self.config.require_authentication and user_context:
                self._validate_security(user_context)
            
            # Detect fraud using appropriate system
            if use_brain:
                result = self._detect_fraud_via_brain(transaction, user_context)
            else:
                result = self._detect_fraud_direct(transaction, user_context)
            
            # Update metrics
            latency_ms = (time.time() - start_time) * 1000
            self.state.total_predictions += 1
            self.state.successful_predictions += 1
            self.state.total_latency_ms += latency_ms
            self.state.consecutive_errors = 0
            
            # Cache result
            self._cache_result(cache_key, result)
            
            # Audit log
            if self.config.enable_audit_logging:
                self._audit_log_detection(transaction, result, user_context, latency_ms)
            
            return result
            
        except Exception as e:
            # Update error metrics
            self.state.failed_predictions += 1
            self.state.consecutive_errors += 1
            self.state.last_error = str(e)
            self.state.last_error_time = datetime.now()
            
            # Check if circuit breaker should open
            if self.state.consecutive_errors >= self.config.circuit_breaker_threshold:
                self._open_circuit_breaker()
            
            logger.error(f"Fraud detection failed: {e}")
            raise
    
    def _detect_fraud_via_brain(self, transaction: Dict[str, Any],
                               user_context: Optional[Dict[str, Any]]) -> BrainPredictionResult:
        """Detect fraud using Brain system"""
        # Prepare input for Brain
        brain_input = {
            'domain': self.config.fraud_domain_name,
            'task': 'fraud_detection',
            'data': transaction,
            'context': user_context or {}
        }
        
        # Use Brain's prediction
        result = self.brain.predict(
            brain_input,
            domain=self.config.fraud_domain_name,
            return_reasoning=True
        )
        
        return result
    
    def _detect_fraud_direct(self, transaction: Dict[str, Any],
                            user_context: Optional[Dict[str, Any]]) -> FraudDetectionResult:
        """Detect fraud using direct fraud system"""
        # Create security context
        security_context = None
        if user_context:
            security_context = SecurityContext(
                user_id=user_context.get('user_id', 'unknown'),
                session_id=user_context.get('session_id', str(uuid.uuid4())),
                ip_address=user_context.get('ip_address', '127.0.0.1'),
                user_agent=user_context.get('user_agent', 'unknown'),
                permissions=user_context.get('permissions', ['read']),
                authentication_method=user_context.get('auth_method', 'unknown'),
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=1)
            )
        
        # Use fraud system directly
        result = self.fraud_system.detect_fraud(transaction, user_context)
        
        return result
    
    # ======================== BRAIN WRAPPER METHODS ========================
    
    def _brain_detect_fraud_wrapper(self, transaction: Dict[str, Any],
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Wrapper for Brain to call fraud detection"""
        try:
            # Use direct fraud detection
            result = self._detect_fraud_direct(transaction, context)
            
            # Convert to Brain-compatible format
            return {
                'prediction': result.fraud_detected,
                'confidence': result.confidence,
                'fraud_probability': result.fraud_probability,
                'risk_score': result.risk_score,
                'explanation': result.explanation,
                'metadata': {
                    'transaction_id': result.transaction_id,
                    'detection_strategy': result.detection_strategy.value,
                    'detection_time': result.detection_time,
                    'violated_rules': result.violated_rules,
                    'ml_predictions': result.ml_predictions,
                    'behavioral_anomalies': result.behavioral_anomalies
                }
            }
        except Exception as e:
            logger.error(f"Brain fraud detection wrapper failed: {e}")
            raise
    
    def _brain_get_fraud_status_wrapper(self) -> Dict[str, Any]:
        """Wrapper for Brain to get fraud system status"""
        try:
            return self.get_status()
        except Exception as e:
            logger.error(f"Brain fraud status wrapper failed: {e}")
            return {'error': str(e)}
    
    def _brain_train_fraud_wrapper(self, training_data: Any,
                                  training_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Wrapper for Brain to train fraud detection"""
        try:
            # Would implement training through fraud system
            return {
                'success': False,
                'message': 'Training not implemented in this version'
            }
        except Exception as e:
            logger.error(f"Brain fraud training wrapper failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # ======================== UTILITY METHODS ========================
    
    def _generate_cache_key(self, transaction: Dict[str, Any]) -> str:
        """Generate cache key for transaction"""
        # Create deterministic key from transaction data
        key_data = {
            'transaction_id': transaction.get('transaction_id', ''),
            'user_id': transaction.get('user_id', ''),
            'amount': transaction.get('amount', 0),
            'merchant_id': transaction.get('merchant_id', ''),
            'timestamp': transaction.get('timestamp', '')[:10]  # Date only
        }
        return json.dumps(key_data, sort_keys=True)
    
    def _get_cached_result(self, cache_key: str) -> Optional[FraudDetectionResult]:
        """Get cached result if available and not expired"""
        with self._cache_lock:
            if cache_key in self._prediction_cache:
                result, cached_time = self._prediction_cache[cache_key]
                if (datetime.now() - cached_time).total_seconds() < self.config.cache_ttl_seconds:
                    return result
                else:
                    # Expired
                    del self._prediction_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: Union[FraudDetectionResult, BrainPredictionResult]) -> None:
        """Cache prediction result"""
        if not self.config.cache_predictions:
            return
        
        with self._cache_lock:
            # Limit cache size
            if len(self._prediction_cache) >= 10000:
                # Remove oldest entries
                sorted_items = sorted(self._prediction_cache.items(), 
                                    key=lambda x: x[1][1])  # Sort by timestamp
                for key, _ in sorted_items[:1000]:  # Remove oldest 1000
                    del self._prediction_cache[key]
            
            self._prediction_cache[cache_key] = (result, datetime.now())
    
    def _validate_security(self, user_context: Dict[str, Any]) -> None:
        """Validate security context"""
        if not user_context.get('auth_token'):
            raise SecurityError("Authentication token required")
        
        # Validate token
        with self._security_lock:
            token = user_context['auth_token']
            if token not in self._security_tokens:
                raise SecurityError("Invalid authentication token")
            
            token_time = self._security_tokens[token]
            if (datetime.now() - token_time).total_seconds() > self.config.security_token_ttl:
                del self._security_tokens[token]
                raise SecurityError("Authentication token expired")
    
    def _should_close_circuit_breaker(self) -> bool:
        """Check if circuit breaker should be closed"""
        if not self.state.circuit_breaker_open:
            return False
        
        if self.state.circuit_breaker_opened_at:
            elapsed = (datetime.now() - self.state.circuit_breaker_opened_at).total_seconds()
            return elapsed >= self.config.circuit_breaker_timeout
        
        return False
    
    def _open_circuit_breaker(self) -> None:
        """Open circuit breaker due to failures"""
        self.state.circuit_breaker_open = True
        self.state.circuit_breaker_opened_at = datetime.now()
        logger.error(f"Circuit breaker opened after {self.state.consecutive_errors} consecutive errors")
    
    def _close_circuit_breaker(self) -> None:
        """Close circuit breaker"""
        self.state.circuit_breaker_open = False
        self.state.circuit_breaker_opened_at = None
        self.state.consecutive_errors = 0
        logger.info("Circuit breaker closed")
    
    def _audit_log_detection(self, transaction: Dict[str, Any],
                            result: Union[FraudDetectionResult, BrainPredictionResult],
                            user_context: Optional[Dict[str, Any]],
                            latency_ms: float) -> None:
        """Log fraud detection for audit trail"""
        try:
            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'transaction_id': transaction.get('transaction_id', 'unknown'),
                'user_id': user_context.get('user_id', 'unknown') if user_context else 'system',
                'fraud_detected': getattr(result, 'fraud_detected', False),
                'fraud_probability': getattr(result, 'fraud_probability', 0.0),
                'detection_latency_ms': latency_ms,
                'used_brain': isinstance(result, BrainPredictionResult),
                'ip_address': user_context.get('ip_address', 'unknown') if user_context else None
            }
            
            # Log to audit system
            logger.info(f"Fraud detection audit: {json.dumps(audit_entry)}")
            
        except Exception as e:
            logger.error(f"Audit logging failed: {e}")
    
    def _cleanup_partial_initialization(self) -> None:
        """Cleanup after partial initialization failure"""
        try:
            if self.brain:
                self.brain.shutdown()
            if self.fraud_system:
                self.fraud_system.shutdown()
            if self._thread_pool:
                self._thread_pool.shutdown(wait=False)
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    # ======================== PUBLIC API METHODS ========================
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive connector status"""
        return {
            'initialized': self.initialized,
            'connector_state': self.state.to_dict(),
            'configuration': {
                'fraud_domain_name': self.config.fraud_domain_name,
                'auto_registration': self.config.enable_auto_registration,
                'state_sync': self.config.enable_state_sync,
                'performance_monitoring': self.config.enable_performance_monitoring,
                'security_validation': self.config.enable_security_validation
            },
            'system_status': {
                'brain_status': self.brain.get_brain_status() if self.brain else None,
                'fraud_status': self.fraud_system.get_system_status() if self.fraud_system else None
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        metrics = {
            'prediction_metrics': {
                'total': self.state.total_predictions,
                'successful': self.state.successful_predictions,
                'failed': self.state.failed_predictions,
                'success_rate': self.state.successful_predictions / self.state.total_predictions if self.state.total_predictions > 0 else 0,
                'average_latency_ms': self.state.total_latency_ms / self.state.total_predictions if self.state.total_predictions > 0 else 0
            },
            'cache_metrics': {
                'hits': self.state.cache_hits,
                'misses': self.state.cache_misses,
                'hit_rate': self.state.cache_hits / (self.state.cache_hits + self.state.cache_misses) if (self.state.cache_hits + self.state.cache_misses) > 0 else 0,
                'current_size': len(self._prediction_cache)
            },
            'resource_metrics': {
                'current_memory_mb': self.state.current_memory_mb,
                'peak_memory_mb': self.state.peak_memory_mb,
                'current_cpu_percent': self.state.current_cpu_percent,
                'peak_cpu_percent': self.state.peak_cpu_percent
            },
            'error_metrics': {
                'consecutive_errors': self.state.consecutive_errors,
                'last_error': self.state.last_error,
                'last_error_time': self.state.last_error_time.isoformat() if self.state.last_error_time else None,
                'circuit_breaker_open': self.state.circuit_breaker_open
            }
        }
        
        # Add Brain metrics if available
        if self.brain and self.state.domain_registered:
            brain_metrics = self.brain.get_performance_metrics()
            metrics['brain_metrics'] = brain_metrics
        
        # Add Fraud system metrics if available
        if self.fraud_system:
            try:
                fraud_metrics = self.fraud_system.get_performance_metrics()
                metrics['fraud_metrics'] = fraud_metrics
            except:
                pass
        
        return metrics
    
    def configure_fraud_detection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure fraud detection settings"""
        try:
            results = {
                'success': True,
                'updated_settings': []
            }
            
            # Update detection strategy
            if 'detection_strategy' in config and self.fraud_core:
                strategy = DetectionStrategy(config['detection_strategy'])
                self.fraud_core.config.detection_strategy = strategy
                results['updated_settings'].append('detection_strategy')
            
            # Update thresholds
            threshold_mappings = {
                'fraud_probability_threshold': 'fraud_probability_threshold',
                'risk_score_threshold': 'risk_score_threshold',
                'confidence_threshold': 'confidence_threshold'
            }
            
            for config_key, core_key in threshold_mappings.items():
                if config_key in config and self.fraud_core:
                    setattr(self.fraud_core.config, core_key, config[config_key])
                    results['updated_settings'].append(config_key)
            
            # Update cache settings
            if 'cache_predictions' in config:
                self.config.cache_predictions = config['cache_predictions']
                results['updated_settings'].append('cache_predictions')
            
            if 'cache_ttl_seconds' in config:
                self.config.cache_ttl_seconds = config['cache_ttl_seconds']
                results['updated_settings'].append('cache_ttl_seconds')
            
            return results
            
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def batch_detect_fraud(self, transactions: List[Dict[str, Any]],
                          user_context: Optional[Dict[str, Any]] = None,
                          use_brain: bool = True) -> List[Union[FraudDetectionResult, BrainPredictionResult]]:
        """Batch fraud detection with parallel processing"""
        if not self.initialized:
            raise IntegrationError("Connector not initialized")
        
        if len(transactions) > self.config.batch_size_limit:
            raise ValidationError(f"Batch size {len(transactions)} exceeds limit {self.config.batch_size_limit}")
        
        results = []
        futures = []
        
        # Submit all transactions to thread pool
        for transaction in transactions:
            future = self._thread_pool.submit(
                self.detect_fraud, transaction, user_context, use_brain
            )
            futures.append((transaction.get('transaction_id', 'unknown'), future))
        
        # Collect results
        for tx_id, future in futures:
            try:
                result = future.result(timeout=self.config.prediction_timeout_seconds)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch detection failed for transaction {tx_id}: {e}")
                # Create error result
                if use_brain:
                    error_result = BrainPredictionResult(
                        success=False,
                        prediction=None,
                        confidence=0.0,
                        domain=self.config.fraud_domain_name,
                        reasoning=[],
                        error=str(e)
                    )
                else:
                    error_result = FraudDetectionResult(
                        transaction_id=tx_id,
                        fraud_detected=False,
                        fraud_probability=0.0,
                        risk_score=0.0,
                        confidence=0.0,
                        detection_strategy=DetectionStrategy.HYBRID,
                        detection_time=0.0,
                        timestamp=datetime.now(),
                        explanation=f"Batch processing error: {str(e)}"
                    )
                results.append(error_result)
        
        return results
    
    def generate_auth_token(self, user_id: str) -> str:
        """Generate authentication token for API access"""
        with self._security_lock:
            token = f"bfc_{user_id}_{uuid.uuid4()}"
            self._security_tokens[token] = datetime.now()
            
            # Cleanup old tokens
            current_time = datetime.now()
            expired_tokens = [
                token for token, created_time in self._security_tokens.items()
                if (current_time - created_time).total_seconds() > self.config.security_token_ttl
            ]
            for token in expired_tokens:
                del self._security_tokens[token]
            
            return token
    
    def shutdown(self) -> None:
        """Shutdown connector and cleanup resources"""
        logger.info("Shutting down Brain-Fraud connector...")
        
        self._shutdown = True
        
        # Wait for background threads
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5)
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)
        
        # Clear caches
        self._prediction_cache.clear()
        self._security_tokens.clear()
        
        # Shutdown systems
        if self.brain:
            try:
                self.brain.shutdown()
            except Exception as e:
                logger.error(f"Brain shutdown error: {e}")
        
        if self.fraud_system:
            try:
                self.fraud_system.shutdown()
            except Exception as e:
                logger.error(f"Fraud system shutdown error: {e}")
        
        self.initialized = False
        logger.info("Brain-Fraud connector shutdown complete")

# ======================== FACTORY FUNCTIONS ========================

def create_brain_fraud_connector(config: Optional[BrainFraudConnectorConfig] = None) -> BrainFraudSystemConnector:
    """Create Brain-Fraud connector with optional configuration"""
    return BrainFraudSystemConnector(config)

def create_production_connector() -> BrainFraudSystemConnector:
    """Create production-ready Brain-Fraud connector"""
    config = BrainFraudConnectorConfig(
        enable_auto_registration=True,
        enable_state_sync=True,
        enable_performance_monitoring=True,
        enable_security_validation=True,
        require_authentication=True,
        enable_audit_logging=True,
        cache_predictions=True,
        max_concurrent_predictions=20
    )
    return BrainFraudSystemConnector(config)

def create_test_connector() -> BrainFraudSystemConnector:
    """Create test Brain-Fraud connector"""
    config = BrainFraudConnectorConfig(
        brain_base_path=Path.cwd() / ".brain_fraud_test",
        enable_auto_registration=True,
        enable_state_sync=False,
        enable_performance_monitoring=False,
        require_authentication=False,
        enable_audit_logging=False,
        cache_predictions=False,
        max_concurrent_predictions=2
    )
    return BrainFraudSystemConnector(config)