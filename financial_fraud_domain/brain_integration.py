"""
Brain Integration for Financial Fraud Detection
Integration with Universal AI Core Brain system
"""

import logging
import asyncio
import json
import time
import threading
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import numpy as np
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
import psutil
import traceback

# Import IEEE fraud data loader for Brain integration
try:
    from ieee_fraud_data_loader import IEEEFraudDataLoader, load_ieee_fraud_data
    from training_integrator import IEEEFraudTrainingIntegrator
    IEEE_INTEGRATION_AVAILABLE = True
except ImportError as e:
    IEEE_INTEGRATION_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"IEEE integration components not available: {e}")

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegrationStatus(Enum):
    """Integration status types"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    SYNCING = "syncing"
    SYNCED = "synced"
    ERROR = "error"
    DEGRADED = "degraded"


class IntegrationStrategy(Enum):
    """Integration strategy types"""
    DIRECT = "direct"
    CACHED = "cached"
    ASYNC = "async"
    BATCH = "batch"
    STREAMING = "streaming"


class SyncMode(Enum):
    """State synchronization modes"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SELECTIVE = "selective"


class RouteType(Enum):
    """Routing types for brain integration"""
    QUERY = "query"
    UPDATE = "update"
    TRAINING = "training"
    MONITORING = "monitoring"
    ADMIN = "admin"


@dataclass
class IntegrationMetrics:
    """Metrics for brain integration performance"""
    integration_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: IntegrationStatus = IntegrationStatus.DISCONNECTED
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    errors: int = 0
    warnings: int = 0
    latency_ms: List[float] = field(default_factory=list)
    sync_operations: int = 0
    route_operations: int = 0
    training_operations: int = 0
    
    def calculate_average_latency(self) -> float:
        """Calculate average latency"""
        if not self.latency_ms:
            return 0.0
        return sum(self.latency_ms) / len(self.latency_ms)
    
    def calculate_throughput(self) -> Dict[str, float]:
        """Calculate throughput metrics"""
        if not self.end_time or self.start_time == self.end_time:
            return {"messages_per_second": 0.0, "bytes_per_second": 0.0}
        
        duration = (self.end_time - self.start_time).total_seconds()
        return {
            "messages_per_second": (self.messages_sent + self.messages_received) / duration,
            "bytes_per_second": (self.bytes_sent + self.bytes_received) / duration
        }


@dataclass
class DomainState:
    """Domain state for synchronization"""
    domain_id: str
    version: int
    timestamp: datetime
    models: Dict[str, Any] = field(default_factory=dict)
    configurations: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    active_sessions: int = 0
    checksum: Optional[str] = None
    
    def calculate_checksum(self) -> str:
        """Calculate state checksum"""
        state_str = json.dumps({
            "domain_id": self.domain_id,
            "version": self.version,
            "models": sorted(self.models.keys()),
            "configurations": sorted(self.configurations.keys()),
            "metrics": sorted(self.metrics.keys())
        }, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()


@dataclass
class Route:
    """Route configuration for brain integration"""
    route_id: str
    route_type: RouteType
    path: str
    handler: str
    priority: int = 0
    enabled: bool = True
    filters: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingRequest:
    """Training request for brain integration"""
    request_id: str
    model_name: str
    dataset_path: str
    parameters: Dict[str, Any]
    priority: int = 0
    scheduled_time: Optional[datetime] = None
    status: str = "pending"
    results: Optional[Dict[str, Any]] = None


class IntegrationStrategyBase(ABC):
    """Abstract base class for integration strategies"""
    
    @abstractmethod
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect to brain system"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from brain system"""
        pass
    
    @abstractmethod
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send message to brain"""
        pass
    
    @abstractmethod
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message from brain"""
        pass


class DirectIntegrationStrategy(IntegrationStrategyBase):
    """Direct integration strategy for real-time communication"""
    
    def __init__(self):
        self.connected = False
        self.connection = None
        
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect directly to brain system"""
        try:
            logger.info("Establishing direct connection to brain system")
            # Simulate connection - replace with actual brain connection
            await asyncio.sleep(0.5)
            self.connected = True
            logger.info("Direct connection established")
            return True
        except Exception as e:
            logger.error(f"Direct connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from brain system"""
        try:
            if self.connected:
                logger.info("Closing direct connection")
                await asyncio.sleep(0.1)
                self.connected = False
            return True
        except Exception as e:
            logger.error(f"Disconnect failed: {e}")
            return False
    
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send message directly to brain"""
        if not self.connected:
            return False
        
        try:
            # Simulate message sending
            await asyncio.sleep(0.01)
            logger.debug(f"Sent message: {message.get('type', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message from brain"""
        if not self.connected:
            return None
        
        try:
            # Simulate message receiving
            await asyncio.sleep(0.01)
            # Return simulated message
            return {
                "type": "response",
                "timestamp": datetime.now().isoformat(),
                "data": {}
            }
        except Exception as e:
            logger.error(f"Failed to receive message: {e}")
            return None


class CachedIntegrationStrategy(IntegrationStrategyBase):
    """Cached integration strategy for improved performance"""
    
    def __init__(self, cache_size: int = 1000, ttl_seconds: int = 300):
        self.connected = False
        self.cache: Dict[str, Tuple[Dict[str, Any], datetime]] = {}
        self.cache_size = cache_size
        self.ttl_seconds = ttl_seconds
        self.base_strategy = DirectIntegrationStrategy()
        
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect with caching layer"""
        return await self.base_strategy.connect(config)
    
    async def disconnect(self) -> bool:
        """Disconnect and clear cache"""
        self.cache.clear()
        return await self.base_strategy.disconnect()
    
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send message with caching"""
        # Invalidate related cache entries
        message_type = message.get('type', '')
        if message_type in ['update', 'delete']:
            self._invalidate_cache(message.get('key', ''))
        
        return await self.base_strategy.send_message(message)
    
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message with caching"""
        # Check cache first
        cache_key = "latest_message"
        if cache_key in self.cache:
            cached_msg, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.ttl_seconds:
                logger.debug("Returning cached message")
                return cached_msg
        
        # Get fresh message
        message = await self.base_strategy.receive_message()
        if message:
            self._add_to_cache(cache_key, message)
        
        return message
    
    def _add_to_cache(self, key: str, value: Dict[str, Any]):
        """Add item to cache"""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (value, datetime.now())
    
    def _invalidate_cache(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        keys_to_remove = [k for k in self.cache.keys() if pattern in k]
        for key in keys_to_remove:
            del self.cache[key]


class BatchIntegrationStrategy(IntegrationStrategyBase):
    """Batch integration strategy for high-throughput scenarios"""
    
    def __init__(self, batch_size: int = 100, batch_timeout: float = 1.0):
        self.connected = False
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.message_queue: deque = deque()
        self.response_queue: deque = deque()
        self.base_strategy = DirectIntegrationStrategy()
        self.batch_task = None
        
    async def connect(self, config: Dict[str, Any]) -> bool:
        """Connect and start batch processor"""
        result = await self.base_strategy.connect(config)
        if result:
            self.connected = True
            self.batch_task = asyncio.create_task(self._batch_processor())
        return result
    
    async def disconnect(self) -> bool:
        """Disconnect and stop batch processor"""
        self.connected = False
        if self.batch_task:
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass
        
        # Flush remaining messages
        await self._flush_batch()
        
        return await self.base_strategy.disconnect()
    
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Add message to batch queue"""
        if not self.connected:
            return False
        
        self.message_queue.append(message)
        
        # Trigger immediate flush if batch is full
        if len(self.message_queue) >= self.batch_size:
            await self._flush_batch()
        
        return True
    
    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message from response queue"""
        if self.response_queue:
            return self.response_queue.popleft()
        
        # Wait for responses
        await asyncio.sleep(0.1)
        
        if self.response_queue:
            return self.response_queue.popleft()
        
        return None
    
    async def _batch_processor(self):
        """Process messages in batches"""
        while self.connected:
            try:
                await asyncio.sleep(self.batch_timeout)
                await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
    
    async def _flush_batch(self):
        """Flush current batch"""
        if not self.message_queue:
            return
        
        batch = []
        while self.message_queue and len(batch) < self.batch_size:
            batch.append(self.message_queue.popleft())
        
        if batch:
            # Send batch
            batch_message = {
                "type": "batch",
                "messages": batch,
                "count": len(batch)
            }
            
            success = await self.base_strategy.send_message(batch_message)
            if success:
                logger.info(f"Sent batch of {len(batch)} messages")
                
                # Simulate batch response
                for _ in batch:
                    response = await self.base_strategy.receive_message()
                    if response:
                        self.response_queue.append(response)


class FinancialBrainIntegration:
    """
    Brain Integration for Financial Fraud Detection domain.
    Manages integration with the Universal AI Core Brain system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize brain integration"""
        self.config = config or {}
        self.domain_id = "financial_fraud_detection"
        self.brain_instance = None
        self.status = IntegrationStatus.DISCONNECTED
        self.metrics = None
        self.current_state = None
        self.routes: Dict[str, Route] = {}
        self.training_queue: List[TrainingRequest] = []
        self.integration_lock = threading.Lock()
        self.event_loop = None
        
        # IEEE fraud data integration
        self.ieee_data_loader = None
        self.ieee_training_integrator = None
        if IEEE_INTEGRATION_AVAILABLE:
            try:
                self.ieee_data_loader = IEEEFraudDataLoader(
                    data_dir=self.config.get('ieee_data_dir', 'training_data/ieee-fraud-detection'),
                    fraud_domain_config=self.config.get('fraud_domain_config', {})
                )
                self.ieee_training_integrator = IEEEFraudTrainingIntegrator(config=self.config)
                logger.info("IEEE fraud data integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize IEEE integration: {e}")
        
        # Integration strategies
        self.strategies = {
            IntegrationStrategy.DIRECT: DirectIntegrationStrategy(),
            IntegrationStrategy.CACHED: CachedIntegrationStrategy(),
            IntegrationStrategy.BATCH: BatchIntegrationStrategy()
        }
        self.current_strategy = IntegrationStrategy.DIRECT
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=1000)
        self.error_history: deque = deque(maxlen=100)
        
        # Persistence
        self.state_file = Path("brain_integration_state.pkl")
        
        logger.info(f"FinancialBrainIntegration initialized for domain: {self.domain_id}")
    
    def connect_to_brain(self) -> bool:
        """Connect to Brain system"""
        try:
            logger.info("Attempting to connect to Brain system")
            
            with self.integration_lock:
                self.status = IntegrationStatus.CONNECTING
                self.metrics = IntegrationMetrics(
                    integration_id=self._generate_integration_id(),
                    start_time=datetime.now()
                )
                
                # Initialize event loop if needed
                try:
                    self.event_loop = asyncio.get_event_loop()
                except RuntimeError:
                    self.event_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self.event_loop)
                
                # Connect using current strategy
                strategy = self.strategies[self.current_strategy]
                result = self.event_loop.run_until_complete(
                    strategy.connect(self.config)
                )
                
                if result:
                    self.status = IntegrationStatus.CONNECTED
                    logger.info("Successfully connected to Brain system")
                    
                    # Register domain
                    self.register_with_brain()
                    
                    # Initialize state
                    self.current_state = DomainState(
                        domain_id=self.domain_id,
                        version=1,
                        timestamp=datetime.now()
                    )
                    
                    return True
                else:
                    self.status = IntegrationStatus.ERROR
                    self.metrics.errors += 1
                    logger.error("Failed to connect to Brain system")
                    return False
                    
        except Exception as e:
            logger.error(f"Connection error: {e}")
            logger.error(traceback.format_exc())
            self.status = IntegrationStatus.ERROR
            if self.metrics:
                self.metrics.errors += 1
            self._record_error("connect", str(e))
            return False
    
    def register_with_brain(self) -> bool:
        """Register domain with Brain"""
        try:
            logger.info(f"Registering domain {self.domain_id} with Brain")
            
            registration_message = {
                "type": "register",
                "domain_id": self.domain_id,
                "domain_type": "financial_fraud_detection",
                "capabilities": [
                    "fraud_detection",
                    "risk_assessment",
                    "transaction_monitoring",
                    "pattern_recognition"
                ],
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat()
            }
            
            # Send registration
            strategy = self.strategies[self.current_strategy]
            result = self.event_loop.run_until_complete(
                strategy.send_message(registration_message)
            )
            
            if result:
                logger.info("Domain registration successful")
                if self.metrics:
                    self.metrics.messages_sent += 1
                return True
            else:
                logger.error("Domain registration failed")
                return False
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            self._record_error("register", str(e))
            return False
    
    def sync_with_brain(self) -> bool:
        """Synchronize with Brain system"""
        return self.sync_state(SyncMode.FULL)
    
    def integrate_with_brain(self, 
                           strategy: Optional[IntegrationStrategy] = None,
                           async_mode: bool = False) -> bool:
        """
        Integrate with Brain system using specified strategy.
        
        Args:
            strategy: Integration strategy to use
            async_mode: Whether to run integration asynchronously
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Starting brain integration with strategy: {strategy or self.current_strategy}")
            
            # Update strategy if provided
            if strategy and strategy in self.strategies:
                self.current_strategy = strategy
                logger.info(f"Switched to {strategy.value} integration strategy")
            
            # Connect if not already connected
            if self.status != IntegrationStatus.CONNECTED:
                if not self.connect_to_brain():
                    return False
            
            # Perform integration steps
            if async_mode:
                # Run integration asynchronously
                integration_task = asyncio.create_task(self._async_integration())
                logger.info("Started asynchronous integration")
                return True
            else:
                # Run integration synchronously
                return self._sync_integration()
                
        except Exception as e:
            logger.error(f"Integration error: {e}")
            self._record_error("integrate", str(e))
            return False
    
    def _sync_integration(self) -> bool:
        """Perform synchronous integration"""
        try:
            # Sync state
            if not self.sync_state():
                logger.error("State synchronization failed")
                return False
            
            # Setup routing
            if not self.integrate_routing():
                logger.error("Routing integration failed")
                return False
            
            # Setup training pipeline
            if not self.integrate_training():
                logger.error("Training integration failed")
                return False
            
            # Validate integration
            if not self.validate_integration():
                logger.error("Integration validation failed")
                return False
            
            logger.info("Brain integration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Synchronous integration error: {e}")
            return False
    
    async def _async_integration(self) -> bool:
        """Perform asynchronous integration"""
        try:
            # Run integration steps concurrently
            tasks = [
                self._async_sync_state(),
                self._async_integrate_routing(),
                self._async_integrate_training()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Async integration task {i} failed: {result}")
                    return False
            
            # Validate integration
            if not self.validate_integration():
                logger.error("Async integration validation failed")
                return False
            
            logger.info("Asynchronous brain integration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Asynchronous integration error: {e}")
            return False
    
    def sync_state(self, mode: SyncMode = SyncMode.INCREMENTAL) -> bool:
        """
        Synchronize domain state with Brain.
        
        Args:
            mode: Synchronization mode
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"Starting state synchronization with mode: {mode.value}")
            
            if self.status != IntegrationStatus.CONNECTED:
                logger.error("Not connected to Brain system")
                return False
            
            start_time = time.time()
            self.status = IntegrationStatus.SYNCING
            
            # Prepare state data
            if mode == SyncMode.FULL:
                state_data = self._prepare_full_state()
            elif mode == SyncMode.INCREMENTAL:
                state_data = self._prepare_incremental_state()
            elif mode == SyncMode.DIFFERENTIAL:
                state_data = self._prepare_differential_state()
            else:  # SELECTIVE
                state_data = self._prepare_selective_state()
            
            # Send state to Brain
            sync_message = {
                "type": "sync_state",
                "domain_id": self.domain_id,
                "sync_mode": mode.value,
                "state": state_data,
                "timestamp": datetime.now().isoformat()
            }
            
            strategy = self.strategies[self.current_strategy]
            result = self.event_loop.run_until_complete(
                strategy.send_message(sync_message)
            )
            
            if result:
                # Wait for acknowledgment
                response = self.event_loop.run_until_complete(
                    strategy.receive_message()
                )
                
                if response and response.get("type") == "sync_ack":
                    self.status = IntegrationStatus.SYNCED
                    
                    # Update metrics
                    if self.metrics:
                        self.metrics.sync_operations += 1
                        self.metrics.messages_sent += 1
                        self.metrics.messages_received += 1
                        sync_time = (time.time() - start_time) * 1000
                        self.metrics.latency_ms.append(sync_time)
                    
                    # Record performance
                    self._record_performance("sync_state", sync_time)
                    
                    logger.info(f"State synchronization completed in {sync_time:.2f}ms")
                    return True
            
            self.status = IntegrationStatus.ERROR
            logger.error("State synchronization failed")
            return False
            
        except Exception as e:
            logger.error(f"State sync error: {e}")
            self.status = IntegrationStatus.ERROR
            self._record_error("sync_state", str(e))
            return False
    
    def integrate_routing(self, routes: Optional[List[Route]] = None) -> bool:
        """
        Integrate routing with Brain system.
        
        Args:
            routes: Optional list of routes to register
            
        Returns:
            bool: Success status
        """
        try:
            logger.info("Integrating routing with Brain system")
            
            # Use default routes if none provided
            if not routes:
                routes = self._get_default_routes()
            
            # Register each route
            for route in routes:
                route_message = {
                    "type": "register_route",
                    "domain_id": self.domain_id,
                    "route": {
                        "id": route.route_id,
                        "type": route.route_type.value,
                        "path": route.path,
                        "handler": route.handler,
                        "priority": route.priority,
                        "enabled": route.enabled,
                        "filters": route.filters,
                        "metadata": route.metadata
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                strategy = self.strategies[self.current_strategy]
                result = self.event_loop.run_until_complete(
                    strategy.send_message(route_message)
                )
                
                if result:
                    self.routes[route.route_id] = route
                    logger.info(f"Registered route: {route.route_id}")
                    
                    if self.metrics:
                        self.metrics.route_operations += 1
                        self.metrics.messages_sent += 1
                else:
                    logger.error(f"Failed to register route: {route.route_id}")
                    return False
            
            logger.info(f"Successfully registered {len(routes)} routes")
            return True
            
        except Exception as e:
            logger.error(f"Routing integration error: {e}")
            self._record_error("integrate_routing", str(e))
            return False
    
    def integrate_training(self, training_config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Integrate training pipeline with Brain.
        
        Args:
            training_config: Optional training configuration
            
        Returns:
            bool: Success status
        """
        try:
            logger.info("Integrating training pipeline with Brain system")
            
            # Default training configuration
            if not training_config:
                training_config = {
                    "models": ["fraud_detector", "risk_assessor"],
                    "update_frequency": "daily",
                    "batch_size": 1000,
                    "validation_split": 0.2,
                    "metrics": ["accuracy", "precision", "recall", "f1"],
                    "use_ieee_dataset": IEEE_INTEGRATION_AVAILABLE
                }
            
            # Register training pipeline
            training_message = {
                "type": "register_training",
                "domain_id": self.domain_id,
                "config": training_config,
                "timestamp": datetime.now().isoformat(),
                "ieee_integration_available": IEEE_INTEGRATION_AVAILABLE
            }
            
            strategy = self.strategies[self.current_strategy]
            result = self.event_loop.run_until_complete(
                strategy.send_message(training_message)
            )
            
            if result:
                logger.info("Training pipeline registered successfully")
                
                if self.metrics:
                    self.metrics.training_operations += 1
                    self.metrics.messages_sent += 1
                
                return True
            else:
                logger.error("Failed to register training pipeline")
                return False
                
        except Exception as e:
            logger.error(f"Training integration error: {e}")
            self._record_error("integrate_training", str(e))
            return False
    
    def validate_integration(self) -> bool:
        """
        Validate the brain integration.
        
        Returns:
            bool: True if integration is valid
        """
        try:
            logger.info("Validating brain integration")
            
            validations = {
                "connection": self.status in [IntegrationStatus.CONNECTED, IntegrationStatus.SYNCED],
                "state": self.current_state is not None,
                "routes": len(self.routes) > 0,
                "metrics": self.metrics is not None
            }
            
            # Send validation request
            validation_message = {
                "type": "validate_integration",
                "domain_id": self.domain_id,
                "validations": validations,
                "timestamp": datetime.now().isoformat()
            }
            
            strategy = self.strategies[self.current_strategy]
            result = self.event_loop.run_until_complete(
                strategy.send_message(validation_message)
            )
            
            if result:
                response = self.event_loop.run_until_complete(
                    strategy.receive_message()
                )
                
                if response and response.get("type") == "validation_result":
                    validation_passed = all(validations.values())
                    
                    if validation_passed:
                        logger.info("Integration validation passed")
                    else:
                        failed_checks = [k for k, v in validations.items() if not v]
                        logger.error(f"Integration validation failed: {failed_checks}")
                    
                    return validation_passed
            
            logger.error("Integration validation request failed")
            return False
            
        except Exception as e:
            logger.error(f"Integration validation error: {e}")
            self._record_error("validate_integration", str(e))
            return False
    
    def save_state(self, file_path: Optional[Path] = None) -> bool:
        """Save integration state to file"""
        try:
            path = file_path or self.state_file
            
            state = {
                "domain_id": self.domain_id,
                "status": self.status.value,
                "current_strategy": self.current_strategy.value,
                "routes": {k: asdict(v) for k, v in self.routes.items()},
                "current_state": asdict(self.current_state) if self.current_state else None,
                "metrics": asdict(self.metrics) if self.metrics else None,
                "performance_history": list(self.performance_history),
                "error_history": list(self.error_history),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(path, 'wb') as f:
                pickle.dump(state, f)
            
            logger.info(f"Integration state saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    def load_state(self, file_path: Optional[Path] = None) -> bool:
        """Load integration state from file"""
        try:
            path = file_path or self.state_file
            
            if not path.exists():
                logger.warning(f"State file not found: {path}")
                return False
            
            with open(path, 'rb') as f:
                state = pickle.load(f)
            
            # Restore state
            self.domain_id = state.get("domain_id", self.domain_id)
            self.status = IntegrationStatus(state.get("status", "disconnected"))
            self.current_strategy = IntegrationStrategy(state.get("current_strategy", "direct"))
            
            # Restore routes
            self.routes = {}
            for route_id, route_data in state.get("routes", {}).items():
                route_data["route_type"] = RouteType(route_data["route_type"])
                self.routes[route_id] = Route(**route_data)
            
            # Restore metrics and history
            if state.get("performance_history"):
                self.performance_history = deque(state["performance_history"], maxlen=1000)
            
            if state.get("error_history"):
                self.error_history = deque(state["error_history"], maxlen=100)
            
            logger.info(f"Integration state loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of integration metrics"""
        if not self.metrics:
            return {"status": "No metrics available"}
        
        summary = {
            "integration_id": self.metrics.integration_id,
            "status": self.status.value,
            "uptime_seconds": (datetime.now() - self.metrics.start_time).total_seconds() if self.metrics.start_time else 0,
            "messages": {
                "sent": self.metrics.messages_sent,
                "received": self.metrics.messages_received
            },
            "bytes": {
                "sent": self.metrics.bytes_sent,
                "received": self.metrics.bytes_received
            },
            "operations": {
                "sync": self.metrics.sync_operations,
                "route": self.metrics.route_operations,
                "training": self.metrics.training_operations
            },
            "performance": {
                "average_latency_ms": self.metrics.calculate_average_latency(),
                "throughput": self.metrics.calculate_throughput()
            },
            "errors": self.metrics.errors,
            "warnings": self.metrics.warnings
        }
        
        return summary
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report"""
        report = {
            "metrics_summary": self.get_metrics_summary(),
            "performance_history": {
                "total_operations": len(self.performance_history),
                "operations_by_type": defaultdict(list)
            },
            "error_analysis": {
                "total_errors": len(self.error_history),
                "errors_by_type": defaultdict(int)
            }
        }
        
        # Analyze performance history
        for entry in self.performance_history:
            op_type = entry.get("operation")
            duration = entry.get("duration_ms", 0)
            report["performance_history"]["operations_by_type"][op_type].append(duration)
        
        # Calculate statistics per operation type
        for op_type, durations in report["performance_history"]["operations_by_type"].items():
            if durations:
                report["performance_history"]["operations_by_type"][op_type] = {
                    "count": len(durations),
                    "average_ms": sum(durations) / len(durations),
                    "min_ms": min(durations),
                    "max_ms": max(durations)
                }
        
        # Analyze errors
        for error in self.error_history:
            error_type = error.get("operation", "unknown")
            report["error_analysis"]["errors_by_type"][error_type] += 1
        
        return report
    
    # Helper methods
    
    def _generate_integration_id(self) -> str:
        """Generate unique integration ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = hashlib.md5(
            f"{self.domain_id}{timestamp}".encode()
        ).hexdigest()[:8]
        return f"int-{timestamp}-{random_suffix}"
    
    def _prepare_full_state(self) -> Dict[str, Any]:
        """Prepare full state for synchronization"""
        if not self.current_state:
            return {}
        
        return {
            "domain_id": self.current_state.domain_id,
            "version": self.current_state.version,
            "timestamp": self.current_state.timestamp.isoformat(),
            "models": self.current_state.models,
            "configurations": self.current_state.configurations,
            "metrics": self.current_state.metrics,
            "active_sessions": self.current_state.active_sessions,
            "checksum": self.current_state.calculate_checksum()
        }
    
    def _prepare_incremental_state(self) -> Dict[str, Any]:
        """Prepare incremental state update"""
        # Only send changes since last sync
        return {
            "domain_id": self.current_state.domain_id,
            "version": self.current_state.version,
            "timestamp": datetime.now().isoformat(),
            "changes": {
                "models": {},  # Only changed models
                "configurations": {},  # Only changed configs
                "metrics": self.current_state.metrics  # Always send current metrics
            }
        }
    
    def _prepare_differential_state(self) -> Dict[str, Any]:
        """Prepare differential state update"""
        # Send differences from baseline
        return {
            "domain_id": self.current_state.domain_id,
            "version": self.current_state.version,
            "timestamp": datetime.now().isoformat(),
            "diff": {
                "added": {},
                "modified": {},
                "removed": []
            }
        }
    
    def _prepare_selective_state(self) -> Dict[str, Any]:
        """Prepare selective state update"""
        # Only send specific components
        return {
            "domain_id": self.current_state.domain_id,
            "version": self.current_state.version,
            "timestamp": datetime.now().isoformat(),
            "selected_components": ["metrics", "active_sessions"]
        }
    
    def _get_default_routes(self) -> List[Route]:
        """Get default routes for domain"""
        return [
            Route(
                route_id="fraud_query",
                route_type=RouteType.QUERY,
                path="/api/fraud/query",
                handler="FraudQueryHandler",
                priority=10,
                filters=["authenticated", "rate_limited"]
            ),
            Route(
                route_id="fraud_update",
                route_type=RouteType.UPDATE,
                path="/api/fraud/update",
                handler="FraudUpdateHandler",
                priority=5,
                filters=["authenticated", "authorized"]
            ),
            Route(
                route_id="fraud_training",
                route_type=RouteType.TRAINING,
                path="/api/fraud/train",
                handler="FraudTrainingHandler",
                priority=1,
                filters=["admin_only"]
            ),
            Route(
                route_id="fraud_monitoring",
                route_type=RouteType.MONITORING,
                path="/api/fraud/monitor",
                handler="FraudMonitoringHandler",
                priority=8,
                filters=["authenticated"]
            )
        ]
    
    def _record_performance(self, operation: str, duration_ms: float):
        """Record performance metrics"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration_ms": duration_ms
        }
        self.performance_history.append(entry)
    
    def _record_error(self, operation: str, error: str):
        """Record error information"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "error": error
        }
        self.error_history.append(entry)
        
        if self.metrics:
            self.metrics.errors += 1
    
    async def _async_sync_state(self) -> bool:
        """Asynchronous state synchronization"""
        return self.sync_state()
    
    async def _async_integrate_routing(self) -> bool:
        """Asynchronous routing integration"""
        return self.integrate_routing()
    
    async def _async_integrate_training(self) -> bool:
        """Asynchronous training integration"""
        return self.integrate_training()
    
    def disconnect(self) -> bool:
        """Disconnect from Brain system"""
        try:
            logger.info("Disconnecting from Brain system")
            
            # Save current state
            self.save_state()
            
            # Disconnect using current strategy
            strategy = self.strategies[self.current_strategy]
            result = self.event_loop.run_until_complete(
                strategy.disconnect()
            )
            
            if result:
                self.status = IntegrationStatus.DISCONNECTED
                
                if self.metrics:
                    self.metrics.end_time = datetime.now()
                
                logger.info("Successfully disconnected from Brain system")
                return True
            else:
                logger.error("Failed to disconnect from Brain system")
                return False
                
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
            return False
    
    def start_ieee_training_session(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start IEEE fraud detection training session integrated with Brain
        
        Args:
            training_config: Training configuration
            
        Returns:
            Training session information
        """
        try:
            if not self.ieee_training_integrator:
                raise Exception("IEEE training integrator not available")
            
            logger.info("Starting IEEE training session with Brain integration")
            
            # Add Brain integration context to training config
            enhanced_config = training_config.copy()
            enhanced_config['brain_integration'] = {
                'enabled': True,
                'domain_id': self.domain_id,
                'integration_metrics': self.metrics.integration_id if self.metrics else None
            }
            
            # Start training session
            training_result = self.ieee_training_integrator.start_ieee_training(enhanced_config)
            
            # Register training session with Brain
            if training_result.get('success', False):
                brain_registration = {
                    "type": "training_session_started",
                    "domain_id": self.domain_id,
                    "session_id": training_result['session_id'],
                    "config": enhanced_config,
                    "data_info": training_result.get('data_info', {}),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Send to Brain system
                if self.status == IntegrationStatus.CONNECTED:
                    strategy = self.strategies[self.current_strategy]
                    self.event_loop.run_until_complete(
                        strategy.send_message(brain_registration)
                    )
                    
                    if self.metrics:
                        self.metrics.training_operations += 1
                        self.metrics.messages_sent += 1
            
            return training_result
            
        except Exception as e:
            logger.error(f"IEEE training session failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'ieee_training_error'
            }
    
    def get_ieee_data_quality_report(self) -> Dict[str, Any]:
        """
        Get IEEE dataset quality report for Brain monitoring
        
        Returns:
            Data quality report
        """
        try:
            if not self.ieee_data_loader:
                return {'error': 'IEEE data loader not available'}
            
            quality_report = self.ieee_data_loader.get_data_quality_report()
            
            # Add Brain integration context
            quality_report['brain_integration'] = {
                'domain_id': self.domain_id,
                'integration_status': self.status.value,
                'timestamp': datetime.now().isoformat()
            }
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Failed to get IEEE data quality report: {e}")
            return {
                'error': str(e),
                'error_type': 'data_quality_error'
            }
    
    def load_ieee_dataset_for_brain(self, dataset_type: str = "train", 
                                   validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Load IEEE dataset and prepare for Brain system integration
        
        Args:
            dataset_type: Type of dataset to load
            validation_split: Validation split ratio
            
        Returns:
            Dataset loading results
        """
        try:
            if not self.ieee_data_loader:
                return {'error': 'IEEE data loader not available'}
            
            logger.info(f"Loading IEEE {dataset_type} dataset for Brain integration")
            
            # Load dataset
            start_time = time.time()
            X, y, val_data = self.ieee_data_loader.load_data(
                dataset_type=dataset_type,
                validation_split=validation_split
            )
            load_time = time.time() - start_time
            
            # Prepare Brain integration metadata
            dataset_info = {
                'dataset_type': dataset_type,
                'samples': X.shape[0],
                'features': X.shape[1],
                'fraud_rate': float(y.mean()) if y is not None else None,
                'validation_samples': val_data[0].shape[0] if val_data else 0,
                'load_time_seconds': load_time,
                'data_quality': self.ieee_data_loader.get_data_quality_report(),
                'brain_integration': {
                    'domain_id': self.domain_id,
                    'loaded_at': datetime.now().isoformat(),
                    'integration_status': self.status.value
                }
            }
            
            # Notify Brain system of dataset loading
            if self.status == IntegrationStatus.CONNECTED:
                brain_notification = {
                    "type": "dataset_loaded",
                    "domain_id": self.domain_id,
                    "dataset_info": dataset_info,
                    "timestamp": datetime.now().isoformat()
                }
                
                strategy = self.strategies[self.current_strategy]
                self.event_loop.run_until_complete(
                    strategy.send_message(brain_notification)
                )
                
                if self.metrics:
                    self.metrics.messages_sent += 1
            
            return {
                'success': True,
                'dataset_info': dataset_info,
                'X_shape': X.shape,
                'y_shape': y.shape if y is not None else None,
                'validation_available': val_data is not None
            }
            
        except Exception as e:
            logger.error(f"Failed to load IEEE dataset for Brain: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'dataset_loading_error'
            }

    def __del__(self):
        """Cleanup on deletion"""
        try:
            if self.status == IntegrationStatus.CONNECTED:
                self.disconnect()
        except:
            pass


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for testing
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Create integration instance
    integration_config = {
        "brain_url": "brain.example.com",
        "brain_port": 8080,
        "auth_token": "test_token",
        "timeout": 30
    }
    
    integration = FinancialBrainIntegration(integration_config)
    
    print("\n=== Financial Brain Integration Test ===\n")
    
    # Test connection
    print("1. Testing connection to Brain system...")
    if integration.connect_to_brain():
        print("✓ Successfully connected to Brain system")
    else:
        print("✗ Failed to connect to Brain system")
    
    # Test integration with different strategies
    print("\n2. Testing integration strategies...")
    
    # Direct strategy
    print("   - Testing DIRECT strategy...")
    if integration.integrate_with_brain(strategy=IntegrationStrategy.DIRECT):
        print("   ✓ Direct integration successful")
    
    # Cached strategy
    print("   - Testing CACHED strategy...")
    if integration.integrate_with_brain(strategy=IntegrationStrategy.CACHED):
        print("   ✓ Cached integration successful")
    
    # Batch strategy
    print("   - Testing BATCH strategy...")
    if integration.integrate_with_brain(strategy=IntegrationStrategy.BATCH):
        print("   ✓ Batch integration successful")
    
    # Test state synchronization
    print("\n3. Testing state synchronization...")
    if integration.sync_state(SyncMode.FULL):
        print("✓ Full state sync successful")
    
    if integration.sync_state(SyncMode.INCREMENTAL):
        print("✓ Incremental state sync successful")
    
    # Test routing integration
    print("\n4. Testing routing integration...")
    custom_route = Route(
        route_id="custom_fraud_api",
        route_type=RouteType.QUERY,
        path="/api/v2/fraud/detect",
        handler="CustomFraudHandler",
        priority=15
    )
    
    if integration.integrate_routing([custom_route]):
        print("✓ Routing integration successful")
    
    # Test training integration
    print("\n5. Testing training integration...")
    training_config = {
        "models": ["advanced_fraud_detector"],
        "update_frequency": "hourly",
        "batch_size": 500
    }
    
    if integration.integrate_training(training_config):
        print("✓ Training integration successful")
    
    # Test validation
    print("\n6. Testing integration validation...")
    if integration.validate_integration():
        print("✓ Integration validation passed")
    else:
        print("✗ Integration validation failed")
    
    # Get metrics summary
    print("\n7. Integration Metrics Summary:")
    metrics = integration.get_metrics_summary()
    print(json.dumps(metrics, indent=2))
    
    # Get performance report
    print("\n8. Performance Report:")
    report = integration.get_performance_report()
    print(json.dumps(report, indent=2, default=str))
    
    # Test state persistence
    print("\n9. Testing state persistence...")
    if integration.save_state():
        print("✓ State saved successfully")
    
    # Test async integration
    print("\n10. Testing asynchronous integration...")
    if integration.integrate_with_brain(async_mode=True):
        print("✓ Asynchronous integration started")
    
    # Wait for async operations
    time.sleep(2)
    
    # Disconnect
    print("\n11. Testing disconnection...")
    if integration.disconnect():
        print("✓ Successfully disconnected from Brain system")
    
    print("\n=== Brain Integration Test Complete ===")

# IEEE Integration Methods for Brain System - Add to FinancialBrainIntegration class
def add_ieee_methods_to_brain_integration():
    """Add IEEE integration methods to FinancialBrainIntegration class"""
    
    def start_ieee_training_session(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start IEEE fraud detection training session integrated with Brain
        
        Args:
            training_config: Training configuration
            
        Returns:
            Training session information
        """
        try:
            if not self.ieee_training_integrator:
                raise Exception("IEEE training integrator not available")
            
            logger.info("Starting IEEE training session with Brain integration")
            
            # Add Brain integration context to training config
            enhanced_config = training_config.copy()
            enhanced_config['brain_integration'] = {
                'enabled': True,
                'domain_id': self.domain_id,
                'integration_metrics': self.metrics.integration_id if self.metrics else None
            }
            
            # Start training session
            training_result = self.ieee_training_integrator.start_ieee_training(enhanced_config)
            
            # Register training session with Brain
            if training_result.get('success', False):
                brain_registration = {
                    "type": "training_session_started",
                    "domain_id": self.domain_id,
                    "session_id": training_result['session_id'],
                    "config": enhanced_config,
                    "data_info": training_result.get('data_info', {}),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Send to Brain system
                if self.status == IntegrationStatus.CONNECTED:
                    strategy = self.strategies[self.current_strategy]
                    self.event_loop.run_until_complete(
                        strategy.send_message(brain_registration)
                    )
                    
                    if self.metrics:
                        self.metrics.training_operations += 1
                        self.metrics.messages_sent += 1
            
            return training_result
            
        except Exception as e:
            logger.error(f"IEEE training session failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'ieee_training_error'
            }
    
    def get_ieee_data_quality_report(self) -> Dict[str, Any]:
        """
        Get IEEE dataset quality report for Brain monitoring
        
        Returns:
            Data quality report
        """
        try:
            if not self.ieee_data_loader:
                return {'error': 'IEEE data loader not available'}
            
            quality_report = self.ieee_data_loader.get_data_quality_report()
            
            # Add Brain integration context
            quality_report['brain_integration'] = {
                'domain_id': self.domain_id,
                'integration_status': self.status.value,
                'timestamp': datetime.now().isoformat()
            }
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Failed to get IEEE data quality report: {e}")
            return {
                'error': str(e),
                'error_type': 'data_quality_error'
            }
    
    def load_ieee_dataset_for_brain(self, dataset_type: str = "train", 
                                   validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Load IEEE dataset and prepare for Brain system integration
        
        Args:
            dataset_type: Type of dataset to load
            validation_split: Validation split ratio
            
        Returns:
            Dataset loading results
        """
        try:
            if not self.ieee_data_loader:
                return {'error': 'IEEE data loader not available'}
            
            logger.info(f"Loading IEEE {dataset_type} dataset for Brain integration")
            
            # Load dataset
            start_time = time.time()
            X, y, val_data = self.ieee_data_loader.load_data(
                dataset_type=dataset_type,
                validation_split=validation_split
            )
            load_time = time.time() - start_time
            
            # Prepare Brain integration metadata
            dataset_info = {
                'dataset_type': dataset_type,
                'samples': X.shape[0],
                'features': X.shape[1],
                'fraud_rate': float(y.mean()) if y is not None else None,
                'validation_samples': val_data[0].shape[0] if val_data else 0,
                'load_time_seconds': load_time,
                'data_quality': self.ieee_data_loader.get_data_quality_report(),
                'brain_integration': {
                    'domain_id': self.domain_id,
                    'loaded_at': datetime.now().isoformat(),
                    'integration_status': self.status.value
                }
            }
            
            # Notify Brain system of dataset loading
            if self.status == IntegrationStatus.CONNECTED:
                brain_notification = {
                    "type": "dataset_loaded",
                    "domain_id": self.domain_id,
                    "dataset_info": dataset_info,
                    "timestamp": datetime.now().isoformat()
                }
                
                strategy = self.strategies[self.current_strategy]
                self.event_loop.run_until_complete(
                    strategy.send_message(brain_notification)
                )
                
                if self.metrics:
                    self.metrics.messages_sent += 1
            
            return {
                'success': True,
                'dataset_info': dataset_info,
                'X_shape': X.shape,
                'y_shape': y.shape if y is not None else None,
                'validation_available': val_data is not None
            }
            
        except Exception as e:
            logger.error(f"Failed to load IEEE dataset for Brain: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'dataset_loading_error'
            }
    
    def get_training_history_for_brain(self) -> Dict[str, Any]:
        """
        Get IEEE training history formatted for Brain system
        
        Returns:
            Training history with Brain integration metadata
        """
        try:
            if not self.ieee_training_integrator:
                return {'error': 'IEEE training integrator not available'}
            
            training_history = self.ieee_training_integrator.get_training_history()
            
            # Add Brain integration context
            brain_formatted_history = {
                'domain_id': self.domain_id,
                'integration_status': self.status.value,
                'total_sessions': len(training_history),
                'sessions': [],
                'summary': {
                    'successful_sessions': 0,
                    'failed_sessions': 0,
                    'total_training_time': 0,
                    'average_fraud_rate': 0
                },
                'timestamp': datetime.now().isoformat()
            }
            
            total_fraud_rate = 0
            for session in training_history:
                # Format session for Brain
                brain_session = {
                    'session_id': session['session_id'],
                    'status': session['status'],
                    'duration': session.get('duration', 0),
                    'data_info': session.get('data_info', {}),
                    'final_metrics': session.get('validation_metrics', {}),
                    'brain_integration': session.get('brain_integration', False)
                }
                brain_formatted_history['sessions'].append(brain_session)
                
                # Update summary
                if session['status'] == 'completed':
                    brain_formatted_history['summary']['successful_sessions'] += 1
                else:
                    brain_formatted_history['summary']['failed_sessions'] += 1
                
                brain_formatted_history['summary']['total_training_time'] += session.get('duration', 0)
                total_fraud_rate += session.get('data_info', {}).get('fraud_rate', 0)
            
            if training_history:
                brain_formatted_history['summary']['average_fraud_rate'] = total_fraud_rate / len(training_history)
            
            return brain_formatted_history
            
        except Exception as e:
            logger.error(f"Failed to get training history for Brain: {e}")
            return {
                'error': str(e),
                'error_type': 'training_history_error'
            }
    
    def clear_ieee_cache_for_brain(self) -> Dict[str, Any]:
        """
        Clear IEEE dataset cache and notify Brain system
        
        Returns:
            Cache clearing results
        """
        try:
            if not self.ieee_data_loader:
                return {'error': 'IEEE data loader not available'}
            
            # Clear cache
            cache_cleared = self.ieee_data_loader.clear_cache()
            
            # Notify Brain system
            if self.status == IntegrationStatus.CONNECTED:
                brain_notification = {
                    "type": "cache_cleared",
                    "domain_id": self.domain_id,
                    "cache_type": "ieee_dataset",
                    "success": cache_cleared,
                    "timestamp": datetime.now().isoformat()
                }
                
                strategy = self.strategies[self.current_strategy]
                self.event_loop.run_until_complete(
                    strategy.send_message(brain_notification)
                )
                
                if self.metrics:
                    self.metrics.messages_sent += 1
            
            return {
                'success': cache_cleared,
                'cache_type': 'ieee_dataset',
                'domain_id': self.domain_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to clear IEEE cache for Brain: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'cache_clearing_error'
            }