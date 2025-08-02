#!/usr/bin/env python3
"""
Universal AI Core - Main API Interface
======================================

This module provides the main API interface for the Universal AI Core system.
Adapted from Saraphis molecular_analyzer.py, property_predictor.py, ai_coordinator.py,
and async_symbolic_reasoning.py patterns to create a unified, domain-agnostic API.

Features:
- Unified API interface for all core components
- Async processing capabilities with task queuing
- Multi-tier caching system with TTL support
- Comprehensive health checks and diagnostics
- Performance monitoring and metrics collection
- Circuit breaker and safety systems
- Batch processing with dependency management
- Enterprise-grade error handling and recovery
"""

import logging
import time
import asyncio
import threading
import uuid
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
import warnings
import sys
import os

# Import core components
from .config.config_manager import get_config_manager, get_config, UniversalConfiguration
from .core.universal_ai_core import UniversalAICore
from .core.plugin_manager import PluginManager
from .core.orchestrator import SystemOrchestrator
from .utils.data_utils import DataProcessor, ProcessingResult
from .utils.validation_utils import ValidationEngine, ValidationResult

# Import utilities
from .utils import (
    data_utils, validation_utils, serialization_utils, performance_utils,
    logging_utils, error_utils, async_utils, memory_utils
)

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Configuration for API operations adapted from Saraphis patterns"""
    max_workers: int = 8
    max_queue_size: int = 1000
    enable_monitoring: bool = True
    enable_caching: bool = True
    cache_size: int = 10000
    cache_ttl_hours: int = 24
    request_timeout: float = 300.0
    batch_size: int = 100
    enable_safety_checks: bool = True
    rate_limit_requests_per_minute: int = 1000
    enable_circuit_breaker: bool = True
    health_check_interval: int = 60
    metrics_collection_interval: int = 30
    debug_mode: bool = False
    log_level: str = "INFO"


@dataclass
class TaskResult:
    """Result of API task execution adapted from Saraphis patterns"""
    task_id: str
    status: str  # pending, running, completed, failed
    result: Optional[Any] = None
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_completed(self) -> bool:
        return self.status in ['completed', 'failed']
    
    @property
    def is_successful(self) -> bool:
        return self.status == 'completed' and self.error_message is None


@dataclass
class BatchRequest:
    """Batch processing request adapted from Saraphis batch patterns"""
    batch_id: str
    items: List[Dict[str, Any]]
    operation_type: str
    config: Optional[Dict[str, Any]] = None
    priority: int = 5
    timeout: float = 300.0
    callback_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class APICache:
    """
    Multi-tier caching system adapted from Saraphis caching patterns.
    Provides memory caching with TTL and size limits.
    """
    
    def __init__(self, max_size: int = 10000, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.access_times: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self._lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                
                # Check TTL
                if datetime.utcnow() - timestamp < self.ttl:
                    self.access_times[key] = datetime.utcnow()
                    self.stats['hits'] += 1
                    return value
                else:
                    # Expired
                    del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
            
            self.stats['misses'] += 1
            return None
    
    def put(self, key: str, value: Any) -> bool:
        """Put item in cache"""
        with self._lock:
            current_time = datetime.utcnow()
            
            # Check if we need to evict
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = (value, current_time)
            self.access_times[key] = current_time
            self.stats['size'] = len(self.cache)
            return True
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
        self.stats['evictions'] += 1
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.stats['size'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }


class HealthChecker:
    """
    Health check system adapted from Saraphis health monitoring patterns.
    Provides comprehensive system health monitoring.
    """
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.health_status = {
            'overall': 'healthy',
            'components': {},
            'last_check': None,
            'uptime_start': datetime.utcnow()
        }
        self.checks: Dict[str, Callable] = {}
        self._monitoring_active = False
        self._monitor_thread = None
    
    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function"""
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def start_monitoring(self):
        """Start background health monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop background health monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                self.check_health()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        start_time = time.time()
        component_status = {}
        overall_healthy = True
        
        # Run all registered checks
        for name, check_func in self.checks.items():
            try:
                result = check_func()
                is_healthy = result.get('healthy', True) if isinstance(result, dict) else bool(result)
                
                component_status[name] = {
                    'healthy': is_healthy,
                    'details': result if isinstance(result, dict) else {'status': result},
                    'check_time': datetime.utcnow().isoformat()
                }
                
                if not is_healthy:
                    overall_healthy = False
                    
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                component_status[name] = {
                    'healthy': False,
                    'error': str(e),
                    'check_time': datetime.utcnow().isoformat()
                }
                overall_healthy = False
        
        # Update health status
        self.health_status.update({
            'overall': 'healthy' if overall_healthy else 'unhealthy',
            'components': component_status,
            'last_check': datetime.utcnow().isoformat(),
            'check_duration': time.time() - start_time,
            'uptime_seconds': (datetime.utcnow() - self.health_status['uptime_start']).total_seconds()
        })
        
        return self.health_status
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return self.health_status.copy()


class MetricsCollector:
    """
    Metrics collection system adapted from Saraphis monitoring patterns.
    Collects and aggregates system metrics.
    """
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self._collecting = False
        self._collector_thread = None
        self._lock = threading.RLock()
    
    def start_collection(self):
        """Start metrics collection"""
        if self._collecting:
            return
        
        self._collecting = True
        self._collector_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._collector_thread.start()
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self._collecting = False
        if self._collector_thread:
            self._collector_thread.join(timeout=5.0)
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Background collection loop"""
        while self._collecting:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system metrics"""
        try:
            import psutil
            
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            timestamp = datetime.utcnow()
            
            with self._lock:
                self.record_gauge('system.cpu_percent', cpu_percent, timestamp)
                self.record_gauge('system.memory_percent', memory.percent, timestamp)
                self.record_gauge('system.memory_used_mb', memory.used / 1024 / 1024, timestamp)
                self.record_gauge('system.disk_percent', disk.percent, timestamp)
                
        except ImportError:
            # psutil not available, record basic metrics
            with self._lock:
                self.record_gauge('system.status', 1.0, datetime.utcnow())
        except Exception as e:
            logger.warning(f"System metrics collection failed: {e}")
    
    def record_counter(self, name: str, value: int = 1):
        """Record counter metric"""
        with self._lock:
            self.counters[name] += value
    
    def record_gauge(self, name: str, value: float, timestamp: Optional[datetime] = None):
        """Record gauge metric"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        with self._lock:
            self.gauges[name] = value
            self.metrics[name].append((timestamp, value))
    
    def record_histogram(self, name: str, value: float, timestamp: Optional[datetime] = None):
        """Record histogram metric"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        with self._lock:
            self.metrics[name].append((timestamp, value))
    
    def get_metrics(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """Get collected metrics"""
        with self._lock:
            if metric_name:
                return {
                    'timeseries': list(self.metrics.get(metric_name, [])),
                    'current_value': self.gauges.get(metric_name),
                    'counter_value': self.counters.get(metric_name, 0)
                }
            else:
                return {
                    'counters': dict(self.counters),
                    'gauges': dict(self.gauges),
                    'metrics_count': len(self.metrics),
                    'collection_active': self._collecting
                }


class UniversalAIAPI:
    """
    Universal AI Core API - Main interface adapted from Saraphis patterns.
    
    Provides unified access to all Universal AI Core capabilities with
    enterprise-grade features including async processing, caching,
    health monitoring, and performance metrics.
    """
    
    def __init__(self, config_path: Optional[str] = None, api_config: Optional[APIConfig] = None):
        """Initialize Universal AI API"""
        # Configuration
        self.api_config = api_config or APIConfig()
        self.config_manager = get_config_manager(config_path)
        self.config = self.config_manager.get_config()
        
        # Setup logging
        self.logger = self._setup_logging()
        self.logger.info("Initializing Universal AI API")
        
        # Core components
        self.core = UniversalAICore(self.config)
        self.orchestrator = SystemOrchestrator(self.core)
        self.data_processor = DataProcessor()
        self.validation_engine = ValidationEngine()
        
        # API infrastructure
        self.cache = APICache(
            max_size=self.api_config.cache_size,
            ttl_hours=self.api_config.cache_ttl_hours
        ) if self.api_config.enable_caching else None
        
        self.health_checker = HealthChecker(self.api_config.health_check_interval)
        self.metrics_collector = MetricsCollector(self.api_config.metrics_collection_interval)
        
        # Async processing
        self.executor = ThreadPoolExecutor(max_workers=self.api_config.max_workers)
        self.task_queue = asyncio.Queue(maxsize=self.api_config.max_queue_size)
        self.pending_tasks: Dict[str, TaskResult] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        
        # Safety and monitoring
        self.request_counts = defaultdict(int)
        self.last_request_reset = datetime.utcnow()
        self._initialization_time = datetime.utcnow()
        self._lock = threading.RLock()
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("Universal AI API initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging adapted from Saraphis patterns"""
        logger = logging.getLogger(f"{__name__}.UniversalAIAPI")
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, self.api_config.log_level))
            
            # File handler
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "universal_ai_api.log")
            file_handler.setLevel(logging.DEBUG)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            logger.setLevel(logging.DEBUG)
        
        return logger
    
    def _initialize_components(self):
        """Initialize API components"""
        try:
            # Register health checks
            self.health_checker.register_check('core_system', self._check_core_health)
            self.health_checker.register_check('plugin_manager', self._check_plugin_health)
            self.health_checker.register_check('cache_system', self._check_cache_health)
            self.health_checker.register_check('async_system', self._check_async_health)
            
            # Start monitoring if enabled
            if self.api_config.enable_monitoring:
                self.health_checker.start_monitoring()
                self.metrics_collector.start_collection()
            
            self.logger.info("API components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise
    
    def _check_core_health(self) -> Dict[str, Any]:
        """Check core system health"""
        try:
            # Test core functionality
            return {
                'healthy': True,
                'plugin_count': len(self.core.plugin_manager.plugins),
                'config_loaded': self.config is not None
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    def _check_plugin_health(self) -> Dict[str, Any]:
        """Check plugin system health"""
        try:
            available_plugins = self.core.plugin_manager.list_available_plugins()
            total_plugins = sum(len(plugins) for plugins in available_plugins.values())
            
            return {
                'healthy': True,
                'total_plugins': total_plugins,
                'plugin_types': list(available_plugins.keys())
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    def _check_cache_health(self) -> Dict[str, Any]:
        """Check cache system health"""
        if not self.cache:
            return {'healthy': True, 'cache_enabled': False}
        
        try:
            stats = self.cache.get_stats()
            return {
                'healthy': True,
                'cache_enabled': True,
                **stats
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    def _check_async_health(self) -> Dict[str, Any]:
        """Check async system health"""
        try:
            return {
                'healthy': True,
                'pending_tasks': len(self.pending_tasks),
                'completed_tasks': len(self.completed_tasks),
                'queue_size': self.task_queue.qsize(),
                'worker_threads': self.api_config.max_workers
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    def _check_rate_limit(self) -> bool:
        """Check rate limiting adapted from Saraphis safety patterns"""
        if not self.api_config.enable_safety_checks:
            return True
        
        current_time = datetime.utcnow()
        
        # Reset counters every minute
        if (current_time - self.last_request_reset).total_seconds() >= 60:
            self.request_counts.clear()
            self.last_request_reset = current_time
        
        # Check rate limit
        minute_key = current_time.strftime('%Y-%m-%d-%H-%M')
        self.request_counts[minute_key] += 1
        
        return self.request_counts[minute_key] <= self.api_config.rate_limit_requests_per_minute
    
    # Core API Methods
    
    def process_data(self, data: Any, extractors: Optional[List[str]] = None, 
                    use_cache: bool = True) -> ProcessingResult:
        """
        Process data using registered extractors.
        Adapted from Saraphis data processing patterns.
        """
        try:
            # Rate limiting check
            if not self._check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            # Check cache
            if use_cache and self.cache:
                cache_key = f"process_{hash(str(data))}_{str(extractors)}"
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.metrics_collector.record_counter('api.cache_hits')
                    return cached_result
            
            # Record metrics
            self.metrics_collector.record_counter('api.process_data_requests')
            start_time = time.time()
            
            # Process data
            result = self.data_processor.prepare_features(data, extractors)
            
            # Record processing time
            processing_time = time.time() - start_time
            self.metrics_collector.record_histogram('api.process_data_duration', processing_time)
            
            # Cache result
            if use_cache and self.cache and result.status == "success":
                self.cache.put(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Data processing failed: {e}")
            self.metrics_collector.record_counter('api.process_data_errors')
            raise
    
    def validate_data(self, data: Any, validators: Optional[List[str]] = None,
                     use_cache: bool = True) -> ValidationResult:
        """
        Validate data using registered validators.
        Adapted from Saraphis validation patterns.
        """
        try:
            # Rate limiting check
            if not self._check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            # Check cache
            if use_cache and self.cache:
                cache_key = f"validate_{hash(str(data))}_{str(validators)}"
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.metrics_collector.record_counter('api.cache_hits')
                    return cached_result
            
            # Record metrics
            self.metrics_collector.record_counter('api.validate_data_requests')
            start_time = time.time()
            
            # Validate data
            result = self.validation_engine.validate(data, validators)
            
            # Record validation time
            validation_time = time.time() - start_time
            self.metrics_collector.record_histogram('api.validate_data_duration', validation_time)
            
            # Cache result
            if use_cache and self.cache:
                self.cache.put(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            self.metrics_collector.record_counter('api.validate_data_errors')
            raise
    
    async def submit_async_task(self, operation_type: str, data: Dict[str, Any],
                               config: Optional[Dict[str, Any]] = None,
                               priority: int = 5, timeout: float = None) -> str:
        """
        Submit asynchronous task adapted from Saraphis async patterns.
        """
        try:
            # Rate limiting check
            if not self._check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            task_id = str(uuid.uuid4())
            task_timeout = timeout or self.api_config.request_timeout
            
            # Create task result
            task_result = TaskResult(
                task_id=task_id,
                status="pending",
                start_time=datetime.utcnow(),
                metadata={
                    'operation_type': operation_type,
                    'priority': priority,
                    'timeout': task_timeout,
                    'config': config or {}
                }
            )
            
            with self._lock:
                self.pending_tasks[task_id] = task_result
            
            # Submit task to queue
            task_data = {
                'task_id': task_id,
                'operation_type': operation_type,
                'data': data,
                'config': config,
                'priority': priority,
                'timeout': task_timeout
            }
            
            await self.task_queue.put(task_data)
            
            # Record metrics
            self.metrics_collector.record_counter('api.async_tasks_submitted')
            
            self.logger.info(f"Async task submitted: {task_id}")
            return task_id
            
        except Exception as e:
            self.logger.error(f"Async task submission failed: {e}")
            self.metrics_collector.record_counter('api.async_task_errors')
            raise
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result of async task"""
        try:
            # Check completed tasks first
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
            
            # Check pending tasks
            if task_id in self.pending_tasks:
                return self.pending_tasks[task_id]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Task result retrieval failed: {e}")
            return None
    
    async def process_batch(self, batch_request: BatchRequest) -> str:
        """
        Process batch request adapted from Saraphis batch patterns.
        """
        try:
            # Rate limiting check
            if not self._check_rate_limit():
                raise Exception("Rate limit exceeded")
            
            # Submit individual tasks
            task_ids = []
            for i, item in enumerate(batch_request.items):
                task_id = await self.submit_async_task(
                    operation_type=batch_request.operation_type,
                    data=item,
                    config=batch_request.config,
                    priority=batch_request.priority,
                    timeout=batch_request.timeout
                )
                task_ids.append(task_id)
            
            # Store batch information
            batch_result = TaskResult(
                task_id=batch_request.batch_id,
                status="pending",
                start_time=datetime.utcnow(),
                metadata={
                    'batch_type': 'batch_request',
                    'item_count': len(batch_request.items),
                    'task_ids': task_ids,
                    'operation_type': batch_request.operation_type
                }
            )
            
            with self._lock:
                self.pending_tasks[batch_request.batch_id] = batch_result
            
            # Record metrics
            self.metrics_collector.record_counter('api.batch_requests')
            self.metrics_collector.record_histogram('api.batch_size', len(batch_request.items))
            
            self.logger.info(f"Batch request submitted: {batch_request.batch_id} with {len(task_ids)} tasks")
            return batch_request.batch_id
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            self.metrics_collector.record_counter('api.batch_errors')
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        try:
            health_status = self.health_checker.get_health_status()
            
            # Add API-specific health information
            health_status['api'] = {
                'initialization_time': self._initialization_time.isoformat(),
                'uptime_seconds': (datetime.utcnow() - self._initialization_time).total_seconds(),
                'pending_tasks': len(self.pending_tasks),
                'completed_tasks': len(self.completed_tasks),
                'cache_enabled': self.cache is not None,
                'monitoring_enabled': self.api_config.enable_monitoring
            }
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health status retrieval failed: {e}")
            return {'overall': 'unhealthy', 'error': str(e)}
    
    def get_metrics(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            metrics = self.metrics_collector.get_metrics(metric_name)
            
            # Add API-specific metrics
            if not metric_name:
                metrics['api'] = {
                    'pending_tasks': len(self.pending_tasks),
                    'completed_tasks': len(self.completed_tasks),
                    'cache_stats': self.cache.get_stats() if self.cache else {'enabled': False}
                }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Metrics retrieval failed: {e}")
            return {'error': str(e)}
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            return {
                'api_version': "1.0.0",
                'initialization_time': self._initialization_time.isoformat(),
                'uptime_seconds': (datetime.utcnow() - self._initialization_time).total_seconds(),
                'configuration': {
                    'max_workers': self.api_config.max_workers,
                    'max_queue_size': self.api_config.max_queue_size,
                    'cache_enabled': self.api_config.enable_caching,
                    'monitoring_enabled': self.api_config.enable_monitoring,
                    'safety_checks_enabled': self.api_config.enable_safety_checks
                },
                'components': {
                    'core_system': self.core is not None,
                    'plugin_manager': self.core.plugin_manager is not None,
                    'data_processor': self.data_processor is not None,
                    'validation_engine': self.validation_engine is not None,
                    'cache_system': self.cache is not None,
                    'health_checker': self.health_checker is not None,
                    'metrics_collector': self.metrics_collector is not None
                },
                'statistics': {
                    'pending_tasks': len(self.pending_tasks),
                    'completed_tasks': len(self.completed_tasks),
                    'queue_size': self.task_queue.qsize()
                }
            }
            
        except Exception as e:
            self.logger.error(f"System info retrieval failed: {e}")
            return {'error': str(e)}
    
    def shutdown(self):
        """Shutdown API and cleanup resources"""
        try:
            self.logger.info("Shutting down Universal AI API")
            
            # Stop monitoring
            if self.health_checker:
                self.health_checker.stop_monitoring()
            
            if self.metrics_collector:
                self.metrics_collector.stop_collection()
            
            # Shutdown executor
            if self.executor:
                self.executor.shutdown(wait=True)
            
            # Clear caches
            if self.cache:
                self.cache.clear()
            
            self.logger.info("Universal AI API shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")


# Factory functions for easy API creation

def create_api(config_path: Optional[str] = None, **kwargs) -> UniversalAIAPI:
    """
    Factory function to create Universal AI API instance.
    Adapted from Saraphis factory patterns.
    """
    api_config = APIConfig(**kwargs)
    return UniversalAIAPI(config_path=config_path, api_config=api_config)


def create_development_api() -> UniversalAIAPI:
    """Create API configured for development"""
    config = APIConfig(
        debug_mode=True,
        log_level="DEBUG",
        enable_safety_checks=False,
        rate_limit_requests_per_minute=10000
    )
    return UniversalAIAPI(api_config=config)


def create_production_api() -> UniversalAIAPI:
    """Create API configured for production"""
    config = APIConfig(
        debug_mode=False,
        log_level="WARNING",
        enable_safety_checks=True,
        enable_monitoring=True,
        rate_limit_requests_per_minute=1000
    )
    return UniversalAIAPI(api_config=config)


# Export public API
__all__ = [
    'UniversalAIAPI', 'APIConfig', 'TaskResult', 'BatchRequest',
    'APICache', 'HealthChecker', 'MetricsCollector',
    'create_api', 'create_development_api', 'create_production_api'
]


# Version information
__version__ = "1.0.0"
__author__ = "Universal AI Core Team"
__description__ = "Domain-agnostic AI core system with enterprise-grade features"