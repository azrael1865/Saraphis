#!/usr/bin/env python3
"""
Universal AI Core Coordinator
=============================

This module provides the main coordination framework for the Universal AI Core system.
Extracted and adapted from the Saraphis coordination patterns, made domain-agnostic while preserving
all sophisticated orchestration capabilities.

Features:
- Multi-engine coordination and orchestration
- Async task management and execution
- Error handling and recovery mechanisms
- Performance monitoring and optimization
- Result aggregation and intelligent caching
- Health checking and diagnostics
- Load balancing and resource management
- Comprehensive logging and metrics
"""

import asyncio
import json
import logging
import hashlib
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import weakref
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import traceback

# Import core engines
from .proof_engine import ProofEngine
from .symbolic_engine import SymbolicReasoningEngine
from .data_engine import DataEngine

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations the coordinator can handle"""
    PROOF_VERIFICATION = "proof_verification"
    SYMBOLIC_REASONING = "symbolic_reasoning"
    DATA_PROCESSING = "data_processing"
    FEATURE_ENGINEERING = "feature_engineering"
    VALIDATION = "validation"
    ANALYSIS = "analysis"
    PREDICTION = "prediction"
    OPTIMIZATION = "optimization"


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class Priority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Task:
    """Task representation for the coordinator"""
    id: str
    operation_type: OperationType
    operation_data: Dict[str, Any]
    priority: Priority = Priority.NORMAL
    timeout: float = 300.0
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def execution_time(self) -> Optional[float]:
        """Get task execution time in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def age(self) -> float:
        """Get task age in seconds"""
        return (datetime.utcnow() - self.created_at).total_seconds()


@dataclass
class CoordinationResult:
    """Result container for coordinated operations"""
    task_id: str
    operation_type: OperationType
    status: TaskStatus
    result: Optional[Any]
    execution_time: float
    memory_used: float
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sub_results: Dict[str, Any] = field(default_factory=dict)


class HealthStatus(Enum):
    """System health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class SystemHealth:
    """System health information"""
    overall_status: HealthStatus
    engine_status: Dict[str, HealthStatus]
    memory_usage: float
    cpu_usage: float
    active_tasks: int
    queue_size: int
    error_rate: float
    response_time: float
    uptime: float
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class UniversalAICoordinator:
    """
    Main coordination framework for the Universal AI Core system.
    
    Extracted and adapted from charon_builder ai_coordinator.py coordination patterns,
    made domain-agnostic while preserving sophisticated orchestration capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Universal AI Coordinator"""
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.UniversalAICoordinator")
        
        # Core engines
        self.proof_engine = None
        self.symbolic_engine = None
        self.data_engine = None
        
        # Task management
        self.task_queue = asyncio.PriorityQueue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        # Coordination state
        self.running = False
        self.workers = []
        self.max_workers = self.config.get('max_workers', 8)
        self.max_queue_size = self.config.get('max_queue_size', 1000)
        
        # Performance monitoring
        self.metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
            'average_response_time': 0.0,
            'peak_memory_usage': 0.0,
            'error_rate': 0.0,
            'cache_hit_rate': 0.0
        }
        
        # Caching system
        self.result_cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 1 hour default
        self.max_cache_size = self.config.get('max_cache_size', 10000)
        
        # Health monitoring
        self.health_check_interval = self.config.get('health_check_interval', 60)
        self.health_monitor_task = None
        self.last_health_check = None
        
        # Error handling
        self.circuit_breaker = CircuitBreaker()
        self.retry_strategies = self._initialize_retry_strategies()
        
        # Resource monitoring
        self.memory_threshold = self.config.get('memory_threshold_mb', 8192)
        self.cpu_threshold = self.config.get('cpu_threshold_percent', 80)
        
        self.logger.info("🎛️ Universal AI Coordinator initialized")
    
    async def initialize(self):
        """Initialize all core engines"""
        try:
            self.logger.info("🚀 Initializing Universal AI Core engines...")
            
            # Initialize engines
            engine_config = self.config.get('engines', {})
            
            self.proof_engine = ProofEngine(engine_config.get('proof', {}))
            await self.proof_engine.start()
            
            self.symbolic_engine = SymbolicReasoningEngine(engine_config.get('symbolic', {}))
            
            self.data_engine = DataEngine(engine_config.get('data', {}))
            await self.data_engine.start()
            
            self.logger.info("✅ All engines initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize engines: {e}")
            raise
    
    async def start(self):
        """Start the coordinator and all worker processes"""
        if self.running:
            self.logger.warning("Coordinator is already running")
            return
        
        try:
            # Initialize engines if not already done
            if not self.proof_engine:
                await self.initialize()
            
            self.running = True
            
            # Start worker tasks
            for i in range(self.max_workers):
                worker = asyncio.create_task(
                    self._worker(f"worker-{i}"),
                    name=f"coordinator-worker-{i}"
                )
                self.workers.append(worker)
            
            # Start health monitoring
            self.health_monitor_task = asyncio.create_task(
                self._health_monitor(),
                name="health-monitor"
            )
            
            # Start cache cleanup
            asyncio.create_task(self._cache_cleanup_loop(), name="cache-cleanup")
            
            self.logger.info(f"🎛️ Coordinator started with {self.max_workers} workers")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start coordinator: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the coordinator and cleanup resources"""
        if not self.running:
            return
        
        self.running = False
        
        try:
            # Cancel all workers
            for worker in self.workers:
                worker.cancel()
            
            # Cancel health monitor
            if self.health_monitor_task:
                self.health_monitor_task.cancel()
            
            # Wait for workers to finish
            if self.workers:
                await asyncio.gather(*self.workers, return_exceptions=True)
            
            # Stop engines
            if self.proof_engine:
                await self.proof_engine.stop()
            
            if self.symbolic_engine:
                self.symbolic_engine.shutdown()
            
            if self.data_engine:
                await self.data_engine.stop()
            
            self.logger.info("🛑 Coordinator stopped successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error during coordinator shutdown: {e}")
    
    async def submit_task(self, operation_type: OperationType, 
                         operation_data: Dict[str, Any],
                         priority: Priority = Priority.NORMAL,
                         timeout: float = 300.0,
                         **kwargs) -> str:
        """Submit a task for execution"""
        if not self.running:
            raise RuntimeError("Coordinator is not running")
        
        # Check queue size
        if self.task_queue.qsize() >= self.max_queue_size:
            raise RuntimeError("Task queue is full")
        
        # Create task
        task = Task(
            id=self._generate_task_id(),
            operation_type=operation_type,
            operation_data=operation_data,
            priority=priority,
            timeout=timeout,
            **kwargs
        )
        
        # Check cache first
        cache_key = self._generate_cache_key(operation_type, operation_data)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            self.logger.info(f"🎯 Using cached result for task: {task.id}")
            self.metrics['cache_hit_rate'] = (
                self.metrics['cache_hit_rate'] * 0.9 + 0.1
            )  # Exponential moving average
            
            # Create completed task
            task.status = TaskStatus.COMPLETED
            task.result = cached_result
            task.completed_at = datetime.utcnow()
            self.completed_tasks[task.id] = task
            
            return task.id
        
        # Add to queue (priority queue uses negative priority for max-heap)
        await self.task_queue.put((-priority.value, task.created_at, task))
        
        self.logger.info(f"📋 Task submitted: {task.id} ({operation_type.value})")
        return task.id
    
    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> CoordinationResult:
        """Get task result, waiting if necessary"""
        start_time = time.time()
        
        while True:
            # Check completed tasks
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                return self._create_coordination_result(task)
            
            # Check failed tasks
            if task_id in self.failed_tasks:
                task = self.failed_tasks[task_id]
                return self._create_coordination_result(task)
            
            # Check active tasks
            if task_id in self.active_tasks:
                if timeout and (time.time() - start_time) > timeout:
                    raise asyncio.TimeoutError(f"Timeout waiting for task {task_id}")
                await asyncio.sleep(0.1)
                continue
            
            # Task not found
            raise ValueError(f"Task {task_id} not found")
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get current status of a task"""
        # Check all task dictionaries
        for task_dict in [self.active_tasks, self.completed_tasks, self.failed_tasks]:
            if task_id in task_dict:
                return task_dict[task_id].status
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        # Check if task is active
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            self.logger.info(f"🚫 Task cancelled: {task_id}")
            return True
        
        return False
    
    async def _worker(self, worker_name: str):
        """Worker coroutine for processing tasks"""
        self.logger.info(f"🔧 Starting worker: {worker_name}")
        
        while self.running:
            try:
                # Get task from queue with timeout
                try:
                    priority, created_at, task = await asyncio.wait_for(
                        self.task_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Move task to active
                self.active_tasks[task.id] = task
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.utcnow()
                
                self.logger.info(f"🔍 {worker_name} processing task: {task.id}")
                
                try:
                    # Execute task with timeout
                    result = await asyncio.wait_for(
                        self._execute_task(task),
                        timeout=task.timeout
                    )
                    
                    # Task completed successfully
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = datetime.utcnow()
                    
                    # Move to completed
                    self.completed_tasks[task.id] = task
                    del self.active_tasks[task.id]
                    
                    # Cache result
                    cache_key = self._generate_cache_key(task.operation_type, task.operation_data)
                    self._cache_result(cache_key, result)
                    
                    # Update metrics
                    self.metrics['tasks_completed'] += 1
                    if task.execution_time:
                        self.metrics['total_execution_time'] += task.execution_time
                        self.metrics['average_response_time'] = (
                            self.metrics['total_execution_time'] / 
                            max(self.metrics['tasks_completed'], 1)
                        )
                    
                    self.logger.info(f"✅ {worker_name} completed task: {task.id}")
                    
                except asyncio.TimeoutError:
                    # Task timed out
                    task.status = TaskStatus.TIMEOUT
                    task.error = f"Task timed out after {task.timeout}s"
                    task.completed_at = datetime.utcnow()
                    
                    self.failed_tasks[task.id] = task
                    del self.active_tasks[task.id]
                    
                    self.metrics['tasks_failed'] += 1
                    self.logger.warning(f"⏰ {worker_name} task timeout: {task.id}")
                    
                except Exception as e:
                    # Task failed
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.utcnow()
                    
                    # Retry logic
                    if task.retry_count < task.max_retries and self._should_retry(e):
                        task.retry_count += 1
                        task.status = TaskStatus.PENDING
                        task.started_at = None
                        task.error = None
                        
                        # Re-queue with exponential backoff
                        delay = 2 ** task.retry_count
                        await asyncio.sleep(delay)
                        await self.task_queue.put((-task.priority.value, task.created_at, task))
                        
                        del self.active_tasks[task.id]
                        self.logger.info(f"🔄 Retrying task {task.id} (attempt {task.retry_count})")
                    else:
                        # Max retries reached or non-retryable error
                        self.failed_tasks[task.id] = task
                        del self.active_tasks[task.id]
                        
                        self.metrics['tasks_failed'] += 1
                        self.logger.error(f"❌ {worker_name} task failed: {task.id} - {e}")
                
                finally:
                    self.task_queue.task_done()
                    
            except asyncio.CancelledError:
                self.logger.info(f"🛑 Worker {worker_name} cancelled")
                break
            except Exception as e:
                self.logger.error(f"❌ Unexpected error in worker {worker_name}: {e}")
                await asyncio.sleep(1)  # Prevent tight error loop
    
    async def _execute_task(self, task: Task) -> Any:
        """Execute a task based on its operation type"""
        operation_type = task.operation_type
        operation_data = task.operation_data
        
        try:
            if operation_type == OperationType.PROOF_VERIFICATION:
                return await self._execute_proof_verification(operation_data)
            
            elif operation_type == OperationType.SYMBOLIC_REASONING:
                return await self._execute_symbolic_reasoning(operation_data)
            
            elif operation_type == OperationType.DATA_PROCESSING:
                return await self._execute_data_processing(operation_data)
            
            elif operation_type == OperationType.FEATURE_ENGINEERING:
                return await self._execute_feature_engineering(operation_data)
            
            elif operation_type == OperationType.VALIDATION:
                return await self._execute_validation(operation_data)
            
            elif operation_type == OperationType.ANALYSIS:
                return await self._execute_analysis(operation_data)
            
            else:
                raise ValueError(f"Unknown operation type: {operation_type}")
                
        except Exception as e:
            self.logger.error(f"❌ Task execution failed: {e}")
            raise
    
    async def _execute_proof_verification(self, operation_data: Dict[str, Any]) -> Any:
        """Execute proof verification operation"""
        if not self.proof_engine:
            raise RuntimeError("Proof engine not initialized")
        
        proof_text = operation_data.get('proof_text')
        language = operation_data.get('language', 'neuroformal')
        
        if not proof_text:
            raise ValueError("proof_text is required for proof verification")
        
        from .proof_engine import ProofLanguage
        proof_lang = ProofLanguage(language) if isinstance(language, str) else language
        
        proof_id = await self.proof_engine.submit_proof(proof_text, proof_lang)
        result = await self.proof_engine.get_verification_result(proof_id)
        
        return {
            'proof_id': proof_id,
            'verification_result': result,
            'verified': result.verified if result else False
        }
    
    async def _execute_symbolic_reasoning(self, operation_data: Dict[str, Any]) -> Any:
        """Execute symbolic reasoning operation"""
        if not self.symbolic_engine:
            raise RuntimeError("Symbolic engine not initialized")
        
        result = await self.symbolic_engine.symbolic_reasoning(operation_data)
        return {
            'result_id': result.result_id,
            'operation_type': result.operation_type.value,
            'output_data': result.output_data,
            'confidence': result.confidence,
            'reasoning_chain': result.reasoning_chain
        }
    
    async def _execute_data_processing(self, operation_data: Dict[str, Any]) -> Any:
        """Execute data processing operation"""
        if not self.data_engine:
            raise RuntimeError("Data engine not initialized")
        
        data_source = operation_data.get('data_source')
        operations = operation_data.get('operations', [])
        
        if not data_source:
            raise ValueError("data_source is required for data processing")
        
        # Load data
        load_result = await self.data_engine.load_data(data_source)
        if load_result.status.value != 'completed':
            raise RuntimeError(f"Failed to load data: {load_result.errors}")
        
        # Process data
        if operations:
            process_result = await self.data_engine.process_batch(load_result.data, operations)
            return {
                'load_result': load_result,
                'process_result': process_result,
                'final_data_shape': process_result.data.shape if process_result.data is not None else None
            }
        else:
            return {
                'load_result': load_result,
                'data_shape': load_result.data.shape if load_result.data is not None else None
            }
    
    async def _execute_feature_engineering(self, operation_data: Dict[str, Any]) -> Any:
        """Execute feature engineering operation"""
        data = operation_data.get('data')
        transformations = operation_data.get('transformations', [])
        target_column = operation_data.get('target_column')
        
        if data is None:
            raise ValueError("data is required for feature engineering")
        
        result = await self.data_engine.feature_engineer.engineer_features(
            data, transformations, target_column
        )
        
        return {
            'engineered_data': result.data,
            'features_generated': result.features_generated,
            'processing_time': result.processing_time,
            'warnings': result.warnings
        }
    
    async def _execute_validation(self, operation_data: Dict[str, Any]) -> Any:
        """Execute data validation operation"""
        data = operation_data.get('data')
        schema = operation_data.get('schema')
        validation_level = operation_data.get('validation_level', 'basic')
        
        if data is None:
            raise ValueError("data is required for validation")
        
        from .data_engine import ValidationLevel
        level = ValidationLevel(validation_level)
        
        result = await self.data_engine.data_validator.validate_data(data, schema, level)
        
        return {
            'validation_result': result,
            'is_valid': result.status.value == 'completed',
            'errors': result.errors,
            'warnings': result.warnings
        }
    
    async def _execute_analysis(self, operation_data: Dict[str, Any]) -> Any:
        """Execute combined analysis operation"""
        # This could coordinate multiple engines for complex analysis
        components = operation_data.get('components', [])
        results = {}
        
        for component in components:
            comp_type = component.get('type')
            comp_data = component.get('data', {})
            
            if comp_type == 'symbolic':
                results['symbolic'] = await self._execute_symbolic_reasoning(comp_data)
            elif comp_type == 'data':
                results['data'] = await self._execute_data_processing(comp_data)
            elif comp_type == 'proof':
                results['proof'] = await self._execute_proof_verification(comp_data)
        
        return {
            'analysis_results': results,
            'components_processed': len(components)
        }
    
    async def _health_monitor(self):
        """Background health monitoring"""
        while self.running:
            try:
                health = await self.get_system_health()
                self.last_health_check = datetime.utcnow()
                
                # Log health issues
                if health.overall_status != HealthStatus.HEALTHY:
                    self.logger.warning(f"🏥 System health: {health.overall_status.value}")
                    for issue in health.issues:
                        self.logger.warning(f"  Issue: {issue}")
                
                # Update error rate
                total_tasks = self.metrics['tasks_completed'] + self.metrics['tasks_failed']
                if total_tasks > 0:
                    self.metrics['error_rate'] = self.metrics['tasks_failed'] / total_tasks
                
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Health monitor error: {e}")
                await asyncio.sleep(10)
    
    async def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health information"""
        try:
            # System metrics
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            cpu_usage = process.cpu_percent()
            
            # Update peak memory
            self.metrics['peak_memory_usage'] = max(
                self.metrics['peak_memory_usage'], memory_usage
            )
            
            # Engine health
            engine_status = {}
            issues = []
            recommendations = []
            
            # Check proof engine
            if self.proof_engine:
                proof_stats = self.proof_engine.get_proof_statistics()
                if proof_stats['running']:
                    engine_status['proof'] = HealthStatus.HEALTHY
                else:
                    engine_status['proof'] = HealthStatus.UNHEALTHY
                    issues.append("Proof engine not running")
            else:
                engine_status['proof'] = HealthStatus.UNHEALTHY
                issues.append("Proof engine not initialized")
            
            # Check symbolic engine
            if self.symbolic_engine:
                symbolic_status = self.symbolic_engine.get_symbolic_status()
                if symbolic_status['processing_active']:
                    engine_status['symbolic'] = HealthStatus.HEALTHY
                else:
                    engine_status['symbolic'] = HealthStatus.UNHEALTHY
                    issues.append("Symbolic engine not active")
            else:
                engine_status['symbolic'] = HealthStatus.UNHEALTHY
                issues.append("Symbolic engine not initialized")
            
            # Check data engine
            if self.data_engine:
                data_stats = self.data_engine.get_statistics()
                if data_stats['running']:
                    engine_status['data'] = HealthStatus.HEALTHY
                else:
                    engine_status['data'] = HealthStatus.UNHEALTHY
                    issues.append("Data engine not running")
            else:
                engine_status['data'] = HealthStatus.UNHEALTHY
                issues.append("Data engine not initialized")
            
            # Check resource usage
            if memory_usage > self.memory_threshold:
                issues.append(f"High memory usage: {memory_usage:.1f}MB")
                recommendations.append("Consider reducing cache sizes or batch sizes")
            
            if cpu_usage > self.cpu_threshold:
                issues.append(f"High CPU usage: {cpu_usage:.1f}%")
                recommendations.append("Consider reducing worker count or task complexity")
            
            # Check queue health
            queue_size = self.task_queue.qsize()
            if queue_size > self.max_queue_size * 0.8:
                issues.append(f"High queue size: {queue_size}")
                recommendations.append("Consider increasing worker count")
            
            # Check error rate
            if self.metrics['error_rate'] > 0.1:  # 10% error rate
                issues.append(f"High error rate: {self.metrics['error_rate']:.2%}")
                recommendations.append("Check logs for recurring errors")
            
            # Determine overall status
            unhealthy_engines = sum(1 for status in engine_status.values() 
                                  if status == HealthStatus.UNHEALTHY)
            
            if unhealthy_engines == 0 and not issues:
                overall_status = HealthStatus.HEALTHY
            elif unhealthy_engines <= 1 and len(issues) <= 2:
                overall_status = HealthStatus.DEGRADED
            elif unhealthy_engines <= 2 and len(issues) <= 5:
                overall_status = HealthStatus.UNHEALTHY
            else:
                overall_status = HealthStatus.CRITICAL
            
            # Calculate uptime
            uptime = time.time() - getattr(self, '_start_time', time.time())
            
            return SystemHealth(
                overall_status=overall_status,
                engine_status=engine_status,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                active_tasks=len(self.active_tasks),
                queue_size=queue_size,
                error_rate=self.metrics['error_rate'],
                response_time=self.metrics['average_response_time'],
                uptime=uptime,
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error getting system health: {e}")
            return SystemHealth(
                overall_status=HealthStatus.CRITICAL,
                engine_status={},
                memory_usage=0.0,
                cpu_usage=0.0,
                active_tasks=0,
                queue_size=0,
                error_rate=1.0,
                response_time=0.0,
                uptime=0.0,
                issues=[f"Health check failed: {e}"]
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            **self.metrics,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'queue_size': self.task_queue.qsize(),
            'cache_size': len(self.result_cache),
            'running': self.running,
            'workers': len(self.workers),
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None
        }
    
    def clear_cache(self):
        """Clear result cache"""
        self.result_cache.clear()
        gc.collect()
        self.logger.info("🧹 Cleared coordinator cache")
    
    async def _cache_cleanup_loop(self):
        """Background cache cleanup"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                # Remove expired cache entries
                current_time = time.time()
                expired_keys = []
                
                for key, (result, timestamp) in self.result_cache.items():
                    if current_time - timestamp > self.cache_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.result_cache[key]
                
                if expired_keys:
                    self.logger.info(f"🧹 Removed {len(expired_keys)} expired cache entries")
                
                # Limit cache size
                if len(self.result_cache) > self.max_cache_size:
                    # Remove oldest entries
                    sorted_items = sorted(self.result_cache.items(), 
                                        key=lambda x: x[1][1])  # Sort by timestamp
                    remove_count = len(self.result_cache) - self.max_cache_size
                    
                    for i in range(remove_count):
                        key = sorted_items[i][0]
                        del self.result_cache[key]
                    
                    self.logger.info(f"🧹 Removed {remove_count} old cache entries")
                
                # Clean up old completed/failed tasks
                current_time = datetime.utcnow()
                
                for task_dict in [self.completed_tasks, self.failed_tasks]:
                    old_tasks = []
                    for task_id, task in task_dict.items():
                        if task.completed_at and (current_time - task.completed_at).total_seconds() > 3600:
                            old_tasks.append(task_id)
                    
                    for task_id in old_tasks:
                        del task_dict[task_id]
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Cache cleanup error: {e}")
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID"""
        return f"task_{int(time.time() * 1000)}_{id(object())}"
    
    def _generate_cache_key(self, operation_type: OperationType, operation_data: Dict[str, Any]) -> str:
        """Generate cache key for operation"""
        cache_data = {
            'operation_type': operation_type.value,
            'operation_data': operation_data
        }
        cache_str = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if valid"""
        if cache_key in self.result_cache:
            result, timestamp = self.result_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return result
            else:
                del self.result_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: Any):
        """Cache operation result"""
        if len(self.result_cache) < self.max_cache_size:
            self.result_cache[cache_key] = (result, time.time())
    
    def _create_coordination_result(self, task: Task) -> CoordinationResult:
        """Create coordination result from task"""
        return CoordinationResult(
            task_id=task.id,
            operation_type=task.operation_type,
            status=task.status,
            result=task.result,
            execution_time=task.execution_time or 0.0,
            memory_used=0.0,  # Could be enhanced to track per-task memory
            error_message=task.error,
            metadata=task.metadata
        )
    
    def _should_retry(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry"""
        # Define non-retryable errors
        non_retryable = (ValueError, TypeError, KeyError)
        return not isinstance(error, non_retryable)
    
    def _initialize_retry_strategies(self) -> Dict[str, Any]:
        """Initialize retry strategies for different error types"""
        return {
            'default': {'max_retries': 3, 'backoff_factor': 2},
            'timeout': {'max_retries': 2, 'backoff_factor': 1.5},
            'connection': {'max_retries': 5, 'backoff_factor': 2},
            'resource': {'max_retries': 3, 'backoff_factor': 3}
        }


class CircuitBreaker:
    """Simple circuit breaker for error handling"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
    
    def call(self, func):
        """Call function with circuit breaker protection"""
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'half-open'
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func()
            if self.state == 'half-open':
                self.state = 'closed'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'open'
            
            raise e


# Main async function for testing
async def main():
    """Main function for testing the coordinator"""
    print("🎛️ UNIVERSAL AI CORE COORDINATOR")
    print("=" * 60)
    
    # Initialize coordinator
    config = {
        'max_workers': 4,
        'max_queue_size': 100,
        'cache_ttl': 600,
        'engines': {
            'proof': {'max_workers': 2},
            'symbolic': {'max_reasoning_depth': 50},
            'data': {'batch_size': 1000}
        }
    }
    
    coordinator = UniversalAICoordinator(config)
    
    try:
        # Start coordinator
        await coordinator.start()
        
        # Test symbolic reasoning
        print("\n🧠 Testing symbolic reasoning coordination...")
        symbolic_task_id = await coordinator.submit_task(
            OperationType.SYMBOLIC_REASONING,
            {
                'premises': ['All humans are mortal', 'Socrates is human'],
                'conclusion': 'Socrates is mortal'
            },
            priority=Priority.HIGH
        )
        
        symbolic_result = await coordinator.get_result(symbolic_task_id, timeout=30)
        print(f"✅ Symbolic result: {symbolic_result.status.value}")
        print(f"🎯 Confidence: {symbolic_result.result.get('confidence', 0):.2f}")
        
        # Test data processing
        print("\n📊 Testing data processing coordination...")
        import pandas as pd
        import numpy as np
        
        test_data = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randint(0, 10, 100),
            'target': np.random.randint(0, 2, 100)
        })
        
        data_task_id = await coordinator.submit_task(
            OperationType.FEATURE_ENGINEERING,
            {
                'data': test_data,
                'transformations': ['normalize', 'polynomial'],
                'target_column': 'target'
            }
        )
        
        data_result = await coordinator.get_result(data_task_id, timeout=30)
        print(f"✅ Data result: {data_result.status.value}")
        print(f"🔧 Features generated: {data_result.result.get('features_generated', 0)}")
        
        # Test health monitoring
        print("\n🏥 Testing health monitoring...")
        health = await coordinator.get_system_health()
        print(f"Overall health: {health.overall_status.value}")
        print(f"Active tasks: {health.active_tasks}")
        print(f"Memory usage: {health.memory_usage:.1f}MB")
        
        # Show metrics
        metrics = coordinator.get_metrics()
        print(f"\n📊 Coordinator Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Coordinator test completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await coordinator.stop()


if __name__ == "__main__":
    asyncio.run(main())