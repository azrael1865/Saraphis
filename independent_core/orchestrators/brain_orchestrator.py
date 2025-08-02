"""
Brain Orchestrator - Main Coordinator for Saraphis Brain System
Manages core orchestration, decision coordination, and system-wide operations
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock, Lock
import threading
from collections import defaultdict, deque
import weakref
import traceback
import json

logger = logging.getLogger(__name__)

class OrchestrationMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    PIPELINE = "pipeline"
    ADAPTIVE = "adaptive"

class SystemState(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    DEGRADED = "degraded"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"

class OperationPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

@dataclass
class OrchestrationTask:
    task_id: str
    operation: str
    priority: OperationPriority
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 30.0
    retry_count: int = 3
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None

@dataclass
class SystemMetrics:
    task_count: int = 0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_execution_time: float = 0.0
    system_load: float = 0.0
    memory_usage: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    last_updated: float = field(default_factory=time.time)

@dataclass
class ComponentStatus:
    component_id: str
    status: str
    last_heartbeat: float
    health_score: float = 1.0
    error_count: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class BrainOrchestrator:
    def __init__(self, brain_instance=None, config: Optional[Dict] = None):
        self.brain = brain_instance
        self.config = config or {}
        
        # Core state management
        self._lock = RLock()
        self._task_lock = Lock()
        self._state = SystemState.INITIALIZING
        self._mode = OrchestrationMode.ADAPTIVE
        
        # Task management
        self._task_queue = deque()
        self._active_tasks: Dict[str, OrchestrationTask] = {}
        self._completed_tasks: Dict[str, OrchestrationTask] = {}
        self._task_history = deque(maxlen=1000)
        
        # Component management
        self._components: Dict[str, Any] = {}
        self._component_status: Dict[str, ComponentStatus] = {}
        self._component_locks: Dict[str, Lock] = defaultdict(Lock)
        
        # Event system
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._event_queue = asyncio.Queue() if hasattr(asyncio, 'Queue') else deque()
        
        # Performance tracking
        self._metrics = SystemMetrics()
        self._performance_history = deque(maxlen=100)
        self._execution_times = deque(maxlen=50)
        
        # Orchestrator components
        self._decision_engine = None
        self._reasoning_orchestrator = None
        self._neural_orchestrator = None
        self._uncertainty_orchestrator = None
        self._domain_orchestrator = None
        
        # Worker management
        self._worker_pool = None
        self._max_workers = self.config.get('max_workers', 8)
        self._worker_timeout = self.config.get('worker_timeout', 30.0)
        
        # Emergency handling
        self._emergency_handlers: Dict[str, Callable] = {}
        self._emergency_threshold = self.config.get('emergency_threshold', 0.8)
        
        # Monitoring
        self._monitoring_enabled = self.config.get('monitoring_enabled', True)
        self._monitoring_interval = self.config.get('monitoring_interval', 5.0)
        self._monitoring_thread = None
        
        logger.info("BrainOrchestrator initialized")
    
    def initialize(self) -> bool:
        """Initialize the orchestrator and all components"""
        try:
            with self._lock:
                if self._state != SystemState.INITIALIZING:
                    logger.warning(f"Already initialized (state: {self._state})")
                    return True
                
                # Initialize core components
                self._initialize_worker_pool()
                self._initialize_monitoring()
                self._register_emergency_handlers()
                
                # Set up event system
                self._setup_event_handlers()
                
                # Initialize orchestrator components if available
                self._initialize_orchestrator_components()
                
                self._state = SystemState.READY
                logger.info("BrainOrchestrator initialization complete")
                return True
                
        except Exception as e:
            logger.error(f"Orchestrator initialization failed: {e}")
            self._state = SystemState.EMERGENCY
            return False
    
    def _initialize_worker_pool(self):
        """Initialize the worker thread pool"""
        try:
            import concurrent.futures
            self._worker_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_workers,
                thread_name_prefix="orchestrator_worker"
            )
            logger.debug(f"Worker pool initialized with {self._max_workers} workers")
        except Exception as e:
            logger.warning(f"Failed to initialize worker pool: {e}")
    
    def _initialize_monitoring(self):
        """Initialize system monitoring"""
        if self._monitoring_enabled:
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="orchestrator_monitor",
                daemon=True
            )
            self._monitoring_thread.start()
            logger.debug("Monitoring thread started")
    
    def _initialize_orchestrator_components(self):
        """Initialize other orchestrator components if available"""
        try:
            # Import and initialize other orchestrators if available
            if hasattr(self.brain, 'decision_engine'):
                self._decision_engine = self.brain.decision_engine
            
            if hasattr(self.brain, 'reasoning_orchestrator'):
                self._reasoning_orchestrator = self.brain.reasoning_orchestrator
            
            if hasattr(self.brain, 'neural_orchestrator'):
                self._neural_orchestrator = self.brain.neural_orchestrator
            
            if hasattr(self.brain, 'uncertainty_orchestrator'):
                self._uncertainty_orchestrator = self.brain.uncertainty_orchestrator
            
            if hasattr(self.brain, 'domain_orchestrator'):
                self._domain_orchestrator = self.brain.domain_orchestrator
                
            logger.debug("Orchestrator components initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize orchestrator components: {e}")
    
    def register_component(self, component_id: str, component: Any) -> bool:
        """Register a component with the orchestrator"""
        try:
            with self._lock:
                self._components[component_id] = component
                self._component_status[component_id] = ComponentStatus(
                    component_id=component_id,
                    status="registered",
                    last_heartbeat=time.time()
                )
                
                # Create component-specific lock
                self._component_locks[component_id] = Lock()
                
                logger.debug(f"Component {component_id} registered")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register component {component_id}: {e}")
            return False
    
    def submit_task(self, task: OrchestrationTask) -> bool:
        """Submit a task for orchestration"""
        try:
            with self._task_lock:
                # Validate task
                if not self._validate_task(task):
                    return False
                
                # Add to queue based on priority
                self._insert_task_by_priority(task)
                
                # Update metrics
                self._metrics.task_count += 1
                
                # Trigger task processing
                self._process_next_task()
                
                logger.debug(f"Task {task.task_id} submitted")
                return True
                
        except Exception as e:
            logger.error(f"Failed to submit task {task.task_id}: {e}")
            return False
    
    def _validate_task(self, task: OrchestrationTask) -> bool:
        """Validate a task before submission"""
        if not task.task_id or not task.operation:
            logger.error("Task missing required fields")
            return False
        
        if task.task_id in self._active_tasks or task.task_id in self._completed_tasks:
            logger.error(f"Task {task.task_id} already exists")
            return False
        
        # Check dependencies
        for dep in task.dependencies:
            if dep not in self._completed_tasks:
                dep_status = self._completed_tasks.get(dep, {}).get('status')
                if dep_status != 'completed':
                    logger.error(f"Task {task.task_id} has unmet dependency: {dep}")
                    return False
        
        return True
    
    def _insert_task_by_priority(self, task: OrchestrationTask):
        """Insert task into queue based on priority"""
        if not self._task_queue:
            self._task_queue.append(task)
            return
        
        # Insert based on priority
        inserted = False
        for i, existing_task in enumerate(self._task_queue):
            if task.priority.value > existing_task.priority.value:
                self._task_queue.insert(i, task)
                inserted = True
                break
        
        if not inserted:
            self._task_queue.append(task)
    
    def _process_next_task(self):
        """Process the next task in the queue"""
        if not self._task_queue or len(self._active_tasks) >= self._max_workers:
            return
        
        task = self._task_queue.popleft()
        
        # Move to active tasks
        self._active_tasks[task.task_id] = task
        task.status = "active"
        task.started_at = time.time()
        self._metrics.active_tasks += 1
        
        # Submit to worker pool
        if self._worker_pool:
            future = self._worker_pool.submit(self._execute_task, task)
            # Store future reference for tracking
            task.future = future
        else:
            # Execute directly if no worker pool
            self._execute_task(task)
    
    def _execute_task(self, task: OrchestrationTask):
        """Execute a task"""
        try:
            start_time = time.time()
            
            # Route task to appropriate handler
            result = self._route_task(task)
            
            # Task completed successfully
            execution_time = time.time() - start_time
            self._complete_task(task, result, execution_time)
            
        except Exception as e:
            error_msg = f"Task execution failed: {e}"
            logger.error(error_msg)
            self._fail_task(task, error_msg)
    
    def _route_task(self, task: OrchestrationTask) -> Any:
        """Route task to appropriate component or handler"""
        operation = task.operation
        parameters = task.parameters
        
        # Route to specific orchestrators
        if operation.startswith('decision'):
            if self._decision_engine:
                return self._decision_engine.process(parameters)
        
        elif operation.startswith('reasoning'):
            if self._reasoning_orchestrator:
                return self._reasoning_orchestrator.orchestrate(parameters)
        
        elif operation.startswith('neural'):
            if self._neural_orchestrator:
                return self._neural_orchestrator.coordinate(parameters)
        
        elif operation.startswith('uncertainty'):
            if self._uncertainty_orchestrator:
                return self._uncertainty_orchestrator.quantify(parameters)
        
        elif operation.startswith('domain'):
            if self._domain_orchestrator:
                return self._domain_orchestrator.handle_domain_operation(parameters)
        
        # Route to brain if available
        elif hasattr(self.brain, operation):
            method = getattr(self.brain, operation)
            if callable(method):
                return method(**parameters)
        
        # Route to registered components
        component_id = parameters.get('component_id')
        if component_id and component_id in self._components:
            component = self._components[component_id]
            if hasattr(component, operation):
                method = getattr(component, operation)
                if callable(method):
                    return method(**{k: v for k, v in parameters.items() if k != 'component_id'})
        
        # Default handling
        return self._handle_generic_operation(task)
    
    def _handle_generic_operation(self, task: OrchestrationTask) -> Any:
        """Handle generic operations"""
        operation = task.operation
        parameters = task.parameters
        
        if operation == 'status_check':
            return self.get_system_status()
        elif operation == 'metrics_update':
            return self.update_metrics()
        elif operation == 'component_health_check':
            return self.check_component_health()
        elif operation == 'emergency_response':
            return self._handle_emergency(parameters)
        else:
            logger.warning(f"Unknown operation: {operation}")
            return {"status": "unknown_operation", "operation": operation}
    
    def _complete_task(self, task: OrchestrationTask, result: Any, execution_time: float):
        """Mark task as completed"""
        with self._task_lock:
            # Update task
            task.status = "completed"
            task.completed_at = time.time()
            task.result = result
            
            # Move from active to completed
            if task.task_id in self._active_tasks:
                del self._active_tasks[task.task_id]
            self._completed_tasks[task.task_id] = task
            
            # Update metrics
            self._metrics.active_tasks = max(0, self._metrics.active_tasks - 1)
            self._metrics.completed_tasks += 1
            self._execution_times.append(execution_time)
            
            # Update average execution time
            if self._execution_times:
                self._metrics.average_execution_time = sum(self._execution_times) / len(self._execution_times)
            
            # Add to history
            self._task_history.append({
                'task_id': task.task_id,
                'operation': task.operation,
                'status': 'completed',
                'execution_time': execution_time,
                'completed_at': task.completed_at
            })
            
            logger.debug(f"Task {task.task_id} completed in {execution_time:.3f}s")
            
            # Process next task
            self._process_next_task()
    
    def _fail_task(self, task: OrchestrationTask, error_msg: str):
        """Mark task as failed"""
        with self._task_lock:
            # Update task
            task.status = "failed"
            task.completed_at = time.time()
            task.error = error_msg
            
            # Move from active to completed
            if task.task_id in self._active_tasks:
                del self._active_tasks[task.task_id]
            self._completed_tasks[task.task_id] = task
            
            # Update metrics
            self._metrics.active_tasks = max(0, self._metrics.active_tasks - 1)
            self._metrics.failed_tasks += 1
            
            # Update error rate
            total_tasks = self._metrics.completed_tasks + self._metrics.failed_tasks
            self._metrics.error_rate = self._metrics.failed_tasks / max(1, total_tasks)
            
            # Add to history
            self._task_history.append({
                'task_id': task.task_id,
                'operation': task.operation,
                'status': 'failed',
                'error': error_msg,
                'failed_at': task.completed_at
            })
            
            logger.error(f"Task {task.task_id} failed: {error_msg}")
            
            # Check if retry is needed
            if task.retry_count > 0:
                self._retry_task(task)
            else:
                # Process next task
                self._process_next_task()
    
    def _retry_task(self, task: OrchestrationTask):
        """Retry a failed task"""
        task.retry_count -= 1
        task.status = "retrying"
        task.started_at = None
        task.completed_at = None
        task.error = None
        
        # Re-submit task
        self._insert_task_by_priority(task)
        logger.info(f"Retrying task {task.task_id} ({task.retry_count} attempts remaining)")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with self._lock:
            return {
                'state': self._state.value,
                'mode': self._mode.value,
                'metrics': {
                    'task_count': self._metrics.task_count,
                    'active_tasks': self._metrics.active_tasks,
                    'completed_tasks': self._metrics.completed_tasks,
                    'failed_tasks': self._metrics.failed_tasks,
                    'error_rate': self._metrics.error_rate,
                    'average_execution_time': self._metrics.average_execution_time,
                    'throughput': self._metrics.throughput
                },
                'components': {
                    comp_id: {
                        'status': status.status,
                        'health_score': status.health_score,
                        'last_heartbeat': status.last_heartbeat,
                        'error_count': status.error_count
                    }
                    for comp_id, status in self._component_status.items()
                },
                'queue_size': len(self._task_queue),
                'worker_pool_active': self._worker_pool is not None,
                'monitoring_active': self._monitoring_enabled
            }
    
    def update_metrics(self) -> Dict[str, float]:
        """Update system metrics"""
        current_time = time.time()
        
        # Calculate throughput
        recent_completions = sum(1 for entry in self._task_history 
                               if current_time - entry.get('completed_at', 0) < 60)
        self._metrics.throughput = recent_completions / 60.0
        
        # Update system load
        self._metrics.system_load = len(self._active_tasks) / max(1, self._max_workers)
        
        # Update last updated time
        self._metrics.last_updated = current_time
        
        return {
            'throughput': self._metrics.throughput,
            'system_load': self._metrics.system_load,
            'error_rate': self._metrics.error_rate,
            'average_execution_time': self._metrics.average_execution_time
        }
    
    def check_component_health(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all registered components"""
        current_time = time.time()
        health_report = {}
        
        for comp_id, status in self._component_status.items():
            # Check heartbeat freshness
            heartbeat_age = current_time - status.last_heartbeat
            
            # Calculate health score
            health_score = 1.0
            if heartbeat_age > 60:  # No heartbeat for 1 minute
                health_score *= 0.5
            if heartbeat_age > 300:  # No heartbeat for 5 minutes
                health_score *= 0.1
            
            # Factor in error count
            if status.error_count > 0:
                health_score *= max(0.1, 1.0 - (status.error_count * 0.1))
            
            status.health_score = health_score
            
            health_report[comp_id] = {
                'status': status.status,
                'health_score': health_score,
                'heartbeat_age': heartbeat_age,
                'error_count': status.error_count,
                'performance_metrics': status.performance_metrics
            }
        
        return health_report
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._monitoring_enabled and self._state != SystemState.SHUTDOWN:
            try:
                # Update metrics
                self.update_metrics()
                
                # Check component health
                self.check_component_health()
                
                # Check for emergency conditions
                self._check_emergency_conditions()
                
                # Sleep for monitoring interval
                time.sleep(self._monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(1)  # Brief pause before retry
    
    def _check_emergency_conditions(self):
        """Check for emergency conditions"""
        # Check error rate
        if self._metrics.error_rate > self._emergency_threshold:
            self._trigger_emergency("High error rate", {
                'error_rate': self._metrics.error_rate,
                'threshold': self._emergency_threshold
            })
        
        # Check system load
        if self._metrics.system_load > 0.95:
            self._trigger_emergency("System overload", {
                'system_load': self._metrics.system_load
            })
        
        # Check component health
        unhealthy_components = [
            comp_id for comp_id, status in self._component_status.items()
            if status.health_score < 0.3
        ]
        
        if len(unhealthy_components) > len(self._components) * 0.5:
            self._trigger_emergency("Multiple component failures", {
                'unhealthy_components': unhealthy_components
            })
    
    def _trigger_emergency(self, reason: str, details: Dict[str, Any]):
        """Trigger emergency response"""
        if self._state == SystemState.EMERGENCY:
            return  # Already in emergency state
        
        logger.critical(f"Emergency triggered: {reason}")
        self._state = SystemState.EMERGENCY
        
        # Execute emergency handlers
        for handler_name, handler in self._emergency_handlers.items():
            try:
                handler(reason, details)
            except Exception as e:
                logger.error(f"Emergency handler {handler_name} failed: {e}")
        
        # Emit emergency event
        self._emit_event('emergency', {
            'reason': reason,
            'details': details,
            'timestamp': time.time()
        })
    
    def _register_emergency_handlers(self):
        """Register default emergency handlers"""
        self._emergency_handlers['task_cleanup'] = self._emergency_task_cleanup
        self._emergency_handlers['component_isolation'] = self._emergency_component_isolation
        self._emergency_handlers['resource_conservation'] = self._emergency_resource_conservation
    
    def _emergency_task_cleanup(self, reason: str, details: Dict[str, Any]):
        """Emergency task cleanup"""
        logger.warning("Performing emergency task cleanup")
        
        # Cancel all pending tasks
        with self._task_lock:
            cancelled_count = len(self._task_queue)
            self._task_queue.clear()
            
            # Mark active tasks for termination
            for task in self._active_tasks.values():
                task.status = "cancelled_emergency"
            
            logger.warning(f"Cancelled {cancelled_count} pending tasks")
    
    def _emergency_component_isolation(self, reason: str, details: Dict[str, Any]):
        """Emergency component isolation"""
        logger.warning("Isolating unhealthy components")
        
        for comp_id, status in self._component_status.items():
            if status.health_score < 0.3:
                status.status = "isolated"
                logger.warning(f"Component {comp_id} isolated due to poor health")
    
    def _emergency_resource_conservation(self, reason: str, details: Dict[str, Any]):
        """Emergency resource conservation"""
        logger.warning("Activating resource conservation mode")
        
        # Reduce worker pool size
        if self._worker_pool:
            self._max_workers = max(1, self._max_workers // 2)
        
        # Increase monitoring interval
        self._monitoring_interval = min(30.0, self._monitoring_interval * 2)
    
    def _setup_event_handlers(self):
        """Setup event handling system"""
        # Register default event handlers
        self.register_event_handler('task_completed', self._on_task_completed)
        self.register_event_handler('task_failed', self._on_task_failed)
        self.register_event_handler('component_registered', self._on_component_registered)
        self.register_event_handler('emergency', self._on_emergency)
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler"""
        self._event_handlers[event_type].append(handler)
        logger.debug(f"Event handler registered for {event_type}")
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to all registered handlers"""
        for handler in self._event_handlers.get(event_type, []):
            try:
                handler(event_type, data)
            except Exception as e:
                logger.error(f"Event handler error for {event_type}: {e}")
    
    def _on_task_completed(self, event_type: str, data: Dict[str, Any]):
        """Handle task completion event"""
        logger.debug(f"Task completed: {data}")
    
    def _on_task_failed(self, event_type: str, data: Dict[str, Any]):
        """Handle task failure event"""
        logger.warning(f"Task failed: {data}")
    
    def _on_component_registered(self, event_type: str, data: Dict[str, Any]):
        """Handle component registration event"""
        logger.info(f"Component registered: {data}")
    
    def _on_emergency(self, event_type: str, data: Dict[str, Any]):
        """Handle emergency event"""
        logger.critical(f"Emergency event: {data}")
    
    def _handle_emergency(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emergency response operations"""
        emergency_type = parameters.get('type', 'general')
        emergency_data = parameters.get('data', {})
        
        logger.critical(f"Emergency response triggered: {emergency_type}")
        
        response = {
            'type': emergency_type,
            'timestamp': time.time(),
            'actions_taken': []
        }
        
        # Execute emergency procedures based on type
        if emergency_type == 'system_overload':
            self._emergency_task_cleanup(emergency_type, emergency_data)
            response['actions_taken'].append('task_cleanup')
        
        elif emergency_type == 'component_failure':
            self._emergency_component_isolation(emergency_type, emergency_data)
            response['actions_taken'].append('component_isolation')
        
        elif emergency_type == 'resource_exhaustion':
            self._emergency_resource_conservation(emergency_type, emergency_data)
            response['actions_taken'].append('resource_conservation')
        
        return response
    
    def shutdown(self, timeout: float = 10.0) -> bool:
        """Shutdown the orchestrator gracefully"""
        logger.info("Shutting down BrainOrchestrator")
        
        try:
            with self._lock:
                self._state = SystemState.SHUTDOWN
                self._monitoring_enabled = False
            
            # Wait for monitoring thread to finish
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=5.0)
            
            # Shutdown worker pool
            if self._worker_pool:
                self._worker_pool.shutdown(wait=True, timeout=timeout)
            
            # Clear active tasks
            with self._task_lock:
                for task in self._active_tasks.values():
                    task.status = "cancelled_shutdown"
                self._active_tasks.clear()
                self._task_queue.clear()
            
            logger.info("BrainOrchestrator shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
            return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        # Check active tasks
        if task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            return {
                'task_id': task.task_id,
                'status': task.status,
                'progress': 'active',
                'created_at': task.created_at,
                'started_at': task.started_at
            }
        
        # Check completed tasks
        if task_id in self._completed_tasks:
            task = self._completed_tasks[task_id]
            return {
                'task_id': task.task_id,
                'status': task.status,
                'progress': 'completed',
                'created_at': task.created_at,
                'started_at': task.started_at,
                'completed_at': task.completed_at,
                'result': task.result,
                'error': task.error
            }
        
        # Check pending tasks
        for task in self._task_queue:
            if task.task_id == task_id:
                return {
                    'task_id': task.task_id,
                    'status': task.status,
                    'progress': 'pending',
                    'created_at': task.created_at
                }
        
        return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            'system_metrics': {
                'total_tasks': self._metrics.task_count,
                'completed_tasks': self._metrics.completed_tasks,
                'failed_tasks': self._metrics.failed_tasks,
                'active_tasks': self._metrics.active_tasks,
                'error_rate': self._metrics.error_rate,
                'throughput': self._metrics.throughput,
                'average_execution_time': self._metrics.average_execution_time,
                'system_load': self._metrics.system_load
            },
            'component_health': self.check_component_health(),
            'recent_tasks': list(self._task_history)[-10:],  # Last 10 tasks
            'queue_status': {
                'pending_tasks': len(self._task_queue),
                'queue_priorities': [task.priority.name for task in list(self._task_queue)[:5]]
            },
            'system_state': self._state.value,
            'orchestration_mode': self._mode.value
        }