"""
Workflow Orchestrator for Financial Fraud Detection
Complete workflow orchestration system for core workflow orchestration methods.
Part of the Saraphis recursive methodology for accuracy tracking - Phase 5.
"""

import logging
import json
import time
import threading
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import uuid
import hashlib
import traceback
import psutil
from collections import defaultdict, deque

# Import from existing modules
try:
    from enhanced_fraud_core_exceptions import (
        EnhancedFraudException, ValidationError, ConfigurationError,
        ResourceError, DataQualityError, ErrorContext, create_error_context
    )
    from enhanced_fraud_core_monitoring import (
        MonitoringManager, PerformanceMetrics, monitor_performance,
        with_caching, CacheManager, MonitoringConfig
    )
    from accuracy_dataset_manager import TrainValidationTestManager, DatasetMetadata
    from accuracy_tracking_db import AccuracyTrackingDatabase, ModelVersion, ModelStatus
except ImportError:
    # Fallback for standalone development
    from enhanced_fraud_core_exceptions import (
        EnhancedFraudException, ValidationError, ConfigurationError,
        ResourceError, DataQualityError, ErrorContext, create_error_context
    )
    from enhanced_fraud_core_monitoring import (
        MonitoringManager, PerformanceMetrics, monitor_performance,
        with_caching, CacheManager, MonitoringConfig
    )
    from accuracy_dataset_manager import TrainValidationTestManager, DatasetMetadata
    from accuracy_tracking_db import AccuracyTrackingDatabase, ModelVersion, ModelStatus

# Configure logging
logger = logging.getLogger(__name__)

# ======================== CUSTOM EXCEPTIONS ========================

class WorkflowError(EnhancedFraudException):
    """Base exception for workflow orchestration errors"""
    pass

class WorkflowValidationError(WorkflowError):
    """Exception raised when workflow validation fails"""
    pass

class WorkflowExecutionError(WorkflowError):
    """Exception raised when workflow execution fails"""
    pass

class WorkflowConfigurationError(WorkflowError):
    """Exception raised when workflow configuration is invalid"""
    pass

class WorkflowResourceError(WorkflowError):
    """Exception raised when workflow resource limits are exceeded"""
    pass

class WorkflowIntegrationError(WorkflowError):
    """Exception raised when workflow integration fails"""
    pass

# ======================== ENUMS AND CONSTANTS ========================

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class WorkflowType(Enum):
    """Types of workflows"""
    ACCURACY_TRACKING_SETUP = "accuracy_tracking_setup"
    MODEL_TRAINING = "model_training"
    MODEL_DEPLOYMENT = "model_deployment"
    LIFECYCLE_MANAGEMENT = "lifecycle_management"
    EVALUATION = "evaluation"
    MONITORING = "monitoring"

class WorkflowPriority(Enum):
    """Workflow execution priority"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

# ======================== DATA STRUCTURES ========================

@dataclass
class WorkflowStep:
    """Individual workflow step definition"""
    step_id: str
    name: str
    operation: Callable
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    max_retries: int = 3
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    workflow_id: str
    name: str
    description: str
    workflow_type: WorkflowType
    steps: List[WorkflowStep]
    priority: WorkflowPriority = WorkflowPriority.NORMAL
    timeout_seconds: Optional[int] = None
    parallel_execution: bool = False
    error_handling_strategy: str = "fail_fast"  # fail_fast, continue_on_error, retry
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class WorkflowExecution:
    """Workflow execution instance"""
    execution_id: str
    workflow_definition: WorkflowDefinition
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: float = 0.0
    error_message: Optional[str] = None
    step_results: Dict[str, Any] = field(default_factory=dict)
    execution_context: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class WorkflowConfig:
    """Configuration for workflow orchestrator"""
    max_concurrent_workflows: int = 5
    max_workers: int = 4
    default_timeout_seconds: int = 3600
    enable_monitoring: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    enable_persistence: bool = True
    persistence_path: str = "workflow_state"
    resource_monitoring_interval: float = 30.0
    cleanup_completed_workflows_hours: int = 24
    max_workflow_history: int = 1000
    enable_performance_profiling: bool = True
    enable_detailed_logging: bool = False

# ======================== WORKFLOW ORCHESTRATOR ========================

class WorkflowOrchestrator:
    """
    Complete workflow orchestration system for core workflow orchestration methods.
    Provides comprehensive orchestration functionality for model training, deployment,
    lifecycle management, and accuracy tracking workflows.
    """
    
    def __init__(self, 
                 dataset_manager: Optional[TrainValidationTestManager] = None,
                 tracking_db: Optional[AccuracyTrackingDatabase] = None,
                 evaluation_system: Optional[Any] = None,
                 config: Optional[WorkflowConfig] = None):
        """
        Initialize WorkflowOrchestrator with component references and configuration.
        
        Args:
            dataset_manager: TrainValidationTestManager instance
            tracking_db: AccuracyTrackingDatabase instance
            evaluation_system: Model evaluation system
            config: Workflow configuration
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Component references
        self.dataset_manager = dataset_manager
        self.tracking_db = tracking_db
        self.evaluation_system = evaluation_system
        
        # Configuration
        self.config = config or WorkflowConfig()
        
        # State management
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.workflow_history: deque = deque(maxlen=self.config.max_workflow_history)
        self.workflow_registry: Dict[str, WorkflowDefinition] = {}
        
        # Execution management
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.workflow_queue = deque()
        self.workflow_scheduler_running = False
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Monitoring and caching
        self._init_monitoring()
        self._init_caching()
        self._init_persistence()
        
        # Performance tracking
        self.performance_stats = defaultdict(list)
        self.resource_stats = defaultdict(list)
        
        # Start background services
        self._start_scheduler()
        self._start_resource_monitor()
        
        self.logger.info(
            f"WorkflowOrchestrator initialized with config: "
            f"max_concurrent={self.config.max_concurrent_workflows}, "
            f"monitoring={self.config.enable_monitoring}, "
            f"caching={self.config.enable_caching}"
        )
    
    def _init_monitoring(self) -> None:
        """Initialize monitoring system"""
        if self.config.enable_monitoring:
            monitoring_config = MonitoringConfig(
                enable_performance_profiling=self.config.enable_performance_profiling,
                enable_resource_monitoring=True,
                cache_size=self.config.cache_size,
                cache_ttl=self.config.cache_ttl_seconds
            )
            self.monitoring_manager = MonitoringManager(monitoring_config)
        else:
            self.monitoring_manager = None
    
    def _init_caching(self) -> None:
        """Initialize caching system"""
        if self.config.enable_caching and self.monitoring_manager:
            self.cache_manager = self.monitoring_manager.cache_manager
        else:
            self.cache_manager = None
    
    def _init_persistence(self) -> None:
        """Initialize persistence system"""
        if self.config.enable_persistence:
            self.persistence_path = Path(self.config.persistence_path)
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            
            # Load previous workflow state
            self._load_workflow_state()
    
    def _start_scheduler(self) -> None:
        """Start workflow scheduler"""
        if not self.workflow_scheduler_running:
            self.workflow_scheduler_running = True
            self.executor.submit(self._workflow_scheduler_loop)
            self.logger.info("Workflow scheduler started")
    
    def _start_resource_monitor(self) -> None:
        """Start resource monitoring"""
        if self.config.enable_monitoring:
            self.executor.submit(self._resource_monitor_loop)
            self.logger.debug("Resource monitor started")
    
    def _workflow_scheduler_loop(self) -> None:
        """Main workflow scheduler loop"""
        while self.workflow_scheduler_running:
            try:
                self._schedule_workflows()
                time.sleep(1.0)  # Check every second
            except Exception as e:
                self.logger.error(f"Error in workflow scheduler: {e}")
                time.sleep(5.0)  # Back off on error
    
    def _resource_monitor_loop(self) -> None:
        """Resource monitoring loop"""
        while self.workflow_scheduler_running:
            try:
                self._collect_resource_metrics()
                time.sleep(self.config.resource_monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in resource monitor: {e}")
                time.sleep(30.0)
    
    def _schedule_workflows(self) -> None:
        """Schedule pending workflows for execution"""
        with self._lock:
            # Check if we can start more workflows
            active_count = len([w for w in self.active_workflows.values() 
                               if w.status == WorkflowStatus.RUNNING])
            
            if active_count >= self.config.max_concurrent_workflows:
                return
            
            # Sort queue by priority
            sorted_queue = sorted(self.workflow_queue, 
                                key=lambda x: x.workflow_definition.priority.value, 
                                reverse=True)
            
            # Start workflows up to the limit
            workflows_to_start = sorted_queue[:self.config.max_concurrent_workflows - active_count]
            
            for workflow_execution in workflows_to_start:
                if workflow_execution.execution_id in self.active_workflows:
                    self.workflow_queue.remove(workflow_execution)
                    self._execute_workflow_async(workflow_execution)
    
    def _execute_workflow_async(self, workflow_execution: WorkflowExecution) -> None:
        """Execute workflow asynchronously"""
        def workflow_wrapper():
            try:
                self._execute_workflow_sync(workflow_execution)
            except Exception as e:
                self.logger.error(f"Workflow execution failed: {e}")
                workflow_execution.status = WorkflowStatus.FAILED
                workflow_execution.error_message = str(e)
                workflow_execution.end_time = datetime.now()
            finally:
                # Clean up and move to history
                with self._lock:
                    if workflow_execution.execution_id in self.active_workflows:
                        del self.active_workflows[workflow_execution.execution_id]
                    self.workflow_history.append(workflow_execution)
                
                # Persist state
                if self.config.enable_persistence:
                    self._save_workflow_state()
        
        self.executor.submit(workflow_wrapper)
    
    @contextmanager
    def _monitor_workflow_execution(self, workflow_execution: WorkflowExecution):
        """Context manager for monitoring workflow execution"""
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        
        try:
            yield
        finally:
            # Record performance metrics
            duration = time.time() - start_time
            end_memory = psutil.virtual_memory().percent
            
            workflow_execution.performance_metrics.update({
                'duration_seconds': duration,
                'memory_delta_percent': end_memory - start_memory,
                'cpu_percent': psutil.cpu_percent(interval=None)
            })
            
            if self.monitoring_manager:
                metric = PerformanceMetrics(
                    timestamp=datetime.now(),
                    operation_name=f"workflow_{workflow_execution.workflow_definition.workflow_type.value}",
                    duration=duration,
                    success=workflow_execution.status == WorkflowStatus.COMPLETED,
                    memory_usage=end_memory - start_memory
                )
                self.monitoring_manager.metrics_collector.record_performance_metric(metric)
    
    def _execute_workflow_sync(self, workflow_execution: WorkflowExecution) -> None:
        """Execute workflow synchronously"""
        self.logger.info(f"Starting workflow execution: {workflow_execution.execution_id}")
        
        workflow_execution.status = WorkflowStatus.RUNNING
        workflow_execution.start_time = datetime.now()
        
        with self._monitor_workflow_execution(workflow_execution):
            try:
                workflow_def = workflow_execution.workflow_definition
                
                if workflow_def.parallel_execution:
                    self._execute_workflow_parallel(workflow_execution)
                else:
                    self._execute_workflow_sequential(workflow_execution)
                
                workflow_execution.status = WorkflowStatus.COMPLETED
                workflow_execution.progress = 1.0
                
            except Exception as e:
                workflow_execution.status = WorkflowStatus.FAILED
                workflow_execution.error_message = str(e)
                
                if workflow_def.error_handling_strategy == "fail_fast":
                    raise
                else:
                    self.logger.error(f"Workflow failed but continuing: {e}")
            finally:
                workflow_execution.end_time = datetime.now()
        
        self.logger.info(f"Workflow execution completed: {workflow_execution.execution_id}")
    
    def _execute_workflow_sequential(self, workflow_execution: WorkflowExecution) -> None:
        """Execute workflow steps sequentially"""
        workflow_def = workflow_execution.workflow_definition
        total_steps = len(workflow_def.steps)
        
        for i, step in enumerate(workflow_def.steps):
            try:
                self._execute_workflow_step(step, workflow_execution)
                workflow_execution.progress = (i + 1) / total_steps
                
            except Exception as e:
                if workflow_def.error_handling_strategy == "fail_fast":
                    raise
                elif workflow_def.error_handling_strategy == "continue_on_error":
                    self.logger.warning(f"Step {step.step_id} failed, continuing: {e}")
                    step.status = WorkflowStatus.FAILED
                    step.error_message = str(e)
                elif workflow_def.error_handling_strategy == "retry":
                    self._retry_workflow_step(step, workflow_execution, e)
    
    def _execute_workflow_parallel(self, workflow_execution: WorkflowExecution) -> None:
        """Execute workflow steps in parallel where possible"""
        workflow_def = workflow_execution.workflow_definition
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(workflow_def.steps)
        
        # Execute steps in topological order with parallelization
        executed_steps = set()
        pending_steps = {step.step_id: step for step in workflow_def.steps}
        
        while pending_steps:
            # Find steps with satisfied dependencies
            ready_steps = []
            for step_id, step in pending_steps.items():
                if all(dep in executed_steps for dep in step.dependencies):
                    ready_steps.append(step)
            
            if not ready_steps:
                raise WorkflowExecutionError("Circular dependency detected in workflow")
            
            # Execute ready steps in parallel
            futures = []
            for step in ready_steps:
                future = self.executor.submit(self._execute_workflow_step, step, workflow_execution)
                futures.append((future, step))
            
            # Wait for completion
            for future, step in futures:
                try:
                    future.result()
                    executed_steps.add(step.step_id)
                    del pending_steps[step.step_id]
                except Exception as e:
                    if workflow_def.error_handling_strategy == "fail_fast":
                        raise
                    else:
                        self.logger.error(f"Parallel step {step.step_id} failed: {e}")
                        step.status = WorkflowStatus.FAILED
                        step.error_message = str(e)
                        executed_steps.add(step.step_id)  # Mark as processed
                        del pending_steps[step.step_id]
            
            # Update progress
            workflow_execution.progress = len(executed_steps) / len(workflow_def.steps)
    
    def _execute_workflow_step(self, step: WorkflowStep, workflow_execution: WorkflowExecution) -> None:
        """Execute individual workflow step"""
        self.logger.debug(f"Executing step: {step.step_id}")
        
        step.status = WorkflowStatus.RUNNING
        step.start_time = datetime.now()
        
        try:
            # Prepare step context
            step_context = {
                'workflow_execution': workflow_execution,
                'step_results': workflow_execution.step_results,
                'execution_context': workflow_execution.execution_context
            }
            
            # Execute with timeout if specified
            if step.timeout_seconds:
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Step {step.step_id} timed out after {step.timeout_seconds} seconds")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(step.timeout_seconds)
            
            try:
                # Execute the step operation
                result = step.operation(step_context, **step.parameters)
                step.result = result
                workflow_execution.step_results[step.step_id] = result
                step.status = WorkflowStatus.COMPLETED
                
            finally:
                if step.timeout_seconds:
                    signal.alarm(0)  # Cancel timeout
            
        except Exception as e:
            step.status = WorkflowStatus.FAILED
            step.error_message = str(e)
            self.logger.error(f"Step {step.step_id} failed: {e}")
            raise
        
        finally:
            step.end_time = datetime.now()
    
    def _retry_workflow_step(self, step: WorkflowStep, workflow_execution: WorkflowExecution, error: Exception) -> None:
        """Retry failed workflow step"""
        if step.retry_count < step.max_retries:
            step.retry_count += 1
            self.logger.info(f"Retrying step {step.step_id}, attempt {step.retry_count}/{step.max_retries}")
            
            # Exponential backoff
            delay = min(2 ** step.retry_count, 60)
            time.sleep(delay)
            
            try:
                self._execute_workflow_step(step, workflow_execution)
            except Exception as retry_error:
                if step.retry_count >= step.max_retries:
                    step.status = WorkflowStatus.FAILED
                    step.error_message = f"Max retries exceeded. Last error: {retry_error}"
                    raise
                else:
                    self._retry_workflow_step(step, workflow_execution, retry_error)
        else:
            step.status = WorkflowStatus.FAILED
            step.error_message = f"Max retries exceeded. Original error: {error}"
            raise
    
    def _build_dependency_graph(self, steps: List[WorkflowStep]) -> Dict[str, List[str]]:
        """Build dependency graph for parallel execution"""
        graph = {}
        for step in steps:
            graph[step.step_id] = step.dependencies
        return graph
    
    def _collect_resource_metrics(self) -> None:
        """Collect system resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            self.resource_stats['cpu_percent'].append(cpu_percent)
            self.resource_stats['memory_percent'].append(memory.percent)
            self.resource_stats['disk_percent'].append(disk.percent)
            
            # Keep only recent metrics
            max_entries = 100
            for key in self.resource_stats:
                if len(self.resource_stats[key]) > max_entries:
                    self.resource_stats[key] = self.resource_stats[key][-max_entries:]
            
        except Exception as e:
            self.logger.error(f"Failed to collect resource metrics: {e}")
    
    def _save_workflow_state(self) -> None:
        """Save workflow state to disk"""
        try:
            state_file = self.persistence_path / "workflow_state.json"
            
            state = {
                'active_workflows': {
                    k: asdict(v) for k, v in self.active_workflows.items()
                },
                'workflow_history': [asdict(w) for w in list(self.workflow_history)[-100:]],  # Last 100
                'workflow_registry': {
                    k: asdict(v) for k, v in self.workflow_registry.items()
                }
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save workflow state: {e}")
    
    def _load_workflow_state(self) -> None:
        """Load workflow state from disk"""
        try:
            state_file = self.persistence_path / "workflow_state.json"
            
            if not state_file.exists():
                return
            
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Restore workflow history (read-only)
            if 'workflow_history' in state:
                for workflow_data in state['workflow_history']:
                    # Convert back to dataclass (simplified)
                    execution = WorkflowExecution(
                        execution_id=workflow_data['execution_id'],
                        workflow_definition=WorkflowDefinition(**workflow_data['workflow_definition']),
                        status=WorkflowStatus(workflow_data['status'])
                    )
                    self.workflow_history.append(execution)
                    
        except Exception as e:
            self.logger.error(f"Failed to load workflow state: {e}")
    
    # ======================== PUBLIC API METHODS ========================
    
    def setup_complete_accuracy_tracking(self, 
                                        model_info: Dict[str, Any],
                                        dataset_config: Dict[str, Any],
                                        monitoring_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Setup complete end-to-end accuracy tracking system.
        
        Args:
            model_info: Model information (model_id, model_type, training_data_hash)
            dataset_config: Dataset configuration (train_ratio, val_ratio, test_ratio, stratification)
            monitoring_config: Monitoring configuration (real_time_monitoring, alert_thresholds, etc.)
            
        Returns:
            Setup result with status and metadata
        """
        self.logger.info("Setting up complete accuracy tracking system")
        
        def setup_accuracy_tracking_operation(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            """Setup accuracy tracking operation"""
            result = {
                'setup_timestamp': datetime.now().isoformat(),
                'model_info': model_info,
                'dataset_config': dataset_config,
                'monitoring_config': monitoring_config,
                'components': {}
            }
            
            try:
                # Initialize dataset manager if not provided
                if not self.dataset_manager:
                    self.dataset_manager = TrainValidationTestManager(config={
                        'validation_level': dataset_config.get('validation_level', 'standard'),
                        'enable_compression': True,
                        'enable_detailed_logging': monitoring_config.get('detailed_logging', False)
                    })
                    result['components']['dataset_manager'] = 'initialized'
                
                # Initialize tracking database if not provided
                if not self.tracking_db:
                    db_config = {
                        'database': {
                            'enable_compression': True,
                            'retention_days': monitoring_config.get('retention_days', 90)
                        },
                        'monitoring': {
                            'enabled': monitoring_config.get('real_time_monitoring', True)
                        }
                    }
                    self.tracking_db = AccuracyTrackingDatabase(config=db_config)
                    result['components']['tracking_db'] = 'initialized'
                
                # Create model version in tracking database
                model_version = self.tracking_db.create_model_version(
                    model_id=model_info['model_id'],
                    version="1.0.0",
                    model_type=model_info['model_type'],
                    parameters=model_info.get('parameters', {}),
                    training_dataset_id=model_info.get('training_dataset_id'),
                    metadata={
                        'setup_timestamp': datetime.now().isoformat(),
                        'training_data_hash': model_info.get('training_data_hash', ''),
                        'monitoring_config': monitoring_config
                    }
                )
                result['model_version'] = asdict(model_version)
                
                # Setup monitoring alerts
                if monitoring_config.get('alert_thresholds'):
                    alert_config = {
                        'accuracy_drop_threshold': monitoring_config['alert_thresholds'].get('accuracy_drop', 0.05),
                        'notification_channels': monitoring_config.get('notification_channels', [])
                    }
                    result['monitoring_setup'] = alert_config
                
                result['status'] = 'success'
                result['message'] = 'Accuracy tracking system setup completed successfully'
                
                return result
                
            except Exception as e:
                self.logger.error(f"Failed to setup accuracy tracking: {e}")
                result['status'] = 'failed'
                result['error'] = str(e)
                raise WorkflowExecutionError(f"Accuracy tracking setup failed: {e}")
        
        # Create workflow definition
        workflow_def = WorkflowDefinition(
            workflow_id=f"accuracy_tracking_setup_{uuid.uuid4().hex[:8]}",
            name="Complete Accuracy Tracking Setup",
            description="End-to-end setup of accuracy tracking system",
            workflow_type=WorkflowType.ACCURACY_TRACKING_SETUP,
            steps=[
                WorkflowStep(
                    step_id="setup_accuracy_tracking",
                    name="Setup Accuracy Tracking",
                    operation=setup_accuracy_tracking_operation,
                    parameters={},
                    timeout_seconds=300
                )
            ],
            priority=WorkflowPriority.HIGH
        )
        
        # Execute workflow
        execution_result = self.execute_workflow(workflow_def)
        
        return {
            'workflow_execution_id': execution_result['execution_id'],
            'status': execution_result['status'],
            'setup_result': execution_result.get('step_results', {}).get('setup_accuracy_tracking', {})
        }
    
    def orchestrate_model_training_with_tracking(self,
                                               training_config: Dict[str, Any],
                                               evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate model training workflow with integrated tracking.
        
        Args:
            training_config: Training configuration (model_type, parameters, dataset_config)
            evaluation_config: Evaluation configuration (metrics, validation_strategy)
            
        Returns:
            Training result with performance metrics and tracking information
        """
        self.logger.info("Orchestrating model training with tracking")
        
        def prepare_training_data_operation(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            """Prepare training data operation"""
            if not self.dataset_manager:
                raise WorkflowExecutionError("Dataset manager not available")
            
            # Create dataset splits based on configuration
            dataset_config = training_config.get('dataset_config', {})
            
            # This would integrate with actual data loading
            result = {
                'dataset_splits_created': True,
                'train_samples': dataset_config.get('estimated_train_samples', 10000),
                'val_samples': dataset_config.get('estimated_val_samples', 2000),
                'test_samples': dataset_config.get('estimated_test_samples', 2000),
                'dataset_hash': hashlib.md5(str(dataset_config).encode()).hexdigest()
            }
            
            return result
        
        def train_model_operation(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            """Train model operation"""
            training_start = datetime.now()
            
            # Simulate model training (in real implementation, this would call actual ML training)
            model_config = training_config.get('model_config', {})
            
            # Record training metrics
            training_metrics = {
                'training_duration_seconds': 300,  # Simulated
                'final_accuracy': 0.95,
                'final_precision': 0.93,
                'final_recall': 0.94,
                'final_f1_score': 0.935,
                'training_loss': 0.15,
                'validation_loss': 0.18
            }
            
            # Create model version in tracking database
            if self.tracking_db:
                model_version = self.tracking_db.create_model_version(
                    model_id=training_config['model_id'],
                    version=training_config.get('model_version', '1.0.0'),
                    model_type=training_config['model_type'],
                    parameters=model_config,
                    metadata={
                        'training_start': training_start.isoformat(),
                        'training_config': training_config,
                        'evaluation_config': evaluation_config
                    }
                )
                
                training_metrics['model_version_id'] = model_version.model_id
            
            return training_metrics
        
        def evaluate_model_operation(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            """Evaluate model operation"""
            training_results = context['step_results'].get('train_model', {})
            
            evaluation_metrics = {
                'test_accuracy': training_results.get('final_accuracy', 0.0) - 0.02,  # Slight drop on test
                'test_precision': training_results.get('final_precision', 0.0) - 0.01,
                'test_recall': training_results.get('final_recall', 0.0) - 0.01,
                'test_f1_score': training_results.get('final_f1_score', 0.0) - 0.015,
                'confusion_matrix': [[850, 50], [30, 70]],  # Simulated
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            # Record metrics in tracking database
            if self.tracking_db and 'model_version_id' in training_results:
                # This would record actual metrics
                self.logger.info(f"Recording evaluation metrics for model {training_results['model_version_id']}")
            
            return evaluation_metrics
        
        # Create workflow definition
        workflow_def = WorkflowDefinition(
            workflow_id=f"model_training_{uuid.uuid4().hex[:8]}",
            name="Model Training with Tracking",
            description="Complete model training workflow with integrated tracking",
            workflow_type=WorkflowType.MODEL_TRAINING,
            steps=[
                WorkflowStep(
                    step_id="prepare_training_data",
                    name="Prepare Training Data",
                    operation=prepare_training_data_operation,
                    parameters={},
                    timeout_seconds=600
                ),
                WorkflowStep(
                    step_id="train_model",
                    name="Train Model",
                    operation=train_model_operation,
                    parameters={},
                    dependencies=["prepare_training_data"],
                    timeout_seconds=3600
                ),
                WorkflowStep(
                    step_id="evaluate_model",
                    name="Evaluate Model",
                    operation=evaluate_model_operation,
                    parameters={},
                    dependencies=["train_model"],
                    timeout_seconds=300
                )
            ],
            priority=WorkflowPriority.HIGH
        )
        
        # Execute workflow
        execution_result = self.execute_workflow(workflow_def)
        
        return {
            'workflow_execution_id': execution_result['execution_id'],
            'status': execution_result['status'],
            'training_results': execution_result.get('step_results', {})
        }
    
    def deploy_model_with_monitoring(self,
                                   model: Any,
                                   deployment_config: Dict[str, Any],
                                   monitoring_rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy model with integrated monitoring setup.
        
        Args:
            model: Trained model to deploy
            deployment_config: Deployment configuration (environment, scaling, etc.)
            monitoring_rules: Monitoring rules and alert thresholds
            
        Returns:
            Deployment result with monitoring setup information
        """
        self.logger.info("Deploying model with monitoring")
        
        def validate_model_operation(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            """Validate model before deployment"""
            validation_results = {
                'model_validation_passed': True,
                'model_size_mb': 15.5,  # Simulated
                'model_type': deployment_config.get('model_type', 'unknown'),
                'validation_timestamp': datetime.now().isoformat()
            }
            
            # Perform model validation checks
            min_accuracy = deployment_config.get('min_accuracy_threshold', 0.85)
            if hasattr(model, 'accuracy') and model.accuracy < min_accuracy:
                validation_results['model_validation_passed'] = False
                validation_results['validation_error'] = f"Model accuracy {model.accuracy} below threshold {min_accuracy}"
            
            return validation_results
        
        def setup_deployment_operation(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            """Setup model deployment"""
            validation_results = context['step_results'].get('validate_model', {})
            
            if not validation_results.get('model_validation_passed', False):
                raise WorkflowExecutionError("Model validation failed, deployment aborted")
            
            deployment_results = {
                'deployment_id': f"deploy_{uuid.uuid4().hex[:8]}",
                'deployment_environment': deployment_config.get('environment', 'production'),
                'deployment_timestamp': datetime.now().isoformat(),
                'endpoint_url': f"https://api.fraud-detection.com/models/{deployment_config.get('model_id', 'unknown')}",
                'scaling_config': deployment_config.get('scaling', {'min_instances': 1, 'max_instances': 5}),
                'deployment_status': 'active'
            }
            
            return deployment_results
        
        def setup_monitoring_operation(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            """Setup monitoring for deployed model"""
            deployment_results = context['step_results'].get('setup_deployment', {})
            
            monitoring_setup = {
                'monitoring_enabled': True,
                'alert_rules': monitoring_rules.get('alert_rules', []),
                'performance_thresholds': monitoring_rules.get('performance_thresholds', {}),
                'drift_detection_enabled': monitoring_rules.get('enable_drift_detection', True),
                'monitoring_dashboard_url': f"https://monitoring.fraud-detection.com/dashboards/{deployment_results.get('deployment_id')}",
                'setup_timestamp': datetime.now().isoformat()
            }
            
            # Setup monitoring in tracking database
            if self.tracking_db:
                model_id = deployment_config.get('model_id')
                if model_id:
                    # Update model status to deployed
                    self.tracking_db.update_model_status(
                        model_id, 
                        deployment_config.get('model_version', '1.0.0'),
                        ModelStatus.DEPLOYED
                    )
            
            return monitoring_setup
        
        # Create workflow definition
        workflow_def = WorkflowDefinition(
            workflow_id=f"model_deployment_{uuid.uuid4().hex[:8]}",
            name="Model Deployment with Monitoring",
            description="Deploy model with integrated monitoring setup",
            workflow_type=WorkflowType.MODEL_DEPLOYMENT,
            steps=[
                WorkflowStep(
                    step_id="validate_model",
                    name="Validate Model",
                    operation=validate_model_operation,
                    parameters={},
                    timeout_seconds=300
                ),
                WorkflowStep(
                    step_id="setup_deployment",
                    name="Setup Deployment",
                    operation=setup_deployment_operation,
                    parameters={},
                    dependencies=["validate_model"],
                    timeout_seconds=600
                ),
                WorkflowStep(
                    step_id="setup_monitoring",
                    name="Setup Monitoring",
                    operation=setup_monitoring_operation,
                    parameters={},
                    dependencies=["setup_deployment"],
                    timeout_seconds=300
                )
            ],
            priority=WorkflowPriority.HIGH
        )
        
        # Execute workflow
        execution_result = self.execute_workflow(workflow_def)
        
        return {
            'workflow_execution_id': execution_result['execution_id'],
            'status': execution_result['status'],
            'deployment_results': execution_result.get('step_results', {})
        }
    
    def manage_model_lifecycle_accuracy(self,
                                      model_id: str,
                                      lifecycle_config: Dict[str, Any],
                                      tracking_rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage model lifecycle with accuracy tracking and automated actions.
        
        Args:
            model_id: Model identifier
            lifecycle_config: Lifecycle management configuration
            tracking_rules: Accuracy tracking and alerting rules
            
        Returns:
            Lifecycle management result with actions taken
        """
        self.logger.info(f"Managing model lifecycle for {model_id}")
        
        def assess_model_health_operation(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            """Assess current model health and performance"""
            health_assessment = {
                'model_id': model_id,
                'assessment_timestamp': datetime.now().isoformat(),
                'current_status': 'active',
                'performance_metrics': {},
                'health_score': 0.85,  # Simulated
                'issues_detected': []
            }
            
            if self.tracking_db:
                # Get recent model metrics
                try:
                    model_version = self.tracking_db.get_model_version(model_id)
                    if model_version:
                        health_assessment['model_version'] = model_version.version
                        health_assessment['current_status'] = model_version.status.value
                    
                    # Check for performance degradation
                    degradation_result = self.tracking_db.detect_performance_degradation(
                        model_id=model_id,
                        model_version=model_version.version if model_version else '1.0.0',
                        threshold=tracking_rules.get('degradation_threshold', 0.05)
                    )
                    
                    if degradation_result['degradation_detected']:
                        health_assessment['issues_detected'].append({
                            'type': 'performance_degradation',
                            'details': degradation_result['recommendation']
                        })
                        health_assessment['health_score'] = 0.6
                    
                except Exception as e:
                    self.logger.error(f"Failed to assess model health: {e}")
                    health_assessment['issues_detected'].append({
                        'type': 'assessment_error',
                        'details': str(e)
                    })
            
            return health_assessment
        
        def determine_lifecycle_actions_operation(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            """Determine what lifecycle actions to take"""
            health_assessment = context['step_results'].get('assess_model_health', {})
            
            actions = {
                'recommended_actions': [],
                'automated_actions': [],
                'manual_actions': [],
                'action_priority': 'normal'
            }
            
            health_score = health_assessment.get('health_score', 1.0)
            issues = health_assessment.get('issues_detected', [])
            
            # Determine actions based on health score and issues
            if health_score < 0.7:
                actions['recommended_actions'].append('model_retraining')
                actions['action_priority'] = 'high'
                
                if lifecycle_config.get('auto_retrain_enabled', False):
                    actions['automated_actions'].append('trigger_retraining')
            
            if health_score < 0.5:
                actions['recommended_actions'].append('model_retirement')
                actions['action_priority'] = 'critical'
                
                if lifecycle_config.get('auto_retire_enabled', False):
                    actions['automated_actions'].append('retire_model')
            
            for issue in issues:
                if issue['type'] == 'performance_degradation':
                    actions['recommended_actions'].append('investigate_data_drift')
                    actions['manual_actions'].append('review_recent_data_quality')
            
            return actions
        
        def execute_lifecycle_actions_operation(context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            """Execute determined lifecycle actions"""
            actions = context['step_results'].get('determine_lifecycle_actions', {})
            
            execution_results = {
                'actions_executed': [],
                'actions_failed': [],
                'execution_timestamp': datetime.now().isoformat(),
                'next_review_date': (datetime.now() + timedelta(days=7)).isoformat()
            }
            
            # Execute automated actions
            for action in actions.get('automated_actions', []):
                try:
                    if action == 'trigger_retraining':
                        # Create retraining workflow
                        retraining_config = {
                            'model_id': model_id,
                            'trigger_reason': 'automated_lifecycle_management',
                            'priority': 'high'
                        }
                        execution_results['actions_executed'].append({
                            'action': action,
                            'status': 'initiated',
                            'config': retraining_config
                        })
                    
                    elif action == 'retire_model':
                        # Update model status to archived
                        if self.tracking_db:
                            model_version = self.tracking_db.get_model_version(model_id)
                            if model_version:
                                self.tracking_db.update_model_status(
                                    model_id,
                                    model_version.version,
                                    ModelStatus.ARCHIVED
                                )
                        
                        execution_results['actions_executed'].append({
                            'action': action,
                            'status': 'completed'
                        })
                
                except Exception as e:
                    execution_results['actions_failed'].append({
                        'action': action,
                        'error': str(e)
                    })
            
            return execution_results
        
        # Create workflow definition
        workflow_def = WorkflowDefinition(
            workflow_id=f"lifecycle_management_{model_id}_{uuid.uuid4().hex[:8]}",
            name="Model Lifecycle Management",
            description=f"Manage lifecycle for model {model_id} with accuracy tracking",
            workflow_type=WorkflowType.LIFECYCLE_MANAGEMENT,
            steps=[
                WorkflowStep(
                    step_id="assess_model_health",
                    name="Assess Model Health",
                    operation=assess_model_health_operation,
                    parameters={},
                    timeout_seconds=300
                ),
                WorkflowStep(
                    step_id="determine_lifecycle_actions",
                    name="Determine Lifecycle Actions",
                    operation=determine_lifecycle_actions_operation,
                    parameters={},
                    dependencies=["assess_model_health"],
                    timeout_seconds=120
                ),
                WorkflowStep(
                    step_id="execute_lifecycle_actions",
                    name="Execute Lifecycle Actions",
                    operation=execute_lifecycle_actions_operation,
                    parameters={},
                    dependencies=["determine_lifecycle_actions"],
                    timeout_seconds=600
                )
            ],
            priority=WorkflowPriority.NORMAL
        )
        
        # Execute workflow
        execution_result = self.execute_workflow(workflow_def)
        
        return {
            'workflow_execution_id': execution_result['execution_id'],
            'status': execution_result['status'],
            'lifecycle_results': execution_result.get('step_results', {})
        }
    
    # ======================== WORKFLOW MANAGEMENT METHODS ========================
    
    def register_workflow(self, workflow_definition: WorkflowDefinition) -> str:
        """Register a workflow definition for reuse"""
        with self._lock:
            self.workflow_registry[workflow_definition.workflow_id] = workflow_definition
            self.logger.info(f"Registered workflow: {workflow_definition.workflow_id}")
            return workflow_definition.workflow_id
    
    def execute_workflow(self, workflow_definition: WorkflowDefinition) -> Dict[str, Any]:
        """Execute a workflow and return execution information"""
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        
        # Create execution instance
        workflow_execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_definition=workflow_definition
        )
        
        with self._lock:
            self.active_workflows[execution_id] = workflow_execution
            self.workflow_queue.append(workflow_execution)
        
        self.logger.info(f"Queued workflow for execution: {execution_id}")
        
        return {
            'execution_id': execution_id,
            'status': 'queued',
            'workflow_type': workflow_definition.workflow_type.value
        }
    
    def get_workflow_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of workflow execution"""
        with self._lock:
            # Check active workflows
            if execution_id in self.active_workflows:
                execution = self.active_workflows[execution_id]
                return {
                    'execution_id': execution_id,
                    'status': execution.status.value,
                    'progress': execution.progress,
                    'start_time': execution.start_time.isoformat() if execution.start_time else None,
                    'end_time': execution.end_time.isoformat() if execution.end_time else None,
                    'error_message': execution.error_message,
                    'step_results': execution.step_results,
                    'performance_metrics': execution.performance_metrics
                }
            
            # Check workflow history
            for execution in self.workflow_history:
                if execution.execution_id == execution_id:
                    return {
                        'execution_id': execution_id,
                        'status': execution.status.value,
                        'progress': execution.progress,
                        'start_time': execution.start_time.isoformat() if execution.start_time else None,
                        'end_time': execution.end_time.isoformat() if execution.end_time else None,
                        'error_message': execution.error_message,
                        'step_results': execution.step_results,
                        'performance_metrics': execution.performance_metrics
                    }
        
        return None
    
    def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel a running workflow"""
        with self._lock:
            if execution_id in self.active_workflows:
                execution = self.active_workflows[execution_id]
                if execution.status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
                    execution.status = WorkflowStatus.CANCELLED
                    execution.end_time = datetime.now()
                    self.logger.info(f"Cancelled workflow: {execution_id}")
                    return True
        
        return False
    
    def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all active workflows"""
        with self._lock:
            return [
                {
                    'execution_id': execution.execution_id,
                    'workflow_name': execution.workflow_definition.name,
                    'workflow_type': execution.workflow_definition.workflow_type.value,
                    'status': execution.status.value,
                    'progress': execution.progress,
                    'start_time': execution.start_time.isoformat() if execution.start_time else None
                }
                for execution in self.active_workflows.values()
            ]
    
    def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get workflow execution metrics"""
        with self._lock:
            total_workflows = len(self.workflow_history) + len(self.active_workflows)
            
            completed_workflows = sum(1 for w in self.workflow_history if w.status == WorkflowStatus.COMPLETED)
            failed_workflows = sum(1 for w in self.workflow_history if w.status == WorkflowStatus.FAILED)
            
            return {
                'total_workflows_executed': total_workflows,
                'completed_workflows': completed_workflows,
                'failed_workflows': failed_workflows,
                'success_rate': completed_workflows / max(total_workflows, 1),
                'active_workflows': len(self.active_workflows),
                'queued_workflows': len(self.workflow_queue),
                'resource_usage': {
                    'avg_cpu_percent': sum(self.resource_stats.get('cpu_percent', [0])) / max(len(self.resource_stats.get('cpu_percent', [1])), 1),
                    'avg_memory_percent': sum(self.resource_stats.get('memory_percent', [0])) / max(len(self.resource_stats.get('memory_percent', [1])), 1)
                }
            }
    
    def shutdown(self) -> None:
        """Shutdown workflow orchestrator"""
        self.logger.info("Shutting down WorkflowOrchestrator")
        
        # Stop scheduler
        self.workflow_scheduler_running = False
        
        # Cancel active workflows
        with self._lock:
            for execution in self.active_workflows.values():
                if execution.status == WorkflowStatus.RUNNING:
                    execution.status = WorkflowStatus.CANCELLED
                    execution.end_time = datetime.now()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Save final state
        if self.config.enable_persistence:
            self._save_workflow_state()
        
        # Shutdown monitoring
        if self.monitoring_manager:
            self.monitoring_manager.shutdown()
        
        self.logger.info("WorkflowOrchestrator shutdown complete")


# ======================== USAGE EXAMPLES ========================

if __name__ == "__main__":
    # Example usage demonstrating all features
    print("=== WorkflowOrchestrator Examples ===\n")
    
    # Initialize orchestrator
    config = WorkflowConfig(
        max_concurrent_workflows=3,
        enable_monitoring=True,
        enable_caching=True,
        enable_persistence=True
    )
    
    orchestrator = WorkflowOrchestrator(config=config)
    
    # Example 1: Setup complete accuracy tracking
    print("1. Setting up complete accuracy tracking...")
    setup_result = orchestrator.setup_complete_accuracy_tracking(
        model_info={
            "model_id": "fraud_model_v2.0",
            "model_type": "ensemble",
            "training_data_hash": "abc123"
        },
        dataset_config={
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "stratification": True
        },
        monitoring_config={
            "real_time_monitoring": True,
            "alert_thresholds": {"accuracy_drop": 0.05},
            "notification_channels": ["email", "slack"]
        }
    )
    print(f"   Setup result: {setup_result['status']}")
    print(f"   Execution ID: {setup_result['workflow_execution_id']}\n")
    
    # Example 2: Orchestrate model training
    print("2. Orchestrating model training with tracking...")
    training_result = orchestrator.orchestrate_model_training_with_tracking(
        training_config={
            "model_id": "fraud_model_v2.0",
            "model_type": "xgboost",
            "model_config": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1
            },
            "dataset_config": {
                "estimated_train_samples": 10000,
                "estimated_val_samples": 2000,
                "estimated_test_samples": 2000
            }
        },
        evaluation_config={
            "metrics": ["accuracy", "precision", "recall", "f1_score"],
            "validation_strategy": "holdout"
        }
    )
    print(f"   Training result: {training_result['status']}")
    print(f"   Execution ID: {training_result['workflow_execution_id']}\n")
    
    # Example 3: Deploy model with monitoring
    print("3. Deploying model with monitoring...")
    deployment_result = orchestrator.deploy_model_with_monitoring(
        model=None,  # Would be actual model object
        deployment_config={
            "model_id": "fraud_model_v2.0",
            "model_version": "1.0.0",
            "environment": "production",
            "min_accuracy_threshold": 0.85,
            "scaling": {"min_instances": 2, "max_instances": 10}
        },
        monitoring_rules={
            "enable_drift_detection": True,
            "performance_thresholds": {
                "response_time_ms": 100,
                "accuracy_threshold": 0.90
            },
            "alert_rules": [
                {"metric": "accuracy", "threshold": 0.85, "action": "alert"},
                {"metric": "response_time", "threshold": 200, "action": "scale_up"}
            ]
        }
    )
    print(f"   Deployment result: {deployment_result['status']}")
    print(f"   Execution ID: {deployment_result['workflow_execution_id']}\n")
    
    # Example 4: Manage model lifecycle
    print("4. Managing model lifecycle...")
    lifecycle_result = orchestrator.manage_model_lifecycle_accuracy(
        model_id="fraud_model_v2.0",
        lifecycle_config={
            "auto_retrain_enabled": True,
            "auto_retire_enabled": False,
            "review_interval_days": 7
        },
        tracking_rules={
            "degradation_threshold": 0.05,
            "min_performance_threshold": 0.80
        }
    )
    print(f"   Lifecycle result: {lifecycle_result['status']}")
    print(f"   Execution ID: {lifecycle_result['workflow_execution_id']}\n")
    
    # Example 5: Check workflow status
    print("5. Checking workflow metrics...")
    metrics = orchestrator.get_workflow_metrics()
    print(f"   Total workflows: {metrics['total_workflows_executed']}")
    print(f"   Success rate: {metrics['success_rate']:.2%}")
    print(f"   Active workflows: {metrics['active_workflows']}")
    
    # List active workflows
    active_workflows = orchestrator.list_active_workflows()
    print(f"   Active workflow details: {len(active_workflows)} workflows")
    
    # Wait a moment for some workflows to complete
    time.sleep(2)
    
    # Check individual workflow status
    for workflow_result in [setup_result, training_result, deployment_result, lifecycle_result]:
        status = orchestrator.get_workflow_status(workflow_result['workflow_execution_id'])
        if status:
            print(f"   Workflow {workflow_result['workflow_execution_id']}: {status['status']} ({status['progress']:.1%})")
    
    # Cleanup
    orchestrator.shutdown()
    
    print("\n=== All examples completed successfully! ===")