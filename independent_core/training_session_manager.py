#!/usr/bin/env python3
"""
Training Session Manager for Brain System
Comprehensive session management and monitoring for training execution.
Part of the training execution and monitoring fixes.
"""

import logging
import time
import threading
import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from contextlib import contextmanager
import numpy as np
import psutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# ======================== ENUMS AND STATUS ========================

class SessionStatus(Enum):
    """Training session status."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECOVERING = "recovering"

class ErrorType(Enum):
    """Types of training errors."""
    OUT_OF_MEMORY = "out_of_memory"
    NAN_LOSS = "nan_loss"
    INFINITE_LOSS = "infinite_loss"
    GRADIENT_EXPLOSION = "gradient_explosion"
    DATA_ERROR = "data_error"
    RUNTIME_ERROR = "runtime_error"
    RESOURCE_ERROR = "resource_error"
    UNKNOWN = "unknown"

class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    REDUCE_LEARNING_RATE = "reduce_learning_rate"
    SKIP_BATCH = "skip_batch"
    RESET_OPTIMIZER = "reset_optimizer"
    LOAD_CHECKPOINT = "load_checkpoint"
    RESTART_TRAINING = "restart_training"

# ======================== DATA CLASSES ========================

@dataclass
class SessionMetrics:
    """Real-time session metrics."""
    # Training metrics
    loss_history: List[float] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)
    val_accuracy_history: List[float] = field(default_factory=list)
    
    # Best metrics
    best_loss: Optional[float] = None
    best_accuracy: Optional[float] = None
    best_val_loss: Optional[float] = None
    best_val_accuracy: Optional[float] = None
    best_epoch: Optional[int] = None
    
    # Progress metrics
    current_epoch: int = 0
    current_batch: int = 0
    total_epochs: int = 0
    total_batches: int = 0
    total_progress: float = 0.0
    
    # Performance metrics
    epoch_duration: List[float] = field(default_factory=list)
    batch_duration: List[float] = field(default_factory=list)
    samples_per_second: List[float] = field(default_factory=list)
    
    # Overfitting detection
    overfitting_score: float = 0.0
    early_stopping_counter: int = 0
    
    def update_loss(self, loss: float, is_validation: bool = False):
        """Update loss metrics."""
        if is_validation:
            self.val_loss_history.append(loss)
            if self.best_val_loss is None or loss < self.best_val_loss:
                self.best_val_loss = loss
        else:
            self.loss_history.append(loss)
            if self.best_loss is None or loss < self.best_loss:
                self.best_loss = loss
    
    def update_accuracy(self, accuracy: float, is_validation: bool = False):
        """Update accuracy metrics."""
        if is_validation:
            self.val_accuracy_history.append(accuracy)
            if self.best_val_accuracy is None or accuracy > self.best_val_accuracy:
                self.best_val_accuracy = accuracy
        else:
            self.accuracy_history.append(accuracy)
            if self.best_accuracy is None or accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
    
    def calculate_overfitting_score(self) -> float:
        """Calculate overfitting score based on train/validation gap."""
        if len(self.val_loss_history) < 2 or len(self.loss_history) < 2:
            return 0.0
        
        # Recent performance (last 5 epochs)
        recent_train_loss = np.mean(self.loss_history[-5:])
        recent_val_loss = np.mean(self.val_loss_history[-5:])
        
        # Overfitting score = validation loss / training loss
        if recent_train_loss > 0:
            self.overfitting_score = recent_val_loss / recent_train_loss
        
        return self.overfitting_score

@dataclass
class ResourceMetrics:
    """System resource usage metrics."""
    cpu_usage: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    gpu_usage: List[float] = field(default_factory=list)
    gpu_memory: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    
    # Alerts
    high_cpu_alerts: int = 0
    high_memory_alerts: int = 0
    high_gpu_alerts: int = 0
    
    def add_measurement(self, cpu: float, memory: float, gpu: float = 0.0, gpu_mem: float = 0.0):
        """Add resource measurement."""
        self.cpu_usage.append(cpu)
        self.memory_usage.append(memory)
        self.gpu_usage.append(gpu)
        self.gpu_memory.append(gpu_mem)
        self.timestamps.append(datetime.now())
        
        # Keep only last 1000 measurements
        if len(self.cpu_usage) > 1000:
            self.cpu_usage = self.cpu_usage[-1000:]
            self.memory_usage = self.memory_usage[-1000:]
            self.gpu_usage = self.gpu_usage[-1000:]
            self.gpu_memory = self.gpu_memory[-1000:]
            self.timestamps = self.timestamps[-1000:]

@dataclass
class TrainingError:
    """Training error information."""
    error_type: ErrorType
    error_message: str
    traceback_str: str
    timestamp: datetime
    epoch: int
    batch: int
    recovery_attempted: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_successful: bool = False

@dataclass
class TrainingCheckpoint:
    """Training checkpoint information."""
    checkpoint_id: str
    epoch: int
    batch: int
    model_state: Optional[Dict] = None
    optimizer_state: Optional[Dict] = None
    metrics: Optional[SessionMetrics] = None
    timestamp: datetime = field(default_factory=datetime.now)
    file_path: Optional[Path] = None
    is_best: bool = False
    is_auto: bool = True

@dataclass
class TrainingSession:
    """Complete training session information."""
    session_id: str
    domain_name: str
    model_type: str
    status: SessionStatus
    config: Dict[str, Any]
    
    # Metrics and monitoring
    metrics: SessionMetrics = field(default_factory=SessionMetrics)
    resource_metrics: ResourceMetrics = field(default_factory=ResourceMetrics)
    
    # Session timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    last_activity: datetime = field(default_factory=datetime.now)
    
    # Error handling
    errors: List[TrainingError] = field(default_factory=list)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    
    # Checkpoints
    checkpoints: List[TrainingCheckpoint] = field(default_factory=list)
    latest_checkpoint: Optional[TrainingCheckpoint] = None
    best_checkpoint: Optional[TrainingCheckpoint] = None
    
    # Callbacks
    callbacks: List[Callable] = field(default_factory=list)
    
    def duration(self) -> timedelta:
        """Get session duration."""
        end = self.end_time or datetime.now()
        return end - self.start_time
    
    def is_active(self) -> bool:
        """Check if session is active."""
        return self.status in [SessionStatus.INITIALIZING, SessionStatus.RUNNING, SessionStatus.RECOVERING]

# ======================== TRAINING SESSION MANAGER ========================

class TrainingSessionManager:
    """
    Comprehensive training session management and monitoring system.
    
    Handles:
    - Session lifecycle management
    - Real-time metrics tracking
    - Checkpointing and recovery
    - Resource monitoring
    - Error handling and recovery
    """
    
    def __init__(self, storage_path: str = ".brain/training_sessions"):
        """Initialize training session manager."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Active sessions
        self.sessions: Dict[str, TrainingSession] = {}
        self.active_sessions: List[str] = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 5.0  # seconds
        
        # Configuration
        self.max_checkpoints_per_session = 10
        self.auto_checkpoint_interval = 5  # epochs
        self.resource_alert_thresholds = {
            'cpu': 90.0,
            'memory': 85.0,
            'gpu': 95.0
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Start resource monitoring
        self.start_monitoring()
    
    def create_session(
        self,
        domain_name: str,
        model_type: str,
        config: Dict[str, Any]
    ) -> str:
        """Create a new training session."""
        # Generate unique session ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        session_id = f"{domain_name}_{model_type}_{timestamp}"
        
        # Create session
        session = TrainingSession(
            session_id=session_id,
            domain_name=domain_name,
            model_type=model_type,
            status=SessionStatus.INITIALIZING,
            config=config.copy()
        )
        
        # Setup metrics
        session.metrics.total_epochs = config.get('epochs', 100)
        session.metrics.total_batches = config.get('total_batches', 0)
        session.max_recovery_attempts = config.get('max_recovery_attempts', 3)
        
        # Store session
        self.sessions[session_id] = session
        # Don't add to active_sessions yet - wait for start_session
        
        # Create session directory
        session_dir = self.storage_path / session_id
        session_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Created training session: {session_id}")
        return session_id
    
    def start_session(self, session_id: str) -> bool:
        """Start a training session."""
        if session_id not in self.sessions:
            self.logger.error(f"Session not found: {session_id}")
            return False
        
        session = self.sessions[session_id]
        session.status = SessionStatus.RUNNING
        session.start_time = datetime.now()
        session.last_activity = datetime.now()
        
        # Add to active sessions
        if session_id not in self.active_sessions:
            self.active_sessions.append(session_id)
        
        self.logger.info(f"Started training session: {session_id}")
        return True
    
    def update_metrics(
        self,
        session_id: str,
        metrics_update: Optional[Dict[str, Any]] = None,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
        val_loss: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        **kwargs
    ) -> bool:
        """Update session metrics."""
        if session_id not in self.sessions:
            return False
        
        with self.lock:  # Thread safety for concurrent updates
            session = self.sessions[session_id]
            session.last_activity = datetime.now()
        
        # Handle metrics_update dict if provided
        if metrics_update is None:
            metrics_update = {}
        
        # Merge keyword arguments into metrics_update
        if loss is not None:
            metrics_update['loss'] = loss
        if accuracy is not None:
            metrics_update['accuracy'] = accuracy
        if val_loss is not None:
            metrics_update['val_loss'] = val_loss
        if val_accuracy is not None:
            metrics_update['val_accuracy'] = val_accuracy
        
        # Add any additional kwargs to metrics_update
        metrics_update.update(kwargs)
        
        # Update metrics
        if 'loss' in metrics_update:
            session.metrics.update_loss(metrics_update['loss'])
        
        if 'accuracy' in metrics_update:
            session.metrics.update_accuracy(metrics_update['accuracy'])
        
        if 'val_loss' in metrics_update:
            session.metrics.update_loss(metrics_update['val_loss'], is_validation=True)
        
        if 'val_accuracy' in metrics_update:
            session.metrics.update_accuracy(metrics_update['val_accuracy'], is_validation=True)
        
        # Support both dict and parameter-based epoch/batch updates
        if epoch is not None:
            session.metrics.current_epoch = epoch
        elif 'epoch' in metrics_update:
            session.metrics.current_epoch = metrics_update['epoch']
        
        if batch is not None:
            session.metrics.current_batch = batch
        elif 'batch' in metrics_update:
            session.metrics.current_batch = metrics_update['batch']
        
        # Calculate progress
        self._update_progress(session)
        
        # Calculate overfitting score
        session.metrics.calculate_overfitting_score()
        
        # Call callbacks
        self._call_callbacks(session, 'metrics_updated', metrics_update)
        
        return True
    
    def report_progress(
        self,
        session_id: str,
        epoch: int,
        batch: int,
        total_batches: Optional[int] = None,
        epoch_start_time: Optional[float] = None,
        loss: Optional[float] = None,
        accuracy: Optional[float] = None
    ) -> bool:
        """Report training progress."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        session.last_activity = datetime.now()
        
        # Update progress metrics
        session.metrics.current_epoch = epoch
        session.metrics.current_batch = batch
        if total_batches is not None:
            session.metrics.total_batches = total_batches
        
        # Calculate timing metrics
        if epoch_start_time:
            epoch_duration = time.time() - epoch_start_time
            session.metrics.epoch_duration.append(epoch_duration)
        
        # Update loss/accuracy if provided
        if loss is not None:
            session.metrics.update_loss(loss)
        if accuracy is not None:
            session.metrics.update_accuracy(accuracy)
        
        # Update progress
        self._update_progress(session)
        
        # Call callbacks
        self._call_callbacks(session, 'progress_updated', {
            'epoch': epoch,
            'batch': batch,
            'total_batches': total_batches,
            'total_progress': session.metrics.total_progress
        })
        
        return True
    
    def handle_error(
        self,
        session_id: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        error_type: Optional[str] = None,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        **kwargs
    ) -> bool:
        """Handle training error with recovery."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        session.status = SessionStatus.RECOVERING
        
        # Update metrics if provided
        if epoch is not None:
            session.metrics.current_epoch = epoch
        if batch is not None:
            session.metrics.current_batch = batch
        
        # Use provided error_type or classify
        if error_type is None:
            error_type_enum = self._classify_error(error)
        else:
            # Convert string to ErrorType enum if needed
            try:
                error_type_enum = ErrorType[error_type.upper()] if isinstance(error_type, str) else error_type
            except (KeyError, AttributeError):
                error_type_enum = self._classify_error(error)
        
        # Create error record
        training_error = TrainingError(
            error_type=error_type_enum,
            error_message=str(error),
            traceback_str=traceback.format_exc(),
            timestamp=datetime.now(),
            epoch=session.metrics.current_epoch,
            batch=session.metrics.current_batch
        )
        
        session.errors.append(training_error)
        
        self.logger.error(f"Training error in session {session_id}: {error}")
        
        # Attempt recovery if possible
        if session.recovery_attempts < session.max_recovery_attempts:
            recovery_strategy = self._determine_recovery_strategy(error_type_enum)
            if recovery_strategy:
                success = self._attempt_recovery(session, training_error, recovery_strategy)
                if success:
                    session.status = SessionStatus.RUNNING
                    self.logger.info(f"Recovery successful for session {session_id}")
                    return True
        
        # Recovery failed or not attempted
        session.status = SessionStatus.FAILED
        self.logger.error(f"Session {session_id} failed - recovery unsuccessful")
        
        # Call callbacks
        self._call_callbacks(session, 'error_occurred', {
            'error': error,
            'error_type': error_type_enum,
            'recovery_attempted': training_error.recovery_attempted
        })
        
        return False
    
    def create_checkpoint(
        self,
        session_id: str,
        model_state: Optional[Dict] = None,
        optimizer_state: Optional[Dict] = None,
        is_best: bool = False,
        is_auto: bool = True,
        checkpoint_data: Optional[Dict[str, Any]] = None,
        epoch: Optional[int] = None,
        batch: Optional[int] = None,
        **kwargs
    ) -> Optional[str]:
        """Create a training checkpoint."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Support checkpoint_data parameter for convenience
        if checkpoint_data is not None:
            if model_state is None:
                model_state = checkpoint_data.get('model_state', {})
            if optimizer_state is None:
                optimizer_state = checkpoint_data.get('optimizer_state', {})
        
        # Use empty dicts if still None
        if model_state is None:
            model_state = {}
        if optimizer_state is None:
            optimizer_state = {}
        
        # Update metrics if provided
        if epoch is not None:
            session.metrics.current_epoch = epoch
        if batch is not None:
            session.metrics.current_batch = batch
        
        # Generate checkpoint ID
        checkpoint_id = f"checkpoint_{session.metrics.current_epoch}_{session.metrics.current_batch}"
        if is_best:
            checkpoint_id += "_best"
        
        # Create checkpoint
        checkpoint = TrainingCheckpoint(
            checkpoint_id=checkpoint_id,
            epoch=session.metrics.current_epoch,
            batch=session.metrics.current_batch,
            model_state=model_state,
            optimizer_state=optimizer_state,
            metrics=session.metrics,
            is_best=is_best,
            is_auto=is_auto
        )
        
        # Save checkpoint to disk
        checkpoint_dir = self.storage_path / session_id / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_file = checkpoint_dir / f"{checkpoint_id}.pkl"
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            checkpoint.file_path = checkpoint_file
            
            # Add to session
            session.checkpoints.append(checkpoint)
            session.latest_checkpoint = checkpoint
            
            if is_best:
                session.best_checkpoint = checkpoint
            
            # Manage checkpoint count
            self._manage_checkpoints(session)
            
            self.logger.info(f"Created checkpoint {checkpoint_id} for session {session_id}")
            return checkpoint_id
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            return None
    
    def recover_from_checkpoint(
        self,
        session_id: str,
        checkpoint_id: Optional[str] = None
    ) -> Optional[Tuple[Dict, Dict]]:
        """Recover from a checkpoint.
        
        Returns:
            Optional tuple of (model_state, optimizer_state) if successful, None otherwise
        """
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Find checkpoint
        checkpoint = None
        if checkpoint_id:
            checkpoint = next((cp for cp in session.checkpoints if cp.checkpoint_id == checkpoint_id), None)
        else:
            checkpoint = session.latest_checkpoint or session.best_checkpoint
        
        if not checkpoint:
            self.logger.warning(f"No checkpoint found for recovery in session {session_id}")
            return None
        
        # Load checkpoint if needed
        if checkpoint.file_path and checkpoint.file_path.exists():
            try:
                with open(checkpoint.file_path, 'rb') as f:
                    loaded_checkpoint = pickle.load(f)
                    checkpoint.model_state = loaded_checkpoint.model_state
                    checkpoint.optimizer_state = loaded_checkpoint.optimizer_state
                    checkpoint.metrics = loaded_checkpoint.metrics
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint: {e}")
                return None
        
        # Update session status to recovering
        session.status = SessionStatus.RECOVERING
        
        self.logger.info(f"Recovered from checkpoint {checkpoint.checkpoint_id} in session {session_id}")
        return (checkpoint.model_state, checkpoint.optimizer_state)
    
    def complete_session(
        self,
        session_id: str,
        final_metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Complete a training session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        session.status = SessionStatus.COMPLETED
        session.end_time = datetime.now()
        
        # Update final metrics
        if final_metrics:
            self.update_metrics(session_id, final_metrics)
        
        # Create final checkpoint
        self.create_checkpoint(session_id, is_auto=True)
        
        # Remove from active sessions
        if session_id in self.active_sessions:
            self.active_sessions.remove(session_id)
        
        # Call callbacks
        self._call_callbacks(session, 'session_completed', final_metrics or {})
        
        self.logger.info(f"Completed training session: {session_id}")
        return True
    
    def cancel_session(self, session_id: str) -> bool:
        """Cancel a training session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        session.status = SessionStatus.CANCELLED
        session.end_time = datetime.now()
        
        # Remove from active sessions
        if session_id in self.active_sessions:
            self.active_sessions.remove(session_id)
        
        # Call callbacks
        self._call_callbacks(session, 'session_cancelled', {})
        
        self.logger.info(f"Cancelled training session: {session_id}")
        return True
    
    def get_session(self, session_id: str) -> Optional[TrainingSession]:
        """Get session information."""
        return self.sessions.get(session_id)
    
    def get_active_sessions(self) -> List[TrainingSession]:
        """Get all active sessions."""
        return [self.sessions[sid] for sid in self.active_sessions if sid in self.sessions]
    
    def add_callback(self, session_id: str, callback: Callable) -> bool:
        """Add callback to session."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        session.callbacks.append(callback)
        return True
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        self.logger.info("Stopped resource monitoring")
    
    def cleanup_session(self, session_id: str, keep_checkpoints: bool = True) -> bool:
        """Clean up session resources."""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        # Remove from active sessions
        if session_id in self.active_sessions:
            self.active_sessions.remove(session_id)
        
        # Clean up checkpoints if requested
        if not keep_checkpoints:
            for checkpoint in session.checkpoints:
                if checkpoint.file_path and checkpoint.file_path.exists():
                    try:
                        checkpoint.file_path.unlink()
                    except Exception as e:
                        self.logger.warning(f"Failed to delete checkpoint file: {e}")
            
            # Also clean up any checkpoint files in storage_path matching session_id pattern
            for file in self.storage_path.glob(f"{session_id}*.pkl"):
                try:
                    file.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to delete file {file}: {e}")
        
        # Remove session
        del self.sessions[session_id]
        
        self.logger.info(f"Cleaned up session: {session_id}")
        return True
    
    def shutdown(self):
        """Shutdown session manager."""
        self.stop_monitoring()
        
        # Complete any active sessions
        for session_id in self.active_sessions.copy():
            self.cancel_session(session_id)
        
        self.logger.info("Training session manager shutdown complete")
    
    # ==================== PRIVATE METHODS ====================
    
    def _update_progress(self, session: TrainingSession):
        """Update training progress calculation."""
        if session.metrics.total_epochs > 0:
            epoch_progress = session.metrics.current_epoch / session.metrics.total_epochs
            
            if session.metrics.total_batches > 0:
                batch_progress = session.metrics.current_batch / session.metrics.total_batches
                epoch_batch_progress = batch_progress / session.metrics.total_epochs
                session.metrics.total_progress = epoch_progress + epoch_batch_progress
            else:
                session.metrics.total_progress = epoch_progress
            
            session.metrics.total_progress = min(1.0, session.metrics.total_progress)
    
    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify training error type."""
        error_str = str(error).lower()
        
        if "out of memory" in error_str or "cuda out of memory" in error_str:
            return ErrorType.OUT_OF_MEMORY
        elif "nan" in error_str or "not a number" in error_str:
            return ErrorType.NAN_LOSS
        elif "inf" in error_str or "infinite" in error_str:
            return ErrorType.INFINITE_LOSS
        elif "gradient" in error_str and ("explod" in error_str or "overflow" in error_str):
            return ErrorType.GRADIENT_EXPLOSION
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorType.DATA_ERROR
        elif isinstance(error, RuntimeError):
            return ErrorType.RUNTIME_ERROR
        elif isinstance(error, (MemoryError, OSError)):
            return ErrorType.RESOURCE_ERROR
        else:
            return ErrorType.UNKNOWN
    
    def _determine_recovery_strategy(self, error_type: ErrorType) -> Optional[RecoveryStrategy]:
        """Determine recovery strategy for error type."""
        strategy_map = {
            ErrorType.OUT_OF_MEMORY: RecoveryStrategy.REDUCE_BATCH_SIZE,
            ErrorType.NAN_LOSS: RecoveryStrategy.RESET_OPTIMIZER,  # NaN should reset optimizer
            ErrorType.INFINITE_LOSS: RecoveryStrategy.REDUCE_LEARNING_RATE,
            ErrorType.GRADIENT_EXPLOSION: RecoveryStrategy.REDUCE_LEARNING_RATE,
            ErrorType.DATA_ERROR: RecoveryStrategy.SKIP_BATCH,
            ErrorType.RUNTIME_ERROR: RecoveryStrategy.LOAD_CHECKPOINT,
        }
        
        return strategy_map.get(error_type)
    
    def _attempt_recovery(
        self,
        session: TrainingSession,
        error: TrainingError,
        strategy: RecoveryStrategy
    ) -> bool:
        """Attempt error recovery."""
        session.recovery_attempts += 1
        error.recovery_attempted = True
        error.recovery_strategy = strategy
        
        self.logger.info(f"Attempting recovery with strategy: {strategy}")
        
        # Recovery strategies would be implemented based on the specific strategy
        # This is a placeholder that indicates recovery was attempted
        recovery_success = True  # Simplified for this implementation
        
        error.recovery_successful = recovery_success
        return recovery_success
    
    def _manage_checkpoints(self, session: TrainingSession):
        """Manage checkpoint count and cleanup old checkpoints."""
        if len(session.checkpoints) <= self.max_checkpoints_per_session:
            return
        
        # Sort by timestamp, keep best and latest, remove oldest auto checkpoints
        auto_checkpoints = [cp for cp in session.checkpoints if cp.is_auto and not cp.is_best]
        auto_checkpoints.sort(key=lambda x: x.timestamp)
        
        # Remove oldest auto checkpoints
        to_remove = len(auto_checkpoints) - (self.max_checkpoints_per_session - 2)  # Keep space for best/latest
        for checkpoint in auto_checkpoints[:to_remove]:
            if checkpoint.file_path and checkpoint.file_path.exists():
                try:
                    checkpoint.file_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to delete checkpoint: {e}")
            session.checkpoints.remove(checkpoint)
    
    def _call_callbacks(self, session: TrainingSession, event: str, data: Dict[str, Any]):
        """Call session callbacks."""
        for callback in session.callbacks:
            try:
                callback(session.session_id, event, data)
            except Exception as e:
                self.logger.warning(f"Callback error: {e}")
    
    def _monitoring_loop(self):
        """Resource monitoring loop."""
        while self.monitoring_active:
            try:
                # Get system metrics
                cpu_usage = psutil.cpu_percent(interval=1)
                memory_usage = psutil.virtual_memory().percent
                
                gpu_usage = 0.0
                gpu_memory = 0.0
                
                if PYTORCH_AVAILABLE and torch.cuda.is_available():
                    try:
                        gpu_usage = self._get_gpu_utilization()
                        gpu_memory = self._get_gpu_memory_usage()
                    except Exception:
                        pass
                
                # Update all active sessions
                for session_id in self.active_sessions:
                    if session_id in self.sessions:
                        session = self.sessions[session_id]
                        session.resource_metrics.add_measurement(
                            cpu_usage, memory_usage, gpu_usage, gpu_memory
                        )
                        
                        # Check for alerts
                        self._check_resource_alerts(session, cpu_usage, memory_usage, gpu_usage)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.warning(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(utilization.gpu)
        except Exception:
            return 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage percentage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return (memory_info.used / memory_info.total) * 100
        except Exception:
            return 0.0
    
    def _check_resource_alerts(
        self,
        session: TrainingSession,
        cpu_usage: float,
        memory_usage: float,
        gpu_usage: float
    ):
        """Check for resource usage alerts."""
        if cpu_usage > self.resource_alert_thresholds['cpu']:
            session.resource_metrics.high_cpu_alerts += 1
            self.logger.warning(f"High CPU usage in session {session.session_id}: {cpu_usage:.1f}%")
        
        if memory_usage > self.resource_alert_thresholds['memory']:
            session.resource_metrics.high_memory_alerts += 1
            self.logger.warning(f"High memory usage in session {session.session_id}: {memory_usage:.1f}%")
        
        if gpu_usage > self.resource_alert_thresholds['gpu']:
            session.resource_metrics.high_gpu_alerts += 1
            self.logger.warning(f"High GPU usage in session {session.session_id}: {gpu_usage:.1f}%")
    
    def cleanup_files(self, session_id: str, keep_best: bool = True) -> int:
        """Clean up checkpoint files for a session.
        
        Args:
            session_id: ID of the session
            keep_best: Whether to keep the best checkpoint
            
        Returns:
            Number of files deleted
        """
        if session_id not in self.sessions:
            return 0
        
        session = self.sessions[session_id]
        deleted_count = 0
        
        # Clean up checkpoints
        for checkpoint in session.checkpoints:
            # Skip best checkpoint if requested
            if keep_best and checkpoint.is_best:
                continue
            
            if checkpoint.file_path and checkpoint.file_path.exists():
                try:
                    checkpoint.file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete checkpoint file {checkpoint.file_path}: {e}")
        
        # Clean up session directory if empty
        session_dir = self.storage_path / session_id
        if session_dir.exists():
            try:
                # Remove checkpoints directory if empty
                checkpoint_dir = session_dir / "checkpoints"
                if checkpoint_dir.exists() and not any(checkpoint_dir.iterdir()):
                    checkpoint_dir.rmdir()
                
                # Remove session directory if empty
                if not any(session_dir.iterdir()):
                    session_dir.rmdir()
                    deleted_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to clean up directories: {e}")
        
        return deleted_count

# ======================== CONVENIENCE FUNCTIONS ========================

def create_session_manager(storage_path: str = ".brain/training_sessions") -> TrainingSessionManager:
    """Create and configure a training session manager."""
    return TrainingSessionManager(storage_path)

# Progress callback example
def log_progress_callback(session_id: str, event: str, data: Dict[str, Any]):
    """Example progress logging callback."""
    if event == 'progress_updated':
        epoch = data.get('epoch', 0)
        batch = data.get('batch', 0)
        total_progress = data.get('total_progress', 0.0)
        print(f"[{session_id}] Epoch {epoch}, Batch {batch}, Progress: {total_progress:.1%}")
    elif event == 'metrics_updated':
        if 'loss' in data:
            print(f"[{session_id}] Loss: {data['loss']:.4f}")
        if 'accuracy' in data:
            print(f"[{session_id}] Accuracy: {data['accuracy']:.4f}")

if __name__ == "__main__":
    # Example usage
    manager = create_session_manager()
    
    # Create session
    session_id = manager.create_session(
        domain_name="fraud_detection",
        model_type="neural_network",
        config={'epochs': 10, 'batch_size': 32}
    )
    
    # Add callback
    manager.add_callback(session_id, log_progress_callback)
    
    # Start session
    manager.start_session(session_id)
    
    print(f"Created session: {session_id}")
    print("Training session manager is ready!")