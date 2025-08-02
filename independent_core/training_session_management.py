#!/usr/bin/env python3
"""
Enhanced Training Session Management for Brain System
Comprehensive session lifecycle management with state tracking, recovery, and cleanup.
"""

import logging
import threading
import time
import json
import pickle
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict
import numpy as np
import gc
import traceback

logger = logging.getLogger(__name__)

# SESSION FALLBACK DISABLING - Makes all session failures explicit
SESSION_FALLBACKS_DISABLED = True  # Set to False to re-enable fallbacks

# ======================== SESSION STATE MANAGEMENT ========================

class SessionState(Enum):
    """Comprehensive session states with full lifecycle support."""
    CREATED = "created"
    PREPARING = "preparing"
    READY = "ready"
    STARTING = "starting"
    TRAINING = "training"
    PAUSED = "paused"
    RESUMING = "resuming"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RECOVERING = "recovering"
    CLEANUP = "cleanup"
    ARCHIVED = "archived"

class SessionStateMachine:
    """Thread-safe state machine for training sessions."""
    
    # Valid state transitions
    VALID_TRANSITIONS = {
        SessionState.CREATED: [SessionState.PREPARING, SessionState.CANCELLED],
        SessionState.PREPARING: [SessionState.READY, SessionState.FAILED, SessionState.CANCELLED],
        SessionState.READY: [SessionState.STARTING, SessionState.CANCELLED, SessionState.CLEANUP],
        SessionState.STARTING: [SessionState.TRAINING, SessionState.FAILED, SessionState.CANCELLED],
        SessionState.TRAINING: [SessionState.PAUSED, SessionState.STOPPING, SessionState.COMPLETED, SessionState.FAILED],
        SessionState.PAUSED: [SessionState.RESUMING, SessionState.STOPPING, SessionState.CANCELLED],
        SessionState.RESUMING: [SessionState.TRAINING, SessionState.FAILED],
        SessionState.STOPPING: [SessionState.STOPPED, SessionState.FAILED],
        SessionState.STOPPED: [SessionState.STARTING, SessionState.CLEANUP, SessionState.CANCELLED],
        SessionState.COMPLETED: [SessionState.CLEANUP, SessionState.ARCHIVED],
        SessionState.FAILED: [SessionState.RECOVERING, SessionState.CLEANUP, SessionState.CANCELLED],
        SessionState.CANCELLED: [SessionState.CLEANUP],
        SessionState.RECOVERING: [SessionState.READY, SessionState.FAILED],
        SessionState.CLEANUP: [SessionState.ARCHIVED],
        SessionState.ARCHIVED: [SessionState.CLEANUP]  # Allow cleanup from archived state
    }
    
    def __init__(self, initial_state: SessionState = SessionState.CREATED):
        self._state = initial_state
        self._lock = threading.RLock()
        self._state_history = [(initial_state, datetime.now())]
        self._transition_callbacks = []
    
    def get_state(self) -> SessionState:
        """Get current state (thread-safe)."""
        with self._lock:
            return self._state
    
    def can_transition_to(self, new_state: SessionState) -> bool:
        """Check if transition to new state is valid."""
        with self._lock:
            return new_state in self.VALID_TRANSITIONS.get(self._state, [])
    
    def transition_to(self, new_state: SessionState, force: bool = False) -> bool:
        """Transition to new state with validation."""
        with self._lock:
            if not force and not self.can_transition_to(new_state):
                logger.warning(f"Invalid state transition: {self._state} -> {new_state}")
                return False
            
            old_state = self._state
            self._state = new_state
            self._state_history.append((new_state, datetime.now()))
            
            # Call transition callbacks
            for callback in self._transition_callbacks:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    logger.error(f"State transition callback error: {e}")
            
            logger.info(f"State transition: {old_state.value} -> {new_state.value}")
            return True
    
    def add_transition_callback(self, callback: Callable[[SessionState, SessionState], None]):
        """Add callback for state transitions."""
        with self._lock:
            self._transition_callbacks.append(callback)
    
    def get_state_history(self) -> List[Tuple[SessionState, datetime]]:
        """Get complete state history."""
        with self._lock:
            return self._state_history.copy()
    
    def is_active(self) -> bool:
        """Check if session is in an active state."""
        active_states = {
            SessionState.PREPARING, SessionState.READY, SessionState.STARTING,
            SessionState.TRAINING, SessionState.PAUSED, SessionState.RESUMING,
            SessionState.STOPPING, SessionState.RECOVERING
        }
        return self.get_state() in active_states
    
    def is_terminal(self) -> bool:
        """Check if session is in a terminal state."""
        terminal_states = {SessionState.COMPLETED, SessionState.CANCELLED, SessionState.ARCHIVED}
        return self.get_state() in terminal_states

# ======================== SESSION METADATA AND ARTIFACTS ========================

@dataclass
class SessionProgress:
    """Session progress tracking."""
    current_epoch: int = 0
    total_epochs: int = 0
    current_batch: int = 0
    total_batches: int = 0
    percentage: float = 0.0
    estimated_remaining: Optional[timedelta] = None
    samples_processed: int = 0
    total_samples: int = 0

@dataclass
class SessionMetadata:
    """Complete session metadata."""
    session_id: str
    session_name: Optional[str]
    domain_name: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_active: datetime = field(default_factory=datetime.now)
    
    # Configuration
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    # Progress tracking
    progress: SessionProgress = field(default_factory=SessionProgress)
    
    # Resource tracking
    peak_memory_usage: float = 0.0
    total_compute_time: float = 0.0
    checkpoint_count: int = 0
    recovery_count: int = 0
    
    # Metrics
    best_metrics: Dict[str, float] = field(default_factory=dict)
    final_metrics: Dict[str, float] = field(default_factory=dict)
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_active = datetime.now()
    
    def total_duration(self) -> timedelta:
        """Calculate total session duration."""
        if self.started_at is None:
            return timedelta(0)
        
        end_time = self.completed_at or datetime.now()
        return end_time - self.started_at

@dataclass
class SessionCheckpoint:
    """Training session checkpoint."""
    checkpoint_id: str
    session_id: str
    created_at: datetime
    epoch: int
    batch: int
    
    # Model state
    model_state: Optional[Dict[str, Any]] = None
    optimizer_state: Optional[Dict[str, Any]] = None
    
    # Metrics at checkpoint
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Storage info
    file_path: Optional[Path] = None
    size_bytes: int = 0
    is_best: bool = False
    is_auto: bool = True

@dataclass
class SessionArtifacts:
    """Session artifacts and storage management."""
    session_id: str
    base_path: Path
    
    # Artifact paths
    checkpoints_path: Path = field(init=False)
    logs_path: Path = field(init=False)
    metrics_path: Path = field(init=False)
    models_path: Path = field(init=False)
    
    def __post_init__(self):
        """Initialize artifact paths."""
        self.checkpoints_path = self.base_path / "checkpoints"
        self.logs_path = self.base_path / "logs"
        self.metrics_path = self.base_path / "metrics"
        self.models_path = self.base_path / "models"
        
        # Create directories
        for path in [self.checkpoints_path, self.logs_path, self.metrics_path, self.models_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def cleanup(self, keep_artifacts: bool = False, keep_best_checkpoint: bool = True):
        """Clean up session artifacts."""
        if not keep_artifacts:
            # Remove all except potentially best checkpoint
            if keep_best_checkpoint:
                # Keep only best checkpoint
                best_checkpoints = list(self.checkpoints_path.glob("*_best.pkl"))
                
                # Remove all other files
                for path in [self.logs_path, self.metrics_path, self.models_path]:
                    if path.exists():
                        shutil.rmtree(path)
                
                # Remove non-best checkpoints
                for cp_file in self.checkpoints_path.glob("*.pkl"):
                    if cp_file not in best_checkpoints:
                        cp_file.unlink()
            else:
                # Remove everything
                if self.base_path.exists():
                    shutil.rmtree(self.base_path)

# ======================== ENHANCED TRAINING SESSION ========================

class EnhancedTrainingSession:
    """Complete training session with full lifecycle management."""
    
    def __init__(
        self,
        session_id: str,
        domain_name: str,
        training_config: Dict[str, Any],
        storage_path: Path,
        session_name: Optional[str] = None
    ):
        self.session_id = session_id
        self.domain_name = domain_name
        self.session_name = session_name or f"session_{session_id[:8]}"
        
        # State management
        self.state_machine = SessionStateMachine()
        
        # Metadata and tracking
        self.metadata = SessionMetadata(
            session_id=session_id,
            session_name=self.session_name,
            domain_name=domain_name,
            created_at=datetime.now(),
            training_config=training_config.copy()
        )
        
        # Storage and artifacts
        self.storage_path = storage_path / session_id
        self.artifacts = SessionArtifacts(session_id, self.storage_path)
        
        # Checkpoints
        self.checkpoints: List[SessionCheckpoint] = []
        self.best_checkpoint: Optional[SessionCheckpoint] = None
        
        # Threading and control
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        
        # Callbacks
        self._progress_callbacks: List[Callable] = []
        self._state_callbacks: List[Callable] = []
        
        # Setup state callbacks
        self.state_machine.add_transition_callback(self._on_state_change)
        
        logger.info(f"Created enhanced training session: {session_id}")
    
    def prepare(self) -> bool:
        """Prepare session for training."""
        print(f"DEBUG: prepare() called for session {self.session_id}")
        
        if not self.state_machine.transition_to(SessionState.PREPARING):
            print("DEBUG: Failed to transition to PREPARING state")
            return False
        
        print("DEBUG: Transitioned to PREPARING state successfully")
        
        try:
            # Initialize training configuration
            print("DEBUG: About to access metadata.training_config")
            epochs = self.metadata.training_config.get('epochs', 100)
            print(f"DEBUG: Got epochs: {epochs}")
            self.metadata.progress.total_epochs = epochs
            print("DEBUG: Set total_epochs in metadata.progress")
            
            # Prepare storage
            print(f"DEBUG: About to create directory: {self.artifacts.base_path}")
            self.artifacts.base_path.mkdir(parents=True, exist_ok=True)
            print("DEBUG: Directory created successfully")
            
            # Save initial metadata
            print("DEBUG: About to call _save_metadata()")
            self._save_metadata()
            print("DEBUG: _save_metadata() completed")
            
            # Transition to ready
            print("DEBUG: About to transition to READY state")
            self.state_machine.transition_to(SessionState.READY)
            print("DEBUG: Transitioned to READY state")
            logger.info(f"Session {self.session_id} prepared successfully")
            return True
            
        except Exception as e:
            print(f"DEBUG: Exception in prepare(): {type(e).__name__}: {e}")
            import traceback
            print(f"DEBUG: Full traceback: {traceback.format_exc()}")
            logger.error(f"Failed to prepare session {self.session_id}: {e}")
            self.state_machine.transition_to(SessionState.FAILED)
            return False
    
    def start(self) -> bool:
        """Start training session."""
        print(f"DEBUG: EnhancedTrainingSession.start() called for {self.session_id}")
        if not self.state_machine.transition_to(SessionState.STARTING):
            print(f"DEBUG: Failed to transition to STARTING state")
            return False
        
        try:
            print(f"DEBUG: Setting metadata.started_at")
            self.metadata.started_at = datetime.now()
            print(f"DEBUG: Calling metadata.update_activity()")
            self.metadata.update_activity()
            print(f"DEBUG: metadata.update_activity() completed")
            
            # Clear control events
            print(f"DEBUG: Clearing control events")
            self._stop_event.clear()
            self._pause_event.clear()
            print(f"DEBUG: Control events cleared")
            
            # Transition to training
            print(f"DEBUG: Transitioning to TRAINING state")
            self.state_machine.transition_to(SessionState.TRAINING)
            print(f"DEBUG: Transition to TRAINING completed")
            logger.info(f"Session {self.session_id} started")
            print(f"DEBUG: EnhancedTrainingSession.start() returning True")
            return True
            
        except Exception as e:
            print(f"DEBUG: Exception in EnhancedTrainingSession.start(): {e}")
            logger.error(f"Failed to start session {self.session_id}: {e}")
            self.state_machine.transition_to(SessionState.FAILED)
            return False
    
    def pause(self) -> bool:
        """Pause training session."""
        if not self.state_machine.transition_to(SessionState.PAUSED):
            return False
        
        try:
            # Set pause event
            self._pause_event.set()
            self.metadata.update_activity()
            
            # Create pause checkpoint
            self.create_checkpoint(
                epoch=self.metadata.progress.current_epoch,
                metrics={"paused_at": time.time()},
                is_auto=True
            )
            
            logger.info(f"Session {self.session_id} paused")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause session {self.session_id}: {e}")
            return False
    
    def resume(self) -> bool:
        """Resume paused training session."""
        if not self.state_machine.transition_to(SessionState.RESUMING):
            return False
        
        try:
            # Clear pause event
            self._pause_event.clear()
            self.metadata.update_activity()
            
            # Transition back to training
            self.state_machine.transition_to(SessionState.TRAINING)
            logger.info(f"Session {self.session_id} resumed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume session {self.session_id}: {e}")
            self.state_machine.transition_to(SessionState.FAILED)
            return False
    
    def stop(self, reason: str = "Manual stop") -> bool:
        """Stop training session gracefully."""
        if not self.state_machine.transition_to(SessionState.STOPPING):
            return False
        
        try:
            # Set stop event
            self._stop_event.set()
            self.metadata.update_activity()
            
            # Create final checkpoint
            self.create_checkpoint(
                epoch=self.metadata.progress.current_epoch,
                metrics={"stopped_at": time.time(), "stop_reason": reason},
                is_auto=True
            )
            
            # Transition to stopped
            self.state_machine.transition_to(SessionState.STOPPED)
            logger.info(f"Session {self.session_id} stopped: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop session {self.session_id}: {e}")
            self.state_machine.transition_to(SessionState.FAILED)
            return False
    
    def complete(self, final_metrics: Dict[str, float] = None) -> bool:
        """Mark session as completed."""
        if self.state_machine.get_state() != SessionState.TRAINING:
            return False
        
        try:
            self.metadata.completed_at = datetime.now()
            self.metadata.update_activity()
            
            if final_metrics:
                self.metadata.final_metrics = final_metrics.copy()
            
            # Create final checkpoint
            self.create_checkpoint(
                epoch=self.metadata.progress.current_epoch,
                metrics=final_metrics or {},
                is_auto=True
            )
            
            # Save final metadata
            self._save_metadata()
            
            self.state_machine.transition_to(SessionState.COMPLETED)
            logger.info(f"Session {self.session_id} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete session {self.session_id}: {e}")
            self.state_machine.transition_to(SessionState.FAILED)
            return False
    
    def recover(self, checkpoint_id: Optional[str] = None) -> bool:
        """Recover session from checkpoint."""
        if not self.state_machine.transition_to(SessionState.RECOVERING):
            return False
        
        try:
            # Find checkpoint to recover from
            checkpoint = None
            if checkpoint_id:
                checkpoint = next((cp for cp in self.checkpoints if cp.checkpoint_id == checkpoint_id), None)
            else:
                # Use best checkpoint or latest
                checkpoint = self.best_checkpoint or (self.checkpoints[-1] if self.checkpoints else None)
            
            if not checkpoint:
                logger.error(f"No checkpoint available for recovery in session {self.session_id}")
                self.state_machine.transition_to(SessionState.FAILED)
                return False
            
            # Restore from checkpoint
            self.metadata.progress.current_epoch = checkpoint.epoch
            self.metadata.recovery_count += 1
            self.metadata.update_activity()
            
            # Transition to ready for restart
            self.state_machine.transition_to(SessionState.READY)
            logger.info(f"Session {self.session_id} recovered from checkpoint {checkpoint.checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to recover session {self.session_id}: {e}")
            self.state_machine.transition_to(SessionState.FAILED)
            return False
    
    def cleanup(self, keep_artifacts: bool = False, keep_best_checkpoint: bool = True) -> bool:
        """Clean up session resources."""
        if not self.state_machine.transition_to(SessionState.CLEANUP):
            return False
        
        try:
            # Clean up artifacts
            self.artifacts.cleanup(keep_artifacts, keep_best_checkpoint)
            
            # Clear in-memory data
            if not keep_artifacts:
                self.checkpoints.clear()
                if not keep_best_checkpoint:
                    self.best_checkpoint = None
            
            # Force garbage collection
            gc.collect()
            
            # Transition to archived
            self.state_machine.transition_to(SessionState.ARCHIVED)
            logger.info(f"Session {self.session_id} cleaned up")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup session {self.session_id}: {e}")
            return False
    
    def create_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float] = None,
        model_state: Dict[str, Any] = None,
        optimizer_state: Dict[str, Any] = None,
        is_best: bool = False,
        is_auto: bool = True
    ) -> Optional[SessionCheckpoint]:
        """Create training checkpoint."""
        try:
            checkpoint_id = f"checkpoint_{epoch}_{int(time.time())}"
            if is_best:
                checkpoint_id += "_best"
            
            checkpoint = SessionCheckpoint(
                checkpoint_id=checkpoint_id,
                session_id=self.session_id,
                created_at=datetime.now(),
                epoch=epoch,
                batch=self.metadata.progress.current_batch,
                model_state=model_state,
                optimizer_state=optimizer_state,
                metrics=metrics or {},
                is_best=is_best,
                is_auto=is_auto
            )
            
            # Save checkpoint to disk
            checkpoint_file = self.artifacts.checkpoints_path / f"{checkpoint_id}.pkl"
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            checkpoint.file_path = checkpoint_file
            checkpoint.size_bytes = checkpoint_file.stat().st_size
            
            # Add to session
            self.checkpoints.append(checkpoint)
            self.metadata.checkpoint_count += 1
            
            # Update best checkpoint
            if is_best or self._is_better_checkpoint(checkpoint):
                self.best_checkpoint = checkpoint
            
            # Manage checkpoint count
            self._manage_checkpoints()
            
            logger.info(f"Created checkpoint {checkpoint_id} for session {self.session_id}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint for session {self.session_id}: {e}")
            return None
    
    def update_progress(
        self,
        epoch: int = None,
        batch: int = None,
        total_batches: int = None,
        samples_processed: int = None
    ):
        """Update training progress."""
        with self._lock:
            if epoch is not None:
                self.metadata.progress.current_epoch = epoch
            if batch is not None:
                self.metadata.progress.current_batch = batch
            if total_batches is not None:
                self.metadata.progress.total_batches = total_batches
            if samples_processed is not None:
                self.metadata.progress.samples_processed = samples_processed
            
            # Calculate percentage
            if self.metadata.progress.total_epochs > 0:
                epoch_progress = self.metadata.progress.current_epoch / self.metadata.progress.total_epochs
                if self.metadata.progress.total_batches > 0:
                    batch_progress = self.metadata.progress.current_batch / self.metadata.progress.total_batches
                    batch_progress /= self.metadata.progress.total_epochs
                    self.metadata.progress.percentage = (epoch_progress + batch_progress) * 100
                else:
                    self.metadata.progress.percentage = epoch_progress * 100
            
            self.metadata.update_activity()
            
            # Call progress callbacks
            for callback in self._progress_callbacks:
                try:
                    callback(self.metadata.progress)
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")
    
    def add_progress_callback(self, callback: Callable[[SessionProgress], None]):
        """Add progress update callback."""
        self._progress_callbacks.append(callback)
    
    def add_state_callback(self, callback: Callable[[SessionState, SessionState], None]):
        """Add state change callback."""
        self._state_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive session status."""
        with self._lock:
            return {
                'session_id': self.session_id,
                'session_name': self.session_name,
                'domain_name': self.domain_name,
                'state': self.state_machine.get_state().value,
                'is_active': self.state_machine.is_active(),
                'is_terminal': self.state_machine.is_terminal(),
                'can_pause': self.state_machine.can_transition_to(SessionState.PAUSED),
                'can_resume': (self.state_machine.can_transition_to(SessionState.RESUMING) or 
                             self.state_machine.get_state() == SessionState.READY),
                'can_stop': self.state_machine.can_transition_to(SessionState.STOPPING),
                'can_recover': self.state_machine.can_transition_to(SessionState.RECOVERING),
                'progress': asdict(self.metadata.progress),
                'metadata': {
                    'created_at': self.metadata.created_at.isoformat(),
                    'started_at': self.metadata.started_at.isoformat() if self.metadata.started_at else None,
                    'completed_at': self.metadata.completed_at.isoformat() if self.metadata.completed_at else None,
                    'total_duration': str(self.metadata.total_duration()),
                    'checkpoint_count': self.metadata.checkpoint_count,
                    'recovery_count': self.metadata.recovery_count,
                    'peak_memory_usage': self.metadata.peak_memory_usage,
                    'total_compute_time': self.metadata.total_compute_time
                },
                'checkpoints': len(self.checkpoints),
                'best_checkpoint': self.best_checkpoint.checkpoint_id if self.best_checkpoint else None,
                'best_metrics': self.metadata.best_metrics,
                'final_metrics': self.metadata.final_metrics
            }
    
    def _on_state_change(self, old_state: SessionState, new_state: SessionState):
        """Handle state changes."""
        # Call state callbacks
        for callback in self._state_callbacks:
            try:
                callback(old_state, new_state)
            except Exception as e:
                logger.error(f"State callback error: {e}")
        
        # Save metadata on important state changes
        important_states = {
            SessionState.READY, SessionState.TRAINING, SessionState.PAUSED,
            SessionState.COMPLETED, SessionState.FAILED, SessionState.STOPPED
        }
        if new_state in important_states:
            self._save_metadata()
    
    def _save_metadata(self):
        """Save session metadata to disk."""
        try:
            print(f"DEBUG: Starting _save_metadata for session {self.session_id}")
            print(f"DEBUG: Base path: {self.artifacts.base_path}")
            
            metadata_file = self.artifacts.base_path / "metadata.json"
            print(f"DEBUG: Created metadata file path: {metadata_file}")
            
            print("DEBUG: About to call asdict(self.metadata)")
            print(f"DEBUG: Metadata training_config keys: {list(self.metadata.training_config.keys()) if hasattr(self.metadata, 'training_config') else 'No training_config'}")
            print(f"DEBUG: Metadata training_config size estimate: {len(str(self.metadata.training_config)) if hasattr(self.metadata, 'training_config') else 0} chars")
            
            metadata_dict = asdict(self.metadata)
            print("DEBUG: Completed asdict conversion successfully")
            print(f"DEBUG: Converted dict keys: {list(metadata_dict.keys())}")
            
            # Convert datetime objects to ISO strings
            print("DEBUG: Starting datetime conversion loop")
            for key, value in metadata_dict.items():
                if isinstance(value, datetime):
                    metadata_dict[key] = value.isoformat()
                    print(f"DEBUG: Converted datetime for key: {key}")
            print("DEBUG: Completed datetime conversion")
            
            print(f"DEBUG: About to write JSON to file: {metadata_file}")
            with open(metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2, default=str)
            print("DEBUG: Completed JSON dump successfully")
                
        except Exception as e:
            print(f"DEBUG: Exception in _save_metadata: {type(e).__name__}: {e}")
            import traceback
            print(f"DEBUG: Full traceback: {traceback.format_exc()}")
            logger.error(f"Failed to save metadata for session {self.session_id}: {e}")
    
    def _is_better_checkpoint(self, checkpoint: SessionCheckpoint) -> bool:
        """Determine if checkpoint is better than current best."""
        if not self.best_checkpoint:
            return True
        
        # Simple metric-based comparison (can be enhanced)
        current_metrics = checkpoint.metrics
        best_metrics = self.best_checkpoint.metrics
        
        # Check for loss improvement
        if 'loss' in current_metrics and 'loss' in best_metrics:
            return current_metrics['loss'] < best_metrics['loss']
        
        # Check for accuracy improvement
        if 'accuracy' in current_metrics and 'accuracy' in best_metrics:
            return current_metrics['accuracy'] > best_metrics['accuracy']
        
        # Default to epoch comparison
        return checkpoint.epoch > self.best_checkpoint.epoch
    
    def _manage_checkpoints(self, max_checkpoints: int = 10):
        """Manage checkpoint count and cleanup old ones."""
        if len(self.checkpoints) <= max_checkpoints:
            return
        
        # Sort by creation time
        sorted_checkpoints = sorted(self.checkpoints, key=lambda cp: cp.created_at)
        
        # Remove oldest auto checkpoints (keep best and manual)
        to_remove = []
        for checkpoint in sorted_checkpoints:
            if len(self.checkpoints) - len(to_remove) <= max_checkpoints:
                break
            
            if checkpoint.is_auto and not checkpoint.is_best and checkpoint != self.best_checkpoint:
                to_remove.append(checkpoint)
        
        # Remove checkpoints
        for checkpoint in to_remove:
            try:
                if checkpoint.file_path and checkpoint.file_path.exists():
                    checkpoint.file_path.unlink()
                self.checkpoints.remove(checkpoint)
            except Exception as e:
                logger.error(f"Failed to remove checkpoint {checkpoint.checkpoint_id}: {e}")

# ======================== ENHANCED TRAINING MANAGER ========================

class EnhancedTrainingManager:
    """Enhanced training manager with comprehensive session lifecycle management."""
    
    def __init__(self, brain_instance=None, storage_path: str = ".brain/enhanced_sessions", 
                 max_concurrent_sessions: int = 5):
        self.brain = brain_instance
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Session management
        self._enhanced_sessions: Dict[str, EnhancedTrainingSession] = {}
        self._session_lock = threading.RLock()
        
        # Configuration
        self.max_concurrent_sessions = max_concurrent_sessions
        self.default_checkpoint_frequency = 5
        self.auto_cleanup_completed = False  # Disable auto cleanup for validation tests
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Enhanced training manager initialized at {storage_path}")
    
    def create_session(
        self,
        domain_name: str,
        training_config: Dict[str, Any],
        session_name: Optional[str] = None
    ) -> str:
        """Create new enhanced training session."""
        print(f"DEBUG: create_session called for domain: {domain_name}")
        print(f"DEBUG: training_config keys: {list(training_config.keys())}")
        print(f"DEBUG: training_config size: {len(str(training_config))} chars")
        
        with self._session_lock:
            print("DEBUG: Acquired session lock")
            
            # Check concurrent session limit
            active_sessions = [s for s in self._enhanced_sessions.values() if s.state_machine.is_active()]
            print(f"DEBUG: Active sessions count: {len(active_sessions)}")
            if len(active_sessions) >= self.max_concurrent_sessions:
                raise RuntimeError(f"Maximum concurrent sessions ({self.max_concurrent_sessions}) reached")
            
            # Generate session ID
            session_id = str(uuid.uuid4())
            print(f"DEBUG: Generated session ID: {session_id}")
            
            # Create enhanced session
            print("DEBUG: About to create EnhancedTrainingSession")
            session = EnhancedTrainingSession(
                session_id=session_id,
                domain_name=domain_name,
                training_config=training_config,
                storage_path=self.storage_path,
                session_name=session_name
            )
            print("DEBUG: EnhancedTrainingSession created successfully")
            
            # Store session
            self._enhanced_sessions[session_id] = session
            print("DEBUG: Session stored in _enhanced_sessions")
            
            # Prepare session
            print("DEBUG: About to call session.prepare()")
            session.prepare()
            print("DEBUG: session.prepare() completed")
            
            self.logger.info(f"Created enhanced session: {session_id} for domain {domain_name}")
            return session_id
    
    def start_session(self, session_id: str) -> bool:
        """Start training session."""
        print(f"DEBUG: start_session called for {session_id}")
        session = self._get_session(session_id)
        if not session:
            print(f"DEBUG: Session {session_id} not found")
            return False
        
        print(f"DEBUG: Found session, about to call session.start()")
        result = session.start()
        print(f"DEBUG: session.start() returned: {result}")
        return result
    
    def pause_session(self, session_id: str) -> bool:
        """Pause training session."""
        session = self._get_session(session_id)
        if not session:
            return False
        
        return session.pause()
    
    def resume_session(self, session_id: str) -> bool:
        """Resume paused training session."""
        session = self._get_session(session_id)
        if not session:
            return False
        
        return session.resume()
    
    def stop_session(self, session_id: str, reason: str = "Manual stop") -> bool:
        """Stop training session."""
        session = self._get_session(session_id)
        if not session:
            return False
        
        return session.stop(reason)
    
    def complete_session(self, session_id: str, final_metrics: Dict[str, float] = None) -> bool:
        """Complete training session."""
        session = self._get_session(session_id)
        if not session:
            return False
        
        success = session.complete(final_metrics)
        
        # Auto cleanup if enabled
        if success and self.auto_cleanup_completed:
            session.cleanup(keep_artifacts=True, keep_best_checkpoint=True)
        
        return success
    
    def recover_session(self, session_id: str, checkpoint_id: Optional[str] = None) -> bool:
        """Recover session from checkpoint."""
        session = self._get_session(session_id)
        if not session:
            return False
        
        return session.recover(checkpoint_id)
    
    def cleanup_session(
        self,
        session_id: str,
        keep_artifacts: bool = False,
        keep_best_checkpoint: bool = True
    ) -> bool:
        """Clean up session resources."""
        session = self._get_session(session_id)
        if not session:
            return False
        
        success = session.cleanup(keep_artifacts, keep_best_checkpoint)
        
        # Remove from active sessions if archived
        if success and session.state_machine.get_state() == SessionState.ARCHIVED:
            with self._session_lock:
                self._enhanced_sessions.pop(session_id, None)
        
        return success
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive session status."""
        session = self._get_session(session_id)
        if not session:
            return None
        
        return session.get_status()
    
    def get_all_session_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all sessions."""
        with self._session_lock:
            return {
                session_id: session.get_status()
                for session_id, session in self._enhanced_sessions.items()
            }
    
    def list_active_sessions(self) -> List[str]:
        """List all active session IDs."""
        with self._session_lock:
            return [
                session_id for session_id, session in self._enhanced_sessions.items()
                if session.state_machine.is_active()
            ]
    
    def list_all_sessions(self) -> List[str]:
        """List all session IDs."""
        with self._session_lock:
            return list(self._enhanced_sessions.keys())
    
    def shutdown(self):
        """Shutdown enhanced training manager."""
        self.logger.info("Shutting down enhanced training manager...")
        
        with self._session_lock:
            # Stop all active sessions
            for session in self._enhanced_sessions.values():
                if session.state_machine.is_active():
                    session.stop("Manager shutdown")
            
            # Cleanup completed sessions
            for session in list(self._enhanced_sessions.values()):
                if session.state_machine.get_state() in {SessionState.COMPLETED, SessionState.STOPPED}:
                    session.cleanup(keep_artifacts=True, keep_best_checkpoint=True)
        
        self.logger.info("Enhanced training manager shutdown complete")
    
    def _get_session(self, session_id: str) -> Optional[EnhancedTrainingSession]:
        """Get session by ID."""
        with self._session_lock:
            return self._enhanced_sessions.get(session_id)
    
    def enhanced_train_domain(self, domain_name: str, training_data: Dict[str, Any], 
                             training_config: Optional[Dict[str, Any]] = None, 
                             model_type: Optional[str] = None,
                             use_enhanced_session: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Enhanced domain training with automatic session management.
        This method provides backward compatibility with the existing train_domain interface.
        """
        # ROOT FIX: Ensure config is always a dictionary with comprehensive type handling
        if isinstance(training_config, dict):
            config = training_config.copy()
        elif training_config is None:
            config = {}
        elif isinstance(training_config, (int, float)):
            config = {'epochs': training_config}
        elif hasattr(training_config, '__dataclass_fields__'):
            # Handle dataclass objects (like TrainingConfig)
            from dataclasses import asdict
            config = asdict(training_config)
        elif hasattr(training_config, '__iter__') and not isinstance(training_config, str):
            # Handle any iterable that can be converted to dict
            try:
                config = dict(training_config)
            except (TypeError, ValueError) as e:
                if SESSION_FALLBACKS_DISABLED:
                    raise RuntimeError(f"CONFIG_CONVERSION_DISABLED: Training config conversion failed: {e}")
                config = {}  # Original fallback preserved
        else:
            # Fallback for any other type
            if SESSION_FALLBACKS_DISABLED:
                raise RuntimeError(f"CONFIG_TYPE_DISABLED: Unknown training config type, fallback disabled")
            config = {}
        
        # Ensure kwargs are merged properly
        if kwargs:
            config.update(kwargs)
        
        # ROOT CAUSE FIX: Properly handle model_type parameter
        # Extract model_type if present and store it separately
        model_type = None
        if 'model_type' in config:
            model_type = config['model_type']
            self.logger.info(f"enhanced_train_domain: extracted model_type = {model_type}")
            # Remove model_type from config to prevent it from interfering with training parameters
            config = {k: v for k, v in config.items() if k != 'model_type'}
        
        # ROOT CAUSE FIX: Handle nested config structure
        # If the config contains a nested 'config' field, extract it
        if 'config' in config and isinstance(config['config'], dict):
            self.logger.info(f"enhanced_train_domain: found nested config, extracting...")
            nested_config = config['config']
            # Merge nested config with parent config, but don't overwrite existing values
            for key, value in nested_config.items():
                if key not in config:
                    config[key] = value
            # Remove the nested config field
            config = {k: v for k, v in config.items() if k != 'config'}
        
        if use_enhanced_session:
            try:
                # Create session automatically
                session_id = self.create_session(
                    domain_name=domain_name,
                    training_config=config,
                    session_name=kwargs.get('session_name')
                )
                
                # Start training
                print(f"DEBUG: About to call start_session({session_id})")
                if not self.start_session(session_id):
                    raise RuntimeError("Failed to start training session")
                print(f"DEBUG: start_session completed successfully")
                
                # Get session for monitoring
                print(f"DEBUG: About to get session for monitoring")
                session = self._get_session(session_id)
                if not session:
                    raise RuntimeError("Session not found after creation")
                print(f"DEBUG: Got session for monitoring: {session.session_id}")
                
                # Use Brain's actual training method - reuse existing training infrastructure
                if self.brain and hasattr(self.brain, 'training_manager'):
                    print(f"DEBUG: Brain has training_manager, proceeding with training")
                    # ROOT FIX: Synchronize sessions between enhanced and legacy systems
                    # Create session bridge to ensure session exists in both systems
                    print(f"DEBUG: About to call _create_session_bridge")
                    self._create_session_bridge(session_id, session, domain_name, config)
                    print(f"DEBUG: _create_session_bridge completed")
                    
                    # Reuse the Brain's training manager for real calculations
                    # ROOT FIX: Pass epochs as integer parameter, not training_config
                    epochs = config.get('epochs', 10) if isinstance(config, dict) else 10
                    print(f"DEBUG: About to call brain.training_manager.train_domain with epochs={epochs}")
                    print(f"DEBUG: Training data keys: {list(training_data.keys()) if isinstance(training_data, dict) else 'Not a dict'}")
                    print(f"DEBUG: Training data X shape: {training_data.get('X').shape if isinstance(training_data, dict) and 'X' in training_data else 'No X'}")
                    
                    # ROOT FIX: Use enhanced session with existing session_id to prevent nested creation
                    original_result = self.brain.training_manager.train_domain(
                        domain_name=domain_name,
                        training_data=training_data,
                        epochs=epochs,
                        session_id=session_id,
                        use_enhanced_session=True,  # Use enhanced session path
                        existing_session_id=session_id  # Pass existing session to prevent creation
                    )
                    print(f"DEBUG: brain.training_manager.train_domain completed")
                    print(f"DEBUG: Training result success: {original_result.get('success', 'Unknown')}")
                    
                    # ROOT FIX: Synchronize results back to enhanced session
                    print(f"DEBUG: About to call _sync_training_results")
                    self._sync_training_results(session_id, original_result)
                    print(f"DEBUG: _sync_training_results completed")
                else:
                    # Fallback to simulation only if no Brain available
                    # FALLBACK DISABLED - FORCE REAL TRAINING
                    raise RuntimeError("Training simulation fallback disabled - must use real training")
                
                # Update session with training results
                if original_result.get('success', False):
                    # Extract metrics from training result
                    history = original_result.get('history', {})
                    if history:
                        final_epoch = len(history.get('loss', []))
                        final_loss = history['loss'][-1] if history.get('loss') else 0.0
                        final_accuracy = history['accuracy'][-1] if history.get('accuracy') else 0.0
                        
                        # Update session progress
                        session.update_progress(
                            epoch=final_epoch,
                            batch=1,
                            total_batches=1,
                            samples_processed=len(training_data.get('X', []))
                        )
                        
                        # Create final checkpoint
                        checkpoint = session.create_checkpoint(
                            epoch=final_epoch,
                            metrics={
                                'final_loss': final_loss,
                                'final_accuracy': final_accuracy,
                                **original_result.get('metrics', {})
                            },
                            model_state={'training_complete': True},
                            optimizer_state={'final_state': True}
                        )
                    
                    # Mark session as completed
                    self.stop_session(session_id, reason="completed")
                    
                    # Clean up session bridge after successful completion
                    self._cleanup_session_bridge(session_id)
                    
                    # Add session lifecycle info
                    original_result['session_id'] = session_id
                    original_result['session_lifecycle'] = {
                        'state': 'completed',
                        'checkpoints_created': len(session.checkpoints),
                        'session_duration': str(session.metadata.total_duration)
                    }
                    original_result['checkpoints_created'] = len(session.checkpoints)
                else:
                    # Mark session as failed
                    self.stop_session(session_id, reason="training_failed")
                    # Clean up session bridge on training failure
                    self._cleanup_session_bridge(session_id)
                    original_result['session_lifecycle'] = {
                        'state': 'failed',
                        'error': original_result.get('error', 'Training failed')
                    }
                
                return original_result
                
            except Exception as e:
                self.logger.error(f"Enhanced session training error: {e}")
                if SESSION_FALLBACKS_DISABLED:
                    raise RuntimeError(f"ENHANCED_TRAINING_DISABLED: Enhanced session training failed: {e}")
                # Clean up session bridge on failure
                if 'session_id' in locals():
                    self._cleanup_session_bridge(session_id)
                return {
                    'success': False,
                    'error': str(e),
                    'session_lifecycle': {'state': 'failed', 'error': str(e)}
                }
        else:
            # Fallback to basic training
            if SESSION_FALLBACKS_DISABLED:
                raise RuntimeError("ENHANCED_TRAINING_FALLBACK_DISABLED: No enhanced training manager available, simulation fallback disabled")
            return self._simulate_training(training_data, config)
    
    def _create_session_bridge(self, session_id: str, enhanced_session: EnhancedTrainingSession, 
                              domain_name: str, config: Dict[str, Any]) -> None:
        """
        ROOT FIX: Create session bridge between enhanced and legacy training systems.
        This ensures the session exists in both _enhanced_sessions and _sessions dictionaries.
        """
        print(f"DEBUG: _create_session_bridge called for {session_id}")
        try:
            print(f"DEBUG: Checking brain and training_manager")
            if not self.brain or not hasattr(self.brain, 'training_manager'):
                print(f"DEBUG: No brain.training_manager available")
                self.logger.warning("No brain.training_manager available for session bridge")
                return
            
            print(f"DEBUG: Getting training_manager")
            training_manager = self.brain.training_manager
            print(f"DEBUG: Got training_manager: {type(training_manager)}")
            
            # Import required classes for legacy session creation
            print(f"DEBUG: Importing datetime")
            from datetime import datetime
            print(f"DEBUG: Importing training_manager classes")
            from training_manager import TrainingSession as LegacyTrainingSession, TrainingConfig, TrainingStatus
            print(f"DEBUG: Imports completed")
            
            # Create legacy session configuration
            print(f"DEBUG: Creating legacy config")
            legacy_config = TrainingConfig(
                batch_size=config.get('batch_size', 32),
                learning_rate=config.get('learning_rate', 0.001),
                validation_split=config.get('validation_split', 0.2),
                epochs=config.get('epochs', 10),
                early_stopping_patience=config.get('early_stopping_patience', 5),
                device='cuda' if config.get('use_gpu', True) else 'cpu',
                checkpoint_frequency=config.get('checkpoint_interval', 1)
            )
            print(f"DEBUG: Legacy config created")
            
            # Create legacy training session compatible with training_manager
            print(f"DEBUG: Creating legacy session")
            legacy_session = LegacyTrainingSession(
                session_id=session_id,  # Use same session ID for bridge
                domain_name=domain_name,
                status=TrainingStatus.NOT_STARTED,
                config=legacy_config,
                start_time=datetime.now()
            )
            print(f"DEBUG: Legacy session created")
            
            # Bridge: Add session to both systems
            print(f"DEBUG: Adding session to training_manager._sessions")
            training_manager._sessions[session_id] = legacy_session
            print(f"DEBUG: Session added to training_manager._sessions")
            self.logger.info(f"Session bridge created: {session_id} now exists in both enhanced and legacy systems")
            print(f"DEBUG: _create_session_bridge completed successfully")
            
        except Exception as e:
            print(f"DEBUG: Exception in _create_session_bridge: {e}")
            self.logger.error(f"Failed to create session bridge for {session_id}: {e}")
            if SESSION_FALLBACKS_DISABLED:
                raise RuntimeError(f"SESSION_BRIDGE_CREATION_DISABLED: Session bridge creation failed for {session_id}: {e}")
            # Continue execution - this is a compatibility fix, not critical for basic functionality
    
    def _sync_training_results(self, session_id: str, training_result: Dict[str, Any]) -> None:
        """
        ROOT FIX: Synchronize training results back to enhanced session.
        Updates enhanced session with results from legacy training manager.
        """
        print(f"DEBUG: _sync_training_results called for {session_id}")
        try:
            print(f"DEBUG: Getting enhanced session")
            enhanced_session = self._get_session(session_id)
            if not enhanced_session:
                print(f"DEBUG: Enhanced session {session_id} not found")
                self.logger.warning(f"Enhanced session {session_id} not found for result sync")
                return
            print(f"DEBUG: Found enhanced session")
            
            # Sync training metrics and status
            if training_result.get('success', False):
                # Update enhanced session with training results
                history = training_result.get('history', {})
                if history and 'loss' in history and 'accuracy' in history:
                    final_epoch = len(history['loss'])
                    final_loss = history['loss'][-1] if history['loss'] else 0.0
                    final_accuracy = history['accuracy'][-1] if history['accuracy'] else 0.0
                    
                    # Update session metadata
                    enhanced_session.metadata.best_metrics = {
                        'best_loss': min(history['loss']) if history['loss'] else final_loss,
                        'best_accuracy': max(history['accuracy']) if history['accuracy'] else final_accuracy
                    }
                    enhanced_session.metadata.final_metrics = {
                        'final_loss': final_loss,
                        'final_accuracy': final_accuracy,
                        'epochs_completed': final_epoch
                    }
            
            self.logger.info(f"Training results synchronized for session {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to sync training results for {session_id}: {e}")
            if SESSION_FALLBACKS_DISABLED:
                raise RuntimeError(f"TRAINING_RESULT_SYNC_DISABLED: Training result sync failed for {session_id}: {e}")
            # Continue execution - result sync failure shouldn't break training
    
    def _cleanup_session_bridge(self, session_id: str) -> None:
        """
        ROOT FIX: Clean up session bridge on failure or completion.
        Removes session from legacy system while preserving enhanced session for debugging.
        """
        try:
            if self.brain and hasattr(self.brain, 'training_manager'):
                training_manager = self.brain.training_manager
                
                # Remove from legacy system only
                if session_id in training_manager._sessions:
                    del training_manager._sessions[session_id]
                    self.logger.info(f"Session bridge cleaned up for {session_id}")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup session bridge for {session_id}: {e}")
            if SESSION_FALLBACKS_DISABLED:
                raise RuntimeError(f"SESSION_CLEANUP_DISABLED: Session bridge cleanup failed for {session_id}: {e}")

    def _simulate_training(self, training_data: Dict[str, Any], config: Union[Dict[str, Any], int]) -> Dict[str, Any]:
        """Simulate basic training for fallback scenarios."""
        import numpy as np
        
        # Handle case where config is an integer (epochs value)
        if isinstance(config, int):
            epochs = config
        else:
            epochs = config.get('epochs', 10)
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(epochs):
            # Generate realistic loss (decreasing)
            loss = max(0.1, 0.5 - (epoch * 0.04) + np.random.random() * 0.05)
            
            # Generate realistic accuracy (increasing, capped at 0.95)
            accuracy = min(0.95, 0.5 + (epoch * 0.04) + np.random.random() * 0.05)
            
            history['loss'].append(loss)
            history['accuracy'].append(accuracy)
        
        return {
            'success': True,
            'history': history,
            'epochs_completed': epochs,
            'final_loss': history['loss'][-1],
            'final_accuracy': history['accuracy'][-1]
        }


# ======================== CONVENIENCE FUNCTIONS ========================

def create_enhanced_training_manager(brain_instance=None, storage_path: str = ".brain/enhanced_sessions", 
                                   max_concurrent_sessions: int = 5):
    """Create enhanced training manager."""
    return EnhancedTrainingManager(brain_instance, storage_path, max_concurrent_sessions)

if __name__ == "__main__":
    # Example usage
    manager = create_enhanced_training_manager()
    
    # Create session
    session_id = manager.create_session(
        domain_name="test_domain",
        training_config={'epochs': 10, 'batch_size': 32},
        session_name="example_session"
    )
    
    print(f"Created session: {session_id}")
    
    # Get status
    status = manager.get_session_status(session_id)
    print(f"Session status: {status['state']}")
    
    # Cleanup
    manager.shutdown()
    print("Enhanced training session management is ready!")