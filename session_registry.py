"""
Unified Session Registry for Saraphis Training System

This module provides a centralized session management system that replaces
the dual session storage approach (legacy _sessions + enhanced _enhanced_sessions)
with a single source of truth for all training sessions.

Key Features:
- Thread-safe singleton registry
- Unified session objects combining legacy and enhanced features
- Standardized configuration schema
- Atomic session lifecycle operations
- Comprehensive resource management
"""

import threading
import uuid
import json
import time
from typing import Dict, Optional, Any, List, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logger = logging.getLogger(__name__)

class SessionState(Enum):
    """Unified session state combining legacy and enhanced state management"""
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
    CLEANUP = "cleanup"
    ARCHIVED = "archived"

@dataclass
class UnifiedTrainingConfig:
    """
    Unified training configuration that reconciles differences between
    legacy and enhanced systems.
    
    Resolves configuration conflicts:
    - Enhanced default: 100 epochs, Legacy default: 10 epochs -> Use 100
    - Enhanced default: batch_size=128, Legacy default: batch_size=32 -> Use 128
    """
    # Core training parameters
    epochs: int = 100  # Reconciled: enhanced default wins
    batch_size: int = 128  # Reconciled: enhanced default wins
    learning_rate: float = 0.001
    validation_split: float = 0.2
    
    # Optimizer configuration
    optimizer: str = "adam"
    optimizer_params: Dict[str, Any] = None
    
    # Scheduler configuration
    scheduler: str = "reduce_on_plateau"
    scheduler_params: Dict[str, Any] = None
    
    # Regularization
    dropout_rate: float = 0.2
    l1_regularization: float = 0.0
    l2_regularization: float = 0.01
    gradient_clip_value: float = 1.0
    
    # GAC System (from enhanced)
    use_gac_system: bool = True
    gac_mode: str = "adaptive"
    gac_components: List[str] = None
    gac_threshold_adaptation: bool = True
    gac_noise_enabled: bool = False
    
    # Device and performance
    device: str = "cuda"
    enable_resource_monitoring: bool = True
    resource_limits: Dict[str, Any] = None
    
    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 10  # Reconciled: enhanced uses 10, legacy uses 5
    early_stopping_min_delta: float = 0.0001
    
    # Training optimization
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    warmup_epochs: int = 0
    warmup_learning_rate: float = 1e-5
    label_smoothing: float = 0.0
    auto_augment: bool = False
    
    # Monitoring and logging
    checkpoint_frequency: int = 10
    max_checkpoints: int = 5
    log_frequency: int = 10
    metric_names: List[str] = None
    
    # Data processing
    normalize_features: bool = True
    handle_missing_values: bool = True
    missing_value_strategy: str = "median"
    encode_categorical: bool = True
    categorical_encoding: str = "auto"
    
    # Session management
    max_memory_mb: int = 1024
    max_training_time_hours: float = 24.0
    num_workers: int = 0
    save_training_history: bool = True
    enable_recovery: bool = True
    
    # Proof system configuration (from enhanced)
    proof_system_config: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values for mutable fields"""
        if self.optimizer_params is None:
            self.optimizer_params = {"weight_decay": 1e-5}
        if self.scheduler_params is None:
            self.scheduler_params = {"patience": 5, "factor": 0.5}
        if self.gac_components is None:
            self.gac_components = ["clipping", "monitoring"]
        if self.resource_limits is None:
            self.resource_limits = {"max_memory_gb": 8.0, "max_cpu_percent": 80.0}
        if self.metric_names is None:
            self.metric_names = ["loss", "accuracy"]
        if self.proof_system_config is None:
            self.proof_system_config = {"enable_proof_system": True}

class SessionArtifacts:
    """Manages session artifacts including checkpoints, logs, and metadata"""
    
    def __init__(self, session_id: str, storage_path: Path):
        self.session_id = session_id
        self.storage_path = storage_path
        self.artifacts_path = storage_path / "artifacts"
        self.checkpoints_path = storage_path / "checkpoints"
        self.logs_path = storage_path / "logs"
        
        # Create directories
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)
        
        # Artifact tracking
        self.checkpoints: List[str] = []
        self.logs: List[str] = []
        self.metadata_files: List[str] = []
        
    def create_checkpoint(self, checkpoint_name: str, data: Any) -> str:
        """Create a checkpoint and return its path"""
        checkpoint_file = self.checkpoints_path / f"{checkpoint_name}.json"
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.checkpoints.append(checkpoint_name)
            logger.info(f"Created checkpoint {checkpoint_name} for session {self.session_id}")
            return str(checkpoint_file)
        except Exception as e:
            logger.error(f"Failed to create checkpoint {checkpoint_name}: {e}")
            raise
            
    def cleanup(self):
        """Clean up all session artifacts"""
        try:
            import shutil
            if self.storage_path.exists():
                shutil.rmtree(self.storage_path)
            logger.info(f"Cleaned up artifacts for session {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup artifacts for session {self.session_id}: {e}")

class UnifiedSession:
    """
    Unified session object that combines features from both TrainingSession (legacy)
    and EnhancedTrainingSession systems.
    """
    
    def __init__(self, session_id: str, domain_name: str, config: UnifiedTrainingConfig, 
                 storage_path: Path):
        # Core session attributes
        self.session_id = session_id
        self.domain_name = domain_name
        self.config = config
        self.storage_path = storage_path
        
        # State management (from enhanced system)
        self.state = SessionState.CREATED
        self.state_lock = threading.RLock()
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        
        # Artifacts management (from enhanced system)
        self.artifacts = SessionArtifacts(session_id, storage_path)
        
        # Resource tracking (from legacy system)
        self.resources: Dict[str, Any] = {
            "gpu_allocated": False,
            "memory_allocated_mb": 0,
            "cpu_threads": 0,
            "storage_used_mb": 0
        }
        
        # Training progress tracking
        self.training_data: Optional[Dict[str, Any]] = None
        self.current_epoch: int = 0
        self.best_performance: Optional[float] = None
        self.training_metrics: List[Dict[str, Any]] = []
        self.error_info: Optional[Dict[str, Any]] = None
        
        # Integration flags
        self.enhanced_features_enabled = True
        self.legacy_compatibility_enabled = True
        
        logger.info(f"Created unified session {session_id} for domain {domain_name}")
    
    def transition_to(self, new_state: SessionState) -> bool:
        """Thread-safe state transition with validation"""
        with self.state_lock:
            old_state = self.state
            
            # Validate state transition
            if not self._is_valid_transition(old_state, new_state):
                logger.warning(f"Invalid state transition: {old_state} -> {new_state}")
                return False
            
            self.state = new_state
            
            # Update timestamps
            if new_state == SessionState.TRAINING and not self.started_at:
                self.started_at = time.time()
            elif new_state in [SessionState.COMPLETED, SessionState.FAILED]:
                self.completed_at = time.time()
            
            logger.info(f"Session {self.session_id} transitioned: {old_state} -> {new_state}")
            return True
    
    def _is_valid_transition(self, old_state: SessionState, new_state: SessionState) -> bool:
        """Validate if state transition is allowed"""
        valid_transitions = {
            SessionState.CREATED: [SessionState.PREPARING, SessionState.FAILED],
            SessionState.PREPARING: [SessionState.READY, SessionState.FAILED],
            SessionState.READY: [SessionState.STARTING, SessionState.FAILED],
            SessionState.STARTING: [SessionState.TRAINING, SessionState.FAILED],
            SessionState.TRAINING: [SessionState.PAUSED, SessionState.STOPPING, SessionState.COMPLETED, SessionState.FAILED],
            SessionState.PAUSED: [SessionState.RESUMING, SessionState.STOPPING, SessionState.FAILED],
            SessionState.RESUMING: [SessionState.TRAINING, SessionState.FAILED],
            SessionState.STOPPING: [SessionState.STOPPED, SessionState.FAILED],
            SessionState.STOPPED: [SessionState.CLEANUP, SessionState.ARCHIVED],
            SessionState.COMPLETED: [SessionState.CLEANUP, SessionState.ARCHIVED],
            SessionState.FAILED: [SessionState.CLEANUP],
            SessionState.CLEANUP: [SessionState.ARCHIVED],
            SessionState.ARCHIVED: []  # Terminal state
        }
        
        return new_state in valid_transitions.get(old_state, [])
    
    def allocate_resources(self, resource_requirements: Dict[str, Any]) -> bool:
        """Allocate resources for training session"""
        try:
            # GPU allocation
            if resource_requirements.get("gpu_required", False):
                self.resources["gpu_allocated"] = True
                
            # Memory allocation
            memory_mb = resource_requirements.get("memory_mb", 512)
            self.resources["memory_allocated_mb"] = memory_mb
            
            # CPU threads
            cpu_threads = resource_requirements.get("cpu_threads", 1)
            self.resources["cpu_threads"] = cpu_threads
            
            logger.info(f"Allocated resources for session {self.session_id}: {self.resources}")
            return True
        except Exception as e:
            logger.error(f"Failed to allocate resources for session {self.session_id}: {e}")
            return False
    
    def deallocate_resources(self):
        """Deallocate all resources for this session"""
        try:
            # Reset resource tracking
            self.resources = {
                "gpu_allocated": False,
                "memory_allocated_mb": 0,
                "cpu_threads": 0,
                "storage_used_mb": 0
            }
            logger.info(f"Deallocated resources for session {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to deallocate resources for session {self.session_id}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "domain_name": self.domain_name,
            "config": asdict(self.config),
            "state": self.state.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "current_epoch": self.current_epoch,
            "best_performance": self.best_performance,
            "resources": self.resources,
            "error_info": self.error_info
        }

class SessionRegistry:
    """
    Thread-safe singleton registry that manages all training sessions.
    Replaces both _sessions (legacy) and _enhanced_sessions (enhanced) dictionaries.
    """
    
    _instance: Optional['SessionRegistry'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        if SessionRegistry._instance is not None:
            raise RuntimeError("SessionRegistry is a singleton. Use get_instance().")
        
        # Core session storage
        self._sessions: Dict[str, UnifiedSession] = {}
        self._session_lock = threading.RLock()
        
        # Session organization
        self._active_sessions: Set[str] = set()
        self._sessions_by_domain: Dict[str, Set[str]] = {}
        
        # Storage configuration
        self.storage_root = Path(".brain/sessions")
        self.storage_root.mkdir(parents=True, exist_ok=True)
        
        # Registry statistics
        self.total_sessions_created = 0
        self.total_sessions_completed = 0
        self.total_sessions_failed = 0
        
        logger.info("Initialized SessionRegistry")
    
    @classmethod
    def get_instance(cls) -> 'SessionRegistry':
        """Get singleton instance of SessionRegistry"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def create_session(self, domain_name: str, config: UnifiedTrainingConfig = None,
                      session_id: str = None) -> str:
        """
        Create a new unified session with atomic operation guarantees.
        Returns session_id on success, raises exception on failure.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        if config is None:
            config = UnifiedTrainingConfig()
        
        with self._session_lock:
            # Check for duplicate session ID
            if session_id in self._sessions:
                raise ValueError(f"Session {session_id} already exists")
            
            try:
                # Create session storage directory
                session_storage = self.storage_root / session_id
                session_storage.mkdir(parents=True, exist_ok=True)
                
                # Create unified session
                session = UnifiedSession(session_id, domain_name, config, session_storage)
                
                # Atomic registration
                self._sessions[session_id] = session
                self._active_sessions.add(session_id)
                
                # Update domain tracking
                if domain_name not in self._sessions_by_domain:
                    self._sessions_by_domain[domain_name] = set()
                self._sessions_by_domain[domain_name].add(session_id)
                
                # Update statistics
                self.total_sessions_created += 1
                
                # Save session metadata
                self._save_session_metadata(session)
                
                logger.info(f"Created session {session_id} for domain {domain_name}")
                return session_id
                
            except Exception as e:
                # Rollback on failure
                self._sessions.pop(session_id, None)
                self._active_sessions.discard(session_id)
                if domain_name in self._sessions_by_domain:
                    self._sessions_by_domain[domain_name].discard(session_id)
                
                logger.error(f"Failed to create session {session_id}: {e}")
                raise
    
    def get_session(self, session_id: str) -> Optional[UnifiedSession]:
        """Get session by ID with thread safety"""
        with self._session_lock:
            return self._sessions.get(session_id)
    
    def has_session(self, session_id: str) -> bool:
        """Check if session exists"""
        with self._session_lock:
            return session_id in self._sessions
    
    def get_sessions_by_domain(self, domain_name: str) -> List[UnifiedSession]:
        """Get all sessions for a specific domain"""
        with self._session_lock:
            session_ids = self._sessions_by_domain.get(domain_name, set())
            return [self._sessions[sid] for sid in session_ids if sid in self._sessions]
    
    def get_active_sessions(self) -> List[UnifiedSession]:
        """Get all currently active sessions"""
        with self._session_lock:
            return [self._sessions[sid] for sid in self._active_sessions if sid in self._sessions]
    
    def cleanup_session(self, session_id: str) -> bool:
        """
        Comprehensive session cleanup with atomic guarantees.
        Removes session from registry and cleans up all resources.
        """
        with self._session_lock:
            session = self._sessions.get(session_id)
            if not session:
                logger.warning(f"Cannot cleanup non-existent session {session_id}")
                return False
            
            try:
                # Transition to cleanup state
                session.transition_to(SessionState.CLEANUP)
                
                # Deallocate resources
                session.deallocate_resources()
                
                # Clean up artifacts
                session.artifacts.cleanup()
                
                # Remove from registry
                self._sessions.pop(session_id, None)
                self._active_sessions.discard(session_id)
                
                # Update domain tracking
                for domain_sessions in self._sessions_by_domain.values():
                    domain_sessions.discard(session_id)
                
                # Update statistics
                if session.state == SessionState.COMPLETED:
                    self.total_sessions_completed += 1
                elif session.state == SessionState.FAILED:
                    self.total_sessions_failed += 1
                
                logger.info(f"Cleaned up session {session_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to cleanup session {session_id}: {e}")
                return False
    
    def _save_session_metadata(self, session: UnifiedSession):
        """Save session metadata to disk"""
        try:
            metadata_file = session.storage_path / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata for session {session.session_id}: {e}")
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        with self._session_lock:
            return {
                "total_sessions": len(self._sessions),
                "active_sessions": len(self._active_sessions),
                "sessions_by_domain": {domain: len(sessions) 
                                     for domain, sessions in self._sessions_by_domain.items()},
                "total_created": self.total_sessions_created,
                "total_completed": self.total_sessions_completed,
                "total_failed": self.total_sessions_failed
            }
    
    def shutdown(self):
        """Shutdown registry and cleanup all active sessions"""
        with self._session_lock:
            logger.info(f"Shutting down SessionRegistry with {len(self._active_sessions)} active sessions")
            
            for session_id in list(self._active_sessions):
                self.cleanup_session(session_id)
            
            logger.info("SessionRegistry shutdown complete")

# Convenience functions for backwards compatibility
def get_session_registry() -> SessionRegistry:
    """Get the global session registry instance"""
    return SessionRegistry.get_instance()

def create_unified_config(**kwargs) -> UnifiedTrainingConfig:
    """Create a unified training configuration with specified parameters"""
    return UnifiedTrainingConfig(**kwargs)