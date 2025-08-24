"""
Domain State Manager for Universal AI Core Brain.
Manages per-domain state including learning progress, model parameters, and performance metrics.
"""

import logging
import json
import pickle
import threading
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import copy


class TensorJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles PyTorch tensors and numpy arrays."""
    
    def default(self, obj):
        try:
            # Handle PyTorch tensors
            if hasattr(obj, 'detach') and hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):
                return obj.detach().cpu().numpy().tolist()
            # Handle numpy arrays
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            # Handle numpy scalars
            elif hasattr(obj, 'item'):
                return obj.item()
            else:
                return super().default(obj)
        except:
            return str(obj)


class StateType(Enum):
    """Types of state data that can be stored."""
    MODEL_PARAMETERS = "model_parameters"
    LEARNING_STATE = "learning_state"
    PERFORMANCE_METRICS = "performance_metrics"
    TRAINING_HISTORY = "training_history"
    CONFIGURATION = "configuration"
    CHECKPOINT = "checkpoint"
    EMBEDDINGS = "embeddings"
    KNOWLEDGE_BASE = "knowledge_base"


class StateUpdateType(Enum):
    """Types of state updates."""
    FULL_REPLACE = "full_replace"
    MERGE = "merge"
    INCREMENTAL = "incremental"
    APPEND = "append"


@dataclass
class StateVersion:
    """Version information for state snapshots."""
    version_id: str
    timestamp: datetime
    description: str
    size_bytes: int
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'version_id': self.version_id,
            'timestamp': self.timestamp.isoformat(),
            'description': self.description,
            'size_bytes': self.size_bytes,
            'checksum': self.checksum,
            'metadata': self.metadata
        }


@dataclass
class DomainState:
    """Complete state for a domain."""
    domain_name: str
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    version: int = 1
    
    # Core state components
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    learning_state: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    training_history: List[Dict[str, Any]] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Advanced state components
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    total_training_steps: int = 0
    total_predictions: int = 0
    best_performance: float = 0.0
    state_size_bytes: int = 0
    
    def get_state_by_type(self, state_type: StateType) -> Any:
        """Get state data by type."""
        type_mapping = {
            StateType.MODEL_PARAMETERS: self.model_parameters,
            StateType.LEARNING_STATE: self.learning_state,
            StateType.PERFORMANCE_METRICS: self.performance_metrics,
            StateType.TRAINING_HISTORY: self.training_history,
            StateType.CONFIGURATION: self.configuration,
            StateType.EMBEDDINGS: self.embeddings,
            StateType.KNOWLEDGE_BASE: self.knowledge_base,
            StateType.CHECKPOINT: self.checkpoints
        }
        return type_mapping.get(state_type, {})
    
    def update_state_by_type(self, state_type: StateType, data: Any, 
                           update_type: StateUpdateType = StateUpdateType.MERGE) -> None:
        """Update state data by type."""
        current_data = self.get_state_by_type(state_type)
        
        if update_type == StateUpdateType.FULL_REPLACE:
            # Complete replacement
            if state_type == StateType.MODEL_PARAMETERS:
                self.model_parameters = data
            elif state_type == StateType.LEARNING_STATE:
                self.learning_state = data
            elif state_type == StateType.PERFORMANCE_METRICS:
                self.performance_metrics = data
            elif state_type == StateType.CONFIGURATION:
                self.configuration = data
            elif state_type == StateType.EMBEDDINGS:
                self.embeddings = data
            elif state_type == StateType.KNOWLEDGE_BASE:
                self.knowledge_base = data
                
        elif update_type == StateUpdateType.MERGE:
            # Merge with existing data
            if isinstance(current_data, dict) and isinstance(data, dict):
                current_data.update(data)
            elif isinstance(current_data, list) and isinstance(data, list):
                current_data.extend(data)
                
        elif update_type == StateUpdateType.INCREMENTAL:
            # Incremental update (for numerical values)
            if isinstance(current_data, dict) and isinstance(data, dict):
                for key, value in data.items():
                    if key in current_data and isinstance(value, (int, float)):
                        current_data[key] += value
                    else:
                        current_data[key] = value
                        
        elif update_type == StateUpdateType.APPEND:
            # Append to lists
            if state_type == StateType.TRAINING_HISTORY:
                self.training_history.append(data)
            elif state_type == StateType.CHECKPOINT:
                self.checkpoints.append(data)
        
        self.last_updated = datetime.now()
        self.version += 1
    
    def calculate_size(self) -> int:
        """Calculate approximate size of state in bytes."""
        try:
            # Serialize to estimate size
            state_dict = {
                'model_parameters': self.model_parameters,
                'learning_state': self.learning_state,
                'performance_metrics': self.performance_metrics,
                'training_history': self.training_history,
                'configuration': self.configuration,
                'knowledge_base': self.knowledge_base,
                'checkpoints': self.checkpoints
            }
            serialized = pickle.dumps(state_dict)
            self.state_size_bytes = len(serialized)
            return self.state_size_bytes
        except Exception:
            return 0
    
    def create_checkpoint(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a checkpoint of current state."""
        checkpoint = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'version': self.version,
            'total_training_steps': self.total_training_steps,
            'total_predictions': self.total_predictions,
            'best_performance': self.best_performance,
            'model_parameters': copy.deepcopy(self.model_parameters),
            'learning_state': copy.deepcopy(self.learning_state),
            'performance_metrics': copy.deepcopy(self.performance_metrics),
            'metadata': metadata or {}
        }
        self.checkpoints.append(checkpoint)
        # Keep only last 10 checkpoints
        if len(self.checkpoints) > 10:
            self.checkpoints = self.checkpoints[-10:]
        return checkpoint
    
    def restore_checkpoint(self, checkpoint_name: str) -> bool:
        """Restore state from checkpoint."""
        for checkpoint in reversed(self.checkpoints):
            if checkpoint['name'] == checkpoint_name:
                self.model_parameters = copy.deepcopy(checkpoint['model_parameters'])
                self.learning_state = copy.deepcopy(checkpoint['learning_state'])
                self.performance_metrics = copy.deepcopy(checkpoint['performance_metrics'])
                self.total_training_steps = checkpoint['total_training_steps']
                self.total_predictions = checkpoint['total_predictions']
                self.best_performance = checkpoint['best_performance']
                self.last_updated = datetime.now()
                self.version += 1
                return True
        return False


class DomainStateManager:
    """
    Manages per-domain state with isolation and persistence.
    Each domain maintains independent learning state, model parameters, and metrics.
    """
    
    def __init__(self, domain_registry, storage_path: Optional[Path] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize domain state manager.
        
        Args:
            domain_registry: DomainRegistry instance
            storage_path: Optional path for persisting state files
            logger: Optional logger instance
        """
        self.domain_registry = domain_registry
        self.storage_path = Path(storage_path) if storage_path else Path.cwd() / ".domain_states"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Logger setup
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Domain states storage
        self._domain_states: Dict[str, DomainState] = {}
        self._state_versions: Dict[str, List[StateVersion]] = defaultdict(list)
        self._state_locks: Dict[str, threading.RLock] = defaultdict(threading.RLock)
        
        # Configuration
        self.max_versions_per_domain = 10
        self.auto_checkpoint_interval = 100  # Auto checkpoint every N updates
        self.update_counters: Dict[str, int] = defaultdict(int)
        
        # Global lock for state management
        self._global_lock = threading.RLock()
        
        # Load existing states
        self._load_existing_states()
        
        self.logger.info(f"DomainStateManager initialized with storage at {self.storage_path}")
    
    def _load_existing_states(self) -> None:
        """Load existing domain states from storage."""
        if not self.storage_path.exists():
            return
        
        state_files = list(self.storage_path.glob("*_state.json"))
        for state_file in state_files:
            try:
                domain_name = state_file.stem.replace("_state", "")
                self.load_domain_state(domain_name, state_file)
                self.logger.debug(f"Loaded existing state for domain: {domain_name}")
            except Exception as e:
                self.logger.warning(f"Failed to load state from {state_file}: {e}")
    
    def _ensure_domain_exists(self, domain_name: str) -> bool:
        """Ensure domain is registered and has isolation."""
        # Check if this looks like a session ID (UUID format)
        import re
        session_id_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
        
        if session_id_pattern.match(domain_name):
            # This is a session ID, not a domain name - skip domain registration check
            self.logger.debug(f"Skipping domain registration check for session ID: {domain_name}")
            return True
        
        if not self.domain_registry.is_domain_registered(domain_name):
            self.logger.error(f"Domain '{domain_name}' not registered")
            return False
        
        # Ensure isolation exists in registry
        if not self.domain_registry._ensure_isolation(domain_name):
            self.logger.error(f"Failed to ensure isolation for domain '{domain_name}'")
            return False
        
        return True
    
    def _initialize_domain_state(self, domain_name: str) -> DomainState:
        """Initialize a new domain state."""
        domain_info = self.domain_registry.get_domain_info(domain_name)
        
        # Create initial state with domain configuration
        initial_state = DomainState(
            domain_name=domain_name,
            configuration={
                'domain_type': domain_info['type'],
                'version': domain_info['version'],
                'max_memory_mb': domain_info['config']['max_memory_mb'],
                'hidden_layers': domain_info['config']['hidden_layers'],
                'activation_function': domain_info['config']['activation_function'],
                'learning_rate': domain_info['config']['learning_rate'],
                'dropout_rate': domain_info['config']['dropout_rate']
            },
            learning_state={
                'optimizer_state': {},
                'learning_rate_schedule': [],
                'gradient_history': [],
                'loss_history': []
            },
            performance_metrics={
                'accuracy': 0.0,
                'loss': float('inf'),
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'confusion_matrix': None
            }
        )
        
        # Initialize model parameters based on hidden layers
        if 'hidden_layers' in domain_info['config']:
            layers = domain_info['config']['hidden_layers']
            initial_state.model_parameters = {
                f'layer_{i}': {
                    'weights': None,  # Will be initialized when model is created
                    'biases': None,
                    'activation': domain_info['config']['activation_function']
                }
                for i, layer_size in enumerate(layers)
            }
        
        return initial_state
    
    def get_domain_state(self, domain_name: str, 
                        state_type: Optional[StateType] = None) -> Optional[Union[DomainState, Any]]:
        """
        Get domain's current state.
        
        Args:
            domain_name: Name of the domain
            state_type: Optional specific state type to retrieve
            
        Returns:
            Complete DomainState or specific state component if state_type specified
        """
        with self._global_lock:
            if not self._ensure_domain_exists(domain_name):
                return None
            
            # Get or create domain state
            if domain_name not in self._domain_states:
                self._domain_states[domain_name] = self._initialize_domain_state(domain_name)
                self.logger.debug(f"Initialized new state for domain: {domain_name}")
            
            domain_state = self._domain_states[domain_name]
            
            # Return specific state type if requested
            if state_type:
                return domain_state.get_state_by_type(state_type)
            
            return domain_state
    
    def update_domain_state(self, domain_name: str, state_update: Dict[str, Any],
                          update_type: StateUpdateType = StateUpdateType.MERGE) -> bool:
        """
        Update domain state with new data.
        
        Args:
            domain_name: Name of the domain
            state_update: Dictionary with state updates
            update_type: Type of update operation
            
        Returns:
            True if update successful, False otherwise
        """
        with self._state_locks[domain_name]:
            try:
                # Ensure domain exists
                if not self._ensure_domain_exists(domain_name):
                    return False
                
                # Get current state
                domain_state = self.get_domain_state(domain_name)
                if not domain_state:
                    return False
                
                # Apply updates
                for key, value in state_update.items():
                    if key == 'model_parameters':
                        domain_state.update_state_by_type(StateType.MODEL_PARAMETERS, value, update_type)
                    elif key == 'learning_state':
                        domain_state.update_state_by_type(StateType.LEARNING_STATE, value, update_type)
                    elif key == 'performance_metrics':
                        domain_state.update_state_by_type(StateType.PERFORMANCE_METRICS, value, update_type)
                    elif key == 'training_history':
                        if update_type == StateUpdateType.APPEND:
                            domain_state.update_state_by_type(StateType.TRAINING_HISTORY, value, update_type)
                        else:
                            domain_state.training_history = value
                    elif key == 'configuration':
                        domain_state.update_state_by_type(StateType.CONFIGURATION, value, update_type)
                    elif key == 'embeddings':
                        domain_state.update_state_by_type(StateType.EMBEDDINGS, value, update_type)
                    elif key == 'knowledge_base':
                        domain_state.update_state_by_type(StateType.KNOWLEDGE_BASE, value, update_type)
                    elif key == 'total_training_steps':
                        domain_state.total_training_steps = value
                    elif key == 'total_predictions':
                        domain_state.total_predictions = value
                    elif key == 'best_performance':
                        domain_state.best_performance = max(domain_state.best_performance, value)
                
                # Update size
                domain_state.calculate_size()
                
                # Update counter for auto-checkpoint
                self.update_counters[domain_name] += 1
                if self.update_counters[domain_name] >= self.auto_checkpoint_interval:
                    domain_state.create_checkpoint(f"auto_{int(time.time())}")
                    self.update_counters[domain_name] = 0
                    self.logger.debug(f"Created auto-checkpoint for domain: {domain_name}")
                
                # Sync with domain registry isolation
                registry_state = {
                    'model_parameters': domain_state.model_parameters,
                    'learning_state': domain_state.learning_state,
                    'performance_metrics': domain_state.performance_metrics,
                    'last_updated': domain_state.last_updated.isoformat(),
                    'version': domain_state.version
                }
                self.domain_registry.set_domain_state(domain_name, registry_state)
                
                self.logger.debug(f"Updated state for domain '{domain_name}' (version: {domain_state.version})")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to update domain state for '{domain_name}': {e}")
                return False
    
    def save_domain_state(self, domain_name: str, filepath: Optional[Path] = None) -> bool:
        """
        Persist domain state to file.
        
        Args:
            domain_name: Name of the domain
            filepath: Optional custom filepath (uses default if None)
            
        Returns:
            True if save successful, False otherwise
        """
        with self._state_locks[domain_name]:
            try:
                domain_state = self.get_domain_state(domain_name)
                if not domain_state:
                    return False
                
                # Determine filepath
                if filepath is None:
                    filepath = self.storage_path / f"{domain_name}_state.json"
                
                # Prepare state for serialization
                state_data = {
                    'domain_name': domain_state.domain_name,
                    'created_at': domain_state.created_at.isoformat(),
                    'last_updated': domain_state.last_updated.isoformat(),
                    'version': domain_state.version,
                    'model_parameters': domain_state.model_parameters,
                    'learning_state': domain_state.learning_state,
                    'performance_metrics': domain_state.performance_metrics,
                    'training_history': domain_state.training_history[-100:],  # Keep last 100
                    'configuration': domain_state.configuration,
                    'knowledge_base': domain_state.knowledge_base,
                    'checkpoints': domain_state.checkpoints,
                    'total_training_steps': domain_state.total_training_steps,
                    'total_predictions': domain_state.total_predictions,
                    'best_performance': domain_state.best_performance,
                    'state_size_bytes': domain_state.state_size_bytes
                }
                
                # Handle numpy arrays in embeddings separately
                embeddings_file = filepath.with_suffix('.embeddings.npz')
                if domain_state.embeddings:
                    np.savez_compressed(embeddings_file, **domain_state.embeddings)
                    state_data['embeddings_file'] = str(embeddings_file.name)
                
                # Calculate checksum
                state_str = json.dumps(state_data, sort_keys=True)
                checksum = hashlib.sha256(state_str.encode()).hexdigest()
                
                # Write state file
                with open(filepath, 'w') as f:
                    json.dump(state_data, f, indent=2, cls=TensorJSONEncoder)
                
                # Create version entry
                version = StateVersion(
                    version_id=f"v{domain_state.version}_{int(time.time())}",
                    timestamp=datetime.now(),
                    description=f"Saved state version {domain_state.version}",
                    size_bytes=filepath.stat().st_size,
                    checksum=checksum
                )
                
                self._state_versions[domain_name].append(version)
                if len(self._state_versions[domain_name]) > self.max_versions_per_domain:
                    self._state_versions[domain_name] = self._state_versions[domain_name][-self.max_versions_per_domain:]
                
                self.logger.info(f"Saved state for domain '{domain_name}' to {filepath}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to save domain state for '{domain_name}': {e}")
                return False
    
    def load_domain_state(self, domain_name: str, filepath: Optional[Path] = None) -> bool:
        """
        Load domain state from file.
        
        Args:
            domain_name: Name of the domain
            filepath: Optional custom filepath (uses default if None)
            
        Returns:
            True if load successful, False otherwise
        """
        with self._state_locks[domain_name]:
            try:
                # Ensure domain exists
                if not self._ensure_domain_exists(domain_name):
                    return False
                
                # Determine filepath
                if filepath is None:
                    filepath = self.storage_path / f"{domain_name}_state.json"
                
                if not filepath.exists():
                    self.logger.warning(f"State file not found: {filepath}")
                    return False
                
                # Load state data
                with open(filepath, 'r') as f:
                    state_data = json.load(f)
                
                # Create domain state
                domain_state = DomainState(
                    domain_name=domain_name,
                    created_at=datetime.fromisoformat(state_data['created_at']),
                    last_updated=datetime.fromisoformat(state_data['last_updated']),
                    version=state_data['version'],
                    model_parameters=state_data.get('model_parameters', {}),
                    learning_state=state_data.get('learning_state', {}),
                    performance_metrics=state_data.get('performance_metrics', {}),
                    training_history=state_data.get('training_history', []),
                    configuration=state_data.get('configuration', {}),
                    knowledge_base=state_data.get('knowledge_base', {}),
                    checkpoints=state_data.get('checkpoints', []),
                    total_training_steps=state_data.get('total_training_steps', 0),
                    total_predictions=state_data.get('total_predictions', 0),
                    best_performance=state_data.get('best_performance', 0.0),
                    state_size_bytes=state_data.get('state_size_bytes', 0)
                )
                
                # Load embeddings if available
                if 'embeddings_file' in state_data:
                    embeddings_file = filepath.parent / state_data['embeddings_file']
                    if embeddings_file.exists():
                        embeddings_data = np.load(embeddings_file)
                        domain_state.embeddings = {key: embeddings_data[key] for key in embeddings_data.files}
                
                # Store in manager
                self._domain_states[domain_name] = domain_state
                
                # Sync with domain registry
                registry_state = {
                    'model_parameters': domain_state.model_parameters,
                    'learning_state': domain_state.learning_state,
                    'performance_metrics': domain_state.performance_metrics,
                    'last_loaded': datetime.now().isoformat(),
                    'loaded_version': domain_state.version
                }
                self.domain_registry.set_domain_state(domain_name, registry_state)
                
                self.logger.info(f"Loaded state for domain '{domain_name}' from {filepath} (version: {domain_state.version})")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to load domain state for '{domain_name}': {e}")
                return False
    
    def reset_domain_state(self, domain_name: str, preserve_config: bool = True) -> bool:
        """
        Reset domain to initial state.
        
        Args:
            domain_name: Name of the domain
            preserve_config: Whether to preserve configuration
            
        Returns:
            True if reset successful, False otherwise
        """
        with self._state_locks[domain_name]:
            try:
                # Ensure domain exists
                if not self._ensure_domain_exists(domain_name):
                    return False
                
                # Get current configuration if preserving
                current_config = None
                if preserve_config and domain_name in self._domain_states:
                    current_config = self._domain_states[domain_name].configuration.copy()
                
                # Create fresh state
                new_state = self._initialize_domain_state(domain_name)
                
                # Restore configuration if requested
                if preserve_config and current_config:
                    new_state.configuration = current_config
                
                # Replace state
                self._domain_states[domain_name] = new_state
                
                # Clear version history
                self._state_versions[domain_name].clear()
                
                # Reset update counter
                self.update_counters[domain_name] = 0
                
                # Sync with domain registry
                self.domain_registry.clear_domain_isolation(domain_name)
                self.domain_registry.create_domain_isolation(domain_name)
                
                self.logger.info(f"Reset state for domain '{domain_name}' (config preserved: {preserve_config})")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to reset domain state for '{domain_name}': {e}")
                return False
    
    def create_checkpoint(self, domain_name: str, checkpoint_name: str,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a named checkpoint for domain state.
        
        Args:
            domain_name: Name of the domain
            checkpoint_name: Name for the checkpoint
            metadata: Optional metadata to store with checkpoint
            
        Returns:
            True if checkpoint created successfully, False otherwise
        """
        with self._state_locks[domain_name]:
            try:
                domain_state = self.get_domain_state(domain_name)
                if not domain_state:
                    return False
                
                checkpoint = domain_state.create_checkpoint(checkpoint_name, metadata)
                
                # Also save to file
                checkpoint_file = self.storage_path / f"{domain_name}_checkpoint_{checkpoint_name}.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint, f, indent=2)
                
                self.logger.info(f"Created checkpoint '{checkpoint_name}' for domain '{domain_name}'")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to create checkpoint for domain '{domain_name}': {e}")
                return False
    
    def restore_checkpoint(self, domain_name: str, checkpoint_name: str) -> bool:
        """
        Restore domain state from checkpoint.
        
        Args:
            domain_name: Name of the domain
            checkpoint_name: Name of checkpoint to restore
            
        Returns:
            True if restore successful, False otherwise
        """
        with self._state_locks[domain_name]:
            try:
                domain_state = self.get_domain_state(domain_name)
                if not domain_state:
                    return False
                
                # Try to restore from in-memory checkpoints first
                if domain_state.restore_checkpoint(checkpoint_name):
                    self.logger.info(f"Restored checkpoint '{checkpoint_name}' for domain '{domain_name}'")
                    return True
                
                # Try to load from file
                checkpoint_file = self.storage_path / f"{domain_name}_checkpoint_{checkpoint_name}.json"
                if checkpoint_file.exists():
                    with open(checkpoint_file, 'r') as f:
                        checkpoint = json.load(f)
                    
                    # Apply checkpoint data
                    domain_state.model_parameters = checkpoint['model_parameters']
                    domain_state.learning_state = checkpoint['learning_state']
                    domain_state.performance_metrics = checkpoint['performance_metrics']
                    domain_state.total_training_steps = checkpoint['total_training_steps']
                    domain_state.total_predictions = checkpoint['total_predictions']
                    domain_state.best_performance = checkpoint['best_performance']
                    domain_state.last_updated = datetime.now()
                    domain_state.version += 1
                    
                    self.logger.info(f"Restored checkpoint '{checkpoint_name}' from file for domain '{domain_name}'")
                    return True
                
                self.logger.warning(f"Checkpoint '{checkpoint_name}' not found for domain '{domain_name}'")
                return False
                
            except Exception as e:
                self.logger.error(f"Failed to restore checkpoint for domain '{domain_name}': {e}")
                return False
    
    def get_state_versions(self, domain_name: str) -> List[StateVersion]:
        """Get list of saved state versions for a domain."""
        with self._global_lock:
            return self._state_versions.get(domain_name, []).copy()
    
    def compare_states(self, domain_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two state versions.
        
        Args:
            domain_name: Name of the domain
            version1: First version ID
            version2: Second version ID
            
        Returns:
            Dictionary with comparison results
        """
        versions = self.get_state_versions(domain_name)
        
        v1_data = None
        v2_data = None
        
        for version in versions:
            if version.version_id == version1:
                v1_data = version
            elif version.version_id == version2:
                v2_data = version
        
        if not v1_data or not v2_data:
            return {'error': 'One or both versions not found'}
        
        return {
            'version1': v1_data.to_dict(),
            'version2': v2_data.to_dict(),
            'size_difference': v2_data.size_bytes - v1_data.size_bytes,
            'time_difference': (v2_data.timestamp - v1_data.timestamp).total_seconds()
        }
    
    def get_state_statistics(self, domain_name: str) -> Dict[str, Any]:
        """Get comprehensive statistics about domain state."""
        with self._state_locks[domain_name]:
            domain_state = self.get_domain_state(domain_name)
            if not domain_state:
                return {'error': 'Domain not found'}
            
            return {
                'domain_name': domain_name,
                'created_at': domain_state.created_at.isoformat(),
                'last_updated': domain_state.last_updated.isoformat(),
                'version': domain_state.version,
                'state_size_bytes': domain_state.calculate_size(),
                'total_training_steps': domain_state.total_training_steps,
                'total_predictions': domain_state.total_predictions,
                'best_performance': domain_state.best_performance,
                'checkpoint_count': len(domain_state.checkpoints),
                'training_history_length': len(domain_state.training_history),
                'embedding_count': len(domain_state.embeddings),
                'knowledge_items': len(domain_state.knowledge_base),
                'version_count': len(self._state_versions.get(domain_name, [])),
                'performance_summary': {
                    'current_accuracy': domain_state.performance_metrics.get('accuracy', 0),
                    'current_loss': domain_state.performance_metrics.get('loss', float('inf')),
                    'current_f1': domain_state.performance_metrics.get('f1_score', 0)
                }
            }
    
    def export_all_states(self, export_path: Path) -> bool:
        """Export all domain states to specified directory."""
        try:
            export_path = Path(export_path)
            export_path.mkdir(parents=True, exist_ok=True)
            
            exported_count = 0
            for domain_name in self._domain_states:
                filepath = export_path / f"{domain_name}_state.json"
                if self.save_domain_state(domain_name, filepath):
                    exported_count += 1
            
            # Export metadata
            metadata = {
                'export_timestamp': datetime.now().isoformat(),
                'total_domains': len(self._domain_states),
                'exported_domains': exported_count,
                'domain_list': list(self._domain_states.keys())
            }
            
            with open(export_path / "export_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Exported {exported_count} domain states to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export states: {e}")
            return False
    
    def sync_all_states(self) -> bool:
        """
        Synchronize all domain states with the domain registry.
        
        Returns:
            True if all states synced successfully, False otherwise
        """
        try:
            sync_count = 0
            for domain_name, domain_state in self._domain_states.items():
                try:
                    registry_state = {
                        'model_parameters': domain_state.model_parameters,
                        'learning_state': domain_state.learning_state,
                        'performance_metrics': domain_state.performance_metrics,
                        'last_synced': datetime.now().isoformat(),
                        'version': domain_state.version,
                        'total_training_steps': domain_state.total_training_steps,
                        'total_predictions': domain_state.total_predictions,
                        'best_performance': domain_state.best_performance
                    }
                    self.domain_registry.set_domain_state(domain_name, registry_state)
                    sync_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to sync state for domain '{domain_name}': {e}")
            
            self.logger.info(f"Synchronized {sync_count}/{len(self._domain_states)} domain states")
            return sync_count == len(self._domain_states)
            
        except Exception as e:
            self.logger.error(f"Failed to sync all domain states: {e}")
            return False
    
    def save_all_states(self) -> int:
        """
        Save all domain states to disk.
        
        Returns:
            Number of states successfully saved
        """
        saved_count = 0
        for domain_name in self._domain_states:
            if self.save_domain_state(domain_name):
                saved_count += 1
        
        self.logger.info(f"Saved {saved_count}/{len(self._domain_states)} domain states")
        return saved_count
    
    def get_all_domains(self) -> List[str]:
        """Get list of all domains with state."""
        with self._global_lock:
            return list(self._domain_states.keys())
    
    def get_domain_count(self) -> int:
        """Get total number of domains with state."""
        with self._global_lock:
            return len(self._domain_states)
    
    def cleanup_old_states(self, max_age_days: int = 30) -> int:
        """
        Clean up old state files and versions.
        
        Args:
            max_age_days: Maximum age in days for state files
            
        Returns:
            Number of files cleaned up
        """
        try:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            cleaned_count = 0
            
            # Clean up old state files
            if self.storage_path.exists():
                for state_file in self.storage_path.glob("*_state.json"):
                    file_time = datetime.fromtimestamp(state_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        state_file.unlink()
                        cleaned_count += 1
                        
                        # Also clean up related embeddings file
                        embeddings_file = state_file.with_suffix('.embeddings.npz')
                        if embeddings_file.exists():
                            embeddings_file.unlink()
                            cleaned_count += 1
            
            # Clean up old versions from memory
            for domain_name in list(self._state_versions.keys()):
                versions = self._state_versions[domain_name]
                self._state_versions[domain_name] = [
                    v for v in versions if v.timestamp >= cutoff_date
                ]
            
            self.logger.info(f"Cleaned up {cleaned_count} old state files")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old states: {e}")
            return 0
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"DomainStateManager(domains={len(self._domain_states)}, "
                f"storage={self.storage_path})")