#!/usr/bin/env python3
"""
Comprehensive Error Recovery and Rollback System for Training Infrastructure.

This module provides sophisticated error detection, classification, and recovery
capabilities that integrate seamlessly with the existing progress tracking system.
It includes checkpoint management, state rollback, and intelligent recovery strategies.
"""

import os
import sys
import json
import time
import threading
import traceback
import hashlib
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
import pickle

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """Types of errors that can occur during training."""
    MEMORY_ERROR = "memory_error"
    CUDA_ERROR = "cuda_error"
    DATA_ERROR = "data_error"
    MODEL_ERROR = "model_error"
    OPTIMIZATION_ERROR = "optimization_error"
    IO_ERROR = "io_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    CHECKPOINT_ERROR = "checkpoint_error"
    UNKNOWN_ERROR = "unknown_error"

class ErrorSeverity(Enum):
    """Severity levels for errors."""
    CRITICAL = "critical"     # Training must stop
    HIGH = "high"            # Major issue, needs immediate attention
    MEDIUM = "medium"        # Recoverable with effort
    LOW = "low"             # Minor issue, easy recovery
    WARNING = "warning"      # Not an error, but worth noting

class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"                    # Simple retry
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    REDUCE_LEARNING_RATE = "reduce_learning_rate"
    GRADIENT_CLIPPING = "gradient_clipping"
    CLEAR_CACHE = "clear_cache"
    ROLLBACK_CHECKPOINT = "rollback_checkpoint"
    ROLLBACK_STATE = "rollback_state"
    RESTART_TRAINING = "restart_training"
    SKIP_BATCH = "skip_batch"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class ErrorRecord:
    """Record of an error and recovery attempt."""
    timestamp: datetime
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    traceback_str: str
    recovery_strategy: Optional[RecoveryStrategy]
    recovery_success: bool
    recovery_time: float
    context: Dict[str, Any]
    session_id: Optional[str] = None
    epoch: Optional[int] = None
    batch: Optional[int] = None

@dataclass
class RecoveryCheckpoint:
    """Represents a recovery checkpoint."""
    checkpoint_id: str
    timestamp: datetime
    session_id: str
    epoch: int
    batch: int
    model_state: Optional[str]  # Path to model state file
    optimizer_state: Optional[str]  # Path to optimizer state file
    training_state: Dict[str, Any]
    metrics: Dict[str, Any]
    checksum: str
    size_bytes: int

def _extract_primitive_parameter(context: Dict[str, Any], param_name: str, param_type: type) -> Any:
    """
    HARD FAILURE: Extract primitive parameter value from context - NO FALLBACKS.
    
    Args:
        context: Context dictionary containing the parameter
        param_name: Name of the parameter to extract
        param_type: Expected type (int, float, etc.)
        
    Returns:
        Primitive value of the specified type
        
    Raises:
        ValueError: If parameter cannot be extracted or converted
    """
    if param_name not in context:
        raise ValueError(f"HARD FAILURE: Required parameter '{param_name}' not found in context")
    
    value = context[param_name]
    
    # Handle various input formats and ensure we get a primitive value
    if isinstance(value, dict):
        # Helper function for recursive value extraction
        def find_value_recursively(obj):
            if isinstance(obj, dict):
                if 'value' in obj:
                    return obj['value']
                elif 'suggested_value' in obj:
                    return obj['suggested_value']
                else:
                    for v in obj.values():
                        result = find_value_recursively(v)
                        if result is not None:
                            return result
            return None
        
        # Extract from nested dictionary if needed
        if param_name in value:
            nested_value = value[param_name]
            # Handle nested dictionaries recursively
            if isinstance(nested_value, dict):
                # Try direct keys first
                if 'value' in nested_value:
                    value = nested_value['value']
                elif 'suggested_value' in nested_value:
                    value = nested_value['suggested_value']
                else:
                    # Try recursive search through all nested values
                    extracted = find_value_recursively(nested_value)
                    if extracted is not None:
                        value = extracted
                    else:
                        raise ValueError(f"HARD FAILURE: Could not extract {param_name} from nested dictionary")
            else:
                value = nested_value
        elif 'value' in value:
            value = value['value']
        elif 'suggested_value' in value:
            value = value['suggested_value']
        else:
            # Try recursive search on the entire dictionary if param_name not found at top level
            extracted = find_value_recursively(value)
            if extracted is not None:
                value = extracted
            else:
                raise ValueError(f"HARD FAILURE: Could not extract {param_name} from dictionary")
    elif isinstance(value, (list, tuple)) and len(value) > 0:
        value = value[0]
        logger.info(f"Extracted {param_name} from list/tuple: {value}")
    
    # Ensure it's the correct type
    try:
        if param_type == int:
            value = int(value)
            if value <= 0:
                raise ValueError(f"HARD FAILURE: Invalid {param_name} value {value} - must be positive")
        elif param_type == float:
            value = float(value)
            if value <= 0:
                raise ValueError(f"HARD FAILURE: Invalid {param_name} value {value} - must be positive")
        else:
            value = param_type(value)
    except (ValueError, TypeError) as e:
        logger.error(f"Could not convert {param_name} to {param_type.__name__}: {e}")
        raise ValueError(f"Parameter {param_name} type conversion failed: {e}")
    
    return value


class ErrorClassifier:
    """Classifies errors and determines appropriate recovery strategies."""
    
    def __init__(self):
        self.error_patterns = {
            ErrorType.MEMORY_ERROR: [
                "out of memory", "cuda out of memory", "memory allocation",
                "insufficient memory", "memory error"
            ],
            ErrorType.CUDA_ERROR: [
                "cuda", "gpu", "device", "cublas", "cudnn", "nvidia"
            ],
            ErrorType.DATA_ERROR: [
                "data", "dataset", "batch", "loader", "input", "target",
                "shape mismatch", "dimension"
            ],
            ErrorType.MODEL_ERROR: [
                "model", "forward", "backward", "layer", "parameter",
                "weight", "bias"
            ],
            ErrorType.OPTIMIZATION_ERROR: [
                "optimizer", "gradient", "learning rate", "loss",
                "nan", "inf", "overflow"
            ],
            ErrorType.IO_ERROR: [
                "file", "directory", "path", "permission", "disk",
                "read", "write", "save", "load"
            ],
            ErrorType.TIMEOUT_ERROR: [
                "timeout", "time", "deadline", "expired"
            ],
            ErrorType.VALIDATION_ERROR: [
                "validation", "test", "eval", "metric"
            ],
            ErrorType.CHECKPOINT_ERROR: [
                "checkpoint", "state_dict", "model.load", "torch.load"
            ],
            ErrorType.UNKNOWN_ERROR: [
                "unknown", "unexpected", "generic", "unhandled"
            ]
        }
        
        self.severity_indicators = {
            ErrorSeverity.CRITICAL: [
                "critical", "fatal", "abort", "emergency", "system"
            ],
            ErrorSeverity.HIGH: [
                "error", "exception", "failed", "corrupt"
            ],
            ErrorSeverity.MEDIUM: [
                "issue", "problem", "moderate"
            ],
            ErrorSeverity.LOW: [
                "notice", "info", "minor"
            ],
            ErrorSeverity.WARNING: [
                "warning", "warn", "caution", "alert"
            ]
        }
    
    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> Tuple[ErrorType, ErrorSeverity]:
        """Classify an error and determine its severity."""
        error_str = str(error).lower()
        error_type_str = type(error).__name__.lower()
        
        context = context or {}
        
        # Determine error type
        error_type = ErrorType.UNKNOWN_ERROR
        max_matches = 0
        
        for etype, patterns in self.error_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in error_str or pattern in error_type_str)
            if matches > max_matches:
                max_matches = matches
                error_type = etype
        
        # Determine severity
        severity = ErrorSeverity.MEDIUM  # Default
        
        # Check for severity indicators in error message
        for sev, indicators in self.severity_indicators.items():
            if any(indicator in error_str for indicator in indicators):
                severity = sev
                break
        
        # Adjust severity based on error type
        if error_type in [ErrorType.MEMORY_ERROR, ErrorType.CUDA_ERROR]:
            if severity in [ErrorSeverity.LOW, ErrorSeverity.WARNING]:
                severity = ErrorSeverity.MEDIUM
        
        # Context-based severity adjustment
        if context:
            if context.get('consecutive_failures', 0) > 3:
                if severity != ErrorSeverity.CRITICAL:
                    severity = ErrorSeverity.HIGH
            
            if context.get('training_progress', 0) > 0.8:  # Near end of training
                if severity == ErrorSeverity.LOW:
                    severity = ErrorSeverity.MEDIUM
        
        return error_type, severity
    
    def get_recovery_strategies(self, error_type: ErrorType, severity: ErrorSeverity, 
                               context: Dict[str, Any] = None) -> List[RecoveryStrategy]:
        """Get appropriate recovery strategies for an error."""
        context = context or {}
        strategies = []
        
        # Basic strategies based on error type
        type_strategies = {
            ErrorType.MEMORY_ERROR: [
                RecoveryStrategy.CLEAR_CACHE,
                RecoveryStrategy.REDUCE_BATCH_SIZE,
                RecoveryStrategy.ROLLBACK_STATE
            ],
            ErrorType.CUDA_ERROR: [
                RecoveryStrategy.CLEAR_CACHE,
                RecoveryStrategy.RETRY,
                RecoveryStrategy.ROLLBACK_STATE
            ],
            ErrorType.DATA_ERROR: [
                RecoveryStrategy.SKIP_BATCH,
                RecoveryStrategy.RETRY,
                RecoveryStrategy.ROLLBACK_STATE
            ],
            ErrorType.MODEL_ERROR: [
                RecoveryStrategy.ROLLBACK_CHECKPOINT,
                RecoveryStrategy.RESTART_TRAINING
            ],
            ErrorType.OPTIMIZATION_ERROR: [
                RecoveryStrategy.REDUCE_LEARNING_RATE,
                RecoveryStrategy.GRADIENT_CLIPPING,
                RecoveryStrategy.ROLLBACK_STATE
            ],
            ErrorType.IO_ERROR: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.ROLLBACK_CHECKPOINT
            ],
            ErrorType.TIMEOUT_ERROR: [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.REDUCE_BATCH_SIZE
            ],
            ErrorType.VALIDATION_ERROR: [
                RecoveryStrategy.ROLLBACK_STATE,
                RecoveryStrategy.RETRY
            ],
            ErrorType.CHECKPOINT_ERROR: [
                RecoveryStrategy.ROLLBACK_CHECKPOINT,
                RecoveryStrategy.RESTART_TRAINING
            ]
        }
        
        strategies.extend(type_strategies.get(error_type, [RecoveryStrategy.RETRY]))
        
        # Severity-based escalation
        if severity == ErrorSeverity.CRITICAL:
            strategies = [RecoveryStrategy.EMERGENCY_STOP]
        elif severity == ErrorSeverity.HIGH:
            if RecoveryStrategy.ROLLBACK_CHECKPOINT not in strategies:
                strategies.insert(0, RecoveryStrategy.ROLLBACK_CHECKPOINT)
        
        # Context-based adjustments
        consecutive_failures = context.get('consecutive_failures', 0)
        if consecutive_failures > 2:
            # Escalate to more aggressive strategies
            if RecoveryStrategy.RESTART_TRAINING not in strategies:
                strategies.insert(-1, RecoveryStrategy.RESTART_TRAINING)  # Insert before last strategy
        
        if consecutive_failures > 5:
            strategies = [RecoveryStrategy.EMERGENCY_STOP]
        
        return strategies[:3]  # Limit to top 3 strategies

class CheckpointRecovery:
    """Manages training checkpoints for recovery purposes."""
    
    def __init__(self, checkpoint_dir: str = ".checkpoints", max_checkpoints: int = 10):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints: Dict[str, RecoveryCheckpoint] = {}
        self._lock = threading.Lock()
        
        # Load existing checkpoints
        self._load_checkpoint_index()
    
    def create_checkpoint(self, session_id: str, epoch: int, batch: int,
                         model_state: Any = None, optimizer_state: Any = None,
                         training_state: Dict[str, Any] = None,
                         metrics: Dict[str, Any] = None) -> str:
        """Create a new recovery checkpoint."""
        timestamp = datetime.now()
        checkpoint_id = f"{session_id}_{epoch}_{batch}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(exist_ok=True)
        
        model_path = None
        optimizer_path = None
        
        try:
            # Save model state if provided
            if model_state is not None:
                model_path = checkpoint_path / "model_state.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model_state, f)
            
            # Save optimizer state if provided
            if optimizer_state is not None:
                optimizer_path = checkpoint_path / "optimizer_state.pkl"
                with open(optimizer_path, 'wb') as f:
                    pickle.dump(optimizer_state, f)
            
            # Calculate checksum
            checksum = self._calculate_checkpoint_checksum(checkpoint_path)
            
            # Get checkpoint size
            size_bytes = sum(f.stat().st_size for f in checkpoint_path.rglob('*') if f.is_file())
            
            # Create checkpoint record
            checkpoint = RecoveryCheckpoint(
                checkpoint_id=checkpoint_id,
                timestamp=timestamp,
                session_id=session_id,
                epoch=epoch,
                batch=batch,
                model_state=str(model_path) if model_path else None,
                optimizer_state=str(optimizer_path) if optimizer_path else None,
                training_state=training_state or {},
                metrics=metrics or {},
                checksum=checksum,
                size_bytes=size_bytes
            )
            
            with self._lock:
                self.checkpoints[checkpoint_id] = checkpoint
                self._save_checkpoint_index()
                self._cleanup_old_checkpoints()
            
            logger.info(f"Created checkpoint {checkpoint_id} for session {session_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            # Cleanup failed checkpoint
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            raise
    
    def restore_checkpoint(self, checkpoint_id: str) -> Optional[RecoveryCheckpoint]:
        """Restore a checkpoint."""
        with self._lock:
            if checkpoint_id not in self.checkpoints:
                logger.warning(f"Checkpoint {checkpoint_id} not found")
                return None
            
            checkpoint = self.checkpoints[checkpoint_id]
            
            # Verify checkpoint integrity
            checkpoint_path = self.checkpoint_dir / checkpoint_id
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint directory {checkpoint_path} not found")
                return None
            
            current_checksum = self._calculate_checkpoint_checksum(checkpoint_path)
            if current_checksum != checkpoint.checksum:
                logger.error(f"Checkpoint {checkpoint_id} integrity check failed")
                return None
            
            logger.info(f"Restored checkpoint {checkpoint_id}")
            return checkpoint
    
    def list_checkpoints(self, session_id: str = None) -> List[RecoveryCheckpoint]:
        """List available checkpoints."""
        with self._lock:
            checkpoints = list(self.checkpoints.values())
            if session_id:
                checkpoints = [cp for cp in checkpoints if cp.session_id == session_id]
            return sorted(checkpoints, key=lambda x: x.timestamp, reverse=True)
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint."""
        with self._lock:
            if checkpoint_id not in self.checkpoints:
                return False
            
            checkpoint_path = self.checkpoint_dir / checkpoint_id
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            
            del self.checkpoints[checkpoint_id]
            self._save_checkpoint_index()
            
            logger.info(f"Deleted checkpoint {checkpoint_id}")
            return True
    
    def _calculate_checkpoint_checksum(self, checkpoint_path: Path) -> str:
        """Calculate SHA256 checksum of checkpoint directory."""
        hasher = hashlib.sha256()
        
        for file_path in sorted(checkpoint_path.rglob('*')):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _load_checkpoint_index(self):
        """Load checkpoint index from disk."""
        index_file = self.checkpoint_dir / "checkpoint_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    data = json.load(f)
                
                for cp_data in data.get('checkpoints', []):
                    cp_data['timestamp'] = datetime.fromisoformat(cp_data['timestamp'])
                    checkpoint = RecoveryCheckpoint(**cp_data)
                    self.checkpoints[checkpoint.checkpoint_id] = checkpoint
                    
            except Exception as e:
                logger.error(f"Failed to load checkpoint index: {e}")
                raise RuntimeError(f"Checkpoint index loading failed: {e}")
    
    def _save_checkpoint_index(self):
        """Save checkpoint index to disk."""
        index_file = self.checkpoint_dir / "checkpoint_index.json"
        try:
            data = {
                'checkpoints': []
            }
            
            for checkpoint in self.checkpoints.values():
                cp_data = asdict(checkpoint)
                cp_data['timestamp'] = checkpoint.timestamp.isoformat()
                data['checkpoints'].append(cp_data)
            
            with open(index_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint index: {e}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if limit exceeded."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by timestamp and keep only the newest
        sorted_checkpoints = sorted(self.checkpoints.values(), 
                                   key=lambda x: x.timestamp, reverse=True)
        
        to_delete = sorted_checkpoints[self.max_checkpoints:]
        
        # Delete checkpoints without acquiring lock (already held)
        for checkpoint in to_delete:
            checkpoint_path = self.checkpoint_dir / checkpoint.checkpoint_id
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            
            if checkpoint.checkpoint_id in self.checkpoints:
                del self.checkpoints[checkpoint.checkpoint_id]
            
            logger.info(f"Cleaned up old checkpoint {checkpoint.checkpoint_id}")
        
        # Save updated index after cleanup
        self._save_checkpoint_index()

class StateRollback:
    """Manages state rollback for quick recovery."""
    
    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.state_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
    
    def save_state(self, state: Dict[str, Any], metadata: Dict[str, Any] = None):
        """Save current state for potential rollback."""
        timestamp = datetime.now()
        state_record = {
            'timestamp': timestamp,
            'state': state.copy(),
            'metadata': metadata or {}
        }
        
        with self._lock:
            self.state_history.append(state_record)
            
            # Cleanup old history
            if len(self.state_history) > self.max_history:
                self.state_history = self.state_history[-self.max_history:]
    
    def rollback(self, steps: int = 1) -> Optional[Dict[str, Any]]:
        """Rollback to a previous state."""
        with self._lock:
            if len(self.state_history) < steps:
                logger.warning(f"Cannot rollback {steps} steps, only {len(self.state_history)} states available")
                return None
            
            # Remove recent states
            for _ in range(steps):
                if self.state_history:
                    self.state_history.pop()
            
            if self.state_history:
                rolled_back_state = self.state_history[-1]['state']
                logger.info(f"Rolled back {steps} steps to state from {self.state_history[-1]['timestamp']}")
                return rolled_back_state
            
            return None
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get state history."""
        with self._lock:
            return self.state_history[-limit:] if self.state_history else []
    
    def clear_history(self):
        """Clear all state history."""
        with self._lock:
            self.state_history.clear()

class ErrorRecoveryManager:
    """Central coordinator for error recovery operations.
    
    By default, operates in 'fail-hard' mode where errors are logged with
    comprehensive diagnostics but NOT automatically recovered from.
    This ensures problems are visible and properly addressed.
    """
    
    def __init__(self, checkpoint_dir: str = ".checkpoints", 
                 max_checkpoints: int = 10, max_state_history: int = 50,
                 fail_hard: bool = True):
        self.fail_hard = fail_hard  # If True, never attempt automatic recovery
        self.classifier = ErrorClassifier()
        self.checkpoint_recovery = CheckpointRecovery(checkpoint_dir, max_checkpoints)
        self.state_rollback = StateRollback(max_state_history)
        
        self.error_history: List[ErrorRecord] = []
        self.recovery_stats = {
            'total_errors': 0,
            'total_recoveries': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'recovery_strategies_used': {},
            'error_types_seen': {},
            'average_recovery_time': 0.0
        }
        
        self._lock = threading.Lock()
        self.recovery_callbacks: List[Callable] = []
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> bool:
        """Handle an error by logging diagnostics and failing hard.
        
        This method captures comprehensive error information but does NOT attempt
        automatic recovery. Instead, it ensures errors fail fast with detailed
        diagnostics for debugging.
        """
        start_time = time.time()
        context = context or {}
        
        # Classify the error for diagnostic purposes
        error_type, severity = self.classifier.classify_error(error, context)
        
        # Get what recovery strategies WOULD be used (for diagnostics only)
        recovery_strategies = self.classifier.get_recovery_strategies(error_type, severity, context)
        
        logger.error(f"HARD FAILURE - Error occurred: {error_type.value} (severity: {severity.value})")
        logger.error(f"Error message: {str(error)}")
        logger.error(f"Error traceback:\n{traceback.format_exc()}")
        logger.error(f"Context at failure: {json.dumps(context, default=str, indent=2)}")
        logger.error(f"Suggested recovery strategies (NOT applied): {[s.value for s in recovery_strategies]}")
        
        recovery_success = False
        recovery_strategy_used = None
        
        if self.fail_hard:
            # DO NOT attempt recovery - just log what we would have done
            logger.error(f"FAILING HARD - No automatic recovery attempted")
            logger.error(f"Manual intervention required to address: {error_type.value}")
            
            # Log diagnostic information about what recovery would be attempted
            for strategy in recovery_strategies[:1]:  # Just show primary strategy
                diagnosis = self._diagnose_recovery_strategy(strategy, error, context)
                logger.error(f"Diagnosis: {json.dumps(diagnosis, default=str, indent=2)}")
        else:
            # Legacy behavior - attempt automatic recovery (NOT RECOMMENDED)
            logger.warning("Automatic recovery mode enabled (NOT RECOMMENDED for production)")
            for strategy in recovery_strategies:
                try:
                    logger.info(f"Attempting recovery with strategy: {strategy.value}")
                    
                    if self._execute_recovery_strategy_legacy(strategy, error, context):
                        recovery_success = True
                        recovery_strategy_used = strategy
                        logger.info(f"Recovery successful with strategy: {strategy.value}")
                        break
                    else:
                        logger.warning(f"Recovery strategy {strategy.value} failed")
                        
                except Exception as recovery_error:
                    logger.error(f"Recovery strategy {strategy.value} raised exception: {recovery_error}")
        
        recovery_time = time.time() - start_time
        
        # Record the error (no recovery attempt)
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            error_type=error_type,
            severity=severity,
            message=str(error),
            traceback_str=traceback.format_exc(),
            recovery_strategy=recovery_strategy_used,  # Always None now
            recovery_success=False,  # Always False - we don't recover
            recovery_time=recovery_time,
            context=context,
            session_id=context.get('session_id'),
            epoch=context.get('epoch'),
            batch=context.get('batch')
        )
        
        with self._lock:
            self.error_history.append(error_record)
            self._update_recovery_stats(error_record)
        
        # Notify callbacks (for logging/monitoring only)
        for callback in self.recovery_callbacks:
            try:
                callback(error_record)
            except Exception as e:
                logger.error(f"Error notification callback failed: {e}")
                raise RuntimeError(f"Callback failure during error handling: {e}")
        
        if self.fail_hard:
            # Always re-raise the original error for hard failure
            logger.error(f"Re-raising original error for hard failure")
            raise error
        else:
            # Legacy behavior - return success/failure status
            return recovery_success
    
    def _diagnose_recovery_strategy(self, strategy: RecoveryStrategy, 
                                  error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose what a recovery strategy WOULD do (without executing it).
        
        This is for diagnostic purposes only - no actual recovery is performed.
        Returns a dictionary describing what would be done.
        """
        diagnosis = {'strategy': strategy.value, 'would_perform': []}
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                diagnosis['would_perform'].append('Retry the failed operation')
                diagnosis['recommended_action'] = 'Check if error is transient'
            
            elif strategy == RecoveryStrategy.REDUCE_BATCH_SIZE:
                if 'batch_size' in context:
                    current_batch_size = _extract_primitive_parameter(context, 'batch_size', int)
                    new_batch_size = max(1, current_batch_size // 2)
                    diagnosis['would_perform'].append(f'Reduce batch size from {current_batch_size} to {new_batch_size}')
                else:
                    diagnosis['would_perform'].append(f'Reduce batch size (current size unknown)')
                diagnosis['recommended_action'] = 'Manually reduce batch size if OOM errors persist'
            
            elif strategy == RecoveryStrategy.REDUCE_LEARNING_RATE:
                if 'learning_rate' in context:
                    current_lr = _extract_primitive_parameter(context, 'learning_rate', float)
                    new_lr = max(1e-8, current_lr * 0.5)
                    diagnosis['would_perform'].append(f'Reduce learning rate from {current_lr} to {new_lr}')
                else:
                    diagnosis['would_perform'].append(f'Reduce learning rate (current rate unknown)')
                diagnosis['recommended_action'] = 'Check gradient magnitudes and adjust LR manually'
            
            elif strategy == RecoveryStrategy.GRADIENT_CLIPPING:
                if 'gradient_clipping' in context:
                    current_clipping = _extract_primitive_parameter(context, 'gradient_clipping', float)
                    new_clipping = min(current_clipping, 0.5)
                    diagnosis['would_perform'].append(f'Set gradient clipping from {current_clipping} to {new_clipping}')
                else:
                    diagnosis['would_perform'].append(f'Enable gradient clipping (current setting unknown)')
                diagnosis['recommended_action'] = 'Enable gradient clipping if experiencing gradient explosion'
            
            elif strategy == RecoveryStrategy.CLEAR_CACHE:
                diagnosis['would_perform'].append('Clear CUDA cache and run garbage collection')
                diagnosis['recommended_action'] = 'Manually clear cache or restart with smaller batch'
            
            elif strategy == RecoveryStrategy.ROLLBACK_STATE:
                diagnosis['would_perform'].append('Rollback to previous training state')
                diagnosis['recommended_action'] = 'Load previous checkpoint manually'
            
            elif strategy == RecoveryStrategy.ROLLBACK_CHECKPOINT:
                diagnosis['would_perform'].append('Restore from last checkpoint')
                diagnosis['recommended_action'] = 'Manually restore from checkpoint and investigate root cause'
            
            elif strategy == RecoveryStrategy.SKIP_BATCH:
                diagnosis['would_perform'].append('Skip the current problematic batch')
                diagnosis['recommended_action'] = 'Inspect batch data for corruption or outliers'
            
            elif strategy == RecoveryStrategy.RESTART_TRAINING:
                diagnosis['would_perform'].append('Restart training from beginning')
                diagnosis['recommended_action'] = 'Fix root cause before restarting training'
            
            elif strategy == RecoveryStrategy.EMERGENCY_STOP:
                diagnosis['would_perform'].append('Emergency stop - critical failure')
                diagnosis['recommended_action'] = 'Investigate critical failure before any restart'
            
        except Exception as e:
            diagnosis['diagnosis_error'] = str(e)
        
        return diagnosis
    
    def _execute_recovery_strategy_legacy(self, strategy: RecoveryStrategy, 
                                  error: Exception, context: Dict[str, Any]) -> bool:
        """[LEGACY] Execute a specific recovery strategy.
        
        This method should only be used when fail_hard=False (not recommended).
        For production use, errors should fail hard with diagnostics.
        """
        try:
            if strategy == RecoveryStrategy.RETRY:
                return True  # Signal that retry should be attempted
            
            elif strategy == RecoveryStrategy.REDUCE_BATCH_SIZE:
                if 'batch_size' not in context:
                    raise ValueError("HARD FAILURE: Cannot reduce batch_size - parameter not provided in context")
                current_batch_size = _extract_primitive_parameter(context, 'batch_size', int)
                new_batch_size = max(1, current_batch_size // 2)
                context['suggested_batch_size'] = int(new_batch_size)
                context['batch_size_reduced'] = True
                logger.info(f"Reduced batch size from {current_batch_size} to {new_batch_size}")
                return True
            
            elif strategy == RecoveryStrategy.REDUCE_LEARNING_RATE:
                if 'learning_rate' not in context:
                    raise ValueError("HARD FAILURE: Cannot reduce learning_rate - parameter not provided in context")
                current_lr = _extract_primitive_parameter(context, 'learning_rate', float)
                new_lr = max(1e-8, current_lr * 0.5)
                context['suggested_learning_rate'] = float(new_lr)
                context['learning_rate_reduced'] = True
                logger.info(f"Reduced learning rate from {current_lr} to {new_lr}")
                return True
            
            elif strategy == RecoveryStrategy.GRADIENT_CLIPPING:
                if 'gradient_clipping' not in context:
                    raise ValueError("HARD FAILURE: Cannot set gradient_clipping - parameter not provided in context")
                current_clipping = _extract_primitive_parameter(context, 'gradient_clipping', float)
                new_clipping = min(current_clipping, 0.5)
                context['suggested_gradient_clipping'] = float(new_clipping)
                context['gradient_clipping_enabled'] = True
                logger.info(f"Set gradient clipping from {current_clipping} to {new_clipping}")
                return True
            
            elif strategy == RecoveryStrategy.CLEAR_CACHE:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except ImportError as e:
                    logger.error(f"PyTorch not available for CUDA cache clearing: {e}")
                    raise RuntimeError(f"CUDA cache clearing failed - PyTorch not available: {e}")
                
                import gc
                gc.collect()
                context['cache_cleared'] = True
                logger.info("Cleared memory cache")
                return True
            
            elif strategy == RecoveryStrategy.ROLLBACK_STATE:
                rolled_back_state = self.state_rollback.rollback(1)
                if rolled_back_state:
                    context['rolled_back_state'] = rolled_back_state
                    return True
                return False
            
            elif strategy == RecoveryStrategy.ROLLBACK_CHECKPOINT:
                session_id = context.get('session_id')
                if session_id:
                    checkpoints = self.checkpoint_recovery.list_checkpoints(session_id)
                    if checkpoints:
                        latest_checkpoint = checkpoints[0]
                        restored = self.checkpoint_recovery.restore_checkpoint(latest_checkpoint.checkpoint_id)
                        if restored:
                            context['restored_checkpoint'] = restored
                            return True
                return False
            
            elif strategy == RecoveryStrategy.SKIP_BATCH:
                context['skip_current_batch'] = True
                return True
            
            elif strategy == RecoveryStrategy.RESTART_TRAINING:
                context['restart_training'] = True
                return True
            
            elif strategy == RecoveryStrategy.EMERGENCY_STOP:
                context['emergency_stop'] = True
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing recovery strategy {strategy.value}: {e}")
            return False
    
    def _update_recovery_stats(self, error_record: ErrorRecord):
        """Update recovery statistics."""
        self.recovery_stats['total_errors'] += 1
        
        if error_record.recovery_strategy:
            self.recovery_stats['total_recoveries'] += 1
            
            strategy_name = error_record.recovery_strategy.value
            if strategy_name not in self.recovery_stats['recovery_strategies_used']:
                self.recovery_stats['recovery_strategies_used'][strategy_name] = 0
            self.recovery_stats['recovery_strategies_used'][strategy_name] += 1
            
            if error_record.recovery_success:
                self.recovery_stats['successful_recoveries'] += 1
            else:
                self.recovery_stats['failed_recoveries'] += 1
        
        error_type_name = error_record.error_type.value
        if error_type_name not in self.recovery_stats['error_types_seen']:
            self.recovery_stats['error_types_seen'][error_type_name] = 0
        self.recovery_stats['error_types_seen'][error_type_name] += 1
        
        # Update average recovery time
        total_time = sum(record.recovery_time for record in self.error_history 
                        if record.recovery_strategy)
        total_recoveries = len([r for r in self.error_history if r.recovery_strategy])
        
        if total_recoveries > 0:
            self.recovery_stats['average_recovery_time'] = total_time / total_recoveries
    
    def register_recovery_callback(self, callback: Callable[[ErrorRecord], None]):
        """Register a callback to be called after error recovery attempts."""
        self.recovery_callbacks.append(callback)
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        with self._lock:
            stats = self.recovery_stats.copy()
            
            # Add success rate
            if stats['total_recoveries'] > 0:
                stats['recovery_success_rate'] = stats['successful_recoveries'] / stats['total_recoveries']
            else:
                stats['recovery_success_rate'] = 0.0
            
            return stats
    
    def get_error_history(self, session_id: str = None, limit: int = 100) -> List[ErrorRecord]:
        """Get error history."""
        with self._lock:
            history = self.error_history
            
            if session_id:
                history = [record for record in history if record.session_id == session_id]
            
            return history[-limit:] if history else []
    
    def export_recovery_report(self, output_file: str = None) -> Dict[str, Any]:
        """Export comprehensive recovery report."""
        stats = self.get_recovery_stats()
        history = self.get_error_history()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'statistics': stats,
            'error_history': [
                {
                    'timestamp': record.timestamp.isoformat(),
                    'error_type': record.error_type.value,
                    'severity': record.severity.value,
                    'message': record.message,
                    'recovery_strategy': record.recovery_strategy.value if record.recovery_strategy else None,
                    'recovery_success': record.recovery_success,
                    'recovery_time': record.recovery_time,
                    'session_id': record.session_id,
                    'epoch': record.epoch,
                    'batch': record.batch
                }
                for record in history
            ],
            'checkpoints': [
                {
                    'checkpoint_id': cp.checkpoint_id,
                    'timestamp': cp.timestamp.isoformat(),
                    'session_id': cp.session_id,
                    'epoch': cp.epoch,
                    'batch': cp.batch,
                    'size_bytes': cp.size_bytes
                }
                for cp in self.checkpoint_recovery.list_checkpoints()
            ]
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report

def integrate_error_recovery(training_manager, progress_tracker=None, 
                           checkpoint_dir: str = ".checkpoints",
                           max_checkpoints: int = 10,
                           max_state_history: int = 50,
                           fail_hard: bool = True) -> ErrorRecoveryManager:
    """
    Integrate error recovery system with existing training infrastructure.
    
    NOTE: By default (fail_hard=True), this system will NOT attempt automatic recovery.
    Instead, it provides comprehensive error diagnostics and fails fast to ensure
    problems are properly addressed rather than masked.
    
    Args:
        training_manager: TrainingManager instance
        progress_tracker: Optional ProgressTracker instance
        checkpoint_dir: Directory for storing recovery checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
        max_state_history: Maximum number of state snapshots to keep
        fail_hard: If True (default), errors fail immediately with diagnostics.
                  If False, attempts automatic recovery (NOT RECOMMENDED).
    
    Returns:
        ErrorRecoveryManager instance
    """
    # Create error recovery manager
    recovery_manager = ErrorRecoveryManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=max_checkpoints,
        max_state_history=max_state_history,
        fail_hard=fail_hard
    )
    
    if fail_hard:
        logger.info("Error recovery system configured for HARD FAILURE mode (recommended)")
        logger.info("Errors will fail fast with comprehensive diagnostics")
    else:
        logger.warning("Error recovery system configured for automatic recovery (NOT RECOMMENDED)")
        logger.warning("Consider using fail_hard=True for production systems")
    
    # Set up integration with training manager
    if hasattr(training_manager, '_error_recovery_manager'):
        logger.warning("Error recovery already integrated with training manager")
    else:
        training_manager._error_recovery_manager = recovery_manager
    
    # Set up integration with progress tracker if provided
    if progress_tracker:
        def progress_error_callback(error_record: ErrorRecord):
            """Callback to report errors to progress tracker."""
            if hasattr(progress_tracker, 'report_error'):
                progress_tracker.report_error(
                    error_type=error_record.error_type.value,
                    message=error_record.message,
                    recovery_attempted=error_record.recovery_strategy is not None,
                    recovery_success=error_record.recovery_success
                )
        
        recovery_manager.register_recovery_callback(progress_error_callback)
    
    logger.info("Error recovery system successfully integrated")
    return recovery_manager

# Export main classes and functions
__all__ = [
    'ErrorType', 'ErrorSeverity', 'RecoveryStrategy',
    'ErrorRecord', 'RecoveryCheckpoint',
    'ErrorClassifier', 'CheckpointRecovery', 'StateRollback', 'ErrorRecoveryManager',
    'integrate_error_recovery'
]