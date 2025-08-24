#!/usr/bin/env python3
"""
Comprehensive test suite for ErrorRecoverySystem.

This test suite comprehensively tests all aspects of the error recovery system
including error classification, recovery strategies, checkpoint management,
state rollback, and integration capabilities.
"""

import unittest
import tempfile
import shutil
import os
import sys
import json
import time
import threading
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from error_recovery_system import (
    ErrorType, ErrorSeverity, RecoveryStrategy,
    ErrorRecord, RecoveryCheckpoint,
    ErrorClassifier, CheckpointRecovery, StateRollback, ErrorRecoveryManager,
    integrate_error_recovery, _extract_primitive_parameter
)


class TestErrorTypeEnum(unittest.TestCase):
    """Test ErrorType enum completeness."""
    
    def test_error_type_enum_values(self):
        """Test that all expected error types are defined."""
        expected_types = [
            'MEMORY_ERROR', 'CUDA_ERROR', 'DATA_ERROR', 'MODEL_ERROR',
            'OPTIMIZATION_ERROR', 'IO_ERROR', 'TIMEOUT_ERROR', 'VALIDATION_ERROR',
            'CHECKPOINT_ERROR', 'UNKNOWN_ERROR'
        ]
        
        for error_type in expected_types:
            self.assertTrue(hasattr(ErrorType, error_type))
            self.assertIsInstance(getattr(ErrorType, error_type), ErrorType)
    
    def test_error_type_string_values(self):
        """Test error type string representations."""
        self.assertEqual(ErrorType.MEMORY_ERROR.value, "memory_error")
        self.assertEqual(ErrorType.CUDA_ERROR.value, "cuda_error")
        self.assertEqual(ErrorType.UNKNOWN_ERROR.value, "unknown_error")


class TestErrorSeverityEnum(unittest.TestCase):
    """Test ErrorSeverity enum completeness."""
    
    def test_error_severity_enum_values(self):
        """Test that all expected severity levels are defined."""
        expected_severities = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'WARNING']
        
        for severity in expected_severities:
            self.assertTrue(hasattr(ErrorSeverity, severity))
            self.assertIsInstance(getattr(ErrorSeverity, severity), ErrorSeverity)
    
    def test_error_severity_string_values(self):
        """Test severity string representations."""
        self.assertEqual(ErrorSeverity.CRITICAL.value, "critical")
        self.assertEqual(ErrorSeverity.HIGH.value, "high")
        self.assertEqual(ErrorSeverity.MEDIUM.value, "medium")


class TestRecoveryStrategyEnum(unittest.TestCase):
    """Test RecoveryStrategy enum completeness."""
    
    def test_recovery_strategy_enum_values(self):
        """Test that all expected recovery strategies are defined."""
        expected_strategies = [
            'RETRY', 'REDUCE_BATCH_SIZE', 'REDUCE_LEARNING_RATE', 'GRADIENT_CLIPPING',
            'CLEAR_CACHE', 'ROLLBACK_CHECKPOINT', 'ROLLBACK_STATE', 'RESTART_TRAINING',
            'SKIP_BATCH', 'EMERGENCY_STOP'
        ]
        
        for strategy in expected_strategies:
            self.assertTrue(hasattr(RecoveryStrategy, strategy))
            self.assertIsInstance(getattr(RecoveryStrategy, strategy), RecoveryStrategy)


class TestErrorRecord(unittest.TestCase):
    """Test ErrorRecord dataclass."""
    
    def test_error_record_creation(self):
        """Test ErrorRecord creation with all fields."""
        timestamp = datetime.now()
        context = {'test': 'data'}
        
        record = ErrorRecord(
            timestamp=timestamp,
            error_type=ErrorType.MEMORY_ERROR,
            severity=ErrorSeverity.HIGH,
            message="Test error",
            traceback_str="Test traceback",
            recovery_strategy=RecoveryStrategy.RETRY,
            recovery_success=True,
            recovery_time=1.5,
            context=context,
            session_id="test_session",
            epoch=5,
            batch=10
        )
        
        self.assertEqual(record.timestamp, timestamp)
        self.assertEqual(record.error_type, ErrorType.MEMORY_ERROR)
        self.assertEqual(record.severity, ErrorSeverity.HIGH)
        self.assertEqual(record.message, "Test error")
        self.assertEqual(record.recovery_strategy, RecoveryStrategy.RETRY)
        self.assertTrue(record.recovery_success)
        self.assertEqual(record.recovery_time, 1.5)
        self.assertEqual(record.context, context)
        self.assertEqual(record.session_id, "test_session")
        self.assertEqual(record.epoch, 5)
        self.assertEqual(record.batch, 10)
    
    def test_error_record_optional_fields(self):
        """Test ErrorRecord with minimal required fields."""
        timestamp = datetime.now()
        
        record = ErrorRecord(
            timestamp=timestamp,
            error_type=ErrorType.UNKNOWN_ERROR,
            severity=ErrorSeverity.MEDIUM,
            message="Test error",
            traceback_str="Test traceback",
            recovery_strategy=None,
            recovery_success=False,
            recovery_time=0.0,
            context={}
        )
        
        self.assertEqual(record.timestamp, timestamp)
        self.assertIsNone(record.recovery_strategy)
        self.assertFalse(record.recovery_success)
        self.assertIsNone(record.session_id)
        self.assertIsNone(record.epoch)
        self.assertIsNone(record.batch)


class TestRecoveryCheckpoint(unittest.TestCase):
    """Test RecoveryCheckpoint dataclass."""
    
    def test_recovery_checkpoint_creation(self):
        """Test RecoveryCheckpoint creation with all fields."""
        timestamp = datetime.now()
        training_state = {'epoch': 5, 'loss': 0.1}
        metrics = {'accuracy': 0.95}
        
        checkpoint = RecoveryCheckpoint(
            checkpoint_id="test_checkpoint",
            timestamp=timestamp,
            session_id="test_session",
            epoch=5,
            batch=100,
            model_state="/path/to/model.pkl",
            optimizer_state="/path/to/optimizer.pkl",
            training_state=training_state,
            metrics=metrics,
            checksum="abc123",
            size_bytes=1024
        )
        
        self.assertEqual(checkpoint.checkpoint_id, "test_checkpoint")
        self.assertEqual(checkpoint.timestamp, timestamp)
        self.assertEqual(checkpoint.session_id, "test_session")
        self.assertEqual(checkpoint.epoch, 5)
        self.assertEqual(checkpoint.batch, 100)
        self.assertEqual(checkpoint.model_state, "/path/to/model.pkl")
        self.assertEqual(checkpoint.optimizer_state, "/path/to/optimizer.pkl")
        self.assertEqual(checkpoint.training_state, training_state)
        self.assertEqual(checkpoint.metrics, metrics)
        self.assertEqual(checkpoint.checksum, "abc123")
        self.assertEqual(checkpoint.size_bytes, 1024)


class TestExtractPrimitiveParameter(unittest.TestCase):
    """Test _extract_primitive_parameter utility function."""
    
    def test_extract_simple_value(self):
        """Test extracting simple values from context."""
        context = {'batch_size': 32, 'learning_rate': 0.001}
        
        batch_size = _extract_primitive_parameter(context, 'batch_size', 16, int)
        learning_rate = _extract_primitive_parameter(context, 'learning_rate', 0.01, float)
        
        self.assertEqual(batch_size, 32)
        self.assertEqual(learning_rate, 0.001)
    
    def test_extract_default_value(self):
        """Test using default value when parameter is missing."""
        context = {}
        
        batch_size = _extract_primitive_parameter(context, 'batch_size', 16, int)
        learning_rate = _extract_primitive_parameter(context, 'learning_rate', 0.01, float)
        
        self.assertEqual(batch_size, 16)
        self.assertEqual(learning_rate, 0.01)
    
    def test_extract_from_dictionary(self):
        """Test extracting from nested dictionary values."""
        context = {
            'batch_size': {'value': 64},
            'learning_rate': {'suggested_value': 0.005}
        }
        
        batch_size = _extract_primitive_parameter(context, 'batch_size', 16, int)
        learning_rate = _extract_primitive_parameter(context, 'learning_rate', 0.01, float)
        
        self.assertEqual(batch_size, 64)
        self.assertEqual(learning_rate, 0.005)
    
    def test_extract_from_list(self):
        """Test extracting from list values."""
        context = {
            'batch_size': [128, 64, 32],
            'learning_rate': [0.002]
        }
        
        batch_size = _extract_primitive_parameter(context, 'batch_size', 16, int)
        learning_rate = _extract_primitive_parameter(context, 'learning_rate', 0.01, float)
        
        self.assertEqual(batch_size, 128)
        self.assertEqual(learning_rate, 0.002)
    
    def test_extract_invalid_values(self):
        """Test handling of invalid values."""
        context = {
            'batch_size': -5,  # Invalid negative value
            'learning_rate': 'invalid',  # Invalid string value
            'timeout': 0  # Invalid zero value
        }
        
        batch_size = _extract_primitive_parameter(context, 'batch_size', 16, int)
        learning_rate = _extract_primitive_parameter(context, 'learning_rate', 0.01, float)
        timeout = _extract_primitive_parameter(context, 'timeout', 30, int)
        
        self.assertEqual(batch_size, 16)  # Should use default
        self.assertEqual(learning_rate, 0.01)  # Should use default
        self.assertEqual(timeout, 30)  # Should use default
    
    def test_extract_type_conversion(self):
        """Test type conversion during extraction."""
        context = {
            'batch_size': '32',  # String to int
            'learning_rate': '0.001',  # String to float
        }
        
        batch_size = _extract_primitive_parameter(context, 'batch_size', 16, int)
        learning_rate = _extract_primitive_parameter(context, 'learning_rate', 0.01, float)
        
        self.assertEqual(batch_size, 32)
        self.assertEqual(learning_rate, 0.001)
        self.assertIsInstance(batch_size, int)
        self.assertIsInstance(learning_rate, float)


class TestErrorClassifier(unittest.TestCase):
    """Test ErrorClassifier functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = ErrorClassifier()
    
    def test_error_classifier_initialization(self):
        """Test ErrorClassifier initialization."""
        self.assertIsInstance(self.classifier.error_patterns, dict)
        self.assertIsInstance(self.classifier.severity_indicators, dict)
        
        # Check that all error types have patterns
        for error_type in ErrorType:
            self.assertIn(error_type, self.classifier.error_patterns)
        
        # Check that all severity levels have indicators
        for severity in ErrorSeverity:
            self.assertIn(severity, self.classifier.severity_indicators)
    
    def test_classify_memory_error(self):
        """Test classification of memory errors."""
        error = RuntimeError("CUDA out of memory")
        error_type, severity = self.classifier.classify_error(error)
        
        self.assertEqual(error_type, ErrorType.MEMORY_ERROR)
        self.assertIn(severity, [ErrorSeverity.MEDIUM, ErrorSeverity.HIGH])
    
    def test_classify_cuda_error(self):
        """Test classification of CUDA errors."""
        error = RuntimeError("CUDA error: device-side assert triggered")
        error_type, severity = self.classifier.classify_error(error)
        
        self.assertEqual(error_type, ErrorType.CUDA_ERROR)
    
    def test_classify_data_error(self):
        """Test classification of data errors."""
        error = ValueError("Input shape mismatch")
        error_type, severity = self.classifier.classify_error(error)
        
        self.assertEqual(error_type, ErrorType.DATA_ERROR)
    
    def test_classify_unknown_error(self):
        """Test classification of unknown errors."""
        error = Exception("Some random error")
        error_type, severity = self.classifier.classify_error(error)
        
        self.assertEqual(error_type, ErrorType.UNKNOWN_ERROR)
    
    def test_classify_with_context(self):
        """Test error classification with context."""
        error = RuntimeError("Some error")
        context = {'consecutive_failures': 5, 'training_progress': 0.9}
        
        error_type, severity = self.classifier.classify_error(error, context)
        
        # High consecutive failures should escalate severity
        self.assertIn(severity, [ErrorSeverity.HIGH, ErrorSeverity.MEDIUM])
    
    def test_get_recovery_strategies_memory_error(self):
        """Test recovery strategies for memory errors."""
        strategies = self.classifier.get_recovery_strategies(
            ErrorType.MEMORY_ERROR, ErrorSeverity.MEDIUM
        )
        
        self.assertIn(RecoveryStrategy.CLEAR_CACHE, strategies)
        self.assertIn(RecoveryStrategy.REDUCE_BATCH_SIZE, strategies)
    
    def test_get_recovery_strategies_critical_error(self):
        """Test recovery strategies for critical errors."""
        strategies = self.classifier.get_recovery_strategies(
            ErrorType.UNKNOWN_ERROR, ErrorSeverity.CRITICAL
        )
        
        self.assertEqual(strategies, [RecoveryStrategy.EMERGENCY_STOP])
    
    def test_get_recovery_strategies_with_failures(self):
        """Test recovery strategy escalation with consecutive failures."""
        context = {'consecutive_failures': 6}
        
        strategies = self.classifier.get_recovery_strategies(
            ErrorType.DATA_ERROR, ErrorSeverity.MEDIUM, context
        )
        
        self.assertEqual(strategies, [RecoveryStrategy.EMERGENCY_STOP])
    
    def test_get_recovery_strategies_escalation(self):
        """Test recovery strategy escalation."""
        context = {'consecutive_failures': 3}
        
        strategies = self.classifier.get_recovery_strategies(
            ErrorType.DATA_ERROR, ErrorSeverity.MEDIUM, context
        )
        
        self.assertIn(RecoveryStrategy.RESTART_TRAINING, strategies)


class TestCheckpointRecovery(unittest.TestCase):
    """Test CheckpointRecovery functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_recovery = CheckpointRecovery(
            checkpoint_dir=self.temp_dir, max_checkpoints=3
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_checkpoint_recovery_initialization(self):
        """Test CheckpointRecovery initialization."""
        self.assertEqual(self.checkpoint_recovery.max_checkpoints, 3)
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertIsInstance(self.checkpoint_recovery.checkpoints, dict)
    
    def test_create_checkpoint_basic(self):
        """Test basic checkpoint creation."""
        checkpoint_id = self.checkpoint_recovery.create_checkpoint(
            session_id="test_session",
            epoch=1,
            batch=10,
            training_state={'loss': 0.5},
            metrics={'accuracy': 0.8}
        )
        
        self.assertIsInstance(checkpoint_id, str)
        self.assertIn(checkpoint_id, self.checkpoint_recovery.checkpoints)
        
        checkpoint = self.checkpoint_recovery.checkpoints[checkpoint_id]
        self.assertEqual(checkpoint.session_id, "test_session")
        self.assertEqual(checkpoint.epoch, 1)
        self.assertEqual(checkpoint.batch, 10)
    
    def test_create_checkpoint_with_states(self):
        """Test checkpoint creation with model and optimizer states."""
        model_state = {'layer1.weight': [1, 2, 3]}
        optimizer_state = {'state': {}, 'param_groups': []}
        
        checkpoint_id = self.checkpoint_recovery.create_checkpoint(
            session_id="test_session",
            epoch=1,
            batch=10,
            model_state=model_state,
            optimizer_state=optimizer_state,
            training_state={'loss': 0.5},
            metrics={'accuracy': 0.8}
        )
        
        checkpoint = self.checkpoint_recovery.checkpoints[checkpoint_id]
        self.assertIsNotNone(checkpoint.model_state)
        self.assertIsNotNone(checkpoint.optimizer_state)
        
        # Verify files were created
        model_path = Path(checkpoint.model_state)
        optimizer_path = Path(checkpoint.optimizer_state)
        self.assertTrue(model_path.exists())
        self.assertTrue(optimizer_path.exists())
    
    def test_restore_checkpoint(self):
        """Test checkpoint restoration."""
        model_state = {'test': 'data'}
        
        checkpoint_id = self.checkpoint_recovery.create_checkpoint(
            session_id="test_session",
            epoch=1,
            batch=10,
            model_state=model_state
        )
        
        restored = self.checkpoint_recovery.restore_checkpoint(checkpoint_id)
        
        self.assertIsNotNone(restored)
        self.assertEqual(restored.checkpoint_id, checkpoint_id)
        self.assertEqual(restored.session_id, "test_session")
    
    def test_restore_nonexistent_checkpoint(self):
        """Test restoration of nonexistent checkpoint."""
        restored = self.checkpoint_recovery.restore_checkpoint("nonexistent")
        self.assertIsNone(restored)
    
    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        # Create multiple checkpoints
        checkpoint1 = self.checkpoint_recovery.create_checkpoint("session1", 1, 10)
        checkpoint2 = self.checkpoint_recovery.create_checkpoint("session2", 2, 20)
        checkpoint3 = self.checkpoint_recovery.create_checkpoint("session1", 3, 30)
        
        # Test listing all checkpoints
        all_checkpoints = self.checkpoint_recovery.list_checkpoints()
        self.assertEqual(len(all_checkpoints), 3)
        
        # Test filtering by session
        session1_checkpoints = self.checkpoint_recovery.list_checkpoints("session1")
        self.assertEqual(len(session1_checkpoints), 2)
        
        session2_checkpoints = self.checkpoint_recovery.list_checkpoints("session2")
        self.assertEqual(len(session2_checkpoints), 1)
    
    def test_delete_checkpoint(self):
        """Test checkpoint deletion."""
        checkpoint_id = self.checkpoint_recovery.create_checkpoint(
            "test_session", 1, 10
        )
        
        # Verify checkpoint exists
        self.assertIn(checkpoint_id, self.checkpoint_recovery.checkpoints)
        
        # Delete checkpoint
        success = self.checkpoint_recovery.delete_checkpoint(checkpoint_id)
        
        self.assertTrue(success)
        self.assertNotIn(checkpoint_id, self.checkpoint_recovery.checkpoints)
    
    def test_delete_nonexistent_checkpoint(self):
        """Test deletion of nonexistent checkpoint."""
        success = self.checkpoint_recovery.delete_checkpoint("nonexistent")
        self.assertFalse(success)
    
    def test_checkpoint_cleanup(self):
        """Test automatic cleanup of old checkpoints."""
        # Create more checkpoints than the limit
        checkpoint_ids = []
        for i in range(5):
            checkpoint_id = self.checkpoint_recovery.create_checkpoint(
                f"session{i}", i, i * 10
            )
            checkpoint_ids.append(checkpoint_id)
            time.sleep(0.01)  # Ensure different timestamps
        
        # Should only keep the most recent 3 checkpoints
        self.assertEqual(len(self.checkpoint_recovery.checkpoints), 3)
        
        # Verify the newest checkpoints are kept
        remaining_checkpoints = list(self.checkpoint_recovery.checkpoints.keys())
        for checkpoint_id in checkpoint_ids[-3:]:
            self.assertIn(checkpoint_id, remaining_checkpoints)
    
    def test_checkpoint_integrity_check(self):
        """Test checkpoint integrity checking."""
        checkpoint_id = self.checkpoint_recovery.create_checkpoint(
            "test_session", 1, 10, training_state={'test': 'data'}
        )
        
        # Verify restoration works with correct checksum
        restored = self.checkpoint_recovery.restore_checkpoint(checkpoint_id)
        self.assertIsNotNone(restored)
        
        # Corrupt the checkpoint by modifying the checksum
        checkpoint = self.checkpoint_recovery.checkpoints[checkpoint_id]
        checkpoint.checksum = "invalid_checksum"
        
        # Restoration should fail with corrupted checksum
        restored = self.checkpoint_recovery.restore_checkpoint(checkpoint_id)
        self.assertIsNone(restored)
    
    def test_checkpoint_index_persistence(self):
        """Test checkpoint index persistence."""
        # Create checkpoint
        checkpoint_id = self.checkpoint_recovery.create_checkpoint(
            "test_session", 1, 10
        )
        
        # Create new instance pointing to same directory
        new_recovery = CheckpointRecovery(checkpoint_dir=self.temp_dir)
        
        # Should load existing checkpoints
        self.assertIn(checkpoint_id, new_recovery.checkpoints)


class TestStateRollback(unittest.TestCase):
    """Test StateRollback functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_rollback = StateRollback(max_history=5)
    
    def test_state_rollback_initialization(self):
        """Test StateRollback initialization."""
        self.assertEqual(self.state_rollback.max_history, 5)
        self.assertEqual(len(self.state_rollback.state_history), 0)
    
    def test_save_state(self):
        """Test saving state."""
        state = {'epoch': 1, 'loss': 0.5}
        metadata = {'timestamp': 'test'}
        
        self.state_rollback.save_state(state, metadata)
        
        self.assertEqual(len(self.state_rollback.state_history), 1)
        saved_record = self.state_rollback.state_history[0]
        
        self.assertEqual(saved_record['state'], state)
        self.assertEqual(saved_record['metadata'], metadata)
        self.assertIsInstance(saved_record['timestamp'], datetime)
    
    def test_rollback_single_step(self):
        """Test rolling back one step."""
        # Save multiple states
        states = [
            {'epoch': 1, 'loss': 0.5},
            {'epoch': 2, 'loss': 0.4},
            {'epoch': 3, 'loss': 0.3}
        ]
        
        for state in states:
            self.state_rollback.save_state(state)
        
        # Rollback one step
        rolled_back = self.state_rollback.rollback(1)
        
        # Should get the second-to-last state (epoch 2)
        self.assertIsNotNone(rolled_back)
        self.assertEqual(rolled_back['epoch'], 2)
        
        # History should be shortened
        self.assertEqual(len(self.state_rollback.state_history), 2)
    
    def test_rollback_multiple_steps(self):
        """Test rolling back multiple steps."""
        # Save multiple states
        for i in range(5):
            self.state_rollback.save_state({'epoch': i + 1})
        
        # Rollback 3 steps
        rolled_back = self.state_rollback.rollback(3)
        
        # Should get epoch 2 (5 - 3 = 2)
        self.assertIsNotNone(rolled_back)
        self.assertEqual(rolled_back['epoch'], 2)
        self.assertEqual(len(self.state_rollback.state_history), 2)
    
    def test_rollback_too_many_steps(self):
        """Test rolling back more steps than available."""
        # Save only 2 states
        self.state_rollback.save_state({'epoch': 1})
        self.state_rollback.save_state({'epoch': 2})
        
        # Try to rollback 5 steps
        rolled_back = self.state_rollback.rollback(5)
        
        self.assertIsNone(rolled_back)
    
    def test_rollback_no_history(self):
        """Test rollback with no history."""
        rolled_back = self.state_rollback.rollback(1)
        self.assertIsNone(rolled_back)
    
    def test_get_history(self):
        """Test getting state history."""
        # Save multiple states
        for i in range(10):
            self.state_rollback.save_state({'epoch': i + 1})
        
        # Get limited history
        history = self.state_rollback.get_history(3)
        self.assertEqual(len(history), 3)
        
        # Should be most recent states
        self.assertEqual(history[-1]['state']['epoch'], 10)
        self.assertEqual(history[-2]['state']['epoch'], 9)
    
    def test_clear_history(self):
        """Test clearing history."""
        # Save some states
        for i in range(3):
            self.state_rollback.save_state({'epoch': i + 1})
        
        self.assertEqual(len(self.state_rollback.state_history), 3)
        
        # Clear history
        self.state_rollback.clear_history()
        
        self.assertEqual(len(self.state_rollback.state_history), 0)
    
    def test_history_limit(self):
        """Test history size limit."""
        # Save more states than the limit
        for i in range(10):
            self.state_rollback.save_state({'epoch': i + 1})
        
        # Should only keep max_history states
        self.assertEqual(len(self.state_rollback.state_history), 5)
        
        # Should keep the most recent ones
        self.assertEqual(self.state_rollback.state_history[-1]['state']['epoch'], 10)
        self.assertEqual(self.state_rollback.state_history[0]['state']['epoch'], 6)


class TestErrorRecoveryManager(unittest.TestCase):
    """Test ErrorRecoveryManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.recovery_manager = ErrorRecoveryManager(
            checkpoint_dir=self.temp_dir,
            max_checkpoints=5,
            max_state_history=10
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_recovery_manager_initialization(self):
        """Test ErrorRecoveryManager initialization."""
        self.assertIsInstance(self.recovery_manager.classifier, ErrorClassifier)
        self.assertIsInstance(self.recovery_manager.checkpoint_recovery, CheckpointRecovery)
        self.assertIsInstance(self.recovery_manager.state_rollback, StateRollback)
        
        self.assertEqual(len(self.recovery_manager.error_history), 0)
        self.assertIsInstance(self.recovery_manager.recovery_stats, dict)
        self.assertEqual(len(self.recovery_manager.recovery_callbacks), 0)
    
    def test_handle_error_basic(self):
        """Test basic error handling."""
        error = RuntimeError("Test error")
        context = {'session_id': 'test', 'epoch': 1, 'batch': 10}
        
        success = self.recovery_manager.handle_error(error, context)
        
        # Should attempt recovery
        self.assertIsInstance(success, bool)
        
        # Should record the error
        self.assertEqual(len(self.recovery_manager.error_history), 1)
        error_record = self.recovery_manager.error_history[0]
        
        self.assertEqual(error_record.message, "Test error")
        self.assertEqual(error_record.session_id, "test")
        self.assertEqual(error_record.epoch, 1)
        self.assertEqual(error_record.batch, 10)
    
    def test_handle_memory_error(self):
        """Test handling memory errors."""
        error = RuntimeError("CUDA out of memory")
        context = {'batch_size': 32}
        
        success = self.recovery_manager.handle_error(error, context)
        
        # Check that suggested batch size reduction was set
        if 'suggested_batch_size' in context:
            self.assertLess(context['suggested_batch_size'], 32)
    
    def test_handle_optimization_error(self):
        """Test handling optimization errors."""
        error = ValueError("Loss is NaN")
        context = {'learning_rate': 0.001}
        
        success = self.recovery_manager.handle_error(error, context)
        
        # Check for suggested learning rate reduction
        if 'suggested_learning_rate' in context:
            self.assertLess(context['suggested_learning_rate'], 0.001)
    
    def test_recovery_callbacks(self):
        """Test recovery callbacks."""
        callback_called = []
        
        def test_callback(error_record):
            callback_called.append(error_record)
        
        self.recovery_manager.register_recovery_callback(test_callback)
        
        error = RuntimeError("Test error")
        self.recovery_manager.handle_error(error)
        
        self.assertEqual(len(callback_called), 1)
        self.assertEqual(callback_called[0].message, "Test error")
    
    def test_recovery_stats_update(self):
        """Test recovery statistics updates."""
        initial_stats = self.recovery_manager.get_recovery_stats()
        self.assertEqual(initial_stats['total_errors'], 0)
        
        # Handle some errors
        errors = [
            RuntimeError("Memory error"),
            ValueError("Data error"),
            Exception("Unknown error")
        ]
        
        for error in errors:
            self.recovery_manager.handle_error(error)
        
        stats = self.recovery_manager.get_recovery_stats()
        self.assertEqual(stats['total_errors'], 3)
        self.assertGreaterEqual(stats['total_recoveries'], 0)
    
    def test_get_error_history(self):
        """Test getting error history."""
        # Handle multiple errors
        for i in range(5):
            error = RuntimeError(f"Error {i}")
            context = {'session_id': f'session{i % 2}'}
            self.recovery_manager.handle_error(error, context)
        
        # Get all history
        all_history = self.recovery_manager.get_error_history()
        self.assertEqual(len(all_history), 5)
        
        # Get session-specific history
        session0_history = self.recovery_manager.get_error_history('session0')
        session1_history = self.recovery_manager.get_error_history('session1')
        
        self.assertEqual(len(session0_history), 3)  # Errors 0, 2, 4
        self.assertEqual(len(session1_history), 2)  # Errors 1, 3
        
        # Test limit
        limited_history = self.recovery_manager.get_error_history(limit=3)
        self.assertEqual(len(limited_history), 3)
    
    def test_export_recovery_report(self):
        """Test exporting recovery report."""
        # Create some error history and checkpoints
        error = RuntimeError("Test error")
        context = {'session_id': 'test'}
        self.recovery_manager.handle_error(error, context)
        
        self.recovery_manager.checkpoint_recovery.create_checkpoint(
            'test', 1, 10
        )
        
        # Export report
        report = self.recovery_manager.export_recovery_report()
        
        self.assertIn('generated_at', report)
        self.assertIn('statistics', report)
        self.assertIn('error_history', report)
        self.assertIn('checkpoints', report)
        
        self.assertEqual(len(report['error_history']), 1)
        self.assertEqual(len(report['checkpoints']), 1)
    
    def test_export_recovery_report_to_file(self):
        """Test exporting recovery report to file."""
        error = RuntimeError("Test error")
        self.recovery_manager.handle_error(error)
        
        output_file = os.path.join(self.temp_dir, "recovery_report.json")
        report = self.recovery_manager.export_recovery_report(output_file)
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Verify file contents
        with open(output_file, 'r') as f:
            file_report = json.load(f)
        
        self.assertEqual(report, file_report)
    
    def test_execute_recovery_strategy_retry(self):
        """Test executing retry recovery strategy."""
        error = RuntimeError("Test error")
        context = {}
        
        success = self.recovery_manager._execute_recovery_strategy(
            RecoveryStrategy.RETRY, error, context
        )
        
        self.assertTrue(success)
    
    def test_execute_recovery_strategy_reduce_batch_size(self):
        """Test executing batch size reduction strategy."""
        error = RuntimeError("Test error")
        context = {'batch_size': 64}
        
        success = self.recovery_manager._execute_recovery_strategy(
            RecoveryStrategy.REDUCE_BATCH_SIZE, error, context
        )
        
        self.assertTrue(success)
        self.assertIn('suggested_batch_size', context)
        self.assertEqual(context['suggested_batch_size'], 32)
    
    def test_execute_recovery_strategy_clear_cache(self):
        """Test executing clear cache strategy."""
        error = RuntimeError("Test error")
        context = {}
        
        success = self.recovery_manager._execute_recovery_strategy(
            RecoveryStrategy.CLEAR_CACHE, error, context
        )
        
        # Should succeed (garbage collection always works)
        self.assertTrue(success)
    
    def test_execute_recovery_strategy_rollback_state(self):
        """Test executing state rollback strategy."""
        # Save some state first
        self.recovery_manager.state_rollback.save_state({'test': 'data'})
        
        error = RuntimeError("Test error")
        context = {}
        
        success = self.recovery_manager._execute_recovery_strategy(
            RecoveryStrategy.ROLLBACK_STATE, error, context
        )
        
        self.assertTrue(success)
        self.assertIn('rolled_back_state', context)
    
    def test_execute_recovery_strategy_rollback_checkpoint(self):
        """Test executing checkpoint rollback strategy."""
        # Create a checkpoint first
        checkpoint_id = self.recovery_manager.checkpoint_recovery.create_checkpoint(
            'test_session', 1, 10
        )
        
        error = RuntimeError("Test error")
        context = {'session_id': 'test_session'}
        
        success = self.recovery_manager._execute_recovery_strategy(
            RecoveryStrategy.ROLLBACK_CHECKPOINT, error, context
        )
        
        self.assertTrue(success)
        self.assertIn('restored_checkpoint', context)
    
    def test_execute_recovery_strategy_skip_batch(self):
        """Test executing skip batch strategy."""
        error = RuntimeError("Test error")
        context = {}
        
        success = self.recovery_manager._execute_recovery_strategy(
            RecoveryStrategy.SKIP_BATCH, error, context
        )
        
        self.assertTrue(success)
        self.assertTrue(context['skip_current_batch'])
    
    def test_execute_recovery_strategy_emergency_stop(self):
        """Test executing emergency stop strategy."""
        error = RuntimeError("Test error")
        context = {}
        
        success = self.recovery_manager._execute_recovery_strategy(
            RecoveryStrategy.EMERGENCY_STOP, error, context
        )
        
        self.assertTrue(success)
        self.assertTrue(context['emergency_stop'])
    
    def test_thread_safety(self):
        """Test thread safety of recovery manager."""
        def handle_errors():
            for i in range(10):
                error = RuntimeError(f"Error {i}")
                self.recovery_manager.handle_error(error)
        
        # Run multiple threads concurrently
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=handle_errors)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have handled all errors without crashes
        self.assertEqual(len(self.recovery_manager.error_history), 30)


class TestIntegrateErrorRecovery(unittest.TestCase):
    """Test integrate_error_recovery function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.training_manager = Mock()
        self.progress_tracker = Mock()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_integrate_error_recovery_basic(self):
        """Test basic error recovery integration."""
        recovery_manager = integrate_error_recovery(
            self.training_manager,
            checkpoint_dir=self.temp_dir
        )
        
        self.assertIsInstance(recovery_manager, ErrorRecoveryManager)
        self.assertTrue(hasattr(self.training_manager, '_error_recovery_manager'))
        self.assertEqual(self.training_manager._error_recovery_manager, recovery_manager)
    
    def test_integrate_with_progress_tracker(self):
        """Test integration with progress tracker."""
        # Mock progress tracker with report_error method
        self.progress_tracker.report_error = Mock()
        
        recovery_manager = integrate_error_recovery(
            self.training_manager,
            self.progress_tracker,
            checkpoint_dir=self.temp_dir
        )
        
        # Handle an error to test callback
        error = RuntimeError("Test error")
        recovery_manager.handle_error(error)
        
        # Progress tracker should have been called
        self.progress_tracker.report_error.assert_called_once()
    
    def test_integrate_already_integrated(self):
        """Test integration when already integrated."""
        # First integration
        recovery_manager1 = integrate_error_recovery(
            self.training_manager,
            checkpoint_dir=self.temp_dir
        )
        
        # Second integration should log warning but continue
        recovery_manager2 = integrate_error_recovery(
            self.training_manager,
            checkpoint_dir=self.temp_dir
        )
        
        self.assertIsInstance(recovery_manager2, ErrorRecoveryManager)
    
    def test_integrate_custom_parameters(self):
        """Test integration with custom parameters."""
        recovery_manager = integrate_error_recovery(
            self.training_manager,
            self.progress_tracker,
            checkpoint_dir=self.temp_dir,
            max_checkpoints=20,
            max_state_history=100
        )
        
        self.assertEqual(recovery_manager.checkpoint_recovery.max_checkpoints, 20)
        self.assertEqual(recovery_manager.state_rollback.max_history, 100)


class TestErrorRecoverySystemComprehensive(unittest.TestCase):
    """Comprehensive end-to-end tests for the error recovery system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.recovery_manager = ErrorRecoveryManager(
            checkpoint_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_recovery_workflow_memory_error(self):
        """Test complete recovery workflow for memory errors."""
        # Simulate training state
        training_state = {'epoch': 5, 'batch': 100, 'loss': 0.3}
        self.recovery_manager.state_rollback.save_state(training_state)
        
        # Create checkpoint
        checkpoint_id = self.recovery_manager.checkpoint_recovery.create_checkpoint(
            'training_session', 5, 95, training_state=training_state
        )
        
        # Simulate memory error during training
        error = RuntimeError("CUDA out of memory")
        context = {
            'session_id': 'training_session',
            'epoch': 5,
            'batch': 100,
            'batch_size': 64
        }
        
        success = self.recovery_manager.handle_error(error, context)
        
        # Verify recovery was attempted
        self.assertIsInstance(success, bool)
        
        # Verify error was recorded
        error_history = self.recovery_manager.get_error_history()
        self.assertEqual(len(error_history), 1)
        self.assertEqual(error_history[0].error_type, ErrorType.MEMORY_ERROR)
        
        # Verify suggested batch size reduction
        if 'suggested_batch_size' in context:
            self.assertLess(context['suggested_batch_size'], 64)
    
    def test_full_recovery_workflow_optimization_error(self):
        """Test complete recovery workflow for optimization errors."""
        # Simulate NaN loss error
        error = ValueError("Loss is NaN")
        context = {
            'session_id': 'training_session',
            'epoch': 10,
            'batch': 50,
            'learning_rate': 0.01,
            'gradient_clipping': 1.0
        }
        
        success = self.recovery_manager.handle_error(error, context)
        
        # Verify error classification
        error_history = self.recovery_manager.get_error_history()
        self.assertEqual(error_history[0].error_type, ErrorType.OPTIMIZATION_ERROR)
        
        # Verify suggested parameter adjustments
        if 'suggested_learning_rate' in context:
            self.assertLess(context['suggested_learning_rate'], 0.01)
        
        if 'suggested_gradient_clipping' in context:
            self.assertLessEqual(context['suggested_gradient_clipping'], 1.0)
    
    def test_escalating_failure_handling(self):
        """Test handling of escalating consecutive failures."""
        error = RuntimeError("Persistent error")
        
        # Simulate multiple consecutive failures
        for i in range(7):
            context = {
                'consecutive_failures': i + 1,
                'session_id': 'failing_session'
            }
            
            success = self.recovery_manager.handle_error(error, context)
            
            if i >= 5:  # After 6 consecutive failures
                # Should trigger emergency stop
                if 'emergency_stop' in context:
                    self.assertTrue(context['emergency_stop'])
    
    def test_checkpoint_recovery_integration(self):
        """Test integration between error recovery and checkpoint system."""
        # Create multiple checkpoints
        for epoch in range(1, 4):
            checkpoint_id = self.recovery_manager.checkpoint_recovery.create_checkpoint(
                'test_session', epoch, epoch * 10,
                training_state={'epoch': epoch, 'loss': 1.0 / epoch}
            )
        
        # Simulate error requiring checkpoint rollback
        error = Exception("Model corruption detected")
        context = {'session_id': 'test_session'}
        
        success = self.recovery_manager.handle_error(error, context)
        
        # Should attempt checkpoint rollback
        if 'restored_checkpoint' in context:
            restored = context['restored_checkpoint']
            self.assertIsInstance(restored, RecoveryCheckpoint)
            self.assertEqual(restored.session_id, 'test_session')
    
    def test_state_rollback_integration(self):
        """Test integration between error recovery and state rollback."""
        # Build up state history
        states = []
        for i in range(5):
            state = {'iteration': i, 'loss': 1.0 / (i + 1)}
            states.append(state)
            self.recovery_manager.state_rollback.save_state(state)
        
        # Simulate error requiring state rollback
        error = ValueError("Invalid gradient detected")
        context = {}
        
        success = self.recovery_manager.handle_error(error, context)
        
        # Should attempt state rollback
        if 'rolled_back_state' in context:
            rolled_back = context['rolled_back_state']
            self.assertIn('iteration', rolled_back)
            self.assertLess(rolled_back['iteration'], 4)  # Should be earlier state
    
    def test_recovery_statistics_accuracy(self):
        """Test accuracy of recovery statistics."""
        # Handle various types of errors
        errors_and_contexts = [
            (RuntimeError("CUDA out of memory"), {'batch_size': 32}),
            (ValueError("Data shape mismatch"), {}),
            (Exception("Unknown error"), {}),
            (RuntimeError("Memory allocation failed"), {'batch_size': 16}),
        ]
        
        for error, context in errors_and_contexts:
            self.recovery_manager.handle_error(error, context)
        
        stats = self.recovery_manager.get_recovery_stats()
        
        # Verify basic statistics
        self.assertEqual(stats['total_errors'], 4)
        self.assertGreaterEqual(stats['total_recoveries'], 0)
        self.assertIn('memory_error', stats['error_types_seen'])
        
        # Verify success rate calculation
        if stats['total_recoveries'] > 0:
            expected_rate = stats['successful_recoveries'] / stats['total_recoveries']
            self.assertAlmostEqual(stats['recovery_success_rate'], expected_rate)
    
    def test_concurrent_error_handling(self):
        """Test concurrent error handling without race conditions."""
        errors_handled = []
        
        def handle_concurrent_errors(thread_id):
            for i in range(5):
                error = RuntimeError(f"Thread {thread_id} Error {i}")
                context = {'thread_id': thread_id, 'error_number': i}
                
                success = self.recovery_manager.handle_error(error, context)
                errors_handled.append((thread_id, i, success))
        
        # Run multiple threads concurrently
        threads = []
        for thread_id in range(3):
            thread = threading.Thread(
                target=handle_concurrent_errors, 
                args=(thread_id,)
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify all errors were handled
        self.assertEqual(len(errors_handled), 15)  # 3 threads × 5 errors
        self.assertEqual(len(self.recovery_manager.error_history), 15)
        
        # Verify no race conditions in statistics
        stats = self.recovery_manager.get_recovery_stats()
        self.assertEqual(stats['total_errors'], 15)


if __name__ == '__main__':
    unittest.main()