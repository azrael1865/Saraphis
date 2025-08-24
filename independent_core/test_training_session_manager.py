#!/usr/bin/env python3
"""
Comprehensive test suite for TrainingSessionManager
Tests all core functionality and identifies issues
"""

import unittest
import tempfile
import shutil
import time
import json
import pickle
import threading
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the module
try:
    from training_session_manager import (
        TrainingSessionManager, TrainingSession, SessionMetrics, ResourceMetrics,
        TrainingError, TrainingCheckpoint, SessionStatus, ErrorType, RecoveryStrategy
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)
    logger.error(f"Failed to import TrainingSessionManager: {e}")


class TestEnumsAndTypes(unittest.TestCase):
    """Test enums and type definitions"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Imports failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_session_status_enum(self):
        """Test SessionStatus enum values"""
        self.assertEqual(SessionStatus.INITIALIZING.value, "initializing")
        self.assertEqual(SessionStatus.RUNNING.value, "running")
        self.assertEqual(SessionStatus.PAUSED.value, "paused")
        self.assertEqual(SessionStatus.COMPLETED.value, "completed")
        self.assertEqual(SessionStatus.FAILED.value, "failed")
        self.assertEqual(SessionStatus.CANCELLED.value, "cancelled")
        self.assertEqual(SessionStatus.RECOVERING.value, "recovering")
    
    def test_error_type_enum(self):
        """Test ErrorType enum values"""
        self.assertEqual(ErrorType.OUT_OF_MEMORY.value, "out_of_memory")
        self.assertEqual(ErrorType.NAN_LOSS.value, "nan_loss")
        self.assertEqual(ErrorType.INFINITE_LOSS.value, "infinite_loss")
        self.assertEqual(ErrorType.GRADIENT_EXPLOSION.value, "gradient_explosion")
        self.assertEqual(ErrorType.DATA_ERROR.value, "data_error")
        self.assertEqual(ErrorType.RUNTIME_ERROR.value, "runtime_error")
        self.assertEqual(ErrorType.RESOURCE_ERROR.value, "resource_error")
        self.assertEqual(ErrorType.UNKNOWN.value, "unknown")
    
    def test_recovery_strategy_enum(self):
        """Test RecoveryStrategy enum values"""
        self.assertEqual(RecoveryStrategy.REDUCE_BATCH_SIZE.value, "reduce_batch_size")
        self.assertEqual(RecoveryStrategy.REDUCE_LEARNING_RATE.value, "reduce_learning_rate")
        self.assertEqual(RecoveryStrategy.SKIP_BATCH.value, "skip_batch")
        self.assertEqual(RecoveryStrategy.RESET_OPTIMIZER.value, "reset_optimizer")
        self.assertEqual(RecoveryStrategy.LOAD_CHECKPOINT.value, "load_checkpoint")
        self.assertEqual(RecoveryStrategy.RESTART_TRAINING.value, "restart_training")


class TestSessionMetrics(unittest.TestCase):
    """Test SessionMetrics data class"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Imports failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def setUp(self):
        self.metrics = SessionMetrics()
    
    def test_metrics_initialization(self):
        """Test metrics initialize with correct defaults"""
        self.assertEqual(len(self.metrics.loss_history), 0)
        self.assertEqual(len(self.metrics.accuracy_history), 0)
        self.assertIsNone(self.metrics.best_loss)
        self.assertIsNone(self.metrics.best_accuracy)
        self.assertEqual(self.metrics.current_epoch, 0)
        self.assertEqual(self.metrics.current_batch, 0)
        self.assertEqual(self.metrics.overfitting_score, 0.0)
    
    def test_update_loss(self):
        """Test loss update tracking"""
        # Update training loss
        self.metrics.update_loss(1.5)
        self.assertEqual(len(self.metrics.loss_history), 1)
        self.assertEqual(self.metrics.loss_history[0], 1.5)
        self.assertEqual(self.metrics.best_loss, 1.5)
        
        # Update with better loss
        self.metrics.update_loss(1.2)
        self.assertEqual(len(self.metrics.loss_history), 2)
        self.assertEqual(self.metrics.best_loss, 1.2)
        
        # Update validation loss
        self.metrics.update_loss(1.8, is_validation=True)
        self.assertEqual(len(self.metrics.val_loss_history), 1)
        self.assertEqual(self.metrics.best_val_loss, 1.8)
    
    def test_update_accuracy(self):
        """Test accuracy update tracking"""
        # Update training accuracy
        self.metrics.update_accuracy(0.75)
        self.assertEqual(len(self.metrics.accuracy_history), 1)
        self.assertEqual(self.metrics.accuracy_history[0], 0.75)
        self.assertEqual(self.metrics.best_accuracy, 0.75)
        
        # Update with better accuracy
        self.metrics.update_accuracy(0.85)
        self.assertEqual(len(self.metrics.accuracy_history), 2)
        self.assertEqual(self.metrics.best_accuracy, 0.85)
        
        # Update validation accuracy
        self.metrics.update_accuracy(0.80, is_validation=True)
        self.assertEqual(len(self.metrics.val_accuracy_history), 1)
        self.assertEqual(self.metrics.best_val_accuracy, 0.80)
    
    def test_calculate_overfitting_score(self):
        """Test overfitting score calculation"""
        # Not enough data
        score = self.metrics.calculate_overfitting_score()
        self.assertEqual(score, 0.0)
        
        # Add some training and validation losses
        for i in range(10):
            self.metrics.update_loss(1.0 - i * 0.05)  # Decreasing training loss
            self.metrics.update_loss(1.2 - i * 0.03, is_validation=True)  # Slower decreasing val loss
        
        score = self.metrics.calculate_overfitting_score()
        self.assertGreater(score, 1.0)  # Validation loss higher than training


class TestResourceMetrics(unittest.TestCase):
    """Test ResourceMetrics data class"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Imports failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def setUp(self):
        self.resources = ResourceMetrics()
    
    def test_add_measurement(self):
        """Test adding resource measurements"""
        self.resources.add_measurement(cpu=50.0, memory=60.0, gpu=70.0, gpu_mem=4000.0)
        
        self.assertEqual(len(self.resources.cpu_usage), 1)
        self.assertEqual(self.resources.cpu_usage[0], 50.0)
        self.assertEqual(self.resources.memory_usage[0], 60.0)
        self.assertEqual(self.resources.gpu_usage[0], 70.0)
        self.assertEqual(self.resources.gpu_memory[0], 4000.0)
        self.assertEqual(len(self.resources.timestamps), 1)
    
    def test_measurement_limit(self):
        """Test that measurements are limited to 1000 entries"""
        # Add 1005 measurements
        for i in range(1005):
            self.resources.add_measurement(cpu=i, memory=i, gpu=i, gpu_mem=i)
        
        # Should only keep last 1000
        self.assertEqual(len(self.resources.cpu_usage), 1000)
        self.assertEqual(self.resources.cpu_usage[0], 5)  # First should be index 5
        self.assertEqual(self.resources.cpu_usage[-1], 1004)  # Last should be index 1004


class TestTrainingError(unittest.TestCase):
    """Test TrainingError data class"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Imports failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_error_creation(self):
        """Test creating training error"""
        error = TrainingError(
            error_type=ErrorType.NAN_LOSS,
            error_message="Loss became NaN",
            traceback_str="Traceback...",
            timestamp=datetime.now(),
            epoch=5,
            batch=100
        )
        
        self.assertEqual(error.error_type, ErrorType.NAN_LOSS)
        self.assertEqual(error.error_message, "Loss became NaN")
        self.assertEqual(error.epoch, 5)
        self.assertEqual(error.batch, 100)
        self.assertFalse(error.recovery_attempted)
        self.assertIsNone(error.recovery_strategy)
        self.assertFalse(error.recovery_successful)


class TestTrainingCheckpoint(unittest.TestCase):
    """Test TrainingCheckpoint data class"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Imports failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_checkpoint_creation(self):
        """Test creating training checkpoint"""
        checkpoint = TrainingCheckpoint(
            checkpoint_id="checkpoint_001",
            epoch=10,
            batch=500,
            model_state={'layer1': 'weights'},
            optimizer_state={'lr': 0.001}
        )
        
        self.assertEqual(checkpoint.checkpoint_id, "checkpoint_001")
        self.assertEqual(checkpoint.epoch, 10)
        self.assertEqual(checkpoint.batch, 500)
        self.assertIsNotNone(checkpoint.model_state)
        self.assertIsNotNone(checkpoint.optimizer_state)
        self.assertFalse(checkpoint.is_best)
        self.assertTrue(checkpoint.is_auto)


class TestTrainingSession(unittest.TestCase):
    """Test TrainingSession data class"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Imports failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def test_session_creation(self):
        """Test creating training session"""
        session = TrainingSession(
            session_id="test_session",
            domain_name="test_domain",
            model_type="test_model",
            status=SessionStatus.INITIALIZING,
            config={'learning_rate': 0.001}
        )
        
        self.assertEqual(session.session_id, "test_session")
        self.assertEqual(session.domain_name, "test_domain")
        self.assertEqual(session.model_type, "test_model")
        self.assertEqual(session.status, SessionStatus.INITIALIZING)
        self.assertEqual(session.config['learning_rate'], 0.001)
    
    def test_session_duration(self):
        """Test session duration calculation"""
        session = TrainingSession(
            session_id="test",
            domain_name="test",
            model_type="test",
            status=SessionStatus.RUNNING,
            config={}
        )
        
        # Set specific times for testing
        session.start_time = datetime(2024, 1, 1, 10, 0, 0)
        session.end_time = datetime(2024, 1, 1, 11, 30, 0)
        
        duration = session.duration()
        self.assertEqual(duration, timedelta(hours=1, minutes=30))
    
    def test_is_active(self):
        """Test active session detection"""
        session = TrainingSession(
            session_id="test",
            domain_name="test",
            model_type="test",
            status=SessionStatus.RUNNING,
            config={}
        )
        
        # Active statuses
        session.status = SessionStatus.INITIALIZING
        self.assertTrue(session.is_active())
        
        session.status = SessionStatus.RUNNING
        self.assertTrue(session.is_active())
        
        session.status = SessionStatus.RECOVERING
        self.assertTrue(session.is_active())
        
        # Inactive statuses
        session.status = SessionStatus.COMPLETED
        self.assertFalse(session.is_active())
        
        session.status = SessionStatus.FAILED
        self.assertFalse(session.is_active())
        
        session.status = SessionStatus.CANCELLED
        self.assertFalse(session.is_active())


class TestTrainingSessionManager(unittest.TestCase):
    """Test TrainingSessionManager main class"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Imports failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TrainingSessionManager(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            self.manager.shutdown()
        except:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_manager_initialization(self):
        """Test manager initializes correctly"""
        self.assertIsNotNone(self.manager)
        self.assertEqual(self.manager.storage_path, Path(self.temp_dir))
        self.assertTrue(self.manager.storage_path.exists())
        self.assertEqual(len(self.manager.sessions), 0)
        self.assertEqual(len(self.manager.active_sessions), 0)
    
    def test_create_session(self):
        """Test creating a new training session"""
        config = {
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001
        }
        
        session_id = self.manager.create_session(
            domain_name="test_domain",
            model_type="test_model",
            config=config
        )
        
        self.assertIsNotNone(session_id)
        self.assertIn(session_id, self.manager.sessions)
        
        session = self.manager.sessions[session_id]
        self.assertEqual(session.domain_name, "test_domain")
        self.assertEqual(session.model_type, "test_model")
        self.assertEqual(session.status, SessionStatus.INITIALIZING)
        self.assertEqual(session.config['epochs'], 10)
    
    def test_start_session(self):
        """Test starting a training session"""
        session_id = self.manager.create_session(
            domain_name="test",
            model_type="model",
            config={}
        )
        
        success = self.manager.start_session(session_id)
        self.assertTrue(success)
        
        session = self.manager.sessions[session_id]
        self.assertEqual(session.status, SessionStatus.RUNNING)
        self.assertIn(session_id, self.manager.active_sessions)
    
    def test_start_nonexistent_session(self):
        """Test starting a non-existent session"""
        success = self.manager.start_session("nonexistent")
        self.assertFalse(success)
    
    def test_update_metrics(self):
        """Test updating session metrics"""
        session_id = self.manager.create_session(
            domain_name="test",
            model_type="model",
            config={}
        )
        self.manager.start_session(session_id)
        
        self.manager.update_metrics(
            session_id=session_id,
            epoch=1,
            batch=10,
            loss=1.5,
            accuracy=0.75,
            val_loss=1.8,
            val_accuracy=0.70
        )
        
        session = self.manager.sessions[session_id]
        self.assertEqual(session.metrics.current_epoch, 1)
        self.assertEqual(session.metrics.current_batch, 10)
        self.assertEqual(len(session.metrics.loss_history), 1)
        self.assertEqual(session.metrics.loss_history[0], 1.5)
        self.assertEqual(len(session.metrics.accuracy_history), 1)
        self.assertEqual(session.metrics.accuracy_history[0], 0.75)
    
    def test_report_progress(self):
        """Test progress reporting"""
        session_id = self.manager.create_session(
            domain_name="test",
            model_type="model",
            config={'epochs': 10, 'total_batches': 100}
        )
        self.manager.start_session(session_id)
        
        self.manager.report_progress(
            session_id=session_id,
            epoch=5,
            batch=50,
            loss=1.0,
            accuracy=0.8
        )
        
        session = self.manager.sessions[session_id]
        self.assertEqual(session.metrics.current_epoch, 5)
        self.assertEqual(session.metrics.current_batch, 50)
        self.assertGreater(session.metrics.total_progress, 0)
    
    def test_handle_error(self):
        """Test error handling"""
        session_id = self.manager.create_session(
            domain_name="test",
            model_type="model",
            config={}
        )
        self.manager.start_session(session_id)
        
        error = RuntimeError("Test error")
        self.manager.handle_error(
            session_id=session_id,
            error=error,
            epoch=2,
            batch=20
        )
        
        session = self.manager.sessions[session_id]
        self.assertEqual(len(session.errors), 1)
        self.assertEqual(session.errors[0].error_message, "Test error")
        self.assertEqual(session.errors[0].epoch, 2)
        self.assertEqual(session.errors[0].batch, 20)
    
    def test_create_checkpoint(self):
        """Test checkpoint creation"""
        session_id = self.manager.create_session(
            domain_name="test",
            model_type="model",
            config={}
        )
        self.manager.start_session(session_id)
        
        # Update some metrics first
        self.manager.update_metrics(
            session_id=session_id,
            epoch=5,
            batch=100,
            loss=0.5,
            accuracy=0.9
        )
        
        # Create checkpoint
        checkpoint_id = self.manager.create_checkpoint(
            session_id=session_id,
            epoch=5,
            batch=100,
            model_state={'weights': 'data'},
            optimizer_state={'lr': 0.001},
            is_best=True
        )
        
        self.assertIsNotNone(checkpoint_id)
        
        session = self.manager.sessions[session_id]
        self.assertEqual(len(session.checkpoints), 1)
        self.assertEqual(session.checkpoints[0].checkpoint_id, checkpoint_id)
        self.assertTrue(session.checkpoints[0].is_best)
        self.assertEqual(session.best_checkpoint.checkpoint_id, checkpoint_id)
    
    def test_complete_session(self):
        """Test completing a session"""
        session_id = self.manager.create_session(
            domain_name="test",
            model_type="model",
            config={}
        )
        self.manager.start_session(session_id)
        
        final_metrics = {
            'final_loss': 0.1,
            'final_accuracy': 0.95
        }
        
        self.manager.complete_session(
            session_id=session_id,
            final_metrics=final_metrics
        )
        
        session = self.manager.sessions[session_id]
        self.assertEqual(session.status, SessionStatus.COMPLETED)
        self.assertNotIn(session_id, self.manager.active_sessions)
        self.assertIsNotNone(session.end_time)
    
    def test_cancel_session(self):
        """Test cancelling a session"""
        session_id = self.manager.create_session(
            domain_name="test",
            model_type="model",
            config={}
        )
        self.manager.start_session(session_id)
        
        success = self.manager.cancel_session(session_id)
        self.assertTrue(success)
        
        session = self.manager.sessions[session_id]
        self.assertEqual(session.status, SessionStatus.CANCELLED)
        self.assertNotIn(session_id, self.manager.active_sessions)
    
    def test_get_session(self):
        """Test getting session by ID"""
        session_id = self.manager.create_session(
            domain_name="test",
            model_type="model",
            config={}
        )
        
        session = self.manager.get_session(session_id)
        self.assertIsNotNone(session)
        self.assertEqual(session.session_id, session_id)
        
        # Test non-existent session
        session = self.manager.get_session("nonexistent")
        self.assertIsNone(session)
    
    def test_get_active_sessions(self):
        """Test getting active sessions"""
        # Create multiple sessions
        session1 = self.manager.create_session("test1", "model", {})
        session2 = self.manager.create_session("test2", "model", {})
        session3 = self.manager.create_session("test3", "model", {})
        
        # Start two sessions
        self.manager.start_session(session1)
        self.manager.start_session(session2)
        
        active_sessions = self.manager.get_active_sessions()
        self.assertEqual(len(active_sessions), 2)
        
        session_ids = [s.session_id for s in active_sessions]
        self.assertIn(session1, session_ids)
        self.assertIn(session2, session_ids)
        self.assertNotIn(session3, session_ids)
    
    def test_add_callback(self):
        """Test adding callbacks to sessions"""
        session_id = self.manager.create_session(
            domain_name="test",
            model_type="model",
            config={}
        )
        
        callback_called = False
        def test_callback(session, event, data):
            nonlocal callback_called
            callback_called = True
        
        success = self.manager.add_callback(session_id, test_callback)
        self.assertTrue(success)
        
        session = self.manager.sessions[session_id]
        self.assertEqual(len(session.callbacks), 1)
    
    def test_cleanup_session(self):
        """Test session cleanup"""
        session_id = self.manager.create_session(
            domain_name="test",
            model_type="model",
            config={}
        )
        
        # Create a checkpoint file
        checkpoint_file = self.manager.storage_path / f"{session_id}_checkpoint.pkl"
        checkpoint_file.touch()
        
        success = self.manager.cleanup_session(session_id, keep_checkpoints=False)
        self.assertTrue(success)
        
        self.assertNotIn(session_id, self.manager.sessions)
        self.assertFalse(checkpoint_file.exists())


class TestErrorRecovery(unittest.TestCase):
    """Test error recovery mechanisms"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Imports failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TrainingSessionManager(storage_path=self.temp_dir)
    
    def tearDown(self):
        try:
            self.manager.shutdown()
        except:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_classify_error(self):
        """Test error classification"""
        # Test NaN error
        nan_error = ValueError("Loss is NaN")
        error_type = self.manager._classify_error(nan_error)
        self.assertEqual(error_type, ErrorType.NAN_LOSS)
        
        # Test infinite error
        inf_error = ValueError("Loss is inf")
        error_type = self.manager._classify_error(inf_error)
        self.assertEqual(error_type, ErrorType.INFINITE_LOSS)
        
        # Test OOM error
        oom_error = RuntimeError("CUDA out of memory")
        error_type = self.manager._classify_error(oom_error)
        self.assertEqual(error_type, ErrorType.OUT_OF_MEMORY)
        
        # Test gradient explosion
        grad_error = RuntimeError("Gradient overflow")
        error_type = self.manager._classify_error(grad_error)
        self.assertEqual(error_type, ErrorType.GRADIENT_EXPLOSION)
        
        # Test unknown error
        unknown_error = Exception("Something else")
        error_type = self.manager._classify_error(unknown_error)
        self.assertEqual(error_type, ErrorType.UNKNOWN)
    
    def test_determine_recovery_strategy(self):
        """Test recovery strategy determination"""
        # OOM should reduce batch size
        strategy = self.manager._determine_recovery_strategy(ErrorType.OUT_OF_MEMORY)
        self.assertEqual(strategy, RecoveryStrategy.REDUCE_BATCH_SIZE)
        
        # NaN loss should reset optimizer
        strategy = self.manager._determine_recovery_strategy(ErrorType.NAN_LOSS)
        self.assertEqual(strategy, RecoveryStrategy.RESET_OPTIMIZER)
        
        # Gradient explosion should reduce learning rate
        strategy = self.manager._determine_recovery_strategy(ErrorType.GRADIENT_EXPLOSION)
        self.assertEqual(strategy, RecoveryStrategy.REDUCE_LEARNING_RATE)
        
        # Data error should skip batch
        strategy = self.manager._determine_recovery_strategy(ErrorType.DATA_ERROR)
        self.assertEqual(strategy, RecoveryStrategy.SKIP_BATCH)


class TestResourceMonitoring(unittest.TestCase):
    """Test resource monitoring functionality"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Imports failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TrainingSessionManager(storage_path=self.temp_dir)
    
    def tearDown(self):
        try:
            self.manager.shutdown()
        except:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_monitoring_start_stop(self):
        """Test starting and stopping monitoring"""
        # Should start automatically
        self.assertTrue(self.manager.monitoring_active)
        self.assertIsNotNone(self.manager.monitoring_thread)
        
        # Stop monitoring
        self.manager.stop_monitoring()
        self.assertFalse(self.manager.monitoring_active)
        
        # Restart monitoring
        self.manager.start_monitoring()
        self.assertTrue(self.manager.monitoring_active)
    
    def test_resource_collection(self):
        """Test that resource metrics are collected"""
        session_id = self.manager.create_session(
            domain_name="test",
            model_type="model",
            config={}
        )
        self.manager.start_session(session_id)
        
        # Wait for monitoring to collect some data
        time.sleep(0.1)
        
        session = self.manager.sessions[session_id]
        # Resource collection happens in background thread
        # Just verify the structure exists
        self.assertIsNotNone(session.resource_metrics)
        self.assertIsInstance(session.resource_metrics, ResourceMetrics)


class TestCheckpointManagement(unittest.TestCase):
    """Test checkpoint management"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Imports failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TrainingSessionManager(storage_path=self.temp_dir)
    
    def tearDown(self):
        try:
            self.manager.shutdown()
        except:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_checkpoint_limit(self):
        """Test that checkpoints are limited"""
        session_id = self.manager.create_session(
            domain_name="test",
            model_type="model",
            config={}
        )
        self.manager.start_session(session_id)
        
        # Create more checkpoints than the limit
        for i in range(15):
            self.manager.create_checkpoint(
                session_id=session_id,
                epoch=i,
                batch=i*10,
                model_state={},
                optimizer_state={}
            )
        
        session = self.manager.sessions[session_id]
        # Should only keep max_checkpoints_per_session
        self.assertLessEqual(len(session.checkpoints), self.manager.max_checkpoints_per_session)
    
    def test_recover_from_checkpoint(self):
        """Test recovering from checkpoint"""
        session_id = self.manager.create_session(
            domain_name="test",
            model_type="model",
            config={}
        )
        self.manager.start_session(session_id)
        
        # Create a checkpoint
        checkpoint_id = self.manager.create_checkpoint(
            session_id=session_id,
            epoch=5,
            batch=100,
            model_state={'test': 'state'},
            optimizer_state={'lr': 0.001}
        )
        
        # Simulate failure
        self.manager.sessions[session_id].status = SessionStatus.FAILED
        
        # Recover from checkpoint
        model_state, optimizer_state = self.manager.recover_from_checkpoint(
            session_id=session_id,
            checkpoint_id=checkpoint_id
        )
        
        self.assertIsNotNone(model_state)
        self.assertIsNotNone(optimizer_state)
        self.assertEqual(model_state['test'], 'state')
        
        session = self.manager.sessions[session_id]
        self.assertEqual(session.status, SessionStatus.RECOVERING)


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of operations"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, f"Imports failed: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TrainingSessionManager(storage_path=self.temp_dir)
    
    def tearDown(self):
        try:
            self.manager.shutdown()
        except:
            pass
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_concurrent_session_creation(self):
        """Test creating sessions concurrently"""
        session_ids = []
        lock = threading.Lock()
        
        def create_session(index):
            session_id = self.manager.create_session(
                domain_name=f"test_{index}",
                model_type="model",
                config={}
            )
            with lock:
                session_ids.append(session_id)
        
        threads = []
        for i in range(10):
            t = threading.Thread(target=create_session, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All sessions should be created successfully
        self.assertEqual(len(session_ids), 10)
        self.assertEqual(len(set(session_ids)), 10)  # All unique
    
    def test_concurrent_metric_updates(self):
        """Test updating metrics concurrently"""
        session_id = self.manager.create_session(
            domain_name="test",
            model_type="model",
            config={}
        )
        self.manager.start_session(session_id)
        
        def update_metrics(index):
            self.manager.update_metrics(
                session_id=session_id,
                epoch=index,
                batch=index*10,
                loss=1.0 / (index + 1),
                accuracy=0.5 + index * 0.01
            )
        
        threads = []
        for i in range(20):
            t = threading.Thread(target=update_metrics, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        session = self.manager.sessions[session_id]
        # Should have recorded all updates
        self.assertEqual(len(session.metrics.loss_history), 20)


def run_tests():
    """Run all tests and report results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestEnumsAndTypes,
        TestSessionMetrics,
        TestResourceMetrics,
        TestTrainingError,
        TestTrainingCheckpoint,
        TestTrainingSession,
        TestTrainingSessionManager,
        TestErrorRecovery,
        TestResourceMonitoring,
        TestCheckpointManagement,
        TestThreadSafety
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print detailed summary
    print("\n" + "="*70)
    print("TRAINING SESSION MANAGER TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, trace in result.failures:
            print(f"  - {test}:")
            lines = trace.split('\n')[:3]
            for line in lines:
                if line.strip():
                    print(f"    {line[:100]}")
    
    if result.errors:
        print("\nERRORS:")
        for test, trace in result.errors:
            print(f"  - {test}:")
            lines = trace.split('\n')[:3]
            for line in lines:
                if line.strip():
                    print(f"    {line[:100]}")
    
    # Import status
    print("\n" + "="*70)
    print("IMPORT STATUS")
    print("="*70)
    print(f"TrainingSessionManager imports: {'✅ SUCCESS' if IMPORTS_AVAILABLE else '❌ FAILED'}")
    if not IMPORTS_AVAILABLE:
        print(f"Import error: {IMPORT_ERROR}")
    
    # Test categories
    print("\n" + "="*70)
    print("TEST CATEGORIES")
    print("="*70)
    for cls in test_classes:
        test_suite = loader.loadTestsFromTestCase(cls)
        test_count = test_suite.countTestCases()
        print(f"  {cls.__name__}: {test_count} tests")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOVERALL RESULT: {'✅ PASS' if success else '❌ FAIL'}")
    
    return success


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)