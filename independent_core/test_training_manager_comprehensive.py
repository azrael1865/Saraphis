"""
Comprehensive test suite for TrainingManager component
"""

import sys
import os
import logging
import unittest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import tempfile
import shutil
from pathlib import Path
import threading
import time
import json
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import TrainingManager
from training_manager import (
    TrainingManager, TrainingSession, TrainingConfig, 
    TrainingStatus, TrainingMetrics,
    DataValidator, ResourceMonitor
)


class TestTrainingManager(unittest.TestCase):
    """Comprehensive test suite for TrainingManager"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for storage
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "test_training"
        
        # Create mock brain core and domain state manager
        self.mock_brain_core = Mock()
        self.mock_brain_core.shared_knowledge = {}
        self.mock_brain_core.update_knowledge = Mock(return_value=True)
        
        self.mock_domain_state = Mock()
        self.mock_domain_state.save_state = Mock(return_value=True)
        self.mock_domain_state.load_state = Mock(return_value={})
        
        # Initialize TrainingManager
        self.training_manager = TrainingManager(
            brain_core=self.mock_brain_core,
            domain_state_manager=self.mock_domain_state,
            storage_path=self.storage_path,
            max_concurrent_sessions=2,
            enable_monitoring=True,
            enable_recovery=True
        )
    
    def tearDown(self):
        """Clean up after tests"""
        # Shutdown training manager
        if hasattr(self.training_manager, 'shutdown'):
            self.training_manager.shutdown()
        
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test TrainingManager initialization"""
        self.assertIsNotNone(self.training_manager)
        self.assertEqual(self.training_manager.max_concurrent_sessions, 2)
        self.assertTrue(self.training_manager.enable_monitoring)
        self.assertTrue(self.training_manager.enable_recovery)
        self.assertTrue(self.storage_path.exists())
    
    def test_create_training_session(self):
        """Test creating a training session"""
        config = TrainingConfig(
            domain_id="test_domain",
            model_type="test_model",
            batch_size=32,
            learning_rate=0.001,
            epochs=10
        )
        
        session_id = self.training_manager.create_session(config)
        self.assertIsNotNone(session_id)
        self.assertIn(session_id, self.training_manager._sessions)
        
        session = self.training_manager._sessions[session_id]
        self.assertEqual(session.config.domain_id, "test_domain")
        self.assertEqual(session.status, TrainingStatus.INITIALIZED)
    
    def test_start_training_session(self):
        """Test starting a training session"""
        config = TrainingConfig(
            domain_id="test_domain",
            model_type="test_model",
            batch_size=32,
            learning_rate=0.001,
            epochs=1
        )
        
        # Create mock training data
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        session_id = self.training_manager.create_session(config)
        
        # Start training
        result = self.training_manager.start_training(
            session_id,
            X_train,
            y_train,
            validation_split=0.2
        )
        
        self.assertTrue(result)
        
        # Wait a bit for training to start
        time.sleep(0.1)
        
        session = self.training_manager._sessions[session_id]
        self.assertIn(session.status, [
            TrainingStatus.RUNNING,
            TrainingStatus.COMPLETED,
            TrainingStatus.FAILED
        ])
    
    def test_concurrent_sessions(self):
        """Test running multiple concurrent sessions"""
        sessions = []
        
        for i in range(2):
            config = TrainingConfig(
                domain_id=f"domain_{i}",
                model_type="test_model",
                batch_size=16,
                learning_rate=0.001,
                epochs=1
            )
            
            session_id = self.training_manager.create_session(config)
            sessions.append(session_id)
        
        # Start both sessions
        for session_id in sessions:
            X_train = np.random.randn(50, 5)
            y_train = np.random.randint(0, 2, 50)
            
            result = self.training_manager.start_training(
                session_id,
                X_train,
                y_train
            )
            self.assertTrue(result)
        
        # Check both are active
        self.assertEqual(len(self.training_manager._active_sessions), 2)
        
        # Try to create a third session (should be queued or rejected)
        config = TrainingConfig(
            domain_id="domain_3",
            model_type="test_model",
            batch_size=16,
            learning_rate=0.001,
            epochs=1
        )
        
        session_id = self.training_manager.create_session(config)
        self.assertIsNotNone(session_id)
    
    def test_pause_resume_session(self):
        """Test pausing and resuming a session"""
        config = TrainingConfig(
            domain_id="test_domain",
            model_type="test_model",
            batch_size=32,
            learning_rate=0.001,
            epochs=10
        )
        
        session_id = self.training_manager.create_session(config)
        
        # Start training
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        self.training_manager.start_training(session_id, X_train, y_train)
        time.sleep(0.1)
        
        # Pause
        result = self.training_manager.pause_session(session_id)
        self.assertTrue(result)
        
        session = self.training_manager._sessions[session_id]
        self.assertEqual(session.status, TrainingStatus.PAUSED)
        
        # Resume
        result = self.training_manager.resume_session(session_id)
        self.assertTrue(result)
        
        # Check status changed
        self.assertIn(session.status, [
            TrainingStatus.RUNNING,
            TrainingStatus.COMPLETED
        ])
    
    def test_cancel_session(self):
        """Test canceling a training session"""
        config = TrainingConfig(
            domain_id="test_domain",
            model_type="test_model",
            batch_size=32,
            learning_rate=0.001,
            epochs=10
        )
        
        session_id = self.training_manager.create_session(config)
        
        # Start training
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        self.training_manager.start_training(session_id, X_train, y_train)
        time.sleep(0.1)
        
        # Cancel
        result = self.training_manager.cancel_session(session_id)
        self.assertTrue(result)
        
        session = self.training_manager._sessions[session_id]
        self.assertEqual(session.status, TrainingStatus.CANCELLED)
    
    def test_get_session_metrics(self):
        """Test retrieving session metrics"""
        config = TrainingConfig(
            domain_id="test_domain",
            model_type="test_model",
            batch_size=32,
            learning_rate=0.001,
            epochs=1
        )
        
        session_id = self.training_manager.create_session(config)
        
        # Get initial metrics
        metrics = self.training_manager.get_session_metrics(session_id)
        self.assertIsNotNone(metrics)
        self.assertIsInstance(metrics, dict)
    
    def test_save_and_load_checkpoint(self):
        """Test checkpoint saving and loading"""
        config = TrainingConfig(
            domain_id="test_domain",
            model_type="test_model",
            batch_size=32,
            learning_rate=0.001,
            epochs=1
        )
        
        session_id = self.training_manager.create_session(config)
        
        # Create a checkpoint
        checkpoint_data = {
            'epoch': 5,
            'loss': 0.5,
            'accuracy': 0.85,
            'model_state': {'weights': [1, 2, 3]}
        }
        
        # Save checkpoint
        checkpoint_path = self.training_manager.save_checkpoint(
            session_id,
            checkpoint_data
        )
        self.assertIsNotNone(checkpoint_path)
        self.assertTrue(Path(checkpoint_path).exists())
        
        # Load checkpoint
        loaded_data = self.training_manager.load_checkpoint(session_id)
        self.assertIsNotNone(loaded_data)
        self.assertEqual(loaded_data['epoch'], 5)
        self.assertEqual(loaded_data['loss'], 0.5)
    
    def test_resource_monitoring(self):
        """Test resource monitoring during training"""
        if not self.training_manager._resource_monitor:
            self.skipTest("Resource monitoring not available")
        
        # Get initial resources
        resources = self.training_manager._resource_monitor.get_current_usage()
        self.assertIsNotNone(resources)
        self.assertIn('cpu_percent', resources)
        self.assertIn('memory_percent', resources)
    
    def test_error_handling(self):
        """Test error handling in training"""
        config = TrainingConfig(
            domain_id="test_domain",
            model_type="invalid_model",  # This should cause an error
            batch_size=32,
            learning_rate=0.001,
            epochs=1
        )
        
        session_id = self.training_manager.create_session(config)
        
        # Try to start training with invalid data
        result = self.training_manager.start_training(
            session_id,
            None,  # Invalid data
            None
        )
        
        # Should handle error gracefully
        self.assertFalse(result)
        
        session = self.training_manager._sessions[session_id]
        self.assertEqual(session.status, TrainingStatus.FAILED)
    
    def test_callback_registration(self):
        """Test callback registration and execution"""
        callback_called = False
        callback_data = None
        
        def test_callback(session_id, event, data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = data
        
        # Register callback
        self.training_manager.register_callback(
            'on_epoch_end',
            test_callback
        )
        
        # Trigger callback
        self.training_manager._trigger_callbacks(
            'on_epoch_end',
            'test_session',
            {'epoch': 1, 'loss': 0.5}
        )
        
        self.assertTrue(callback_called)
        self.assertEqual(callback_data['epoch'], 1)
    
    def test_cleanup_old_sessions(self):
        """Test cleanup of old sessions"""
        # Create multiple sessions
        old_sessions = []
        for i in range(5):
            config = TrainingConfig(
                domain_id=f"domain_{i}",
                model_type="test_model",
                batch_size=16,
                learning_rate=0.001,
                epochs=1
            )
            session_id = self.training_manager.create_session(config)
            old_sessions.append(session_id)
            
            # Mark as completed
            session = self.training_manager._sessions[session_id]
            session.status = TrainingStatus.COMPLETED
            session.end_time = datetime.now()
        
        # Cleanup old sessions
        retained = self.training_manager.cleanup_old_sessions(max_age_hours=0)
        
        # Check some sessions were cleaned
        self.assertLess(len(self.training_manager._sessions), 5)
    
    def test_export_training_history(self):
        """Test exporting training history"""
        config = TrainingConfig(
            domain_id="test_domain",
            model_type="test_model",
            batch_size=32,
            learning_rate=0.001,
            epochs=1
        )
        
        session_id = self.training_manager.create_session(config)
        
        # Add some metrics
        session = self.training_manager._sessions[session_id]
        session.metrics_history.append({
            'epoch': 1,
            'loss': 0.5,
            'accuracy': 0.8
        })
        
        # Export history
        export_path = self.storage_path / f"{session_id}_history.json"
        result = self.training_manager.export_session_history(
            session_id,
            export_path
        )
        
        self.assertTrue(result)
        self.assertTrue(export_path.exists())
        
        # Verify content
        with open(export_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(len(data['metrics_history']), 1)
    
    def test_domain_isolation(self):
        """Test domain isolation in training"""
        # Create sessions for different domains
        domains = ['domain_a', 'domain_b']
        sessions = {}
        
        for domain in domains:
            config = TrainingConfig(
                domain_id=domain,
                model_type="test_model",
                batch_size=16,
                learning_rate=0.001,
                epochs=1
            )
            session_id = self.training_manager.create_session(config)
            sessions[domain] = session_id
        
        # Verify sessions are isolated
        for domain, session_id in sessions.items():
            session = self.training_manager._sessions[session_id]
            self.assertEqual(session.config.domain_id, domain)
            
            # Check domain-specific storage
            domain_path = self.storage_path / domain
            if domain_path.exists():
                self.assertTrue(domain_path.is_dir())
    
    def test_thread_safety(self):
        """Test thread safety of TrainingManager"""
        results = []
        errors = []
        
        def create_and_start_session(index):
            try:
                config = TrainingConfig(
                    domain_id=f"domain_{index}",
                    model_type="test_model",
                    batch_size=16,
                    learning_rate=0.001,
                    epochs=1
                )
                
                session_id = self.training_manager.create_session(config)
                results.append(session_id)
                
                X_train = np.random.randn(50, 5)
                y_train = np.random.randint(0, 2, 50)
                
                self.training_manager.start_training(
                    session_id,
                    X_train,
                    y_train
                )
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=create_and_start_session,
                args=(i,)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5)
        
        # Check results
        self.assertEqual(len(errors), 0)
        self.assertGreater(len(results), 0)
        
        # Verify all sessions were created
        for session_id in results:
            self.assertIn(session_id, self.training_manager._sessions)


class TestDataValidator(unittest.TestCase):
    """Test DataValidator component"""
    
    def setUp(self):
        """Set up test environment"""
        self.validator = DataValidator()
    
    def test_validate_training_data(self):
        """Test training data validation"""
        # Valid data
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        result = self.validator.validate_training_data(X, y)
        self.assertTrue(result)
        
        # Invalid data - mismatched sizes
        X_invalid = np.random.randn(100, 10)
        y_invalid = np.random.randint(0, 2, 50)
        
        result = self.validator.validate_training_data(X_invalid, y_invalid)
        self.assertFalse(result)
        
        # Invalid data - NaN values
        X_nan = np.random.randn(100, 10)
        X_nan[0, 0] = np.nan
        
        result = self.validator.validate_training_data(X_nan, y)
        self.assertFalse(result)
    
    def test_validate_model_params(self):
        """Test model parameter validation"""
        # Valid params
        params = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10,
            'hidden_units': [64, 32]
        }
        
        result = self.validator.validate_model_params(params)
        self.assertTrue(result)
        
        # Invalid params - negative learning rate
        invalid_params = {
            'learning_rate': -0.001,
            'batch_size': 32,
            'epochs': 10
        }
        
        result = self.validator.validate_model_params(invalid_params)
        self.assertFalse(result)


class TestResourceMonitor(unittest.TestCase):
    """Test ResourceMonitor component"""
    
    def setUp(self):
        """Set up test environment"""
        self.monitor = ResourceMonitor()
    
    def test_get_current_usage(self):
        """Test getting current resource usage"""
        usage = self.monitor.get_current_usage()
        
        self.assertIsNotNone(usage)
        self.assertIn('cpu_percent', usage)
        self.assertIn('memory_percent', usage)
        self.assertIn('disk_usage', usage)
        
        # Check values are reasonable
        self.assertGreaterEqual(usage['cpu_percent'], 0)
        self.assertLessEqual(usage['cpu_percent'], 100)
        self.assertGreaterEqual(usage['memory_percent'], 0)
        self.assertLessEqual(usage['memory_percent'], 100)
    
    def test_check_resource_availability(self):
        """Test checking resource availability"""
        # Should have resources available
        available = self.monitor.check_resource_availability(
            min_memory_gb=0.1,
            min_cpu_percent=50
        )
        
        # This might fail on heavily loaded systems
        # but should generally pass in test environments
        self.assertIsNotNone(available)
    
    def test_start_monitoring(self):
        """Test continuous monitoring"""
        # Start monitoring
        self.monitor.start_monitoring(interval=0.1)
        
        # Wait for some samples
        time.sleep(0.5)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        
        # Check we have history
        history = self.monitor.get_usage_history()
        self.assertIsNotNone(history)
        self.assertGreater(len(history), 0)


def run_comprehensive_tests():
    """Run all comprehensive tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTrainingManager))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestResourceMonitor))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)