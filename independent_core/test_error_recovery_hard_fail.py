#!/usr/bin/env python3
"""
Test that ErrorRecoverySystem fails hard by default.

This ensures production systems don't mask errors with automatic recovery.
"""

import unittest
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from error_recovery_system import (
    ErrorRecoveryManager, ErrorType, ErrorSeverity,
    integrate_error_recovery
)


class TestHardFailure(unittest.TestCase):
    """Test hard failure behavior of ErrorRecoverySystem."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_default_fail_hard_mode(self):
        """Test that fail_hard=True is the default."""
        manager = ErrorRecoveryManager(checkpoint_dir=self.temp_dir)
        self.assertTrue(manager.fail_hard)
    
    def test_hard_failure_raises_exception(self):
        """Test that errors are re-raised in fail-hard mode."""
        manager = ErrorRecoveryManager(checkpoint_dir=self.temp_dir, fail_hard=True)
        
        error = RuntimeError("Test error")
        context = {'batch_size': 32}
        
        with self.assertRaises(RuntimeError) as cm:
            manager.handle_error(error, context)
        
        self.assertEqual(str(cm.exception), "Test error")
    
    def test_hard_failure_no_recovery(self):
        """Test that no recovery is attempted in fail-hard mode."""
        manager = ErrorRecoveryManager(checkpoint_dir=self.temp_dir, fail_hard=True)
        
        error = RuntimeError("CUDA out of memory")
        context = {'batch_size': 64}
        
        try:
            manager.handle_error(error, context)
        except RuntimeError:
            pass
        
        # Context should not have recovery modifications
        self.assertNotIn('suggested_batch_size', context)
        self.assertNotIn('batch_size_reduced', context)
        self.assertNotIn('cache_cleared', context)
    
    def test_hard_failure_preserves_diagnostics(self):
        """Test that diagnostics are still captured in fail-hard mode."""
        manager = ErrorRecoveryManager(checkpoint_dir=self.temp_dir, fail_hard=True)
        
        error = ValueError("Loss is NaN")
        context = {'learning_rate': 0.01, 'epoch': 5}
        
        try:
            manager.handle_error(error, context)
        except ValueError:
            pass
        
        # Error should be recorded in history
        error_history = manager.get_error_history()
        self.assertEqual(len(error_history), 1)
        self.assertEqual(error_history[0].message, "Loss is NaN")
        self.assertEqual(error_history[0].error_type, ErrorType.OPTIMIZATION_ERROR)
        self.assertFalse(error_history[0].recovery_success)
    
    def test_legacy_mode_attempts_recovery(self):
        """Test that fail_hard=False enables legacy recovery behavior."""
        manager = ErrorRecoveryManager(checkpoint_dir=self.temp_dir, fail_hard=False)
        
        error = RuntimeError("CUDA out of memory")
        context = {'batch_size': 64}
        
        # Should not raise exception in legacy mode
        success = manager.handle_error(error, context)
        
        # Should have attempted recovery
        self.assertIsInstance(success, bool)
        
        # Context may have recovery modifications
        # (depends on which strategy was applied)
        self.assertTrue(
            'cache_cleared' in context or 
            'suggested_batch_size' in context or
            'rolled_back_state' in context
        )
    
    def test_integration_defaults_to_fail_hard(self):
        """Test that integrate_error_recovery defaults to fail-hard mode."""
        from unittest.mock import Mock
        
        training_manager = Mock()
        recovery_manager = integrate_error_recovery(
            training_manager,
            checkpoint_dir=self.temp_dir
        )
        
        self.assertTrue(recovery_manager.fail_hard)
    
    def test_integration_with_explicit_fail_hard(self):
        """Test explicit fail_hard parameter in integration."""
        from unittest.mock import Mock
        
        training_manager = Mock()
        
        # Test explicit fail_hard=True
        recovery_manager_hard = integrate_error_recovery(
            training_manager,
            checkpoint_dir=self.temp_dir,
            fail_hard=True
        )
        self.assertTrue(recovery_manager_hard.fail_hard)
        
        # Test explicit fail_hard=False (legacy mode)
        recovery_manager_soft = integrate_error_recovery(
            training_manager,
            checkpoint_dir=self.temp_dir,
            fail_hard=False
        )
        self.assertFalse(recovery_manager_soft.fail_hard)
    
    def test_error_classification_still_works(self):
        """Test that error classification works even in fail-hard mode."""
        manager = ErrorRecoveryManager(checkpoint_dir=self.temp_dir, fail_hard=True)
        
        # Test various error types
        errors = [
            (RuntimeError("CUDA out of memory"), ErrorType.MEMORY_ERROR),
            (ValueError("Data shape mismatch"), ErrorType.DATA_ERROR),
            (Exception("Model weights corrupted"), ErrorType.MODEL_ERROR),
        ]
        
        for error, expected_type in errors:
            try:
                manager.handle_error(error, {})
            except:
                pass
            
            error_history = manager.get_error_history(limit=1)
            self.assertEqual(error_history[0].error_type, expected_type)
    
    def test_checkpoints_still_created(self):
        """Test that checkpoints can still be created for manual recovery."""
        manager = ErrorRecoveryManager(checkpoint_dir=self.temp_dir, fail_hard=True)
        
        # Create a checkpoint
        checkpoint_id = manager.checkpoint_recovery.create_checkpoint(
            session_id="test_session",
            epoch=5,
            batch=100,
            training_state={'loss': 0.5}
        )
        
        self.assertIsNotNone(checkpoint_id)
        
        # Verify checkpoint can be listed
        checkpoints = manager.checkpoint_recovery.list_checkpoints("test_session")
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0].session_id, "test_session")


if __name__ == '__main__':
    unittest.main()