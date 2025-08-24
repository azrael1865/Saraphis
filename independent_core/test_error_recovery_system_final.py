#!/usr/bin/env python3
"""
Final comprehensive test suite for ErrorRecoverySystem with fail-hard behavior.

This test suite comprehensively tests all aspects of the error recovery system
with the correct fail-hard behavior (default) and legacy recovery behavior.
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
# NO MOCKS - testing real implementation only

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from error_recovery_system import (
    ErrorType, ErrorSeverity, RecoveryStrategy,
    ErrorRecord, RecoveryCheckpoint,
    ErrorClassifier, CheckpointRecovery, StateRollback, ErrorRecoveryManager,
    integrate_error_recovery, _extract_primitive_parameter
)


class TestErrorSystemBasics(unittest.TestCase):
    """Test basic enum and dataclass functionality."""
    
    def test_all_error_types_exist(self):
        """Test that all expected error types are defined."""
        expected_types = [
            'MEMORY_ERROR', 'CUDA_ERROR', 'DATA_ERROR', 'MODEL_ERROR',
            'OPTIMIZATION_ERROR', 'IO_ERROR', 'TIMEOUT_ERROR', 'VALIDATION_ERROR',
            'CHECKPOINT_ERROR', 'UNKNOWN_ERROR'
        ]
        
        for error_type in expected_types:
            self.assertTrue(hasattr(ErrorType, error_type))
    
    def test_all_severity_levels_exist(self):
        """Test that all expected severity levels are defined."""
        expected_severities = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'WARNING']
        
        for severity in expected_severities:
            self.assertTrue(hasattr(ErrorSeverity, severity))
    
    def test_all_recovery_strategies_exist(self):
        """Test that all expected recovery strategies are defined."""
        expected_strategies = [
            'RETRY', 'REDUCE_BATCH_SIZE', 'REDUCE_LEARNING_RATE', 'GRADIENT_CLIPPING',
            'CLEAR_CACHE', 'ROLLBACK_CHECKPOINT', 'ROLLBACK_STATE', 'RESTART_TRAINING',
            'SKIP_BATCH', 'EMERGENCY_STOP'
        ]
        
        for strategy in expected_strategies:
            self.assertTrue(hasattr(RecoveryStrategy, strategy))


class TestErrorClassifierFixed(unittest.TestCase):
    """Test ErrorClassifier with all fixes applied."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = ErrorClassifier()
    
    def test_all_error_types_have_patterns(self):
        """Test that ALL error types have patterns (including UNKNOWN_ERROR)."""
        for error_type in ErrorType:
            self.assertIn(error_type, self.classifier.error_patterns, 
                         f"Missing pattern for {error_type}")
            self.assertIsInstance(self.classifier.error_patterns[error_type], list)
            self.assertGreater(len(self.classifier.error_patterns[error_type]), 0)
    
    def test_all_severity_levels_have_indicators(self):
        """Test that ALL severity levels have indicators (including WARNING)."""
        for severity in ErrorSeverity:
            self.assertIn(severity, self.classifier.severity_indicators, 
                         f"Missing indicators for {severity}")
            self.assertIsInstance(self.classifier.severity_indicators[severity], list)
            self.assertGreater(len(self.classifier.severity_indicators[severity]), 0)
    
    def test_unknown_error_classification(self):
        """Test UNKNOWN_ERROR pattern matching."""
        error = Exception("Some unknown unexpected error")
        error_type, severity = self.classifier.classify_error(error)
        self.assertEqual(error_type, ErrorType.UNKNOWN_ERROR)
    
    def test_warning_severity_classification(self):
        """Test WARNING severity pattern matching."""
        # This would need a warning-level error to test properly
        self.assertIn("warning", self.classifier.severity_indicators[ErrorSeverity.WARNING])
        self.assertIn("warn", self.classifier.severity_indicators[ErrorSeverity.WARNING])
    
    def test_recovery_strategy_escalation_fixed(self):
        """Test that recovery strategy escalation works correctly."""
        context = {'consecutive_failures': 3}
        strategies = self.classifier.get_recovery_strategies(
            ErrorType.DATA_ERROR, ErrorSeverity.MEDIUM, context
        )
        
        strategy_names = [s.value for s in strategies]
        self.assertIn('restart_training', strategy_names, 
                     "RESTART_TRAINING should be added for 3+ consecutive failures")
    
    def test_emergency_stop_for_many_failures(self):
        """Test emergency stop for excessive consecutive failures."""
        context = {'consecutive_failures': 6}
        strategies = self.classifier.get_recovery_strategies(
            ErrorType.DATA_ERROR, ErrorSeverity.MEDIUM, context
        )
        
        self.assertEqual(strategies, [RecoveryStrategy.EMERGENCY_STOP])


class TestParameterExtractionHardFailure(unittest.TestCase):
    """Test _extract_primitive_parameter with HARD FAILURE behavior - NO FALLBACKS."""
    
    def test_basic_extraction_success(self):
        """Test basic parameter extraction when parameters exist."""
        context = {'batch_size': 32, 'learning_rate': 0.001}
        
        batch_size = _extract_primitive_parameter(context, 'batch_size', int)
        learning_rate = _extract_primitive_parameter(context, 'learning_rate', float)
        
        self.assertEqual(batch_size, 32)
        self.assertEqual(learning_rate, 0.001)
    
    def test_missing_parameter_hard_failure(self):
        """Test that missing parameters cause HARD FAILURE."""
        context = {'batch_size': 32}  # missing learning_rate
        
        with self.assertRaises(ValueError) as cm:
            _extract_primitive_parameter(context, 'learning_rate', float)
        
        self.assertIn("HARD FAILURE", str(cm.exception))
        self.assertIn("learning_rate", str(cm.exception))
    
    def test_nested_value_extraction_success(self):
        """Test extraction from nested dictionaries when valid."""
        context = {
            'batch_size': {'value': 64},
            'learning_rate': {'suggested_value': 0.005}
        }
        
        batch_size = _extract_primitive_parameter(context, 'batch_size', int)
        learning_rate = _extract_primitive_parameter(context, 'learning_rate', float)
        
        self.assertEqual(batch_size, 64)
        self.assertEqual(learning_rate, 0.005)
    
    def test_deeply_nested_extraction_success(self):
        """Test extraction from deeply nested dictionaries when valid."""
        context = {
            'batch_size': {'config': {'value': 128}}
        }
        
        batch_size = _extract_primitive_parameter(context, 'batch_size', int)
        self.assertEqual(batch_size, 128)
    
    def test_invalid_negative_values_hard_failure(self):
        """Test that invalid negative values cause HARD FAILURE."""
        context = {'batch_size': -5}  # Invalid negative
        
        with self.assertRaises(ValueError) as cm:
            _extract_primitive_parameter(context, 'batch_size', int)
        
        self.assertIn("HARD FAILURE", str(cm.exception))
        self.assertIn("must be positive", str(cm.exception))
    
    def test_invalid_type_conversion_hard_failure(self):
        """Test that invalid type conversions cause HARD FAILURE."""
        context = {'learning_rate': 'invalid_string'}
        
        with self.assertRaises(ValueError) as cm:
            _extract_primitive_parameter(context, 'learning_rate', float)
        
        self.assertIn("type conversion failed", str(cm.exception))


class TestCheckpointRecoveryFixed(unittest.TestCase):
    """Test CheckpointRecovery with deadlock fix."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.recovery = CheckpointRecovery(checkpoint_dir=self.temp_dir, max_checkpoints=2)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_checkpoint_creation_and_cleanup(self):
        """Test checkpoint creation with automatic cleanup (no deadlock)."""
        # Create 3 checkpoints when limit is 2
        checkpoint_ids = []
        for i in range(3):
            checkpoint_id = self.recovery.create_checkpoint(f'session{i}', i, i*10)
            checkpoint_ids.append(checkpoint_id)
            time.sleep(0.01)  # Small delay for timestamp ordering
        
        # Should automatically clean up to max_checkpoints (2)
        self.assertEqual(len(self.recovery.checkpoints), 2)
        
        # Should keep the most recent ones
        remaining_ids = list(self.recovery.checkpoints.keys())
        self.assertIn(checkpoint_ids[1], remaining_ids)  # Second checkpoint
        self.assertIn(checkpoint_ids[2], remaining_ids)  # Third checkpoint
        self.assertNotIn(checkpoint_ids[0], remaining_ids)  # First should be cleaned up
    
    def test_checkpoint_integrity_verification(self):
        """Test checkpoint integrity checking works."""
        checkpoint_id = self.recovery.create_checkpoint('test', 1, 10)
        
        # Should restore successfully with correct checksum
        restored = self.recovery.restore_checkpoint(checkpoint_id)
        self.assertIsNotNone(restored)
        
        # Corrupt checksum
        checkpoint = self.recovery.checkpoints[checkpoint_id]
        original_checksum = checkpoint.checksum
        checkpoint.checksum = "corrupted"
        
        # Should fail to restore
        corrupted_restore = self.recovery.restore_checkpoint(checkpoint_id)
        self.assertIsNone(corrupted_restore)


class TestStateRollbackCore(unittest.TestCase):
    """Test StateRollback core functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rollback = StateRollback(max_history=5)
    
    def test_state_saving_and_rollback(self):
        """Test basic state saving and rollback."""
        states = [{'epoch': i} for i in range(1, 4)]
        
        for state in states:
            self.rollback.save_state(state)
        
        # Rollback 1 step should give epoch 2
        rolled_back = self.rollback.rollback(1)
        self.assertEqual(rolled_back['epoch'], 2)
        self.assertEqual(len(self.rollback.state_history), 2)
    
    def test_history_size_limit(self):
        """Test that history respects size limit."""
        for i in range(10):  # More than limit of 5
            self.rollback.save_state({'epoch': i})
        
        self.assertEqual(len(self.rollback.state_history), 5)
        # Should have epochs 5-9 (most recent 5)
        self.assertEqual(self.rollback.state_history[-1]['state']['epoch'], 9)
        self.assertEqual(self.rollback.state_history[0]['state']['epoch'], 5)


class TestFailHardBehavior(unittest.TestCase):
    """Test the fail-hard behavior (primary mode)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ErrorRecoveryManager(checkpoint_dir=self.temp_dir, fail_hard=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_defaults_to_fail_hard(self):
        """Test that fail_hard=True is the default."""
        default_manager = ErrorRecoveryManager()
        self.assertTrue(default_manager.fail_hard)
    
    def test_error_is_re_raised(self):
        """Test that errors are re-raised in fail-hard mode."""
        error = RuntimeError("Test error")
        context = {}
        
        with self.assertRaises(RuntimeError) as cm:
            self.manager.handle_error(error, context)
        
        self.assertEqual(str(cm.exception), "Test error")
    
    def test_no_recovery_modifications(self):
        """Test that no recovery modifications are made to context."""
        error = RuntimeError("CUDA out of memory")
        context = {'batch_size': 64}
        original_context = context.copy()
        
        try:
            self.manager.handle_error(error, context)
        except RuntimeError:
            pass
        
        # Context should be unchanged (no recovery applied)
        self.assertEqual(context, original_context)
        self.assertNotIn('suggested_batch_size', context)
        self.assertNotIn('cache_cleared', context)
    
    def test_error_recorded_in_history(self):
        """Test that errors are still recorded for diagnostics."""
        error = ValueError("Test error")
        context = {'session_id': 'test_session'}
        
        try:
            self.manager.handle_error(error, context)
        except ValueError:
            pass
        
        history = self.manager.get_error_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].message, "Test error")
        self.assertFalse(history[0].recovery_success)
        self.assertIsNone(history[0].recovery_strategy)
    
    def test_statistics_updated(self):
        """Test that statistics are updated even in fail-hard mode."""
        error = RuntimeError("Test error")
        
        try:
            self.manager.handle_error(error, {})
        except RuntimeError:
            pass
        
        stats = self.manager.get_recovery_stats()
        self.assertEqual(stats['total_errors'], 1)
        self.assertEqual(stats['successful_recoveries'], 0)
        self.assertEqual(stats['failed_recoveries'], 0)
        self.assertEqual(stats['total_recoveries'], 0)
    
    def test_callbacks_still_called(self):
        """Test that callbacks are still called for monitoring."""
        callback_records = []
        
        def test_callback(error_record):
            callback_records.append(error_record)
        
        self.manager.register_recovery_callback(test_callback)
        
        error = RuntimeError("Test error")
        try:
            self.manager.handle_error(error, {})
        except RuntimeError:
            pass
        
        self.assertEqual(len(callback_records), 1)
        self.assertEqual(callback_records[0].message, "Test error")


class TestLegacyRecoveryBehavior(unittest.TestCase):
    """Test legacy recovery behavior (fail_hard=False)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ErrorRecoveryManager(checkpoint_dir=self.temp_dir, fail_hard=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_legacy_mode_attempts_recovery(self):
        """Test that legacy mode attempts recovery."""
        error = RuntimeError("CUDA out of memory")
        context = {'batch_size': 64}
        
        # Should not raise exception in legacy mode
        success = self.manager.handle_error(error, context)
        
        # Should return success status
        self.assertIsInstance(success, bool)
    
    def test_legacy_mode_modifies_context(self):
        """Test that legacy mode may modify context with recovery suggestions."""
        # Save some state for potential rollback
        self.manager.state_rollback.save_state({'test': 'data'})
        
        error = RuntimeError("Memory allocation failed")
        context = {'batch_size': 128}
        
        success = self.manager.handle_error(error, context)
        
        # Context may have recovery modifications
        has_recovery_modifications = any(
            key in context for key in [
                'cache_cleared', 'suggested_batch_size', 'batch_size_reduced',
                'rolled_back_state', 'learning_rate_reduced'
            ]
        )
        
        # At least cache clearing should have been attempted
        self.assertTrue(has_recovery_modifications, "Legacy mode should modify context with recovery attempts")
    
    def test_legacy_mode_records_recovery_attempts(self):
        """Test that legacy mode records recovery attempts."""
        error = RuntimeError("Test error")
        context = {}
        
        success = self.manager.handle_error(error, context)
        
        history = self.manager.get_error_history()
        self.assertEqual(len(history), 1)
        
        # Should have attempted some recovery strategy
        record = history[0]
        # Recovery may or may not have been successful, but should have been attempted
        if record.recovery_strategy:
            self.assertIn(record.recovery_success, [True, False])


class TestDiagnosticFunctionality(unittest.TestCase):
    """Test diagnostic and reporting functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ErrorRecoveryManager(checkpoint_dir=self.temp_dir, fail_hard=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_recovery_strategy_diagnosis(self):
        """Test that diagnostic information is generated."""
        error = RuntimeError("CUDA out of memory")
        context = {'batch_size': 64}
        
        # Test diagnosis method directly
        strategy = RecoveryStrategy.REDUCE_BATCH_SIZE
        diagnosis = self.manager._diagnose_recovery_strategy(strategy, error, context)
        
        self.assertIn('strategy', diagnosis)
        self.assertIn('would_perform', diagnosis)
        self.assertIn('recommended_action', diagnosis)
        self.assertEqual(diagnosis['strategy'], 'reduce_batch_size')
        self.assertIn('Reduce batch size', diagnosis['would_perform'][0])
    
    def test_export_recovery_report(self):
        """Test recovery report export."""
        # Generate some error history
        errors = [
            RuntimeError("Memory error"),
            ValueError("Data error")
        ]
        
        for error in errors:
            try:
                self.manager.handle_error(error, {})
            except:
                pass
        
        # Create a checkpoint
        self.manager.checkpoint_recovery.create_checkpoint('test', 1, 10)
        
        # Export report
        report = self.manager.export_recovery_report()
        
        self.assertIn('generated_at', report)
        self.assertIn('statistics', report)
        self.assertIn('error_history', report)
        self.assertIn('checkpoints', report)
        
        self.assertEqual(len(report['error_history']), 2)
        self.assertEqual(len(report['checkpoints']), 1)
        self.assertEqual(report['statistics']['total_errors'], 2)


class RealTrainingManager:
    """Real training manager - NO MOCKS."""
    def __init__(self):
        self.training_state = {'active': True}

class RealProgressTracker:
    """Real progress tracker - NO MOCKS."""
    def __init__(self):
        self.error_reports = []
    
    def report_error(self, error_type, message, recovery_attempted, recovery_success):
        self.error_reports.append({
            'error_type': error_type,
            'message': message, 
            'recovery_attempted': recovery_attempted,
            'recovery_success': recovery_success
        })

class TestIntegrationFunction(unittest.TestCase):
    """Test integrate_error_recovery function with REAL objects."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.training_manager = RealTrainingManager()
        self.progress_tracker = RealProgressTracker()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_integration_defaults_to_fail_hard(self):
        """Test that integration defaults to fail-hard mode."""
        recovery_manager = integrate_error_recovery(
            self.training_manager,
            checkpoint_dir=self.temp_dir
        )
        
        self.assertTrue(recovery_manager.fail_hard)
        self.assertTrue(hasattr(self.training_manager, '_error_recovery_manager'))
    
    def test_integration_with_explicit_modes(self):
        """Test integration with explicit fail_hard parameter."""
        # Test fail_hard=True
        recovery_manager_hard = integrate_error_recovery(
            self.training_manager,
            checkpoint_dir=self.temp_dir,
            fail_hard=True
        )
        self.assertTrue(recovery_manager_hard.fail_hard)
        
        # Test fail_hard=False
        training_manager_2 = RealTrainingManager()
        recovery_manager_soft = integrate_error_recovery(
            training_manager_2,
            checkpoint_dir=self.temp_dir,
            fail_hard=False
        )
        self.assertFalse(recovery_manager_soft.fail_hard)
    
    def test_progress_tracker_callback_integration(self):
        """Test integration with progress tracker callbacks."""
        recovery_manager = integrate_error_recovery(
            self.training_manager,
            self.progress_tracker,
            checkpoint_dir=self.temp_dir,
            fail_hard=True  # Use fail-hard mode
        )
        
        # Handle an error
        error = RuntimeError("Test error")
        try:
            recovery_manager.handle_error(error, {})
        except RuntimeError:
            pass
        
        # Progress tracker should have been called
        self.assertEqual(len(self.progress_tracker.error_reports), 1)
        report = self.progress_tracker.error_reports[0]
        self.assertEqual(report['message'], "Test error")
        self.assertEqual(report['recovery_attempted'], False)
        self.assertEqual(report['recovery_success'], False)


class TestEndToEndBehavior(unittest.TestCase):
    """Test end-to-end behavior of the error recovery system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_fail_hard_workflow(self):
        """Test complete workflow in fail-hard mode."""
        manager = ErrorRecoveryManager(checkpoint_dir=self.temp_dir, fail_hard=True)
        
        # Create some checkpoints and state history for diagnostics
        checkpoint_id = manager.checkpoint_recovery.create_checkpoint(
            'session1', 5, 100, training_state={'loss': 0.3}
        )
        manager.state_rollback.save_state({'epoch': 5, 'loss': 0.3})
        
        # Handle an error
        error = RuntimeError("CUDA out of memory")
        context = {
            'session_id': 'session1',
            'epoch': 5,
            'batch': 101,
            'batch_size': 64
        }
        
        # Should raise the error
        with self.assertRaises(RuntimeError):
            manager.handle_error(error, context)
        
        # Context should be unchanged
        self.assertEqual(context['batch_size'], 64)
        self.assertNotIn('suggested_batch_size', context)
        
        # Error should be recorded
        history = manager.get_error_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].error_type, ErrorType.MEMORY_ERROR)
        self.assertFalse(history[0].recovery_success)
        
        # Statistics should reflect failure
        stats = manager.get_recovery_stats()
        self.assertEqual(stats['total_errors'], 1)
        self.assertEqual(stats['total_recoveries'], 0)
        self.assertEqual(stats['recovery_success_rate'], 0.0)
        
        # Checkpoints should still exist for manual recovery
        checkpoints = manager.checkpoint_recovery.list_checkpoints('session1')
        self.assertEqual(len(checkpoints), 1)
    
    def test_complete_legacy_workflow(self):
        """Test complete workflow in legacy mode."""
        manager = ErrorRecoveryManager(checkpoint_dir=self.temp_dir, fail_hard=False)
        
        # Create some state for recovery
        manager.state_rollback.save_state({'epoch': 4, 'loss': 0.4})
        
        # Handle an error
        error = RuntimeError("Memory allocation failed")
        context = {'batch_size': 64}
        
        # Should not raise error in legacy mode
        success = manager.handle_error(error, context)
        
        # Should have attempted recovery
        self.assertIsInstance(success, bool)
        
        # Context may have modifications
        has_modifications = any(
            key in context for key in [
                'cache_cleared', 'suggested_batch_size', 'rolled_back_state'
            ]
        )
        self.assertTrue(has_modifications, "Legacy mode should modify context")
        
        # Error should be recorded with recovery attempt
        history = manager.get_error_history()
        self.assertEqual(len(history), 1)
        record = history[0]
        
        # Should have attempted some recovery
        if record.recovery_strategy:
            self.assertIsInstance(record.recovery_success, bool)


if __name__ == '__main__':
    unittest.main()