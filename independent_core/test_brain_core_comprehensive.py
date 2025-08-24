#!/usr/bin/env python3
"""
Comprehensive test suite for BrainCore
Tests all functionality, integration, performance, and edge cases.
"""

import sys
import os
import traceback
import json
import time
import tempfile
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import numpy as np

# Add the independent_core to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_component(test_name: str, test_func, *args, **kwargs) -> Tuple[bool, str]:
    """Test a single component and return status."""
    try:
        test_func(*args, **kwargs)
        return True, "OK"
    except ImportError as e:
        return False, f"Import Error: {e}"
    except AttributeError as e:
        return False, f"Attribute Error: {e}"
    except Exception as e:
        # Get full traceback for debugging
        tb = traceback.format_exc()
        return False, f"Error: {e}\nTraceback: {tb[-500:]}"

class TestBrainCore:
    """Comprehensive test suite for BrainCore"""
    
    def __init__(self):
        self.results = {}
        self.failed_tests = []
        self.temp_dirs = []
        
    def cleanup(self):
        """Clean up temporary directories"""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def test_imports(self):
        """Test that all required imports work"""
        from brain_core import BrainCore, BrainConfig, PredictionResult, UncertaintyMetrics, UncertaintyType
        assert BrainCore is not None
        assert BrainConfig is not None
        assert PredictionResult is not None
        assert UncertaintyMetrics is not None
        assert UncertaintyType is not None
    
    def test_basic_initialization(self):
        """Test basic initialization with default parameters"""
        from brain_core import BrainCore, BrainConfig
        
        # Test default initialization
        brain = BrainCore()
        assert brain is not None
        assert brain.config is not None
        
        # Test with custom config
        config = BrainConfig(
            shared_memory_size=5000,
            reasoning_depth=3,
            enable_uncertainty=True,
            confidence_threshold=0.8
        )
        brain_custom = BrainCore(config)
        assert brain_custom.config.shared_memory_size == 5000
        assert brain_custom.config.reasoning_depth == 3
        assert brain_custom.config.enable_uncertainty == True
        assert brain_custom.config.confidence_threshold == 0.8
        
        # Test with dict config
        config_dict = {
            'shared_memory_size': 7000,
            'cache_size': 500
        }
        brain_dict = BrainCore(config_dict)
        assert brain_dict.config.shared_memory_size == 7000
        assert brain_dict.config.cache_size == 500
    
    def test_configuration_validation(self):
        """Test configuration validation"""
        from brain_core import BrainConfig
        
        # Test valid configurations
        valid_config = BrainConfig(
            shared_memory_size=1000,
            reasoning_depth=5,
            confidence_threshold=0.5
        )
        assert valid_config.shared_memory_size == 1000
        
        # Test invalid configurations
        try:
            invalid_config = BrainConfig(shared_memory_size=-1)
            assert False, "Should have raised ValueError for negative memory size"
        except ValueError as e:
            assert "shared_memory_size must be positive" in str(e)
        
        try:
            invalid_config = BrainConfig(reasoning_depth=0)
            assert False, "Should have raised ValueError for zero reasoning depth"
        except ValueError as e:
            assert "reasoning_depth must be positive" in str(e)
        
        try:
            invalid_config = BrainConfig(confidence_threshold=1.5)
            assert False, "Should have raised ValueError for invalid confidence threshold"
        except ValueError as e:
            assert "confidence_threshold must be between 0 and 1" in str(e)
    
    def test_prediction_basic(self):
        """Test basic prediction functionality"""
        from brain_core import BrainCore, PredictionResult
        
        brain = BrainCore()
        
        # Test with simple inputs
        result = brain.predict("test input")
        assert isinstance(result, PredictionResult)
        assert result.success in [True, False]
        assert result.prediction_id is not None
        assert 0.0 <= result.confidence <= 1.0
        
        # Test with dictionary input
        dict_input = {"key": "value", "number": 42}
        result = brain.predict(dict_input)
        assert isinstance(result, PredictionResult)
        
        # Test with list input
        list_input = [1, 2, 3, 4, 5]
        result = brain.predict(list_input)
        assert isinstance(result, PredictionResult)
        
        # Test with domain specification
        result = brain.predict("test", domain="test_domain")
        assert result.domain == "test_domain"
    
    def test_uncertainty_calculation(self):
        """Test uncertainty calculation functionality"""
        from brain_core import BrainCore, UncertaintyMetrics
        
        brain = BrainCore({"enable_uncertainty": True})
        
        # Test basic uncertainty calculation
        prediction_data = {
            "base_confidence": 0.8,
            "context": {"input_type": "string", "input_size": 10},
            "domain": "test"
        }
        
        uncertainty = brain.calculate_uncertainty(prediction_data)
        assert isinstance(uncertainty, UncertaintyMetrics)
        assert uncertainty.mean >= 0
        assert uncertainty.variance >= 0
        assert uncertainty.std >= 0
        assert len(uncertainty.confidence_interval) == 2
        assert uncertainty.confidence_interval[0] <= uncertainty.confidence_interval[1]
        assert 0.0 <= uncertainty.model_confidence <= 1.0
        assert uncertainty.epistemic_uncertainty >= 0
        assert uncertainty.aleatoric_uncertainty >= 0
    
    def test_shared_knowledge(self):
        """Test shared knowledge management"""
        from brain_core import BrainCore
        
        brain = BrainCore()
        
        # Test adding knowledge
        success = brain.add_shared_knowledge(
            key="test_key",
            value="test_value",
            domain="test_domain"
        )
        assert success == True
        
        # Test retrieving knowledge
        knowledge = brain.get_shared_knowledge("test_key")
        assert knowledge is not None
        assert knowledge['value'] == "test_value"
        assert knowledge['domain'] == "test_domain"
        
        # Test searching knowledge
        results = brain.search_shared_knowledge("test", max_results=5)
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Test updating knowledge
        success = brain.add_shared_knowledge(
            key="test_key",
            value="updated_value",
            domain="test_domain",
            overwrite=True
        )
        assert success == True
        
        knowledge = brain.get_shared_knowledge("test_key")
        assert knowledge['value'] == "updated_value"
    
    def test_state_management(self):
        """Test state saving and loading"""
        from brain_core import BrainCore
        
        brain = BrainCore()
        
        # Add some state
        brain.add_shared_knowledge("state_test", "value1")
        initial_state = brain.get_shared_state()
        
        # Create temp file for state
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        state_file = Path(temp_dir) / "brain_state.json"
        
        # Save state
        success = brain.save_state(state_file)
        assert success == True
        assert state_file.exists()
        
        # Create new brain and load state
        brain2 = BrainCore()
        success = brain2.load_state(state_file)
        assert success == True
        
        # Verify state was loaded
        knowledge = brain2.get_shared_knowledge("state_test")
        assert knowledge is not None
        assert knowledge['value'] == "value1"
        
        # Test state export/import
        exported = brain.export_state(format='json')
        assert exported is not None
        assert isinstance(exported, str)
        
        brain3 = BrainCore()
        success = brain3.import_state(exported, format='json')
        assert success == True
    
    def test_caching(self):
        """Test caching functionality"""
        from brain_core import BrainCore
        
        brain = BrainCore({"enable_caching": True, "cache_size": 100})
        
        # Make a prediction to cache
        input_data = "cache_test_input"
        result1 = brain.predict(input_data)
        
        # Make same prediction again (should be cached)
        start_time = time.time()
        result2 = brain.predict(input_data)
        cached_time = time.time() - start_time
        
        # Results should be similar
        assert result1.predicted_value == result2.predicted_value
        assert result1.confidence == result2.confidence
        
        # Clear cache
        brain.clear_cache()
        
        # After clearing, prediction should compute again
        result3 = brain.predict(input_data)
        assert isinstance(result3.prediction_id, str)
    
    def test_thread_safety(self):
        """Test thread safety of BrainCore operations"""
        from brain_core import BrainCore
        
        brain = BrainCore({"thread_safe": True})
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(10):
                    result = brain.predict(f"thread_{thread_id}_input_{i}")
                    brain.add_shared_knowledge(
                        f"thread_{thread_id}_key_{i}",
                        f"value_{i}"
                    )
                    results.append((thread_id, result.success))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Create and start threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join(timeout=10)
        
        # Check results
        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) > 0
    
    def test_checkpoints(self):
        """Test checkpoint creation and restoration"""
        from brain_core import BrainCore
        
        brain = BrainCore()
        
        # Add some state
        brain.add_shared_knowledge("checkpoint_test", "original_value")
        
        # Create checkpoint
        checkpoint_path = brain.create_checkpoint("test_checkpoint")
        assert checkpoint_path is not None
        assert checkpoint_path.exists()
        self.temp_dirs.append(checkpoint_path.parent)
        
        # Modify state
        brain.add_shared_knowledge("checkpoint_test", "modified_value", overwrite=True)
        brain.add_shared_knowledge("new_key", "new_value")
        
        # Restore checkpoint
        success = brain.restore_checkpoint(checkpoint_path)
        assert success == True
        
        # Verify state was restored
        knowledge = brain.get_shared_knowledge("checkpoint_test")
        assert knowledge['value'] == "original_value"
        
        # New key should not exist after restore
        new_knowledge = brain.get_shared_knowledge("new_key")
        assert new_knowledge is None
        
        # List checkpoints
        checkpoints = brain.list_checkpoints()
        assert isinstance(checkpoints, list)
    
    def test_auto_backup(self):
        """Test automatic backup functionality"""
        from brain_core import BrainCore
        
        brain = BrainCore()
        
        # Create temp directory for backups
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        # Enable auto backup with short interval
        success = brain.enable_auto_backup(
            interval_seconds=1,
            backup_path=temp_dir
        )
        assert success == True
        
        # Add some data
        brain.add_shared_knowledge("backup_test", "value")
        
        # Wait for backup to occur
        time.sleep(2)
        
        # Check backup status
        status = brain.get_backup_status()
        assert status['enabled'] == True
        assert status['interval_seconds'] == 1
        
        # Disable auto backup
        brain.disable_auto_backup()
        
        status = brain.get_backup_status()
        assert status['enabled'] == False
    
    def test_statistics(self):
        """Test statistics calculation"""
        from brain_core import BrainCore
        
        brain = BrainCore()
        
        # Make some predictions to generate statistics
        for i in range(5):
            brain.predict(f"test_input_{i}")
            brain.add_shared_knowledge(f"key_{i}", f"value_{i}")
        
        # Get statistics
        stats = brain.get_statistics()
        assert isinstance(stats, dict)
        assert 'total_predictions' in stats
        assert stats['total_predictions'] >= 5
        assert 'knowledge_entries' in stats
        assert 'average_confidence' in stats
        assert 'cache_stats' in stats
        
        # Get state summary
        summary = brain.get_state_summary()
        assert isinstance(summary, dict)
        assert 'config' in summary
        assert 'statistics' in summary
        assert 'memory_usage' in summary
    
    def test_config_updates(self):
        """Test configuration updates"""
        from brain_core import BrainCore
        
        brain = BrainCore()
        
        # Get initial config
        initial_config = brain.get_config()
        initial_cache_size = initial_config['cache_size']
        
        # Update config
        success = brain.update_config({
            'cache_size': initial_cache_size + 100,
            'confidence_threshold': 0.9
        })
        assert success == True
        
        # Verify updates
        updated_config = brain.get_config()
        assert updated_config['cache_size'] == initial_cache_size + 100
        assert updated_config['confidence_threshold'] == 0.9
        
        # Test invalid update
        success = brain.update_config({
            'confidence_threshold': 2.0  # Invalid value
        })
        assert success == False
    
    def test_reliability_assessment(self):
        """Test reliability assessment functionality"""
        from brain_core import BrainCore
        
        brain = BrainCore()
        
        prediction_data = {
            "predicted_value": "test_prediction",
            "base_confidence": 0.85,
            "domain": "test_domain"
        }
        
        # Test without historical data
        reliability = brain.assess_reliability(prediction_data)
        assert 0.0 <= reliability <= 1.0
        
        # Test with historical data
        historical = [
            {"confidence": 0.8, "success": True},
            {"confidence": 0.9, "success": True},
            {"confidence": 0.7, "success": False}
        ]
        reliability = brain.assess_reliability(prediction_data, historical)
        assert 0.0 <= reliability <= 1.0
    
    def test_context_management(self):
        """Test context preparation and management"""
        from brain_core import BrainCore
        
        brain = BrainCore()
        
        # Use private method for testing (normally not recommended)
        validated_input = "test_input"
        context = brain._prepare_prediction_context(validated_input, "test_domain")
        
        assert isinstance(context, dict)
        assert 'timestamp' in context
        assert 'domain' in context
        assert context['domain'] == "test_domain"
        assert 'input_characteristics' in context
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        from brain_core import BrainCore
        
        brain = BrainCore()
        
        # Test with None input
        result = brain.predict(None)
        assert isinstance(result.prediction_id, str)
        
        # Test with empty string
        result = brain.predict("")
        assert isinstance(result.prediction_id, str)
        
        # Test with very large input
        large_input = "x" * 10000
        result = brain.predict(large_input)
        assert isinstance(result.prediction_id, str)
        
        # Test with special characters
        special_input = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        result = brain.predict(special_input)
        assert isinstance(result.prediction_id, str)
        
        # Test with nested structures
        nested = {"a": {"b": {"c": {"d": {"e": "value"}}}}}
        result = brain.predict(nested)
        assert isinstance(result.prediction_id, str)
    
    def test_memory_persistence(self):
        """Test knowledge persistence functionality"""
        from brain_core import BrainCore, BrainConfig
        
        # Create temp directory for persistence
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        knowledge_path = Path(temp_dir) / "knowledge.json"
        
        # Create brain with persistence
        config = BrainConfig(
            knowledge_persistence=True,
            knowledge_path=knowledge_path
        )
        brain1 = BrainCore(config)
        
        # Add knowledge
        brain1.add_shared_knowledge("persist_key", "persist_value")
        
        # Force persistence
        brain1._persist_knowledge()
        
        # Create new brain with same config
        brain2 = BrainCore(config)
        
        # Knowledge should be loaded
        knowledge = brain2.get_shared_knowledge("persist_key")
        assert knowledge is not None
        assert knowledge['value'] == "persist_value"
    
    def test_performance(self):
        """Test performance benchmarks"""
        from brain_core import BrainCore
        
        brain = BrainCore()
        
        # Test prediction performance
        start_time = time.time()
        for i in range(100):
            brain.predict(f"perf_test_{i}")
        prediction_time = time.time() - start_time
        avg_prediction_time = prediction_time / 100
        
        print(f"\nAverage prediction time: {avg_prediction_time*1000:.2f}ms")
        assert avg_prediction_time < 0.1  # Should be < 100ms per prediction
        
        # Test knowledge operations performance
        start_time = time.time()
        for i in range(100):
            brain.add_shared_knowledge(f"perf_key_{i}", f"value_{i}")
        add_time = time.time() - start_time
        
        start_time = time.time()
        for i in range(100):
            brain.get_shared_knowledge(f"perf_key_{i}")
        get_time = time.time() - start_time
        
        print(f"Knowledge add time: {add_time*10:.2f}ms per 100 ops")
        print(f"Knowledge get time: {get_time*10:.2f}ms per 100 ops")
        
        assert add_time < 1.0  # Should be < 1s for 100 operations
        assert get_time < 0.5  # Getting should be faster
    
    def test_reset_functionality(self):
        """Test state reset functionality"""
        from brain_core import BrainCore
        
        brain = BrainCore()
        
        # Add state
        brain.add_shared_knowledge("reset_test", "value")
        brain.predict("test_input")
        
        # Get initial stats
        stats_before = brain.get_statistics()
        assert stats_before['total_predictions'] > 0
        
        # Reset state
        brain.reset_state(preserve_config=True)
        
        # Check state was reset
        knowledge = brain.get_shared_knowledge("reset_test")
        assert knowledge is None
        
        stats_after = brain.get_statistics()
        assert stats_after['total_predictions'] == 0
        
        # Config should be preserved
        config = brain.get_config()
        assert config is not None
    
    def test_context_manager(self):
        """Test context manager functionality"""
        from brain_core import BrainCore
        
        # Test using with statement
        with BrainCore() as brain:
            brain.add_shared_knowledge("context_test", "value")
            result = brain.predict("test")
            assert result is not None
        
        # Brain should handle cleanup properly
        assert True  # If we get here, context manager worked
    
    def run_all_tests(self):
        """Run all test methods"""
        test_methods = [
            ('Imports', self.test_imports),
            ('Basic Initialization', self.test_basic_initialization),
            ('Configuration Validation', self.test_configuration_validation),
            ('Basic Prediction', self.test_prediction_basic),
            ('Uncertainty Calculation', self.test_uncertainty_calculation),
            ('Shared Knowledge', self.test_shared_knowledge),
            ('State Management', self.test_state_management),
            ('Caching', self.test_caching),
            ('Thread Safety', self.test_thread_safety),
            ('Checkpoints', self.test_checkpoints),
            ('Auto Backup', self.test_auto_backup),
            ('Statistics', self.test_statistics),
            ('Config Updates', self.test_config_updates),
            ('Reliability Assessment', self.test_reliability_assessment),
            ('Context Management', self.test_context_management),
            ('Edge Cases', self.test_edge_cases),
            ('Memory Persistence', self.test_memory_persistence),
            ('Performance', self.test_performance),
            ('Reset Functionality', self.test_reset_functionality),
            ('Context Manager', self.test_context_manager)
        ]
        
        print("=" * 80)
        print("COMPREHENSIVE BRAIN CORE TESTING")
        print("=" * 80)
        
        passed = 0
        failed = 0
        
        for test_name, test_method in test_methods:
            success, msg = test_component(test_name, test_method)
            status = "✅" if success else "❌"
            print(f"{status} {test_name:30} - {msg}")
            
            self.results[test_name] = (success, msg)
            if success:
                passed += 1
            else:
                failed += 1
                self.failed_tests.append((test_name, msg))
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {len(test_methods)}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        
        if failed > 0:
            print("\nFAILED TESTS:")
            for test_name, error in self.failed_tests:
                print(f"  - {test_name}: {error[:200]}")
        
        print("\n" + "=" * 80)
        print("ACTION ITEMS")
        print("=" * 80)
        
        if failed > 0:
            print(f"\nIssues to fix (in order of priority):")
            for i, (test_name, error) in enumerate(self.failed_tests, 1):
                print(f"{i}. {test_name}: {error[:100]}")
        else:
            print("\n✅ All BrainCore tests passed!")
        
        # Cleanup
        self.cleanup()
        
        return passed, failed

def main():
    """Main test execution"""
    tester = TestBrainCore()
    passed, failed = tester.run_all_tests()
    
    # Exit with error code if there are failures
    sys.exit(failed)

if __name__ == "__main__":
    main()