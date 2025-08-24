#!/usr/bin/env python3
"""
Comprehensive Test Suite for DomainState and DomainStateManager
Tests all core functionality with edge cases and integration scenarios
"""

import os
import sys
import json
import tempfile
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock
import traceback

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from domain_state import (
    DomainState, DomainStateManager, StateType, StateUpdateType,
    TensorJSONEncoder, StateVersion
)


class MockDomainRegistry:
    """Mock domain registry for testing"""
    
    def __init__(self):
        self.registered_domains = set()
        self.domain_states = {}
        self.domain_info = {}
    
    def is_domain_registered(self, domain_name: str) -> bool:
        # Always return True for test domains and UUIDs
        import re
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
        return domain_name.startswith('test_') or uuid_pattern.match(domain_name) or domain_name in self.registered_domains
    
    def get_domain_info(self, domain_name: str):
        if domain_name in self.domain_info:
            return self.domain_info[domain_name]
        return {
            'type': 'test',
            'version': 1,
            'config': {
                'max_memory_mb': 512,
                'hidden_layers': [64, 32, 16],
                'activation_function': 'relu',
                'learning_rate': 0.001,
                'dropout_rate': 0.1
            }
        }
    
    def set_domain_state(self, domain_name: str, state):
        self.domain_states[domain_name] = state
    
    def _ensure_isolation(self, domain_name: str) -> bool:
        return True
    
    def clear_domain_isolation(self, domain_name: str):
        pass
    
    def create_domain_isolation(self, domain_name: str):
        pass
    
    def register_domain(self, domain_name: str, domain_info: dict):
        self.registered_domains.add(domain_name)
        self.domain_info[domain_name] = domain_info


def run_comprehensive_domain_state_tests():
    """Run comprehensive test suite for DomainState components"""
    
    print("=" * 70)
    print("COMPREHENSIVE DOMAIN STATE TEST SUITE")
    print("=" * 70)
    
    test_results = []
    temp_dirs = []
    
    try:
        # Test 1: DomainState Basic Operations
        print("\n[TEST GROUP 1: DomainState Basic Operations]")
        
        # Test 1.1: Basic creation and properties
        try:
            state = DomainState("test_domain_1")
            assert state.domain_name == "test_domain_1"
            assert state.version == 1
            assert isinstance(state.created_at, datetime)
            assert isinstance(state.last_updated, datetime)
            test_results.append(("DomainState creation", "PASSED"))
        except Exception as e:
            test_results.append(("DomainState creation", f"FAILED: {e}"))
        
        # Test 1.2: State type operations
        try:
            state = DomainState("test_domain_2")
            
            # Test model parameters
            model_params = {"layer1": {"weights": np.random.randn(5, 3).tolist(), "bias": [0.1, 0.2, 0.3]}}
            state.update_state_by_type(StateType.MODEL_PARAMETERS, model_params)
            retrieved = state.get_state_by_type(StateType.MODEL_PARAMETERS)
            assert "layer1" in retrieved
            
            # Test learning state
            learning_state = {"optimizer": "adam", "lr": 0.001, "momentum": 0.9}
            state.update_state_by_type(StateType.LEARNING_STATE, learning_state)
            
            # Test performance metrics
            metrics = {"accuracy": 0.85, "loss": 0.15, "f1_score": 0.82}
            state.update_state_by_type(StateType.PERFORMANCE_METRICS, metrics)
            
            # Test embeddings with numpy arrays
            embeddings = {
                "word_vectors": np.random.randn(100, 50),
                "sentence_vectors": np.random.randn(50, 128)
            }
            state.update_state_by_type(StateType.EMBEDDINGS, embeddings)
            
            test_results.append(("State type operations", "PASSED"))
        except Exception as e:
            test_results.append(("State type operations", f"FAILED: {e}"))
        
        # Test 1.3: Update types
        try:
            state = DomainState("test_domain_3")
            
            # Test MERGE update
            initial_params = {"layer1": {"weights": [1, 2, 3]}}
            state.update_state_by_type(StateType.MODEL_PARAMETERS, initial_params)
            
            additional_params = {"layer2": {"weights": [4, 5, 6]}}
            state.update_state_by_type(StateType.MODEL_PARAMETERS, additional_params, StateUpdateType.MERGE)
            
            params = state.get_state_by_type(StateType.MODEL_PARAMETERS)
            assert "layer1" in params and "layer2" in params
            
            # Test FULL_REPLACE update
            replacement_params = {"layer3": {"weights": [7, 8, 9]}}
            state.update_state_by_type(StateType.MODEL_PARAMETERS, replacement_params, StateUpdateType.FULL_REPLACE)
            
            params = state.get_state_by_type(StateType.MODEL_PARAMETERS)
            assert "layer3" in params and "layer1" not in params
            
            # Test INCREMENTAL update
            initial_metrics = {"accuracy": 0.8, "loss": 0.2}
            state.update_state_by_type(StateType.PERFORMANCE_METRICS, initial_metrics)
            
            incremental_metrics = {"accuracy": 0.05, "loss": -0.02, "precision": 0.85}
            state.update_state_by_type(StateType.PERFORMANCE_METRICS, incremental_metrics, StateUpdateType.INCREMENTAL)
            
            metrics = state.get_state_by_type(StateType.PERFORMANCE_METRICS)
            assert abs(metrics["accuracy"] - 0.85) < 1e-6  # 0.8 + 0.05
            assert abs(metrics["loss"] - 0.18) < 1e-6      # 0.2 - 0.02
            assert metrics["precision"] == 0.85            # New value
            
            # Test APPEND update
            history_entry1 = {"epoch": 1, "loss": 0.5, "accuracy": 0.7}
            state.update_state_by_type(StateType.TRAINING_HISTORY, history_entry1, StateUpdateType.APPEND)
            
            history_entry2 = {"epoch": 2, "loss": 0.3, "accuracy": 0.8}
            state.update_state_by_type(StateType.TRAINING_HISTORY, history_entry2, StateUpdateType.APPEND)
            
            history = state.get_state_by_type(StateType.TRAINING_HISTORY)
            assert len(history) == 2
            assert history[0]["epoch"] == 1
            assert history[1]["epoch"] == 2
            
            test_results.append(("Update type operations", "PASSED"))
        except Exception as e:
            test_results.append(("Update type operations", f"FAILED: {e}"))
        
        # Test 1.4: Checkpoint operations
        try:
            state = DomainState("test_domain_4")
            
            # Set up some state
            state.update_state_by_type(StateType.MODEL_PARAMETERS, {"weights": [1, 2, 3]})
            state.update_state_by_type(StateType.PERFORMANCE_METRICS, {"accuracy": 0.9})
            state.total_training_steps = 1000
            state.best_performance = 0.9
            
            # Create checkpoint
            checkpoint = state.create_checkpoint("test_checkpoint", {"note": "test checkpoint"})
            assert checkpoint["name"] == "test_checkpoint"
            assert checkpoint["version"] == state.version
            assert checkpoint["total_training_steps"] == 1000
            assert "timestamp" in checkpoint
            
            # Modify state
            state.update_state_by_type(StateType.MODEL_PARAMETERS, {"weights": [4, 5, 6]}, StateUpdateType.FULL_REPLACE)
            state.update_state_by_type(StateType.PERFORMANCE_METRICS, {"accuracy": 0.7}, StateUpdateType.FULL_REPLACE)
            state.total_training_steps = 2000
            state.best_performance = 0.7
            
            # Restore checkpoint
            restored = state.restore_checkpoint("test_checkpoint")
            assert restored == True
            
            # Verify restoration
            params = state.get_state_by_type(StateType.MODEL_PARAMETERS)
            metrics = state.get_state_by_type(StateType.PERFORMANCE_METRICS)
            assert params["weights"] == [1, 2, 3]
            assert metrics["accuracy"] == 0.9
            assert state.total_training_steps == 1000
            assert state.best_performance == 0.9
            
            test_results.append(("Checkpoint operations", "PASSED"))
        except Exception as e:
            test_results.append(("Checkpoint operations", f"FAILED: {e}"))
        
        # Test 1.5: Size calculation
        try:
            state = DomainState("test_domain_5")
            
            # Add various types of data
            state.update_state_by_type(StateType.MODEL_PARAMETERS, {
                "layer1": {"weights": np.random.randn(100, 50).tolist()},
                "layer2": {"weights": np.random.randn(50, 25).tolist()}
            })
            state.update_state_by_type(StateType.EMBEDDINGS, {
                "vectors": np.random.randn(1000, 128)
            })
            
            size = state.calculate_size()
            assert size > 0
            assert state.state_size_bytes == size
            
            test_results.append(("Size calculation", "PASSED"))
        except Exception as e:
            test_results.append(("Size calculation", f"FAILED: {e}"))
        
        # Test 2: DomainStateManager Operations
        print("\n[TEST GROUP 2: DomainStateManager Operations]")
        
        # Test 2.1: Manager initialization
        try:
            temp_dir = tempfile.mkdtemp(prefix="domain_state_test_")
            temp_dirs.append(temp_dir)
            
            registry = MockDomainRegistry()
            manager = DomainStateManager(registry, storage_path=Path(temp_dir))
            
            assert manager.storage_path.exists()
            assert len(manager._domain_states) == 0
            
            test_results.append(("Manager initialization", "PASSED"))
        except Exception as e:
            test_results.append(("Manager initialization", f"FAILED: {e}"))
        
        # Test 2.2: Domain state retrieval and creation
        try:
            temp_dir = tempfile.mkdtemp(prefix="domain_state_test_")
            temp_dirs.append(temp_dir)
            
            registry = MockDomainRegistry()
            manager = DomainStateManager(registry, storage_path=Path(temp_dir))
            
            # Get non-existent domain (should create it)
            domain_state = manager.get_domain_state("test_domain_new")
            assert domain_state is not None
            assert domain_state.domain_name == "test_domain_new"
            assert "test_domain_new" in manager._domain_states
            
            # Get specific state type
            config = manager.get_domain_state("test_domain_new", StateType.CONFIGURATION)
            assert isinstance(config, dict)
            assert "domain_type" in config
            
            test_results.append(("Domain state retrieval", "PASSED"))
        except Exception as e:
            test_results.append(("Domain state retrieval", f"FAILED: {e}"))
        
        # Test 2.3: State updates via manager
        try:
            temp_dir = tempfile.mkdtemp(prefix="domain_state_test_")
            temp_dirs.append(temp_dir)
            
            registry = MockDomainRegistry()
            manager = DomainStateManager(registry, storage_path=Path(temp_dir))
            
            # Complex state update
            update_data = {
                "model_parameters": {
                    "layer1": {"weights": np.random.randn(10, 5).tolist(), "bias": [0.1] * 5},
                    "layer2": {"weights": np.random.randn(5, 3).tolist(), "bias": [0.2] * 3}
                },
                "learning_state": {
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "beta1": 0.9,
                    "beta2": 0.999
                },
                "performance_metrics": {
                    "accuracy": 0.92,
                    "loss": 0.08,
                    "precision": 0.89,
                    "recall": 0.94,
                    "f1_score": 0.915
                },
                "total_training_steps": 5000,
                "total_predictions": 50000,
                "best_performance": 0.92
            }
            
            success = manager.update_domain_state("test_complex_domain", update_data)
            assert success == True
            
            # Verify update
            domain_state = manager.get_domain_state("test_complex_domain")
            assert domain_state.total_training_steps == 5000
            assert domain_state.best_performance == 0.92
            
            params = domain_state.get_state_by_type(StateType.MODEL_PARAMETERS)
            assert "layer1" in params and "layer2" in params
            
            metrics = domain_state.get_state_by_type(StateType.PERFORMANCE_METRICS)
            assert metrics["accuracy"] == 0.92
            
            test_results.append(("State updates via manager", "PASSED"))
        except Exception as e:
            test_results.append(("State updates via manager", f"FAILED: {e}"))
        
        # Test 2.4: Save and load operations
        try:
            temp_dir = tempfile.mkdtemp(prefix="domain_state_test_")
            temp_dirs.append(temp_dir)
            
            registry = MockDomainRegistry()
            manager = DomainStateManager(registry, storage_path=Path(temp_dir))
            
            # Create and populate domain state
            update_data = {
                "model_parameters": {"weights": [1, 2, 3, 4, 5]},
                "performance_metrics": {"accuracy": 0.95},
                "total_training_steps": 10000
            }
            manager.update_domain_state("test_save_load", update_data)
            
            # Add embeddings
            domain_state = manager.get_domain_state("test_save_load")
            domain_state.update_state_by_type(StateType.EMBEDDINGS, {
                "test_embeddings": np.random.randn(100, 64)
            })
            
            # Save state
            save_success = manager.save_domain_state("test_save_load")
            assert save_success == True
            
            # Verify files were created
            json_file = Path(temp_dir) / "test_save_load_state.json"
            embeddings_file = Path(temp_dir) / "test_save_load_state.embeddings.npz"
            assert json_file.exists()
            assert embeddings_file.exists()
            
            # Clear state from memory
            del manager._domain_states["test_save_load"]
            
            # Load state
            load_success = manager.load_domain_state("test_save_load")
            assert load_success == True
            
            # Verify loaded data
            loaded_state = manager.get_domain_state("test_save_load")
            assert loaded_state.total_training_steps == 10000
            
            params = loaded_state.get_state_by_type(StateType.MODEL_PARAMETERS)
            assert params["weights"] == [1, 2, 3, 4, 5]
            
            metrics = loaded_state.get_state_by_type(StateType.PERFORMANCE_METRICS)
            assert metrics["accuracy"] == 0.95
            
            embeddings = loaded_state.get_state_by_type(StateType.EMBEDDINGS)
            assert "test_embeddings" in embeddings
            assert embeddings["test_embeddings"].shape == (100, 64)
            
            test_results.append(("Save and load operations", "PASSED"))
        except Exception as e:
            test_results.append(("Save and load operations", f"FAILED: {e}"))
        
        # Test 2.5: Checkpoint management via manager
        try:
            temp_dir = tempfile.mkdtemp(prefix="domain_state_test_")
            temp_dirs.append(temp_dir)
            
            registry = MockDomainRegistry()
            manager = DomainStateManager(registry, storage_path=Path(temp_dir))
            
            # Set up domain
            update_data = {"performance_metrics": {"accuracy": 0.8}, "total_training_steps": 1000}
            manager.update_domain_state("test_checkpoint_mgr", update_data)
            
            # Create checkpoint
            checkpoint_success = manager.create_checkpoint(
                "test_checkpoint_mgr", 
                "milestone_1",
                {"milestone": "first major improvement"}
            )
            assert checkpoint_success == True
            
            # Verify checkpoint file exists
            checkpoint_file = Path(temp_dir) / "test_checkpoint_mgr_checkpoint_milestone_1.json"
            assert checkpoint_file.exists()
            
            # Modify state
            manager.update_domain_state("test_checkpoint_mgr", {"performance_metrics": {"accuracy": 0.6}})
            
            # Restore checkpoint
            restore_success = manager.restore_checkpoint("test_checkpoint_mgr", "milestone_1")
            assert restore_success == True
            
            # Verify restoration
            domain_state = manager.get_domain_state("test_checkpoint_mgr")
            metrics = domain_state.get_state_by_type(StateType.PERFORMANCE_METRICS)
            assert metrics["accuracy"] == 0.8
            
            test_results.append(("Checkpoint management", "PASSED"))
        except Exception as e:
            test_results.append(("Checkpoint management", f"FAILED: {e}"))
        
        # Test 2.6: State reset
        try:
            temp_dir = tempfile.mkdtemp(prefix="domain_state_test_")
            temp_dirs.append(temp_dir)
            
            registry = MockDomainRegistry()
            manager = DomainStateManager(registry, storage_path=Path(temp_dir))
            
            # Populate domain with data
            update_data = {
                "model_parameters": {"complex_weights": list(range(100))},
                "performance_metrics": {"accuracy": 0.99, "loss": 0.01},
                "total_training_steps": 50000,
                "best_performance": 0.99
            }
            manager.update_domain_state("test_reset", update_data)
            
            # Verify data is there
            domain_state = manager.get_domain_state("test_reset")
            assert domain_state.total_training_steps == 50000
            assert len(domain_state.get_state_by_type(StateType.MODEL_PARAMETERS)["complex_weights"]) == 100
            
            # Reset state (preserve config)
            reset_success = manager.reset_domain_state("test_reset", preserve_config=True)
            assert reset_success == True
            
            # Verify reset
            reset_state = manager.get_domain_state("test_reset")
            assert reset_state.total_training_steps == 0
            assert reset_state.best_performance == 0.0
            
            # Configuration should be preserved
            config = reset_state.get_state_by_type(StateType.CONFIGURATION)
            assert "domain_type" in config  # From initial setup
            
            test_results.append(("State reset", "PASSED"))
        except Exception as e:
            test_results.append(("State reset", f"FAILED: {e}"))
        
        # Test 2.7: Statistics and analysis
        try:
            temp_dir = tempfile.mkdtemp(prefix="domain_state_test_")
            temp_dirs.append(temp_dir)
            
            registry = MockDomainRegistry()
            manager = DomainStateManager(registry, storage_path=Path(temp_dir))
            
            # Create rich domain state
            update_data = {
                "performance_metrics": {"accuracy": 0.87, "loss": 0.13, "f1_score": 0.85},
                "total_training_steps": 25000,
                "total_predictions": 250000,
                "best_performance": 0.87
            }
            manager.update_domain_state("test_stats", update_data)
            
            # Add training history
            for i in range(10):
                history_entry = {"epoch": i+1, "loss": 0.5 - i*0.04, "accuracy": 0.5 + i*0.04}
                manager.update_domain_state("test_stats", {"training_history": history_entry}, StateUpdateType.APPEND)
            
            # Add embeddings
            domain_state = manager.get_domain_state("test_stats")
            domain_state.update_state_by_type(StateType.EMBEDDINGS, {
                "word_vectors": np.random.randn(500, 100),
                "phrase_vectors": np.random.randn(200, 150)
            })
            
            # Add knowledge base items
            domain_state.update_state_by_type(StateType.KNOWLEDGE_BASE, {
                "facts": ["fact1", "fact2", "fact3"],
                "rules": {"rule1": "if A then B", "rule2": "if B then C"}
            })
            
            # Create checkpoints
            manager.create_checkpoint("test_stats", "checkpoint1")
            manager.create_checkpoint("test_stats", "checkpoint2")
            
            # Get statistics
            stats = manager.get_state_statistics("test_stats")
            
            assert stats["domain_name"] == "test_stats"
            assert stats["total_training_steps"] == 25000
            assert stats["total_predictions"] == 250000
            assert stats["best_performance"] == 0.87
            assert stats["training_history_length"] == 10
            assert stats["embedding_count"] == 2
            assert stats["checkpoint_count"] == 2
            assert "performance_summary" in stats
            assert stats["performance_summary"]["current_accuracy"] == 0.87
            
            test_results.append(("Statistics and analysis", "PASSED"))
        except Exception as e:
            test_results.append(("Statistics and analysis", f"FAILED: {e}"))
        
        # Test 3: Edge Cases and Error Handling
        print("\n[TEST GROUP 3: Edge Cases and Error Handling]")
        
        # Test 3.1: Invalid operations
        try:
            # Test invalid state type access
            state = DomainState("test_invalid")
            invalid_result = state.get_state_by_type("invalid_type")
            assert invalid_result == {}  # Should return empty dict for invalid type
            
            # Test non-existent checkpoint restoration
            restore_result = state.restore_checkpoint("non_existent_checkpoint")
            assert restore_result == False
            
            test_results.append(("Invalid operations handling", "PASSED"))
        except Exception as e:
            test_results.append(("Invalid operations handling", f"FAILED: {e}"))
        
        # Test 3.2: File system errors
        try:
            # Test save to non-existent directory
            registry = MockDomainRegistry()
            manager = DomainStateManager(registry, storage_path=Path("/non/existent/path"))
            
            update_data = {"performance_metrics": {"accuracy": 0.5}}
            manager.update_domain_state("test_fs_error", update_data)
            
            # This should handle the error gracefully
            save_result = manager.save_domain_state("test_fs_error")
            # Result may be True or False depending on implementation, but shouldn't crash
            
            test_results.append(("File system error handling", "PASSED"))
        except Exception as e:
            test_results.append(("File system error handling", f"FAILED: {e}"))
        
        # Test 3.3: Large data handling
        try:
            temp_dir = tempfile.mkdtemp(prefix="domain_state_test_")
            temp_dirs.append(temp_dir)
            
            registry = MockDomainRegistry()
            manager = DomainStateManager(registry, storage_path=Path(temp_dir))
            
            # Create large embeddings
            large_embeddings = {
                f"embedding_set_{i}": np.random.randn(1000, 100) 
                for i in range(5)
            }
            
            domain_state = manager.get_domain_state("test_large_data")
            domain_state.update_state_by_type(StateType.EMBEDDINGS, large_embeddings)
            
            # Large training history
            large_history = [
                {"epoch": i, "loss": np.random.random(), "accuracy": np.random.random()}
                for i in range(1000)
            ]
            domain_state.update_state_by_type(StateType.TRAINING_HISTORY, large_history, StateUpdateType.FULL_REPLACE)
            
            # Test size calculation
            size = domain_state.calculate_size()
            assert size > 10000  # Should be reasonably large
            
            # Test save/load with large data
            save_success = manager.save_domain_state("test_large_data")
            assert save_success == True
            
            test_results.append(("Large data handling", "PASSED"))
        except Exception as e:
            test_results.append(("Large data handling", f"FAILED: {e}"))
        
        # Test 3.4: Concurrent access simulation
        try:
            temp_dir = tempfile.mkdtemp(prefix="domain_state_test_")
            temp_dirs.append(temp_dir)
            
            registry = MockDomainRegistry()
            manager = DomainStateManager(registry, storage_path=Path(temp_dir))
            
            # Simulate multiple rapid updates (testing thread safety concepts)
            for i in range(50):
                update_data = {
                    "performance_metrics": {"accuracy": 0.5 + i * 0.01},
                    "total_training_steps": i * 100
                }
                success = manager.update_domain_state("test_concurrent", update_data)
                assert success == True
            
            # Verify final state
            final_state = manager.get_domain_state("test_concurrent")
            assert final_state.total_training_steps == 4900  # 49 * 100
            
            test_results.append(("Concurrent access simulation", "PASSED"))
        except Exception as e:
            test_results.append(("Concurrent access simulation", f"FAILED: {e}"))
        
        # Test 4: Integration Scenarios
        print("\n[TEST GROUP 4: Integration Scenarios]")
        
        # Test 4.1: Multi-domain management
        try:
            temp_dir = tempfile.mkdtemp(prefix="domain_state_test_")
            temp_dirs.append(temp_dir)
            
            registry = MockDomainRegistry()
            manager = DomainStateManager(registry, storage_path=Path(temp_dir))
            
            # Create multiple domains with different configurations
            domains = ["nlp_domain", "cv_domain", "rl_domain"]
            
            for domain in domains:
                domain_config = {
                    "model_parameters": {f"{domain}_weights": list(range(10))},
                    "performance_metrics": {"accuracy": np.random.random()},
                    "configuration": {"domain_specific_param": f"value_for_{domain}"}
                }
                manager.update_domain_state(domain, domain_config)
                
                # Create checkpoint for each
                manager.create_checkpoint(domain, f"{domain}_checkpoint")
            
            # Verify all domains exist
            for domain in domains:
                state = manager.get_domain_state(domain)
                assert state is not None
                assert state.domain_name == domain
            
            # Test export all states
            export_dir = Path(temp_dir) / "export"
            export_success = manager.export_all_states(export_dir)
            assert export_success == True
            
            # Verify export files
            assert (export_dir / "export_metadata.json").exists()
            for domain in domains:
                assert (export_dir / f"{domain}_state.json").exists()
            
            test_results.append(("Multi-domain management", "PASSED"))
        except Exception as e:
            test_results.append(("Multi-domain management", f"FAILED: {e}"))
        
        # Test 4.2: Session ID handling (UUID format)
        try:
            temp_dir = tempfile.mkdtemp(prefix="domain_state_test_")
            temp_dirs.append(temp_dir)
            
            registry = MockDomainRegistry()
            manager = DomainStateManager(registry, storage_path=Path(temp_dir))
            
            # Use UUID-format session ID
            session_id = "550e8400-e29b-41d4-a716-446655440000"
            
            update_data = {
                "performance_metrics": {"session_accuracy": 0.75},
                "total_predictions": 1000
            }
            
            success = manager.update_domain_state(session_id, update_data)
            assert success == True
            
            # Verify session state
            session_state = manager.get_domain_state(session_id)
            assert session_state is not None
            assert session_state.total_predictions == 1000
            
            test_results.append(("Session ID handling", "PASSED"))
        except Exception as e:
            test_results.append(("Session ID handling", f"FAILED: {e}"))
        
        # Test 4.3: Version and state history
        try:
            temp_dir = tempfile.mkdtemp(prefix="domain_state_test_")
            temp_dirs.append(temp_dir)
            
            registry = MockDomainRegistry()
            manager = DomainStateManager(registry, storage_path=Path(temp_dir))
            
            # Create domain and make multiple updates
            initial_data = {"performance_metrics": {"accuracy": 0.5}}
            manager.update_domain_state("test_versioning", initial_data)
            
            # Save initial version
            manager.save_domain_state("test_versioning")
            
            # Make updates and save versions
            for i in range(3):
                update_data = {"performance_metrics": {"accuracy": 0.5 + (i+1) * 0.1}}
                manager.update_domain_state("test_versioning", update_data)
                manager.save_domain_state("test_versioning")
            
            # Get version history
            versions = manager.get_state_versions("test_versioning")
            assert len(versions) >= 4  # At least 4 versions saved
            
            # Each version should have required fields
            for version in versions:
                assert hasattr(version, 'version_id')
                assert hasattr(version, 'timestamp')
                assert hasattr(version, 'size_bytes')
                assert hasattr(version, 'checksum')
            
            test_results.append(("Version and state history", "PASSED"))
        except Exception as e:
            test_results.append(("Version and state history", f"FAILED: {e}"))
        
        # Test 4.4: JSON serialization edge cases
        try:
            # Test TensorJSONEncoder with various data types
            encoder = TensorJSONEncoder()
            
            # Test with mock tensor-like object
            class MockTensor:
                def detach(self): return self
                def cpu(self): return self  
                def numpy(self): return np.array([1, 2, 3])
            
            mock_tensor = MockTensor()
            result = encoder.default(mock_tensor)
            assert result == [1, 2, 3]
            
            # Test with numpy array
            np_array = np.array([[1, 2], [3, 4]])
            result = encoder.default(np_array)
            assert result == [[1, 2], [3, 4]]
            
            # Test with numpy scalar
            np_scalar = np.float64(3.14)
            result = encoder.default(np_scalar)
            assert abs(result - 3.14) < 1e-10
            
            test_results.append(("JSON serialization edge cases", "PASSED"))
        except Exception as e:
            test_results.append(("JSON serialization edge cases", f"FAILED: {e}"))
        
    except Exception as e:
        test_results.append(("Test suite execution", f"FAILED: {e}"))
        traceback.print_exc()
    
    finally:
        # Clean up temporary directories
        for temp_dir in temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
    
    # Print final results
    print("\n" + "=" * 70)
    print("COMPREHENSIVE TEST RESULTS")
    print("=" * 70)
    
    passed_count = 0
    total_count = len(test_results)
    
    for test_name, result in test_results:
        status = "✓" if "PASSED" in result else "✗"
        print(f"{status} {test_name}: {result}")
        if "PASSED" in result:
            passed_count += 1
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    print(f"Success Rate: {(passed_count/total_count)*100:.1f}%")
    
    if passed_count == total_count:
        print("\n🎉 ALL COMPREHENSIVE TESTS PASSED!")
        print("DomainState and DomainStateManager are fully functional!")
    else:
        print(f"\n❌ {total_count - passed_count} TEST(S) FAILED")
        print("Review the failed tests above for issues.")
    
    return passed_count == total_count


if __name__ == "__main__":
    success = run_comprehensive_domain_state_tests()
    sys.exit(0 if success else 1)