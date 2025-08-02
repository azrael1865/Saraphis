"""
Comprehensive Test Suite for Universal AI Core Brain System.
Tests all components including isolation, anti-catastrophic forgetting, and system integration.
"""

import pytest
import tempfile
import shutil
import json
import numpy as np
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional

# Import Brain system components
from brain import Brain, BrainSystemConfig
from brain_core import BrainCore, BrainConfig, PredictionResult, UncertaintyMetrics
from domain_registry import DomainRegistry, DomainConfig, DomainStatus, DomainType
from domain_router import DomainRouter, RoutingStrategy
from domain_state import DomainStateManager, DomainState, StateType, StateUpdateType
from training_manager import TrainingManager, TrainingConfig, TrainingStatus


class TestBrainInitialization:
    """Test Brain system initialization and configuration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def basic_config(self, temp_dir):
        """Create basic configuration for testing."""
        return BrainSystemConfig(
            base_path=temp_dir / ".brain_test",
            enable_persistence=True,
            enable_monitoring=True,
            max_domains=10
        )
    
    def test_brain_initialization_success(self, basic_config):
        """Test successful Brain initialization."""
        brain = Brain(config=basic_config)
        
        assert brain.config == basic_config
        assert brain.is_initialized
        assert len(brain.list_available_domains()) == 0
        
        # Check component initialization
        assert brain.brain_core is not None
        assert brain.domain_registry is not None
        assert brain.domain_router is not None
        assert brain.domain_state_manager is not None
        assert brain.training_manager is not None
    
    def test_brain_initialization_with_invalid_config(self):
        """Test Brain initialization with invalid configuration."""
        invalid_config = BrainSystemConfig(max_domains=-1)
        
        with pytest.raises(ValueError):
            Brain(config=invalid_config)
    
    def test_brain_initialization_without_config(self, temp_dir):
        """Test Brain initialization with default configuration."""
        brain = Brain()
        
        assert brain.is_initialized
        assert brain.config is not None
        assert brain.config.base_path.exists()
    
    def test_brain_directory_creation(self, basic_config):
        """Test that Brain creates necessary directories."""
        brain = Brain(config=basic_config)
        
        assert basic_config.base_path.exists()
        assert basic_config.knowledge_path.exists()
        assert basic_config.models_path.exists()
        assert basic_config.training_path.exists()
        assert basic_config.logs_path.exists()


class TestDomainManagement:
    """Test domain registration, management, and isolation."""
    
    @pytest.fixture
    def brain(self, temp_dir):
        """Create Brain instance for testing."""
        config = BrainSystemConfig(
            base_path=temp_dir / ".brain_test",
            enable_persistence=True,
            enable_isolation=True
        )
        return Brain(config=config)
    
    @pytest.fixture
    def domain_config(self):
        """Create domain configuration for testing."""
        return DomainConfig(
            domain_type=DomainType.STANDARD,
            description="Test domain for unit testing",
            hidden_layers=[128, 64, 32],
            learning_rate=0.001,
            max_memory_mb=256
        )
    
    def test_domain_registration_success(self, brain, domain_config):
        """Test successful domain registration."""
        result = brain.register_domain("test_domain", domain_config)
        
        assert result["success"] is True
        assert result["domain_name"] == "test_domain"
        assert "test_domain" in [d["name"] for d in brain.list_available_domains()]
    
    def test_domain_registration_duplicate(self, brain, domain_config):
        """Test registration of duplicate domain names."""
        brain.register_domain("test_domain", domain_config)
        
        result = brain.register_domain("test_domain", domain_config)
        assert result["success"] is False
        assert "already registered" in result["message"].lower()
    
    def test_domain_isolation_validation(self, brain, domain_config):
        """Test that domains are properly isolated."""
        # Register two domains
        brain.register_domain("domain_a", domain_config)
        brain.register_domain("domain_b", domain_config)
        
        # Get domain capabilities
        caps_a = brain.get_domain_capabilities("domain_a")
        caps_b = brain.get_domain_capabilities("domain_b")
        
        # Verify isolation
        assert caps_a["isolation"]["enabled"] is True
        assert caps_b["isolation"]["enabled"] is True
        assert caps_a["isolation"]["namespace"] != caps_b["isolation"]["namespace"]
    
    def test_domain_deregistration(self, brain, domain_config):
        """Test domain deregistration and cleanup."""
        brain.register_domain("test_domain", domain_config)
        
        # Verify domain exists
        domains = brain.list_available_domains()
        assert "test_domain" in [d["name"] for d in domains]
        
        # Deregister domain
        result = brain.deregister_domain("test_domain")
        assert result["success"] is True
        
        # Verify domain removed
        domains = brain.list_available_domains()
        assert "test_domain" not in [d["name"] for d in domains]
    
    def test_max_domains_limit(self, brain, domain_config):
        """Test maximum domains limit enforcement."""
        brain.config.max_domains = 2
        
        # Register domains up to limit
        brain.register_domain("domain_1", domain_config)
        brain.register_domain("domain_2", domain_config)
        
        # Try to register beyond limit
        result = brain.register_domain("domain_3", domain_config)
        assert result["success"] is False
        assert "maximum" in result["message"].lower()


class TestPredictionFunctionality:
    """Test prediction capabilities across domains."""
    
    @pytest.fixture
    def brain_with_domains(self, temp_dir):
        """Create Brain with registered domains."""
        config = BrainSystemConfig(
            base_path=temp_dir / ".brain_test",
            enable_persistence=True,
            enable_parallel_predictions=True
        )
        brain = Brain(config=config)
        
        # Register test domains
        domain_config = DomainConfig(
            hidden_layers=[64, 32],
            learning_rate=0.001
        )
        
        brain.register_domain("classification", domain_config)
        brain.register_domain("regression", domain_config)
        
        return brain
    
    def test_single_domain_prediction(self, brain_with_domains):
        """Test prediction on single domain."""
        test_data = {
            "features": [1.0, 2.0, 3.0, 4.0],
            "context": "test prediction"
        }
        
        result = brain_with_domains.predict("classification", test_data)
        
        assert result.success is True
        assert result.domain_name == "classification"
        assert "prediction" in result.result
        assert "confidence" in result.result
        assert "uncertainty" in result.result
    
    def test_multi_domain_prediction(self, brain_with_domains):
        """Test prediction across multiple domains."""
        test_data = {
            "features": [1.0, 2.0, 3.0, 4.0],
            "context": "multi-domain test"
        }
        
        results = brain_with_domains.predict_multi_domain(
            ["classification", "regression"], 
            test_data
        )
        
        assert len(results) == 2
        for result in results:
            assert result.success is True
            assert result.domain_name in ["classification", "regression"]
    
    def test_prediction_confidence_analysis(self, brain_with_domains):
        """Test prediction confidence analysis."""
        test_data = {
            "features": [1.0, 2.0, 3.0, 4.0],
            "context": "confidence test"
        }
        
        result = brain_with_domains.predict("classification", test_data)
        confidence_analysis = brain_with_domains.get_prediction_confidence(result)
        
        assert "overall_confidence" in confidence_analysis
        assert "uncertainty_breakdown" in confidence_analysis
        assert "reliability_score" in confidence_analysis
        assert "confidence_intervals" in confidence_analysis
    
    def test_prediction_with_invalid_domain(self, brain_with_domains):
        """Test prediction on non-existent domain."""
        test_data = {"features": [1.0, 2.0, 3.0]}
        
        result = brain_with_domains.predict("nonexistent_domain", test_data)
        
        assert result.success is False
        assert "not found" in result.error_message.lower()
    
    def test_concurrent_predictions(self, brain_with_domains):
        """Test concurrent prediction handling."""
        test_data = {"features": [1.0, 2.0, 3.0, 4.0]}
        
        def make_prediction():
            return brain_with_domains.predict("classification", test_data)
        
        # Run concurrent predictions
        threads = []
        results = []
        
        for _ in range(5):
            thread = threading.Thread(target=lambda: results.append(make_prediction()))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify all predictions succeeded
        assert len(results) == 5
        for result in results:
            assert result.success is True


class TestTrainingWithoutForgetting:
    """Test training capabilities with anti-catastrophic forgetting."""
    
    @pytest.fixture
    def brain_for_training(self, temp_dir):
        """Create Brain configured for training tests."""
        config = BrainSystemConfig(
            base_path=temp_dir / ".brain_test",
            enable_persistence=True,
            max_concurrent_training=2,
            enable_adaptation=True
        )
        brain = Brain(config=config)
        
        # Register domains for training
        domain_config = DomainConfig(
            hidden_layers=[32, 16],
            learning_rate=0.01,
            max_memory_mb=128
        )
        
        brain.register_domain("task_a", domain_config)
        brain.register_domain("task_b", domain_config)
        
        return brain
    
    def test_single_domain_training(self, brain_for_training):
        """Test training on single domain."""
        # Prepare training data
        training_data = {
            "X": np.random.rand(100, 10).tolist(),
            "y": np.random.randint(0, 2, 100).tolist()
        }
        
        training_config = TrainingConfig(
            epochs=10,
            batch_size=16,
            learning_rate=0.01
        )
        
        result = brain_for_training.train_domain(
            "task_a", 
            training_data, 
            training_config
        )
        
        assert result["success"] is True
        assert result["domain_name"] == "task_a"
        assert "training_id" in result
        assert "status" in result
    
    def test_catastrophic_forgetting_prevention(self, brain_for_training):
        """Test that training new tasks doesn't forget old ones."""
        # Create distinct training datasets
        task_a_data = {
            "X": np.random.rand(50, 10).tolist(),
            "y": [0] * 50  # All class 0
        }
        
        task_b_data = {
            "X": np.random.rand(50, 10).tolist(),
            "y": [1] * 50  # All class 1
        }
        
        training_config = TrainingConfig(epochs=5, batch_size=8)
        
        # Train on task A
        result_a = brain_for_training.train_domain("task_a", task_a_data, training_config)
        assert result_a["success"] is True
        
        # Get performance on task A after training
        test_data_a = {"features": task_a_data["X"][0]}
        prediction_a_before = brain_for_training.predict("task_a", test_data_a)
        
        # Train on task B
        result_b = brain_for_training.train_domain("task_b", task_b_data, training_config)
        assert result_b["success"] is True
        
        # Test that task A performance is preserved
        prediction_a_after = brain_for_training.predict("task_a", test_data_a)
        
        # Verify predictions are still reasonable (no catastrophic forgetting)
        assert prediction_a_after.success is True
        assert abs(prediction_a_before.result["confidence"] - 
                  prediction_a_after.result["confidence"]) < 0.5
    
    def test_concurrent_training_limit(self, brain_for_training):
        """Test concurrent training limit enforcement."""
        training_data = {
            "X": np.random.rand(50, 10).tolist(),
            "y": np.random.randint(0, 2, 50).tolist()
        }
        
        training_config = TrainingConfig(epochs=20, batch_size=8)
        
        # Start first training (should succeed)
        result1 = brain_for_training.train_domain("task_a", training_data, training_config)
        assert result1["success"] is True
        
        # Start second training (should succeed)  
        result2 = brain_for_training.train_domain("task_b", training_data, training_config)
        assert result2["success"] is True
        
        # Try third training (should be queued or rejected due to limit)
        domain_config = DomainConfig(hidden_layers=[16])
        brain_for_training.register_domain("task_c", domain_config)
        
        result3 = brain_for_training.train_domain("task_c", training_data, training_config)
        
        # Should either be queued or rejected
        assert ("queued" in result3.get("message", "").lower() or 
                "limit" in result3.get("message", "").lower())
    
    def test_training_status_monitoring(self, brain_for_training):
        """Test training status monitoring."""
        training_data = {
            "X": np.random.rand(30, 10).tolist(),
            "y": np.random.randint(0, 2, 30).tolist()
        }
        
        training_config = TrainingConfig(epochs=5, batch_size=8)
        
        result = brain_for_training.train_domain("task_a", training_data, training_config)
        training_id = result["training_id"]
        
        # Monitor training status
        status = brain_for_training.get_training_status(training_id)
        
        assert "status" in status
        assert "progress" in status
        assert "domain_name" in status
        assert status["domain_name"] == "task_a"


class TestStatePersistence:
    """Test state saving and loading functionality."""
    
    @pytest.fixture
    def brain_with_state(self, temp_dir):
        """Create Brain with some state for testing."""
        config = BrainSystemConfig(
            base_path=temp_dir / ".brain_test",
            enable_persistence=True
        )
        brain = Brain(config=config)
        
        # Add domains and some training
        domain_config = DomainConfig(hidden_layers=[32, 16])
        brain.register_domain("persistent_domain", domain_config)
        
        # Add some training data to create state
        training_data = {
            "X": np.random.rand(20, 5).tolist(),
            "y": np.random.randint(0, 2, 20).tolist()
        }
        training_config = TrainingConfig(epochs=3, batch_size=4)
        brain.train_domain("persistent_domain", training_data, training_config)
        
        return brain
    
    def test_save_brain_state(self, brain_with_state, temp_dir):
        """Test saving complete Brain state."""
        save_path = temp_dir / "brain_state.json"
        
        result = brain_with_state.save_brain_state(save_path)
        
        assert result["success"] is True
        assert save_path.exists()
        
        # Verify saved data structure
        with open(save_path, 'r') as f:
            saved_data = json.load(f)
        
        assert "brain_config" in saved_data
        assert "domains" in saved_data
        assert "domain_states" in saved_data
        assert "system_metrics" in saved_data
        assert "save_timestamp" in saved_data
    
    def test_load_brain_state(self, brain_with_state, temp_dir):
        """Test loading complete Brain state."""
        save_path = temp_dir / "brain_state.json"
        
        # Save current state
        brain_with_state.save_brain_state(save_path)
        
        # Create new Brain instance
        new_config = BrainSystemConfig(base_path=temp_dir / ".brain_test2")
        new_brain = Brain(config=new_config)
        
        # Load state
        result = new_brain.load_brain_state(save_path)
        
        assert result["success"] is True
        assert len(new_brain.list_available_domains()) > 0
        
        # Verify domain exists and can make predictions
        domains = new_brain.list_available_domains()
        domain_names = [d["name"] for d in domains]
        assert "persistent_domain" in domain_names
    
    def test_state_versioning(self, brain_with_state, temp_dir):
        """Test state versioning functionality."""
        save_path = temp_dir / "brain_state_v1.json"
        
        # Save initial state
        result1 = brain_with_state.save_brain_state(save_path)
        
        # Modify state
        test_data = {"features": [1.0, 2.0, 3.0, 4.0, 5.0]}
        brain_with_state.predict("persistent_domain", test_data)
        
        # Save updated state
        save_path_v2 = temp_dir / "brain_state_v2.json"
        result2 = brain_with_state.save_brain_state(save_path_v2)
        
        assert result1["success"] is True
        assert result2["success"] is True
        
        # Verify different versions
        with open(save_path, 'r') as f:
            state_v1 = json.load(f)
        with open(save_path_v2, 'r') as f:
            state_v2 = json.load(f)
        
        assert state_v1["save_timestamp"] != state_v2["save_timestamp"]
    
    def test_backup_and_restore(self, brain_with_state, temp_dir):
        """Test backup and restore functionality."""
        backup_path = temp_dir / "backup"
        
        # Create backup
        result = brain_with_state.create_backup(backup_path)
        
        assert result["success"] is True
        assert backup_path.exists()
        
        # Verify backup contains expected files
        backup_files = list(backup_path.glob("*"))
        assert len(backup_files) > 0


class TestSystemMonitoring:
    """Test system monitoring and health check capabilities."""
    
    @pytest.fixture
    def monitored_brain(self, temp_dir):
        """Create Brain with monitoring enabled."""
        config = BrainSystemConfig(
            base_path=temp_dir / ".brain_test",
            enable_monitoring=True,
            enable_persistence=True
        )
        brain = Brain(config=config)
        
        # Add domain for monitoring tests
        domain_config = DomainConfig(hidden_layers=[32, 16])
        brain.register_domain("monitored_domain", domain_config)
        
        return brain
    
    def test_health_status_check(self, monitored_brain):
        """Test comprehensive health status check."""
        health = monitored_brain.get_health_status()
        
        assert "overall_health" in health
        assert "grade" in health["overall_health"]
        assert "score" in health["overall_health"]
        assert "components" in health
        
        # Check component health
        components = health["components"]
        expected_components = ["brain_core", "domain_registry", "domain_router", 
                             "domain_state_manager", "training_manager"]
        
        for component in expected_components:
            assert component in components
            assert "health_score" in components[component]
            assert "status" in components[component]
    
    def test_performance_metrics(self, monitored_brain):
        """Test performance metrics collection."""
        # Generate some activity
        test_data = {"features": [1.0, 2.0, 3.0]}
        monitored_brain.predict("monitored_domain", test_data)
        
        metrics = monitored_brain.get_performance_metrics()
        
        assert "system_performance" in metrics
        assert "domain_performance" in metrics
        assert "resource_usage" in metrics
        assert "prediction_stats" in metrics
        
        # Check prediction statistics
        pred_stats = metrics["prediction_stats"]
        assert "total_predictions" in pred_stats
        assert "average_response_time" in pred_stats
        assert "success_rate" in pred_stats
    
    def test_resource_monitoring(self, monitored_brain):
        """Test resource usage monitoring."""
        resource_usage = monitored_brain.get_resource_usage()
        
        assert "memory" in resource_usage
        assert "cpu" in resource_usage
        assert "disk" in resource_usage
        assert "network" in resource_usage
        
        # Check memory metrics
        memory = resource_usage["memory"]
        assert "total_mb" in memory
        assert "used_mb" in memory
        assert "available_mb" in memory
        assert "usage_percent" in memory
    
    def test_diagnostic_check(self, monitored_brain):
        """Test comprehensive diagnostic check."""
        diagnostic = monitored_brain.run_diagnostic()
        
        assert "overall_status" in diagnostic
        assert "checks_performed" in diagnostic
        assert "issues_found" in diagnostic
        assert "recommendations" in diagnostic
        
        # Verify diagnostic checks
        checks = diagnostic["checks_performed"]
        expected_checks = ["component_health", "memory_usage", "disk_space", 
                          "domain_integrity", "state_consistency"]
        
        for check in expected_checks:
            assert any(check in performed for performed in checks)
    
    def test_domain_health_monitoring(self, monitored_brain):
        """Test domain-specific health monitoring."""
        domain_health = monitored_brain.get_domain_health("monitored_domain")
        
        assert "domain_name" in domain_health
        assert "health_score" in domain_health
        assert "status" in domain_health
        assert "performance_metrics" in domain_health
        assert "resource_usage" in domain_health
        assert "last_activity" in domain_health


class TestAdvancedFeatures:
    """Test advanced Brain features and edge cases."""
    
    @pytest.fixture
    def advanced_brain(self, temp_dir):
        """Create Brain with advanced features enabled."""
        config = BrainSystemConfig(
            base_path=temp_dir / ".brain_test",
            enable_persistence=True,
            enable_monitoring=True,
            enable_adaptation=True,
            enable_parallel_predictions=True,
            max_prediction_threads=4
        )
        brain = Brain(config=config)
        
        # Register multiple domains
        domain_config = DomainConfig(hidden_layers=[64, 32, 16])
        brain.register_domain("advanced_domain_1", domain_config)
        brain.register_domain("advanced_domain_2", domain_config)
        
        return brain
    
    def test_adaptive_routing(self, advanced_brain):
        """Test adaptive routing between domains."""
        test_data = {"features": [1.0, 2.0, 3.0, 4.0]}
        
        # Make predictions to generate routing data
        for _ in range(10):
            advanced_brain.predict("advanced_domain_1", test_data)
            advanced_brain.predict("advanced_domain_2", test_data)
        
        # Test auto-routing
        result = advanced_brain.predict_auto_route(test_data)
        
        assert result.success is True
        assert result.domain_name in ["advanced_domain_1", "advanced_domain_2"]
    
    def test_cross_domain_knowledge_transfer(self, advanced_brain):
        """Test knowledge transfer between domains."""
        # Train first domain
        training_data_1 = {
            "X": np.random.rand(30, 10).tolist(),
            "y": np.random.randint(0, 2, 30).tolist()
        }
        
        training_config = TrainingConfig(epochs=5, batch_size=8)
        advanced_brain.train_domain("advanced_domain_1", training_data_1, training_config)
        
        # Transfer knowledge to second domain
        result = advanced_brain.transfer_knowledge(
            source_domain="advanced_domain_1",
            target_domain="advanced_domain_2",
            transfer_layers=["shared_foundation"]
        )
        
        assert result["success"] is True
        assert "transferred_parameters" in result
    
    def test_uncertainty_quantification(self, advanced_brain):
        """Test uncertainty quantification in predictions."""
        test_data = {"features": [1.0, 2.0, 3.0, 4.0]}
        
        result = advanced_brain.predict("advanced_domain_1", test_data)
        
        assert "uncertainty" in result.result
        uncertainty = result.result["uncertainty"]
        
        assert "epistemic" in uncertainty
        assert "aleatoric" in uncertainty
        assert "total" in uncertainty
        assert "confidence_interval" in uncertainty
    
    def test_batch_prediction_processing(self, advanced_brain):
        """Test batch prediction processing."""
        batch_data = [
            {"features": [1.0, 2.0, 3.0, 4.0]},
            {"features": [2.0, 3.0, 4.0, 5.0]},
            {"features": [3.0, 4.0, 5.0, 6.0]}
        ]
        
        results = advanced_brain.predict_batch("advanced_domain_1", batch_data)
        
        assert len(results) == 3
        for result in results:
            assert result.success is True
            assert result.domain_name == "advanced_domain_1"
    
    def test_model_versioning(self, advanced_brain):
        """Test model versioning and rollback."""
        # Create initial version
        training_data = {
            "X": np.random.rand(30, 10).tolist(),
            "y": np.random.randint(0, 2, 30).tolist()
        }
        
        training_config = TrainingConfig(epochs=3, batch_size=8)
        advanced_brain.train_domain("advanced_domain_1", training_data, training_config)
        
        # Create checkpoint
        checkpoint_result = advanced_brain.create_domain_checkpoint("advanced_domain_1", "v1.0")
        assert checkpoint_result["success"] is True
        
        # Train more (create new version)
        advanced_brain.train_domain("advanced_domain_1", training_data, training_config)
        
        # Rollback to checkpoint
        rollback_result = advanced_brain.rollback_domain("advanced_domain_1", "v1.0")
        assert rollback_result["success"] is True


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge case scenarios."""
    
    @pytest.fixture
    def error_test_brain(self, temp_dir):
        """Create Brain for error testing."""
        config = BrainSystemConfig(base_path=temp_dir / ".brain_test")
        return Brain(config=config)
    
    def test_prediction_with_invalid_data(self, error_test_brain):
        """Test prediction with invalid input data."""
        domain_config = DomainConfig(hidden_layers=[32, 16])
        error_test_brain.register_domain("test_domain", domain_config)
        
        # Test with None data
        result = error_test_brain.predict("test_domain", None)
        assert result.success is False
        
        # Test with empty data
        result = error_test_brain.predict("test_domain", {})
        assert result.success is False
        
        # Test with malformed data
        result = error_test_brain.predict("test_domain", {"invalid": "data"})
        assert result.success is False
    
    def test_training_with_insufficient_data(self, error_test_brain):
        """Test training with insufficient data."""
        domain_config = DomainConfig(hidden_layers=[32, 16])
        error_test_brain.register_domain("test_domain", domain_config)
        
        # Empty training data
        empty_data = {"X": [], "y": []}
        training_config = TrainingConfig(epochs=5, batch_size=8)
        
        result = error_test_brain.train_domain("test_domain", empty_data, training_config)
        assert result["success"] is False
        assert "insufficient" in result["message"].lower()
    
    def test_memory_limit_handling(self, error_test_brain):
        """Test handling of memory limits."""
        # Create domain with very low memory limit
        domain_config = DomainConfig(
            hidden_layers=[1024, 1024, 1024],  # Large layers
            max_memory_mb=1  # Very low limit
        )
        
        result = error_test_brain.register_domain("memory_test", domain_config)
        
        # Should either reject or warn about memory constraints
        if result["success"]:
            # If registered, training should handle memory limits
            large_data = {
                "X": np.random.rand(10000, 1000).tolist(),
                "y": np.random.randint(0, 2, 10000).tolist()
            }
            
            training_config = TrainingConfig(epochs=5, batch_size=1000)
            train_result = error_test_brain.train_domain("memory_test", large_data, training_config)
            
            # Should fail gracefully
            assert "memory" in train_result.get("message", "").lower()
    
    def test_concurrent_access_safety(self, error_test_brain):
        """Test thread safety under concurrent access."""
        domain_config = DomainConfig(hidden_layers=[32, 16])
        error_test_brain.register_domain("concurrent_test", domain_config)
        
        test_data = {"features": [1.0, 2.0, 3.0]}
        results = []
        errors = []
        
        def make_prediction():
            try:
                result = error_test_brain.predict("concurrent_test", test_data)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent operations
        threads = []
        for _ in range(20):
            thread = threading.Thread(target=make_prediction)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should handle concurrent access without errors
        assert len(errors) == 0
        assert len(results) == 20
    
    def test_disk_space_handling(self, error_test_brain, temp_dir):
        """Test handling of disk space limitations."""
        # This test would be platform-specific and complex to implement
        # For now, just test that save operations handle errors gracefully
        
        domain_config = DomainConfig(hidden_layers=[32, 16])
        error_test_brain.register_domain("disk_test", domain_config)
        
        # Try to save to invalid path
        invalid_path = Path("/invalid/nonexistent/path/brain_state.json")
        result = error_test_brain.save_brain_state(invalid_path)
        
        assert result["success"] is False
        assert "error" in result


class TestAntiForgettingMechanisms:
    """Test specific anti-catastrophic forgetting mechanisms."""
    
    @pytest.fixture
    def forgetting_test_brain(self, temp_dir):
        """Create Brain for forgetting tests."""
        config = BrainSystemConfig(
            base_path=temp_dir / ".brain_test",
            enable_persistence=True,
            enable_adaptation=True
        )
        brain = Brain(config=config)
        
        domain_config = DomainConfig(
            hidden_layers=[64, 32],
            learning_rate=0.01
        )
        brain.register_domain("memory_domain", domain_config)
        
        return brain
    
    def test_knowledge_preservation_during_training(self, forgetting_test_brain):
        """Test that old knowledge is preserved during new training."""
        # Create distinct datasets for different "tasks"
        task1_data = {
            "X": [[1, 0, 0, 0]] * 20 + [[0, 1, 0, 0]] * 20,
            "y": [0] * 20 + [1] * 20
        }
        
        task2_data = {
            "X": [[0, 0, 1, 0]] * 20 + [[0, 0, 0, 1]] * 20,
            "y": [2] * 20 + [3] * 20
        }
        
        training_config = TrainingConfig(epochs=10, batch_size=8)
        
        # Train on task 1
        forgetting_test_brain.train_domain("memory_domain", task1_data, training_config)
        
        # Test task 1 performance
        test1 = {"features": [1, 0, 0, 0]}
        pred1_before = forgetting_test_brain.predict("memory_domain", test1)
        
        # Train on task 2
        forgetting_test_brain.train_domain("memory_domain", task2_data, training_config)
        
        # Test that task 1 is still remembered
        pred1_after = forgetting_test_brain.predict("memory_domain", test1)
        
        assert pred1_after.success is True
        # Performance shouldn't degrade significantly
        confidence_drop = abs(pred1_before.result["confidence"] - pred1_after.result["confidence"])
        assert confidence_drop < 0.3  # Allow some degradation but not catastrophic
    
    def test_elastic_weight_consolidation(self, forgetting_test_brain):
        """Test EWC-like mechanisms for preventing forgetting."""
        # This would test the implementation of EWC or similar techniques
        # For now, test that the mechanism is available
        
        training_data = {
            "X": np.random.rand(50, 10).tolist(),
            "y": np.random.randint(0, 2, 50).tolist()
        }
        
        training_config = TrainingConfig(
            epochs=5,
            batch_size=8,
            enable_ewc=True,  # Assuming this option exists
            ewc_lambda=0.1
        )
        
        result = forgetting_test_brain.train_domain(
            "memory_domain", 
            training_data, 
            training_config
        )
        
        # Should complete successfully with EWC enabled
        assert result["success"] is True
    
    def test_knowledge_distillation(self, forgetting_test_brain):
        """Test knowledge distillation mechanisms."""
        # Register teacher and student domains
        teacher_config = DomainConfig(hidden_layers=[128, 64, 32])
        student_config = DomainConfig(hidden_layers=[64, 32])
        
        forgetting_test_brain.register_domain("teacher", teacher_config)
        forgetting_test_brain.register_domain("student", student_config)
        
        # Train teacher
        training_data = {
            "X": np.random.rand(50, 10).tolist(),
            "y": np.random.randint(0, 2, 50).tolist()
        }
        
        training_config = TrainingConfig(epochs=10, batch_size=8)
        forgetting_test_brain.train_domain("teacher", training_data, training_config)
        
        # Distill knowledge to student
        distillation_result = forgetting_test_brain.distill_knowledge(
            teacher_domain="teacher",
            student_domain="student",
            distillation_data=training_data,
            temperature=3.0
        )
        
        assert distillation_result["success"] is True
    
    def test_rehearsal_mechanisms(self, forgetting_test_brain):
        """Test rehearsal-based forgetting prevention."""
        # Train on initial data
        initial_data = {
            "X": np.random.rand(30, 10).tolist(),
            "y": np.random.randint(0, 2, 30).tolist()
        }
        
        training_config = TrainingConfig(epochs=5, batch_size=8)
        forgetting_test_brain.train_domain("memory_domain", initial_data, training_config)
        
        # Enable rehearsal for new training
        new_data = {
            "X": np.random.rand(30, 10).tolist(),
            "y": np.random.randint(0, 2, 30).tolist()
        }
        
        rehearsal_config = TrainingConfig(
            epochs=5,
            batch_size=8,
            enable_rehearsal=True,
            rehearsal_ratio=0.3
        )
        
        result = forgetting_test_brain.train_domain(
            "memory_domain", 
            new_data, 
            rehearsal_config
        )
        
        assert result["success"] is True


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""
    
    @pytest.fixture
    def performance_brain(self, temp_dir):
        """Create Brain for performance testing."""
        config = BrainSystemConfig(
            base_path=temp_dir / ".brain_test",
            enable_parallel_predictions=True,
            max_prediction_threads=8,
            prediction_cache_size=1000
        )
        brain = Brain(config=config)
        
        # Register multiple domains
        domain_config = DomainConfig(hidden_layers=[32, 16])
        for i in range(5):
            brain.register_domain(f"perf_domain_{i}", domain_config)
        
        return brain
    
    def test_prediction_throughput(self, performance_brain):
        """Test prediction throughput under load."""
        test_data = {"features": [1.0, 2.0, 3.0, 4.0]}
        
        start_time = time.time()
        num_predictions = 100
        
        for _ in range(num_predictions):
            result = performance_brain.predict("perf_domain_0", test_data)
            assert result.success is True
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = num_predictions / duration
        
        # Should achieve reasonable throughput (adjust threshold as needed)
        assert throughput > 10  # At least 10 predictions per second
    
    def test_memory_usage_scaling(self, performance_brain):
        """Test memory usage with increasing load."""
        initial_memory = performance_brain.get_resource_usage()["memory"]["used_mb"]
        
        # Add training data to multiple domains
        training_data = {
            "X": np.random.rand(100, 20).tolist(),
            "y": np.random.randint(0, 2, 100).tolist()
        }
        
        training_config = TrainingConfig(epochs=3, batch_size=16)
        
        for i in range(3):
            performance_brain.train_domain(f"perf_domain_{i}", training_data, training_config)
        
        final_memory = performance_brain.get_resource_usage()["memory"]["used_mb"]
        memory_increase = final_memory - initial_memory
        
        # Memory should increase but not excessively
        assert memory_increase > 0
        assert memory_increase < 1000  # Less than 1GB increase
    
    def test_concurrent_domain_operations(self, performance_brain):
        """Test concurrent operations across domains."""
        test_data = {"features": [1.0, 2.0, 3.0, 4.0]}
        
        def domain_operation(domain_name):
            results = []
            for _ in range(10):
                result = performance_brain.predict(domain_name, test_data)
                results.append(result.success)
            return all(results)
        
        # Run operations on all domains concurrently
        threads = []
        thread_results = []
        
        for i in range(5):
            thread = threading.Thread(
                target=lambda dn=f"perf_domain_{i}": thread_results.append(domain_operation(dn))
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All operations should succeed
        assert all(thread_results)
    
    def test_cache_effectiveness(self, performance_brain):
        """Test prediction caching effectiveness."""
        test_data = {"features": [1.0, 2.0, 3.0, 4.0]}
        
        # First prediction (cache miss)
        start_time = time.time()
        result1 = performance_brain.predict("perf_domain_0", test_data)
        first_duration = time.time() - start_time
        
        # Second prediction (cache hit)
        start_time = time.time()
        result2 = performance_brain.predict("perf_domain_0", test_data)
        second_duration = time.time() - start_time
        
        assert result1.success is True
        assert result2.success is True
        
        # Second prediction should be faster (cached)
        assert second_duration < first_duration
    
    def test_large_batch_processing(self, performance_brain):
        """Test processing of large prediction batches."""
        large_batch = [
            {"features": np.random.rand(10).tolist()}
            for _ in range(500)
        ]
        
        start_time = time.time()
        results = performance_brain.predict_batch("perf_domain_0", large_batch)
        duration = time.time() - start_time
        
        assert len(results) == 500
        assert all(result.success for result in results)
        
        # Should process large batch in reasonable time
        assert duration < 30  # Less than 30 seconds for 500 predictions


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    @pytest.fixture
    def integration_brain(self, temp_dir):
        """Create Brain for integration testing."""
        config = BrainSystemConfig(
            base_path=temp_dir / ".brain_test",
            enable_persistence=True,
            enable_monitoring=True,
            enable_adaptation=True,
            enable_parallel_predictions=True
        )
        return Brain(config=config)
    
    def test_complete_workflow(self, integration_brain):
        """Test complete AI workflow from registration to deployment."""
        # 1. Domain registration
        domain_config = DomainConfig(
            domain_type=DomainType.STANDARD,
            description="Integration test domain",
            hidden_layers=[64, 32, 16],
            learning_rate=0.01
        )
        
        reg_result = integration_brain.register_domain("workflow_domain", domain_config)
        assert reg_result["success"] is True
        
        # 2. Training
        training_data = {
            "X": np.random.rand(100, 15).tolist(),
            "y": np.random.randint(0, 3, 100).tolist()
        }
        
        training_config = TrainingConfig(epochs=10, batch_size=16)
        train_result = integration_brain.train_domain("workflow_domain", training_data, training_config)
        assert train_result["success"] is True
        
        # 3. Validation
        test_data = {"features": np.random.rand(15).tolist()}
        pred_result = integration_brain.predict("workflow_domain", test_data)
        assert pred_result.success is True
        
        # 4. Monitoring
        health = integration_brain.get_health_status()
        assert health["overall_health"]["grade"] in ["A", "B", "C", "D", "F"]
        
        # 5. Persistence
        save_path = integration_brain.config.base_path / "workflow_state.json"
        save_result = integration_brain.save_brain_state(save_path)
        assert save_result["success"] is True
        
        # 6. Restoration
        new_brain = Brain(config=integration_brain.config)
        load_result = new_brain.load_brain_state(save_path)
        assert load_result["success"] is True
        
        # 7. Verification
        final_pred = new_brain.predict("workflow_domain", test_data)
        assert final_pred.success is True
    
    def test_multi_domain_interaction(self, integration_brain):
        """Test interactions between multiple domains."""
        # Register multiple related domains
        base_config = DomainConfig(hidden_layers=[32, 16])
        
        domains = ["text_processing", "image_analysis", "decision_making"]
        for domain_name in domains:
            integration_brain.register_domain(domain_name, base_config)
        
        # Train each domain
        for domain_name in domains:
            training_data = {
                "X": np.random.rand(50, 10).tolist(),
                "y": np.random.randint(0, 2, 50).tolist()
            }
            training_config = TrainingConfig(epochs=5, batch_size=8)
            integration_brain.train_domain(domain_name, training_data, training_config)
        
        # Test cross-domain predictions
        test_data = {"features": np.random.rand(10).tolist()}
        
        results = integration_brain.predict_multi_domain(domains, test_data)
        assert len(results) == 3
        assert all(result.success for result in results)
        
        # Test knowledge transfer between domains
        transfer_result = integration_brain.transfer_knowledge(
            source_domain="text_processing",
            target_domain="decision_making",
            transfer_layers=["shared_foundation"]
        )
        assert transfer_result["success"] is True
    
    def test_production_readiness(self, integration_brain):
        """Test production readiness scenarios."""
        # Setup production-like environment
        domain_config = DomainConfig(
            hidden_layers=[128, 64, 32],
            max_memory_mb=512,
            enable_caching=True,
            enable_logging=True
        )
        
        integration_brain.register_domain("production_domain", domain_config)
        
        # Train with realistic data size
        training_data = {
            "X": np.random.rand(1000, 50).tolist(),
            "y": np.random.randint(0, 5, 1000).tolist()
        }
        
        training_config = TrainingConfig(
            epochs=20,
            batch_size=32,
            early_stopping_enabled=True,
            checkpoint_frequency=5
        )
        
        train_result = integration_brain.train_domain("production_domain", training_data, training_config)
        assert train_result["success"] is True
        
        # Stress test with concurrent predictions
        test_data = {"features": np.random.rand(50).tolist()}
        
        def stress_predict():
            results = []
            for _ in range(50):
                result = integration_brain.predict("production_domain", test_data)
                results.append(result.success)
            return all(results)
        
        # Run stress test
        threads = []
        stress_results = []
        
        for _ in range(10):
            thread = threading.Thread(target=lambda: stress_results.append(stress_predict()))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert all(stress_results)
        
        # Verify system health after stress test
        health = integration_brain.get_health_status()
        assert health["overall_health"]["score"] > 50  # Should maintain reasonable health


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])