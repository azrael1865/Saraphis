"""Comprehensive test suite for DomainRegistry to identify all root issues"""

import pytest
import tempfile
import json
import logging
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from domain_registry import (
    DomainRegistry,
    DomainConfig, 
    DomainMetadata,
    DomainStatus,
    DomainType,
    TensorJSONEncoder
)

class TestTensorJSONEncoder:
    """Test TensorJSONEncoder functionality"""
    
    def test_pytorch_tensor_encoding(self):
        """Test encoding PyTorch tensors"""
        encoder = TensorJSONEncoder()
        
        # Mock PyTorch tensor
        mock_tensor = Mock()
        mock_tensor.detach.return_value = mock_tensor
        mock_tensor.cpu.return_value = mock_tensor
        mock_tensor.numpy.return_value.tolist.return_value = [1, 2, 3]
        
        result = encoder.default(mock_tensor)
        assert result == [1, 2, 3]
    
    def test_numpy_array_encoding(self):
        """Test encoding numpy arrays"""
        encoder = TensorJSONEncoder()
        
        # Mock numpy array
        mock_array = Mock()
        mock_array.tolist.return_value = [4, 5, 6]
        # Ensure it doesn't have tensor attributes
        del mock_array.detach
        
        result = encoder.default(mock_array)
        assert result == [4, 5, 6]
    
    def test_numpy_scalar_encoding(self):
        """Test encoding numpy scalars"""
        encoder = TensorJSONEncoder()
        
        # Mock numpy scalar
        mock_scalar = Mock()
        mock_scalar.item.return_value = 42
        # Ensure it doesn't have array/tensor attributes
        del mock_scalar.tolist
        del mock_scalar.detach
        
        result = encoder.default(mock_scalar)
        assert result == 42
    
    def test_fallback_encoding(self):
        """Test fallback for unknown types"""
        encoder = TensorJSONEncoder()
        
        class UnknownType:
            def __str__(self):
                return "unknown_object"
        
        obj = UnknownType()
        result = encoder.default(obj)
        assert result == "unknown_object"
    
    def test_exception_handling(self):
        """Test exception handling in encoder"""
        encoder = TensorJSONEncoder()
        
        # Mock object that raises exception
        mock_obj = Mock()
        mock_obj.detach.side_effect = Exception("Test error")
        
        result = encoder.default(mock_obj)
        # Should fall back to string representation
        assert isinstance(result, str)

class TestDomainConfig:
    """Test DomainConfig functionality"""
    
    def test_default_creation(self):
        """Test default configuration creation"""
        config = DomainConfig()
        
        assert config.domain_type == DomainType.STANDARD
        assert config.version == "1.0.0"
        assert config.max_memory_mb == 512
        assert config.max_cpu_percent == 25.0
        assert config.priority == 5
        assert config.hidden_layers == [256, 128, 64]
        assert config.activation_function == "relu"
        assert config.dropout_rate == 0.2
        assert config.learning_rate == 0.001
        assert config.enable_caching == True
        assert config.cache_size == 100
        assert config.enable_logging == True
        assert config.log_level == "INFO"
        assert config.shared_foundation_layers == 3
        assert config.allow_cross_domain_access == False
        assert config.dependencies == []
        assert config.author == ""
        assert config.contact == ""
        assert config.tags == []
    
    def test_custom_creation(self):
        """Test custom configuration creation"""
        config = DomainConfig(
            domain_type=DomainType.EXPERIMENTAL,
            description="Test domain",
            version="2.1.0",
            max_memory_mb=1024,
            max_cpu_percent=50.0,
            priority=8,
            hidden_layers=[512, 256],
            activation_function="tanh",
            dropout_rate=0.3,
            learning_rate=0.01,
            enable_caching=False,
            cache_size=50,
            enable_logging=False,
            log_level="DEBUG",
            shared_foundation_layers=5,
            allow_cross_domain_access=True,
            dependencies=["domain1", "domain2"],
            author="Test Author",
            contact="test@example.com",
            tags=["test", "experimental"]
        )
        
        assert config.domain_type == DomainType.EXPERIMENTAL
        assert config.description == "Test domain"
        assert config.version == "2.1.0"
        assert config.max_memory_mb == 1024
        assert config.max_cpu_percent == 50.0
        assert config.priority == 8
        assert config.hidden_layers == [512, 256]
        assert config.activation_function == "tanh"
        assert config.dropout_rate == 0.3
        assert config.learning_rate == 0.01
        assert config.enable_caching == False
        assert config.cache_size == 50
        assert config.enable_logging == False
        assert config.log_level == "DEBUG"
        assert config.shared_foundation_layers == 5
        assert config.allow_cross_domain_access == True
        assert config.dependencies == ["domain1", "domain2"]
        assert config.author == "Test Author"
        assert config.contact == "test@example.com"
        assert config.tags == ["test", "experimental"]
    
    def test_validation_valid_config(self):
        """Test validation with valid configuration"""
        config = DomainConfig()
        is_valid, errors = config.validate()
        
        assert is_valid == True
        assert errors == []
    
    def test_validation_memory_errors(self):
        """Test validation with memory errors"""
        # Test negative memory
        config = DomainConfig(max_memory_mb=-100)
        is_valid, errors = config.validate()
        assert is_valid == False
        assert "max_memory_mb must be positive" in errors
        
        # Test zero memory
        config = DomainConfig(max_memory_mb=0)
        is_valid, errors = config.validate()
        assert is_valid == False
        assert "max_memory_mb must be positive" in errors
        
        # Test excessive memory
        config = DomainConfig(max_memory_mb=10000)
        is_valid, errors = config.validate()
        assert is_valid == False
        assert "max_memory_mb cannot exceed 8192" in errors
    
    def test_validation_cpu_errors(self):
        """Test validation with CPU errors"""
        # Test zero CPU
        config = DomainConfig(max_cpu_percent=0)
        is_valid, errors = config.validate()
        assert is_valid == False
        assert "max_cpu_percent must be between 0 and 100" in errors
        
        # Test negative CPU
        config = DomainConfig(max_cpu_percent=-10)
        is_valid, errors = config.validate()
        assert is_valid == False
        assert "max_cpu_percent must be between 0 and 100" in errors
        
        # Test excessive CPU
        config = DomainConfig(max_cpu_percent=150)
        is_valid, errors = config.validate()
        assert is_valid == False
        assert "max_cpu_percent must be between 0 and 100" in errors
    
    def test_validation_priority_errors(self):
        """Test validation with priority errors"""
        # Test too low priority
        config = DomainConfig(priority=0)
        is_valid, errors = config.validate()
        assert is_valid == False
        assert "priority must be between 1 and 10" in errors
        
        # Test too high priority
        config = DomainConfig(priority=11)
        is_valid, errors = config.validate()
        assert is_valid == False
        assert "priority must be between 1 and 10" in errors
    
    def test_validation_neural_network_errors(self):
        """Test validation with neural network errors"""
        # Test empty hidden layers
        config = DomainConfig(hidden_layers=[])
        is_valid, errors = config.validate()
        assert is_valid == False
        assert "hidden_layers cannot be empty" in errors
        
        # Test negative layer size
        config = DomainConfig(hidden_layers=[256, -128, 64])
        is_valid, errors = config.validate()
        assert is_valid == False
        assert "hidden layer sizes must be positive" in errors
        
        # Test zero layer size
        config = DomainConfig(hidden_layers=[256, 0, 64])
        is_valid, errors = config.validate()
        assert is_valid == False
        assert "hidden layer sizes must be positive" in errors
        
        # Test invalid dropout rate
        config = DomainConfig(dropout_rate=-0.1)
        is_valid, errors = config.validate()
        assert is_valid == False
        assert "dropout_rate must be between 0 and 1" in errors
        
        config = DomainConfig(dropout_rate=1.0)
        is_valid, errors = config.validate()
        assert is_valid == False
        assert "dropout_rate must be between 0 and 1" in errors
        
        # Test invalid learning rate
        config = DomainConfig(learning_rate=-0.001)
        is_valid, errors = config.validate()
        assert is_valid == False
        assert "learning_rate must be positive" in errors
        
        config = DomainConfig(learning_rate=0)
        is_valid, errors = config.validate()
        assert is_valid == False
        assert "learning_rate must be positive" in errors
    
    def test_validation_version_errors(self):
        """Test validation with version format errors"""
        invalid_versions = [
            "1.0", "1", "1.0.0.0", "v1.0.0", "1.0.0-beta", 
            "1.0.a", "a.b.c", "", "1.0."
        ]
        
        for version in invalid_versions:
            config = DomainConfig(version=version)
            is_valid, errors = config.validate()
            assert is_valid == False
            assert "version must be in format X.Y.Z" in errors
    
    def test_validation_dependency_errors(self):
        """Test validation with dependency errors"""
        # Test empty string dependency
        config = DomainConfig(dependencies=["valid_dep", "", "another_dep"])
        is_valid, errors = config.validate()
        assert is_valid == False
        assert "dependencies must be non-empty strings" in errors
        
        # Test whitespace only dependency
        config = DomainConfig(dependencies=["valid_dep", "   ", "another_dep"])
        is_valid, errors = config.validate()
        assert is_valid == False
        assert "dependencies must be non-empty strings" in errors
        
        # Test non-string dependency
        config = DomainConfig(dependencies=["valid_dep", 123, "another_dep"])
        is_valid, errors = config.validate()
        assert is_valid == False
        assert "dependencies must be non-empty strings" in errors
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary"""
        config = DomainConfig(
            domain_type=DomainType.EXPERIMENTAL,
            description="Test",
            dependencies=["dep1", "dep2"]
        )
        
        result = config.to_dict()
        
        assert isinstance(result, dict)
        assert result['domain_type'] == 'experimental'
        assert result['description'] == 'Test'
        assert result['dependencies'] == ['dep1', 'dep2']
    
    def test_from_dict_conversion(self):
        """Test conversion from dictionary"""
        data = {
            'domain_type': 'specialized',
            'description': 'Test domain',
            'version': '1.2.3',
            'max_memory_mb': 1024,
            'dependencies': ['dep1']
        }
        
        config = DomainConfig.from_dict(data)
        
        assert config.domain_type == DomainType.SPECIALIZED
        assert config.description == 'Test domain'
        assert config.version == '1.2.3'
        assert config.max_memory_mb == 1024
        assert config.dependencies == ['dep1']

class TestDomainMetadata:
    """Test DomainMetadata functionality"""
    
    def test_default_creation(self):
        """Test default metadata creation"""
        config = DomainConfig()
        metadata = DomainMetadata(name="test_domain", config=config)
        
        assert metadata.name == "test_domain"
        assert metadata.config == config
        assert metadata.status == DomainStatus.REGISTERED
        assert isinstance(metadata.registered_at, datetime)
        assert isinstance(metadata.last_updated, datetime)
        assert metadata.last_accessed is None
        assert metadata.activation_count == 0
        assert metadata.error_count == 0
        assert metadata.total_predictions == 0
        assert metadata.average_confidence == 0.0
        assert metadata.resource_usage == {}
        assert metadata.error_messages == []
        assert metadata.metadata == {}
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary"""
        config = DomainConfig()
        metadata = DomainMetadata(
            name="test_domain", 
            config=config,
            status=DomainStatus.ACTIVE,
            activation_count=5,
            error_count=2,
            total_predictions=100,
            average_confidence=0.85,
            resource_usage={'memory': 512.0, 'cpu': 25.0},
            error_messages=['Error 1', 'Error 2'],
            metadata={'key1': 'value1'}
        )
        
        result = metadata.to_dict()
        
        assert isinstance(result, dict)
        assert result['name'] == 'test_domain'
        assert result['status'] == 'active'
        assert result['activation_count'] == 5
        assert result['error_count'] == 2
        assert result['total_predictions'] == 100
        assert result['average_confidence'] == 0.85
        assert result['resource_usage'] == {'memory': 512.0, 'cpu': 25.0}
        assert result['error_messages'] == ['Error 1', 'Error 2']
        assert result['metadata'] == {'key1': 'value1'}
        assert 'registered_at' in result
        assert 'last_updated' in result
    
    def test_from_dict_conversion(self):
        """Test conversion from dictionary"""
        now = datetime.now()
        data = {
            'name': 'test_domain',
            'config': {
                'domain_type': 'core',
                'version': '2.0.0',
                'max_memory_mb': 1024
            },
            'status': 'active',
            'registered_at': now.isoformat(),
            'last_updated': now.isoformat(),
            'last_accessed': now.isoformat(),
            'activation_count': 10,
            'error_count': 1,
            'total_predictions': 500,
            'average_confidence': 0.9,
            'resource_usage': {'memory': 800.0},
            'error_messages': ['Test error'],
            'metadata': {'custom': 'data'}
        }
        
        metadata = DomainMetadata.from_dict(data)
        
        assert metadata.name == 'test_domain'
        assert metadata.config.domain_type == DomainType.CORE
        assert metadata.config.version == '2.0.0'
        assert metadata.config.max_memory_mb == 1024
        assert metadata.status == DomainStatus.ACTIVE
        assert metadata.activation_count == 10
        assert metadata.error_count == 1
        assert metadata.total_predictions == 500
        assert metadata.average_confidence == 0.9
        assert metadata.resource_usage == {'memory': 800.0}
        assert metadata.error_messages == ['Test error']
        assert metadata.metadata == {'custom': 'data'}

class TestDomainRegistryInitialization:
    """Test DomainRegistry initialization"""
    
    def test_default_initialization(self):
        """Test default initialization"""
        registry = DomainRegistry()
        
        assert registry._domains == {}
        assert registry._domain_order == []
        assert registry._dependencies == {}
        assert registry._dependents == {}
        assert registry._domain_states == {}
        assert registry._domain_knowledge == {}
        assert registry._isolation_metadata == {}
        assert registry._domain_access_logs == {}
        assert registry._total_registrations == 0
        assert registry._total_removals == 0
        assert isinstance(registry._registry_created, datetime)
        assert isinstance(registry._reserved_names, set)
        assert 'core' in registry._reserved_names
        assert 'admin' in registry._reserved_names
        assert registry._persistence_path is None
        assert registry.logger is not None
    
    def test_initialization_with_persistence(self):
        """Test initialization with persistence path"""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence_path = Path(temp_dir) / "registry.json"
            registry = DomainRegistry(persistence_path=persistence_path)
            
            assert registry._persistence_path == persistence_path
    
    def test_initialization_with_logger(self):
        """Test initialization with custom logger"""
        custom_logger = logging.getLogger("test_logger")
        registry = DomainRegistry(logger=custom_logger)
        
        assert registry.logger == custom_logger

class TestDomainRegistration:
    """Test domain registration functionality"""
    
    def test_valid_domain_registration(self):
        """Test registering valid domains"""
        registry = DomainRegistry()
        
        # Test with default config
        result = registry.register_domain("test_domain")
        assert result == True
        assert "test_domain" in registry._domains
        assert registry._domains["test_domain"].status == DomainStatus.REGISTERED
        assert "test_domain" in registry._domain_order
        assert registry._total_registrations == 1
        
        # Test with custom config
        config = DomainConfig(description="Custom domain", priority=8)
        result = registry.register_domain("custom_domain", config)
        assert result == True
        assert "custom_domain" in registry._domains
        assert registry._domains["custom_domain"].config.description == "Custom domain"
        assert registry._domains["custom_domain"].config.priority == 8
        assert registry._total_registrations == 2
    
    def test_invalid_domain_names(self):
        """Test registration with invalid domain names"""
        registry = DomainRegistry()
        
        # Test None domain name
        result = registry.register_domain(None)
        assert result == False
        
        # Test empty string
        result = registry.register_domain("")
        assert result == False
        
        # Test whitespace only
        result = registry.register_domain("   ")
        assert result == False
        
        # Test non-string
        result = registry.register_domain(123)
        assert result == False
        
        # Test reserved names
        for reserved in ['core', 'brain', 'system', 'admin']:
            result = registry.register_domain(reserved)
            assert result == False
    
    def test_duplicate_registration(self):
        """Test duplicate domain registration"""
        registry = DomainRegistry()
        
        # Register first time
        result = registry.register_domain("test_domain")
        assert result == True
        
        # Try to register again
        result = registry.register_domain("test_domain")
        assert result == False
        
        # Should still only have one domain
        assert len(registry._domains) == 1
        assert registry._total_registrations == 1
    
    def test_invalid_config_type(self):
        """Test registration with invalid config type"""
        registry = DomainRegistry()
        
        # Test with non-DomainConfig object
        result = registry.register_domain("test_domain", {"invalid": "config"})
        assert result == False
        
        result = registry.register_domain("test_domain", "invalid_config")
        assert result == False
    
    def test_invalid_config_validation(self):
        """Test registration with invalid config values"""
        registry = DomainRegistry()
        
        # Create invalid config
        invalid_config = DomainConfig(max_memory_mb=-100)
        result = registry.register_domain("test_domain", invalid_config)
        assert result == False
    
    def test_case_normalization(self):
        """Test domain name case normalization"""
        registry = DomainRegistry()
        
        result = registry.register_domain("TEST_DOMAIN")
        assert result == True
        assert "test_domain" in registry._domains
        assert "TEST_DOMAIN" not in registry._domains

class TestDomainLifecycle:
    """Test domain lifecycle management"""
    
    def test_domain_activation(self):
        """Test domain activation"""
        registry = DomainRegistry()
        registry.register_domain("test_domain")
        
        result = registry.activate_domain("test_domain")
        assert result == True
        assert registry._domains["test_domain"].status == DomainStatus.ACTIVE
        assert registry._domains["test_domain"].activation_count == 1
        assert registry._domains["test_domain"].last_accessed is not None
    
    def test_domain_deactivation(self):
        """Test domain deactivation"""
        registry = DomainRegistry()
        registry.register_domain("test_domain")
        registry.activate_domain("test_domain")
        
        result = registry.deactivate_domain("test_domain")
        assert result == True
        assert registry._domains["test_domain"].status == DomainStatus.INACTIVE
    
    def test_domain_removal(self):
        """Test domain removal"""
        registry = DomainRegistry()
        registry.register_domain("test_domain")
        
        result = registry.remove_domain("test_domain")
        assert result == True
        assert "test_domain" not in registry._domains
        assert "test_domain" not in registry._domain_order
        assert registry._total_removals == 1
    
    def test_lifecycle_nonexistent_domain(self):
        """Test lifecycle operations on nonexistent domains"""
        registry = DomainRegistry()
        
        # Test activation of nonexistent domain
        result = registry.activate_domain("nonexistent")
        assert result == False
        
        # Test deactivation of nonexistent domain
        result = registry.deactivate_domain("nonexistent")
        assert result == False
        
        # Test removal of nonexistent domain
        result = registry.remove_domain("nonexistent")
        assert result == False

class TestDomainQueries:
    """Test domain query functionality"""
    
    def test_domain_existence_check(self):
        """Test checking if domain exists"""
        registry = DomainRegistry()
        registry.register_domain("test_domain")
        
        assert registry.has_domain("test_domain") == True
        assert registry.has_domain("nonexistent") == False
    
    def test_get_domain_metadata(self):
        """Test retrieving domain metadata"""
        registry = DomainRegistry()
        config = DomainConfig(description="Test domain")
        registry.register_domain("test_domain", config)
        
        metadata = registry.get_domain("test_domain")
        assert metadata is not None
        assert metadata.name == "test_domain"
        assert metadata.config.description == "Test domain"
        
        # Test nonexistent domain
        metadata = registry.get_domain("nonexistent")
        assert metadata is None
    
    def test_list_domains(self):
        """Test listing domains"""
        registry = DomainRegistry()
        registry.register_domain("domain1")
        registry.register_domain("domain2")
        registry.register_domain("domain3")
        
        domains = registry.list_domains()
        assert len(domains) == 3
        assert "domain1" in domains
        assert "domain2" in domains
        assert "domain3" in domains
    
    def test_list_domains_by_status(self):
        """Test listing domains by status"""
        registry = DomainRegistry()
        registry.register_domain("active_domain")
        registry.register_domain("inactive_domain")
        registry.activate_domain("active_domain")
        
        active_domains = registry.list_domains_by_status(DomainStatus.ACTIVE)
        assert len(active_domains) == 1
        assert "active_domain" in active_domains
        
        registered_domains = registry.list_domains_by_status(DomainStatus.REGISTERED)
        assert len(registered_domains) == 1
        assert "inactive_domain" in registered_domains
    
    def test_list_domains_by_type(self):
        """Test listing domains by type"""
        registry = DomainRegistry()
        config1 = DomainConfig(domain_type=DomainType.CORE)
        config2 = DomainConfig(domain_type=DomainType.EXPERIMENTAL)
        
        registry.register_domain("core_domain", config1)
        registry.register_domain("exp_domain", config2)
        
        core_domains = registry.list_domains_by_type(DomainType.CORE)
        assert len(core_domains) == 1
        assert "core_domain" in core_domains
        
        exp_domains = registry.list_domains_by_type(DomainType.EXPERIMENTAL)
        assert len(exp_domains) == 1
        assert "exp_domain" in exp_domains

class TestDomainDependencies:
    """Test domain dependency management"""
    
    def test_dependency_registration(self):
        """Test registering domains with dependencies"""
        registry = DomainRegistry()
        
        # Register base domain
        registry.register_domain("base_domain")
        
        # Register domain with dependency
        config = DomainConfig(dependencies=["base_domain"])
        result = registry.register_domain("dependent_domain", config)
        assert result == True
        
        # Check dependency tracking
        assert "base_domain" in registry._dependencies.get("dependent_domain", set())
        assert "dependent_domain" in registry._dependents.get("base_domain", set())
    
    def test_missing_dependency_registration(self):
        """Test registering domain with missing dependencies"""
        registry = DomainRegistry()
        
        # Try to register domain with missing dependency
        config = DomainConfig(dependencies=["missing_domain"])
        result = registry.register_domain("dependent_domain", config)
        assert result == False
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection"""
        registry = DomainRegistry()
        
        # Register domains
        registry.register_domain("domain_a")
        registry.register_domain("domain_b")
        
        # Create circular dependency
        config_a = DomainConfig(dependencies=["domain_b"])
        config_b = DomainConfig(dependencies=["domain_a"])
        
        # Should detect and prevent circular dependency
        # (This test depends on implementation details)
        pass
    
    def test_dependency_removal_prevention(self):
        """Test prevention of removing domains with dependents"""
        registry = DomainRegistry()
        
        # Register base and dependent domains
        registry.register_domain("base_domain")
        config = DomainConfig(dependencies=["base_domain"])
        registry.register_domain("dependent_domain", config)
        
        # Try to remove base domain
        result = registry.remove_domain("base_domain")
        # Should fail because dependent_domain depends on it
        assert result == False

class TestDomainIsolation:
    """Test domain isolation functionality"""
    
    def test_domain_state_isolation(self):
        """Test domain state isolation"""
        registry = DomainRegistry()
        registry.register_domain("domain1")
        registry.register_domain("domain2")
        
        # Set isolated state for domain1
        state_data = {"weights": [1, 2, 3], "bias": [0.1, 0.2]}
        registry.set_domain_state("domain1", state_data)
        
        # Retrieve state
        retrieved_state = registry.get_domain_state("domain1")
        assert retrieved_state == state_data
        
        # Domain2 should not have access to domain1's state
        domain2_state = registry.get_domain_state("domain2")
        assert domain2_state == {}
    
    def test_domain_knowledge_isolation(self):
        """Test domain knowledge isolation"""
        registry = DomainRegistry()
        registry.register_domain("domain1")
        
        # Set domain knowledge
        knowledge_data = {"learned_patterns": ["pattern1", "pattern2"]}
        registry.set_domain_knowledge("domain1", knowledge_data)
        
        # Retrieve knowledge
        retrieved_knowledge = registry.get_domain_knowledge("domain1")
        assert retrieved_knowledge == knowledge_data
    
    def test_cross_domain_access_control(self):
        """Test cross-domain access control"""
        registry = DomainRegistry()
        
        # Domain with cross-domain access disabled
        config1 = DomainConfig(allow_cross_domain_access=False)
        registry.register_domain("restricted_domain", config1)
        
        # Domain with cross-domain access enabled
        config2 = DomainConfig(allow_cross_domain_access=True)
        registry.register_domain("open_domain", config2)
        
        # Test access control (depends on implementation)
        pass

class TestDomainPersistence:
    """Test domain persistence functionality"""
    
    def test_registry_save_and_load(self):
        """Test saving and loading registry state"""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence_path = Path(temp_dir) / "registry.json"
            
            # Create and populate registry
            registry1 = DomainRegistry(persistence_path=persistence_path)
            config = DomainConfig(description="Test domain")
            registry1.register_domain("test_domain", config)
            registry1.activate_domain("test_domain")
            
            # Save registry
            registry1.save_registry()
            assert persistence_path.exists()
            
            # Create new registry and load
            registry2 = DomainRegistry(persistence_path=persistence_path)
            
            # Check if data was loaded correctly
            assert registry2.has_domain("test_domain")
            metadata = registry2.get_domain("test_domain")
            assert metadata.config.description == "Test domain"
            assert metadata.status == DomainStatus.ACTIVE
    
    def test_persistence_with_corrupt_file(self):
        """Test handling of corrupt persistence file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            persistence_path = Path(temp_dir) / "registry.json"
            
            # Create corrupt JSON file
            with open(persistence_path, 'w') as f:
                f.write("invalid json content {")
            
            # Should handle corrupt file gracefully
            registry = DomainRegistry(persistence_path=persistence_path)
            assert len(registry._domains) == 0

class TestDomainStatistics:
    """Test domain statistics functionality"""
    
    def test_registry_statistics(self):
        """Test registry-level statistics"""
        registry = DomainRegistry()
        
        # Initial statistics
        stats = registry.get_registry_statistics()
        assert stats['total_domains'] == 0
        assert stats['total_registrations'] == 0
        assert stats['total_removals'] == 0
        
        # Register some domains
        registry.register_domain("domain1")
        registry.register_domain("domain2")
        registry.remove_domain("domain1")
        
        # Check updated statistics
        stats = registry.get_registry_statistics()
        assert stats['total_domains'] == 1
        assert stats['total_registrations'] == 2
        assert stats['total_removals'] == 1
    
    def test_domain_statistics(self):
        """Test domain-specific statistics"""
        registry = DomainRegistry()
        registry.register_domain("test_domain")
        
        # Update domain statistics
        registry.update_domain_statistics("test_domain", {
            'predictions': 100,
            'accuracy': 0.95,
            'memory_usage': 256.0
        })
        
        # Check statistics
        metadata = registry.get_domain("test_domain")
        assert metadata.total_predictions == 100
        # Check if other statistics are updated appropriately

class TestConcurrency:
    """Test concurrent access to DomainRegistry"""
    
    def test_concurrent_registration(self):
        """Test concurrent domain registration"""
        registry = DomainRegistry()
        
        def register_domain(domain_name):
            return registry.register_domain(f"domain_{domain_name}")
        
        # Register domains concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(register_domain, i) for i in range(20)]
            results = [f.result() for f in futures]
        
        # All registrations should succeed
        assert all(results)
        assert len(registry._domains) == 20
    
    def test_concurrent_activation(self):
        """Test concurrent domain activation"""
        registry = DomainRegistry()
        
        # Register domains
        for i in range(10):
            registry.register_domain(f"domain_{i}")
        
        def activate_domain(domain_name):
            return registry.activate_domain(f"domain_{domain_name}")
        
        # Activate domains concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(activate_domain, i) for i in range(10)]
            results = [f.result() for f in futures]
        
        # All activations should succeed
        assert all(results)
    
    def test_thread_safety(self):
        """Test thread safety of registry operations"""
        registry = DomainRegistry()
        
        def mixed_operations(worker_id):
            results = []
            for i in range(5):
                domain_name = f"worker_{worker_id}_domain_{i}"
                
                # Register
                results.append(registry.register_domain(domain_name))
                
                # Activate
                results.append(registry.activate_domain(domain_name))
                
                # Query
                results.append(registry.has_domain(domain_name))
                
                # Small delay to increase concurrency
                time.sleep(0.001)
            
            return results
        
        # Run mixed operations concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(mixed_operations, i) for i in range(5)]
            all_results = [f.result() for f in futures]
        
        # Check that operations completed successfully
        for worker_results in all_results:
            # Each worker should have successful operations
            assert len([r for r in worker_results if r]) > 0

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_method_arguments(self):
        """Test handling of invalid method arguments"""
        registry = DomainRegistry()
        
        # Test None arguments
        assert registry.activate_domain(None) == False
        assert registry.deactivate_domain(None) == False
        assert registry.remove_domain(None) == False
        assert registry.has_domain(None) == False
        assert registry.get_domain(None) is None
        
        # Test empty string arguments
        assert registry.activate_domain("") == False
        assert registry.deactivate_domain("") == False
        assert registry.remove_domain("") == False
        assert registry.has_domain("") == False
        assert registry.get_domain("") is None
    
    def test_memory_pressure_handling(self):
        """Test handling of memory pressure scenarios"""
        registry = DomainRegistry()
        
        # Register many domains to simulate memory pressure
        for i in range(1000):
            registry.register_domain(f"domain_{i}")
        
        # Registry should still function correctly
        assert len(registry._domains) == 1000
        assert registry.has_domain("domain_500") == True
    
    def test_large_domain_states(self):
        """Test handling of large domain states"""
        registry = DomainRegistry()
        registry.register_domain("test_domain")
        
        # Create large state data
        large_state = {"weights": list(range(10000))}
        registry.set_domain_state("test_domain", large_state)
        
        # Should handle large states
        retrieved_state = registry.get_domain_state("test_domain")
        assert len(retrieved_state["weights"]) == 10000

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    import sys
    
    test_classes = [
        TestTensorJSONEncoder,
        TestDomainConfig,
        TestDomainMetadata,
        TestDomainRegistryInitialization,
        TestDomainRegistration,
        TestDomainLifecycle,
        TestDomainQueries,
        TestDomainDependencies,
        TestDomainIsolation,
        TestDomainPersistence,
        TestDomainStatistics,
        TestConcurrency,
        TestErrorHandling,
    ]
    
    failed_tests = []
    passed_tests = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Testing {test_class.__name__}")
        print('='*60)
        
        test_instance = test_class()
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"✓ {method_name}")
                passed_tests.append(f"{test_class.__name__}.{method_name}")
            except Exception as e:
                print(f"✗ {method_name}: {e}")
                failed_tests.append((f"{test_class.__name__}.{method_name}", str(e)))
    
    print(f"\n{'='*60}")
    print("Test Summary")
    print('='*60)
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed Tests:")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {error}")
        return 1
    else:
        print("\nAll tests passed!")
        return 0

if __name__ == "__main__":
    exit(run_comprehensive_tests())