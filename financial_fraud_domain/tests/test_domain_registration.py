"""
Unit tests for Financial Fraud Domain Registration
Tests domain registration, validation, configuration, and lifecycle management
"""

import asyncio
import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Import the domain registration module
from domain_registration import (
    DomainConfiguration,
    DomainConfigurationError,
    DomainMetadata,
    DomainRegistrationError,
    DomainStatus,
    DomainValidationError,
    FinancialFraudDomain,
    FraudTaskType,
    register_fraud_domain
)


class TestDomainConfiguration(unittest.TestCase):
    """Test DomainConfiguration class"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        config = DomainConfiguration()
        
        self.assertTrue(config.enabled)
        self.assertTrue(config.auto_start)
        self.assertEqual(config.max_concurrent_tasks, 100)
        self.assertEqual(config.task_timeout, 300)
        self.assertEqual(config.model_threshold, 0.85)
        self.assertTrue(config.real_time_processing)
        self.assertIn("PCI_DSS", config.compliance_frameworks)
        self.assertIn("SOX", config.compliance_frameworks)
        self.assertIn("GDPR", config.compliance_frameworks)
    
    def test_configuration_customization(self):
        """Test configuration customization"""
        config = DomainConfiguration(
            enabled=False,
            max_concurrent_tasks=50,
            model_threshold=0.95
        )
        
        self.assertFalse(config.enabled)
        self.assertEqual(config.max_concurrent_tasks, 50)
        self.assertEqual(config.model_threshold, 0.95)


class TestDomainMetadata(unittest.TestCase):
    """Test DomainMetadata class"""
    
    def test_metadata_creation(self):
        """Test metadata creation"""
        metadata = DomainMetadata(
            name="Test Domain",
            version="1.0.0",
            description="Test description"
        )
        
        self.assertEqual(metadata.name, "Test Domain")
        self.assertEqual(metadata.version, "1.0.0")
        self.assertEqual(metadata.description, "Test description")
        self.assertIsInstance(metadata.created_at, datetime)
        self.assertIsInstance(metadata.updated_at, datetime)


class TestFinancialFraudDomain(unittest.TestCase):
    """Test FinancialFraudDomain class"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_domain_initialization(self):
        """Test domain initialization"""
        domain = FinancialFraudDomain(config_path=self.config_path)
        
        self.assertEqual(domain.status, DomainStatus.UNREGISTERED)
        self.assertEqual(domain.domain_id, "financial-fraud-detection")
        self.assertIsInstance(domain.metadata, DomainMetadata)
        self.assertIsInstance(domain.configuration, DomainConfiguration)
        self.assertEqual(domain.metadata.name, "Financial Fraud Detection Domain")
        self.assertEqual(domain.metadata.version, "1.0.0")
    
    def test_configuration_loading_default(self):
        """Test loading default configuration when no file exists"""
        domain = FinancialFraudDomain(config_path=self.config_path)
        
        self.assertEqual(domain.configuration.model_threshold, 0.85)
        self.assertTrue(domain.configuration.real_time_processing)
        
        # Check that default config was saved
        self.assertTrue(os.path.exists(self.config_path))
    
    def test_configuration_loading_from_file(self):
        """Test loading configuration from file"""
        # Create a config file
        config_data = {
            "model_threshold": 0.95,
            "real_time_processing": False,
            "batch_size": 500
        }
        
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f)
        
        domain = FinancialFraudDomain(config_path=self.config_path)
        
        self.assertEqual(domain.configuration.model_threshold, 0.95)
        self.assertFalse(domain.configuration.real_time_processing)
        self.assertEqual(domain.configuration.batch_size, 500)
    
    def test_configuration_update(self):
        """Test configuration update"""
        domain = FinancialFraudDomain(config_path=self.config_path)
        
        updates = {
            "model_threshold": 0.92,
            "alert_threshold": 0.88,
            "cache_ttl": 7200
        }
        
        domain.update_configuration(updates)
        
        self.assertEqual(domain.configuration.model_threshold, 0.92)
        self.assertEqual(domain.configuration.alert_threshold, 0.88)
        self.assertEqual(domain.configuration.cache_ttl, 7200)
        
        # Check that config was saved
        with open(self.config_path, 'r') as f:
            saved_config = json.load(f)
        
        self.assertEqual(saved_config["model_threshold"], 0.92)
    
    def test_get_metrics(self):
        """Test getting domain metrics"""
        domain = FinancialFraudDomain(config_path=self.config_path)
        
        metrics = domain.get_metrics()
        
        self.assertIn("tasks_processed", metrics)
        self.assertIn("fraud_detected", metrics)
        self.assertIn("processing_time_avg", metrics)
        self.assertIn("current_status", metrics)
        self.assertEqual(metrics["current_status"], "unregistered")
    
    def test_get_capabilities(self):
        """Test getting domain capabilities"""
        domain = FinancialFraudDomain(config_path=self.config_path)
        
        capabilities = domain.get_capabilities()
        
        self.assertIn("real-time-transaction-analysis", capabilities)
        self.assertIn("pattern-detection", capabilities)
        self.assertIn("risk-scoring", capabilities)
        self.assertIn("compliance-checking", capabilities)
    
    def test_get_info(self):
        """Test getting complete domain information"""
        domain = FinancialFraudDomain(config_path=self.config_path)
        
        info = domain.get_info()
        
        self.assertEqual(info["id"], "financial-fraud-detection")
        self.assertIn("metadata", info)
        self.assertIn("status", info)
        self.assertIn("capabilities", info)
        self.assertIn("configuration", info)
        self.assertIn("metrics", info)


@pytest.mark.asyncio
class TestAsyncDomainOperations:
    """Test async domain operations"""
    
    async def test_domain_registration_success(self):
        """Test successful domain registration"""
        domain = FinancialFraudDomain()
        mock_registry = AsyncMock()
        mock_registry.register_domain = AsyncMock()
        
        # Mock validation to return True
        with patch.object(domain, 'validate', return_value=True):
            result = await domain.register(mock_registry)
        
        assert result is True
        assert domain.status == DomainStatus.ACTIVE  # Auto-start is enabled
        mock_registry.register_domain.assert_called_once()
    
    async def test_domain_registration_validation_failure(self):
        """Test domain registration with validation failure"""
        domain = FinancialFraudDomain()
        mock_registry = AsyncMock()
        
        # Mock validation to return False
        with patch.object(domain, 'validate', return_value=False):
            with pytest.raises(DomainRegistrationError):
                await domain.register(mock_registry)
        
        assert domain.status == DomainStatus.ERROR
    
    async def test_domain_validation_all_pass(self):
        """Test domain validation when all checks pass"""
        domain = FinancialFraudDomain()
        
        # Mock all validation methods to return True
        with patch.object(domain, '_validate_configuration', return_value=True), \
             patch.object(domain, '_validate_dependencies', return_value=True), \
             patch.object(domain, '_validate_resources', return_value=True), \
             patch.object(domain, '_validate_integration', return_value=True):
            
            result = await domain.validate()
        
        assert result is True
    
    async def test_domain_validation_configuration_failure(self):
        """Test domain validation with configuration failure"""
        domain = FinancialFraudDomain()
        domain.configuration.model_threshold = 1.5  # Invalid value
        
        result = await domain.validate()
        
        assert result is False
    
    async def test_domain_start(self):
        """Test domain start"""
        domain = FinancialFraudDomain()
        domain.status = DomainStatus.REGISTERED
        
        # Mock resource initialization and health monitoring
        with patch.object(domain, '_initialize_resources', new_callable=AsyncMock), \
             patch.object(domain, '_monitor_health', new_callable=AsyncMock), \
             patch('asyncio.create_task'):
            
            await domain.start()
        
        assert domain.status == DomainStatus.ACTIVE
        assert domain.metrics["uptime"] is not None
    
    async def test_domain_start_already_active(self):
        """Test starting an already active domain"""
        domain = FinancialFraudDomain()
        domain.status = DomainStatus.ACTIVE
        
        # Should not raise error, just log warning
        await domain.start()
        
        assert domain.status == DomainStatus.ACTIVE
    
    async def test_domain_shutdown(self):
        """Test domain shutdown"""
        domain = FinancialFraudDomain()
        domain.status = DomainStatus.ACTIVE
        
        # Mock cleanup
        with patch.object(domain, '_cleanup_resources', new_callable=AsyncMock):
            await domain.shutdown()
        
        assert domain.status == DomainStatus.SHUTDOWN
    
    async def test_health_check_all_healthy(self):
        """Test health check when all checks pass"""
        domain = FinancialFraudDomain()
        domain.status = DomainStatus.ACTIVE
        
        # Mock health check methods
        with patch.object(domain, '_check_executors_health', return_value=True), \
             patch.object(domain, '_check_resource_availability', return_value=True), \
             patch.object(domain, '_check_integration_health', return_value=True):
            
            health = await domain.health_check()
        
        assert health["overall_status"] == "healthy"
        assert health["domain_id"] == "financial-fraud-detection"
        assert "timestamp" in health
        assert len(health["checks"]) > 0
    
    async def test_health_check_partial_failure(self):
        """Test health check with partial failure"""
        domain = FinancialFraudDomain()
        domain.status = DomainStatus.ACTIVE
        
        # Mock one health check to fail
        with patch.object(domain, '_check_executors_health', return_value=True), \
             patch.object(domain, '_check_resource_availability', return_value=False), \
             patch.object(domain, '_check_integration_health', return_value=True):
            
            health = await domain.health_check()
        
        assert health["overall_status"] == "unhealthy"
    
    async def test_register_fraud_domain_function(self):
        """Test the register_fraud_domain function"""
        mock_registry = AsyncMock()
        mock_registry.register_domain = AsyncMock()
        
        with patch('domain_registration.FinancialFraudDomain.validate', return_value=True):
            domain = await register_fraud_domain(mock_registry)
        
        assert isinstance(domain, FinancialFraudDomain)
        assert domain.status == DomainStatus.ACTIVE


class TestFraudTaskType(unittest.TestCase):
    """Test FraudTaskType enum"""
    
    def test_task_types_defined(self):
        """Test that all expected task types are defined"""
        expected_types = [
            "transaction_analysis",
            "pattern_detection",
            "risk_scoring",
            "compliance_check",
            "alert_generation",
            "anomaly_detection",
            "fraud_investigation",
            "model_training",
            "report_generation"
        ]
        
        actual_types = [task_type.value for task_type in FraudTaskType]
        
        for expected in expected_types:
            self.assertIn(expected, actual_types)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and exceptions"""
    
    def test_domain_registration_error(self):
        """Test DomainRegistrationError"""
        with self.assertRaises(DomainRegistrationError):
            raise DomainRegistrationError("Test error")
    
    def test_domain_validation_error(self):
        """Test DomainValidationError"""
        with self.assertRaises(DomainValidationError):
            raise DomainValidationError("Test error")
    
    def test_domain_configuration_error(self):
        """Test DomainConfigurationError"""
        with self.assertRaises(DomainConfigurationError):
            raise DomainConfigurationError("Test error")


# Integration test
@pytest.mark.asyncio
async def test_full_domain_lifecycle():
    """Test complete domain lifecycle from registration to shutdown"""
    # Create mock registry
    mock_registry = AsyncMock()
    mock_registry.register_domain = AsyncMock()
    
    # Create and register domain
    with patch('domain_registration.FinancialFraudDomain.validate', return_value=True):
        domain = await register_fraud_domain(mock_registry)
    
    assert domain.status == DomainStatus.ACTIVE
    
    # Perform health check
    with patch.object(domain, '_check_executors_health', return_value=True), \
         patch.object(domain, '_check_resource_availability', return_value=True), \
         patch.object(domain, '_check_integration_health', return_value=True):
        
        health = await domain.health_check()
    
    assert health["overall_status"] == "healthy"
    
    # Update configuration
    domain.update_configuration({"model_threshold": 0.9})
    assert domain.configuration.model_threshold == 0.9
    
    # Shutdown domain
    with patch.object(domain, '_cleanup_resources', new_callable=AsyncMock):
        await domain.shutdown()
    
    assert domain.status == DomainStatus.SHUTDOWN


if __name__ == "__main__":
    # Run synchronous tests
    unittest.main()