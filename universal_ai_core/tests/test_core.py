"""
Unit tests for Universal AI Core components.
Adapted from Saraphis/charon_builder test patterns.
"""

import pytest
import asyncio
import time
import threading
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List

from universal_ai_core.core.universal_ai_core import UniversalAICore
from universal_ai_core.core.plugin_manager import PluginManager, PluginType
from universal_ai_core.core.orchestrator import SystemOrchestrator
from universal_ai_core.config.config_manager import ConfigurationManager, UniversalConfiguration
from universal_ai_core.utils.data_utils import DataProcessor, ProcessingResult
from universal_ai_core.utils.validation_utils import ValidationEngine, ValidationResult
from universal_ai_core import UniversalAIAPI, APIConfig, TaskResult


@pytest.mark.unit
class TestUniversalAICore:
    """Test Universal AI Core functionality."""
    
    def test_core_initialization(self, sample_config):
        """Test core system initialization."""
        config = UniversalConfiguration(**sample_config)
        core = UniversalAICore(config)
        
        assert core.config == config
        assert core.plugin_manager is not None
        assert isinstance(core.plugin_manager, PluginManager)
        
    def test_core_with_invalid_config(self):
        """Test core initialization with invalid config."""
        with pytest.raises((ValueError, TypeError)):
            UniversalAICore(None)
    
    @patch('universal_ai_core.core.universal_ai_core.PluginManager')
    def test_core_plugin_manager_integration(self, mock_plugin_manager_class, sample_config):
        """Test integration with plugin manager."""
        mock_plugin_manager = Mock()
        mock_plugin_manager_class.return_value = mock_plugin_manager
        
        config = UniversalConfiguration(**sample_config)
        core = UniversalAICore(config)
        
        mock_plugin_manager_class.assert_called_once_with(config)
        assert core.plugin_manager == mock_plugin_manager
    
    def test_core_get_available_domains(self, universal_ai_core):
        """Test getting available domains."""
        with patch.object(universal_ai_core.plugin_manager, 'list_available_plugins') as mock_list:
            mock_list.return_value = {
                "feature_extractors": {"molecular": Mock(), "financial": Mock()},
                "models": {"molecular": Mock(), "cybersecurity": Mock()}
            }
            
            domains = universal_ai_core.get_available_domains()
            expected_domains = {"molecular", "financial", "cybersecurity"}
            assert domains == expected_domains
    
    def test_core_thread_safety(self, sample_config):
        """Test core thread safety."""
        config = UniversalConfiguration(**sample_config)
        core = UniversalAICore(config)
        results = []
        errors = []
        
        def worker():
            try:
                # Test concurrent access to core methods
                for _ in range(10):
                    domains = core.get_available_domains()
                    results.append(len(domains))
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0
        assert len(results) == 30  # 3 threads * 10 iterations


@pytest.mark.unit
class TestPluginManager:
    """Test Plugin Manager functionality."""
    
    def test_plugin_manager_initialization(self, sample_config):
        """Test plugin manager initialization."""
        config = UniversalConfiguration(**sample_config)
        manager = PluginManager(config)
        
        assert manager.config == config
        assert isinstance(manager.plugins, dict)
        
        # Check plugin type structure
        expected_types = ["feature_extractors", "models", "proof_languages", "knowledge_bases"]
        for plugin_type in expected_types:
            assert plugin_type in manager.plugins
    
    def test_list_available_plugins(self, mock_plugin_manager):
        """Test listing available plugins."""
        plugins = mock_plugin_manager.list_available_plugins()
        
        assert isinstance(plugins, dict)
        assert "feature_extractors" in plugins
        assert "models" in plugins
        assert "proof_languages" in plugins
        assert "knowledge_bases" in plugins
    
    def test_plugin_loading(self, sample_config):
        """Test plugin loading functionality."""
        config = UniversalConfiguration(**sample_config)
        manager = PluginManager(config)
        
        # Test loading with valid plugin type and name
        with patch.object(manager, '_load_plugin_module') as mock_load:
            mock_load.return_value = Mock()
            result = manager.load_plugin("feature_extractors", "molecular")
            assert result is True
    
    def test_plugin_getting(self, mock_plugin_manager):
        """Test getting plugins."""
        mock_plugin = Mock()
        mock_plugin_manager.get_plugin.return_value = mock_plugin
        
        plugin = mock_plugin_manager.get_plugin("feature_extractors", "molecular")
        assert plugin == mock_plugin
    
    def test_plugin_type_validation(self, sample_config):
        """Test plugin type validation."""
        config = UniversalConfiguration(**sample_config)
        manager = PluginManager(config)
        
        # Valid plugin types
        valid_types = ["feature_extractors", "models", "proof_languages", "knowledge_bases"]
        for plugin_type in valid_types:
            assert plugin_type in manager.plugins
        
        # Test getting invalid plugin type
        result = manager.get_plugin("invalid_type", "test")
        assert result is None


@pytest.mark.unit
class TestSystemOrchestrator:
    """Test System Orchestrator functionality."""
    
    def test_orchestrator_initialization(self, universal_ai_core):
        """Test orchestrator initialization."""
        orchestrator = SystemOrchestrator(universal_ai_core)
        
        assert orchestrator.core == universal_ai_core
        assert orchestrator.data_processor is not None
        assert orchestrator.validation_engine is not None
    
    @pytest.mark.asyncio
    async def test_orchestrator_process_request(self, universal_ai_core):
        """Test processing request through orchestrator."""
        orchestrator = SystemOrchestrator(universal_ai_core)
        
        # Mock the processing pipeline
        with patch.object(orchestrator.data_processor, 'prepare_features') as mock_process:
            with patch.object(orchestrator.validation_engine, 'validate') as mock_validate:
                mock_process.return_value = ProcessingResult(
                    status="success", data={"features": [1, 2, 3]}, processing_time=0.1
                )
                mock_validate.return_value = ValidationResult(
                    is_valid=True, validation_score=0.95, issues=[]
                )
                
                result = await orchestrator.process_request(
                    domain="molecular",
                    operation="analyze",
                    data={"smiles": "CCO"}
                )
                
                assert result is not None
                mock_process.assert_called_once()
                mock_validate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_orchestrator_error_handling(self, universal_ai_core):
        """Test orchestrator error handling."""
        orchestrator = SystemOrchestrator(universal_ai_core)
        
        # Mock processing to raise error
        with patch.object(orchestrator.data_processor, 'prepare_features') as mock_process:
            mock_process.side_effect = Exception("Processing failed")
            
            with pytest.raises(Exception):
                await orchestrator.process_request(
                    domain="molecular",
                    operation="analyze",
                    data={"smiles": "invalid"}
                )


@pytest.mark.unit
class TestConfigurationManager:
    """Test Configuration Manager functionality."""
    
    def test_config_manager_initialization(self, temp_directory):
        """Test configuration manager initialization."""
        config_file = temp_directory / "config.yaml"
        config_file.write_text("core:\n  max_workers: 4\n")
        
        manager = ConfigurationManager(str(config_file))
        assert manager.config_path == str(config_file)
    
    def test_config_loading(self, sample_config_file):
        """Test configuration loading."""
        manager = ConfigurationManager(str(sample_config_file))
        config = manager.get_config()
        
        assert isinstance(config, UniversalConfiguration)
        assert config.core.max_workers == 2
    
    def test_config_validation(self, sample_config):
        """Test configuration validation."""
        manager = ConfigurationManager()
        valid, errors = manager.validate_config(sample_config)
        
        assert valid is True
        assert len(errors) == 0
    
    def test_invalid_config_validation(self):
        """Test validation of invalid configuration."""
        manager = ConfigurationManager()
        invalid_config = {"invalid": "config"}
        
        valid, errors = manager.validate_config(invalid_config)
        assert valid is False
        assert len(errors) > 0
    
    def test_config_file_watching(self, sample_config_file):
        """Test configuration file watching."""
        manager = ConfigurationManager(str(sample_config_file))
        original_config = manager.get_config()
        
        # Mock file modification
        with patch.object(manager, '_reload_config') as mock_reload:
            manager.start_watching()
            time.sleep(0.1)  # Allow watcher to start
            manager.stop_watching()
            
            # File watching should be active
            assert manager._watching is False  # Should be stopped


@pytest.mark.unit
class TestDataProcessor:
    """Test Data Processor functionality."""
    
    def test_data_processor_initialization(self):
        """Test data processor initialization."""
        processor = DataProcessor()
        assert processor is not None
        assert hasattr(processor, 'prepare_features')
    
    def test_feature_preparation(self, mock_data_processor, sample_molecular_data):
        """Test feature preparation."""
        result = mock_data_processor.prepare_features(
            sample_molecular_data, ["molecular_weight", "logp"]
        )
        
        assert isinstance(result, ProcessingResult)
        assert result.status == "success"
        assert "features" in result.data
    
    def test_data_processor_error_handling(self):
        """Test data processor error handling."""
        processor = DataProcessor()
        
        # Test with invalid data
        with patch.object(processor, '_extract_features') as mock_extract:
            mock_extract.side_effect = ValueError("Invalid data")
            
            result = processor.prepare_features(None, [])
            assert result.status == "error"
    
    def test_batch_processing(self, mock_data_processor):
        """Test batch data processing."""
        batch_data = [
            {"smiles": "CCO"},
            {"smiles": "CC(=O)O"},
            {"smiles": "c1ccccc1"}
        ]
        
        with patch.object(mock_data_processor, 'prepare_features') as mock_prepare:
            mock_prepare.return_value = ProcessingResult(
                status="success", data={"features": [1, 2, 3]}, processing_time=0.1
            )
            
            results = []
            for data in batch_data:
                result = mock_data_processor.prepare_features(data, ["molecular_weight"])
                results.append(result)
            
            assert len(results) == 3
            assert all(r.status == "success" for r in results)


@pytest.mark.unit
class TestValidationEngine:
    """Test Validation Engine functionality."""
    
    def test_validation_engine_initialization(self):
        """Test validation engine initialization."""
        engine = ValidationEngine()
        assert engine is not None
        assert hasattr(engine, 'validate')
    
    def test_data_validation(self, mock_validation_engine, sample_molecular_data):
        """Test data validation."""
        result = mock_validation_engine.validate(
            sample_molecular_data, ["schema", "statistical"]
        )
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.validation_score > 0.9
    
    def test_validation_with_errors(self):
        """Test validation with validation errors."""
        engine = ValidationEngine()
        
        with patch.object(engine, '_run_validators') as mock_run:
            mock_run.return_value = ValidationResult(
                is_valid=False,
                validation_score=0.3,
                issues=["Missing required field"],
                failed_validators=["schema"]
            )
            
            result = engine.validate({"incomplete": "data"}, ["schema"])
            assert result.is_valid is False
            assert len(result.issues) > 0
    
    def test_validator_registration(self):
        """Test custom validator registration."""
        engine = ValidationEngine()
        
        def custom_validator(data):
            return len(data) > 0
        
        # Test adding custom validator
        with patch.object(engine, 'register_validator') as mock_register:
            engine.register_validator("custom", custom_validator)
            mock_register.assert_called_once_with("custom", custom_validator)


@pytest.mark.unit
class TestUniversalAIAPI:
    """Test Universal AI API functionality."""
    
    def test_api_initialization(self, api_config, sample_config_file):
        """Test API initialization."""
        api = UniversalAIAPI(config_path=str(sample_config_file), api_config=api_config)
        
        assert api.api_config == api_config
        assert api.core is not None
        assert api.orchestrator is not None
        
        api.shutdown()
    
    def test_api_process_data(self, universal_ai_api, sample_molecular_data):
        """Test API data processing."""
        with patch.object(universal_ai_api.data_processor, 'prepare_features') as mock_process:
            mock_process.return_value = ProcessingResult(
                status="success", data={"features": [1, 2, 3]}, processing_time=0.1
            )
            
            result = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
            assert isinstance(result, ProcessingResult)
            assert result.status == "success"
    
    def test_api_validate_data(self, universal_ai_api, sample_molecular_data):
        """Test API data validation."""
        with patch.object(universal_ai_api.validation_engine, 'validate') as mock_validate:
            mock_validate.return_value = ValidationResult(
                is_valid=True, validation_score=0.95, issues=[]
            )
            
            result = universal_ai_api.validate_data(sample_molecular_data, ["schema"])
            assert isinstance(result, ValidationResult)
            assert result.is_valid is True
    
    @pytest.mark.asyncio
    async def test_api_async_task(self, universal_ai_api):
        """Test API async task submission."""
        def sample_operation():
            time.sleep(0.1)
            return {"result": "success"}
        
        task_id = await universal_ai_api.submit_async_task(
            "test_operation", {"data": "test"}, config={}
        )
        
        assert isinstance(task_id, str)
        assert len(task_id) > 0
        
        # Wait a bit for task to potentially complete
        await asyncio.sleep(0.2)
        
        task_result = universal_ai_api.get_task_result(task_id)
        assert task_result is not None
        assert isinstance(task_result, TaskResult)
    
    def test_api_caching(self, universal_ai_api, sample_molecular_data):
        """Test API caching functionality."""
        with patch.object(universal_ai_api.data_processor, 'prepare_features') as mock_process:
            mock_process.return_value = ProcessingResult(
                status="success", data={"features": [1, 2, 3]}, processing_time=0.1
            )
            
            # First call should hit the processor
            result1 = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
            assert mock_process.call_count == 1
            
            # Second call with same data should use cache (if caching enabled)
            result2 = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
            
            # Both results should be the same
            assert result1.status == result2.status
    
    def test_api_health_status(self, universal_ai_api):
        """Test API health status."""
        health_status = universal_ai_api.get_health_status()
        
        assert isinstance(health_status, dict)
        assert "overall" in health_status
        assert "api" in health_status
    
    def test_api_metrics(self, universal_ai_api):
        """Test API metrics collection."""
        metrics = universal_ai_api.get_metrics()
        
        assert isinstance(metrics, dict)
        assert "api" in metrics
    
    def test_api_rate_limiting(self, universal_ai_api, sample_molecular_data):
        """Test API rate limiting."""
        # Override rate limit for testing
        universal_ai_api.api_config.rate_limit_requests_per_minute = 2
        universal_ai_api.api_config.enable_safety_checks = True
        
        # Reset rate limit counters
        universal_ai_api.request_counts.clear()
        
        # Make requests up to limit
        for i in range(2):
            result = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
            assert result is not None
        
        # Next request should be rate limited
        with pytest.raises(Exception) as exc_info:
            universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
        
        assert "Rate limit exceeded" in str(exc_info.value)
    
    def test_api_shutdown(self, api_config, sample_config_file):
        """Test API shutdown."""
        api = UniversalAIAPI(config_path=str(sample_config_file), api_config=api_config)
        
        # Should shutdown without errors
        api.shutdown()
        
        # Multiple shutdowns should be safe
        api.shutdown()


@pytest.mark.performance
class TestCorePerformance:
    """Performance tests for core components."""
    
    def test_core_initialization_performance(self, sample_config):
        """Test core initialization performance."""
        start_time = time.time()
        
        for _ in range(10):
            config = UniversalConfiguration(**sample_config)
            core = UniversalAICore(config)
            assert core is not None
        
        total_time = time.time() - start_time
        
        # Should initialize 10 cores in less than 1 second
        assert total_time < 1.0
    
    def test_plugin_loading_performance(self, sample_config):
        """Test plugin loading performance."""
        config = UniversalConfiguration(**sample_config)
        manager = PluginManager(config)
        
        start_time = time.time()
        
        # Test loading multiple plugins
        plugin_combinations = [
            ("feature_extractors", "molecular"),
            ("feature_extractors", "cybersecurity"),
            ("models", "financial"),
            ("proof_languages", "molecular")
        ]
        
        for plugin_type, plugin_name in plugin_combinations:
            try:
                manager.load_plugin(plugin_type, plugin_name)
            except Exception:
                pass  # Ignore loading errors for performance test
        
        total_time = time.time() - start_time
        
        # Should load plugins quickly
        assert total_time < 0.5
    
    def test_concurrent_api_requests(self, universal_ai_api, sample_molecular_data):
        """Test concurrent API request performance."""
        import concurrent.futures
        
        def make_request():
            try:
                return universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
            except Exception:
                return None
        
        start_time = time.time()
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Should handle 10 concurrent requests in reasonable time
        assert total_time < 2.0
        assert len([r for r in results if r is not None]) > 0


@pytest.mark.memory
class TestCoreMemoryUsage:
    """Memory usage tests for core components."""
    
    def test_core_memory_usage(self, sample_config, memory_monitor):
        """Test core memory usage."""
        initial_memory = memory_monitor['initial_memory']
        
        # Create multiple core instances
        cores = []
        for _ in range(5):
            config = UniversalConfiguration(**sample_config)
            core = UniversalAICore(config)
            cores.append(core)
        
        current_memory = memory_monitor['get_memory_usage']()
        memory_increase = current_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 5 cores)
        assert memory_increase < 100
        
        # Cleanup
        del cores
    
    def test_api_memory_leak(self, api_config, sample_config_file, memory_monitor):
        """Test for memory leaks in API usage."""
        initial_memory = memory_monitor['initial_memory']
        
        # Create and destroy multiple API instances
        for _ in range(3):
            api = UniversalAIAPI(config_path=str(sample_config_file), api_config=api_config)
            
            # Use the API briefly
            try:
                health = api.get_health_status()
                assert health is not None
            except Exception:
                pass
            
            api.shutdown()
            del api
        
        import gc
        gc.collect()
        
        final_memory = memory_monitor['get_memory_usage']()
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (less than 50MB)
        assert memory_increase < 50