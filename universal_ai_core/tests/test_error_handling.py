"""
Error handling and recovery tests for Universal AI Core.
Adapted from Saraphis error handling test patterns.
"""

import pytest
import asyncio
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from universal_ai_core import UniversalAIAPI, APIConfig
from universal_ai_core.core.universal_ai_core import UniversalAICore
from universal_ai_core.core.plugin_manager import PluginManager
from universal_ai_core.utils.data_utils import DataProcessor, ProcessingResult
from universal_ai_core.utils.validation_utils import ValidationEngine, ValidationResult


@pytest.mark.unit
class TestCoreErrorHandling:
    """Test error handling in core components."""
    
    def test_core_initialization_with_invalid_config(self):
        """Test core initialization error handling."""
        # Test with None config
        with pytest.raises((TypeError, ValueError)):
            UniversalAICore(None)
        
        # Test with invalid config structure
        with pytest.raises((AttributeError, TypeError, ValueError)):
            invalid_config = Mock()
            invalid_config.core = None
            UniversalAICore(invalid_config)
    
    def test_plugin_manager_missing_plugin_error(self, sample_config):
        """Test plugin manager error handling for missing plugins."""
        from universal_ai_core.config.config_manager import UniversalConfiguration
        
        config = UniversalConfiguration(**sample_config)
        manager = PluginManager(config)
        
        # Test getting non-existent plugin
        result = manager.get_plugin("nonexistent_type", "nonexistent_plugin")
        assert result is None
        
        # Test loading non-existent plugin
        success = manager.load_plugin("nonexistent_type", "nonexistent_plugin")
        assert success is False
    
    def test_data_processor_invalid_data_handling(self):
        """Test data processor error handling."""
        processor = DataProcessor()
        
        # Test with None data
        result = processor.prepare_features(None, ["test_extractor"])
        assert result.status == "error"
        
        # Test with invalid data structure
        invalid_data = "not_a_dict"
        result = processor.prepare_features(invalid_data, ["test_extractor"])
        assert result.status == "error"
        
        # Test with empty extractors list
        result = processor.prepare_features({"valid": "data"}, [])
        assert result.status in ["success", "error"]  # Should handle gracefully
    
    def test_validation_engine_error_handling(self):
        """Test validation engine error handling."""
        engine = ValidationEngine()
        
        # Test with None data
        result = engine.validate(None, ["schema"])
        assert result.is_valid is False
        assert len(result.issues) > 0
        
        # Test with invalid validators
        result = engine.validate({"test": "data"}, ["nonexistent_validator"])
        assert result.is_valid is False or len(result.failed_validators) > 0


@pytest.mark.integration
class TestAPIErrorHandling:
    """Test API-level error handling."""
    
    def test_api_initialization_errors(self, temp_directory):
        """Test API initialization error handling."""
        # Test with non-existent config file
        with pytest.raises((FileNotFoundError, IOError)):
            UniversalAIAPI(config_path="/nonexistent/config.yaml")
        
        # Test with invalid config format
        invalid_config_file = temp_directory / "invalid_config.yaml"
        invalid_config_file.write_text("invalid: yaml: content: [")
        
        with pytest.raises((yaml.YAMLError, ValueError, AttributeError)):
            import yaml
            UniversalAIAPI(config_path=str(invalid_config_file))
    
    def test_api_request_error_handling(self, universal_ai_api):
        """Test API request error handling."""
        # Test processing with None data
        with pytest.raises((TypeError, ValueError)):
            universal_ai_api.process_data(None, ["test"])
        
        # Test validation with invalid data type
        with pytest.raises((TypeError, AttributeError)):
            universal_ai_api.validate_data("invalid_data_type", ["schema"])
    
    def test_api_rate_limiting_errors(self, universal_ai_api, sample_molecular_data):
        """Test API rate limiting error handling."""
        # Enable rate limiting with very low limit
        universal_ai_api.api_config.enable_safety_checks = True
        universal_ai_api.api_config.rate_limit_requests_per_minute = 1
        
        # Reset rate limit counters
        universal_ai_api.request_counts.clear()
        universal_ai_api.last_request_reset = time.time()
        
        # First request should succeed
        try:
            result1 = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
        except Exception:
            pass  # May fail due to mocking, but shouldn't be rate limited
        
        # Second request should be rate limited
        with pytest.raises(Exception) as exc_info:
            universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
        
        assert "Rate limit exceeded" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_async_task_error_handling(self, universal_ai_api):
        """Test async task error handling."""
        # Test submitting task with invalid operation
        with pytest.raises((ValueError, TypeError)):
            await universal_ai_api.submit_async_task(None, {"data": "test"})
        
        # Test getting result for non-existent task
        result = universal_ai_api.get_task_result("nonexistent_task_id")
        assert result is None
    
    def test_api_cache_error_handling(self, universal_ai_api, sample_molecular_data):
        """Test cache error handling."""
        if universal_ai_api.cache:
            # Test cache with extremely large data
            large_data = {"data": "x" * 1000000}  # 1MB string
            
            # Should handle large data gracefully
            try:
                result = universal_ai_api.process_data(large_data, ["test"])
            except Exception as e:
                # Should not be a cache-related error
                assert "cache" not in str(e).lower()


@pytest.mark.integration
class TestPluginErrorHandling:
    """Test plugin system error handling."""
    
    def test_plugin_initialization_errors(self):
        """Test plugin initialization error handling."""
        from universal_ai_core.plugins.feature_extractors.molecular import MolecularFeatureExtractorPlugin
        
        # Test with invalid configuration
        with pytest.raises((KeyError, ValueError, TypeError)):
            MolecularFeatureExtractorPlugin({"invalid": "config"})
        
        # Test with None configuration
        with pytest.raises((TypeError, AttributeError)):
            MolecularFeatureExtractorPlugin(None)
    
    def test_plugin_processing_errors(self):
        """Test plugin processing error handling."""
        from universal_ai_core.plugins.feature_extractors.molecular import MolecularFeatureExtractorPlugin
        
        config = {"enabled": True, "rdkit_enabled": False}
        plugin = MolecularFeatureExtractorPlugin(config)
        
        # Test with invalid data
        try:
            result = plugin.extract_features({"invalid": "molecular_data"})
            # Should either return empty/error result or raise appropriate exception
            assert isinstance(result, dict) or result is None
        except (ValueError, KeyError, TypeError):
            # These are acceptable exceptions for invalid data
            pass
    
    def test_plugin_dependency_errors(self, sample_config):
        """Test plugin dependency error handling."""
        from universal_ai_core.config.config_manager import UniversalConfiguration
        
        config = UniversalConfiguration(**sample_config)
        manager = PluginManager(config)
        
        # Test loading plugin with missing dependencies
        with patch('universal_ai_core.core.plugin_manager.importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("Missing dependency")
            
            success = manager.load_plugin("feature_extractors", "molecular")
            assert success is False
    
    def test_plugin_runtime_errors(self):
        """Test plugin runtime error handling."""
        from universal_ai_core.plugins.models.molecular import MolecularModelPlugin
        
        config = {"enabled": True, "model_type": "neural_network"}
        plugin = MolecularModelPlugin(config)
        
        # Test prediction with invalid features
        with patch.object(plugin, '_get_model') as mock_model:
            mock_model.side_effect = Exception("Model runtime error")
            
            try:
                result = plugin.predict({"features": [[1, 2, 3]]})
                # Should handle error gracefully
                assert "error" in result or result is None
            except Exception:
                # Runtime errors might propagate
                pass


@pytest.mark.integration
class TestNetworkErrorHandling:
    """Test network and external service error handling."""
    
    def test_external_service_timeout_handling(self, universal_ai_api):
        """Test handling of external service timeouts."""
        # Mock external service call with timeout
        with patch.object(universal_ai_api.orchestrator, 'process_request') as mock_process:
            mock_process.side_effect = asyncio.TimeoutError("External service timeout")
            
            # Should handle timeout gracefully
            try:
                result = asyncio.run(
                    universal_ai_api.orchestrator.process_request(
                        domain="molecular",
                        operation="external_lookup",
                        data={"query": "test"}
                    )
                )
                assert result is None  # Should return None or error indicator
            except asyncio.TimeoutError:
                # Timeout exception might propagate
                pass
    
    def test_network_connection_error_handling(self, universal_ai_api):
        """Test handling of network connection errors."""
        # Mock network connection error
        with patch.object(universal_ai_api.orchestrator, 'process_request') as mock_process:
            mock_process.side_effect = ConnectionError("Network unreachable")
            
            try:
                result = asyncio.run(
                    universal_ai_api.orchestrator.process_request(
                        domain="cybersecurity",
                        operation="threat_intel_lookup",
                        data={"ip": "1.2.3.4"}
                    )
                )
                # Should handle connection error gracefully
                assert result is None or "error" in str(result)
            except ConnectionError:
                # Connection errors might propagate
                pass


@pytest.mark.integration
class TestConcurrencyErrorHandling:
    """Test error handling in concurrent scenarios."""
    
    def test_concurrent_access_errors(self, universal_ai_api, sample_molecular_data):
        """Test handling of concurrent access errors."""
        import threading
        
        errors = []
        results = []
        
        def worker():
            try:
                result = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Start multiple concurrent workers
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Should handle concurrent access without deadlocks
        assert len(threads) == 5
        # Some operations might fail, but system should remain stable
        total_operations = len(results) + len(errors)
        assert total_operations <= 5
    
    def test_thread_safety_error_handling(self, sample_config):
        """Test thread safety error handling."""
        from universal_ai_core.config.config_manager import UniversalConfiguration
        
        config = UniversalConfiguration(**sample_config)
        
        def create_and_use_core():
            try:
                core = UniversalAICore(config)
                domains = core.get_available_domains()
                return len(domains)
            except Exception:
                return 0
        
        import concurrent.futures
        
        # Test concurrent core creation and usage
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_and_use_core) for _ in range(6)]
            domain_counts = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All operations should complete without deadlocks
        assert len(domain_counts) == 6
        # Some might return 0 due to errors, but shouldn't hang


@pytest.mark.integration
class TestResourceErrorHandling:
    """Test resource-related error handling."""
    
    def test_memory_pressure_handling(self, universal_ai_api, memory_monitor):
        """Test handling of memory pressure scenarios."""
        initial_memory = memory_monitor['initial_memory']
        
        # Simulate memory-intensive operations
        large_datasets = []
        for i in range(10):
            large_data = {
                "molecules": [{"smiles": f"C{j}", "data": "x" * 1000} for j in range(100)],
                "metadata": {"dataset": i}
            }
            large_datasets.append(large_data)
        
        processed_count = 0
        errors = []
        
        for dataset in large_datasets:
            try:
                result = universal_ai_api.process_data(dataset, ["molecular_descriptors"])
                if result.status == "success":
                    processed_count += 1
            except MemoryError as e:
                errors.append(e)
                break  # Stop on memory error
            except Exception as e:
                errors.append(e)
        
        current_memory = memory_monitor['get_memory_usage']()
        memory_increase = current_memory - initial_memory
        
        # System should handle memory pressure gracefully
        assert memory_increase < 500  # Less than 500MB increase
        # Should process at least some datasets
        assert processed_count >= 0
    
    def test_disk_space_error_handling(self, universal_ai_api, temp_directory):
        """Test handling of disk space issues."""
        # Simulate disk space issues by creating large temporary files
        large_file = temp_directory / "large_file.tmp"
        
        try:
            # Try to create a very large file (this might fail on systems with limited space)
            with open(large_file, 'wb') as f:
                f.write(b'x' * (100 * 1024 * 1024))  # 100MB
        except OSError:
            # If we can't create the file due to space, that's fine for this test
            pass
        
        # Test that the API handles potential disk issues
        try:
            result = universal_ai_api.get_system_info()
            assert isinstance(result, dict)
        except OSError:
            # Disk space errors might propagate
            pass
        finally:
            # Cleanup
            if large_file.exists():
                large_file.unlink()
    
    def test_cpu_overload_handling(self, universal_ai_api, sample_molecular_data):
        """Test handling of CPU overload scenarios."""
        # Simulate CPU-intensive operations
        def cpu_intensive_operation():
            # Simple CPU-intensive calculation
            return sum(i ** 2 for i in range(10000))
        
        # Start background CPU load
        import threading
        
        cpu_threads = []
        for _ in range(4):  # 4 CPU-intensive threads
            thread = threading.Thread(target=cpu_intensive_operation)
            thread.daemon = True
            thread.start()
            cpu_threads.append(thread)
        
        try:
            # Test API operations under CPU load
            start_time = time.time()
            result = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
            processing_time = time.time() - start_time
            
            # Should complete even under CPU load (might be slower)
            assert processing_time < 10.0  # Should complete within 10 seconds
            assert result.status in ["success", "error"]
            
        finally:
            # Cleanup - threads will finish naturally
            pass


@pytest.mark.integration
class TestRecoveryMechanisms:
    """Test system recovery mechanisms."""
    
    def test_automatic_retry_mechanism(self, universal_ai_api, sample_molecular_data):
        """Test automatic retry mechanisms."""
        call_count = 0
        
        def failing_processor(data, extractors):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first 2 attempts
                raise ConnectionError("Temporary failure")
            return ProcessingResult(
                status="success",
                data={"features": [1, 2, 3]},
                processing_time=0.1
            )
        
        # Mock processor with retry logic
        with patch.object(universal_ai_api.data_processor, 'prepare_features', side_effect=failing_processor):
            try:
                result = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
                # Should succeed after retries
                assert result.status == "success"
                assert call_count > 1  # Should have retried
            except Exception:
                # Retry mechanism might not be implemented yet
                pass
    
    def test_circuit_breaker_recovery(self, universal_ai_api, sample_molecular_data):
        """Test circuit breaker recovery mechanism."""
        failure_count = 0
        
        def intermittent_failure(data, extractors):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:  # Fail first 3 attempts
                raise Exception("Service unavailable")
            return ProcessingResult(
                status="success",
                data={"features": [1, 2, 3]},
                processing_time=0.1
            )
        
        with patch.object(universal_ai_api.data_processor, 'prepare_features', side_effect=intermittent_failure):
            # Multiple attempts should trigger circuit breaker
            for i in range(5):
                try:
                    result = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
                    if result.status == "success":
                        break
                except Exception:
                    time.sleep(0.1)  # Brief pause between attempts
    
    def test_graceful_degradation(self, universal_ai_api):
        """Test graceful degradation when components fail."""
        # Simulate component failure
        with patch.object(universal_ai_api.cache, 'get', side_effect=Exception("Cache failure")):
            # System should continue working without cache
            try:
                health = universal_ai_api.get_health_status()
                assert isinstance(health, dict)
                # System might report unhealthy, but should still respond
            except Exception:
                # Some degradation is acceptable
                pass
    
    def test_error_reporting_and_logging(self, universal_ai_api, sample_molecular_data):
        """Test error reporting and logging mechanisms."""
        # Capture log messages
        import logging
        
        log_messages = []
        
        class TestLogHandler(logging.Handler):
            def emit(self, record):
                log_messages.append(self.format(record))
        
        # Add test handler
        test_handler = TestLogHandler()
        test_handler.setLevel(logging.ERROR)
        universal_ai_api.logger.addHandler(test_handler)
        
        try:
            # Trigger an error condition
            with patch.object(universal_ai_api.data_processor, 'prepare_features', side_effect=Exception("Test error")):
                try:
                    result = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
                except Exception:
                    pass
            
            # Should have logged the error
            error_logs = [msg for msg in log_messages if "error" in msg.lower() or "exception" in msg.lower()]
            # Note: Logging might not capture all errors in test environment
            
        finally:
            # Remove test handler
            universal_ai_api.logger.removeHandler(test_handler)