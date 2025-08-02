"""
Pytest configuration and shared fixtures for Universal AI Core tests.
Adapted from Saraphis/charon_builder test patterns.
"""

import asyncio
import logging
import pytest
import tempfile
import shutil
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, MagicMock, patch
import threading
import time

# Import core modules for testing
try:
    from universal_ai_core.core.universal_ai_core import UniversalAICore
    from universal_ai_core.core.plugin_manager import PluginManager
    from universal_ai_core.config.config_manager import ConfigurationManager, UniversalConfiguration
    from universal_ai_core.utils.data_utils import DataProcessor, ProcessingResult
    from universal_ai_core.utils.validation_utils import ValidationEngine, ValidationResult
    from universal_ai_core import UniversalAIAPI, APIConfig
except ImportError:
    # Fallback imports for test environment
    pass


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_directory():
    """Provide a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_logger():
    """Provide a mock logger for testing."""
    logger = Mock(spec=logging.Logger)
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.critical = Mock()
    logger.getChild = Mock(return_value=logger)
    return logger


@pytest.fixture
def sample_config():
    """Provide sample configuration for testing."""
    return {
        "core": {
            "max_workers": 2,
            "enable_monitoring": True,
            "cache_enabled": True,
            "debug_mode": True
        },
        "plugins": {
            "feature_extractors": {
                "molecular": {
                    "enabled": True,
                    "rdkit_enabled": False,  # Disable for testing
                    "basic_descriptors": ["molecular_weight", "logp"]
                },
                "cybersecurity": {
                    "enabled": True,
                    "threat_detection": True
                },
                "financial": {
                    "enabled": True,
                    "technical_indicators": ["sma", "rsi"]
                }
            },
            "models": {
                "molecular": {"enabled": True, "model_type": "neural_network"},
                "cybersecurity": {"enabled": True, "model_type": "ensemble"},
                "financial": {"enabled": True, "model_type": "lstm"}
            },
            "proof_languages": {
                "molecular": {"enabled": True},
                "cybersecurity": {"enabled": True},
                "financial": {"enabled": True}
            },
            "knowledge_bases": {
                "molecular": {"enabled": True},
                "cybersecurity": {"enabled": True},
                "financial": {"enabled": True}
            }
        },
        "logging": {
            "level": "DEBUG",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }


@pytest.fixture
def sample_config_file(temp_directory, sample_config):
    """Create sample configuration file."""
    config_file = temp_directory / "config.yaml"
    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(sample_config, f)
    return config_file


@pytest.fixture
def mock_config_manager(sample_config):
    """Provide mock configuration manager."""
    manager = Mock(spec=ConfigurationManager)
    manager.get_config.return_value = UniversalConfiguration(**sample_config)
    manager.config = UniversalConfiguration(**sample_config)
    manager.validate_config.return_value = (True, [])
    return manager


@pytest.fixture
def sample_molecular_data():
    """Provide sample molecular data for testing."""
    return {
        "smiles": ["CCO", "CC(=O)O", "c1ccccc1"],
        "properties": [
            {"molecular_weight": 46.07, "logp": -0.31},
            {"molecular_weight": 60.05, "logp": -0.17},
            {"molecular_weight": 78.11, "logp": 2.13}
        ]
    }


@pytest.fixture
def sample_cybersecurity_data():
    """Provide sample cybersecurity data for testing."""
    return {
        "network_traffic": [
            {"src_ip": "192.168.1.1", "dst_ip": "10.0.0.1", "port": 80, "protocol": "TCP"},
            {"src_ip": "192.168.1.2", "dst_ip": "10.0.0.2", "port": 443, "protocol": "TCP"}
        ],
        "logs": [
            {"timestamp": "2024-01-01 10:00:00", "level": "INFO", "message": "User login"},
            {"timestamp": "2024-01-01 10:01:00", "level": "WARN", "message": "Failed login attempt"}
        ]
    }


@pytest.fixture
def sample_financial_data():
    """Provide sample financial data for testing."""
    return {
        "ohlcv": [
            {"timestamp": "2024-01-01", "open": 100.0, "high": 105.0, "low": 98.0, "close": 103.0, "volume": 1000},
            {"timestamp": "2024-01-02", "open": 103.0, "high": 107.0, "low": 101.0, "close": 106.0, "volume": 1200}
        ],
        "indicators": {
            "sma_20": [101.5, 104.5],
            "rsi": [45.2, 58.7]
        }
    }


@pytest.fixture(autouse=True)
def cleanup_resources():
    """Automatically cleanup resources after each test."""
    yield
    
    # Cleanup any global state
    import gc
    gc.collect()
    
    # Clear thread-local storage
    threading.current_thread().__dict__.clear()


@pytest.fixture
def mock_plugin_manager():
    """Provide mock plugin manager."""
    manager = Mock(spec=PluginManager)
    manager.plugins = {
        "feature_extractors": {
            "molecular": Mock(),
            "cybersecurity": Mock(),
            "financial": Mock()
        },
        "models": {
            "molecular": Mock(),
            "cybersecurity": Mock(),
            "financial": Mock()
        },
        "proof_languages": {
            "molecular": Mock(),
            "cybersecurity": Mock(),
            "financial": Mock()
        },
        "knowledge_bases": {
            "molecular": Mock(),
            "cybersecurity": Mock(),
            "financial": Mock()
        }
    }
    manager.list_available_plugins.return_value = manager.plugins
    manager.get_plugin.return_value = Mock()
    manager.load_plugin.return_value = True
    return manager


@pytest.fixture
def mock_data_processor():
    """Provide mock data processor."""
    processor = Mock(spec=DataProcessor)
    processor.prepare_features.return_value = ProcessingResult(
        status="success",
        data={"features": [1, 2, 3], "metadata": {"processor": "test"}},
        processing_time=0.1,
        feature_count=3
    )
    return processor


@pytest.fixture
def mock_validation_engine():
    """Provide mock validation engine."""
    engine = Mock(spec=ValidationEngine)
    engine.validate.return_value = ValidationResult(
        is_valid=True,
        validation_score=0.95,
        issues=[],
        passed_validators=["schema", "statistical"],
        failed_validators=[],
        processing_time=0.05
    )
    return engine


@pytest.fixture
def universal_ai_core(mock_config_manager):
    """Provide Universal AI Core instance."""
    with patch('universal_ai_core.core.universal_ai_core.get_config_manager', return_value=mock_config_manager):
        core = UniversalAICore(mock_config_manager.get_config())
        return core


@pytest.fixture
def api_config():
    """Provide API configuration for testing."""
    return APIConfig(
        max_workers=2,
        max_queue_size=100,
        enable_monitoring=False,  # Disable for testing
        enable_caching=True,
        cache_size=100,
        debug_mode=True,
        log_level="DEBUG",
        rate_limit_requests_per_minute=1000
    )


@pytest.fixture
def universal_ai_api(temp_directory, api_config, sample_config_file):
    """Provide Universal AI API instance."""
    api = UniversalAIAPI(config_path=str(sample_config_file), api_config=api_config)
    yield api
    try:
        api.shutdown()
    except Exception:
        pass


@pytest.fixture
def performance_config():
    """Provide performance testing configuration."""
    return {
        "max_concurrent_operations": 5,
        "timeout_seconds": 10,
        "memory_limit_mb": 256,
        "cpu_limit_percent": 80,
        "benchmark_iterations": 10
    }


class TestUtils:
    """Utility class for testing helpers."""
    
    @staticmethod
    def create_test_file(directory: Path, filename: str, content: str = "test content") -> Path:
        """Create a test file with specified content."""
        file_path = directory / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path
    
    @staticmethod
    def create_test_config(directory: Path, config_data: Dict[str, Any]) -> Path:
        """Create a test configuration file."""
        config_file = directory / "test_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        return config_file
    
    @staticmethod
    def create_sample_data_files(directory: Path) -> Dict[str, Path]:
        """Create sample data files for testing."""
        files = {}
        
        # Molecular data
        molecular_data = {
            "molecules": [
                {"smiles": "CCO", "name": "ethanol"},
                {"smiles": "CC(=O)O", "name": "acetic acid"}
            ]
        }
        files["molecular"] = TestUtils.create_test_file(
            directory, "molecular_data.json", json.dumps(molecular_data)
        )
        
        # Cybersecurity data
        security_data = {
            "events": [
                {"type": "login", "user": "admin", "success": True},
                {"type": "login", "user": "guest", "success": False}
            ]
        }
        files["cybersecurity"] = TestUtils.create_test_file(
            directory, "security_data.json", json.dumps(security_data)
        )
        
        # Financial data
        financial_data = {
            "prices": [
                {"symbol": "AAPL", "price": 150.0, "volume": 1000000},
                {"symbol": "GOOGL", "price": 2800.0, "volume": 500000}
            ]
        }
        files["financial"] = TestUtils.create_test_file(
            directory, "financial_data.json", json.dumps(financial_data)
        )
        
        return files
    
    @staticmethod
    async def wait_for_condition(
        condition_func, 
        timeout: float = 5.0, 
        check_interval: float = 0.1
    ) -> bool:
        """Wait for a condition to become true."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            await asyncio.sleep(check_interval)
        
        return False
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs):
        """Measure execution time of a function."""
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    
    @staticmethod
    async def measure_async_execution_time(func, *args, **kwargs):
        """Measure execution time of an async function."""
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils


# Markers for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow
pytest.mark.async_test = pytest.mark.asyncio
pytest.mark.memory = pytest.mark.memory
pytest.mark.compatibility = pytest.mark.compatibility


# Custom pytest markers configuration
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "async_test: marks tests as async tests")
    config.addinivalue_line("markers", "memory: marks tests as memory tests")
    config.addinivalue_line("markers", "compatibility: marks tests as compatibility tests")


# Async test helpers
@pytest.fixture
def async_test_timeout():
    """Provide timeout for async tests."""
    return 30.0  # 30 seconds


def pytest_runtest_setup(item):
    """Setup for each test run."""
    # Configure logging for tests
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def pytest_runtest_teardown(item):
    """Teardown after each test run."""
    # Clear any cached singletons
    import gc
    gc.collect()


# Memory monitoring fixtures
@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    try:
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        yield {
            'get_memory_usage': lambda: process.memory_info().rss / 1024 / 1024,
            'initial_memory': initial_memory
        }
    except ImportError:
        # Fallback if psutil not available
        yield {
            'get_memory_usage': lambda: 0.0,
            'initial_memory': 0.0
        }


# Stress testing helpers
@pytest.fixture
def stress_test_config():
    """Configuration for stress tests."""
    return {
        'concurrent_requests': 10,
        'request_duration': 5.0,
        'memory_limit_mb': 512,
        'expected_response_time_ms': 1000
    }