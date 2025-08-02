# Universal AI Core API Documentation

## Overview

The Universal AI Core provides a comprehensive API for domain-agnostic AI operations across molecular analysis, cybersecurity, and financial domains. This documentation covers all API endpoints, methods, and usage patterns.

## Core API Classes

### UniversalAIAPI

The main API interface providing unified access to all system capabilities.

```python
from universal_ai_core import UniversalAIAPI, APIConfig

# Initialize API
api = UniversalAIAPI(config_path="config.yaml", api_config=APIConfig())
```

#### Constructor Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `config_path` | `Optional[str]` | Path to configuration file | `None` |
| `api_config` | `Optional[APIConfig]` | API configuration object | `None` |

#### Methods

##### `process_data(data, extractors, use_cache=True)`

Process data using registered feature extractors.

**Parameters:**
- `data` (Any): Input data to process
- `extractors` (Optional[List[str]]): List of extractor names to use
- `use_cache` (bool): Whether to use caching for results

**Returns:**
- `ProcessingResult`: Result object with status, data, and metadata

**Example:**
```python
# Molecular data processing
molecular_data = {"smiles": ["CCO", "CC(=O)O"]}
result = api.process_data(molecular_data, ["molecular_descriptors"])

if result.status == "success":
    features = result.data["features"]
    print(f"Extracted {result.feature_count} features")
```

##### `validate_data(data, validators, use_cache=True)`

Validate data using registered validators.

**Parameters:**
- `data` (Any): Input data to validate
- `validators` (Optional[List[str]]): List of validator names to use
- `use_cache` (bool): Whether to use caching for results

**Returns:**
- `ValidationResult`: Result object with validation status and issues

**Example:**
```python
# Financial data validation
financial_data = {"portfolio": {"assets": ["AAPL", "GOOGL"]}}
result = api.validate_data(financial_data, ["financial_schema"])

if result.is_valid:
    print(f"Validation passed with score: {result.validation_score}")
else:
    print(f"Validation issues: {result.issues}")
```

##### `submit_async_task(operation_type, data, config=None, priority=5, timeout=None)`

Submit asynchronous task for processing.

**Parameters:**
- `operation_type` (str): Type of operation to perform
- `data` (Dict[str, Any]): Input data for the operation
- `config` (Optional[Dict[str, Any]]): Operation-specific configuration
- `priority` (int): Task priority (1-10, higher is more important)
- `timeout` (Optional[float]): Task timeout in seconds

**Returns:**
- `str`: Unique task ID for tracking

**Example:**
```python
# Submit cybersecurity analysis task
task_id = await api.submit_async_task(
    "threat_analysis",
    {"network_events": network_data},
    config={"detection_level": "high"},
    priority=8
)

# Get task result
result = api.get_task_result(task_id)
```

##### `get_task_result(task_id)`

Get result of async task.

**Parameters:**
- `task_id` (str): Task ID returned from submit_async_task

**Returns:**
- `Optional[TaskResult]`: Task result object or None if not found

##### `get_health_status()`

Get comprehensive system health status.

**Returns:**
- `Dict[str, Any]`: Health status information

**Example:**
```python
health = api.get_health_status()
print(f"Overall status: {health['overall']}")
print(f"Component health: {health['components']}")
```

##### `get_metrics(metric_name=None)`

Get performance metrics.

**Parameters:**
- `metric_name` (Optional[str]): Specific metric name to retrieve

**Returns:**
- `Dict[str, Any]`: Metrics data

##### `get_system_info()`

Get comprehensive system information.

**Returns:**
- `Dict[str, Any]`: System information including version, configuration, and statistics

##### `shutdown()`

Gracefully shutdown the API and cleanup resources.

### APIConfig

Configuration class for API behavior.

```python
from universal_ai_core import APIConfig

config = APIConfig(
    max_workers=8,
    enable_monitoring=True,
    enable_caching=True,
    cache_size=10000,
    debug_mode=False
)
```

#### Configuration Options

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `max_workers` | `int` | Maximum number of worker threads | `8` |
| `max_queue_size` | `int` | Maximum async task queue size | `1000` |
| `enable_monitoring` | `bool` | Enable performance monitoring | `True` |
| `enable_caching` | `bool` | Enable result caching | `True` |
| `cache_size` | `int` | Maximum cache entries | `10000` |
| `cache_ttl_hours` | `int` | Cache time-to-live in hours | `24` |
| `request_timeout` | `float` | Default request timeout seconds | `300.0` |
| `batch_size` | `int` | Default batch processing size | `100` |
| `enable_safety_checks` | `bool` | Enable safety and rate limiting | `True` |
| `rate_limit_requests_per_minute` | `int` | Rate limit threshold | `1000` |
| `debug_mode` | `bool` | Enable debug mode | `False` |
| `log_level` | `str` | Logging level | `"INFO"` |

## Data Processing API

### Feature Extractors

#### Molecular Feature Extractors

```python
# Basic molecular descriptors
result = api.process_data(
    {"smiles": ["CCO"]}, 
    ["molecular_descriptors"]
)

# Advanced molecular features
result = api.process_data(
    {"smiles": ["CCO"], "properties": [{"mw": 46.07}]}, 
    ["fingerprints", "pharmacophores"]
)
```

**Available Extractors:**
- `molecular_descriptors`: Basic molecular properties
- `fingerprints`: Molecular fingerprints
- `pharmacophores`: Pharmacophore features
- `admet_properties`: ADMET predictions

#### Cybersecurity Feature Extractors

```python
# Network traffic analysis
result = api.process_data(
    {
        "network_traffic": [
            {"src_ip": "192.168.1.1", "dst_ip": "10.0.0.1", "port": 80}
        ]
    },
    ["network_features"]
)

# Behavioral analysis
result = api.process_data(
    {
        "user_sessions": [
            {"user": "admin", "actions": ["login", "file_access"]}
        ]
    },
    ["behavioral_patterns"]
)
```

**Available Extractors:**
- `network_features`: Network traffic analysis
- `behavioral_patterns`: User behavior analysis
- `anomaly_detection`: Anomaly detection features
- `threat_indicators`: Threat intelligence features

#### Financial Feature Extractors

```python
# Technical indicators
result = api.process_data(
    {
        "ohlcv": [
            {"open": 100, "high": 105, "low": 98, "close": 103, "volume": 1000}
        ]
    },
    ["technical_indicators"]
)

# Risk metrics
result = api.process_data(
    {
        "returns": [0.02, -0.01, 0.03, -0.02],
        "portfolio_weights": [0.25, 0.25, 0.25, 0.25]
    },
    ["risk_metrics"]
)
```

**Available Extractors:**
- `technical_indicators`: Technical analysis indicators
- `risk_metrics`: Financial risk measurements
- `returns_analysis`: Return analysis features
- `market_microstructure`: Market structure features

### Models

#### Molecular Models

```python
# Neural network prediction
await api.orchestrator.process_request(
    domain="molecular",
    operation="predict",
    data={
        "features": molecular_features,
        "model_type": "neural_network"
    }
)

# Ensemble prediction
await api.orchestrator.process_request(
    domain="molecular",
    operation="ensemble_predict",
    data={
        "features": molecular_features,
        "models": ["neural_network", "random_forest", "svm"]
    }
)
```

#### Cybersecurity Models

```python
# Threat classification
await api.orchestrator.process_request(
    domain="cybersecurity",
    operation="classify_threat",
    data={
        "features": security_features,
        "classification_type": "malware_detection"
    }
)

# Anomaly detection
await api.orchestrator.process_request(
    domain="cybersecurity",
    operation="detect_anomaly",
    data={
        "behavioral_data": user_behavior,
        "baseline_model": "isolation_forest"
    }
)
```

#### Financial Models

```python
# Portfolio optimization
await api.orchestrator.process_request(
    domain="financial",
    operation="optimize_portfolio",
    data={
        "returns": historical_returns,
        "constraints": {"max_weight": 0.3, "min_weight": 0.05},
        "objective": "maximize_sharpe"
    }
)

# Risk prediction
await api.orchestrator.process_request(
    domain="financial",
    operation="predict_risk",
    data={
        "portfolio_data": portfolio,
        "time_horizon": "1_month",
        "confidence_level": 0.95
    }
)
```

## Validation API

### Schema Validators

```python
# Molecular schema validation
result = api.validate_data(
    {"molecules": [{"smiles": "CCO"}]},
    ["molecular_schema"]
)

# Financial schema validation
result = api.validate_data(
    {"portfolio": {"assets": ["AAPL"]}},
    ["financial_schema"]
)
```

### Data Quality Validators

```python
# Statistical validation
result = api.validate_data(
    {"returns": [0.01, 0.02, -0.01]},
    ["statistical_validation"]
)

# Completeness validation
result = api.validate_data(
    {"dataset": molecular_data},
    ["completeness"]
)
```

## Async Processing API

### Task Management

```python
# Submit multiple tasks
task_ids = []
for data_batch in data_batches:
    task_id = await api.submit_async_task(
        "batch_analysis",
        data_batch,
        priority=5
    )
    task_ids.append(task_id)

# Monitor task progress
for task_id in task_ids:
    result = api.get_task_result(task_id)
    if result and result.is_completed:
        if result.is_successful:
            print(f"Task {task_id} completed successfully")
            process_result(result.result)
        else:
            print(f"Task {task_id} failed: {result.error_message}")
```

### Batch Processing

```python
from universal_ai_core import BatchRequest

# Create batch request
batch_request = BatchRequest(
    batch_id="analysis_batch_001",
    items=[
        {"smiles": "CCO"},
        {"smiles": "CCN"},
        {"smiles": "CCC"}
    ],
    operation_type="molecular_analysis",
    priority=7
)

# Submit batch
batch_id = await api.process_batch(batch_request)

# Monitor batch progress
batch_result = api.get_task_result(batch_id)
```

## Caching API

### Cache Configuration

```python
# Configure caching
api = UniversalAIAPI(
    api_config=APIConfig(
        enable_caching=True,
        cache_size=20000,
        cache_ttl_hours=48
    )
)

# Use caching
result = api.process_data(data, extractors, use_cache=True)
```

### Cache Management

```python
# Get cache statistics
if api.cache:
    stats = api.cache.get_stats()
    print(f"Cache hit rate: {stats['hit_rate']:.2f}")
    print(f"Total entries: {stats['size']}")
    
    # Clear cache if needed
    api.cache.clear()
```

## Monitoring API

### Health Checks

```python
# Basic health check
health = api.get_health_status()
is_healthy = health['overall'] == 'healthy'

# Component-specific health
component_health = health['components']
for component, status in component_health.items():
    print(f"{component}: {'✓' if status['healthy'] else '✗'}")
```

### Performance Metrics

```python
# Get all metrics
metrics = api.get_metrics()

# API performance metrics
api_metrics = metrics['api']
print(f"Pending tasks: {api_metrics['pending_tasks']}")
print(f"Completed tasks: {api_metrics['completed_tasks']}")

# Cache performance
cache_stats = api_metrics['cache_stats']
print(f"Cache hit rate: {cache_stats['hit_rate']:.2f}")

# Get specific metric
memory_metric = api.get_metrics('system.memory_percent')
```

### System Information

```python
# Comprehensive system info
system_info = api.get_system_info()

print(f"API Version: {system_info['api_version']}")
print(f"Uptime: {system_info['uptime_seconds']} seconds")
print(f"Configuration: {system_info['configuration']}")
print(f"Statistics: {system_info['statistics']}")
```

## Error Handling

### Exception Types

```python
from universal_ai_core.exceptions import (
    ValidationError,
    ProcessingError,
    ConfigurationError,
    PluginError
)

try:
    result = api.process_data(invalid_data, ["molecular_descriptors"])
except ValidationError as e:
    print(f"Validation failed: {e}")
except ProcessingError as e:
    print(f"Processing failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Error Response Format

```python
# Processing errors
result = api.process_data(data, extractors)
if result.status == "error":
    print(f"Error: {result.error_message}")
    print(f"Error code: {result.error_code}")

# Validation errors
validation = api.validate_data(data, validators)
if not validation.is_valid:
    for issue in validation.issues:
        print(f"Issue: {issue['message']} (severity: {issue['severity']})")
```

## Rate Limiting

### Configuration

```python
# Configure rate limiting
api = UniversalAIAPI(
    api_config=APIConfig(
        enable_safety_checks=True,
        rate_limit_requests_per_minute=500
    )
)
```

### Handling Rate Limits

```python
import time
from universal_ai_core.exceptions import RateLimitError

try:
    result = api.process_data(data, extractors)
except RateLimitError as e:
    print(f"Rate limited: {e}")
    # Wait and retry
    time.sleep(60)
    result = api.process_data(data, extractors)
```

## Factory Functions

### Quick API Creation

```python
from universal_ai_core import create_api, create_development_api, create_production_api

# Default API
api = create_api()

# Development API (debug mode, relaxed limits)
dev_api = create_development_api()

# Production API (optimized, strict limits)
prod_api = create_production_api()

# Custom API
custom_api = create_api(
    max_workers=16,
    enable_caching=True,
    cache_size=50000,
    enable_monitoring=True
)
```

## Plugin API

### Plugin Management

```python
# List available plugins
available = api.core.plugin_manager.list_available_plugins()
print(f"Available plugins: {available}")

# Load specific plugin
success = api.core.plugin_manager.load_plugin("feature_extractors", "molecular")

# Get plugin instance
plugin = api.core.plugin_manager.get_plugin("feature_extractors", "molecular")
```

### Custom Plugin Registration

```python
from universal_ai_core.plugins.base import BasePlugin

class CustomPlugin(BasePlugin):
    def __init__(self, config):
        super().__init__(config)
    
    def process(self, data):
        return {"custom_result": "processed"}

# Register plugin
api.core.plugin_manager.register_plugin("custom", CustomPlugin)
```

## Configuration API

### Dynamic Configuration

```python
from universal_ai_core.config import get_config_manager, get_config

# Get configuration manager
config_manager = get_config_manager("config.yaml")

# Get current configuration
config = get_config("config.yaml")

# Validate configuration
is_valid, errors = config_manager.validate_config(config_dict)
```

### Hot Reload

```python
# Start configuration watching
config_manager.start_watching()

# Configuration will reload automatically when file changes

# Stop watching
config_manager.stop_watching()
```

## Best Practices

### Performance Optimization

1. **Use Caching**: Enable caching for repeated operations
```python
api = create_api(enable_caching=True, cache_size=20000)
```

2. **Batch Processing**: Process multiple items together
```python
# Instead of individual requests
for item in items:
    result = api.process_data(item, extractors)

# Use batch processing
batch_request = BatchRequest(
    batch_id="batch_001",
    items=items,
    operation_type="analysis"
)
batch_id = await api.process_batch(batch_request)
```

3. **Async Operations**: Use async for I/O-bound operations
```python
# Submit tasks asynchronously
tasks = []
for data in datasets:
    task_id = await api.submit_async_task("analysis", data)
    tasks.append(task_id)

# Collect results
results = []
for task_id in tasks:
    result = api.get_task_result(task_id)
    if result and result.is_completed:
        results.append(result)
```

### Error Handling

1. **Graceful Degradation**: Handle errors appropriately
```python
try:
    result = api.process_data(data, extractors)
    if result.status == "success":
        return result.data
    else:
        return fallback_processing(data)
except Exception as e:
    logger.error(f"Processing failed: {e}")
    return None
```

2. **Retry Logic**: Implement retry for transient failures
```python
import time

def process_with_retry(data, max_retries=3):
    for attempt in range(max_retries):
        try:
            return api.process_data(data, extractors)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

### Resource Management

1. **Proper Cleanup**: Always shutdown API when done
```python
try:
    # Use API
    result = api.process_data(data, extractors)
finally:
    api.shutdown()
```

2. **Monitor Resources**: Track system health
```python
health = api.get_health_status()
if health['overall'] != 'healthy':
    print("System health issue detected")
    # Take corrective action
```

3. **Memory Management**: Monitor memory usage
```python
metrics = api.get_metrics()
memory_usage = metrics.get('system.memory_percent', 0)
if memory_usage > 80:
    # Consider reducing batch sizes or clearing caches
    if api.cache:
        api.cache.clear()
```