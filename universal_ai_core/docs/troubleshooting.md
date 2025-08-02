# Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting information for Universal AI Core, covering common issues, diagnostic procedures, and solutions adapted from Saraphis troubleshooting patterns.

## Quick Diagnostic Commands

### System Health Check
```python
from universal_ai_core import create_api

# Create API instance
api = create_api()

# Get comprehensive system health
health = api.get_health_status()
print(f"Overall health: {health['overall']}")

# Check individual components
for component, status in health['components'].items():
    print(f"{component}: {'✓' if status['healthy'] else '✗'}")
    if not status['healthy']:
        print(f"  Issues: {status.get('issues', [])}")

# Get system metrics
metrics = api.get_metrics()
print(f"Memory usage: {metrics.get('system.memory_percent', 'Unknown')}%")
print(f"Active workers: {metrics.get('active_workers', 'Unknown')}")

# Cleanup
api.shutdown()
```

### Plugin Status Check
```python
from universal_ai_core import create_api

api = create_api()

# Check available plugins
available = api.core.plugin_manager.list_available_plugins()
print(f"Available plugins: {available}")

# Check loaded plugins
loaded = api.core.plugin_manager.get_loaded_plugins()
print(f"Loaded plugins: {loaded}")

# Validate specific plugin
plugin_types = ["feature_extractors", "models", "proof_languages", "knowledge_bases"]
for plugin_type in plugin_types:
    try:
        plugins = api.core.plugin_manager.get_plugins_by_type(plugin_type)
        print(f"{plugin_type}: {list(plugins.keys())}")
    except Exception as e:
        print(f"{plugin_type}: Error - {e}")

api.shutdown()
```

## Common Issues and Solutions

### 1. Installation and Setup Issues

#### Issue: Import Errors
**Symptoms:**
```
ImportError: No module named 'universal_ai_core'
ModuleNotFoundError: No module named 'rdkit'
```

**Solutions:**
```bash
# Verify installation
pip list | grep universal-ai-core

# Reinstall in development mode
pip uninstall universal-ai-core
pip install -e .

# Install optional dependencies
pip install rdkit-pypi  # For molecular analysis
pip install torch torchvision  # For neural networks
pip install scikit-learn  # For machine learning
```

**Verify Installation:**
```python
# Test basic import
try:
    from universal_ai_core import create_api
    print("✓ Universal AI Core installed correctly")
except ImportError as e:
    print(f"✗ Import error: {e}")

# Test optional dependencies
optional_deps = {
    'rdkit': 'from rdkit import Chem',
    'torch': 'import torch',
    'sklearn': 'import sklearn',
    'numpy': 'import numpy',
    'pandas': 'import pandas'
}

for name, import_cmd in optional_deps.items():
    try:
        exec(import_cmd)
        print(f"✓ {name} available")
    except ImportError:
        print(f"○ {name} not available (optional)")
```

#### Issue: Configuration File Not Found
**Symptoms:**
```
FileNotFoundError: Configuration file not found: config.yaml
ConfigurationError: Invalid configuration format
```

**Solutions:**
```python
# Create minimal configuration
from universal_ai_core.config import UniversalConfiguration

minimal_config = {
    "core": {
        "max_workers": 4,
        "enable_monitoring": True,
        "debug_mode": False
    },
    "plugins": {
        "feature_extractors": {
            "molecular": {"enabled": True},
            "cybersecurity": {"enabled": True},
            "financial": {"enabled": True}
        }
    }
}

# Save configuration
import yaml
with open('config.yaml', 'w') as f:
    yaml.dump(minimal_config, f)

# Or use without configuration file
from universal_ai_core import create_api
api = create_api()  # Uses default configuration
```

### 2. Plugin Issues

#### Issue: Plugin Loading Failures
**Symptoms:**
```
PluginError: Failed to load plugin 'molecular'
ImportError: No module named 'rdkit'
AttributeError: Plugin missing required method
```

**Diagnosis:**
```python
from universal_ai_core import create_api
from universal_ai_core.core.plugin_manager import PluginManager

api = create_api()
manager = api.core.plugin_manager

# Test plugin loading individually
plugin_combinations = [
    ("feature_extractors", "molecular"),
    ("feature_extractors", "cybersecurity"),
    ("feature_extractors", "financial"),
    ("models", "molecular"),
    ("models", "cybersecurity"),
    ("models", "financial")
]

for plugin_type, plugin_name in plugin_combinations:
    try:
        success = manager.load_plugin(plugin_type, plugin_name)
        print(f"✓ {plugin_type}.{plugin_name}: {'Loaded' if success else 'Failed'}")
    except Exception as e:
        print(f"✗ {plugin_type}.{plugin_name}: {e}")

api.shutdown()
```

**Solutions:**
```bash
# Install missing dependencies
pip install rdkit-pypi  # For molecular plugins
pip install scikit-learn torch  # For ML models
pip install pandas numpy  # For data processing

# Check plugin configuration
python -c "
from universal_ai_core.config import get_config
config = get_config()
print('Plugin config:', config.get('plugins', {}))
"
```

#### Issue: Plugin Performance Problems
**Symptoms:**
- Slow plugin execution
- High memory usage
- Plugin timeouts

**Diagnosis:**
```python
import time
from universal_ai_core import create_api

api = create_api()

# Test plugin performance
test_data = {"molecules": [{"smiles": "CCO"}]}

start_time = time.time()
try:
    result = api.process_data(test_data, ["molecular_descriptors"])
    duration = time.time() - start_time
    print(f"Processing time: {duration:.2f}s")
    print(f"Result status: {result.status}")
    if hasattr(result, 'processing_time'):
        print(f"Plugin processing time: {result.processing_time:.2f}s")
except Exception as e:
    print(f"Plugin error: {e}")

# Check memory usage
metrics = api.get_metrics()
print(f"Memory usage: {metrics.get('system.memory_percent', 'Unknown')}%")

api.shutdown()
```

**Solutions:**
```python
# Enable performance monitoring
from universal_ai_core import create_api, APIConfig

config = APIConfig(
    enable_monitoring=True,
    debug_mode=True,
    max_workers=2  # Reduce if memory constrained
)

api = create_api(api_config=config)

# Monitor plugin performance
import time
start_memory = api.get_metrics().get('system.memory_percent', 0)

result = api.process_data(test_data, ["molecular_descriptors"])

end_memory = api.get_metrics().get('system.memory_percent', 0)
print(f"Memory increase: {end_memory - start_memory}%")
```

### 3. Performance Issues

#### Issue: High Memory Usage
**Symptoms:**
- System becomes slow
- Out of memory errors
- High swap usage

**Diagnosis:**
```python
import psutil
import gc
from universal_ai_core import create_api

def get_memory_info():
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'percent': process.memory_percent()
    }

print("Initial memory:", get_memory_info())

# Create API
api = create_api()
print("After API creation:", get_memory_info())

# Process data
large_dataset = {"molecules": [{"smiles": f"C{i}"} for i in range(1000)]}
result = api.process_data(large_dataset, ["molecular_descriptors"])
print("After processing:", get_memory_info())

# Force garbage collection
gc.collect()
print("After GC:", get_memory_info())

api.shutdown()
print("After shutdown:", get_memory_info())
```

**Solutions:**
```python
# Optimize memory usage
from universal_ai_core import create_api, APIConfig

# Reduce memory footprint
config = APIConfig(
    max_workers=2,  # Reduce worker count
    enable_caching=False,  # Disable caching if memory constrained
    batch_size=50  # Process smaller batches
)

api = create_api(api_config=config)

# Process in smaller batches
def process_in_batches(data, batch_size=50):
    items = data.get("molecules", [])
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = {"molecules": items[i:i+batch_size]}
        result = api.process_data(batch, ["molecular_descriptors"])
        if result.status == "success":
            results.extend(result.data.get("features", []))
        
        # Force cleanup between batches
        import gc
        gc.collect()
    
    return results

# Use batch processing for large datasets
results = process_in_batches(large_dataset, batch_size=50)
```

#### Issue: Slow Performance
**Symptoms:**
- Long processing times
- API timeouts
- Poor response times

**Diagnosis:**
```python
import time
from universal_ai_core import create_api

api = create_api()

# Benchmark different operations
operations = [
    ("Health check", lambda: api.get_health_status()),
    ("System info", lambda: api.get_system_info()),
    ("Metrics", lambda: api.get_metrics()),
    ("Small dataset", lambda: api.process_data(
        {"molecules": [{"smiles": "CCO"}]}, 
        ["molecular_descriptors"]
    ))
]

for name, operation in operations:
    start_time = time.time()
    try:
        result = operation()
        duration = time.time() - start_time
        print(f"{name}: {duration:.3f}s")
    except Exception as e:
        print(f"{name}: Error - {e}")

api.shutdown()
```

**Solutions:**
```python
# Performance optimization
from universal_ai_core import create_api, APIConfig

# Optimize for performance
config = APIConfig(
    max_workers=8,  # Increase workers for CPU-bound tasks
    enable_caching=True,  # Enable caching
    cache_size=10000,
    batch_size=100
)

api = create_api(api_config=config)

# Use caching for repeated operations
data = {"molecules": [{"smiles": "CCO"}]}

# First call (uncached)
start_time = time.time()
result1 = api.process_data(data, ["molecular_descriptors"], use_cache=True)
time1 = time.time() - start_time

# Second call (cached)
start_time = time.time()
result2 = api.process_data(data, ["molecular_descriptors"], use_cache=True)
time2 = time.time() - start_time

print(f"First call: {time1:.3f}s")
print(f"Second call (cached): {time2:.3f}s")
print(f"Speedup: {time1/time2:.1f}x")
```

### 4. API and Network Issues

#### Issue: Connection Errors
**Symptoms:**
```
ConnectionError: Failed to connect to API
TimeoutError: Request timed out
```

**Diagnosis:**
```python
import requests
import time

def test_api_connectivity(base_url="http://localhost:8000"):
    endpoints = ["/health", "/metrics", "/system-info"]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            print(f"✓ {endpoint}: {response.status_code}")
        except requests.ConnectionError:
            print(f"✗ {endpoint}: Connection refused")
        except requests.Timeout:
            print(f"✗ {endpoint}: Timeout")
        except Exception as e:
            print(f"✗ {endpoint}: {e}")

# Test local API
test_api_connectivity()
```

**Solutions:**
```python
# Start local API server
from universal_ai_core import create_api

api = create_api()

# Test internal API
try:
    health = api.get_health_status()
    print("✓ Internal API working")
except Exception as e:
    print(f"✗ Internal API error: {e}")

# For external API, check firewall and network settings
import socket

def check_port(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

if check_port("localhost", 8000):
    print("✓ Port 8000 accessible")
else:
    print("✗ Port 8000 not accessible")
```

#### Issue: Rate Limiting
**Symptoms:**
```
RateLimitError: Too many requests
HTTP 429: Rate limit exceeded
```

**Solutions:**
```python
# Configure rate limiting
from universal_ai_core import create_api, APIConfig
import time

config = APIConfig(
    enable_safety_checks=True,
    rate_limit_requests_per_minute=100  # Adjust as needed
)

api = create_api(api_config=config)

# Implement backoff strategy
def process_with_backoff(data, extractors, max_retries=3):
    for attempt in range(max_retries):
        try:
            return api.process_data(data, extractors)
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e

# Use with backoff
result = process_with_backoff(
    {"molecules": [{"smiles": "CCO"}]}, 
    ["molecular_descriptors"]
)
```

### 5. Data Processing Issues

#### Issue: Invalid Input Data
**Symptoms:**
```
ValidationError: Invalid data format
TypeError: Expected dict, got str
ValueError: Invalid SMILES string
```

**Diagnosis:**
```python
from universal_ai_core import create_api

api = create_api()

# Test data validation
test_cases = [
    # Valid data
    {"molecules": [{"smiles": "CCO"}]},
    
    # Invalid formats
    "invalid_string",
    {"molecules": "not_a_list"},
    {"molecules": [{"invalid": "no_smiles"}]},
    {"molecules": [{"smiles": "INVALID_SMILES"}]}
]

for i, test_data in enumerate(test_cases):
    try:
        # Test validation first
        validation = api.validate_data(test_data, ["molecular_schema"])
        print(f"Test {i}: Validation - {'✓' if validation.is_valid else '✗'}")
        
        if not validation.is_valid:
            for issue in validation.issues:
                print(f"  Issue: {issue['message']}")
        
        # Test processing
        result = api.process_data(test_data, ["molecular_descriptors"])
        print(f"Test {i}: Processing - {'✓' if result.status == 'success' else '✗'}")
        
    except Exception as e:
        print(f"Test {i}: Error - {e}")

api.shutdown()
```

**Solutions:**
```python
# Implement data validation and cleaning
from universal_ai_core import create_api

def clean_molecular_data(data):
    """Clean and validate molecular data."""
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary")
    
    if "molecules" not in data:
        raise ValueError("Data must contain 'molecules' key")
    
    molecules = data["molecules"]
    if not isinstance(molecules, list):
        raise ValueError("Molecules must be a list")
    
    cleaned_molecules = []
    for mol in molecules:
        if isinstance(mol, dict) and "smiles" in mol:
            # Basic SMILES validation
            smiles = mol["smiles"].strip()
            if smiles and len(smiles) > 0:
                cleaned_molecules.append({"smiles": smiles})
    
    return {"molecules": cleaned_molecules}

# Usage
api = create_api()

try:
    # Clean data before processing
    raw_data = {"molecules": [{"smiles": " CCO "}, {"smiles": ""}]}
    cleaned_data = clean_molecular_data(raw_data)
    
    result = api.process_data(cleaned_data, ["molecular_descriptors"])
    print("Processing successful")
    
except Exception as e:
    print(f"Error: {e}")

api.shutdown()
```

#### Issue: Processing Errors
**Symptoms:**
```
ProcessingError: Feature extraction failed
RuntimeError: Plugin execution error
```

**Diagnosis:**
```python
from universal_ai_core import create_api
import logging

# Enable debug logging
logging.getLogger('universal_ai_core').setLevel(logging.DEBUG)

api = create_api()

# Test with minimal data first
minimal_data = {"molecules": [{"smiles": "C"}]}  # Methane - simplest

try:
    result = api.process_data(minimal_data, ["molecular_descriptors"])
    print(f"Minimal data result: {result.status}")
    
    if result.status == "success":
        print("Basic processing works, trying more complex data...")
        
        complex_data = {"molecules": [{"smiles": "CC(C)C(=O)N1CCN(CC1)C2=CC=C(C=C2)F"}]}
        result2 = api.process_data(complex_data, ["molecular_descriptors"])
        print(f"Complex data result: {result2.status}")
    
except Exception as e:
    print(f"Processing error: {e}")
    print(f"Error type: {type(e)}")

api.shutdown()
```

### 6. Configuration Issues

#### Issue: Configuration Errors
**Symptoms:**
```
ConfigurationError: Invalid configuration
KeyError: Missing required configuration key
```

**Solutions:**
```python
from universal_ai_core.config import get_config_manager, UniversalConfiguration

# Validate configuration
def validate_configuration(config_path="config.yaml"):
    try:
        manager = get_config_manager(config_path)
        config_dict = manager.load_config()
        
        # Validate configuration
        config = UniversalConfiguration(**config_dict)
        print("✓ Configuration valid")
        return True
        
    except FileNotFoundError:
        print("✗ Configuration file not found")
        print("Creating default configuration...")
        
        default_config = {
            "core": {
                "max_workers": 4,
                "enable_monitoring": True,
                "debug_mode": False
            },
            "plugins": {
                "feature_extractors": {
                    "molecular": {"enabled": True},
                    "cybersecurity": {"enabled": True},
                    "financial": {"enabled": True}
                }
            }
        }
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
        
        return False
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

# Test configuration
if validate_configuration():
    print("Configuration is valid")
else:
    print("Configuration fixed or created")
```

## Debugging Tools and Techniques

### 1. Enable Debug Mode
```python
from universal_ai_core import create_api, APIConfig
import logging

# Set up debug logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('universal_ai_core').setLevel(logging.DEBUG)

# Create API with debug configuration
config = APIConfig(
    debug_mode=True,
    log_level="DEBUG",
    enable_monitoring=True
)

api = create_api(api_config=config)

# All operations will now provide detailed logging
result = api.process_data({"molecules": [{"smiles": "CCO"}]}, ["molecular_descriptors"])
```

### 2. Memory Profiling
```python
import tracemalloc
import gc
from universal_ai_core import create_api

# Start memory tracing
tracemalloc.start()

# Take snapshot before
snapshot_before = tracemalloc.take_snapshot()

# Create API and process data
api = create_api()
result = api.process_data({"molecules": [{"smiles": "CCO"}]}, ["molecular_descriptors"])

# Take snapshot after
snapshot_after = tracemalloc.take_snapshot()

# Compare snapshots
top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')

print("Top 10 memory allocations:")
for stat in top_stats[:10]:
    print(stat)

api.shutdown()
```

### 3. Performance Profiling
```python
import cProfile
import pstats
from universal_ai_core import create_api

def profile_processing():
    api = create_api()
    data = {"molecules": [{"smiles": f"C{i}"} for i in range(10)]}
    result = api.process_data(data, ["molecular_descriptors"])
    api.shutdown()
    return result

# Run profiler
profiler = cProfile.Profile()
profiler.enable()

result = profile_processing()

profiler.disable()

# Print stats
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### 4. Network Debugging
```python
import requests
import time

def debug_api_requests(base_url="http://localhost:8000"):
    """Debug API requests with detailed timing."""
    
    endpoints = [
        ("/health", "GET"),
        ("/metrics", "GET"),
        ("/process", "POST")
    ]
    
    for endpoint, method in endpoints:
        url = f"{base_url}{endpoint}"
        
        start_time = time.time()
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=30)
            else:
                test_data = {"molecules": [{"smiles": "CCO"}]}
                response = requests.post(url, json=test_data, timeout=30)
            
            duration = time.time() - start_time
            
            print(f"{method} {endpoint}:")
            print(f"  Status: {response.status_code}")
            print(f"  Duration: {duration:.3f}s")
            print(f"  Response size: {len(response.content)} bytes")
            
            if response.status_code != 200:
                print(f"  Error: {response.text}")
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"{method} {endpoint}:")
            print(f"  Error: {e}")
            print(f"  Duration: {duration:.3f}s")

# Test API endpoints
debug_api_requests()
```

## Environment-Specific Troubleshooting

### Docker Environment
```bash
# Check container status
docker ps -a

# View container logs
docker logs universal-ai-core

# Access container for debugging
docker exec -it universal-ai-core /bin/bash

# Check resource usage
docker stats universal-ai-core

# Test container health
docker inspect universal-ai-core | grep -A 10 "Health"
```

### Kubernetes Environment
```bash
# Check pod status
kubectl get pods -n universal-ai-core

# View pod logs
kubectl logs -f deployment/universal-ai-core -n universal-ai-core

# Describe pod for events
kubectl describe pod <pod-name> -n universal-ai-core

# Access pod for debugging
kubectl exec -it <pod-name> -n universal-ai-core -- /bin/bash

# Check resource usage
kubectl top pods -n universal-ai-core
```

### Production Environment
```bash
# Check system resources
free -h
df -h
top

# Check network connectivity
netstat -tulpn | grep 8000
curl -v http://localhost:8000/health

# Check application logs
tail -f /var/log/universal-ai-core/app.log

# Monitor system metrics
iostat 1
vmstat 1
```

## Getting Help

### 1. Collect System Information
```python
from universal_ai_core import create_api
import platform
import sys

def collect_debug_info():
    """Collect system information for debugging."""
    
    info = {
        "system": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "architecture": platform.architecture()
        }
    }
    
    try:
        api = create_api()
        
        # Get API information
        system_info = api.get_system_info()
        health = api.get_health_status()
        metrics = api.get_metrics()
        
        info.update({
            "api": {
                "version": system_info.get("api_version"),
                "uptime": system_info.get("uptime_seconds"),
                "health": health["overall"],
                "memory_percent": metrics.get("system.memory_percent")
            }
        })
        
        api.shutdown()
        
    except Exception as e:
        info["api_error"] = str(e)
    
    return info

# Collect and print debug information
debug_info = collect_debug_info()
print("=== Debug Information ===")
for section, data in debug_info.items():
    print(f"\n{section.upper()}:")
    for key, value in data.items():
        print(f"  {key}: {value}")
```

### 2. Create Issue Report
When reporting issues, include:

1. **Environment Information**: OS, Python version, Universal AI Core version
2. **Configuration**: Sanitized configuration file
3. **Error Messages**: Complete error traceback
4. **Reproduction Steps**: Minimal code to reproduce the issue
5. **Expected vs Actual Behavior**: What should happen vs what actually happens

### 3. Enable Verbose Logging
```python
import logging

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('universal_ai_core_debug.log'),
        logging.StreamHandler()
    ]
)

# Enable all Universal AI Core logging
logging.getLogger('universal_ai_core').setLevel(logging.DEBUG)
```

This comprehensive troubleshooting guide should help resolve most common issues with Universal AI Core. For additional support, refer to the documentation or contact the development team.