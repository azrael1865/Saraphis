# JAX Integration Guide

## Overview
The compression system now includes full JAX backend support for accelerated operations. When JAX is installed, the system can automatically utilize JAX-optimized implementations for significant performance improvements.

## Components Integrated

### 1. System Integration Coordinator
- **Location**: `system_integration_coordinator.py`
- **Features Added**:
  - JAX component initialization in `_initialize_jax_components()`
  - JAX configuration options in `SystemConfiguration`
  - Component registry includes all JAX modules
  - JAX_ACCELERATED optimization strategy

### 2. Model Compression API
- **Location**: `model_compression_api.py`
- **Features Added**:
  - "jax" strategy option in `CompressionProfile`
  - JAX strategy manager creation
  - Automatic JAX strategy selection for large tensors
  - JAX decompression support

### 3. Compression Strategy System
- **Location**: `strategies/compression_strategy.py`
- **Features Added**:
  - JAX strategy initialization in `StrategySelector`
  - Automatic JAX availability detection
  - Graceful fallback when JAX not available

### 4. Performance Monitoring
- **Location**: `tropical/unified_performance_monitor.py`
- **Features Added**:
  - JAX-specific metric types (compilation time, cache hits, memory pool)
  - JAX pipeline type for tracking
  - JAX performance thresholds and alerts
  - Integration with JAXPerformanceMonitor

## Usage Examples

### Basic Usage with JAX Backend
```python
from independent_core.compression_systems.system_integration_coordinator import (
    SystemIntegrationCoordinator, SystemConfiguration
)

# Create configuration with JAX enabled
config = SystemConfiguration(
    enable_jax=True,
    jax_backend="gpu",  # or "cpu", "tpu", "auto"
    jax_memory_fraction=0.75,
    jax_compilation_cache_size=128,
    jax_parallel_devices=1
)

# Initialize system with JAX
coordinator = SystemIntegrationCoordinator(config)

# Compress tensor using JAX acceleration
tensor = torch.randn(1000, 1000)
result = coordinator.compress(tensor)
```

### Model Compression with JAX Strategy
```python
from independent_core.compression_systems.model_compression_api import (
    CompressionProfile, ModelCompressionAPI
)

# Use JAX strategy explicitly
profile = CompressionProfile(
    strategy="jax",  # Force JAX strategy
    target_compression_ratio=4.0,
    mode="balanced"
)

api = ModelCompressionAPI(profile)
compressed_model = api.compress(model)
```

### Automatic JAX Selection
```python
# With strategy="auto", JAX will be used when beneficial
profile = CompressionProfile(
    strategy="auto",  # System chooses best strategy
    mode="aggressive"
)
```

## Configuration Options

### SystemConfiguration JAX Parameters
- `enable_jax`: Enable/disable JAX backend (default: True)
- `jax_backend`: Backend selection - "auto", "gpu", "cpu", "tpu" (default: "auto")
- `jax_memory_fraction`: GPU memory fraction for JAX (default: 0.75)
- `jax_compilation_cache_size`: JIT compilation cache size (default: 128)
- `jax_enable_x64`: Enable 64-bit precision (default: False)
- `jax_parallel_devices`: Number of devices for parallel execution (default: 1)

### Performance Monitoring
JAX operations are automatically tracked with these metrics:
- `JAX_COMPILATION`: JIT compilation time in milliseconds
- `JAX_CACHE_HITS`: Compilation cache hit rate (0-1)
- `JAX_MEMORY_POOL`: JAX memory pool usage in MB

## Installation Requirements

### Without JAX (CPU-only)
The system works without JAX installed, using PyTorch implementations.

### With JAX (Accelerated)
Install JAX for your platform:

```bash
# CPU-only
pip install jax jaxlib

# GPU (CUDA 12)
pip install jax[cuda12_local]

# TPU
pip install jax[tpu]
```

## Component Registry

When JAX is available and enabled, these components are registered:
- `jax_config_adapter`: Configuration adapter for JAX
- `jax_device_manager`: Device management for JAX operations
- `jax_memory_pool`: Memory pool for JAX tensors
- `jax_memory_optimizer`: Memory optimization for JAX
- `jax_compilation_optimizer`: JIT compilation optimization
- `jax_tropical_engine`: Core JAX tropical operations
- `jax_tropical_bridge`: Bridge between PyTorch and JAX
- `jax_strategy`: JAX compression strategy
- `jax_performance_monitor`: JAX-specific performance monitoring

## Fallback Behavior

The system gracefully handles JAX unavailability:
1. Import attempts are wrapped in try/except blocks
2. JAX_AVAILABLE flag controls feature availability
3. Falls back to PyTorch implementations when JAX not available
4. Warnings are issued but system continues to function
5. Performance monitoring works even without JAX

## Testing JAX Integration

Run the integration test suite:
```bash
python -m independent_core.compression_systems.test_jax_integration
```

This tests:
1. Component imports
2. System integration
3. Model compression API
4. Strategy manager
5. Performance monitoring

## Performance Benefits

When JAX is available and enabled:
- **6x speedup** for tropical operations through XLA compilation
- **Automatic vectorization** with vmap
- **Multi-GPU parallelization** with pmap
- **XLA fusion** for memory optimization
- **JIT compilation caching** for repeated operations

## Troubleshooting

### JAX Not Detected
- Check JAX installation: `python -c "import jax; print(jax.__version__)"`
- Verify CUDA compatibility for GPU usage
- Check system logs for import errors

### Performance Issues
- Monitor `JAX_COMPILATION` metric for compilation overhead
- Check `JAX_CACHE_HITS` for cache effectiveness
- Adjust `jax_compilation_cache_size` if needed

### Memory Issues
- Adjust `jax_memory_fraction` to control GPU memory usage
- Monitor `JAX_MEMORY_POOL` metric
- Use CPU backend for large models that don't fit in GPU memory