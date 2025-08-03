# CPU_BurstingPipeline - Automatic GPU/CPU Decompression Switching

## Overview

The CPU_BurstingPipeline is an intelligent decompression system that automatically switches between GPU and CPU processing based on memory availability and system load. It ensures continuous operation even when GPU memory is exhausted, making it ideal for processing large models that exceed GPU capacity.

**Key Achievement**: Seamless GPU→CPU fallback with zero data loss and automatic mode selection.

## Architecture

```
CPU_BurstingPipeline
├── GPU Memory Monitor
│   ├── Real-time utilization tracking
│   ├── Threshold detection
│   └── Hysteresis prevention
├── CPUDecompressionEngine
│   ├── Multi-core optimization
│   ├── Mantissa/exponent channel processing
│   └── Result caching
└── Mode Controller
    ├── AUTO_SWITCH (default)
    ├── GPU_ONLY
    ├── CPU_ONLY
    └── HYBRID
```

## Key Components

### 1. CPU_BurstingPipeline
- **Purpose**: Main orchestrator for automatic mode switching
- **Features**:
  - Real-time GPU memory monitoring
  - Intelligent mode selection with hysteresis
  - Performance tracking and statistics
  - Seamless integration with existing GPU pipeline

### 2. CPUDecompressionEngine
- **Purpose**: Optimized CPU-based p-adic decompression
- **Features**:
  - Multi-core parallel processing
  - Broken-down p-adic channels (mantissa/exponent)
  - LRU caching for repeated decompressions
  - Configurable worker pools (threading/multiprocessing)

### 3. Memory Threshold Detection
- **Thresholds**:
  - LOW: < 50% GPU utilization
  - MODERATE: 50-75% GPU utilization
  - HIGH: 75-90% GPU utilization
  - CRITICAL: > 90% GPU utilization
- **Automatic switching at HIGH/CRITICAL levels**

## Usage

### Basic Integration

```python
from compression_systems.gpu_memory import (
    GPUMemoryOptimizer,
    integrate_cpu_bursting,
    CPUBurstingConfig
)

# Create GPU optimizer
gpu_optimizer = GPUMemoryOptimizer(config)

# Integrate CPU bursting
cpu_config = CPUBurstingConfig(
    gpu_memory_threshold_mb=100,  # Min GPU memory before switching
    memory_pressure_threshold=0.9,  # 90% utilization threshold
    num_cpu_workers=-1,  # Auto-detect CPU cores
    cpu_batch_size=100,
    enable_caching=True
)

cpu_pipeline = integrate_cpu_bursting(gpu_optimizer, cpu_config)
```

### Decompression with Auto-Switching

```python
# Automatic mode selection based on GPU memory
result, info = gpu_optimizer.decompress_with_bursting(
    padic_weights, 
    target_precision=32,
    metadata={'original_shape': (1000, 1000), 'dtype': 'torch.float32'}
)

print(f"Decompression mode: {info['mode']}")  # 'gpu', 'cpu', or 'hybrid'
print(f"Total time: {info['total_time']:.3f}s")
```

### Force Specific Mode

```python
from compression_systems.gpu_memory import DecompressionMode

# Force CPU-only mode
gpu_optimizer.force_decompression_mode(DecompressionMode.CPU_ONLY)

# Force GPU-only mode (will fail if OOM)
gpu_optimizer.force_decompression_mode(DecompressionMode.GPU_ONLY)

# Return to automatic switching
gpu_optimizer.force_decompression_mode(DecompressionMode.AUTO_SWITCH)
```

### Advanced Configuration

```python
config = CPUBurstingConfig(
    # Memory settings
    gpu_memory_threshold_mb=200,
    memory_pressure_threshold=0.85,
    
    # CPU settings
    num_cpu_workers=8,
    cpu_batch_size=200,
    use_multiprocessing=True,  # Process pool instead of threads
    cpu_affinity=[0, 1, 2, 3],  # Pin to specific CPU cores
    
    # Performance settings
    enable_profiling=True,
    enable_caching=True,
    cache_size_mb=1024,
    prefetch_factor=2,
    
    # Decompression settings
    progressive_precision=True,
    mantissa_bits=23,
    exponent_bits=8,
    
    # Switching behavior
    switch_delay_ms=50,  # Prevent rapid switching
    hysteresis_factor=0.15  # 15% hysteresis band
)
```

## Decompression Modes

### 1. AUTO_SWITCH (Default)
- Monitors GPU memory in real-time
- Automatically switches to CPU when memory pressure detected
- Returns to GPU when memory available
- Includes hysteresis to prevent rapid switching

### 2. GPU_ONLY
- Forces GPU decompression
- Fails immediately if GPU memory insufficient
- Use for latency-critical operations

### 3. CPU_ONLY
- Forces CPU decompression
- Saves all GPU memory for other operations
- Consistent performance regardless of GPU state

### 4. HYBRID
- Splits workload between GPU and CPU
- Maximizes throughput for large batches
- Automatically balances based on available resources

## Performance Characteristics

### GPU vs CPU Trade-offs

| Aspect | GPU | CPU |
|--------|-----|-----|
| Throughput | High (1000x) | Moderate |
| Latency | Low (~1ms) | Higher (~10-100ms) |
| Memory Limit | Fixed (GPU VRAM) | Flexible (System RAM) |
| Power Usage | High | Moderate |
| Scalability | Limited | Good |

### Optimization Strategies

1. **Batch Processing**: CPU processes in optimized batches
2. **Caching**: LRU cache reduces repeated decompressions
3. **Parallel Processing**: Multi-core utilization
4. **Channel Optimization**: Separate mantissa/exponent processing

## Monitoring and Statistics

```python
# Get comprehensive statistics
stats = cpu_pipeline.get_statistics()

print(f"Total decompressions: {stats['total_decompressions']}")
print(f"GPU decompressions: {stats['gpu_decompressions']}")
print(f"CPU decompressions: {stats['cpu_decompressions']}")
print(f"Mode switches: {stats['mode_switches']}")
print(f"Memory pressure events: {stats['memory_pressure_events']}")
print(f"GPU memory saved: {stats['gpu_memory_saved_mb']:.1f}MB")
print(f"Average GPU time: {stats['average_gpu_time']:.3f}s")
print(f"Average CPU time: {stats['average_cpu_time']:.3f}s")
```

## Integration with Existing Systems

### With SmartPool

```python
# SmartPool handles allocation, CPU bursting handles decompression
gpu_optimizer = GPUMemoryOptimizer({
    'enable_smart_pool': True,
    'enable_autoswap': True
})

cpu_pipeline = integrate_cpu_bursting(gpu_optimizer)
```

### With AutoSwap

```python
# AutoSwap moves tensors, CPU bursting decompresses on CPU
# They work together seamlessly
tensor_id = gpu_optimizer.register_tensor_for_autoswap(tensor)

# If tensor swapped to CPU, decompression happens there
result = gpu_optimizer.decompress_with_bursting(weights, precision, metadata)
```

## Best Practices

1. **Memory Thresholds**:
   - Set `memory_pressure_threshold` to 0.85-0.9
   - Allow headroom for GPU operations
   - Monitor actual switching behavior

2. **CPU Workers**:
   - Use `-1` for auto-detection
   - Leave 1-2 cores for system
   - Use multiprocessing for CPU-bound tasks

3. **Caching**:
   - Enable for repeated decompressions
   - Size cache based on working set
   - Monitor cache hit rates

4. **Mode Selection**:
   - Use AUTO_SWITCH for most cases
   - Force GPU_ONLY for real-time needs
   - Use CPU_ONLY during training

## Production Deployment

### Environment Variables

```bash
# CPU bursting configuration
export SARAPHIS_CPU_WORKERS=8
export SARAPHIS_GPU_THRESHOLD_MB=200
export SARAPHIS_MEMORY_PRESSURE=0.9
export SARAPHIS_ENABLE_CPU_CACHE=1
```

### Monitoring

```python
# Real-time monitoring
while True:
    stats = cpu_pipeline.get_statistics()
    memory_state = stats['gpu_memory_state']
    
    print(f"Mode: {stats['current_mode']}")
    print(f"GPU Free: {memory_state['free_mb']:.0f}MB")
    print(f"CPU Usage: {stats['avg_cpu_utilization']:.1f}%")
    
    time.sleep(1)
```

### Error Handling

The system follows **NO FALLBACKS - HARD FAILURES ONLY**:
- Invalid configurations raise immediate errors
- OOM errors in GPU_ONLY mode fail immediately
- CPU decompression failures are not retried

## Testing

Run the comprehensive test suite:

```bash
python test_cpu_bursting.py
```

Tests include:
- CPU decompression engine validation
- Memory threshold detection
- Automatic mode switching
- Performance benchmarks
- Integration tests
- Stress testing

## Technical Details

### P-adic Channel Breakdown

The CPU engine processes broken-down p-adic components:
- **Mantissa Channel**: Fractional precision bits
- **Exponent Channel**: Scale/magnitude bits
- Optimized for IEEE 754 float reconstruction

### Memory Management

- GPU memory monitored every 100ms
- Hysteresis prevents switching within 10% band
- CPU results cached with LRU eviction
- Zero-copy transfers where possible

### Thread Safety

- All operations are thread-safe
- Monitoring runs in background thread
- Statistics updated atomically
- Cache access synchronized

## Future Enhancements

- [ ] NUMA-aware CPU processing
- [ ] GPU memory prediction
- [ ] Adaptive batch sizing
- [ ] Distributed CPU processing
- [ ] Hardware acceleration (AVX-512)

## License

Part of the Saraphis Independent Core compression system.
**NO FALLBACKS - HARD FAILURES ONLY**