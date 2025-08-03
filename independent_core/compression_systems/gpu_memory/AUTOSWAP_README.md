# AutoSwap Priority-Based Memory Swapping System

## Overview

AutoSwap is an intelligent GPU memory swapping system that uses Duration of Absence (DOA) scoring and priority-based strategies to optimize GPU memory utilization. It seamlessly integrates with the existing GPUMemoryOptimizer to provide automatic memory management under pressure.

## Key Components

### 1. DOAScorer (Duration of Absence)
- **Purpose**: Intelligent memory prioritization based on access patterns
- **Features**:
  - Duration of Absence calculation
  - Access pattern detection (sequential, random, strided, temporal, spatial)
  - Temporal and spatial locality analysis
  - Priority-based scoring with configurable weights
  - Memory block pinning/unpinning

### 2. PrioritySwapper
- **Purpose**: Execute memory swaps between GPU, CPU, and disk
- **Strategies**:
  - IMMEDIATE: Synchronous immediate swap
  - ASYNC: Asynchronous background swap
  - BATCH: Batch multiple swaps for efficiency
  - PREDICTIVE: Predictive pre-swapping based on patterns
- **Features**:
  - Pinned memory pool for fast transfers
  - Optional compression for CPU/disk storage
  - Concurrent swap operations
  - Comprehensive swap tracking

### 3. AutoSwapManager
- **Purpose**: Orchestrate intelligent memory swapping
- **Policies**:
  - AGGRESSIVE: Swap early and often
  - BALANCED: Balance performance and memory
  - CONSERVATIVE: Swap only when necessary
  - ADAPTIVE: Learn and adapt based on workload
- **Features**:
  - Automatic memory pressure monitoring
  - Policy-based swap decisions
  - Real-time optimization
  - Performance tracking and adaptation

## Usage

### Basic Integration

```python
from compression_systems.gpu_memory import GPUMemoryOptimizer

# Create GPU optimizer with AutoSwap
gpu_optimizer = GPUMemoryOptimizer({
    'enable_autoswap': True,
    'swap_policy': 'balanced',  # 'aggressive', 'conservative', 'adaptive'
    'autoswap_monitoring': True,
    'swap_threshold_low': 0.5,
    'swap_threshold_moderate': 0.75,
    'swap_threshold_high': 0.9,
    'swap_threshold_critical': 0.95
})
```

### Register Tensors for AutoSwap

```python
# Register tensor with priority
tensor = torch.randn(10000000, device='cuda:0')  # ~40MB
tensor_id = gpu_optimizer.register_tensor_for_autoswap(
    tensor,
    priority='medium'  # 'critical', 'high', 'medium', 'low', 'idle'
)

# Record access for DOA tracking
gpu_optimizer.record_tensor_access(tensor_id)
```

### Handle Memory Pressure

```python
# Automatic handling
required_bytes = 100 * 1024 * 1024  # 100MB
success = gpu_optimizer.handle_memory_pressure(required_bytes, device_id=0)

# Manual swap decision
decision = gpu_optimizer.autoswap_manager.make_swap_decision(required_bytes, device_id=0)
operations = gpu_optimizer.autoswap_manager.execute_swap_decision(decision)

# Swap tensor back to GPU
tensor = gpu_optimizer.swap_in_tensor(tensor_id, device_id=0)
```

### Advanced Configuration

```python
from compression_systems.gpu_memory import AutoSwapConfig, SwapPolicy

config = AutoSwapConfig(
    swap_policy=SwapPolicy.BALANCED,
    memory_pressure_thresholds={
        'low': 0.5,
        'moderate': 0.75,
        'high': 0.9,
        'critical': 0.95
    },
    min_swap_size_mb=1.0,
    max_swap_size_mb=1024.0,
    swap_ahead_factor=1.2,  # Swap 20% more than needed
    enable_predictive_swapping=True,
    enable_batch_swapping=True,
    monitoring_interval_seconds=1.0,
    auto_adjust_policy=True
)
```

## Architecture

```
GPUMemoryOptimizer
    ├── SmartPool (Task 1.1)
    │   └── Advanced memory allocation
    └── AutoSwap (Task 1.2)
        ├── DOAScorer
        │   ├── Access Pattern Detection
        │   ├── Priority Scoring
        │   └── Swap Candidate Selection
        ├── PrioritySwapper
        │   ├── GPU ↔ CPU Swapping
        │   ├── CPU ↔ Disk Swapping
        │   ├── Compression Support
        │   └── Batch Operations
        └── AutoSwapManager
            ├── Policy Management
            ├── Memory Monitoring
            ├── Swap Orchestration
            └── Performance Adaptation
```

## Memory Pressure Levels

1. **LOW** (< 50% utilization)
   - No swapping needed
   - Normal operation

2. **MODERATE** (50-75% utilization)
   - Consider swapping IDLE blocks
   - Preventive measures

3. **HIGH** (75-90% utilization)
   - Swap LOW and IDLE priority blocks
   - Active memory management

4. **CRITICAL** (> 90% utilization)
   - Aggressive swapping
   - Include MEDIUM priority blocks

## Swap Priorities

1. **CRITICAL**: Never swap (essential data)
2. **HIGH**: Swap only under extreme pressure
3. **MEDIUM**: Normal swap candidate
4. **LOW**: Preferred swap candidate
5. **IDLE**: Immediate swap candidate

## Access Patterns

AutoSwap detects and optimizes for various access patterns:

- **SEQUENTIAL**: Consecutive memory accesses
- **RANDOM**: Unpredictable access pattern
- **STRIDED**: Regular interval accesses
- **TEMPORAL**: Time-based patterns
- **SPATIAL**: Location-based clustering

## Performance Metrics

### DOA Scoring Factors
- **Recency** (30%): Time since last access
- **Frequency** (30%): Access rate
- **Pattern** (20%): Access pattern type
- **Size** (20%): Memory block size

### Swap Performance
- Average swap time tracking
- Compression savings
- Success/failure rates
- Memory distribution monitoring

## Testing

Run the comprehensive test suite:

```bash
python test_autoswap.py
```

Tests include:
- DOAScorer pattern detection and scoring
- PrioritySwapper GPU-CPU-Disk operations
- AutoSwapManager policy decisions
- Integration with GPUMemoryOptimizer
- Stress testing under memory pressure

## Best Practices

1. **Priority Assignment**:
   - Use CRITICAL for model weights
   - Use HIGH for active computations
   - Use MEDIUM for intermediate results
   - Use LOW for cached data
   - Use IDLE for precomputed values

2. **Access Recording**:
   - Record accesses for accurate DOA scoring
   - Update access patterns during computation
   - Use offset/length for partial access tracking

3. **Policy Selection**:
   - AGGRESSIVE for memory-constrained systems
   - BALANCED for general workloads
   - CONSERVATIVE for latency-sensitive tasks
   - ADAPTIVE for dynamic workloads

4. **Monitoring**:
   - Enable autoswap_monitoring for automatic management
   - Check statistics regularly
   - Adjust thresholds based on workload

## Production Considerations

1. **No Fallbacks**: System fails hard on errors
2. **Thread Safety**: All operations are thread-safe
3. **Performance**: Optimized for minimal overhead
4. **Scalability**: Supports multiple GPUs
5. **Integration**: Seamless with existing code

## Future Enhancements

- Machine learning-based policy selection
- Distributed swapping across nodes
- Advanced compression algorithms
- Prefetching based on access patterns
- Integration with PyTorch memory hooks

## License

Part of the Saraphis Independent Core compression system.
NO FALLBACKS - HARD FAILURES ONLY