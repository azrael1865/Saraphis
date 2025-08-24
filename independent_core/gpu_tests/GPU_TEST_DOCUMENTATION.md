# Saraphis GPU Test Suite Documentation

## Overview
Comprehensive GPU testing suite for Saraphis with **HARD FAILURE MODE** - no silent errors or fallbacks. All tests are designed to fail loudly and clearly indicate the exact problem.

## Requirements

### Hardware
- NVIDIA GPU with CUDA support
- Minimum Compute Capability: 3.5
- Recommended: 8GB+ VRAM
- Tested on: RTX 3090, A100, V100

### Software
- Python 3.7+
- PyTorch 2.0+ with CUDA support
- CUDA 11.0+ (or 12.0+ for newer GPUs)
- cuDNN 8.0+

## Test Suite Structure

```
gpu_tests/
├── test_gpu_device_management.py     # Priority 1: Device detection & switching
├── test_gpu_memory_management.py     # Priority 1: Memory allocation & cleanup
├── test_saraphis_gpu_integration.py  # Priority 2: Component integration
├── run_all_gpu_tests.py             # Master test runner
├── GPU_TEST_DOCUMENTATION.md        # This file
├── gpu_test_results.log            # Detailed test log (generated)
└── gpu_test_report.txt             # Summary report (generated)
```

## Running the Tests

### Full Test Suite
```bash
cd independent_core/gpu_tests
python run_all_gpu_tests.py
```

### Individual Test Modules
```bash
# Test device management only
python test_gpu_device_management.py

# Test memory management only
python test_gpu_memory_management.py

# Test Saraphis integration only
python test_saraphis_gpu_integration.py
```

## Test Categories

### Priority 1: Critical Tests (MUST PASS)

#### Device Management (25 tests)
- **CUDA Availability**: Verifies CUDA is available and accessible
- **Device Properties**: Validates GPU properties and compute capability
- **Device Switching**: Tests CPU↔GPU tensor transfers
- **Context Management**: Tests CUDA context and stream management
- **Multi-GPU Support**: Tests multiple GPU detection and usage

#### Memory Management (45 tests)
- **Allocation Tracking**: Verifies accurate memory allocation tracking
- **Large Allocations**: Tests handling of large memory requests
- **OOM Handling**: Ensures proper out-of-memory error handling
- **Memory Cleanup**: Verifies no memory leaks
- **Cache Management**: Tests PyTorch GPU cache behavior
- **Fragmentation**: Tests memory fragmentation handling

### Priority 2: Important Tests

#### Saraphis GPU Integration (40 tests)
- **P-adic Compression**: GPU-accelerated compression/decompression
- **Tropical Algebra**: GPU matrix operations
- **Brain Core**: GPU-accelerated predictions
- **Memory Optimizer**: Automatic memory optimization

## Hard Failure Behaviors

All tests implement **HARD FAILURES**:

1. **No Silent Errors**: Every error raises an exception
2. **No Fallbacks**: No automatic CPU fallback in tests
3. **Fatal Assertions**: Clear FATAL messages for all failures
4. **Memory Leaks**: Any memory leak > 1MB fails the test
5. **Data Corruption**: Any data corruption fails immediately

## Expected Outputs

### Successful Run
```
================================================================================
SARAPHIS GPU TEST SUITE
HARD FAILURE MODE - NO SILENT ERRORS
================================================================================
CHECKING GPU AVAILABILITY
✓ CUDA Available: 1 device(s) found

GPU 0: NVIDIA GeForce RTX 3090
  Compute Capability: 8.6
  Memory: 24.00 GB
  Multiprocessors: 82

RUNNING: Device Management
✅ Device Management PASSED (12.34s)

RUNNING: Memory Management
✅ Memory Management PASSED (23.45s)

RUNNING: Saraphis GPU Integration
✅ Saraphis GPU Integration PASSED (34.56s)

================================================================================
GPU TEST SUITE FINAL REPORT
================================================================================
Test Execution Time: 70.35 seconds
Modules Tested: 3
Passed: 3
Failed: 0

🎉 ALL GPU TESTS PASSED!
Saraphis GPU components are fully functional
================================================================================
```

### Failed Run
```
❌ Device Management FAILED
  FATAL: Compute capability 3.0 < 3.5 minimum required
  FATAL: Memory leak detected: 125.34 MB
  
Review the errors above and fix the issues
```

## Common Issues and Solutions

### Issue: CUDA Not Available
```
FATAL: CUDA NOT AVAILABLE - Cannot run GPU tests without NVIDIA GPU
```
**Solution**: Ensure you have an NVIDIA GPU and PyTorch is installed with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Out of Memory
```
FATAL: Cannot allocate 100MB tensor: CUDA out of memory
```
**Solution**: 
- Close other GPU applications
- Reduce test tensor sizes
- Clear GPU cache: `torch.cuda.empty_cache()`

### Issue: Import Errors
```
FATAL: Cannot import P-adic compression: No module named 'compression_systems'
```
**Solution**: Ensure you're running from the correct directory and all dependencies are installed

### Issue: Memory Leaks
```
FATAL: Memory leak detected: 50.00 MB
```
**Solution**: Check for unreleased tensors, missing `del` statements, or circular references

## Performance Benchmarks

Expected performance on different GPUs:

| GPU | Device Tests | Memory Tests | Integration Tests | Total Time |
|-----|-------------|--------------|-------------------|------------|
| RTX 3090 | ~10s | ~20s | ~30s | ~60s |
| A100 | ~8s | ~15s | ~25s | ~48s |
| V100 | ~12s | ~25s | ~35s | ~72s |
| RTX 2080 Ti | ~15s | ~30s | ~40s | ~85s |

## Debugging Failed Tests

### Enable Debug Logging
```python
# In test files, change logging level
logging.basicConfig(level=logging.DEBUG)
```

### Run with Memory Profiling
```bash
python -m torch.utils.bottleneck test_gpu_memory_management.py
```

### Check GPU Utilization
```bash
# In another terminal while tests run
nvidia-smi -l 1
```

### Analyze Memory Leaks
```python
# Add to test code
import tracemalloc
tracemalloc.start()
# ... test code ...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
```

## Test Coverage Summary

- **103 GPU-enabled modules** in Saraphis
- **473 minimum tests** required for comprehensive coverage
- **3 test modules** currently implemented
- **~110 individual test cases** in current suite

## Future Enhancements

1. **Performance Benchmarks**: Add speed comparison tests
2. **Multi-GPU Tests**: Expand distributed GPU testing
3. **Mixed Precision**: Add FP16/BF16 tests
4. **JIT Compilation**: Test torch.compile optimizations
5. **Profiling Integration**: Add automatic profiling on failures

## Contributing

When adding new GPU tests:

1. **Always use HARD FAILURES**: No silent errors or skips
2. **Clear Error Messages**: Use `FATAL:` prefix for critical errors
3. **Memory Cleanup**: Always cleanup GPU memory in tearDown
4. **Document Expected Behavior**: Clear docstrings for each test
5. **Test Both Success and Failure**: Verify error handling works

## Support

For issues with GPU tests:
1. Check this documentation first
2. Review gpu_test_results.log for detailed errors
3. Ensure GPU drivers and CUDA are up to date
4. Report issues with full error logs and GPU info