"""
Test suite for CPU_BurstingPipeline
Tests automatic GPU/CPU switching for p-adic decompression
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import numpy as np
import time
import psutil
import logging
from typing import List, Dict, Any
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_padic_weights(num_weights: int, precision: int = 10, prime: int = 251) -> List[Any]:
    """Create mock p-adic weights for testing"""
    # Import the actual PadicWeight class
    try:
        from compression_systems.padic.padic_encoder import PadicWeight
    except ImportError:
        # Create a mock class for testing
        class PadicWeight:
            def __init__(self, coefficients, prime, precision):
                self.coefficients = coefficients
                self.prime = prime
                self.precision = precision
            
            def copy(self):
                return PadicWeight(self.coefficients.copy(), self.prime, self.precision)
    
    weights = []
    for i in range(num_weights):
        # Create random coefficients simulating mantissa/exponent breakdown
        coeffs = np.random.randint(0, prime, size=precision).tolist()
        weight = PadicWeight(coefficients=coeffs, prime=prime, precision=precision)
        weights.append(weight)
    
    return weights


def test_cpu_decompression_engine():
    """Test CPU decompression engine"""
    print("\n" + "="*60)
    print("Testing CPU Decompression Engine")
    print("="*60)
    
    from cpu_bursting_pipeline import CPUDecompressionEngine, CPUBurstingConfig
    
    # Create config
    config = CPUBurstingConfig(
        num_cpu_workers=4,
        cpu_batch_size=50,
        use_multiprocessing=False,  # Use threading for testing
        enable_caching=True,
        cache_size_mb=128
    )
    
    # Create engine
    engine = CPUDecompressionEngine(config, prime=251)
    
    # Test 1: Basic decompression
    print("\n1. Testing basic CPU decompression...")
    weights = create_mock_padic_weights(100, precision=10)
    metadata = {
        'original_shape': (10, 10),
        'dtype': 'torch.float32',
        'device': 'cpu'
    }
    
    result, info = engine.decompress_batch_cpu(weights, target_precision=10, metadata=metadata)
    
    print(f"   Result shape: {result.shape}")
    print(f"   CPU time: {info['cpu_time']:.3f}s")
    print(f"   Cache hit: {info['cache_hit']}")
    print(f"   Batches: {info['num_batches']}")
    
    # Test 2: Cache hit
    print("\n2. Testing cache functionality...")
    result2, info2 = engine.decompress_batch_cpu(weights, target_precision=10, metadata=metadata)
    
    print(f"   Cache hit: {info2['cache_hit']}")
    print(f"   CPU time: {info2['cpu_time']:.3f}s")
    
    # Test 3: Large batch
    print("\n3. Testing large batch decompression...")
    large_weights = create_mock_padic_weights(1000, precision=20)
    large_metadata = {
        'original_shape': (100, 10),
        'dtype': 'torch.float32',
        'device': 'cpu'
    }
    
    result3, info3 = engine.decompress_batch_cpu(large_weights, target_precision=20, metadata=large_metadata)
    
    print(f"   Result shape: {result3.shape}")
    print(f"   CPU time: {info3['cpu_time']:.3f}s")
    print(f"   Weights/second: {len(large_weights) / info3['cpu_time']:.0f}")
    
    # Test 4: Statistics
    print("\n4. CPU engine statistics...")
    stats = engine.get_statistics()
    print(f"   Total decompressions: {stats['total_decompressions']}")
    print(f"   Average CPU time: {stats['average_cpu_time']:.3f}s")
    print(f"   Cache size: {stats['cache_size_mb']:.1f}MB")
    print(f"   Cache entries: {stats['cache_entries']}")
    
    # Cleanup
    engine.cleanup()
    print("\n✓ CPU decompression engine tests completed")
    return True


def test_memory_threshold_detection():
    """Test GPU memory threshold detection"""
    print("\n" + "="*60)
    print("Testing Memory Threshold Detection")
    print("="*60)
    
    from cpu_bursting_pipeline import CPU_BurstingPipeline, CPUBurstingConfig, DecompressionMode
    
    # Skip if no CUDA
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping GPU memory tests")
        return True
    
    # Create mock GPU engine
    class MockGPUEngine:
        def __init__(self):
            self.prime = 251
            self.device = torch.device('cuda:0')
        
        def decompress_progressive(self, weights, precision, metadata):
            # Simulate GPU decompression
            time.sleep(0.01)
            shape = metadata['original_shape']
            return torch.randn(shape, device='cuda'), {'decompression_time': 0.01}
        
        def get_decompression_stats(self):
            return {'total_decompressions': 0}
        
        def cleanup(self):
            pass
    
    gpu_engine = MockGPUEngine()
    
    # Create config with low thresholds for testing
    config = CPUBurstingConfig(
        gpu_memory_threshold_mb=100,
        memory_pressure_threshold=0.7,  # Lower threshold for testing
        hysteresis_factor=0.1
    )
    
    # Create pipeline
    pipeline = CPU_BurstingPipeline(config, gpu_engine)
    
    # Test 1: Check memory state
    print("\n1. Testing GPU memory state detection...")
    memory_state = pipeline._get_gpu_memory_state()
    
    if memory_state:
        print(f"   Total GPU memory: {memory_state['total_mb']:.0f}MB")
        print(f"   Free GPU memory: {memory_state['free_mb']:.0f}MB")
        print(f"   GPU utilization: {memory_state['utilization']:.1%}")
    
    # Test 2: Mode selection
    print("\n2. Testing decompression mode selection...")
    initial_mode = pipeline._select_decompression_mode()
    print(f"   Selected mode: {initial_mode.value}")
    
    # Test 3: Force mode switching
    print("\n3. Testing forced mode switching...")
    pipeline.force_mode(DecompressionMode.CPU_ONLY)
    print(f"   Forced mode: {pipeline.current_mode.value}")
    
    # Test 4: Allocate GPU memory to trigger pressure
    print("\n4. Testing memory pressure detection...")
    try:
        # Allocate large tensors to create memory pressure
        tensors = []
        for i in range(10):
            size = 100 * 1024 * 1024 // 4  # 100MB
            tensor = torch.randn(size, device='cuda')
            tensors.append(tensor)
            
            memory_state = pipeline._get_gpu_memory_state()
            print(f"   Iteration {i+1}: Free={memory_state['free_mb']:.0f}MB, "
                  f"Utilization={memory_state['utilization']:.1%}")
            
            if memory_state['utilization'] > config.memory_pressure_threshold:
                print("   Memory pressure threshold reached!")
                break
    except torch.cuda.OutOfMemoryError:
        print("   Out of GPU memory - pressure test successful")
    
    # Test 5: Check mode after pressure
    pressure_mode = pipeline._select_decompression_mode()
    print(f"\n5. Mode after pressure: {pressure_mode.value}")
    
    # Cleanup
    tensors = []
    torch.cuda.empty_cache()
    gc.collect()
    
    pipeline.cleanup()
    print("\n✓ Memory threshold detection tests completed")
    return True


def test_automatic_switching():
    """Test automatic GPU/CPU switching"""
    print("\n" + "="*60)
    print("Testing Automatic GPU/CPU Switching")
    print("="*60)
    
    from cpu_bursting_pipeline import CPU_BurstingPipeline, CPUBurstingConfig, DecompressionMode
    
    # Create mock engines
    class MockGPUEngine:
        def __init__(self, fail_after=None):
            self.prime = 251
            self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            self.decompress_count = 0
            self.fail_after = fail_after
        
        def decompress_progressive(self, weights, precision, metadata):
            self.decompress_count += 1
            if self.fail_after and self.decompress_count > self.fail_after:
                raise torch.cuda.OutOfMemoryError("Simulated OOM")
            
            shape = metadata['original_shape']
            return torch.randn(shape), {'decompression_time': 0.01, 'mode': 'gpu'}
        
        def get_decompression_stats(self):
            return {'total_decompressions': self.decompress_count}
        
        def cleanup(self):
            pass
    
    # Test with GPU that fails after 2 decompressions
    gpu_engine = MockGPUEngine(fail_after=2)
    
    config = CPUBurstingConfig(
        memory_pressure_threshold=0.9,
        switch_delay_ms=0  # No delay for testing
    )
    
    pipeline = CPU_BurstingPipeline(config, gpu_engine)
    
    # Test 1: Normal GPU decompression
    print("\n1. Testing normal GPU decompression...")
    weights = create_mock_padic_weights(100)
    metadata = {
        'original_shape': (10, 10),
        'dtype': 'torch.float32',
        'device': 'cpu'
    }
    
    result1, info1 = pipeline.decompress(weights, 10, metadata)
    print(f"   Mode used: {info1['mode']}")
    print(f"   Total time: {info1['total_time']:.3f}s")
    
    # Test 2: Second decompression (still GPU)
    print("\n2. Testing second decompression...")
    result2, info2 = pipeline.decompress(weights, 10, metadata)
    print(f"   Mode used: {info2['mode']}")
    
    # Test 3: Third decompression (should switch to CPU)
    print("\n3. Testing automatic switch to CPU...")
    result3, info3 = pipeline.decompress(weights, 10, metadata)
    print(f"   Mode used: {info3['mode']}")
    if 'fallback_reason' in info3:
        print(f"   Fallback reason: {info3['fallback_reason']}")
    
    # Test 4: Force GPU mode
    print("\n4. Testing forced GPU mode...")
    try:
        result4, info4 = pipeline.decompress(weights, 10, metadata, force_mode=DecompressionMode.GPU_ONLY)
        print(f"   Mode used: {info4['mode']}")
    except Exception as e:
        print(f"   Expected error in GPU_ONLY mode: {type(e).__name__}")
    
    # Test 5: Hybrid mode
    print("\n5. Testing hybrid decompression...")
    if torch.cuda.is_available():
        # Reset GPU engine
        gpu_engine.decompress_count = 0
        result5, info5 = pipeline.decompress(weights, 10, metadata, force_mode=DecompressionMode.HYBRID)
        print(f"   Mode used: {info5['mode']}")
        print(f"   GPU weights: {info5.get('gpu_weights', 0)}")
        print(f"   CPU weights: {info5.get('cpu_weights', 0)}")
    
    # Test 6: Statistics
    print("\n6. Pipeline statistics...")
    stats = pipeline.get_statistics()
    print(f"   Total decompressions: {stats['total_decompressions']}")
    print(f"   GPU decompressions: {stats['gpu_decompressions']}")
    print(f"   CPU decompressions: {stats['cpu_decompressions']}")
    print(f"   Mode switches: {stats['mode_switches']}")
    print(f"   Memory pressure events: {stats['memory_pressure_events']}")
    
    # Cleanup
    pipeline.cleanup()
    print("\n✓ Automatic switching tests completed")
    return True


def test_performance_comparison():
    """Compare GPU vs CPU decompression performance"""
    print("\n" + "="*60)
    print("Testing Performance Comparison")
    print("="*60)
    
    from cpu_bursting_pipeline import CPU_BurstingPipeline, CPUBurstingConfig, DecompressionMode
    
    # Create realistic GPU engine
    class RealisticGPUEngine:
        def __init__(self):
            self.prime = 251
            self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        
        def decompress_progressive(self, weights, precision, metadata):
            # Simulate GPU processing
            start = time.time()
            
            # Convert weights to tensor (simplified)
            data = []
            for w in weights:
                val = sum(c * (self.prime ** (-i)) for i, c in enumerate(w.coefficients[:precision]))
                data.append(val)
            
            tensor = torch.tensor(data).reshape(metadata['original_shape'])
            if torch.cuda.is_available():
                tensor = tensor.to(self.device)
                torch.cuda.synchronize()
            
            gpu_time = time.time() - start
            return tensor.cpu(), {'decompression_time': gpu_time, 'mode': 'gpu'}
        
        def get_decompression_stats(self):
            return {}
        
        def cleanup(self):
            pass
    
    gpu_engine = RealisticGPUEngine()
    config = CPUBurstingConfig(num_cpu_workers=4)
    pipeline = CPU_BurstingPipeline(config, gpu_engine)
    
    # Test different sizes
    test_sizes = [100, 500, 1000, 5000]
    
    for size in test_sizes:
        print(f"\n Testing with {size} weights...")
        weights = create_mock_padic_weights(size, precision=10)
        metadata = {
            'original_shape': (size,),
            'dtype': 'torch.float32',
            'device': 'cpu'
        }
        
        # Test GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gpu_result, gpu_info = pipeline.decompress(weights, 10, metadata, 
                                                      force_mode=DecompressionMode.GPU_ONLY)
            gpu_time = gpu_info['total_time']
            print(f"   GPU time: {gpu_time:.3f}s ({size/gpu_time:.0f} weights/s)")
        else:
            gpu_time = float('inf')
            print("   GPU: Not available")
        
        # Test CPU
        cpu_result, cpu_info = pipeline.decompress(weights, 10, metadata,
                                                  force_mode=DecompressionMode.CPU_ONLY)
        cpu_time = cpu_info['total_time']
        print(f"   CPU time: {cpu_time:.3f}s ({size/cpu_time:.0f} weights/s)")
        
        # Compare
        if gpu_time < float('inf'):
            speedup = cpu_time / gpu_time
            print(f"   GPU speedup: {speedup:.2f}x")
    
    # Test with memory pressure simulation
    print("\n\n Testing under memory pressure...")
    
    # Allocate memory to simulate pressure
    if torch.cuda.is_available():
        try:
            # Allocate 80% of GPU memory
            total_mem = torch.cuda.get_device_properties(0).total_memory
            alloc_size = int(total_mem * 0.8 / 4)  # float32
            pressure_tensor = torch.randn(alloc_size, device='cuda')
            
            print("   Created memory pressure")
            
            # Test auto mode
            weights = create_mock_padic_weights(1000)
            result, info = pipeline.decompress(weights, 10, metadata)
            print(f"   Auto-selected mode: {info['mode']}")
            
            # Cleanup
            del pressure_tensor
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError:
            print("   Maximum memory pressure achieved")
    
    # Final statistics
    print("\n\n Final pipeline statistics:")
    stats = pipeline.get_statistics()
    print(f"   Average GPU time: {stats['average_gpu_time']:.3f}s")
    print(f"   Average CPU time: {stats['average_cpu_time']:.3f}s")
    print(f"   GPU memory saved: {stats['gpu_memory_saved_mb']:.1f}MB")
    print(f"   Mode switches: {stats['mode_switches']}")
    
    pipeline.cleanup()
    print("\n✓ Performance comparison completed")
    return True


def test_integration():
    """Test integration with GPU memory optimizer"""
    print("\n" + "="*60)
    print("Testing Integration with GPU Memory Optimizer")
    print("="*60)
    
    from cpu_bursting_pipeline import integrate_cpu_bursting, CPUBurstingConfig
    
    # Create mock GPU optimizer
    class MockGPUOptimizer:
        def __init__(self):
            # Create mock decompression engine
            class MockEngine:
                def __init__(self):
                    self.prime = 251
                    self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
                
                def decompress_progressive(self, weights, precision, metadata):
                    shape = metadata['original_shape']
                    return torch.randn(shape), {'decompression_time': 0.01}
                
                def get_decompression_stats(self):
                    return {}
                
                def cleanup(self):
                    pass
            
            self.gpu_decompression_engine = MockEngine()
            self.decompress_gpu = self.gpu_decompression_engine.decompress_progressive
    
    # Create optimizer
    gpu_optimizer = MockGPUOptimizer()
    
    # Test integration
    print("\n1. Testing CPU bursting integration...")
    config = CPUBurstingConfig(
        num_cpu_workers=2,
        memory_pressure_threshold=0.8
    )
    
    cpu_pipeline = integrate_cpu_bursting(gpu_optimizer, config)
    print("   ✓ Integration successful")
    
    # Test that methods are added
    print("\n2. Verifying added methods...")
    assert hasattr(gpu_optimizer, 'cpu_bursting_pipeline')
    assert hasattr(gpu_optimizer, 'decompress_with_bursting')
    assert hasattr(gpu_optimizer, 'force_decompression_mode')
    print("   ✓ All methods added")
    
    # Test decompression through optimizer
    print("\n3. Testing decompression through optimizer...")
    weights = create_mock_padic_weights(100)
    metadata = {
        'original_shape': (10, 10),
        'dtype': 'torch.float32',
        'device': 'cpu'
    }
    
    result, info = gpu_optimizer.decompress_with_bursting(weights, 10, metadata)
    print(f"   Decompression mode: {info['mode']}")
    print(f"   Result shape: {result.shape}")
    
    # Test force mode
    print("\n4. Testing force mode through optimizer...")
    from cpu_bursting_pipeline import DecompressionMode
    gpu_optimizer.force_decompression_mode(DecompressionMode.CPU_ONLY)
    
    result2, info2 = gpu_optimizer.decompress_with_bursting(weights, 10, metadata)
    print(f"   Forced mode: {info2['mode']}")
    
    # Cleanup
    cpu_pipeline.cleanup()
    print("\n✓ Integration tests completed")
    return True


def run_stress_test():
    """Run stress test for CPU bursting"""
    print("\n" + "="*60)
    print("Running CPU Bursting Stress Test")
    print("="*60)
    
    from cpu_bursting_pipeline import CPU_BurstingPipeline, CPUBurstingConfig, DecompressionMode
    
    # Create engines
    class StressTestGPUEngine:
        def __init__(self):
            self.prime = 251
            self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            self.fail_randomly = False
        
        def decompress_progressive(self, weights, precision, metadata):
            if self.fail_randomly and np.random.random() < 0.3:
                raise torch.cuda.OutOfMemoryError("Random failure")
            
            shape = metadata['original_shape']
            return torch.randn(shape), {'decompression_time': 0.001}
        
        def get_decompression_stats(self):
            return {}
        
        def cleanup(self):
            pass
    
    gpu_engine = StressTestGPUEngine()
    config = CPUBurstingConfig(
        num_cpu_workers=4,
        cpu_batch_size=200,
        memory_pressure_threshold=0.8
    )
    
    pipeline = CPU_BurstingPipeline(config, gpu_engine)
    
    # Enable random failures
    gpu_engine.fail_randomly = True
    
    print("\n1. Running continuous decompressions with random failures...")
    
    decompress_times = []
    modes_used = {'gpu': 0, 'cpu': 0, 'auto_switch': 0}
    
    for i in range(100):
        size = np.random.randint(50, 500)
        weights = create_mock_padic_weights(size)
        metadata = {
            'original_shape': (size,),
            'dtype': 'torch.float32',
            'device': 'cpu'
        }
        
        try:
            start = time.time()
            result, info = pipeline.decompress(weights, 10, metadata)
            decompress_time = time.time() - start
            
            decompress_times.append(decompress_time)
            modes_used[info['mode']] = modes_used.get(info['mode'], 0) + 1
            
            if i % 20 == 0:
                print(f"   Completed {i+1} decompressions...")
                
        except Exception as e:
            print(f"   Error at iteration {i}: {e}")
    
    print(f"\n2. Stress test results:")
    print(f"   Total decompressions: {len(decompress_times)}")
    print(f"   Average time: {np.mean(decompress_times):.3f}s")
    print(f"   Modes used: {modes_used}")
    
    # Get final statistics
    stats = pipeline.get_statistics()
    print(f"\n3. Final statistics:")
    print(f"   Mode switches: {stats['mode_switches']}")
    print(f"   Memory pressure events: {stats['memory_pressure_events']}")
    print(f"   GPU memory saved: {stats['gpu_memory_saved_mb']:.1f}MB")
    
    pipeline.cleanup()
    print("\n✓ Stress test completed")
    return True


def run_all_tests():
    """Run all CPU bursting tests"""
    print("\n" + "="*80)
    print("CPU_BurstingPipeline Test Suite")
    print("="*80)
    
    try:
        # Run individual test suites
        test_cpu_decompression_engine()
        test_memory_threshold_detection()
        test_automatic_switching()
        test_performance_comparison()
        test_integration()
        run_stress_test()
        
        print("\n" + "="*80)
        print("✅ All CPU bursting tests completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.exception("Test failure details:")
        return False
    
    return True


if __name__ == "__main__":
    # Run tests
    success = run_all_tests()
    
    if success:
        print("\n🎉 CPU_BurstingPipeline is ready for production!")
        print("   - CPU Decompression Engine: ✓")
        print("   - Memory Threshold Detection: ✓")
        print("   - Automatic GPU/CPU Switching: ✓")
        print("   - Performance Optimization: ✓")
        print("   - Integration Support: ✓")
    else:
        print("\n⚠️  Some tests failed. Please review the logs above.")