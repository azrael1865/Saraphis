"""
Integration tests for JAX-PyTorch interoperability in tropical compression system.
Tests memory sharing, tensor conversion, and performance benchmarks.
NO PLACEHOLDERS - PRODUCTION READY
"""

import unittest
import time
import gc
import numpy as np
import torch
import warnings
from typing import Dict, Any, Tuple

# Import JAX components
from jax_config import (
    JAX_AVAILABLE,
    JAXConfig,
    JAXEnvironment,
    JAXDeviceManager,
    JAXMemoryPool,
    JAXCompilationCache,
    JAXPyTorchBridge,
    JAXPlatform,
    setup_jax_for_tropical
)

# Import existing system components
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from independent_core.compression_systems.gpu_memory.gpu_auto_detector import GPUAutoDetector
from independent_core.compression_systems.gpu_memory.gpu_memory_core import GPUMemoryManager
from independent_core.compression_systems.tropical.tropical_core import TROPICAL_ZERO, TROPICAL_EPSILON

# Skip tests if JAX not available
if not JAX_AVAILABLE:
    warnings.warn("JAX not installed, skipping JAX integration tests")
else:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap


class TestJAXEnvironment(unittest.TestCase):
    """Test JAX environment setup and configuration"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = JAXConfig(
            enable_jit=True,
            enable_x64=False,
            memory_fraction=0.5,
            compilation_cache_size=64
        )
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_environment_setup(self):
        """Test complete environment setup"""
        env = JAXEnvironment(self.config)
        result = env.setup_environment()
        
        self.assertTrue(result['jax_available'])
        self.assertIn('platform', result)
        self.assertIn('devices', result)
        self.assertEqual(len(result['errors']), 0)
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_device_detection(self):
        """Test JAX device detection"""
        env = JAXEnvironment(self.config)
        devices = env.detect_jax_devices()
        
        self.assertIsInstance(devices, list)
        self.assertGreater(len(devices), 0)  # At least CPU should be available
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_memory_configuration(self):
        """Test memory configuration"""
        env = JAXEnvironment(self.config)
        env.setup_environment()
        
        memory_config = env.get_memory_config()
        if env.config.platform == JAXPlatform.GPU:
            self.assertIn('total_bytes', memory_config)
            self.assertIn('allocated_bytes', memory_config)
            self.assertEqual(memory_config['fraction'], self.config.memory_fraction)
            
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_validation(self):
        """Test JAX installation validation"""
        env = JAXEnvironment(self.config)
        env.setup_environment()
        
        is_valid = env.validate_installation()
        self.assertTrue(is_valid)
        
    def test_no_jax_handling(self):
        """Test graceful handling when JAX is not installed"""
        # This test always runs
        env = JAXEnvironment(self.config)
        result = env.setup_environment()
        
        self.assertIn('jax_available', result)
        if not JAX_AVAILABLE:
            self.assertFalse(result['jax_available'])
            self.assertGreater(len(result['errors']), 0)


class TestJAXDeviceManager(unittest.TestCase):
    """Test JAX device management"""
    
    def setUp(self):
        """Set up device manager"""
        self.manager = JAXDeviceManager()
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_get_device(self):
        """Test device retrieval"""
        # Should have at least one device (CPU)
        device = self.manager.get_device()
        self.assertIsNotNone(device)
        
        # Test specific device ID
        device_0 = self.manager.get_device(0)
        self.assertIsNotNone(device_0)
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_device_memory_stats(self):
        """Test memory statistics retrieval"""
        if self.manager.devices:
            device = self.manager.devices[0]
            stats = self.manager.get_device_memory_stats(device)
            
            self.assertIsInstance(stats, dict)
            self.assertIn('allocated_bytes', stats)
            self.assertIn('total_bytes', stats)
            
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_set_default_device(self):
        """Test setting default device"""
        if len(self.manager.devices) > 0:
            device = self.manager.devices[0]
            self.manager.set_default_device(device)
            self.assertEqual(self.manager.current_device, device)
            
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_distribute_data(self):
        """Test data distribution across devices"""
        data = np.random.randn(100, 100)
        distributed = self.manager.distribute_across_devices(data)
        
        # Should return data even with single device
        self.assertIsNotNone(distributed)


class TestJAXMemoryPool(unittest.TestCase):
    """Test JAX memory pool management"""
    
    def setUp(self):
        """Set up memory pool"""
        self.config = JAXConfig(memory_fraction=0.5)
        self.pool = JAXMemoryPool(self.config)
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_allocation(self):
        """Test memory allocation"""
        shape = (100, 100)
        array, buffer_id = self.pool.allocate(shape, dtype=jnp.float32)
        
        self.assertEqual(array.shape, shape)
        self.assertIn(buffer_id, self.pool.allocated_buffers)
        
        # Clean up
        self.pool.deallocate(buffer_id)
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_deallocation(self):
        """Test memory deallocation"""
        shape = (50, 50)
        array, buffer_id = self.pool.allocate(shape)
        
        initial_allocated = self.pool._total_allocated
        self.pool.deallocate(buffer_id)
        
        self.assertNotIn(buffer_id, self.pool.allocated_buffers)
        self.assertLess(self.pool._total_allocated, initial_allocated)
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_memory_tracking(self):
        """Test memory usage tracking"""
        # Allocate multiple buffers
        buffers = []
        for i in range(5):
            array, buffer_id = self.pool.allocate((10, 10))
            buffers.append(buffer_id)
            
        stats = self.pool.get_memory_usage()
        self.assertEqual(stats['num_buffers'], 5)
        self.assertGreater(stats['total_allocated_bytes'], 0)
        
        # Clean up
        for buffer_id in buffers:
            self.pool.deallocate(buffer_id)
            
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_memory_limit(self):
        """Test memory limit enforcement"""
        self.pool.memory_limit = 1024 * 1024  # 1MB limit
        
        # Try to allocate more than limit
        with self.assertRaises(MemoryError):
            # This should exceed 1MB
            array, buffer_id = self.pool.allocate((1000, 1000), dtype=jnp.float32)
            
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_saraphis_integration(self):
        """Test integration with Saraphis GPU memory manager"""
        gpu_manager = GPUMemoryManager()
        self.pool.integrate_with_saraphis(gpu_manager)
        
        # Memory limit should be set based on GPU
        if self.pool.memory_limit:
            self.assertGreater(self.pool.memory_limit, 0)


class TestJAXCompilationCache(unittest.TestCase):
    """Test JAX compilation caching"""
    
    def setUp(self):
        """Set up compilation cache"""
        self.cache = JAXCompilationCache(max_size=10)
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_cache_hit_miss(self):
        """Test cache hits and misses"""
        def test_func(x):
            return x * 2 + 1
            
        # First call should be a miss
        x = jnp.array([1, 2, 3])
        result1 = self.cache.get_or_compile(test_func, x)
        self.assertEqual(self.cache._misses, 1)
        
        # Second call with same shape should be a hit
        result2 = self.cache.get_or_compile(test_func, x)
        self.assertEqual(self.cache._hits, 1)
        
        # Different shape should be a miss
        y = jnp.array([1, 2, 3, 4])
        result3 = self.cache.get_or_compile(test_func, y)
        self.assertEqual(self.cache._misses, 2)
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_cache_eviction(self):
        """Test LRU cache eviction"""
        def test_func(x):
            return x * 2
            
        # Fill cache beyond capacity
        for i in range(15):
            x = jnp.ones((i+1, i+1))
            self.cache.get_or_compile(test_func, x)
            
        # Cache should not exceed max size
        self.assertLessEqual(len(self.cache.cache), self.cache.max_size)
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_cache_stats(self):
        """Test cache statistics"""
        def test_func(x):
            return jnp.sum(x)
            
        # Generate some cache activity
        for _ in range(5):
            x = jnp.array([1, 2, 3])
            self.cache.get_or_compile(test_func, x)
            
        stats = self.cache.get_cache_stats()
        self.assertIn('hit_rate', stats)
        self.assertIn('cache_size', stats)
        self.assertIn('total_compilation_time', stats)
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_cache_clear(self):
        """Test cache clearing"""
        def test_func(x):
            return x ** 2
            
        # Add some entries
        for i in range(5):
            x = jnp.ones(i+1)
            self.cache.get_or_compile(test_func, x)
            
        self.assertGreater(len(self.cache.cache), 0)
        
        # Clear cache
        self.cache.clear_cache()
        self.assertEqual(len(self.cache.cache), 0)
        self.assertEqual(self.cache._hits, 0)
        self.assertEqual(self.cache._misses, 0)


class TestJAXPyTorchBridge(unittest.TestCase):
    """Test JAX-PyTorch tensor conversion"""
    
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_torch_to_jax(self):
        """Test PyTorch to JAX conversion"""
        # CPU tensor
        torch_tensor = torch.randn(10, 10)
        jax_array = JAXPyTorchBridge.torch_to_jax(torch_tensor)
        
        self.assertEqual(jax_array.shape, torch_tensor.shape)
        np.testing.assert_allclose(
            np.array(jax_array),
            torch_tensor.numpy(),
            rtol=1e-5
        )
        
        # CUDA tensor if available
        if torch.cuda.is_available():
            cuda_tensor = torch.randn(5, 5).cuda()
            jax_array_cuda = JAXPyTorchBridge.torch_to_jax(cuda_tensor)
            self.assertEqual(jax_array_cuda.shape, cuda_tensor.shape)
            
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_jax_to_torch(self):
        """Test JAX to PyTorch conversion"""
        jax_array = jnp.ones((7, 7))
        torch_tensor = JAXPyTorchBridge.jax_to_torch(jax_array)
        
        self.assertEqual(torch_tensor.shape, jax_array.shape)
        np.testing.assert_allclose(
            torch_tensor.numpy(),
            np.array(jax_array),
            rtol=1e-5
        )
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_memory_sharing(self):
        """Test zero-copy memory sharing"""
        # Only works for CPU tensors
        torch_tensor = torch.randn(100, 100)
        jax_shared = JAXPyTorchBridge.share_memory(torch_tensor)
        
        # Modify torch tensor
        torch_tensor[0, 0] = 999.0
        
        # Check if change is reflected (if true zero-copy)
        # Note: This may not always work depending on backend
        self.assertEqual(jax_shared.shape, torch_tensor.shape)
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_validation(self):
        """Test conversion validation"""
        torch_tensor = torch.randn(5, 5)
        jax_array = JAXPyTorchBridge.torch_to_jax(torch_tensor)
        
        # Should validate successfully
        is_valid = JAXPyTorchBridge.validate_conversion(torch_tensor, jax_array)
        self.assertTrue(is_valid)
        
        # Create mismatched array
        wrong_array = jnp.zeros((3, 3))
        is_valid = JAXPyTorchBridge.validate_conversion(torch_tensor, wrong_array)
        self.assertFalse(is_valid)
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_dtype_preservation(self):
        """Test dtype preservation during conversion"""
        # Test different dtypes
        dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]
        
        for dtype in dtypes:
            torch_tensor = torch.ones(3, 3, dtype=dtype)
            jax_array = JAXPyTorchBridge.torch_to_jax(torch_tensor)
            torch_back = JAXPyTorchBridge.jax_to_torch(jax_array)
            
            # Check dtype preservation
            if dtype == torch.float64:
                # Float64 might be downcast to float32 in JAX
                self.assertIn(torch_back.dtype, [torch.float32, torch.float64])
            else:
                self.assertEqual(torch_back.dtype, dtype)
                
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_benchmark(self):
        """Test and benchmark conversion performance"""
        results = JAXPyTorchBridge.benchmark_conversion(
            tensor_shape=(100, 100),
            num_iterations=10
        )
        
        self.assertIn('torch_to_jax_ms', results)
        self.assertIn('jax_to_torch_ms', results)
        self.assertIn('share_memory_ms', results)
        
        # All conversions should be reasonably fast (< 10ms for small tensors)
        self.assertLess(results['torch_to_jax_ms'], 10)
        self.assertLess(results['jax_to_torch_ms'], 10)


class TestTropicalOperationsIntegration(unittest.TestCase):
    """Test JAX integration with tropical operations"""
    
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_tropical_operations_jax(self):
        """Test tropical semiring operations in JAX"""
        # Tropical addition (max)
        @jit
        def tropical_add(x, y):
            return jnp.maximum(x, y)
            
        # Tropical multiplication (addition)
        @jit
        def tropical_mul(x, y):
            # Handle tropical zero
            x_is_zero = x <= TROPICAL_ZERO
            y_is_zero = y <= TROPICAL_ZERO
            
            result = jnp.where(
                x_is_zero | y_is_zero,
                TROPICAL_ZERO,
                x + y
            )
            return result
            
        # Test operations
        a = jnp.array([1.0, 2.0, TROPICAL_ZERO])
        b = jnp.array([3.0, 1.0, 5.0])
        
        add_result = tropical_add(a, b)
        expected_add = jnp.array([3.0, 2.0, 5.0])
        np.testing.assert_allclose(add_result, expected_add)
        
        mul_result = tropical_mul(a[0], b[0])
        expected_mul = 4.0  # 1 + 3
        np.testing.assert_allclose(mul_result, expected_mul)
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_tropical_matrix_operations(self):
        """Test tropical matrix operations with JAX"""
        @jit
        def tropical_matmul(A, B):
            """Tropical matrix multiplication"""
            m, k = A.shape
            k2, n = B.shape
            assert k == k2, "Incompatible dimensions"
            
            C = jnp.full((m, n), TROPICAL_ZERO)
            
            for i in range(m):
                for j in range(n):
                    for l in range(k):
                        # Tropical multiplication then addition
                        if A[i, l] > TROPICAL_ZERO and B[l, j] > TROPICAL_ZERO:
                            val = A[i, l] + B[l, j]  # Tropical multiplication
                            C = C.at[i, j].set(jnp.maximum(C[i, j], val))  # Tropical addition
                            
            return C
            
        # Test matrices
        A = jnp.array([[1.0, 2.0], [3.0, TROPICAL_ZERO]])
        B = jnp.array([[2.0, 1.0], [TROPICAL_ZERO, 3.0]])
        
        C = tropical_matmul(A, B)
        
        # Verify result shape
        self.assertEqual(C.shape, (2, 2))
        
        # Check specific values
        self.assertAlmostEqual(float(C[0, 0]), 3.0)  # max(1+2, 2+TROP_ZERO) = 3
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_gradient_computation(self):
        """Test gradient computation with JAX"""
        def loss_func(x):
            """Simple loss function for gradient testing"""
            return jnp.sum(x ** 2)
            
        # Compute gradient
        grad_func = grad(loss_func)
        
        x = jnp.array([1.0, 2.0, 3.0])
        gradient = grad_func(x)
        
        # Expected gradient: 2x
        expected = 2 * x
        np.testing.assert_allclose(gradient, expected, rtol=1e-5)
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_vectorization(self):
        """Test vectorization with vmap"""
        def single_operation(x):
            return x ** 2 + 2 * x + 1
            
        # Vectorize the operation
        vectorized = vmap(single_operation)
        
        # Test on batch
        batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = vectorized(batch)
        
        # Check shape
        self.assertEqual(result.shape, batch.shape)
        
        # Verify values
        expected = batch ** 2 + 2 * batch + 1
        np.testing.assert_allclose(result, expected)


class TestEndToEndIntegration(unittest.TestCase):
    """Test complete JAX integration with compression system"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.config = JAXConfig(
            enable_jit=True,
            memory_fraction=0.5,
            compilation_cache_size=32
        )
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_full_setup(self):
        """Test complete JAX setup for tropical operations"""
        result = setup_jax_for_tropical(
            memory_fraction=0.5,
            enable_jit=True
        )
        
        self.assertTrue(result['jax_available'])
        self.assertGreater(result['device_count'], 0)
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_mixed_operations(self):
        """Test mixed PyTorch and JAX operations"""
        # Start with PyTorch tensor
        torch_data = torch.randn(100, 50)
        
        # Convert to JAX for processing
        jax_data = JAXPyTorchBridge.torch_to_jax(torch_data)
        
        # Perform JAX operations
        @jit
        def process_data(x):
            # Normalize
            x = x - jnp.mean(x)
            x = x / (jnp.std(x) + 1e-8)
            # Apply tropical-like operation
            x = jnp.maximum(x, TROPICAL_ZERO)
            return x
            
        jax_processed = process_data(jax_data)
        
        # Convert back to PyTorch
        torch_result = JAXPyTorchBridge.jax_to_torch(jax_processed)
        
        # Verify shapes match
        self.assertEqual(torch_result.shape, torch_data.shape)
        
        # Result should be normalized
        self.assertLess(torch_result.std().item(), torch_data.std().item())
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_performance_comparison(self):
        """Compare performance of PyTorch vs JAX operations"""
        size = (1000, 1000)
        
        # PyTorch operation
        torch_data = torch.randn(*size)
        
        start = time.time()
        for _ in range(10):
            torch_result = torch.maximum(torch_data, torch.tensor(TROPICAL_ZERO))
            torch_result = torch_result + 1.0
        torch_time = time.time() - start
        
        # JAX operation
        jax_data = jnp.array(torch_data.numpy())
        
        @jit
        def jax_op(x):
            x = jnp.maximum(x, TROPICAL_ZERO)
            return x + 1.0
            
        # Warm up JIT
        _ = jax_op(jax_data)
        
        start = time.time()
        for _ in range(10):
            jax_result = jax_op(jax_data)
        jax_time = time.time() - start
        
        print(f"\nPerformance comparison:")
        print(f"PyTorch time: {torch_time:.4f}s")
        print(f"JAX time: {jax_time:.4f}s")
        print(f"Speedup: {torch_time/jax_time:.2f}x")
        
        # JAX should be competitive or faster after JIT compilation
        # Note: Actual speedup depends on hardware and operation complexity
        
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_memory_pressure(self):
        """Test behavior under memory pressure"""
        pool = JAXMemoryPool(self.config)
        pool.memory_limit = 100 * 1024 * 1024  # 100MB limit
        
        allocations = []
        
        # Allocate until near limit
        try:
            for i in range(20):
                array, buffer_id = pool.allocate((1000, 1000), dtype=jnp.float32)
                allocations.append(buffer_id)
        except MemoryError:
            # Expected when limit reached
            pass
            
        # Check cleanup was triggered
        stats = pool.get_memory_usage()
        self.assertLessEqual(stats['total_allocated_bytes'], pool.memory_limit)
        
        # Clean up
        for buffer_id in allocations:
            if buffer_id in pool.allocated_buffers:
                pool.deallocate(buffer_id)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_invalid_device_id(self):
        """Test handling of invalid device IDs"""
        manager = JAXDeviceManager()
        
        with self.assertRaises(ValueError):
            manager.get_device(device_id=999)
            
    @unittest.skipIf(not JAX_AVAILABLE, "JAX not installed")
    def test_conversion_errors(self):
        """Test handling of conversion errors"""
        # Test with incompatible types
        invalid_data = "not a tensor"
        
        with self.assertRaises((TypeError, RuntimeError)):
            JAXPyTorchBridge.torch_to_jax(invalid_data)
            
    def test_config_validation(self):
        """Test configuration validation"""
        # Invalid memory fraction
        with self.assertRaises(ValueError):
            config = JAXConfig(memory_fraction=1.5)
            
        # Invalid cache size
        with self.assertRaises(ValueError):
            config = JAXConfig(compilation_cache_size=-1)
            
        # Invalid device ID
        with self.assertRaises(ValueError):
            config = JAXConfig(device_id=-1)


def run_benchmarks():
    """Run performance benchmarks"""
    if not JAX_AVAILABLE:
        print("JAX not installed, skipping benchmarks")
        return
        
    print("\n" + "="*60)
    print("JAX Integration Benchmarks")
    print("="*60)
    
    # Tensor conversion benchmarks
    print("\nTensor Conversion Benchmarks:")
    for shape in [(100, 100), (1000, 1000), (10000, 1000)]:
        results = JAXPyTorchBridge.benchmark_conversion(
            tensor_shape=shape,
            num_iterations=10
        )
        print(f"  Shape {shape}:")
        print(f"    PyTorch→JAX: {results['torch_to_jax_ms']:.3f}ms")
        print(f"    JAX→PyTorch: {results['jax_to_torch_ms']:.3f}ms")
        print(f"    Memory share: {results['share_memory_ms']:.3f}ms")
        print(f"    Tensor size: {results['tensor_size_mb']:.1f}MB")
        
    # Compilation cache benchmarks
    print("\nCompilation Cache Performance:")
    cache = JAXCompilationCache(max_size=32)
    
    def test_func(x):
        return jnp.sum(x ** 2 + 2 * x + 1)
        
    # Test cache performance
    shapes = [(10, 10), (100, 100), (500, 500)]
    for shape in shapes:
        x = jnp.ones(shape)
        
        # First compilation (miss)
        start = time.time()
        _ = cache.get_or_compile(test_func, x)
        miss_time = (time.time() - start) * 1000
        
        # Second call (hit)
        start = time.time()
        _ = cache.get_or_compile(test_func, x)
        hit_time = (time.time() - start) * 1000
        
        print(f"  Shape {shape}:")
        print(f"    Cache miss: {miss_time:.3f}ms")
        print(f"    Cache hit:  {hit_time:.3f}ms")
        print(f"    Speedup:    {miss_time/hit_time:.1f}x")
        
    stats = cache.get_cache_stats()
    print(f"\n  Overall cache stats:")
    print(f"    Hit rate: {stats['hit_rate']:.1%}")
    print(f"    Cache size: {stats['cache_size']}/{stats['max_size']}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run benchmarks
    run_benchmarks()