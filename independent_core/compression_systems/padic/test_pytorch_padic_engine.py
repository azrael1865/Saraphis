"""
Comprehensive test suite for PyTorch P-adic Engine
Tests basic operations, batch processing, device compatibility, and performance
NO PLACEHOLDERS - PRODUCTION READY
"""

import torch
import numpy as np
import pytest
import time
import math
from typing import List, Tuple
from fractions import Fraction

from .pytorch_padic_engine import (
    PyTorchPAdicEngine,
    PyTorchPAdicConfig,
    differentiable_padic_encode,
    TRITON_AVAILABLE
)
from .padic_encoder import PadicWeight, create_real_padic_weights


class TestPyTorchPAdicEngine:
    """Test suite for PyTorch P-adic Engine"""
    
    @pytest.fixture
    def engine(self):
        """Create default engine for testing"""
        config = PyTorchPAdicConfig(
            prime=257,
            precision=4,
            device="cpu",  # Use CPU for predictable testing
            enable_triton=False
        )
        return PyTorchPAdicEngine(config=config)
    
    @pytest.fixture
    def gpu_engine(self):
        """Create GPU engine if available"""
        if torch.cuda.is_available():
            config = PyTorchPAdicConfig(
                prime=257,
                precision=4,
                device="cuda",
                enable_triton=TRITON_AVAILABLE
            )
            return PyTorchPAdicEngine(config=config)
        elif torch.backends.mps.is_available():
            config = PyTorchPAdicConfig(
                prime=257,
                precision=4,
                device="mps",
                enable_triton=False
            )
            return PyTorchPAdicEngine(config=config)
        else:
            pytest.skip("No GPU available")
    
    def test_initialization(self):
        """Test engine initialization with various configurations"""
        # Test default initialization
        engine = PyTorchPAdicEngine()
        assert engine.prime == 257
        assert engine.precision == 4
        
        # Test custom initialization
        engine = PyTorchPAdicEngine(prime=251, precision=8)
        assert engine.prime == 251
        assert engine.precision == 8
        
        # Test with config
        config = PyTorchPAdicConfig(prime=241, precision=6, device="cpu")
        engine = PyTorchPAdicEngine(config=config)
        assert engine.prime == 241
        assert engine.precision == 6
        assert engine.device == torch.device("cpu")
    
    def test_invalid_configuration(self):
        """Test validation of invalid configurations"""
        # Invalid prime
        with pytest.raises(ValueError):
            PyTorchPAdicConfig(prime=4)  # Not prime
        
        # Invalid precision
        with pytest.raises(ValueError):
            PyTorchPAdicConfig(precision=0)
        
        # Invalid device
        with pytest.raises(ValueError):
            PyTorchPAdicConfig(device="invalid")
        
        # Invalid batch size
        with pytest.raises(ValueError):
            PyTorchPAdicConfig(batch_size=-1)
    
    def test_basic_encoding_decoding(self, engine):
        """Test basic p-adic encoding and decoding"""
        test_values = [
            0.0,
            1.0,
            -1.0,
            3.14159,
            -2.71828,
            0.5,
            -0.5,
            100.0,
            -100.0,
            1e-5,
            -1e-5
        ]
        
        for value in test_values:
            # Encode
            padic = engine.to_padic(value)
            assert padic.shape == (engine.precision,)
            assert padic.dtype == torch.long
            
            # Decode
            decoded = engine.from_padic(padic)
            assert decoded.shape == ()
            
            # Check accuracy
            rel_error = abs(decoded.item() - value) / (abs(value) + 1e-10)
            assert rel_error < 1e-3, f"Failed for value {value}: decoded={decoded.item()}, error={rel_error}"
    
    def test_tensor_encoding_decoding(self, engine):
        """Test encoding and decoding of tensors"""
        # 1D tensor
        tensor_1d = torch.randn(100)
        padic_1d = engine.to_padic(tensor_1d)
        assert padic_1d.shape == (100, engine.precision)
        decoded_1d = engine.from_padic(padic_1d)
        assert decoded_1d.shape == tensor_1d.shape
        
        # 2D tensor
        tensor_2d = torch.randn(10, 20)
        padic_2d = engine.to_padic(tensor_2d)
        assert padic_2d.shape == (10, 20, engine.precision)
        decoded_2d = engine.from_padic(padic_2d)
        assert decoded_2d.shape == tensor_2d.shape
        
        # 3D tensor
        tensor_3d = torch.randn(5, 10, 15)
        padic_3d = engine.to_padic(tensor_3d)
        assert padic_3d.shape == (5, 10, 15, engine.precision)
        decoded_3d = engine.from_padic(padic_3d)
        assert decoded_3d.shape == tensor_3d.shape
    
    def test_numpy_array_support(self, engine):
        """Test support for numpy arrays"""
        np_array = np.random.randn(50).astype(np.float32)
        
        # Encode numpy array
        padic = engine.to_padic(np_array)
        assert padic.shape == (50, engine.precision)
        
        # Decode back
        decoded = engine.from_padic(padic)
        assert decoded.shape == (50,)
        
        # Check accuracy
        rel_error = np.abs(decoded.cpu().numpy() - np_array) / (np.abs(np_array) + 1e-10)
        assert np.max(rel_error) < 1e-3
    
    def test_padic_addition(self, engine):
        """Test p-adic addition"""
        # Test scalar addition
        a = engine.to_padic(3.5)
        b = engine.to_padic(2.7)
        c = engine.padic_add(a, b)
        result = engine.from_padic(c)
        expected = 3.5 + 2.7
        assert abs(result.item() - expected) < 1e-3
        
        # Test tensor addition
        tensor_a = torch.randn(10, 10)
        tensor_b = torch.randn(10, 10)
        padic_a = engine.to_padic(tensor_a)
        padic_b = engine.to_padic(tensor_b)
        padic_c = engine.padic_add(padic_a, padic_b)
        result = engine.from_padic(padic_c)
        expected = tensor_a + tensor_b
        rel_error = torch.abs(result - expected) / (torch.abs(expected) + 1e-10)
        assert torch.max(rel_error) < 1e-2
    
    def test_padic_multiplication(self, engine):
        """Test p-adic multiplication"""
        # Test scalar multiplication
        a = engine.to_padic(2.5)
        b = engine.to_padic(3.0)
        c = engine.padic_multiply(a, b)
        result = engine.from_padic(c)
        expected = 2.5 * 3.0
        assert abs(result.item() - expected) < 1e-2
        
        # Test tensor multiplication
        tensor_a = torch.randn(5, 5) * 0.1  # Small values to avoid overflow
        tensor_b = torch.randn(5, 5) * 0.1
        padic_a = engine.to_padic(tensor_a)
        padic_b = engine.to_padic(tensor_b)
        padic_c = engine.padic_multiply(padic_a, padic_b)
        result = engine.from_padic(padic_c)
        expected = tensor_a * tensor_b
        rel_error = torch.abs(result - expected) / (torch.abs(expected) + 1e-10)
        assert torch.max(rel_error) < 0.1  # Higher tolerance for multiplication
    
    def test_padic_norm(self, engine):
        """Test p-adic norm calculation"""
        # Test various values
        test_values = [1.0, 0.5, 2.0, engine.prime, 1.0/engine.prime]
        
        for value in test_values:
            padic = engine.to_padic(value)
            norm = engine.padic_norm(padic)
            assert norm.shape == ()
            assert 0 <= norm.item() <= 1
    
    def test_hensel_lifting(self, engine):
        """Test Hensel lifting for improved precision"""
        test_value = 3.14159265359
        
        # Initial approximation
        initial = torch.tensor(3.14)
        
        # Perform Hensel lifting
        lifted = engine.hensel_lift_torch(initial, target_error=1e-6)
        
        # Decode and check
        result = engine.from_padic(lifted)
        error = abs(result.item() - test_value)
        assert error < 1e-3  # Should be more accurate after lifting
    
    def test_batch_operations(self, engine):
        """Test batch processing capabilities"""
        batch_size = 1000
        tensor = torch.randn(batch_size)
        
        # Batch encoding
        start_time = time.time()
        padic_batch = engine.batch_to_padic(tensor)
        encode_time = time.time() - start_time
        
        assert padic_batch.shape == (batch_size, engine.precision)
        
        # Batch decoding
        start_time = time.time()
        decoded_batch = engine.batch_from_padic(padic_batch)
        decode_time = time.time() - start_time
        
        assert decoded_batch.shape == (batch_size,)
        
        # Check accuracy
        rel_error = torch.abs(decoded_batch - tensor) / (torch.abs(tensor) + 1e-10)
        assert torch.max(rel_error) < 1e-3
        
        print(f"Batch encoding time: {encode_time:.4f}s")
        print(f"Batch decoding time: {decode_time:.4f}s")
    
    def test_device_compatibility(self):
        """Test compatibility with different devices"""
        devices = []
        
        # Check available devices
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
        devices.append("cpu")
        
        test_tensor = torch.randn(100)
        
        for device in devices:
            config = PyTorchPAdicConfig(device=device)
            engine = PyTorchPAdicEngine(config=config)
            
            # Test encoding/decoding
            padic = engine.to_padic(test_tensor)
            decoded = engine.from_padic(padic)
            
            # Check device placement
            assert str(padic.device).startswith(device)
            assert str(decoded.device).startswith(device)
            
            # Check accuracy
            rel_error = torch.abs(decoded.cpu() - test_tensor) / (torch.abs(test_tensor) + 1e-10)
            assert torch.max(rel_error) < 1e-3
    
    def test_gpu_acceleration(self, gpu_engine):
        """Test GPU acceleration if available"""
        # Large tensor for GPU testing
        large_tensor = torch.randn(10000, device=gpu_engine.device)
        
        # Measure GPU performance
        start_time = time.time()
        padic = gpu_engine.to_padic(large_tensor)
        gpu_encode_time = time.time() - start_time
        
        start_time = time.time()
        decoded = gpu_engine.from_padic(padic)
        gpu_decode_time = time.time() - start_time
        
        # Compare with CPU
        cpu_engine = PyTorchPAdicEngine(
            PyTorchPAdicConfig(device="cpu", prime=gpu_engine.prime, precision=gpu_engine.precision)
        )
        cpu_tensor = large_tensor.cpu()
        
        start_time = time.time()
        cpu_padic = cpu_engine.to_padic(cpu_tensor)
        cpu_encode_time = time.time() - start_time
        
        start_time = time.time()
        cpu_decoded = cpu_engine.from_padic(cpu_padic)
        cpu_decode_time = time.time() - start_time
        
        print(f"GPU encode time: {gpu_encode_time:.4f}s")
        print(f"CPU encode time: {cpu_encode_time:.4f}s")
        print(f"GPU speedup: {cpu_encode_time/gpu_encode_time:.2f}x")
        
        # Check if Triton is being used
        if gpu_engine.triton_enabled:
            assert gpu_engine.stats['triton_calls'] > 0
            print(f"Triton calls: {gpu_engine.stats['triton_calls']}")
    
    def test_gradient_support(self, engine):
        """Test gradient support for differentiable operations"""
        if not engine.config.gradient_enabled:
            pytest.skip("Gradient support disabled")
        
        # Create tensor with gradient
        x = torch.randn(10, requires_grad=True)
        
        # Apply differentiable p-adic encoding
        padic = differentiable_padic_encode(x, engine)
        
        # Create loss and backpropagate
        loss = padic.sum()
        loss.backward()
        
        # Check gradient exists
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_mixed_precision(self, engine):
        """Test mixed precision support"""
        if not engine.config.enable_mixed_precision:
            pytest.skip("Mixed precision disabled")
        
        # Test with different dtypes
        for dtype in [torch.float16, torch.float32, torch.float64]:
            tensor = torch.randn(100, dtype=dtype)
            padic = engine.to_padic(tensor)
            decoded = engine.from_padic(padic)
            
            # Check shape preservation
            assert decoded.shape == tensor.shape
    
    def test_padic_weight_compatibility(self, engine):
        """Test compatibility with existing PadicWeight structures"""
        test_value = 3.14159
        
        # Convert to PadicWeight
        weight = engine.to_padic_weight(test_value)
        assert isinstance(weight, PadicWeight)
        assert weight.prime == engine.prime
        assert weight.precision == engine.precision
        assert len(weight.digits) == engine.precision
        
        # Convert back from PadicWeight
        decoded = engine.from_padic_weight(weight)
        assert abs(decoded - test_value) < 1e-3
    
    def test_dynamic_prime_switching(self, engine):
        """Test dynamic prime switching"""
        original_prime = engine.prime
        test_tensor = torch.randn(10)
        
        # Encode with original prime
        padic1 = engine.to_padic(test_tensor)
        
        # Switch to new prime
        new_prime = 251
        engine.switch_prime_dynamically(new_prime)
        assert engine.prime == new_prime
        
        # Encode with new prime
        padic2 = engine.to_padic(test_tensor)
        
        # Results should be different
        assert not torch.allclose(padic1, padic2)
        
        # Switch back
        engine.switch_prime_dynamically(original_prime)
        assert engine.prime == original_prime
    
    def test_performance_statistics(self, engine):
        """Test performance tracking"""
        engine.reset_stats()
        
        # Perform operations
        tensor = torch.randn(100)
        padic = engine.to_padic(tensor)
        decoded = engine.from_padic(padic)
        
        # Check statistics
        stats = engine.get_stats()
        assert stats['total_conversions'] > 0
        assert 'device' in stats
        assert 'triton_enabled' in stats
        assert 'compile_enabled' in stats
    
    def test_edge_cases(self, engine):
        """Test edge cases and boundary conditions"""
        edge_values = [
            0.0,
            float('inf'),  # Should raise error
            float('-inf'),  # Should raise error
            float('nan'),  # Should raise error
            1e-10,
            1e10,
            engine.prime,
            1.0 / engine.prime,
            engine.prime ** 2,
            1.0 / (engine.prime ** 2)
        ]
        
        for value in edge_values:
            if math.isnan(value) or math.isinf(value):
                # Should handle gracefully or raise appropriate error
                try:
                    padic = engine.to_padic(value)
                    assert False, f"Should have raised error for {value}"
                except (ValueError, RuntimeError):
                    pass  # Expected
            else:
                padic = engine.to_padic(value)
                decoded = engine.from_padic(padic)
                # Some loss of precision is expected for extreme values
                if abs(value) < 1e-5 or abs(value) > 1e5:
                    assert not math.isnan(decoded.item())
                else:
                    rel_error = abs(decoded.item() - value) / (abs(value) + 1e-10)
                    assert rel_error < 0.1
    
    def test_memory_efficiency(self, engine):
        """Test memory-efficient operations"""
        if not engine.config.memory_efficient:
            pytest.skip("Memory efficiency disabled")
        
        # Check pre-allocated buffers exist
        assert hasattr(engine, 'digit_buffer')
        assert hasattr(engine, 'carry_buffer')
        
        # Test that buffers are reused
        initial_buffer_id = id(engine.digit_buffer)
        
        # Perform multiple operations
        for _ in range(10):
            tensor = torch.randn(min(100, engine.config.batch_size))
            padic = engine.to_padic(tensor)
            decoded = engine.from_padic(padic)
        
        # Buffer should still be the same object
        assert id(engine.digit_buffer) == initial_buffer_id
    
    def test_compile_mode(self):
        """Test different compile modes"""
        if not hasattr(torch, 'compile'):
            pytest.skip("torch.compile not available")
        
        compile_modes = ["default", "reduce-overhead", "max-autotune"]
        test_tensor = torch.randn(100)
        
        for mode in compile_modes:
            config = PyTorchPAdicConfig(compile_mode=mode)
            engine = PyTorchPAdicEngine(config=config)
            
            # Test operations work with each compile mode
            padic = engine.to_padic(test_tensor)
            decoded = engine.from_padic(padic)
            
            # Check accuracy
            rel_error = torch.abs(decoded - test_tensor) / (torch.abs(test_tensor) + 1e-10)
            assert torch.max(rel_error) < 1e-3
    
    def test_thread_safety(self, engine):
        """Test thread safety of operations"""
        import threading
        import queue
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def worker(thread_id):
            try:
                # Each thread performs operations
                tensor = torch.randn(10)
                padic = engine.to_padic(tensor)
                decoded = engine.from_padic(padic)
                
                # Check accuracy
                rel_error = torch.abs(decoded - tensor) / (torch.abs(tensor) + 1e-10)
                max_error = torch.max(rel_error).item()
                
                results.put((thread_id, max_error))
            except Exception as e:
                errors.put((thread_id, str(e)))
        
        # Launch multiple threads
        threads = []
        num_threads = 10
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check results
        assert errors.empty(), f"Thread errors: {list(errors.queue)}"
        assert results.qsize() == num_threads
        
        # All results should be accurate
        while not results.empty():
            thread_id, max_error = results.get()
            assert max_error < 1e-3, f"Thread {thread_id} had error {max_error}"


class TestPerformanceBenchmarks:
    """Performance benchmarks for PyTorch P-adic Engine"""
    
    def benchmark_encoding_speed(self):
        """Benchmark encoding speed for different sizes"""
        sizes = [100, 1000, 10000, 100000]
        devices = []
        
        if torch.cuda.is_available():
            devices.append("cuda")
        devices.append("cpu")
        
        results = {}
        
        for device in devices:
            config = PyTorchPAdicConfig(device=device, enable_triton=True)
            engine = PyTorchPAdicEngine(config=config)
            device_results = {}
            
            for size in sizes:
                tensor = torch.randn(size, device=engine.device)
                
                # Warmup
                for _ in range(3):
                    _ = engine.to_padic(tensor)
                
                # Benchmark
                start_time = time.time()
                iterations = 10
                for _ in range(iterations):
                    _ = engine.to_padic(tensor)
                
                elapsed = (time.time() - start_time) / iterations
                throughput = size / elapsed
                device_results[size] = {
                    'time': elapsed,
                    'throughput': throughput
                }
            
            results[device] = device_results
        
        # Print results
        print("\nEncoding Performance Benchmarks:")
        print("-" * 60)
        for device, device_results in results.items():
            print(f"\nDevice: {device}")
            print(f"{'Size':<10} {'Time (s)':<15} {'Throughput (elem/s)':<20}")
            for size, metrics in device_results.items():
                print(f"{size:<10} {metrics['time']:<15.6f} {metrics['throughput']:<20.2f}")
        
        return results
    
    def benchmark_arithmetic_operations(self):
        """Benchmark p-adic arithmetic operations"""
        size = 10000
        devices = []
        
        if torch.cuda.is_available():
            devices.append("cuda")
        devices.append("cpu")
        
        results = {}
        
        for device in devices:
            config = PyTorchPAdicConfig(device=device, enable_triton=True)
            engine = PyTorchPAdicEngine(config=config)
            
            # Prepare data
            tensor_a = torch.randn(size, device=engine.device)
            tensor_b = torch.randn(size, device=engine.device)
            padic_a = engine.to_padic(tensor_a)
            padic_b = engine.to_padic(tensor_b)
            
            # Benchmark addition
            start_time = time.time()
            iterations = 100
            for _ in range(iterations):
                _ = engine.padic_add(padic_a, padic_b)
            add_time = (time.time() - start_time) / iterations
            
            # Benchmark multiplication
            start_time = time.time()
            for _ in range(iterations):
                _ = engine.padic_multiply(padic_a, padic_b)
            mul_time = (time.time() - start_time) / iterations
            
            results[device] = {
                'addition': add_time,
                'multiplication': mul_time
            }
        
        # Print results
        print("\nArithmetic Operations Benchmarks:")
        print("-" * 60)
        for device, metrics in results.items():
            print(f"\nDevice: {device}")
            print(f"Addition: {metrics['addition']:.6f}s")
            print(f"Multiplication: {metrics['multiplication']:.6f}s")
        
        return results
    
    def benchmark_vs_existing_implementation(self):
        """Compare performance with existing p-adic implementation"""
        from .padic_encoder import PadicMathematicalOperations
        
        num_weights = 1000
        prime = 257
        precision = 4
        
        # PyTorch engine
        pytorch_engine = PyTorchPAdicEngine(prime=prime, precision=precision)
        
        # Existing implementation
        existing_ops = PadicMathematicalOperations(prime=prime, precision=precision)
        
        # Generate test data
        test_values = np.random.randn(num_weights).tolist()
        
        # Benchmark PyTorch engine
        start_time = time.time()
        for value in test_values:
            padic = pytorch_engine.to_padic(value)
            decoded = pytorch_engine.from_padic(padic)
        pytorch_time = time.time() - start_time
        
        # Benchmark existing implementation
        start_time = time.time()
        for value in test_values:
            weight = existing_ops.to_padic(value)
            decoded = existing_ops.from_padic(weight)
        existing_time = time.time() - start_time
        
        # Print comparison
        print("\nPerformance Comparison:")
        print("-" * 60)
        print(f"PyTorch Engine: {pytorch_time:.4f}s")
        print(f"Existing Implementation: {existing_time:.4f}s")
        print(f"Speedup: {existing_time/pytorch_time:.2f}x")
        
        return {
            'pytorch': pytorch_time,
            'existing': existing_time,
            'speedup': existing_time/pytorch_time
        }


if __name__ == "__main__":
    # Run basic tests
    print("Running PyTorch P-adic Engine Tests...")
    
    # Create test instance
    tester = TestPyTorchPAdicEngine()
    
    # Create engine
    engine = PyTorchPAdicEngine()
    print(f"Engine: {engine}")
    
    # Run basic tests
    print("\n1. Testing basic encoding/decoding...")
    tester.test_basic_encoding_decoding(engine)
    print("✓ Basic encoding/decoding passed")
    
    print("\n2. Testing tensor operations...")
    tester.test_tensor_encoding_decoding(engine)
    print("✓ Tensor operations passed")
    
    print("\n3. Testing arithmetic operations...")
    tester.test_padic_addition(engine)
    tester.test_padic_multiplication(engine)
    print("✓ Arithmetic operations passed")
    
    print("\n4. Testing batch operations...")
    tester.test_batch_operations(engine)
    print("✓ Batch operations passed")
    
    print("\n5. Testing device compatibility...")
    tester.test_device_compatibility()
    print("✓ Device compatibility passed")
    
    # Run benchmarks
    print("\n" + "="*60)
    print("Running Performance Benchmarks...")
    print("="*60)
    
    benchmarker = TestPerformanceBenchmarks()
    
    print("\n6. Encoding speed benchmark...")
    benchmarker.benchmark_encoding_speed()
    
    print("\n7. Arithmetic operations benchmark...")
    benchmarker.benchmark_arithmetic_operations()
    
    print("\n8. Comparison with existing implementation...")
    benchmarker.benchmark_vs_existing_implementation()
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)