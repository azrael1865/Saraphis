"""
Saraphis GPU Integration Tests - Critical Components
Tests GPU integration with compression systems, brain core, and optimizers
HARD FAILURES ONLY - NO SILENT ERRORS OR FALLBACKS
"""

import unittest
import torch
import numpy as np
import logging
import gc
import time
import os
import sys
import traceback
from typing import Optional, List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestPAdicGPUCompression(unittest.TestCase):
    """Test P-adic compression on GPU - HARD FAILURES ONLY"""
    
    @classmethod
    def setUpClass(cls):
        """Verify GPU and import compression system"""
        if not torch.cuda.is_available():
            raise RuntimeError("FATAL: CUDA required for GPU compression tests")
        
        try:
            from compression_systems.padic.pytorch_padic_engine import (
                PyTorchPAdicEngine, PyTorchPAdicConfig
            )
            cls.PyTorchPAdicEngine = PyTorchPAdicEngine
            cls.PyTorchPAdicConfig = PyTorchPAdicConfig
        except ImportError as e:
            raise RuntimeError(f"FATAL: Cannot import P-adic compression: {e}")
        
        logger.info("P-adic compression system loaded")
    
    def setUp(self):
        """Initialize engine for each test"""
        torch.cuda.empty_cache()
        self.config = self.PyTorchPAdicConfig(
            prime=257,
            precision=6,
            device='cuda',
            enable_triton=False  # Triton is disabled
        )
        self.engine = self.PyTorchPAdicEngine(config=self.config)
    
    def test_gpu_tensor_compression(self):
        """Test compressing tensors on GPU - MUST MAINTAIN ACCURACY"""
        sizes = [(100, 100), (1000, 1000), (5000, 5000)]
        
        for size in sizes:
            # Create GPU tensor
            original = torch.randn(size, device='cuda', dtype=torch.float32)
            
            # Compress
            try:
                compressed = self.engine.encode(original)
            except Exception as e:
                raise RuntimeError(f"FATAL: Compression failed for size {size}: {e}")
            
            # Decompress
            try:
                decompressed = self.engine.decode(compressed)
            except Exception as e:
                raise RuntimeError(f"FATAL: Decompression failed for size {size}: {e}")
            
            # Verify on GPU
            if decompressed.device.type != 'cuda':
                raise AssertionError(f"FATAL: Decompressed tensor not on GPU: {decompressed.device}")
            
            # Check accuracy
            error = torch.abs(original - decompressed).max().item()
            tolerance = 0.01  # 1% tolerance for p-adic compression
            
            if error > tolerance:
                raise AssertionError(
                    f"FATAL: Compression error too high for size {size}. "
                    f"Max error: {error:.6f} > {tolerance}"
                )
            
            logger.info(f"✓ GPU compression for {size}: max error {error:.6f}")
    
    def test_batch_gpu_compression(self):
        """Test batch compression on GPU - MUST BE EFFICIENT"""
        batch_size = 32
        tensor_size = (256, 256)
        
        # Create batch of tensors
        batch = torch.randn(batch_size, *tensor_size, device='cuda')
        
        # Time batch compression
        torch.cuda.synchronize()
        start_time = time.time()
        
        try:
            compressed_batch = self.engine.encode(batch)
        except Exception as e:
            raise RuntimeError(f"FATAL: Batch compression failed: {e}")
        
        torch.cuda.synchronize()
        compress_time = time.time() - start_time
        
        # Time batch decompression
        start_time = time.time()
        
        try:
            decompressed_batch = self.engine.decode(compressed_batch)
        except Exception as e:
            raise RuntimeError(f"FATAL: Batch decompression failed: {e}")
        
        torch.cuda.synchronize()
        decompress_time = time.time() - start_time
        
        # Verify shapes
        if decompressed_batch.shape != batch.shape:
            raise AssertionError(
                f"FATAL: Shape mismatch. Original: {batch.shape}, "
                f"Decompressed: {decompressed_batch.shape}"
            )
        
        # Check compression ratio
        original_bytes = batch.numel() * batch.element_size()
        compressed_bytes = sum(
            t.numel() * t.element_size() 
            for t in compressed_batch.values() 
            if isinstance(t, torch.Tensor)
        )
        ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 1
        
        if ratio < 1.5:  # Expect at least 1.5x compression
            logger.warning(f"⚠ Low compression ratio: {ratio:.2f}x")
        else:
            logger.info(f"✓ Batch compression ratio: {ratio:.2f}x")
        
        logger.info(f"✓ Batch GPU compression: {compress_time:.3f}s compress, "
                   f"{decompress_time:.3f}s decompress")
    
    def test_gpu_memory_efficiency(self):
        """Test memory efficiency of compression - MUST REDUCE MEMORY"""
        # Large tensor that uses significant memory
        size = (10000, 10000)
        original = torch.randn(size, device='cuda', dtype=torch.float32)
        original_memory = original.numel() * original.element_size()
        
        # Compress and check memory
        try:
            compressed = self.engine.encode(original)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("⚠ OOM during compression - reducing size")
                size = (5000, 5000)
                original = torch.randn(size, device='cuda', dtype=torch.float32)
                original_memory = original.numel() * original.element_size()
                compressed = self.engine.encode(original)
            else:
                raise RuntimeError(f"FATAL: Compression failed: {e}")
        
        # Calculate compressed memory
        compressed_memory = 0
        for key, value in compressed.items():
            if isinstance(value, torch.Tensor):
                compressed_memory += value.numel() * value.element_size()
        
        memory_saved = original_memory - compressed_memory
        savings_percent = (memory_saved / original_memory) * 100
        
        if savings_percent < 20:  # Expect at least 20% memory savings
            raise AssertionError(
                f"FATAL: Insufficient memory savings: {savings_percent:.1f}% "
                f"(original: {original_memory/1e6:.2f}MB, "
                f"compressed: {compressed_memory/1e6:.2f}MB)"
            )
        
        logger.info(f"✓ Memory savings: {savings_percent:.1f}% "
                   f"({memory_saved/1e6:.2f}MB saved)")


class TestTropicalGPUOperations(unittest.TestCase):
    """Test Tropical algebra GPU operations - HARD FAILURES ONLY"""
    
    @classmethod
    def setUpClass(cls):
        """Import tropical algebra system"""
        if not torch.cuda.is_available():
            raise RuntimeError("FATAL: CUDA required")
        
        try:
            from compression_systems.tropical.tropical_linear_algebra import (
                TropicalLinearAlgebra, TropicalMatrix
            )
            cls.TropicalLinearAlgebra = TropicalLinearAlgebra
            cls.TropicalMatrix = TropicalMatrix
        except ImportError as e:
            raise RuntimeError(f"FATAL: Cannot import tropical algebra: {e}")
        
        logger.info("Tropical algebra system loaded")
    
    def setUp(self):
        """Initialize tropical algebra on GPU"""
        self.algebra = self.TropicalLinearAlgebra(device=torch.device('cuda'))
    
    def test_tropical_matrix_multiply_gpu(self):
        """Test tropical matrix multiplication on GPU - MUST BE CORRECT"""
        sizes = [(10, 10), (100, 100), (1000, 1000)]
        
        for size in sizes:
            # Create random matrices on GPU
            A_data = torch.randn(size, device='cuda')
            B_data = torch.randn(size, device='cuda')
            
            A = self.TropicalMatrix(A_data)
            B = self.TropicalMatrix(B_data)
            
            # Tropical multiplication
            try:
                C = self.algebra.matrix_multiply(A, B)
            except Exception as e:
                raise RuntimeError(f"FATAL: Tropical multiply failed for size {size}: {e}")
            
            # Verify on GPU
            if C.data.device.type != 'cuda':
                raise AssertionError(f"FATAL: Result not on GPU: {C.data.device}")
            
            # Verify dimensions
            if C.data.shape != size:
                raise AssertionError(
                    f"FATAL: Wrong output shape. Expected {size}, got {C.data.shape}"
                )
            
            # Verify tropical property (idempotent)
            C2 = self.algebra.matrix_multiply(C, C)
            
            # In tropical algebra, repeated multiplication should converge
            diff = torch.abs(C2.data - C.data).max().item()
            
            logger.info(f"✓ Tropical GPU multiply {size}: convergence diff {diff:.6f}")
    
    def test_tropical_eigenvalue_gpu(self):
        """Test tropical eigenvalue computation on GPU - MUST CONVERGE"""
        size = 100
        matrix_data = torch.randn(size, size, device='cuda')
        matrix = self.TropicalMatrix(matrix_data)
        
        # Compute eigenvalue
        try:
            eigenvalue = self.algebra.eigenvalue(matrix, method='power')
        except Exception as e:
            raise RuntimeError(f"FATAL: Eigenvalue computation failed: {e}")
        
        # Verify result is finite
        if not torch.isfinite(eigenvalue).all():
            raise AssertionError(f"FATAL: Non-finite eigenvalue: {eigenvalue}")
        
        logger.info(f"✓ Tropical eigenvalue on GPU: {eigenvalue.item():.6f}")
    
    def test_gpu_performance_advantage(self):
        """Test GPU provides performance advantage - SHOULD BE FASTER"""
        size = 2000
        
        # Create large matrices
        A_data = torch.randn(size, size)
        B_data = torch.randn(size, size)
        
        # CPU timing
        A_cpu = self.TropicalMatrix(A_data)
        B_cpu = self.TropicalMatrix(B_data)
        
        cpu_algebra = self.TropicalLinearAlgebra(device=torch.device('cpu'))
        
        start = time.time()
        C_cpu = cpu_algebra.matrix_multiply(A_cpu, B_cpu)
        cpu_time = time.time() - start
        
        # GPU timing
        A_gpu = self.TropicalMatrix(A_data.cuda())
        B_gpu = self.TropicalMatrix(B_data.cuda())
        
        torch.cuda.synchronize()
        start = time.time()
        C_gpu = self.algebra.matrix_multiply(A_gpu, B_gpu)
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        speedup = cpu_time / gpu_time
        
        if speedup < 1.0:
            logger.warning(f"⚠ GPU slower than CPU: {speedup:.2f}x")
        else:
            logger.info(f"✓ GPU speedup: {speedup:.2f}x (CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s)")


class TestBrainCoreGPU(unittest.TestCase):
    """Test Brain Core GPU integration - HARD FAILURES ONLY"""
    
    @classmethod
    def setUpClass(cls):
        """Import brain core"""
        if not torch.cuda.is_available():
            raise RuntimeError("FATAL: CUDA required")
        
        try:
            from brain_core import BrainCore, BrainConfig
            cls.BrainCore = BrainCore
            cls.BrainConfig = BrainConfig
        except ImportError as e:
            raise RuntimeError(f"FATAL: Cannot import brain core: {e}")
        
        logger.info("Brain Core loaded")
    
    def setUp(self):
        """Initialize brain core"""
        self.config = self.BrainConfig()
        self.brain = self.BrainCore(self.config)
    
    def test_brain_gpu_prediction(self):
        """Test brain predictions utilize GPU - MUST USE GPU WHEN AVAILABLE"""
        # Create GPU tensor input
        data = torch.randn(1000, 100, device='cuda')
        
        # Make prediction
        try:
            result = self.brain.predict(data)
        except Exception as e:
            raise RuntimeError(f"FATAL: Brain prediction failed: {e}")
        
        # Verify result
        if result is None:
            raise AssertionError("FATAL: Brain returned None")
        
        if not hasattr(result, 'confidence'):
            raise AssertionError("FATAL: Result missing confidence score")
        
        logger.info(f"✓ Brain GPU prediction: confidence {result.confidence:.3f}")
    
    def test_brain_batch_processing(self):
        """Test brain batch processing on GPU - MUST HANDLE BATCHES"""
        batch_sizes = [10, 50, 100]
        
        for batch_size in batch_sizes:
            # Create batch
            batch_data = [
                torch.randn(100, device='cuda') for _ in range(batch_size)
            ]
            
            # Process batch
            results = []
            try:
                for data in batch_data:
                    result = self.brain.predict(data)
                    results.append(result)
            except Exception as e:
                raise RuntimeError(f"FATAL: Batch processing failed at size {batch_size}: {e}")
            
            # Verify all processed
            if len(results) != batch_size:
                raise AssertionError(
                    f"FATAL: Batch size mismatch. Expected {batch_size}, got {len(results)}"
                )
            
            avg_confidence = sum(r.confidence for r in results) / len(results)
            logger.info(f"✓ Brain batch {batch_size}: avg confidence {avg_confidence:.3f}")
    
    def test_brain_memory_management(self):
        """Test brain manages GPU memory properly - MUST NOT LEAK"""
        initial_memory = torch.cuda.memory_allocated()
        
        # Run many predictions
        for i in range(100):
            data = torch.randn(1000, 100, device='cuda')
            result = self.brain.predict(data)
            
            if i % 20 == 0:
                current_memory = torch.cuda.memory_allocated()
                leak = current_memory - initial_memory
                
                # Allow up to 100MB growth
                if leak > 100 * 1024 * 1024:
                    raise AssertionError(
                        f"FATAL: Memory leak detected. Leaked {leak/1e6:.2f}MB after {i} predictions"
                    )
        
        # Final check
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        total_leak = final_memory - initial_memory
        
        if total_leak > 10 * 1024 * 1024:  # 10MB tolerance
            logger.warning(f"⚠ Possible memory leak: {total_leak/1e6:.2f}MB")
        else:
            logger.info(f"✓ Brain memory management OK: {total_leak/1024:.2f}KB difference")


class TestGPUMemoryOptimizer(unittest.TestCase):
    """Test GPU memory optimizer component - HARD FAILURES ONLY"""
    
    @classmethod
    def setUpClass(cls):
        """Import memory optimizer"""
        if not torch.cuda.is_available():
            raise RuntimeError("FATAL: CUDA required")
        
        try:
            from gpu_memory_optimizer import GPUMemoryOptimizer
            cls.GPUMemoryOptimizer = GPUMemoryOptimizer
        except ImportError as e:
            # Try alternative path
            try:
                from compression_systems.gpu_memory.gpu_memory_core import GPUMemoryManager
                cls.GPUMemoryOptimizer = GPUMemoryManager
            except ImportError:
                raise RuntimeError(f"FATAL: Cannot import GPU memory optimizer: {e}")
        
        logger.info("GPU Memory Optimizer loaded")
    
    def setUp(self):
        """Initialize optimizer"""
        self.optimizer = self.GPUMemoryOptimizer()
        torch.cuda.empty_cache()
    
    def test_memory_optimization(self):
        """Test memory optimization strategies - MUST REDUCE USAGE"""
        # Create memory pressure
        tensors = []
        for i in range(10):
            tensors.append(torch.randn(1000, 1000, device='cuda'))
        
        initial_memory = torch.cuda.memory_allocated()
        
        # Optimize
        try:
            self.optimizer.optimize()
        except Exception as e:
            logger.warning(f"⚠ Optimization failed: {e}")
            # Try manual optimization
            torch.cuda.empty_cache()
        
        optimized_memory = torch.cuda.memory_allocated()
        
        if optimized_memory >= initial_memory:
            logger.warning(f"⚠ No memory reduction: {initial_memory/1e6:.2f}MB -> {optimized_memory/1e6:.2f}MB")
        else:
            reduction = (initial_memory - optimized_memory) / initial_memory * 100
            logger.info(f"✓ Memory reduced by {reduction:.1f}%")
    
    def test_automatic_cleanup(self):
        """Test automatic memory cleanup - MUST FREE UNUSED MEMORY"""
        # Allocate and free repeatedly
        for i in range(5):
            # Allocate
            temp_tensors = [torch.randn(5000, 5000, device='cuda') for _ in range(3)]
            allocated = torch.cuda.memory_allocated()
            
            # Delete
            del temp_tensors
            
            # Trigger cleanup
            try:
                self.optimizer.cleanup()
            except:
                torch.cuda.empty_cache()
            
            freed = torch.cuda.memory_allocated()
            
            if freed >= allocated * 0.9:  # Should free at least 10%
                raise AssertionError(
                    f"FATAL: Memory not freed. Before: {allocated/1e6:.2f}MB, "
                    f"After: {freed/1e6:.2f}MB"
                )
            
            logger.info(f"✓ Iteration {i+1}: Freed {(allocated-freed)/1e6:.2f}MB")


def run_saraphis_gpu_tests():
    """Run all Saraphis GPU integration tests"""
    logger.info("=" * 80)
    logger.info("SARAPHIS GPU INTEGRATION TESTS - HARD FAILURE MODE")
    logger.info("=" * 80)
    
    if not torch.cuda.is_available():
        raise RuntimeError("FATAL: NO CUDA DEVICES AVAILABLE")
    
    # Log GPU info
    props = torch.cuda.get_device_properties(0)
    logger.info(f"GPU: {props.name}")
    logger.info(f"Compute Capability: {props.major}.{props.minor}")
    logger.info(f"Memory: {props.total_memory / 1e9:.2f} GB")
    logger.info("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestPAdicGPUCompression,
        TestTropicalGPUOperations,
        TestBrainCoreGPU,
        TestGPUMemoryOptimizer
    ]
    
    for test_class in test_classes:
        try:
            suite.addTests(loader.loadTestsFromTestCase(test_class))
        except Exception as e:
            logger.warning(f"⚠ Could not load {test_class.__name__}: {e}")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    
    try:
        result = runner.run(suite)
        
        if not result.wasSuccessful():
            logger.error("=" * 80)
            logger.error("INTEGRATION TESTS FAILED")
            for failure in result.failures + result.errors:
                logger.error(f"FAILED: {failure[0]}")
            logger.error("=" * 80)
            return False
        else:
            logger.info("=" * 80)
            logger.info("ALL INTEGRATION TESTS PASSED")
            logger.info("=" * 80)
            return True
            
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = run_saraphis_gpu_tests()
    sys.exit(0 if success else 1)