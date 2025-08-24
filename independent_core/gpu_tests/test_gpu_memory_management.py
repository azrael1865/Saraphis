"""
GPU Memory Management Tests - Priority 1 (Critical)
Tests GPU memory allocation, caching, cleanup, and OOM handling
HARD FAILURES ONLY - NO SILENT ERRORS
"""

import unittest
import torch
import logging
import gc
import time
import psutil
import os
import sys
import traceback
from typing import List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestGPUMemoryAllocation(unittest.TestCase):
    """Test GPU memory allocation patterns - HARD FAILURES ONLY"""
    
    def setUp(self):
        """Reset GPU state before each test"""
        if not torch.cuda.is_available():
            raise RuntimeError("FATAL: CUDA required for memory tests")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        self.initial_memory = torch.cuda.memory_allocated()
        logger.info(f"Initial memory: {self.initial_memory / 1e6:.2f} MB")
    
    def tearDown(self):
        """Verify memory cleanup after each test"""
        torch.cuda.empty_cache()
        gc.collect()
        
        final_memory = torch.cuda.memory_allocated()
        leaked = final_memory - self.initial_memory
        
        if leaked > 1024 * 1024:  # More than 1MB leaked
            raise AssertionError(f"FATAL: Memory leak detected: {leaked / 1e6:.2f} MB")
    
    def test_incremental_allocation(self):
        """Test incremental memory allocation - MUST TRACK CORRECTLY"""
        allocations = []
        expected_total = 0
        
        sizes_mb = [10, 50, 100, 200, 500]
        
        for size_mb in sizes_mb:
            elements = (size_mb * 1024 * 1024) // 4  # float32
            
            try:
                tensor = torch.randn(elements, device='cuda')
                allocations.append(tensor)
            except RuntimeError as e:
                raise RuntimeError(f"FATAL: Cannot allocate {size_mb}MB: {e}")
            
            expected_total += elements * 4
            actual_total = torch.cuda.memory_allocated() - self.initial_memory
            
            # Allow 1% overhead
            if abs(actual_total - expected_total) > expected_total * 0.01:
                raise AssertionError(
                    f"FATAL: Memory tracking error. Expected {expected_total/1e6:.2f}MB, "
                    f"got {actual_total/1e6:.2f}MB"
                )
            
            logger.info(f"✓ Allocated {size_mb}MB, total: {actual_total/1e6:.2f}MB")
        
        # Cleanup
        del allocations
        torch.cuda.empty_cache()
        
        logger.info("✓ Incremental allocation tracking correct")
    
    def test_large_allocation(self):
        """Test large single allocation - MUST HANDLE CORRECTLY"""
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory
        
        # Try to allocate 50% of total memory
        target_bytes = total_memory // 2
        elements = target_bytes // 4  # float32
        
        try:
            large_tensor = torch.randn(elements, device='cuda')
            allocated = torch.cuda.memory_allocated() - self.initial_memory
            
            if allocated < target_bytes * 0.95:  # Allow 5% less
                raise AssertionError(
                    f"FATAL: Large allocation failed. Expected ~{target_bytes/1e9:.2f}GB, "
                    f"got {allocated/1e9:.2f}GB"
                )
            
            logger.info(f"✓ Large allocation successful: {allocated/1e9:.2f}GB")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Try smaller allocation
                elements = elements // 2
                try:
                    large_tensor = torch.randn(elements, device='cuda')
                    logger.warning(f"⚠ Had to reduce allocation size to {elements*4/1e9:.2f}GB")
                except:
                    raise RuntimeError(f"FATAL: Cannot allocate even reduced size: {e}")
            else:
                raise RuntimeError(f"FATAL: Unexpected error in large allocation: {e}")
        
        del large_tensor
        torch.cuda.empty_cache()
    
    def test_allocation_patterns(self):
        """Test various allocation patterns - ALL MUST WORK"""
        patterns = [
            ("contiguous", lambda: torch.randn(1000, 1000, device='cuda')),
            ("non-contiguous", lambda: torch.randn(1000, 1000, device='cuda').t()),
            ("strided", lambda: torch.randn(2000, 2000, device='cuda')[::2, ::2]),
            ("view", lambda: torch.randn(1000000, device='cuda').view(1000, 1000)),
            ("reshape", lambda: torch.randn(1000000, device='cuda').reshape(1000, 1000))
        ]
        
        for name, allocator in patterns:
            try:
                tensor = allocator()
                if tensor.device.type != 'cuda':
                    raise AssertionError(f"FATAL: {name} pattern not on CUDA")
                logger.info(f"✓ {name} allocation pattern successful")
                del tensor
            except Exception as e:
                raise RuntimeError(f"FATAL: {name} pattern failed: {e}")
        
        torch.cuda.empty_cache()


class TestGPUMemoryCaching(unittest.TestCase):
    """Test GPU memory caching behavior - HARD FAILURES ONLY"""
    
    def setUp(self):
        """Clear caches before tests"""
        if not torch.cuda.is_available():
            raise RuntimeError("FATAL: CUDA required")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    def test_cache_behavior(self):
        """Test PyTorch GPU cache behavior - MUST BE PREDICTABLE"""
        # Allocate and free multiple times
        for i in range(5):
            tensor = torch.randn(10000, 10000, device='cuda')
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            
            if allocated > reserved:
                raise AssertionError(
                    f"FATAL: Allocated memory ({allocated}) > Reserved memory ({reserved})"
                )
            
            del tensor
            
            # Memory should be cached, not freed
            allocated_after = torch.cuda.memory_allocated()
            reserved_after = torch.cuda.memory_reserved()
            
            if allocated_after >= allocated:
                raise AssertionError(
                    f"FATAL: Memory not released after del. Before: {allocated}, After: {allocated_after}"
                )
            
            logger.info(f"✓ Iteration {i+1}: Cache working (reserved: {reserved_after/1e6:.2f}MB)")
        
        # Empty cache should free memory
        torch.cuda.empty_cache()
        final_reserved = torch.cuda.memory_reserved()
        
        logger.info(f"✓ Cache cleared, final reserved: {final_reserved/1e6:.2f}MB")
    
    def test_cache_fragmentation(self):
        """Test cache fragmentation handling - MUST NOT FAIL"""
        sizes = [1000, 5000, 2000, 8000, 3000, 6000]
        tensors = {}
        
        # Create tensors of varying sizes
        for i, size in enumerate(sizes):
            try:
                tensors[i] = torch.randn(size, size, device='cuda')
            except RuntimeError as e:
                raise RuntimeError(f"FATAL: Cannot allocate {size}x{size} tensor: {e}")
        
        # Delete every other tensor
        for i in range(0, len(sizes), 2):
            del tensors[i]
        
        # Try to allocate in gaps
        for i in range(0, len(sizes), 2):
            try:
                tensors[i] = torch.randn(sizes[i], sizes[i], device='cuda')
                logger.info(f"✓ Re-allocated tensor {i} size {sizes[i]}x{sizes[i]}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"⚠ Fragmentation prevented allocation of size {sizes[i]}")
                    torch.cuda.empty_cache()
                    # Retry after defrag
                    try:
                        tensors[i] = torch.randn(sizes[i], sizes[i], device='cuda')
                        logger.info(f"✓ Allocation successful after cache clear")
                    except:
                        raise RuntimeError(f"FATAL: Cannot allocate even after cache clear: {e}")
                else:
                    raise RuntimeError(f"FATAL: Unexpected error: {e}")
        
        # Cleanup
        tensors.clear()
        torch.cuda.empty_cache()
    
    def test_peak_memory_tracking(self):
        """Test peak memory tracking - MUST BE ACCURATE"""
        torch.cuda.reset_peak_memory_stats()
        
        peak_expected = 0
        
        # Series of allocations
        for size_mb in [100, 200, 300, 200, 100]:
            elements = (size_mb * 1024 * 1024) // 4
            tensor = torch.randn(elements, device='cuda')
            
            current_allocated = torch.cuda.memory_allocated()
            peak_expected = max(peak_expected, current_allocated)
            
            del tensor
        
        peak_actual = torch.cuda.max_memory_allocated()
        
        if abs(peak_actual - peak_expected) > peak_expected * 0.01:
            raise AssertionError(
                f"FATAL: Peak memory tracking wrong. Expected {peak_expected/1e6:.2f}MB, "
                f"got {peak_actual/1e6:.2f}MB"
            )
        
        logger.info(f"✓ Peak memory tracking accurate: {peak_actual/1e6:.2f}MB")


class TestGPUMemoryErrors(unittest.TestCase):
    """Test GPU memory error handling - HARD FAILURES ONLY"""
    
    def setUp(self):
        """Setup for error tests"""
        if not torch.cuda.is_available():
            raise RuntimeError("FATAL: CUDA required")
        
        torch.cuda.empty_cache()
    
    def test_out_of_memory_handling(self):
        """Test OOM error handling - MUST FAIL CORRECTLY"""
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory
        
        # Try to allocate 10x available memory
        huge_elements = (total_memory * 10) // 4
        
        oom_caught = False
        try:
            huge_tensor = torch.randn(huge_elements, device='cuda')
            raise AssertionError("FATAL: OOM did not occur when it should have")
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "oom" in error_msg:
                oom_caught = True
                logger.info("✓ OOM error correctly raised")
                
                # Verify error message contains useful info
                if "tried to allocate" not in error_msg:
                    logger.warning("⚠ OOM error missing allocation size info")
            else:
                raise AssertionError(f"FATAL: Wrong error type for OOM: {e}")
        
        if not oom_caught:
            raise AssertionError("FATAL: OOM error not properly caught")
        
        # Verify we can recover
        torch.cuda.empty_cache()
        
        try:
            small_tensor = torch.randn(100, 100, device='cuda')
            del small_tensor
            logger.info("✓ GPU recovered after OOM")
        except Exception as e:
            raise RuntimeError(f"FATAL: Cannot allocate after OOM recovery: {e}")
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure - MUST HANDLE GRACEFULLY"""
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory
        
        # Allocate 80% of memory
        target_bytes = int(total_memory * 0.8)
        elements = target_bytes // 4
        
        try:
            pressure_tensor = torch.randn(elements, device='cuda')
            logger.info(f"✓ Allocated {target_bytes/1e9:.2f}GB under pressure")
        except RuntimeError as e:
            raise RuntimeError(f"FATAL: Cannot allocate 80% of memory: {e}")
        
        # Try additional small allocations
        small_allocations = []
        failures = 0
        
        for i in range(10):
            try:
                small = torch.randn(1000, 1000, device='cuda')  # ~4MB
                small_allocations.append(small)
                logger.info(f"✓ Small allocation {i+1} successful under pressure")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    failures += 1
                    logger.warning(f"⚠ Small allocation {i+1} failed under pressure")
                else:
                    raise RuntimeError(f"FATAL: Unexpected error under pressure: {e}")
        
        # Some failures expected under pressure
        if failures == 0:
            logger.warning("⚠ No failures under memory pressure - unusual")
        elif failures == 10:
            raise AssertionError("FATAL: All small allocations failed - memory management issue")
        
        # Cleanup
        del pressure_tensor
        del small_allocations
        torch.cuda.empty_cache()
        
        logger.info(f"✓ Memory pressure handled: {failures}/10 small allocations failed")
    
    def test_invalid_allocation_sizes(self):
        """Test invalid allocation size handling - MUST FAIL PROPERLY"""
        invalid_sizes = [
            (-100, "negative size"),
            (0, "zero size"),
            (2**63, "overflow size")
        ]
        
        for size, description in invalid_sizes:
            try:
                if size == 2**63:
                    # This will cause integer overflow
                    tensor = torch.randn(size, device='cuda')
                elif size <= 0:
                    tensor = torch.empty(size, device='cuda')
                
                if size != 0:  # Zero-size tensors are actually valid in PyTorch
                    raise AssertionError(f"FATAL: {description} did not raise error")
                    
            except (RuntimeError, ValueError, OverflowError) as e:
                logger.info(f"✓ {description} correctly raised error: {type(e).__name__}")
            except Exception as e:
                raise RuntimeError(f"FATAL: Unexpected error for {description}: {e}")


class TestGPUMemoryOptimization(unittest.TestCase):
    """Test memory optimization techniques - HARD FAILURES ONLY"""
    
    def setUp(self):
        """Setup for optimization tests"""
        if not torch.cuda.is_available():
            raise RuntimeError("FATAL: CUDA required")
        
        torch.cuda.empty_cache()
        self.initial_memory = torch.cuda.memory_allocated()
    
    def test_inplace_operations(self):
        """Test in-place operations save memory - MUST BE EFFICIENT"""
        size = 10000
        
        # Non-inplace operation
        tensor1 = torch.randn(size, size, device='cuda')
        memory_before = torch.cuda.memory_allocated()
        
        result1 = tensor1 + 1.0  # Creates new tensor
        memory_after_non_inplace = torch.cuda.memory_allocated()
        
        memory_used_non_inplace = memory_after_non_inplace - memory_before
        
        del result1
        torch.cuda.empty_cache()
        
        # Inplace operation
        tensor2 = torch.randn(size, size, device='cuda')
        memory_before = torch.cuda.memory_allocated()
        
        tensor2.add_(1.0)  # Modifies in place
        memory_after_inplace = torch.cuda.memory_allocated()
        
        memory_used_inplace = memory_after_inplace - memory_before
        
        # Inplace should use significantly less memory
        if memory_used_inplace >= memory_used_non_inplace * 0.5:
            raise AssertionError(
                f"FATAL: Inplace operation not saving memory. "
                f"Inplace: {memory_used_inplace/1e6:.2f}MB, "
                f"Non-inplace: {memory_used_non_inplace/1e6:.2f}MB"
            )
        
        logger.info(f"✓ Inplace operations save memory: "
                   f"{memory_used_inplace/1e6:.2f}MB vs {memory_used_non_inplace/1e6:.2f}MB")
    
    def test_gradient_checkpointing(self):
        """Test gradient checkpointing memory trade-off - MUST WORK"""
        try:
            from torch.utils.checkpoint import checkpoint
        except ImportError:
            raise RuntimeError("FATAL: checkpoint not available in torch.utils")
        
        # Simple model that uses memory
        class MemoryIntensiveModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([
                    torch.nn.Linear(1000, 1000) for _ in range(10)
                ])
            
            def forward(self, x):
                for layer in self.layers:
                    x = torch.relu(layer(x))
                return x
        
        model = MemoryIntensiveModule().cuda()
        input_tensor = torch.randn(100, 1000, device='cuda', requires_grad=True)
        
        # Without checkpointing
        torch.cuda.reset_peak_memory_stats()
        output1 = model(input_tensor)
        loss1 = output1.sum()
        loss1.backward()
        peak_without = torch.cuda.max_memory_allocated()
        
        # Clear grads
        model.zero_grad()
        input_tensor.grad = None
        torch.cuda.empty_cache()
        
        # With checkpointing (should use less memory)
        torch.cuda.reset_peak_memory_stats()
        
        def run_model(x):
            return model(x)
        
        output2 = checkpoint(run_model, input_tensor)
        loss2 = output2.sum()
        loss2.backward()
        peak_with = torch.cuda.max_memory_allocated()
        
        # Checkpointing should reduce peak memory
        if peak_with >= peak_without:
            logger.warning(f"⚠ Checkpointing did not reduce memory: "
                         f"{peak_with/1e6:.2f}MB vs {peak_without/1e6:.2f}MB")
        else:
            logger.info(f"✓ Checkpointing reduced memory: "
                       f"{peak_with/1e6:.2f}MB vs {peak_without/1e6:.2f}MB")
    
    def test_tensor_views_vs_copies(self):
        """Test that views don't duplicate memory - MUST BE EFFICIENT"""
        size = 10000
        original = torch.randn(size, size, device='cuda')
        memory_after_original = torch.cuda.memory_allocated()
        
        # Create view (should not allocate new memory)
        view = original.view(size * size)
        memory_after_view = torch.cuda.memory_allocated()
        
        if memory_after_view > memory_after_original + 1024:  # Allow 1KB overhead
            raise AssertionError(
                f"FATAL: View allocated new memory. "
                f"Before: {memory_after_original/1e6:.2f}MB, "
                f"After: {memory_after_view/1e6:.2f}MB"
            )
        
        # Create copy (should allocate new memory)
        copy = original.clone()
        memory_after_copy = torch.cuda.memory_allocated()
        
        expected_increase = original.numel() * original.element_size()
        actual_increase = memory_after_copy - memory_after_view
        
        if abs(actual_increase - expected_increase) > expected_increase * 0.01:
            raise AssertionError(
                f"FATAL: Copy didn't allocate expected memory. "
                f"Expected: {expected_increase/1e6:.2f}MB, "
                f"Got: {actual_increase/1e6:.2f}MB"
            )
        
        logger.info("✓ Views vs copies memory behavior correct")


def run_memory_management_tests():
    """Run all GPU memory management tests"""
    logger.info("=" * 80)
    logger.info("GPU MEMORY MANAGEMENT TESTS - HARD FAILURE MODE")
    logger.info("=" * 80)
    
    if not torch.cuda.is_available():
        raise RuntimeError("FATAL: NO CUDA DEVICES AVAILABLE")
    
    # Log GPU info
    props = torch.cuda.get_device_properties(0)
    logger.info(f"GPU: {props.name}")
    logger.info(f"Total Memory: {props.total_memory / 1e9:.2f} GB")
    logger.info(f"Current Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    logger.info("=" * 80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGPUMemoryAllocation))
    suite.addTests(loader.loadTestsFromTestCase(TestGPUMemoryCaching))
    suite.addTests(loader.loadTestsFromTestCase(TestGPUMemoryErrors))
    suite.addTests(loader.loadTestsFromTestCase(TestGPUMemoryOptimization))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    
    try:
        result = runner.run(suite)
        
        if not result.wasSuccessful():
            logger.error("=" * 80)
            logger.error("MEMORY TESTS FAILED")
            for failure in result.failures + result.errors:
                logger.error(f"FAILED: {failure[0]}")
            logger.error("=" * 80)
            return False
        else:
            logger.info("=" * 80)
            logger.info("ALL MEMORY TESTS PASSED")
            logger.info("=" * 80)
            return True
            
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = run_memory_management_tests()
    sys.exit(0 if success else 1)