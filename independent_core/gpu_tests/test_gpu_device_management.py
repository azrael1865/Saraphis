"""
GPU Device Management Tests - Priority 1 (Critical)
Tests device detection, allocation, switching, and fallback mechanisms
HARD FAILURES ONLY - NO SILENT ERRORS
"""

import unittest
import torch
import logging
import gc
from typing import Optional, List, Tuple
import os
import sys
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestGPUDeviceDetection(unittest.TestCase):
    """Test GPU device detection and availability - HARD FAILURES ONLY"""
    
    @classmethod
    def setUpClass(cls):
        """Check if CUDA is available - FAIL HARD if not"""
        cls.cuda_available = torch.cuda.is_available()
        if not cls.cuda_available:
            raise RuntimeError("FATAL: CUDA NOT AVAILABLE - Cannot run GPU tests without NVIDIA GPU")
        
        cls.device_count = torch.cuda.device_count()
        if cls.device_count == 0:
            raise RuntimeError("FATAL: No CUDA devices found despite CUDA being available")
        
        cls.device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA VERIFIED: {cls.device_count} device(s) - {cls.device_name}")
    
    def test_cuda_availability(self):
        """Test CUDA availability detection - MUST PASS"""
        if not torch.cuda.is_available():
            raise AssertionError("FATAL: CUDA not available - GPU tests cannot proceed")
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise AssertionError("FATAL: CUDA available but no devices found")
        
        logger.info(f"✓ CUDA availability confirmed: {device_count} device(s)")
        self.assertGreater(device_count, 0, "FATAL: Must have at least 1 CUDA device")
    
    def test_device_properties(self):
        """Test GPU device property queries - HARD FAIL if properties invalid"""
        device_count = torch.cuda.device_count()
        
        for i in range(device_count):
            try:
                props = torch.cuda.get_device_properties(i)
            except Exception as e:
                raise RuntimeError(f"FATAL: Cannot get properties for device {i}: {e}")
            
            # HARD ASSERTIONS - must all pass
            if props.total_memory <= 0:
                raise AssertionError(f"FATAL: Device {i} reports invalid memory: {props.total_memory}")
            
            if props.major <= 0:
                raise AssertionError(f"FATAL: Device {i} reports invalid compute capability: {props.major}.{props.minor}")
            
            if not props.name:
                raise AssertionError(f"FATAL: Device {i} has no name")
            
            logger.info(f"✓ Device {i}: {props.name}, Memory: {props.total_memory / 1e9:.2f} GB, Compute: {props.major}.{props.minor}")
    
    def test_device_capability(self):
        """Test compute capability requirements - HARD FAIL if insufficient"""
        props = torch.cuda.get_device_properties(0)
        compute_capability = float(f"{props.major}.{props.minor}")
        
        # HARD REQUIREMENT: Minimum compute capability 3.5
        if compute_capability < 3.5:
            raise AssertionError(f"FATAL: Compute capability {compute_capability} < 3.5 minimum required")
        
        logger.info(f"✓ Compute capability {compute_capability} meets requirements")
        
        # Check for advanced features
        if compute_capability >= 7.0:
            logger.info("✓ GPU supports Tensor Cores")
        else:
            logger.warning("⚠ GPU does not support Tensor Cores (compute < 7.0)")
        
        if compute_capability >= 8.0:
            logger.info("✓ GPU supports BF16")
        else:
            logger.warning("⚠ GPU does not support BF16 (compute < 8.0)")
    
    def test_multi_gpu_detection(self):
        """Test multi-GPU setup detection - HARD FAIL on access errors"""
        device_count = torch.cuda.device_count()
        logger.info(f"Testing {device_count} GPU(s)")
        
        for i in range(device_count):
            try:
                device = torch.device(f'cuda:{i}')
                tensor = torch.ones(100, device=device)
            except Exception as e:
                raise RuntimeError(f"FATAL: Cannot create tensor on GPU {i}: {e}")
            
            if tensor.device.index != i:
                raise AssertionError(f"FATAL: Tensor created on wrong device. Expected {i}, got {tensor.device.index}")
            
            logger.info(f"✓ GPU {i} accessible and functional")
    
    def test_cuda_initialization(self):
        """Test CUDA context initialization - HARD FAIL if init fails"""
        try:
            torch.cuda.init()
        except Exception as e:
            raise RuntimeError(f"FATAL: CUDA initialization failed: {e}")
        
        if not torch.cuda.is_initialized():
            raise AssertionError("FATAL: CUDA failed to initialize despite no exceptions")
        
        # Test creating tensor on GPU - MUST SUCCEED
        try:
            tensor = torch.randn(100, 100, device='cuda')
        except Exception as e:
            raise RuntimeError(f"FATAL: Cannot create tensor after CUDA init: {e}")
        
        if tensor.device.type != 'cuda':
            raise AssertionError(f"FATAL: Tensor not on CUDA device: {tensor.device}")
        
        logger.info("✓ CUDA initialization successful")


class TestDeviceSwitching(unittest.TestCase):
    """Test device switching and tensor transfers - HARD FAILURES ONLY"""
    
    def setUp(self):
        """Set up test fixtures - FAIL HARD if CUDA not available"""
        if not torch.cuda.is_available():
            raise RuntimeError("FATAL: CUDA required for device switching tests")
        torch.cuda.empty_cache()
        
    def test_cpu_to_gpu_transfer(self):
        """Test transferring tensors from CPU to GPU - MUST SUCCEED"""
        # Create CPU tensor
        cpu_tensor = torch.randn(1000, 1000)
        if cpu_tensor.device.type != 'cpu':
            raise AssertionError(f"FATAL: Tensor not created on CPU: {cpu_tensor.device}")
        
        # Transfer to GPU - MUST SUCCEED
        try:
            gpu_tensor = cpu_tensor.cuda()
        except Exception as e:
            raise RuntimeError(f"FATAL: CPU to GPU transfer failed: {e}")
        
        if gpu_tensor.device.type != 'cuda':
            raise AssertionError(f"FATAL: Tensor not on GPU after .cuda(): {gpu_tensor.device}")
        
        # Verify data integrity - MUST BE EXACT
        cpu_verify = gpu_tensor.cpu()
        if not torch.allclose(cpu_tensor, cpu_verify, rtol=1e-5, atol=1e-8):
            max_diff = torch.max(torch.abs(cpu_tensor - cpu_verify)).item()
            raise AssertionError(f"FATAL: Data corruption during transfer. Max diff: {max_diff}")
        
        logger.info("✓ CPU to GPU transfer successful with data integrity")
    
    def test_gpu_to_cpu_transfer(self):
        """Test transferring tensors from GPU to CPU - MUST SUCCEED"""
        # Create GPU tensor
        try:
            gpu_tensor = torch.randn(1000, 1000, device='cuda')
        except Exception as e:
            raise RuntimeError(f"FATAL: Cannot create GPU tensor: {e}")
        
        if gpu_tensor.device.type != 'cuda':
            raise AssertionError(f"FATAL: Tensor not on GPU: {gpu_tensor.device}")
        
        # Transfer to CPU - MUST SUCCEED
        try:
            cpu_tensor = gpu_tensor.cpu()
        except Exception as e:
            raise RuntimeError(f"FATAL: GPU to CPU transfer failed: {e}")
        
        if cpu_tensor.device.type != 'cpu':
            raise AssertionError(f"FATAL: Tensor not on CPU after .cpu(): {cpu_tensor.device}")
        
        # Verify data integrity
        gpu_verify = cpu_tensor.cuda()
        if not torch.allclose(gpu_tensor, gpu_verify, rtol=1e-5, atol=1e-8):
            max_diff = torch.max(torch.abs(gpu_tensor - gpu_verify)).item()
            raise AssertionError(f"FATAL: Data corruption during transfer. Max diff: {max_diff}")
        
        logger.info("✓ GPU to CPU transfer successful with data integrity")
    
    def test_device_to_method(self):
        """Test the .to() method for device switching - ALL PATHS MUST WORK"""
        tensor = torch.randn(500, 500)
        
        # Test to CUDA - MUST SUCCEED
        try:
            cuda_tensor = tensor.to('cuda')
        except Exception as e:
            raise RuntimeError(f"FATAL: .to('cuda') failed: {e}")
        
        if cuda_tensor.device.type != 'cuda':
            raise AssertionError(f"FATAL: .to('cuda') didn't move to CUDA: {cuda_tensor.device}")
        
        # Test to specific GPU - MUST SUCCEED
        try:
            cuda_tensor = tensor.to('cuda:0')
        except Exception as e:
            raise RuntimeError(f"FATAL: .to('cuda:0') failed: {e}")
        
        if cuda_tensor.device.index != 0:
            raise AssertionError(f"FATAL: .to('cuda:0') on wrong device: {cuda_tensor.device.index}")
        
        # Test back to CPU - MUST SUCCEED
        try:
            cpu_tensor = cuda_tensor.to('cpu')
        except Exception as e:
            raise RuntimeError(f"FATAL: .to('cpu') failed: {e}")
        
        if cpu_tensor.device.type != 'cpu':
            raise AssertionError(f"FATAL: .to('cpu') didn't move to CPU: {cpu_tensor.device}")
        
        logger.info("✓ .to() method device switching working correctly")
    
    def test_device_mismatch_operations(self):
        """Test that device mismatches fail properly - MUST RAISE ERRORS"""
        cpu_tensor = torch.randn(100, 100)
        gpu_tensor = torch.randn(100, 100, device='cuda')
        
        # This MUST raise an error
        error_raised = False
        try:
            result = cpu_tensor + gpu_tensor
        except RuntimeError as e:
            error_raised = True
            if "must be on the same device" not in str(e):
                raise AssertionError(f"FATAL: Wrong error for device mismatch: {e}")
            logger.info("✓ Device mismatch correctly raised RuntimeError")
        
        if not error_raised:
            raise AssertionError("FATAL: Device mismatch did not raise error - this is a bug!")
        
        # Correct way - verify it works
        try:
            result = cpu_tensor.cuda() + gpu_tensor
        except Exception as e:
            raise RuntimeError(f"FATAL: Same-device operation failed: {e}")
        
        if result.device.type != 'cuda':
            raise AssertionError(f"FATAL: Result not on expected device: {result.device}")
        
        logger.info("✓ Device mismatch handling working correctly")
    
    def test_device_context_manager(self):
        """Test device context manager - MUST WORK CORRECTLY"""
        original_device = torch.cuda.current_device()
        
        try:
            with torch.cuda.device(0):
                tensor = torch.randn(100, 100, device='cuda')
                if tensor.device.index != 0:
                    raise AssertionError(f"FATAL: Context manager didn't set device 0: {tensor.device.index}")
            
            # Check device restored
            if torch.cuda.current_device() != original_device:
                raise AssertionError(f"FATAL: Context manager didn't restore device: {torch.cuda.current_device()}")
            
            logger.info("✓ Device context manager working correctly")
            
        except Exception as e:
            raise RuntimeError(f"FATAL: Device context manager failed: {e}")
        
        # Test with multiple GPUs if available
        if torch.cuda.device_count() > 1:
            try:
                with torch.cuda.device(1):
                    tensor = torch.randn(100, 100, device='cuda')
                    if tensor.device.index != 1:
                        raise AssertionError(f"FATAL: Context manager didn't set device 1: {tensor.device.index}")
                logger.info("✓ Multi-GPU context switching working")
            except Exception as e:
                raise RuntimeError(f"FATAL: Multi-GPU context failed: {e}")


class TestMemoryManagement(unittest.TestCase):
    """Test GPU memory management - HARD FAILURES ONLY"""
    
    def setUp(self):
        """Clear GPU cache before each test"""
        if not torch.cuda.is_available():
            raise RuntimeError("FATAL: CUDA required for memory tests")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    def test_memory_allocation(self):
        """Test GPU memory allocation tracking - MUST BE ACCURATE"""
        initial_memory = torch.cuda.memory_allocated()
        
        # Allocate known size
        size_mb = 100
        elements = (size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
        
        try:
            tensor = torch.randn(elements, device='cuda')
        except Exception as e:
            raise RuntimeError(f"FATAL: Cannot allocate {size_mb}MB tensor: {e}")
        
        allocated = torch.cuda.memory_allocated() - initial_memory
        expected = elements * 4  # float32 size
        
        # Allow 1% tolerance for overhead
        if abs(allocated - expected) > expected * 0.01:
            raise AssertionError(f"FATAL: Memory allocation mismatch. Expected ~{expected}, got {allocated}")
        
        logger.info(f"✓ Memory allocation tracking accurate: {allocated/1e6:.2f} MB")
        
        # Cleanup and verify
        del tensor
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated()
        if final_memory > initial_memory + 1024:  # Allow 1KB tolerance
            raise AssertionError(f"FATAL: Memory not freed. Leaked {final_memory - initial_memory} bytes")
        
        logger.info("✓ Memory cleanup successful")
    
    def test_memory_overflow_handling(self):
        """Test GPU memory overflow handling - MUST FAIL CORRECTLY"""
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory
        
        # Try to allocate more than available
        try:
            # Allocate 2x total memory (guaranteed to fail)
            elements = (total_memory * 2) // 4
            tensor = torch.randn(elements, device='cuda')
            raise AssertionError("FATAL: Memory overflow did not raise error - this is a bug!")
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise AssertionError(f"FATAL: Wrong error for memory overflow: {e}")
            logger.info("✓ Memory overflow correctly detected and reported")
        except Exception as e:
            raise RuntimeError(f"FATAL: Unexpected error during memory overflow test: {e}")
    
    def test_memory_fragmentation(self):
        """Test memory fragmentation handling - MUST HANDLE CORRECTLY"""
        # Create and delete many small tensors
        tensors = []
        try:
            for i in range(100):
                tensors.append(torch.randn(1000, 1000, device='cuda'))
                if i % 2 == 0 and i > 0:
                    # Delete every other tensor to create fragmentation
                    del tensors[i//2]
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"⚠ Memory fragmentation detected at iteration {i}")
            else:
                raise RuntimeError(f"FATAL: Unexpected error during fragmentation test: {e}")
        
        # Clear and verify
        del tensors
        torch.cuda.empty_cache()
        
        # Should be able to allocate after cleanup
        try:
            test_tensor = torch.randn(10000, 10000, device='cuda')
            del test_tensor
            logger.info("✓ Memory fragmentation handled correctly")
        except RuntimeError as e:
            raise AssertionError(f"FATAL: Cannot allocate after cleanup - memory fragmented: {e}")


class TestDeviceOperations(unittest.TestCase):
    """Test GPU operations correctness - HARD FAILURES ONLY"""
    
    def setUp(self):
        """Ensure CUDA available"""
        if not torch.cuda.is_available():
            raise RuntimeError("FATAL: CUDA required for operation tests")
    
    def test_computation_correctness(self):
        """Test that GPU computations match CPU exactly"""
        # Create identical tensors
        cpu_a = torch.randn(100, 100)
        cpu_b = torch.randn(100, 100)
        
        gpu_a = cpu_a.cuda()
        gpu_b = cpu_b.cuda()
        
        # Compute on both devices
        cpu_result = cpu_a @ cpu_b
        gpu_result = gpu_a @ gpu_b
        
        # Compare results
        gpu_on_cpu = gpu_result.cpu()
        
        if not torch.allclose(cpu_result, gpu_on_cpu, rtol=1e-5, atol=1e-7):
            max_diff = torch.max(torch.abs(cpu_result - gpu_on_cpu)).item()
            raise AssertionError(f"FATAL: GPU computation differs from CPU. Max diff: {max_diff}")
        
        logger.info("✓ GPU computations match CPU exactly")
    
    def test_synchronization_required(self):
        """Test that synchronization works correctly"""
        # Large computation that takes time
        size = 5000
        a = torch.randn(size, size, device='cuda')
        b = torch.randn(size, size, device='cuda')
        
        # Start async computation
        result = a @ b
        
        # Force synchronization
        torch.cuda.synchronize()
        
        # Result must be ready
        if result.shape != (size, size):
            raise AssertionError(f"FATAL: Result shape wrong after sync: {result.shape}")
        
        # Verify result is computed (check a value)
        try:
            value = result[0, 0].item()
            logger.info(f"✓ Synchronization working correctly, sample value: {value:.4f}")
        except Exception as e:
            raise RuntimeError(f"FATAL: Cannot access result after sync: {e}")


def run_all_gpu_tests():
    """Run all GPU tests with hard failures"""
    logger.info("=" * 80)
    logger.info("STARTING GPU TEST SUITE - HARD FAILURE MODE")
    logger.info("=" * 80)
    
    # Check CUDA first
    if not torch.cuda.is_available():
        raise RuntimeError("FATAL: CANNOT RUN GPU TESTS - NO CUDA DEVICES AVAILABLE")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGPUDeviceDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestDeviceSwitching))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestDeviceOperations))
    
    # Run with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    
    try:
        result = runner.run(suite)
        
        if not result.wasSuccessful():
            logger.error("=" * 80)
            logger.error("TESTS FAILED - CHECK ERRORS ABOVE")
            logger.error("=" * 80)
            
            # Print summary of failures
            for failure in result.failures:
                logger.error(f"FAILED: {failure[0]}")
                logger.error(f"  {failure[1]}")
            
            for error in result.errors:
                logger.error(f"ERROR: {error[0]}")
                logger.error(f"  {error[1]}")
            
            return False
        else:
            logger.info("=" * 80)
            logger.info("ALL GPU TESTS PASSED")
            logger.info("=" * 80)
            return True
            
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"FATAL ERROR DURING TEST EXECUTION: {e}")
        logger.error(traceback.format_exc())
        logger.error("=" * 80)
        return False


if __name__ == "__main__":
    success = run_all_gpu_tests()
    sys.exit(0 if success else 1)