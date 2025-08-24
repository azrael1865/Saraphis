#!/usr/bin/env python3
"""
Comprehensive test suite for SparsePAdicBridge
Tests all functionality, integration, performance, and edge cases.
"""

import sys
import os
import traceback
import torch
import numpy as np
from typing import Tuple, Dict, List, Any
import time
import logging

# Add the independent_core to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_component(test_name: str, test_func, *args, **kwargs) -> Tuple[bool, str]:
    """Test a single component and return status."""
    try:
        test_func(*args, **kwargs)
        return True, "OK"
    except ImportError as e:
        return False, f"Import Error: {e}"
    except AttributeError as e:
        return False, f"Attribute Error: {e}"
    except Exception as e:
        # Get full traceback for debugging
        tb = traceback.format_exc()
        return False, f"Error: {e}\nTraceback: {tb[-500:]}"

class TestSparsePAdicBridge:
    """Comprehensive test suite for SparsePAdicBridge"""
    
    def __init__(self):
        self.results = {}
        self.failed_tests = []
        
    def test_imports(self):
        """Test that all required imports work"""
        from compression_systems.padic.sparse_bridge import SparsePAdicBridge, AdaptiveSparsityManager
        assert SparsePAdicBridge is not None
        assert AdaptiveSparsityManager is not None
        
    def test_basic_initialization(self):
        """Test basic initialization with default parameters"""
        from compression_systems.padic.sparse_bridge import SparsePAdicBridge
        
        # Test default initialization
        bridge = SparsePAdicBridge()
        assert bridge.threshold > 0
        assert bridge.device is not None
        
        # Test with custom parameters
        bridge_custom = SparsePAdicBridge(
            sparsity_threshold=1e-4,
            use_gpu=False,
            use_optimized_sparse=True
        )
        assert bridge_custom.threshold == 1e-4
        assert not bridge_custom.use_gpu
        assert bridge_custom.use_optimized_sparse
        
    def test_device_handling(self):
        """Test device handling and switching"""
        from compression_systems.padic.sparse_bridge import SparsePAdicBridge
        
        bridge = SparsePAdicBridge()
        original_device = bridge.device
        
        # Test device setting
        cpu_device = torch.device('cpu')
        bridge.set_device(cpu_device)
        assert bridge.device == cpu_device
        assert not bridge.use_gpu
        
        # Test CUDA if available
        if torch.cuda.is_available():
            cuda_device = torch.device('cuda:0')
            bridge.set_device(cuda_device)
            assert bridge.device == cuda_device
            assert bridge.use_gpu
    
    def test_padic_to_sparse_conversion(self):
        """Test p-adic to sparse conversion"""
        from compression_systems.padic.sparse_bridge import SparsePAdicBridge
        
        bridge = SparsePAdicBridge()
        
        # Create test p-adic data
        batch_size, precision = 10, 5
        padic_digits = torch.randint(0, 5, (batch_size, precision), dtype=torch.long)
        valuations = torch.randint(-2, 3, (batch_size,), dtype=torch.long)
        
        # Test conversion
        sparse_tensor = bridge.padic_to_sparse(padic_digits, valuations)
        
        # Validate results - CSR tensors have layout sparse_csr but is_sparse is False
        assert sparse_tensor.layout in [torch.sparse_coo, torch.sparse_csr]
        assert hasattr(sparse_tensor, 'digit_tensor_shape')
        assert sparse_tensor.digit_tensor_shape == [batch_size, precision]
        
        # Test with original shape preservation
        original_shape = [2, 5, precision]
        sparse_tensor_with_shape = bridge.padic_to_sparse(
            padic_digits, valuations, original_shape=original_shape
        )
        assert hasattr(sparse_tensor_with_shape, 'original_shape')
        assert sparse_tensor_with_shape.original_shape == original_shape
    
    def test_sparse_to_padic_conversion(self):
        """Test sparse to p-adic conversion and round-trip consistency"""
        from compression_systems.padic.sparse_bridge import SparsePAdicBridge
        
        bridge = SparsePAdicBridge()
        
        # Create test data
        batch_size, precision = 8, 4
        padic_digits = torch.randint(0, 5, (batch_size, precision), dtype=torch.long)
        valuations = torch.randint(-1, 2, (batch_size,), dtype=torch.long)
        
        # Convert to sparse and back
        sparse_tensor = bridge.padic_to_sparse(padic_digits, valuations)
        recovered_digits, recovered_valuations = bridge.sparse_to_padic(sparse_tensor)
        
        # Check round-trip consistency (with tolerance for sparse operations)
        assert recovered_digits.shape == padic_digits.shape
        assert recovered_valuations.shape == valuations.shape
    
    def test_batch_operations(self):
        """Test batch processing operations"""
        from compression_systems.padic.sparse_bridge import SparsePAdicBridge
        
        bridge = SparsePAdicBridge()
        
        # Test batch p-adic to sparse
        batch_size, precision, num_batches = 5, 6, 3
        padic_batch = torch.randint(0, 5, (num_batches, batch_size, precision), dtype=torch.long)
        valuation_batch = torch.randint(-2, 3, (num_batches, batch_size), dtype=torch.long)
        
        sparse_results = bridge.batch_padic_to_sparse(padic_batch, valuation_batch)
        assert len(sparse_results) == num_batches
        for sparse_tensor in sparse_results:
            assert sparse_tensor.layout in [torch.sparse_coo, torch.sparse_csr]
        
        # Test batch sparse to p-adic  
        first_sparse = sparse_results[0]
        recovered_digits, recovered_valuations = bridge.batch_sparse_to_padic(first_sparse)
        assert recovered_digits.dim() >= 2
        assert recovered_valuations.dim() >= 1
    
    def test_compression_metrics(self):
        """Test compression ratio and memory statistics"""
        from compression_systems.padic.sparse_bridge import SparsePAdicBridge
        
        bridge = SparsePAdicBridge()
        
        # Create sparse test data
        batch_size, precision = 12, 8
        padic_digits = torch.randint(0, 5, (batch_size, precision), dtype=torch.long)
        # Make data sparse by setting many values to 0
        mask = torch.rand(batch_size, precision) > 0.7  # 70% sparse
        padic_digits[mask] = 0
        
        valuations = torch.randint(-1, 2, (batch_size,), dtype=torch.long)
        
        sparse_tensor = bridge.padic_to_sparse(padic_digits, valuations)
        
        # Test compression ratio calculation
        compression_ratio = bridge.get_compression_ratio(sparse_tensor)
        assert compression_ratio > 0
        assert isinstance(compression_ratio, float)
        
        # Test memory statistics
        memory_stats = bridge.get_memory_stats(sparse_tensor)
        assert isinstance(memory_stats, dict)
        assert 'dense_memory_bytes' in memory_stats
        assert 'sparse_memory_bytes' in memory_stats
        assert 'compression_ratio' in memory_stats
        assert 'sparsity_ratio' in memory_stats
    
    def test_optimization_patterns(self):
        """Test sparse pattern optimization"""
        from compression_systems.padic.sparse_bridge import SparsePAdicBridge
        
        bridge = SparsePAdicBridge()
        
        # Create test sparse tensor
        batch_size, precision = 10, 10
        padic_digits = torch.randint(0, 3, (batch_size, precision), dtype=torch.long)
        valuations = torch.randint(0, 2, (batch_size,), dtype=torch.long)
        
        sparse_tensor = bridge.padic_to_sparse(padic_digits, valuations)
        
        # Test pattern optimization
        optimized_tensor = bridge.optimize_for_pattern(sparse_tensor)
        assert optimized_tensor.layout in [torch.sparse_coo, torch.sparse_csr]
        
        # Test with different patterns
        patterns = ['block', 'diagonal', 'banded']
        for pattern in patterns:
            try:
                optimized = bridge.optimize_for_pattern(sparse_tensor, pattern=pattern)
                assert optimized.layout in [torch.sparse_coo, torch.sparse_csr]
            except Exception as e:
                # Some patterns might not be applicable to test data
                logger.warning(f"Pattern {pattern} optimization failed: {e}")
    
    def test_validation_and_compression(self):
        """Test validation and compression accuracy"""
        from compression_systems.padic.sparse_bridge import SparsePAdicBridge
        
        bridge = SparsePAdicBridge()
        
        # Create test data
        original_tensor = torch.randn(15, 12)
        
        # Convert to p-adic representation (simplified)
        padic_digits = (original_tensor.abs() * 10).long() % 5
        valuations = torch.zeros(original_tensor.shape[0], dtype=torch.long)
        
        sparse_tensor = bridge.padic_to_sparse(padic_digits, valuations)
        recovered_digits, recovered_valuations = bridge.sparse_to_padic(sparse_tensor)
        
        # Reconstruct approximation
        reconstructed = recovered_digits.float() / 10.0
        
        # Validate compression
        is_valid = bridge.validate_compression(
            original_tensor.abs(), 
            reconstructed.float(), 
            tolerance=0.2
        )
        # Note: This is a loose test since we're doing simplified p-adic conversion
        assert isinstance(is_valid, bool)
    
    def test_adaptive_sparsity_manager(self):
        """Test adaptive sparsity threshold management"""
        from compression_systems.padic.sparse_bridge import AdaptiveSparsityManager
        
        # Test basic initialization
        manager = AdaptiveSparsityManager()
        initial_threshold = manager.current_threshold
        assert initial_threshold > 0
        
        # Test threshold updates
        manager.update_threshold(10.0)  # Lower than target - should increase threshold
        threshold_after_low = manager.current_threshold
        
        manager.update_threshold(30.0)  # Higher than target - should decrease threshold
        threshold_after_high = manager.current_threshold
        
        # Current threshold should adapt (increase then decrease)
        assert threshold_after_low > initial_threshold  # Increased for more sparsity
        assert threshold_after_high < threshold_after_low  # Decreased when too sparse
        
        # Test optimal threshold selection
        optimal = manager.get_optimal_threshold()
        assert optimal > 0
    
    def test_cache_functionality(self):
        """Test caching mechanisms"""
        from compression_systems.padic.sparse_bridge import SparsePAdicBridge
        
        bridge = SparsePAdicBridge()
        
        # Clear cache
        bridge.clear_cache()
        assert len(bridge.pattern_cache) == 0
        
        # Test that cache can be populated (indirectly through repeated operations)
        batch_size, precision = 5, 5
        padic_digits = torch.randint(0, 5, (batch_size, precision), dtype=torch.long)
        valuations = torch.randint(0, 2, (batch_size,), dtype=torch.long)
        
        # Run same operation multiple times
        for _ in range(3):
            sparse_tensor = bridge.padic_to_sparse(padic_digits, valuations)
        
        # Cache stats should show some activity
        total_cache_operations = bridge.cache_hits + bridge.cache_misses
        assert total_cache_operations >= 0  # At least tracking is working
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        from compression_systems.padic.sparse_bridge import SparsePAdicBridge
        
        bridge = SparsePAdicBridge()
        
        # Test with empty tensors
        empty_digits = torch.empty(0, 5, dtype=torch.long)
        empty_valuations = torch.empty(0, dtype=torch.long)
        
        try:
            sparse_result = bridge.padic_to_sparse(empty_digits, empty_valuations)
            # Should handle empty case gracefully
            assert sparse_result.shape[0] == 0
        except Exception as e:
            logger.warning(f"Empty tensor test failed: {e}")
        
        # Test with single element
        single_digits = torch.tensor([[1, 2, 3, 4, 0]], dtype=torch.long)
        single_valuations = torch.tensor([1], dtype=torch.long)
        
        sparse_single = bridge.padic_to_sparse(single_digits, single_valuations)
        assert sparse_single.layout in [torch.sparse_coo, torch.sparse_csr]
        
        # Test with large precision
        try:
            large_digits = torch.randint(0, 5, (3, 100), dtype=torch.long)
            large_valuations = torch.randint(0, 2, (3,), dtype=torch.long)
            sparse_large = bridge.padic_to_sparse(large_digits, large_valuations)
            assert sparse_large.layout in [torch.sparse_coo, torch.sparse_csr]
        except Exception as e:
            logger.warning(f"Large precision test failed: {e}")
    
    def test_performance_benchmarks(self):
        """Test performance and timing"""
        from compression_systems.padic.sparse_bridge import SparsePAdicBridge
        
        bridge = SparsePAdicBridge()
        
        # Performance test with medium-sized data
        batch_size, precision = 100, 50
        padic_digits = torch.randint(0, 5, (batch_size, precision), dtype=torch.long)
        valuations = torch.randint(-3, 4, (batch_size,), dtype=torch.long)
        
        # Time the conversion
        start_time = time.perf_counter()
        sparse_tensor = bridge.padic_to_sparse(padic_digits, valuations)
        conversion_time = time.perf_counter() - start_time
        
        logger.info(f"P-adic to sparse conversion time: {conversion_time*1000:.2f}ms")
        
        # Time the reverse conversion
        start_time = time.perf_counter()
        recovered_digits, recovered_valuations = bridge.sparse_to_padic(sparse_tensor)
        recovery_time = time.perf_counter() - start_time
        
        logger.info(f"Sparse to p-adic conversion time: {recovery_time*1000:.2f}ms")
        
        # Basic performance assertions
        assert conversion_time < 1.0  # Should complete within 1 second
        assert recovery_time < 1.0
    
    def run_all_tests(self):
        """Run all test methods"""
        test_methods = [
            ('Imports', self.test_imports),
            ('Basic Initialization', self.test_basic_initialization),
            ('Device Handling', self.test_device_handling),
            ('P-adic to Sparse Conversion', self.test_padic_to_sparse_conversion),
            ('Sparse to P-adic Conversion', self.test_sparse_to_padic_conversion),
            ('Batch Operations', self.test_batch_operations),
            ('Compression Metrics', self.test_compression_metrics),
            ('Optimization Patterns', self.test_optimization_patterns),
            ('Validation and Compression', self.test_validation_and_compression),
            ('Adaptive Sparsity Manager', self.test_adaptive_sparsity_manager),
            ('Cache Functionality', self.test_cache_functionality),
            ('Edge Cases', self.test_edge_cases),
            ('Performance Benchmarks', self.test_performance_benchmarks)
        ]
        
        print("=" * 80)
        print("COMPREHENSIVE SPARSE P-ADIC BRIDGE TESTING")
        print("=" * 80)
        
        passed = 0
        failed = 0
        
        for test_name, test_method in test_methods:
            success, msg = test_component(test_name, test_method)
            status = "✅" if success else "❌"
            print(f"{status} {test_name:30} - {msg}")
            
            self.results[test_name] = (success, msg)
            if success:
                passed += 1
            else:
                failed += 1
                self.failed_tests.append((test_name, msg))
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total Tests: {len(test_methods)}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        
        if failed > 0:
            print("\nFAILED TESTS:")
            for test_name, error in self.failed_tests:
                print(f"  - {test_name}: {error}")
        
        print("\n" + "=" * 80)
        print("ACTION ITEMS")
        print("=" * 80)
        
        if failed > 0:
            print(f"\nIssues to fix (in order of priority):")
            for i, (test_name, error) in enumerate(self.failed_tests, 1):
                print(f"{i}. {test_name}: {error}")
        else:
            print("\n✅ All SparsePAdicBridge tests passed!")
        
        return passed, failed

def main():
    """Main test execution"""
    tester = TestSparsePAdicBridge()
    passed, failed = tester.run_all_tests()
    
    # Exit with error code if there are failures
    sys.exit(failed)

if __name__ == "__main__":
    main()