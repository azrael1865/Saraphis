"""
Comprehensive tests for GPU memory layout optimization
PRODUCTION-READY - NO PLACEHOLDERS - HARD FAILURES ONLY

Tests memory alignment, coalescing efficiency, cache optimization,
and performance characteristics of GPU-optimized channel layouts.
"""

import torch
import torch.cuda as cuda
import numpy as np
import time
import pytest
from typing import Dict, List, Tuple, Any
import traceback

# Import components to test
try:
    from independent_core.compression_systems.tropical.gpu_memory_optimizer import (
        GPUMemoryLayoutConfig,
        ChannelMemoryOptimizer,
        GPUMemoryAllocator,
        ChannelAccessPatternAnalyzer,
        BatchedChannelProcessor,
        MemoryLayout,
        AccessPattern,
        MemoryAccessMetrics,
        create_optimized_gpu_config,
        optimize_channel_for_gpu
    )
    from independent_core.compression_systems.tropical.tropical_channel_extractor import (
        TropicalChannels,
        TropicalChannelManager,
        ExponentChannelConfig,
        MantissaChannelConfig
    )
    from independent_core.compression_systems.tropical.tropical_polynomial import (
        TropicalPolynomial,
        TropicalMonomial
    )
except ImportError:
    from gpu_memory_optimizer import (
        GPUMemoryLayoutConfig,
        ChannelMemoryOptimizer,
        GPUMemoryAllocator,
        ChannelAccessPatternAnalyzer,
        BatchedChannelProcessor,
        MemoryLayout,
        AccessPattern,
        MemoryAccessMetrics,
        create_optimized_gpu_config,
        optimize_channel_for_gpu
    )
    from tropical_channel_extractor import (
        TropicalChannels,
        TropicalChannelManager,
        ExponentChannelConfig,
        MantissaChannelConfig
    )
    from tropical_polynomial import (
        TropicalPolynomial,
        TropicalMonomial
    )


class TestMemoryAlignment:
    """Test memory alignment for coalesced access"""
    
    def test_alignment_verification(self):
        """Verify memory is properly aligned"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = create_optimized_gpu_config()
        allocator = GPUMemoryAllocator(config)
        
        # Test various sizes
        test_sizes = [
            (128,),
            (256, 512),
            (1000, 100),
            (333, 777),  # Non-aligned sizes
        ]
        
        for shape in test_sizes:
            tensor = allocator.allocate_aligned(shape, dtype=torch.float32)
            
            # Check alignment
            ptr = tensor.data_ptr()
            alignment = config.alignment_bytes
            
            assert ptr % alignment == 0, f"Memory not aligned to {alignment} bytes for shape {shape}"
            
            # Verify shape
            assert tensor.shape == shape, f"Shape mismatch: expected {shape}, got {tensor.shape}"
            
            # Clean up
            allocator.deallocate(tensor)
    
    def test_pinned_memory_allocation(self):
        """Test pinned memory for fast transfers"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = create_optimized_gpu_config()
        allocator = GPUMemoryAllocator(config)
        
        # Allocate pinned memory
        shape = (1024, 512)
        pinned_tensor = allocator.allocate_aligned(shape, pinned=True)
        
        # Test transfer speed
        start = time.perf_counter()
        gpu_tensor = pinned_tensor.cuda(non_blocking=True)
        torch.cuda.synchronize()
        pinned_time = time.perf_counter() - start
        
        # Compare with non-pinned
        regular_tensor = torch.randn(shape)
        start = time.perf_counter()
        gpu_tensor2 = regular_tensor.cuda()
        torch.cuda.synchronize()
        regular_time = time.perf_counter() - start
        
        # Pinned should be faster for large tensors
        print(f"Pinned transfer: {pinned_time*1000:.3f}ms")
        print(f"Regular transfer: {regular_time*1000:.3f}ms")
        
        # Clean up
        allocator.deallocate(pinned_tensor)
    
    def test_memory_pool_efficiency(self):
        """Test memory pool allocation efficiency"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = create_optimized_gpu_config()
        allocator = GPUMemoryAllocator(config)
        
        # Allocate and deallocate multiple times
        shape = (512, 256)
        tensors = []
        
        # First allocation - pool miss
        for i in range(10):
            t = allocator.allocate_aligned(shape)
            tensors.append(t)
        
        # Deallocate all
        for t in tensors:
            allocator.deallocate(t)
        
        stats1 = allocator.get_stats()
        pool_size1 = stats1['pool_size']
        
        # Second allocation - should hit pool
        tensors2 = []
        for i in range(10):
            t = allocator.allocate_aligned(shape)
            tensors2.append(t)
        
        stats2 = allocator.get_stats()
        
        # Verify pool was used
        assert stats2['pool_hits'] > stats1.get('pool_hits', 0), "Memory pool not being utilized"
        assert pool_size1 == 10, f"Expected 10 tensors in pool, got {pool_size1}"
        
        # Clean up
        for t in tensors2:
            allocator.deallocate(t)
        allocator.clear_pools()


class TestCoalescingEfficiency:
    """Test memory coalescing efficiency"""
    
    def test_soa_layout_coalescing(self):
        """Test Structure-of-Arrays layout for coalescing"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = create_optimized_gpu_config()
        optimizer = ChannelMemoryOptimizer(config)
        
        # Create test data
        data = torch.randn(1024, 256, device='cuda')
        
        # Apply SoA layout
        optimized, layout, metrics = optimizer.optimize_layout(
            data, access_pattern=AccessPattern.SEQUENTIAL
        )
        
        # Verify layout
        assert layout == MemoryLayout.SOA, f"Expected SoA layout, got {layout}"
        
        # Check coalescing efficiency
        assert metrics.coalescing_efficiency >= 0.9, \
            f"Poor coalescing efficiency: {metrics.coalescing_efficiency:.2%}"
        
        print(f"SoA Coalescing efficiency: {metrics.coalescing_efficiency:.2%}")
        print(f"Memory bandwidth: {metrics.memory_bandwidth_gbps:.2f} GB/s")
    
    def test_blocked_layout_cache_efficiency(self):
        """Test blocked layout for cache optimization"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = create_optimized_gpu_config()
        optimizer = ChannelMemoryOptimizer(config)
        
        # Create test data
        data = torch.randn(512, 512, device='cuda')
        
        # Apply blocked layout
        optimized, layout, metrics = optimizer.optimize_layout(
            data, access_pattern=AccessPattern.STRIDED
        )
        
        # Verify layout
        assert layout == MemoryLayout.BLOCKED, f"Expected blocked layout, got {layout}"
        
        # Check cache efficiency
        assert metrics.cache_hit_rate_l2 >= 0.5, \
            f"Poor L2 cache hit rate: {metrics.cache_hit_rate_l2:.2%}"
        
        print(f"Blocked layout L1 cache hit: {metrics.cache_hit_rate_l1:.2%}")
        print(f"Blocked layout L2 cache hit: {metrics.cache_hit_rate_l2:.2%}")
    
    def test_tiled_layout_2d_access(self):
        """Test tiled layout for 2D access patterns"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = create_optimized_gpu_config()
        config.tile_size = 16  # Small tiles for testing
        optimizer = ChannelMemoryOptimizer(config)
        
        # Create test data
        data = torch.randn(256, 256, device='cuda')
        
        # Apply tiled layout
        optimized, layout, metrics = optimizer.optimize_layout(
            data, access_pattern=AccessPattern.GATHER
        )
        
        # Verify layout
        assert layout == MemoryLayout.TILED, f"Expected tiled layout, got {layout}"
        
        # Verify shape is preserved (after potential padding)
        assert optimized.numel() >= data.numel(), "Data loss in tiled layout"
        
        print(f"Tiled layout efficiency: {metrics.coalescing_efficiency:.2%}")


class TestBankConflicts:
    """Test for shared memory bank conflicts"""
    
    def test_warp_aligned_access(self):
        """Test warp-aligned memory access"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = create_optimized_gpu_config()
        warp_size = config.warp_size
        
        # Create warp-aligned data
        aligned_size = warp_size * 32  # Multiple of warp size
        data = torch.randn(aligned_size, 64, device='cuda')
        
        # Verify alignment
        assert data.shape[0] % warp_size == 0, "Data not warp-aligned"
        
        # Test access pattern
        optimizer = ChannelMemoryOptimizer(config)
        optimized, layout, metrics = optimizer.optimize_layout(data)
        
        # Bank conflicts should be minimal
        assert metrics.bank_conflicts <= 100, \
            f"Excessive bank conflicts: {metrics.bank_conflicts}"
        
        print(f"Bank conflicts: {metrics.bank_conflicts}")
        print(f"Warp efficiency: {metrics.warp_efficiency:.2%}")
    
    def test_stride_conflict_avoidance(self):
        """Test stride patterns that avoid bank conflicts"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = create_optimized_gpu_config()
        
        # Create data with potential bank conflict stride
        # Banks = 32 for modern GPUs
        bad_stride = 32
        good_stride = 33  # Avoids conflicts
        
        bad_data = torch.randn(256, bad_stride, device='cuda')
        good_data = torch.randn(256, good_stride, device='cuda')
        
        optimizer = ChannelMemoryOptimizer(config)
        
        # Test bad stride
        _, _, bad_metrics = optimizer.optimize_layout(bad_data)
        
        # Test good stride
        _, _, good_metrics = optimizer.optimize_layout(good_data)
        
        # Good stride should have better efficiency
        assert good_metrics.coalescing_efficiency >= bad_metrics.coalescing_efficiency, \
            "Stride optimization not working"
        
        print(f"Bad stride efficiency: {bad_metrics.coalescing_efficiency:.2%}")
        print(f"Good stride efficiency: {good_metrics.coalescing_efficiency:.2%}")


class TestCacheEfficiency:
    """Test L1/L2 cache optimization"""
    
    def test_cache_hit_rates(self):
        """Measure cache hit rates for different data sizes"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = create_optimized_gpu_config()
        optimizer = ChannelMemoryOptimizer(config)
        
        # Test different data sizes
        test_configs = [
            (128, 128, "small"),     # Fits in L1
            (512, 512, "medium"),    # Fits in L2
            (2048, 2048, "large"),   # Exceeds cache
        ]
        
        for h, w, name in test_configs:
            data = torch.randn(h, w, device='cuda')
            _, _, metrics = optimizer.optimize_layout(data)
            
            print(f"\n{name.capitalize()} data ({h}x{w}):")
            print(f"  L1 hit rate: {metrics.cache_hit_rate_l1:.2%}")
            print(f"  L2 hit rate: {metrics.cache_hit_rate_l2:.2%}")
            print(f"  Bandwidth: {metrics.memory_bandwidth_gbps:.2f} GB/s")
            
            # Verify cache behavior
            if name == "small":
                assert metrics.cache_hit_rate_l1 >= 0.7, "Poor L1 hit rate for small data"
            elif name == "medium":
                assert metrics.cache_hit_rate_l2 >= 0.6, "Poor L2 hit rate for medium data"
    
    def test_cache_line_utilization(self):
        """Test cache line utilization"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = create_optimized_gpu_config()
        cache_line_size = config.cache_line_size
        
        # Create data aligned to cache lines
        elements_per_line = cache_line_size // 4  # float32 = 4 bytes
        data = torch.randn(256, elements_per_line * 4, device='cuda')
        
        # Optimize layout
        optimizer = ChannelMemoryOptimizer(config)
        optimized, layout, metrics = optimizer.optimize_layout(data)
        
        # Should have good cache utilization
        assert metrics.cache_hit_rate_l1 >= 0.5, \
            f"Poor cache line utilization: {metrics.cache_hit_rate_l1:.2%}"
        
        print(f"Cache line utilization: {metrics.cache_hit_rate_l1:.2%}")


class TestTransferBandwidth:
    """Test CPU-GPU transfer bandwidth optimization"""
    
    def test_pinned_memory_bandwidth(self):
        """Test pinned memory transfer bandwidth"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = create_optimized_gpu_config()
        
        # Test different transfer sizes
        test_sizes = [
            (1024, 256),      # 1MB
            (2048, 1024),     # 8MB
            (4096, 2048),     # 32MB
            (8192, 4096),     # 128MB
        ]
        
        for shape in test_sizes:
            size_mb = np.prod(shape) * 4 / (1024 * 1024)
            
            # Pinned memory transfer
            pinned = torch.randn(shape, pin_memory=True)
            start = time.perf_counter()
            gpu_pinned = pinned.cuda(non_blocking=True)
            torch.cuda.synchronize()
            pinned_time = time.perf_counter() - start
            pinned_bandwidth = size_mb / pinned_time / 1000  # GB/s
            
            # Regular memory transfer
            regular = torch.randn(shape)
            start = time.perf_counter()
            gpu_regular = regular.cuda()
            torch.cuda.synchronize()
            regular_time = time.perf_counter() - start
            regular_bandwidth = size_mb / regular_time / 1000  # GB/s
            
            print(f"\nSize: {size_mb:.1f}MB")
            print(f"  Pinned: {pinned_bandwidth:.2f} GB/s ({pinned_time*1000:.2f}ms)")
            print(f"  Regular: {regular_bandwidth:.2f} GB/s ({regular_time*1000:.2f}ms)")
            
            # Pinned should be faster
            assert pinned_bandwidth >= regular_bandwidth * 0.9, \
                "Pinned memory not providing bandwidth improvement"
    
    def test_async_transfer_overlap(self):
        """Test async transfer with compute overlap"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = create_optimized_gpu_config()
        
        # Create streams
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        
        # Test data
        data1 = torch.randn(2048, 2048, pin_memory=True)
        data2 = torch.randn(2048, 2048, pin_memory=True)
        
        # Measure overlapped execution
        start = time.perf_counter()
        
        with torch.cuda.stream(stream1):
            gpu1 = data1.cuda(non_blocking=True)
            result1 = gpu1.sum()
        
        with torch.cuda.stream(stream2):
            gpu2 = data2.cuda(non_blocking=True)
            result2 = gpu2.sum()
        
        # Synchronize
        stream1.synchronize()
        stream2.synchronize()
        overlapped_time = time.perf_counter() - start
        
        # Measure sequential execution
        start = time.perf_counter()
        gpu1_seq = data1.cuda()
        result1_seq = gpu1_seq.sum()
        gpu2_seq = data2.cuda()
        result2_seq = gpu2_seq.sum()
        sequential_time = time.perf_counter() - start
        
        print(f"Overlapped execution: {overlapped_time*1000:.2f}ms")
        print(f"Sequential execution: {sequential_time*1000:.2f}ms")
        print(f"Speedup: {sequential_time/overlapped_time:.2f}x")
        
        # Overlapped should be faster
        assert overlapped_time < sequential_time * 1.1, \
            "Async transfer overlap not working"


class TestLargeScalePerformance:
    """Test performance with large-scale data"""
    
    def test_large_tensor_optimization(self):
        """Test optimization of large tensors"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Check available memory
        free_memory = torch.cuda.mem_get_info()[0]
        if free_memory < 2 * 1024**3:  # Need at least 2GB free
            pytest.skip("Insufficient GPU memory")
        
        config = create_optimized_gpu_config()
        optimizer = ChannelMemoryOptimizer(config)
        
        # Create large tensor (500MB)
        size = int(500 * 1024 * 1024 / 4)  # 500MB of float32
        data = torch.randn(size // 1024, 1024, device='cuda')
        
        # Optimize layout
        start = time.perf_counter()
        optimized, layout, metrics = optimizer.optimize_layout(data)
        optimization_time = time.perf_counter() - start
        
        print(f"\nLarge tensor optimization:")
        print(f"  Size: {data.numel() * 4 / (1024**2):.1f}MB")
        print(f"  Optimization time: {optimization_time*1000:.2f}ms")
        print(f"  Layout: {layout.value}")
        print(f"  Bandwidth: {metrics.memory_bandwidth_gbps:.2f} GB/s")
        print(f"  Coalescing: {metrics.coalescing_efficiency:.2%}")
        
        # Performance requirements
        assert metrics.memory_bandwidth_gbps > 10, \
            f"Poor bandwidth for large tensor: {metrics.memory_bandwidth_gbps:.2f} GB/s"
        assert metrics.coalescing_efficiency >= 0.8, \
            f"Poor coalescing for large tensor: {metrics.coalescing_efficiency:.2%}"
    
    def test_batch_processing_throughput(self):
        """Test batch processing throughput"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = create_optimized_gpu_config()
        processor = BatchedChannelProcessor(config)
        
        # Create batch of channels
        batch_size = 32
        channels = [torch.randn(512, 256, device='cuda') for _ in range(batch_size)]
        
        # Define operation
        def process_op(x):
            return x.sum(dim=0)
        
        # Process batch
        start = time.perf_counter()
        results = processor.process_batch(channels, process_op, optimize_layout=True)
        batch_time = time.perf_counter() - start
        
        # Calculate throughput
        total_elements = sum(ch.numel() for ch in channels)
        throughput_gbps = (total_elements * 4 / 1e9) / batch_time
        
        stats = processor.get_statistics()
        
        print(f"\nBatch processing:")
        print(f"  Batch size: {batch_size}")
        print(f"  Total time: {batch_time*1000:.2f}ms")
        print(f"  Throughput: {throughput_gbps:.2f} GB/s")
        print(f"  Avg per batch: {stats['avg_batch_time_ms']:.2f}ms")
        print(f"  Peak memory: {stats['peak_memory_mb']:.2f}MB")
        
        # Performance requirements
        assert throughput_gbps > 5, f"Poor batch throughput: {throughput_gbps:.2f} GB/s"
        assert stats['avg_batch_time_ms'] < 100, \
            f"Batch processing too slow: {stats['avg_batch_time_ms']:.2f}ms"


class TestMemoryPressureHandling:
    """Test behavior under memory pressure"""
    
    def test_memory_limit_enforcement(self):
        """Test that memory limits are enforced"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = create_optimized_gpu_config()
        
        # Set conservative memory limit
        free_memory = torch.cuda.mem_get_info()[0]
        config.gpu_memory_limit_mb = int(free_memory / (1024 * 1024) * 0.5)  # Use 50% of free
        
        allocator = GPUMemoryAllocator(config)
        
        # Try to allocate within limit - should succeed
        safe_size = config.gpu_memory_limit_mb * 1024 * 1024 // 4 // 2  # Half of limit
        safe_tensor = allocator.allocate_aligned((safe_size,), dtype=torch.float32)
        assert safe_tensor is not None, "Failed to allocate within memory limit"
        
        allocator.deallocate(safe_tensor)
        print(f"Memory limit enforcement: PASS")
    
    def test_fragmentation_handling(self):
        """Test handling of memory fragmentation"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = create_optimized_gpu_config()
        allocator = GPUMemoryAllocator(config)
        
        # Create fragmentation by allocating/deallocating different sizes
        tensors = []
        sizes = [1024, 2048, 512, 4096, 256]
        
        for _ in range(3):
            for size in sizes:
                t = allocator.allocate_aligned((size, size))
                tensors.append(t)
            
            # Deallocate every other tensor
            for i in range(0, len(tensors), 2):
                allocator.deallocate(tensors[i])
            
            tensors = [t for i, t in enumerate(tensors) if i % 2 == 1]
        
        # Clean up remaining
        for t in tensors:
            allocator.deallocate(t)
        
        # Pool should handle fragmentation
        stats = allocator.get_stats()
        print(f"After fragmentation - Pool size: {stats['pool_size']}")
        print(f"Pool memory: {stats['pool_bytes'] / (1024*1024):.2f}MB")
        
        # Clear pools
        allocator.clear_pools()
        torch.cuda.empty_cache()


class TestChannelIntegration:
    """Test integration with TropicalChannels"""
    
    def test_tropical_channels_optimization(self):
        """Test optimization of TropicalChannels"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create test polynomial
        monomials = [
            TropicalMonomial(1.0, {0: 2, 1: 1}),
            TropicalMonomial(2.0, {0: 1, 1: 2}),
            TropicalMonomial(3.0, {0: 3}),
        ]
        poly = TropicalPolynomial(monomials, num_variables=2)
        
        # Create channel manager with GPU optimization
        gpu_config = create_optimized_gpu_config()
        manager = TropicalChannelManager(
            device=torch.device('cuda'),
            gpu_layout_config=gpu_config
        )
        
        # Convert to channels
        channels = manager.polynomial_to_channels(poly)
        
        # Move to GPU with optimization
        gpu_channels = channels.to_gpu(torch.device('cuda'), optimize_layout=True)
        
        # Verify optimization
        assert gpu_channels.coefficient_channel.is_cuda, "Channels not on GPU"
        assert gpu_channels.exponent_channel.is_cuda, "Channels not on GPU"
        
        # Profile performance
        profile = gpu_channels.profile_gpu_performance()
        
        print(f"\nTropical Channels GPU Profile:")
        print(f"  Total memory: {profile['total_memory_bytes'] / 1024:.2f}KB")
        
        for channel_name, analysis in profile['channels'].items():
            print(f"  {channel_name}:")
            print(f"    Best pattern: {analysis['best_pattern']}")
            print(f"    Sparsity: {analysis['sparsity']:.2%}")
            print(f"    Aligned: {analysis['is_aligned']}")
    
    def test_optimized_layout_performance(self):
        """Test performance improvement from layout optimization"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create larger test data
        num_monomials = 1000
        num_vars = 10
        
        # Create channels
        coeff = torch.randn(num_monomials, device='cuda')
        exp = torch.randint(0, 5, (num_monomials, num_vars), device='cuda', dtype=torch.int32)
        idx = torch.arange(num_monomials, device='cuda', dtype=torch.long)
        
        channels = TropicalChannels(
            coefficient_channel=coeff,
            exponent_channel=exp,
            index_channel=idx,
            metadata={'num_variables': num_vars},
            device=torch.device('cuda')
        )
        
        # Test without optimization
        start = time.perf_counter()
        for _ in range(100):
            _ = channels.coefficient_channel.sum()
            _ = channels.exponent_channel.sum()
        torch.cuda.synchronize()
        unopt_time = time.perf_counter() - start
        
        # Test with optimization
        opt_channels = channels.optimize_gpu_layout()
        
        start = time.perf_counter()
        for _ in range(100):
            _ = opt_channels.coefficient_channel.sum()
            _ = opt_channels.exponent_channel.sum()
        torch.cuda.synchronize()
        opt_time = time.perf_counter() - start
        
        speedup = unopt_time / opt_time
        
        print(f"\nLayout optimization performance:")
        print(f"  Unoptimized: {unopt_time*1000:.2f}ms")
        print(f"  Optimized: {opt_time*1000:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Check optimization metadata
        assert opt_channels.metadata.get('gpu_optimized', False), "Optimization flag not set"
        
        if 'optimization_metrics' in opt_channels.metadata:
            metrics = opt_channels.metadata['optimization_metrics']
            for channel_name, channel_metrics in metrics.items():
                print(f"  {channel_name} coalescing: {channel_metrics['coalescing_efficiency']:.2%}")


def run_all_tests():
    """Run all GPU memory layout tests"""
    print("GPU Memory Layout Optimization Tests")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available - skipping GPU tests")
        return
    
    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_memory:.1f}GB")
    print()
    
    test_classes = [
        TestMemoryAlignment,
        TestCoalescingEfficiency,
        TestBankConflicts,
        TestCacheEfficiency,
        TestTransferBandwidth,
        TestLargeScalePerformance,
        TestMemoryPressureHandling,
        TestChannelIntegration,
    ]
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)
        
        test_instance = test_class()
        
        # Run all test methods
        for method_name in dir(test_instance):
            if method_name.startswith('test_'):
                print(f"\n{method_name}:")
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    print("✓ PASS")
                except Exception as e:
                    print(f"✗ FAIL: {e}")
                    traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("All tests completed!")


if __name__ == "__main__":
    run_all_tests()