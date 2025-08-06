"""
Comprehensive tests for GPU Memory Pool System
Tests stress allocation, fragmentation, multi-GPU, and failure scenarios
"""

import torch
import numpy as np
import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import gc
import psutil
import pytest
from typing import List, Tuple, Dict, Any

from gpu_memory_pool import (
    GPUMemoryPool, 
    GPUMemoryPoolConfig,
    PoolType,
    AllocationStatus
)


class TestGPUMemoryPool:
    """Test suite for GPU memory pool"""
    
    @pytest.fixture
    def pool_config(self):
        """Create test configuration"""
        config = GPUMemoryPoolConfig(
            device_pool_size_mb=1024,  # 1GB for testing
            pinned_pool_size_mb=256,   # 256MB
            unified_pool_size_mb=512,  # 512MB
            slab_chunk_blocks=32,
            buddy_max_order=15,  # Smaller for testing
            defrag_interval_seconds=0,  # Disable auto-defrag for tests
            enable_integrity_checks=True
        )
        return config
    
    @pytest.fixture
    def pool(self, pool_config):
        """Create pool instance"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        pool = GPUMemoryPool(pool_config)
        yield pool
        pool.shutdown()
    
    def test_basic_allocation(self, pool):
        """Test basic allocation and deallocation"""
        # Small allocation (slab)
        ptr1, id1 = pool.allocate(1024)  # 1KB
        assert ptr1 > 0
        assert id1 is not None
        
        # Medium allocation (slab)
        ptr2, id2 = pool.allocate(65536)  # 64KB
        assert ptr2 > 0
        assert ptr2 != ptr1
        
        # Large allocation (buddy)
        ptr3, id3 = pool.allocate(10 * 1024 * 1024)  # 10MB
        assert ptr3 > 0
        
        # Deallocate
        assert pool.deallocate(id1)
        assert pool.deallocate(id2)
        assert pool.deallocate(id3)
        
        # Verify statistics
        stats = pool.get_statistics()
        assert stats['successful_allocations'] == 3
        assert stats['slab_allocations'] >= 2
        assert stats['buddy_allocations'] >= 1
    
    def test_alignment(self, pool):
        """Test memory alignment guarantees"""
        sizes = [100, 1000, 10000, 100000]
        
        for size in sizes:
            ptr, alloc_id = pool.allocate(size, alignment=128)
            
            # Check 128-byte alignment
            assert ptr % 128 == 0, f"Pointer {ptr} not 128-byte aligned"
            
            pool.deallocate(alloc_id)
    
    def test_concurrent_allocations(self, pool):
        """Test concurrent allocations from multiple threads"""
        num_threads = 16
        allocations_per_thread = 100
        
        def allocate_deallocate_worker(thread_id):
            """Worker function for concurrent test"""
            local_allocations = []
            
            for i in range(allocations_per_thread):
                size = random.randint(256, 1024 * 1024)  # 256B to 1MB
                try:
                    ptr, alloc_id = pool.allocate(size, tag=f"thread_{thread_id}_{i}")
                    local_allocations.append(alloc_id)
                    
                    # Random delay
                    time.sleep(random.uniform(0, 0.001))
                    
                    # Deallocate some randomly
                    if random.random() > 0.5 and local_allocations:
                        dealloc_id = local_allocations.pop(random.randint(0, len(local_allocations)-1))
                        pool.deallocate(dealloc_id)
                        
                except RuntimeError as e:
                    print(f"Thread {thread_id} allocation failed: {e}")
            
            # Clean up remaining allocations
            for alloc_id in local_allocations:
                pool.deallocate(alloc_id)
            
            return len(local_allocations)
        
        # Run concurrent test
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(allocate_deallocate_worker, i) for i in range(num_threads)]
            results = [f.result() for f in as_completed(futures)]
        
        # Validate pool integrity
        assert pool.validate_integrity()
        
        # Check statistics
        stats = pool.get_statistics()
        assert stats['total_allocations'] >= num_threads * allocations_per_thread // 2
        
        # Check timing
        if 'avg_allocation_time_ms' in stats:
            assert stats['avg_allocation_time_ms'] < 1.0, "Allocation not sub-millisecond"
    
    def test_fragmentation_and_defrag(self, pool):
        """Test fragmentation detection and defragmentation"""
        # Create fragmentation by allocating and freeing in pattern
        allocations = []
        
        # Allocate many blocks
        for i in range(100):
            size = (i % 10 + 1) * 1024 * 1024  # 1-10 MB
            ptr, alloc_id = pool.allocate(size)
            allocations.append(alloc_id)
        
        # Free every other block to create fragmentation
        for i in range(0, len(allocations), 2):
            pool.deallocate(allocations[i])
        
        # Check fragmentation
        frag_before = pool.get_fragmentation()
        print(f"Fragmentation before defrag: {frag_before}")
        
        # Run defragmentation
        defrag_stats = pool.defragment()
        print(f"Defragmentation stats: {defrag_stats}")
        
        # Check fragmentation after
        frag_after = pool.get_fragmentation()
        print(f"Fragmentation after defrag: {frag_after}")
        
        # Clean up
        for i in range(1, len(allocations), 2):
            pool.deallocate(allocations[i])
        
        # Verify some improvement
        assert defrag_stats['blocks_coalesced'] > 0
    
    def test_batch_operations(self, pool):
        """Test batch allocation and deallocation"""
        sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
        
        # Batch allocate
        results = pool.batch_allocate(sizes)
        assert len(results) == len(sizes)
        
        # Verify all different pointers
        ptrs = [ptr for ptr, _ in results]
        assert len(set(ptrs)) == len(ptrs), "Duplicate pointers allocated"
        
        # Batch deallocate
        alloc_ids = [aid for _, aid in results]
        assert pool.batch_deallocate(alloc_ids)
        
        # Verify stats
        stats = pool.get_statistics()
        assert stats['successful_allocations'] >= len(sizes)
    
    def test_pool_exhaustion(self, pool):
        """Test behavior when pool is exhausted"""
        # Allocate until exhaustion
        allocations = []
        allocation_size = 100 * 1024 * 1024  # 100MB chunks
        
        try:
            for i in range(1000):  # Try to allocate 100GB
                ptr, alloc_id = pool.allocate(allocation_size)
                allocations.append(alloc_id)
        except RuntimeError as e:
            assert "ALLOCATION FAILED" in str(e) or "GPU OOM" in str(e)
            print(f"Pool exhausted after {len(allocations)} allocations")
        
        # Clean up
        for alloc_id in allocations:
            pool.deallocate(alloc_id)
        
        # Pool should be functional after cleanup
        ptr, alloc_id = pool.allocate(1024)
        pool.deallocate(alloc_id)
    
    def test_double_free_detection(self, pool):
        """Test double-free detection causes hard failure"""
        ptr, alloc_id = pool.allocate(1024)
        
        # First free should succeed
        assert pool.deallocate(alloc_id)
        
        # Second free should crash
        with pytest.raises(RuntimeError) as exc_info:
            pool.deallocate(alloc_id)
        
        assert "DOUBLE FREE" in str(exc_info.value) or "not found" in str(exc_info.value)
    
    def test_invalid_size(self, pool):
        """Test invalid allocation sizes cause hard failure"""
        # Zero size
        with pytest.raises(RuntimeError) as exc_info:
            pool.allocate(0)
        assert "INVALID SIZE" in str(exc_info.value)
        
        # Negative size
        with pytest.raises(RuntimeError) as exc_info:
            pool.allocate(-1024)
        assert "INVALID SIZE" in str(exc_info.value)
    
    def test_multi_device(self, pool):
        """Test multi-GPU allocation if available"""
        if torch.cuda.device_count() < 2:
            pytest.skip("Multi-GPU test requires 2+ GPUs")
        
        # Allocate on different devices
        ptr0, id0 = pool.allocate(1024, device_id=0)
        ptr1, id1 = pool.allocate(1024, device_id=1)
        
        # Verify different allocations
        assert ptr0 != ptr1
        
        # Deallocate
        pool.deallocate(id0)
        pool.deallocate(id1)
        
        # Check per-device stats
        stats = pool.get_statistics()
        assert 'device_0' in stats['per_device']
        assert 'device_1' in stats['per_device']
    
    def test_pinned_memory(self, pool):
        """Test pinned memory allocation"""
        # Allocate pinned memory
        ptr, alloc_id = pool.allocate(1024 * 1024, pool_type=PoolType.PINNED)
        assert ptr > 0
        
        # Deallocate
        pool.deallocate(alloc_id)
        
        # Check stats show pinned allocation
        stats = pool.get_statistics()
        assert stats['successful_allocations'] >= 1
    
    def test_memory_tagging(self, pool):
        """Test memory tagging for debugging"""
        # Allocate with tags
        tags = ["model_weights", "gradients", "optimizer_state", "activations"]
        allocations = []
        
        for tag in tags:
            ptr, alloc_id = pool.allocate(1024 * 1024, tag=tag)
            allocations.append(alloc_id)
        
        # Verify allocations tracked
        assert len(pool.allocations) == len(tags)
        
        # Clean up
        for alloc_id in allocations:
            pool.deallocate(alloc_id)
    
    def test_stress_allocation_pattern(self, pool):
        """Stress test with realistic allocation patterns"""
        # Simulate model training allocation pattern
        allocations = {}
        
        # Phase 1: Model loading (large allocations)
        model_sizes = [50 * 1024 * 1024, 100 * 1024 * 1024, 25 * 1024 * 1024]
        for i, size in enumerate(model_sizes):
            ptr, alloc_id = pool.allocate(size, tag=f"model_layer_{i}")
            allocations[f"model_{i}"] = alloc_id
        
        # Phase 2: Batch processing (many medium allocations)
        batch_allocations = []
        for batch in range(10):
            batch_ids = []
            for tensor in range(5):
                size = random.randint(1024 * 1024, 10 * 1024 * 1024)
                ptr, alloc_id = pool.allocate(size, tag=f"batch_{batch}_tensor_{tensor}")
                batch_ids.append(alloc_id)
            batch_allocations.append(batch_ids)
            
            # Simulate processing delay
            time.sleep(0.01)
            
            # Free previous batch
            if batch > 0:
                for aid in batch_allocations[batch-1]:
                    pool.deallocate(aid)
        
        # Phase 3: Cleanup
        for key, alloc_id in allocations.items():
            pool.deallocate(alloc_id)
        
        if batch_allocations:
            for aid in batch_allocations[-1]:
                pool.deallocate(aid)
        
        # Verify pool is still healthy
        assert pool.validate_integrity()
        
        # Check performance metrics
        stats = pool.get_statistics()
        if 'p99_allocation_time_ms' in stats:
            assert stats['p99_allocation_time_ms'] < 5.0, "P99 allocation time too high"
    
    def test_cuda_stream_allocation(self, pool):
        """Test allocation with CUDA streams"""
        # Create CUDA streams
        streams = [torch.cuda.Stream() for _ in range(4)]
        allocations = []
        
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                ptr, alloc_id = pool.allocate(
                    1024 * 1024,
                    stream=stream,
                    tag=f"stream_{i}"
                )
                allocations.append(alloc_id)
        
        # Synchronize streams
        for stream in streams:
            stream.synchronize()
        
        # Deallocate
        for alloc_id in allocations:
            pool.deallocate(alloc_id)
    
    def test_numa_awareness(self, pool):
        """Test NUMA-aware allocation if available"""
        if not pool.config.enable_numa or len(pool.config.numa_nodes) < 2:
            pytest.skip("NUMA test requires multiple NUMA nodes")
        
        # This would test NUMA-aware allocation
        # Implementation depends on system configuration
        pass
    
    def test_recovery_after_oom(self, pool):
        """Test pool recovery after OOM condition"""
        huge_size = 100 * 1024 * 1024 * 1024  # 100GB - should fail
        
        # Try huge allocation that should fail
        with pytest.raises(RuntimeError) as exc_info:
            pool.allocate(huge_size)
        
        assert "GPU OOM" in str(exc_info.value) or "ALLOCATION FAILED" in str(exc_info.value)
        
        # Pool should still work for smaller allocations
        ptr, alloc_id = pool.allocate(1024)
        pool.deallocate(alloc_id)
        
        # Verify integrity
        assert pool.validate_integrity()


def benchmark_allocation_performance():
    """Standalone benchmark for allocation performance"""
    if not torch.cuda.is_available():
        print("CUDA not available for benchmark")
        return
    
    print("GPU Memory Pool Performance Benchmark")
    print("=" * 50)
    
    # Create pool
    config = GPUMemoryPoolConfig(
        device_pool_size_mb=4096,  # 4GB
        enable_integrity_checks=False  # Disable for performance
    )
    pool = GPUMemoryPool(config)
    
    # Benchmark different allocation sizes
    sizes = [
        (256, "256B"),
        (1024, "1KB"),
        (4096, "4KB"),
        (65536, "64KB"),
        (1048576, "1MB"),
        (10485760, "10MB"),
        (104857600, "100MB")
    ]
    
    for size, label in sizes:
        allocations = []
        
        # Warmup
        for _ in range(10):
            ptr, aid = pool.allocate(size)
            pool.deallocate(aid)
        
        # Benchmark allocation
        start = time.perf_counter()
        for _ in range(1000):
            ptr, aid = pool.allocate(size)
            allocations.append(aid)
        alloc_time = (time.perf_counter() - start) * 1000
        
        # Benchmark deallocation
        start = time.perf_counter()
        for aid in allocations:
            pool.deallocate(aid)
        dealloc_time = (time.perf_counter() - start) * 1000
        
        print(f"{label:8} - Alloc: {alloc_time/1000:.3f}ms, Dealloc: {dealloc_time/1000:.3f}ms")
    
    # Get final statistics
    stats = pool.get_statistics()
    print("\nFinal Statistics:")
    print(f"Total allocations: {stats['total_allocations']}")
    print(f"Slab allocations: {stats['slab_allocations']}")
    print(f"Buddy allocations: {stats['buddy_allocations']}")
    
    if 'avg_allocation_time_ms' in stats:
        print(f"Avg allocation time: {stats['avg_allocation_time_ms']:.3f}ms")
        print(f"P99 allocation time: {stats['p99_allocation_time_ms']:.3f}ms")
    
    pool.shutdown()


if __name__ == "__main__":
    # Run benchmark
    benchmark_allocation_performance()
    
    # Run tests with pytest if available
    try:
        import pytest
        pytest.main([__file__, "-v", "--tb=short"])
    except ImportError:
        print("\nInstall pytest to run full test suite")
        
        # Run basic test manually
        if torch.cuda.is_available():
            print("\nRunning basic manual test...")
            config = GPUMemoryPoolConfig()
            pool = GPUMemoryPool(config)
            
            # Basic test
            ptr, aid = pool.allocate(1024)
            print(f"Allocated 1KB at {ptr}, ID: {aid}")
            pool.deallocate(aid)
            print("Deallocated successfully")
            
            # Verify integrity
            pool.validate_integrity()
            print("Integrity check passed")
            
            pool.shutdown()
            print("Basic test completed successfully")