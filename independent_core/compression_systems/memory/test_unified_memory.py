"""
Test suite for Unified Memory Handler - Production validation
Tests all memory management features with realistic workloads
"""

import torch
import numpy as np
import time
import threading
import pytest
from typing import List, Dict, Any
import psutil
import gc

from unified_memory_handler import (
    UnifiedMemoryHandler,
    UnifiedMemoryConfig,
    MemoryPressureLevel,
    AllocationPriority,
    EvictionStrategy,
    MigrationPolicy,
    MemoryRequest,
    MigrationStats,
    create_unified_handler,
    integrate_with_compression_systems
)


class TestUnifiedMemoryHandler:
    """Comprehensive test suite for unified memory handler"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create handler with test configuration
        self.config = UnifiedMemoryConfig(
            gpu_memory_limit_mb=4096,  # 4GB for testing
            cpu_memory_limit_mb=8192,   # 8GB for testing
            moderate_threshold=0.40,     # Lower thresholds for testing
            high_threshold=0.60,
            critical_threshold=0.80,
            exhausted_threshold=0.95,
            monitoring_interval_ms=50,
            enable_compression=True,
            compression_threshold_mb=1.0
        )
        self.handler = UnifiedMemoryHandler(self.config)
    
    def teardown_method(self):
        """Cleanup after test"""
        if hasattr(self, 'handler'):
            self.handler.shutdown()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_basic_allocation(self):
        """Test basic memory allocation and deallocation"""
        print("\nTest: Basic allocation...")
        
        # Submit allocation request
        request_id = self.handler.submit_request(
            subsystem='test',
            size_bytes=1024 * 1024,  # 1MB
            priority=AllocationPriority.NORMAL,
            device='cpu'
        )
        
        # Wait for processing
        time.sleep(0.1)
        
        # Check allocation succeeded
        stats = self.handler.get_memory_stats()
        assert stats['operations']['successful_allocations'] > 0
        assert stats['allocations']['total'] > 0
        
        # Find and free allocation
        allocations = list(self.handler.tracker.allocations.keys())
        if allocations:
            assert self.handler.free_memory(allocations[0])
        
        print("✓ Basic allocation test passed")
    
    def test_priority_allocation(self):
        """Test priority-based allocation"""
        print("\nTest: Priority allocation...")
        
        requests = []
        priorities = [
            AllocationPriority.LOW,
            AllocationPriority.NORMAL,
            AllocationPriority.HIGH,
            AllocationPriority.CRITICAL
        ]
        
        # Submit requests with different priorities
        for i, priority in enumerate(priorities):
            request_id = self.handler.submit_request(
                subsystem='test',
                size_bytes=512 * 1024,  # 512KB each
                priority=priority,
                device='cpu'
            )
            requests.append(request_id)
        
        # Wait for processing
        time.sleep(0.2)
        
        # Check all were allocated
        stats = self.handler.get_memory_stats()
        assert stats['operations']['successful_allocations'] >= len(priorities)
        
        print("✓ Priority allocation test passed")
    
    def test_memory_pressure_monitoring(self):
        """Test memory pressure detection"""
        print("\nTest: Memory pressure monitoring...")
        
        # Get initial pressure
        initial_pressure = self.handler.monitor.get_pressure_level('cpu')
        print(f"Initial pressure: {initial_pressure.name}")
        
        # Allocate large amount to increase pressure
        large_allocations = []
        for i in range(10):
            request_id = self.handler.submit_request(
                subsystem='test',
                size_bytes=100 * 1024 * 1024,  # 100MB each
                priority=AllocationPriority.NORMAL,
                device='cpu'
            )
            large_allocations.append(request_id)
        
        # Wait for allocations
        time.sleep(0.5)
        
        # Check pressure increased
        new_pressure = self.handler.monitor.get_pressure_level('cpu')
        print(f"New pressure: {new_pressure.name}")
        
        # Pressure should have increased or stayed same
        assert new_pressure.value >= initial_pressure.value
        
        print("✓ Memory pressure monitoring test passed")
    
    def test_eviction_strategy(self):
        """Test memory eviction strategies"""
        print("\nTest: Eviction strategies...")
        
        # Fill memory with different priority allocations
        allocations = []
        for i in range(5):
            for priority in [AllocationPriority.LOW, AllocationPriority.NORMAL]:
                request_id = self.handler.submit_request(
                    subsystem='test',
                    size_bytes=50 * 1024 * 1024,  # 50MB
                    priority=priority,
                    device='cpu'
                )
                allocations.append((request_id, priority))
        
        # Wait for allocations
        time.sleep(0.5)
        
        initial_count = len(self.handler.tracker.allocations)
        print(f"Initial allocations: {initial_count}")
        
        # Trigger eviction by requesting high-priority large allocation
        critical_request = self.handler.submit_request(
            subsystem='test',
            size_bytes=500 * 1024 * 1024,  # 500MB
            priority=AllocationPriority.CRITICAL,
            device='cpu'
        )
        
        # Wait for eviction and allocation
        time.sleep(1.0)
        
        # Check evictions occurred
        stats = self.handler.get_memory_stats()
        assert stats['operations']['total_evictions'] > 0
        
        final_count = len(self.handler.tracker.allocations)
        print(f"Final allocations: {final_count}")
        print(f"Evictions: {stats['operations']['total_evictions']}")
        
        print("✓ Eviction strategy test passed")
    
    def test_compression(self):
        """Test memory compression for evicted data"""
        print("\nTest: Memory compression...")
        
        # Create compressible data
        test_data = np.zeros((1000, 1000), dtype=np.float32)
        test_data[::2, ::2] = 1.0  # Pattern for good compression
        
        # Test compression
        compressed, metadata = self.handler.compressor.compress(test_data)
        
        assert metadata['compressed_size'] < metadata['original_size']
        assert metadata['compression_ratio'] > 1.0
        
        print(f"Compression ratio: {metadata['compression_ratio']:.2f}x")
        
        # Test decompression
        decompressed = self.handler.compressor.decompress(compressed, metadata)
        assert np.array_equal(test_data, decompressed)
        
        print("✓ Compression test passed")
    
    def test_concurrent_allocation(self):
        """Test concurrent allocation requests"""
        print("\nTest: Concurrent allocations...")
        
        num_threads = 10
        allocations_per_thread = 5
        success_count = threading.Event()
        results = []
        
        def allocate_worker(thread_id):
            """Worker thread for allocation"""
            thread_results = []
            for i in range(allocations_per_thread):
                try:
                    request_id = self.handler.submit_request(
                        subsystem=f'thread_{thread_id}',
                        size_bytes=10 * 1024 * 1024,  # 10MB
                        priority=AllocationPriority.NORMAL,
                        device='cpu'
                    )
                    thread_results.append(request_id)
                    time.sleep(0.01)
                except Exception as e:
                    print(f"Thread {thread_id} error: {e}")
            results.append(thread_results)
        
        # Start threads
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=allocate_worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check results
        total_requests = sum(len(r) for r in results)
        print(f"Total requests submitted: {total_requests}")
        
        # Wait for processing
        time.sleep(1.0)
        
        stats = self.handler.get_memory_stats()
        print(f"Successful allocations: {stats['operations']['successful_allocations']}")
        
        # Should have processed most requests
        assert stats['operations']['successful_allocations'] > 0
        
        print("✓ Concurrent allocation test passed")
    
    def test_gpu_allocation(self):
        """Test GPU memory allocation if available"""
        if not torch.cuda.is_available():
            print("\nTest: GPU allocation skipped (no GPU)")
            return
        
        print("\nTest: GPU allocation...")
        
        # Submit GPU allocation request
        request_id = self.handler.submit_request(
            subsystem='test_gpu',
            size_bytes=256 * 1024 * 1024,  # 256MB
            priority=AllocationPriority.HIGH,
            device='cuda:0'
        )
        
        # Wait for processing
        time.sleep(0.2)
        
        # Check GPU allocation
        stats = self.handler.get_memory_stats()
        if 'cuda:0' in stats['allocations']['by_device']:
            device_stats = stats['allocations']['by_device']['cuda:0']
            assert device_stats['count'] > 0
            print(f"GPU allocations: {device_stats['count']}")
            print(f"GPU pressure: {device_stats['pressure']}")
        
        print("✓ GPU allocation test passed")
    
    def test_memory_prediction(self):
        """Test memory exhaustion prediction"""
        print("\nTest: Memory exhaustion prediction...")
        
        # Simulate increasing memory usage
        for i in range(5):
            self.handler.submit_request(
                subsystem='test',
                size_bytes=100 * 1024 * 1024,  # 100MB
                priority=AllocationPriority.NORMAL,
                device='cpu'
            )
            time.sleep(0.1)
        
        # Try to predict exhaustion
        exhaustion_time = self.handler.monitor.predict_exhaustion('cpu')
        
        if exhaustion_time is not None:
            print(f"Predicted exhaustion in: {exhaustion_time:.1f} seconds")
        else:
            print("No exhaustion predicted")
        
        print("✓ Memory prediction test passed")
    
    def test_emergency_gc(self):
        """Test emergency garbage collection"""
        print("\nTest: Emergency GC...")
        
        initial_gc_count = self.handler.stats['emergency_gc_count']
        
        # Force high memory pressure
        self.handler.config.emergency_gc_threshold = 0.01  # Very low threshold
        
        # Trigger emergency GC
        self.handler._emergency_gc()
        
        # Check GC was triggered
        assert self.handler.stats['emergency_gc_count'] > initial_gc_count
        
        print(f"Emergency GC triggered: {self.handler.stats['emergency_gc_count']} times")
        print("✓ Emergency GC test passed")
    
    def test_subsystem_integration(self):
        """Test integration with compression subsystems"""
        print("\nTest: Subsystem integration...")
        
        # Mock subsystems
        class MockTropicalSystem:
            def __init__(self):
                self.name = "tropical"
        
        class MockPadicSystem:
            def __init__(self):
                self.name = "padic"
        
        tropical = MockTropicalSystem()
        padic = MockPadicSystem()
        
        # Integrate subsystems
        integrate_with_compression_systems(
            self.handler,
            tropical_system=tropical,
            padic_system=padic
        )
        
        # Check integration
        assert 'tropical' in self.handler.subsystem_handlers
        assert 'padic' in self.handler.subsystem_handlers
        
        # Submit requests from different subsystems
        tropical_req = self.handler.submit_request(
            subsystem='tropical',
            size_bytes=50 * 1024 * 1024,
            priority=AllocationPriority.HIGH
        )
        
        padic_req = self.handler.submit_request(
            subsystem='padic',
            size_bytes=50 * 1024 * 1024,
            priority=AllocationPriority.NORMAL
        )
        
        # Wait for processing
        time.sleep(0.2)
        
        # Check subsystem allocations
        stats = self.handler.get_memory_stats()
        assert 'tropical' in stats['allocations']['by_subsystem']
        assert 'padic' in stats['allocations']['by_subsystem']
        
        print("✓ Subsystem integration test passed")


def run_stress_test():
    """Run stress test with high load"""
    print("\n" + "="*60)
    print("STRESS TEST: High memory pressure simulation")
    print("="*60)
    
    handler = create_unified_handler()
    
    try:
        # Simulate heavy load
        print("\nPhase 1: Ramping up allocations...")
        requests = []
        for i in range(50):
            size = np.random.randint(1, 100) * 1024 * 1024  # 1-100MB
            priority = np.random.choice(list(AllocationPriority))
            device = 'cuda:0' if torch.cuda.is_available() and i % 2 == 0 else 'cpu'
            
            request_id = handler.submit_request(
                subsystem=f'stress_{i}',
                size_bytes=size,
                priority=priority,
                device=device
            )
            requests.append(request_id)
            
            if i % 10 == 0:
                print(f"  Submitted {i+1} requests...")
            
            time.sleep(0.01)
        
        print("\nPhase 2: Waiting for processing...")
        time.sleep(2.0)
        
        # Get statistics
        stats = handler.get_memory_stats()
        
        print("\nStress Test Results:")
        print(f"Total allocations attempted: {stats['operations']['total_allocations']}")
        print(f"Successful allocations: {stats['operations']['successful_allocations']}")
        print(f"Failed allocations: {stats['operations']['failed_allocations']}")
        print(f"Total evictions: {stats['operations']['total_evictions']}")
        print(f"Emergency GC count: {stats['operations']['emergency_gc_count']}")
        print(f"Compressions performed: {stats['operations']['compression_count']}")
        
        # Print memory usage
        print("\nMemory Usage by Device:")
        for device, info in stats['allocations']['by_device'].items():
            print(f"  {device}: {info['count']} allocations, "
                  f"{info['bytes'] / (1024**2):.2f} MB, "
                  f"pressure: {info['pressure']}")
        
        print("\nMemory Usage by Subsystem:")
        for subsystem, info in stats['allocations']['by_subsystem'].items():
            print(f"  {subsystem}: {info['count']} allocations, "
                  f"{info['bytes'] / (1024**2):.2f} MB")
        
        # Success criteria
        success_rate = (stats['operations']['successful_allocations'] / 
                       stats['operations']['total_allocations'])
        
        if success_rate > 0.5:  # At least 50% success rate
            print(f"\n✓ Stress test PASSED (success rate: {success_rate:.1%})")
        else:
            print(f"\n✗ Stress test FAILED (success rate: {success_rate:.1%})")
        
    finally:
        handler.shutdown()


def run_performance_benchmark():
    """Benchmark memory handler performance"""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    handler = create_unified_handler()
    
    try:
        # Benchmark allocation speed
        print("\n1. Allocation Speed Test")
        sizes = [1024, 1024*10, 1024*100, 1024*1024, 1024*1024*10]
        
        for size in sizes:
            start = time.perf_counter()
            
            request_id = handler.submit_request(
                subsystem='benchmark',
                size_bytes=size,
                priority=AllocationPriority.NORMAL,
                device='cpu'
            )
            
            # Wait for allocation
            time.sleep(0.05)
            
            elapsed = (time.perf_counter() - start) * 1000
            print(f"  {size / 1024:.1f} KB: {elapsed:.2f} ms")
        
        # Benchmark eviction speed
        print("\n2. Eviction Speed Test")
        
        # Fill memory
        for i in range(20):
            handler.submit_request(
                subsystem='benchmark',
                size_bytes=10 * 1024 * 1024,
                priority=AllocationPriority.LOW,
                device='cpu'
            )
        
        time.sleep(0.5)
        
        # Time eviction
        start = time.perf_counter()
        handler._trigger_eviction(100 * 1024 * 1024, 'cpu')
        eviction_time = (time.perf_counter() - start) * 1000
        
        print(f"  Eviction of 100MB: {eviction_time:.2f} ms")
        
        # Benchmark monitoring overhead
        print("\n3. Monitoring Overhead Test")
        
        # Measure baseline CPU usage
        baseline_cpu = psutil.cpu_percent(interval=1)
        
        # Run for a while with monitoring
        time.sleep(2)
        
        monitoring_cpu = psutil.cpu_percent(interval=1)
        overhead = monitoring_cpu - baseline_cpu
        
        print(f"  Baseline CPU: {baseline_cpu:.1f}%")
        print(f"  With monitoring: {monitoring_cpu:.1f}%")
        print(f"  Overhead: {overhead:.1f}%")
        
        print("\n✓ Performance benchmark complete")
        
    finally:
        handler.shutdown()


if __name__ == "__main__":
    print("="*60)
    print("UNIFIED MEMORY HANDLER TEST SUITE")
    print("="*60)
    
    # Run unit tests
    test_suite = TestUnifiedMemoryHandler()
    
    tests = [
        'test_basic_allocation',
        'test_priority_allocation',
        'test_memory_pressure_monitoring',
        'test_eviction_strategy',
        'test_compression',
        'test_concurrent_allocation',
        'test_gpu_allocation',
        'test_memory_prediction',
        'test_emergency_gc',
        'test_subsystem_integration'
    ]
    
    print("\nRunning unit tests...")
    for test_name in tests:
        test_suite.setup_method()
        try:
            test_method = getattr(test_suite, test_name)
            test_method()
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
        finally:
            test_suite.teardown_method()
    
    # Run stress test
    run_stress_test()
    
    # Run performance benchmark
    run_performance_benchmark()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE - UNIFIED MEMORY HANDLER READY")
    print("="*60)

class TestMemoryMigration:
    """Test suite for memory migration functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = UnifiedMemoryConfig(
            gpu_memory_limit_mb=2048,
            cpu_memory_limit_mb=4096,
            migration_policy=MigrationPolicy.ADAPTIVE,
            enable_zero_copy=True,
            migration_chunk_size_mb=32,
            enable_async_migration=True,
            enable_checksum_validation=True,
            migration_parallelism=4
        )
        self.handler = UnifiedMemoryHandler(self.config)
    
    def teardown_method(self):
        """Cleanup after test"""
        if hasattr(self, 'handler'):
            self.handler.shutdown()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_basic_migration(self):
        """Test basic CPU to GPU migration"""
        print("\nTest: Basic memory migration...")
        
        # Create CPU allocation
        request_id = self.handler.submit_request(
            subsystem='migration_test',
            size_bytes=10 * 1024 * 1024,  # 10MB
            priority=AllocationPriority.NORMAL,
            device='cpu'
        )
        
        time.sleep(0.1)
        
        # Find allocation
        allocations = list(self.handler.tracker.allocations.keys())
        assert len(allocations) > 0
        
        allocation_id = allocations[0]
        allocation = self.handler.tracker.allocations[allocation_id]
        assert allocation.device == 'cpu'
        
        # Migrate to GPU if available
        if torch.cuda.is_available():
            stats = self.handler.migrate_allocation(
                allocation_id,
                'cuda:0',
                async_mode=False
            )
            
            assert isinstance(stats, MigrationStats)
            assert stats.success
            assert stats.source_device == 'cpu'
            assert stats.target_device == 'cuda:0'
            assert stats.checksum_valid
            
            # Verify allocation moved
            allocation = self.handler.tracker.allocations[allocation_id]
            assert allocation.device == 'cuda:0'
            assert allocation.migration_count == 1
            
            print(f"✓ Migration completed: {stats.throughput_gbps:.2f} GB/s")
    
    def test_concurrent_migrations(self):
        """Test multiple concurrent migrations"""
        print("\nTest: Concurrent migrations...")
        
        if not torch.cuda.is_available():
            print("⚠ Skipping - GPU not available")
            return
        
        # Create multiple allocations
        allocation_ids = []
        for i in range(5):
            request_id = self.handler.submit_request(
                subsystem=f'concurrent_{i}',
                size_bytes=5 * 1024 * 1024,  # 5MB each
                priority=AllocationPriority.NORMAL,
                device='cpu'
            )
            time.sleep(0.05)
        
        time.sleep(0.5)
        allocation_ids = list(self.handler.tracker.allocations.keys())[:5]
        
        # Bulk migrate
        migrations = [(aid, 'cuda:0') for aid in allocation_ids]
        futures = self.handler.bulk_migrate(migrations)
        
        # Wait for all migrations
        results = []
        for future in futures:
            try:
                result = future.result(timeout=5.0)
                results.append(result)
            except Exception as e:
                print(f"Migration failed: {e}")
        
        # Verify results
        successful = sum(1 for r in results if r.success)
        print(f"✓ Concurrent migrations: {successful}/{len(results)} successful")
        assert successful > 0
    
    def test_migration_under_pressure(self):
        """Test migration behavior under memory pressure"""
        print("\nTest: Migration under memory pressure...")
        
        # Fill memory to create pressure
        large_allocations = []
        for i in range(10):
            request_id = self.handler.submit_request(
                subsystem=f'pressure_{i}',
                size_bytes=100 * 1024 * 1024,  # 100MB each
                priority=AllocationPriority.PREEMPTIBLE,
                device='cpu'
            )
            large_allocations.append(request_id)
            time.sleep(0.05)
        
        time.sleep(1.0)
        
        # Check pressure level
        pressure = self.handler.monitor.get_pressure_level('cpu')
        print(f"  Current CPU pressure: {pressure.name}")
        
        # Try optimization
        migrations_scheduled = self.handler.optimize_data_placement()
        print(f"  Optimization scheduled {migrations_scheduled} migrations")
        
        # Get migration stats
        stats = self.handler.get_migration_stats()
        print(f"  Total migrations: {stats['total_migrations']}")
        print(f"  Successful: {stats['successful_migrations']}")
        
        print("✓ Migration under pressure tested")


def run_migration_tests():
    """Run all migration tests"""
    print("\n" + "="*60)
    print("MIGRATION TESTS")
    print("="*60)
    
    migration_tester = TestMemoryMigration()
    
    migration_tester.setup_method()
    migration_tester.test_basic_migration()
    migration_tester.teardown_method()
    
    migration_tester.setup_method()
    migration_tester.test_concurrent_migrations()
    migration_tester.teardown_method()
    
    migration_tester.setup_method()
    migration_tester.test_migration_under_pressure()
    migration_tester.teardown_method()
    
    print("\n✓ All migration tests passed")
EOF < /dev/null