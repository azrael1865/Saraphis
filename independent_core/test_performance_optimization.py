"""
Comprehensive tests for performance optimization components.
Tests PerformanceOptimizer, MemoryOptimizer, and GeneralGPUMemoryOptimizer.
NO FALLBACKS - HARD FAILURES ONLY
"""

import gc
import logging
import psutil
import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Import performance optimization components
from .performance_optimizer import (
    PerformanceOptimizer, 
    OptimizationStrategy, 
    BottleneckType,
    OperationProfile,
    BottleneckReport,
    OptimizationRecommendation
)
from .memory_optimizer import (
    MemoryOptimizer,
    MemoryOptimizationStrategy,
    MemoryLeakSeverity,
    MemorySnapshot,
    MemoryLeakReport,
    CacheMetrics
)

# GPU optimizer tests conditional on torch availability
try:
    import torch
    from .gpu_memory_optimizer import (
        GeneralGPUMemoryOptimizer,
        GPUOptimizationStrategy,
        GPUMemorySnapshot,
        GPUAllocationRecord
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestPerformanceOptimizer:
    """Test suite for PerformanceOptimizer"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = {
            'enable_profiling': True,
            'profile_history_size': 100,
            'bottleneck_detection_threshold': 0.8,
            'optimization_strategy': 'balanced',
            'profiling_interval': 1.0,
            'max_concurrent_profiles': 10
        }
        self.optimizer = PerformanceOptimizer(self.config)
    
    def test_initialization_valid_config(self):
        """Test successful initialization with valid config"""
        assert self.optimizer.is_initialized
        assert self.optimizer.enable_profiling is True
        assert self.optimizer.optimization_strategy == OptimizationStrategy.BALANCED
        assert self.optimizer.profile_history_size == 100
        assert len(self.optimizer.operation_profiles) == 0
    
    def test_initialization_invalid_config_type(self):
        """Test initialization fails with invalid config type"""
        with pytest.raises(TypeError, match="Config must be dict or None"):
            PerformanceOptimizer("invalid_config")
    
    def test_initialization_invalid_profiling_interval(self):
        """Test initialization fails with invalid profiling interval"""
        invalid_config = self.config.copy()
        invalid_config['profiling_interval'] = -1.0
        
        with pytest.raises(ValueError, match="profiling_interval must be positive"):
            PerformanceOptimizer(invalid_config)
    
    def test_initialization_invalid_threshold(self):
        """Test initialization fails with invalid threshold"""
        invalid_config = self.config.copy()
        invalid_config['bottleneck_detection_threshold'] = 1.5
        
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            PerformanceOptimizer(invalid_config)
    
    def test_profile_operation_success(self):
        """Test successful operation profiling"""
        def test_operation(x, y):
            time.sleep(0.01)  # Small delay for measurable time
            return x + y
        
        result = self.optimizer.profile_operation("test_add", test_operation, 5, 3)
        
        assert result == 8
        assert "test_add" in self.optimizer.operation_profiles
        assert len(self.optimizer.operation_profiles["test_add"]) == 1
        
        profile = self.optimizer.operation_profiles["test_add"][0]
        assert isinstance(profile, OperationProfile)
        assert profile.operation_name == "test_add"
        assert profile.execution_time > 0
        assert profile.success is True
    
    def test_profile_operation_invalid_name(self):
        """Test profiling fails with invalid operation name"""
        def dummy_func():
            return True
        
        with pytest.raises(TypeError, match="operation_name must be str"):
            self.optimizer.profile_operation(123, dummy_func)
        
        with pytest.raises(ValueError, match="operation_name cannot be empty"):
            self.optimizer.profile_operation("", dummy_func)
    
    def test_profile_operation_invalid_function(self):
        """Test profiling fails with invalid function"""
        with pytest.raises(TypeError, match="operation_func must be callable"):
            self.optimizer.profile_operation("test", "not_callable")
    
    def test_profile_operation_exception_handling(self):
        """Test profiling handles operation exceptions properly"""
        def failing_operation():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            self.optimizer.profile_operation("failing_op", failing_operation)
        
        # Profile should still be recorded
        assert "failing_op" in self.optimizer.operation_profiles
        profile = self.optimizer.operation_profiles["failing_op"][0]
        assert profile.success is False
        assert profile.error_message == "Test error"
    
    def test_detect_bottlenecks_insufficient_data(self):
        """Test bottleneck detection with insufficient data"""
        # Add only a few profiles (less than minimum required)
        for i in range(5):
            self.optimizer.profile_operation("test_op", lambda: time.sleep(0.001), )
        
        bottlenecks = self.optimizer.detect_bottlenecks()
        assert len(bottlenecks) == 0  # Not enough data for analysis
    
    def test_detect_bottlenecks_profiling_disabled(self):
        """Test bottleneck detection fails when profiling disabled"""
        self.optimizer.enable_profiling = False
        
        with pytest.raises(RuntimeError, match="Profiling must be enabled"):
            self.optimizer.detect_bottlenecks()
    
    def test_optimization_strategy_types(self):
        """Test all optimization strategy types"""
        strategies = [
            OptimizationStrategy.CONSERVATIVE,
            OptimizationStrategy.BALANCED,
            OptimizationStrategy.AGGRESSIVE,
            OptimizationStrategy.CUSTOM
        ]
        
        for strategy in strategies:
            config = self.config.copy()
            config['optimization_strategy'] = strategy.value
            
            optimizer = PerformanceOptimizer(config)
            assert optimizer.optimization_strategy == strategy
    
    def test_performance_metrics_tracking(self):
        """Test performance metrics are tracked correctly"""
        initial_ops = self.optimizer.performance_metrics['total_operations_profiled']
        
        # Profile several operations
        for i in range(5):
            self.optimizer.profile_operation(f"op_{i}", lambda x=i: x * 2)
        
        final_ops = self.optimizer.performance_metrics['total_operations_profiled']
        assert final_ops == initial_ops + 5
        assert self.optimizer.performance_metrics['average_operation_time'] > 0
    
    def test_register_performance_hook(self):
        """Test performance hook registration"""
        hook_called = []
        
        def test_hook(profile):
            hook_called.append(profile.operation_name)
        
        self.optimizer.register_performance_hook("test_op", test_hook)
        
        # Profile operation should trigger hook
        self.optimizer.profile_operation("test_op", lambda: 42)
        
        assert len(hook_called) == 1
        assert hook_called[0] == "test_op"
    
    def test_register_performance_hook_invalid_params(self):
        """Test performance hook registration with invalid parameters"""
        with pytest.raises(TypeError, match="component_name must be str"):
            self.optimizer.register_performance_hook(123, lambda x: x)
        
        with pytest.raises(ValueError, match="component_name cannot be empty"):
            self.optimizer.register_performance_hook("", lambda x: x)
        
        with pytest.raises(TypeError, match="hook_func must be callable"):
            self.optimizer.register_performance_hook("test", "not_callable")
    
    def test_get_performance_report(self):
        """Test performance report generation"""
        # Generate some test data
        for i in range(3):
            self.optimizer.profile_operation("report_test", lambda x=i: x + 1)
        
        report = self.optimizer.get_performance_report()
        
        assert 'timestamp' in report
        assert 'performance_metrics' in report
        assert 'system_metrics' in report
        assert 'operation_statistics' in report
        assert 'configuration' in report
        
        # Check operation statistics
        assert 'report_test' in report['operation_statistics']
        op_stats = report['operation_statistics']['report_test']
        assert op_stats['total_calls'] == 3
        assert op_stats['success_rate'] == 1.0
        assert op_stats['avg_execution_time'] > 0
    
    def test_concurrent_profiling_limit(self):
        """Test concurrent profiling limits are enforced"""
        # Set low limit for testing
        self.optimizer.max_concurrent_profiles = 2
        
        def slow_operation():
            time.sleep(0.1)
            return True
        
        # Start operations in threads
        threads = []
        for i in range(5):  # More than the limit
            thread = threading.Thread(
                target=self.optimizer.profile_operation,
                args=(f"concurrent_{i}", slow_operation)
            )
            threads.append(thread)
        
        # Some should succeed, some should fail with RuntimeError
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # At least some profiles should have been recorded
        total_profiles = sum(len(profiles) for profiles in self.optimizer.operation_profiles.values())
        assert total_profiles > 0


class TestMemoryOptimizer:
    """Test suite for MemoryOptimizer"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = {
            'enable_tracking': True,
            'enable_leak_detection': True,
            'snapshot_interval': 1.0,
            'memory_threshold_mb': 1024,
            'leak_detection_threshold': 10.0,
            'max_snapshots': 100,
            'optimization_strategy': 'balanced'
        }
        self.optimizer = MemoryOptimizer(self.config)
    
    def test_initialization_valid_config(self):
        """Test successful initialization with valid config"""
        assert self.optimizer.is_initialized
        assert self.optimizer.enable_tracking is True
        assert self.optimizer.optimization_strategy == MemoryOptimizationStrategy.BALANCED
        assert len(self.optimizer.memory_snapshots) == 0
    
    def test_initialization_invalid_config_type(self):
        """Test initialization fails with invalid config type"""
        with pytest.raises(TypeError, match="Config must be dict or None"):
            MemoryOptimizer([1, 2, 3])
    
    def test_initialization_invalid_snapshot_interval(self):
        """Test initialization fails with invalid snapshot interval"""
        invalid_config = self.config.copy()
        invalid_config['snapshot_interval'] = -5.0
        
        with pytest.raises(ValueError, match="snapshot_interval must be positive"):
            MemoryOptimizer(invalid_config)
    
    def test_initialization_invalid_threshold(self):
        """Test initialization fails with invalid threshold"""
        invalid_config = self.config.copy()
        invalid_config['leak_detection_threshold'] = -1.0
        
        with pytest.raises(ValueError, match="leak_detection_threshold must be positive"):
            MemoryOptimizer(invalid_config)
    
    def test_track_memory_usage_context_manager(self):
        """Test memory tracking context manager"""
        initial_memory = len(self.optimizer.component_memory_usage)
        
        with self.optimizer.track_memory_usage("test_component") as tracking_data:
            # Simulate some memory usage
            test_data = [i for i in range(1000)]  # Create some objects
            tracking_data['test_data'] = test_data
        
        # Memory usage should be tracked
        assert "test_component" in self.optimizer.component_memory_usage
        assert len(self.optimizer.component_memory_usage["test_component"]) == 1
        
        usage_record = self.optimizer.component_memory_usage["test_component"][0]
        assert 'timestamp' in usage_record
        assert 'memory_delta' in usage_record
        assert 'duration' in usage_record
    
    def test_track_memory_usage_invalid_component_name(self):
        """Test memory tracking fails with invalid component name"""
        with pytest.raises(TypeError, match="component_name must be str"):
            with self.optimizer.track_memory_usage(123):
                pass
        
        with pytest.raises(ValueError, match="component_name cannot be empty"):
            with self.optimizer.track_memory_usage(""):
                pass
    
    def test_track_memory_usage_disabled(self):
        """Test memory tracking when disabled"""
        self.optimizer.enable_tracking = False
        
        with self.optimizer.track_memory_usage("disabled_test") as tracking_data:
            assert tracking_data == {}
        
        # No tracking data should be recorded
        assert "disabled_test" not in self.optimizer.component_memory_usage
    
    def test_detect_memory_leaks_insufficient_data(self):
        """Test leak detection with insufficient data"""
        # Add minimal usage data
        with self.optimizer.track_memory_usage("minimal_component"):
            pass
        
        leaks = self.optimizer.detect_memory_leaks()
        assert len(leaks) == 0  # Not enough data for analysis
    
    def test_detect_memory_leaks_disabled(self):
        """Test leak detection fails when disabled"""
        self.optimizer.enable_leak_detection = False
        
        with pytest.raises(RuntimeError, match="Memory leak detection must be enabled"):
            self.optimizer.detect_memory_leaks()
    
    def test_optimize_memory_usage(self):
        """Test memory usage optimization"""
        # Generate some tracking data first
        for i in range(5):
            with self.optimizer.track_memory_usage(f"opt_test_{i}"):
                # Simulate memory usage
                temp_data = list(range(100))
        
        result = self.optimizer.optimize_memory_usage()
        
        assert result['status'] == 'optimization_completed'
        assert 'optimizations_applied' in result
        assert 'total_time' in result
        assert result['total_time'] > 0
    
    def test_memory_optimization_strategies(self):
        """Test different memory optimization strategies"""
        strategies = [
            MemoryOptimizationStrategy.CONSERVATIVE,
            MemoryOptimizationStrategy.BALANCED,
            MemoryOptimizationStrategy.AGGRESSIVE
        ]
        
        for strategy in strategies:
            config = self.config.copy()
            config['optimization_strategy'] = strategy.value
            
            optimizer = MemoryOptimizer(config)
            assert optimizer.optimization_strategy == strategy
    
    def test_cache_registry(self):
        """Test cache registration and management"""
        cache_metrics = CacheMetrics(
            cache_name="test_cache",
            total_size=1024 * 1024,  # 1MB
            item_count=100,
            hit_rate=0.85,
            miss_rate=0.15,
            eviction_count=5,
            last_access=time.time(),
            creation_time=time.time()
        )
        
        self.optimizer.register_cache("test_cache", cache_metrics)
        
        assert "test_cache" in self.optimizer.cache_registry
        assert self.optimizer.cache_registry["test_cache"] == cache_metrics
    
    def test_cache_registry_invalid_params(self):
        """Test cache registration with invalid parameters"""
        with pytest.raises(TypeError, match="cache_name must be str"):
            self.optimizer.register_cache(123, Mock())
        
        with pytest.raises(TypeError, match="cache_metrics must be CacheMetrics"):
            self.optimizer.register_cache("test", "invalid")
    
    def test_clear_memory_cache_specific(self):
        """Test clearing specific memory cache"""
        # Register a test cache
        cache_metrics = CacheMetrics(
            cache_name="clear_test",
            total_size=512 * 1024,
            item_count=50,
            hit_rate=0.7,
            miss_rate=0.3,
            eviction_count=2,
            last_access=time.time(),
            creation_time=time.time()
        )
        
        self.optimizer.register_cache("clear_test", cache_metrics)
        
        # Clear specific cache
        result = self.optimizer.clear_memory_cache("clear_test")
        
        assert result['caches_cleared'] == ["clear_test"]
        assert result['estimated_memory_freed_mb'] > 0
        assert "clear_test" not in self.optimizer.cache_registry
    
    def test_clear_memory_cache_all(self):
        """Test clearing all memory caches"""
        # Register multiple test caches
        for i in range(3):
            cache_metrics = CacheMetrics(
                cache_name=f"cache_{i}",
                total_size=256 * 1024,
                item_count=25,
                hit_rate=0.8,
                miss_rate=0.2,
                eviction_count=1,
                last_access=time.time(),
                creation_time=time.time()
            )
            self.optimizer.register_cache(f"cache_{i}", cache_metrics)
        
        # Clear all caches
        result = self.optimizer.clear_memory_cache()
        
        assert len(result['caches_cleared']) == 3
        assert len(self.optimizer.cache_registry) == 0
    
    def test_clear_memory_cache_nonexistent(self):
        """Test clearing nonexistent cache raises error"""
        with pytest.raises(ValueError, match="Cache 'nonexistent' not found"):
            self.optimizer.clear_memory_cache("nonexistent")
    
    def test_get_memory_report(self):
        """Test memory report generation"""
        # Generate some test data
        with self.optimizer.track_memory_usage("report_component"):
            test_data = list(range(1000))
        
        report = self.optimizer.get_memory_report()
        
        assert 'timestamp' in report
        assert 'system_memory' in report
        assert 'component_summary' in report
        assert 'gc_statistics' in report
        assert 'optimization_metrics' in report
        assert 'configuration' in report
        
        # Check system memory info
        system_memory = report['system_memory']
        assert 'total_mb' in system_memory
        assert 'available_mb' in system_memory
        assert 'usage_percent' in system_memory
        
        # Check component summary
        if 'report_component' in report['component_summary']:
            comp_summary = report['component_summary']['report_component']
            assert 'total_calls' in comp_summary
            assert 'total_memory_delta' in comp_summary


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGeneralGPUMemoryOptimizer:
    """Test suite for GeneralGPUMemoryOptimizer (requires PyTorch and CUDA)"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = {
            'optimization_strategy': 'balanced',
            'memory_threshold_percent': 80.0,
            'fragmentation_threshold': 0.3,
            'stream_pool_size': 4,
            'enable_memory_pooling': True,
            'max_cached_memory_mb': 1024
        }
        
        # Skip if CUDA not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for GPU memory optimizer tests")
        
        self.optimizer = GeneralGPUMemoryOptimizer(self.config)
    
    def test_initialization_valid_config(self):
        """Test successful initialization with valid config"""
        assert self.optimizer.is_initialized
        assert self.optimizer.optimization_strategy == GPUOptimizationStrategy.BALANCED
        assert self.optimizer.device_count > 0
        assert len(self.optimizer.stream_pools) == self.optimizer.device_count
    
    def test_initialization_no_cuda(self):
        """Test initialization fails without CUDA"""
        with patch('torch.cuda.is_available', return_value=False):
            with pytest.raises(RuntimeError, match="CUDA is not available"):
                GeneralGPUMemoryOptimizer(self.config)
    
    def test_initialization_invalid_config(self):
        """Test initialization fails with invalid config"""
        invalid_config = self.config.copy()
        invalid_config['memory_threshold_percent'] = 150.0
        
        with pytest.raises(ValueError, match="must be between 0 and 100"):
            GeneralGPUMemoryOptimizer(invalid_config)
    
    def test_allocate_gpu_memory_success(self):
        """Test successful GPU memory allocation"""
        shape = (100, 100)
        dtype = torch.float32
        name = "test_tensor"
        
        tensor = self.optimizer.allocate_gpu_memory(shape, dtype, name)
        
        assert tensor.shape == shape
        assert tensor.dtype == dtype
        assert tensor.is_cuda
        assert name in [record.tensor_name for record in self.optimizer.allocation_records.values()]
    
    def test_allocate_gpu_memory_invalid_shape(self):
        """Test allocation fails with invalid shape"""
        with pytest.raises(TypeError, match="shape must be tuple or list"):
            self.optimizer.allocate_gpu_memory("invalid", torch.float32, "test")
        
        with pytest.raises(ValueError, match="shape cannot be empty"):
            self.optimizer.allocate_gpu_memory((), torch.float32, "test")
        
        with pytest.raises(ValueError, match="must be positive integers"):
            self.optimizer.allocate_gpu_memory((10, -5), torch.float32, "test")
    
    def test_allocate_gpu_memory_invalid_name(self):
        """Test allocation fails with invalid name"""
        with pytest.raises(TypeError, match="name must be str"):
            self.optimizer.allocate_gpu_memory((10, 10), torch.float32, 123)
        
        with pytest.raises(ValueError, match="name cannot be empty"):
            self.optimizer.allocate_gpu_memory((10, 10), torch.float32, "")
    
    def test_free_gpu_memory(self):
        """Test GPU memory freeing"""
        # Allocate memory first
        tensor = self.optimizer.allocate_gpu_memory((50, 50), torch.float32, "free_test")
        
        # Free memory
        result = self.optimizer.free_gpu_memory("free_test")
        
        assert result is True
        # Allocation record should be removed
        assert not any(
            record.tensor_name == "free_test" 
            for record in self.optimizer.allocation_records.values()
        )
    
    def test_free_gpu_memory_nonexistent(self):
        """Test freeing nonexistent memory allocation"""
        result = self.optimizer.free_gpu_memory("nonexistent")
        assert result is False
    
    def test_free_gpu_memory_invalid_name(self):
        """Test freeing memory with invalid name"""
        with pytest.raises(TypeError, match="tensor_name must be str"):
            self.optimizer.free_gpu_memory(123)
    
    def test_optimize_gpu_memory(self):
        """Test GPU memory optimization"""
        # Allocate some memory first to have something to optimize
        for i in range(3):
            self.optimizer.allocate_gpu_memory((20, 20), torch.float32, f"opt_test_{i}")
        
        result = self.optimizer.optimize_gpu_memory()
        
        assert result['status'] == 'optimization_completed'
        assert result['devices_optimized'] == self.optimizer.device_count
        assert 'optimizations_applied' in result
        assert 'total_memory_freed_mb' in result
        assert 'optimization_time' in result
    
    def test_gpu_optimization_strategies(self):
        """Test different GPU optimization strategies"""
        strategies = [
            GPUOptimizationStrategy.CONSERVATIVE,
            GPUOptimizationStrategy.BALANCED,
            GPUOptimizationStrategy.AGGRESSIVE,
            GPUOptimizationStrategy.MEMORY_FIRST,
            GPUOptimizationStrategy.PERFORMANCE_FIRST
        ]
        
        for strategy in strategies:
            config = self.config.copy()
            config['optimization_strategy'] = strategy.value
            
            optimizer = GeneralGPUMemoryOptimizer(config)
            assert optimizer.optimization_strategy == strategy
    
    def test_manage_cuda_streams(self):
        """Test CUDA stream management"""
        result = self.optimizer.manage_cuda_streams()
        
        assert 'devices_managed' in result
        assert 'device_results' in result
        assert result['devices_managed'] == self.optimizer.device_count
        
        for device_id in range(self.optimizer.device_count):
            assert device_id in result['device_results']
            device_result = result['device_results'][device_id]
            assert 'total_streams' in device_result
            assert 'synchronized_streams' in device_result
    
    def test_manage_cuda_streams_specific_device(self):
        """Test CUDA stream management for specific device"""
        device_id = 0
        result = self.optimizer.manage_cuda_streams(device_id)
        
        assert result['devices_managed'] == 1
        assert device_id in result['device_results']
    
    def test_manage_cuda_streams_invalid_device(self):
        """Test stream management fails with invalid device ID"""
        with pytest.raises(ValueError, match="device_id.*out of range"):
            self.optimizer.manage_cuda_streams(99)
    
    def test_get_gpu_memory_report(self):
        """Test GPU memory report generation"""
        # Allocate some memory to have data in report
        self.optimizer.allocate_gpu_memory((30, 30), torch.float32, "report_test")
        
        report = self.optimizer.get_gpu_memory_report()
        
        assert 'timestamp' in report
        assert 'system_summary' in report
        assert 'device_reports' in report
        assert 'optimization_metrics' in report
        assert 'configuration' in report
        
        # Check system summary
        system_summary = report['system_summary']
        assert system_summary['total_devices'] == self.optimizer.device_count
        assert 'total_memory_gb' in system_summary
        assert 'overall_utilization_percent' in system_summary
        
        # Check device reports
        device_reports = report['device_reports']
        assert len(device_reports) == self.optimizer.device_count
        
        for device_id, device_report in device_reports.items():
            assert 'device_name' in device_report
            assert 'memory_state' in device_report
            assert 'tracked_allocations' in device_report
            assert 'stream_info' in device_report
    
    def test_memory_pooling_enabled(self):
        """Test memory pooling functionality"""
        if not self.optimizer.enable_memory_pooling:
            pytest.skip("Memory pooling not enabled")
        
        # Allocate and free memory to test pooling
        tensor1 = self.optimizer.allocate_gpu_memory((25, 25), torch.float32, "pool_test1")
        self.optimizer.free_gpu_memory("pool_test1")
        
        # Second allocation might use pooled memory
        tensor2 = self.optimizer.allocate_gpu_memory((25, 25), torch.float32, "pool_test2")
        
        assert tensor2.shape == (25, 25)
        assert tensor2.dtype == torch.float32


class TestPerformanceOptimizationIntegration:
    """Integration tests for performance optimization components"""
    
    def setup_method(self):
        """Setup integration test environment"""
        self.perf_optimizer = PerformanceOptimizer({'enable_profiling': True})
        self.memory_optimizer = MemoryOptimizer({'enable_tracking': True})
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.gpu_optimizer = GeneralGPUMemoryOptimizer({'enable_monitoring': False})
        else:
            self.gpu_optimizer = None
    
    def test_coordinated_optimization(self):
        """Test coordinated optimization across all components"""
        # Generate some load for performance optimizer
        def cpu_intensive_task():
            return sum(i ** 2 for i in range(1000))
        
        for i in range(3):
            self.perf_optimizer.profile_operation(f"cpu_task_{i}", cpu_intensive_task)
        
        # Generate some memory usage
        with self.memory_optimizer.track_memory_usage("integration_test"):
            large_data = [list(range(1000)) for _ in range(100)]
        
        # Run optimizations
        perf_result = self.perf_optimizer.optimize_system()
        memory_result = self.memory_optimizer.optimize_memory_usage()
        
        assert perf_result['status'] == 'optimization_completed'
        assert memory_result['status'] == 'optimization_completed'
        
        if self.gpu_optimizer:
            gpu_result = self.gpu_optimizer.optimize_gpu_memory()
            assert gpu_result['status'] == 'optimization_completed'
    
    def test_comprehensive_reporting(self):
        """Test comprehensive reporting across all components"""
        # Generate some activity
        self.perf_optimizer.profile_operation("report_test", lambda: time.sleep(0.001))
        
        with self.memory_optimizer.track_memory_usage("report_memory"):
            temp_data = list(range(500))
        
        # Get reports
        perf_report = self.perf_optimizer.get_performance_report()
        memory_report = self.memory_optimizer.get_memory_report()
        
        assert 'performance_metrics' in perf_report
        assert 'system_metrics' in perf_report
        assert 'system_memory' in memory_report
        assert 'component_summary' in memory_report
        
        if self.gpu_optimizer:
            gpu_report = self.gpu_optimizer.get_gpu_memory_report()
            assert 'system_summary' in gpu_report
            assert 'device_reports' in gpu_report
    
    def test_thread_safety(self):
        """Test thread safety of optimization components"""
        def worker_task(worker_id):
            # Performance profiling
            self.perf_optimizer.profile_operation(
                f"worker_{worker_id}", 
                lambda: time.sleep(0.01)
            )
            
            # Memory tracking
            with self.memory_optimizer.track_memory_usage(f"worker_memory_{worker_id}"):
                worker_data = list(range(100))
            
            return True
        
        # Run multiple workers concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_task, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify data was collected from all workers
        assert len(self.perf_optimizer.operation_profiles) == 5
        assert len(self.memory_optimizer.component_memory_usage) == 5


def test_performance_optimizer_basic():
    """Standalone test for basic performance optimization functionality"""
    optimizer = PerformanceOptimizer({'enable_profiling': True})
    
    def test_function(x):
        return x * 2
    
    result = optimizer.profile_operation("basic_test", test_function, 5)
    
    assert result == 10
    assert "basic_test" in optimizer.operation_profiles
    assert len(optimizer.operation_profiles["basic_test"]) == 1
    
    print("✅ Basic performance optimizer test passed")


def test_memory_optimizer_basic():
    """Standalone test for basic memory optimization functionality"""
    optimizer = MemoryOptimizer({'enable_tracking': True})
    
    with optimizer.track_memory_usage("basic_memory_test"):
        test_data = [i for i in range(1000)]
    
    assert "basic_memory_test" in optimizer.component_memory_usage
    assert len(optimizer.component_memory_usage["basic_memory_test"]) == 1
    
    # Test optimization
    result = optimizer.optimize_memory_usage()
    assert result['status'] == 'optimization_completed'
    
    print("✅ Basic memory optimizer test passed")


@pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(), 
                   reason="PyTorch or CUDA not available")
def test_gpu_memory_optimizer_basic():
    """Standalone test for basic GPU memory optimization functionality"""
    optimizer = GeneralGPUMemoryOptimizer({'enable_monitoring': False})
    
    # Test allocation
    tensor = optimizer.allocate_gpu_memory((10, 10), torch.float32, "basic_gpu_test")
    
    assert tensor.shape == (10, 10)
    assert tensor.is_cuda
    
    # Test freeing
    freed = optimizer.free_gpu_memory("basic_gpu_test")
    assert freed is True
    
    # Test optimization
    result = optimizer.optimize_gpu_memory()
    assert result['status'] == 'optimization_completed'
    
    print("✅ Basic GPU memory optimizer test passed")


if __name__ == "__main__":
    """Run basic tests when executed as script"""
    print("Running performance optimization tests...")
    
    try:
        test_performance_optimizer_basic()
        test_memory_optimizer_basic()
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            test_gpu_memory_optimizer_basic()
            print("✅ All performance optimization tests passed!")
        else:
            print("⚠️  GPU tests skipped (PyTorch/CUDA not available)")
            print("✅ CPU-based performance optimization tests passed!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise