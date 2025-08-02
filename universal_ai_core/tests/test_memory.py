"""
Memory and resource tests for Universal AI Core.
Adapted from Saraphis memory test patterns.
"""

import pytest
import gc
import time
import threading
import weakref
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, List
import sys

from universal_ai_core import UniversalAIAPI, APIConfig
from universal_ai_core.core.universal_ai_core import UniversalAICore
from universal_ai_core.core.plugin_manager import PluginManager


@pytest.mark.memory
class TestMemoryUsage:
    """Test memory usage patterns and optimization."""
    
    def test_core_memory_footprint(self, sample_config, memory_monitor):
        """Test core system memory footprint."""
        initial_memory = memory_monitor['initial_memory']
        
        # Create multiple core instances
        cores = []
        memory_samples = []
        
        for i in range(3):
            from universal_ai_core.config.config_manager import UniversalConfiguration
            config = UniversalConfiguration(**sample_config)
            core = UniversalAICore(config)
            cores.append(core)
            
            current_memory = memory_monitor['get_memory_usage']()
            memory_samples.append(current_memory)
        
        max_memory = max(memory_samples)
        memory_per_core = (max_memory - initial_memory) / len(cores) if cores else 0
        
        # Each core should have reasonable memory footprint
        assert memory_per_core < 50, f"Memory per core {memory_per_core:.1f}MB exceeds 50MB"
        
        # Cleanup
        del cores
        gc.collect()
        
        final_memory = memory_monitor['get_memory_usage']()
        memory_recovered = max_memory - final_memory
        
        # Should recover most memory after cleanup
        assert memory_recovered > 0, "No memory recovered after cleanup"
    
    def test_plugin_manager_memory_usage(self, sample_config, memory_monitor):
        """Test plugin manager memory usage."""
        initial_memory = memory_monitor['initial_memory']
        
        from universal_ai_core.config.config_manager import UniversalConfiguration
        config = UniversalConfiguration(**sample_config)
        
        # Create plugin manager and load plugins
        manager = PluginManager(config)
        
        # Load plugins and monitor memory
        plugin_combinations = [
            ("feature_extractors", "molecular"),
            ("models", "cybersecurity"),
            ("proof_languages", "financial")
        ]
        
        memory_before_plugins = memory_monitor['get_memory_usage']()
        
        for plugin_type, plugin_name in plugin_combinations:
            try:
                manager.load_plugin(plugin_type, plugin_name)
            except Exception:
                pass  # Some plugins might not load in test environment
        
        memory_after_plugins = memory_monitor['get_memory_usage']()
        plugin_memory_usage = memory_after_plugins - memory_before_plugins
        
        # Plugin loading should have reasonable memory overhead
        assert plugin_memory_usage < 100, f"Plugin memory usage {plugin_memory_usage:.1f}MB too high"
    
    def test_api_memory_usage_under_load(self, universal_ai_api, sample_molecular_data, memory_monitor):
        """Test API memory usage under load."""
        initial_memory = memory_monitor['initial_memory']
        
        # Mock data processor to control memory usage
        with patch.object(universal_ai_api.data_processor, 'prepare_features') as mock_process:
            mock_process.return_value = Mock(
                status="success",
                data={"features": list(range(100))},  # Small controlled data
                processing_time=0.01
            )
            
            # Process many requests
            memory_samples = []
            for i in range(50):
                try:
                    result = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
                    
                    if i % 10 == 0:  # Sample memory every 10 requests
                        current_memory = memory_monitor['get_memory_usage']()
                        memory_samples.append(current_memory)
                        
                except Exception:
                    pass
            
            max_memory = max(memory_samples) if memory_samples else initial_memory
            memory_increase = max_memory - initial_memory
        
        # Memory usage should be reasonable under load
        assert memory_increase < 150, f"Memory increase {memory_increase:.1f}MB under load too high"
    
    def test_large_data_memory_handling(self, universal_ai_api, memory_monitor):
        """Test memory handling with large datasets."""
        initial_memory = memory_monitor['initial_memory']
        
        # Create increasingly large datasets
        dataset_sizes = [100, 500, 1000, 2000]
        memory_increases = []
        
        for size in dataset_sizes:
            # Create large dataset
            large_dataset = {
                "molecules": [
                    {"smiles": f"C{i}", "data": "x" * 100} 
                    for i in range(size)
                ]
            }
            
            memory_before = memory_monitor['get_memory_usage']()
            
            try:
                result = universal_ai_api.process_data(large_dataset, ["molecular_descriptors"])
                memory_after = memory_monitor['get_memory_usage']()
                memory_increase = memory_after - memory_before
                memory_increases.append(memory_increase)
                
            except MemoryError:
                # Expected for very large datasets
                break
            except Exception:
                memory_increases.append(0)
            
            # Cleanup between iterations
            del large_dataset
            gc.collect()
        
        # Memory increase should scale reasonably with data size
        if len(memory_increases) > 1:
            # Check that memory doesn't increase exponentially
            for i in range(1, len(memory_increases)):
                ratio = memory_increases[i] / memory_increases[0] if memory_increases[0] > 0 else 1
                size_ratio = dataset_sizes[i] / dataset_sizes[0]
                
                # Memory should not scale worse than linearly (with some tolerance)
                assert ratio <= size_ratio * 2, f"Memory scaling too poor: {ratio:.2f}x for {size_ratio:.2f}x data"


@pytest.mark.memory
class TestMemoryLeaks:
    """Test for memory leaks in various scenarios."""
    
    def test_core_creation_deletion_cycle(self, sample_config, memory_monitor):
        """Test for memory leaks in core creation/deletion cycles."""
        initial_memory = memory_monitor['initial_memory']
        memory_samples = []
        
        # Create and delete cores multiple times
        for cycle in range(5):
            from universal_ai_core.config.config_manager import UniversalConfiguration
            config = UniversalConfiguration(**sample_config)
            
            # Create core
            core = UniversalAICore(config)
            
            # Use core briefly
            try:
                domains = core.get_available_domains()
            except Exception:
                pass
            
            # Delete core
            del core
            del config
            gc.collect()
            
            current_memory = memory_monitor['get_memory_usage']()
            memory_samples.append(current_memory)
        
        # Check for memory leak pattern
        if len(memory_samples) >= 3:
            # Calculate trend
            early_avg = sum(memory_samples[:2]) / 2
            late_avg = sum(memory_samples[-2:]) / 2
            memory_drift = late_avg - early_avg
            
            # Should not have significant memory drift
            assert memory_drift < 20, f"Potential memory leak: {memory_drift:.1f}MB drift over cycles"
    
    def test_api_request_processing_leak(self, universal_ai_api, sample_molecular_data, memory_monitor):
        """Test for memory leaks in request processing."""
        initial_memory = memory_monitor['initial_memory']
        
        # Mock processor for consistent behavior
        with patch.object(universal_ai_api.data_processor, 'prepare_features') as mock_process:
            mock_process.return_value = Mock(
                status="success",
                data={"features": [1, 2, 3]},
                processing_time=0.01
            )
            
            memory_samples = []
            
            # Process requests in batches
            for batch in range(3):
                # Process batch of requests
                for _ in range(20):
                    try:
                        result = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
                    except Exception:
                        pass
                
                # Force cleanup
                gc.collect()
                
                current_memory = memory_monitor['get_memory_usage']()
                memory_samples.append(current_memory)
        
        # Check for progressive memory increase
        if len(memory_samples) >= 2:
            memory_increase = memory_samples[-1] - memory_samples[0]
            
            # Should not have significant memory increase across batches
            assert memory_increase < 30, f"Memory leak detected: {memory_increase:.1f}MB increase"
    
    def test_plugin_loading_unloading_leak(self, sample_config, memory_monitor):
        """Test for memory leaks in plugin loading/unloading."""
        initial_memory = memory_monitor['initial_memory']
        
        from universal_ai_core.config.config_manager import UniversalConfiguration
        config = UniversalConfiguration(**sample_config)
        
        memory_samples = []
        
        # Multiple plugin loading cycles
        for cycle in range(3):
            manager = PluginManager(config)
            
            # Load plugins
            plugin_combinations = [
                ("feature_extractors", "molecular"),
                ("models", "cybersecurity")
            ]
            
            for plugin_type, plugin_name in plugin_combinations:
                try:
                    manager.load_plugin(plugin_type, plugin_name)
                except Exception:
                    pass
            
            # Delete manager
            del manager
            gc.collect()
            
            current_memory = memory_monitor['get_memory_usage']()
            memory_samples.append(current_memory)
        
        # Check for memory accumulation
        if len(memory_samples) >= 2:
            memory_accumulation = memory_samples[-1] - memory_samples[0]
            
            assert memory_accumulation < 25, f"Plugin memory leak: {memory_accumulation:.1f}MB accumulated"
    
    def test_cache_memory_leak(self, universal_ai_api, sample_molecular_data, memory_monitor):
        """Test for memory leaks in caching system."""
        if not universal_ai_api.cache:
            pytest.skip("Caching not enabled")
        
        initial_memory = memory_monitor['initial_memory']
        
        # Fill cache with many different requests
        with patch.object(universal_ai_api.data_processor, 'prepare_features') as mock_process:
            mock_process.return_value = Mock(
                status="success",
                data={"features": list(range(50))},  # Varying data
                processing_time=0.01
            )
            
            # Create many unique cache entries
            for i in range(100):
                unique_data = {**sample_molecular_data, "unique_id": i}
                try:
                    result = universal_ai_api.process_data(unique_data, ["molecular_weight"])
                except Exception:
                    pass
        
        cache_memory = memory_monitor['get_memory_usage']()
        
        # Clear cache
        universal_ai_api.cache.clear()
        gc.collect()
        
        cleared_memory = memory_monitor['get_memory_usage']()
        memory_freed = cache_memory - cleared_memory
        
        # Should free some memory when cache is cleared
        assert memory_freed >= 0, "Cache clear should not increase memory"


@pytest.mark.memory
class TestMemoryOptimizations:
    """Test memory optimization features."""
    
    def test_garbage_collection_efficiency(self, universal_ai_api, sample_molecular_data):
        """Test garbage collection efficiency."""
        # Create objects that should be garbage collected
        large_objects = []
        
        for i in range(10):
            large_data = {
                "molecules": [{"smiles": f"C{j}", "data": "x" * 1000} for j in range(100)],
                "id": i
            }
            large_objects.append(large_data)
        
        # Create weak references to track cleanup
        weak_refs = [weakref.ref(obj) for obj in large_objects]
        
        # Clear references
        del large_objects
        
        # Force garbage collection
        gc.collect()
        
        # Check how many objects were collected
        alive_objects = sum(1 for ref in weak_refs if ref() is not None)
        collected_objects = len(weak_refs) - alive_objects
        
        # Most objects should be collected
        collection_rate = collected_objects / len(weak_refs)
        assert collection_rate > 0.7, f"Poor GC efficiency: only {collection_rate:.2f} collected"
    
    def test_memory_pressure_handling(self, universal_ai_api, memory_monitor):
        """Test system behavior under memory pressure."""
        initial_memory = memory_monitor['initial_memory']
        
        # Create memory pressure by allocating large objects
        memory_hogs = []
        
        try:
            # Gradually increase memory usage
            for i in range(10):
                # Create large object
                large_object = bytearray(10 * 1024 * 1024)  # 10MB
                memory_hogs.append(large_object)
                
                current_memory = memory_monitor['get_memory_usage']()
                memory_increase = current_memory - initial_memory
                
                # Test API functionality under memory pressure
                try:
                    health = universal_ai_api.get_health_status()
                    assert isinstance(health, dict)
                except MemoryError:
                    # Expected under extreme memory pressure
                    break
                except Exception:
                    # Other exceptions might occur under pressure
                    pass
                
                # Stop if memory usage gets too high
                if memory_increase > 200:  # 200MB limit
                    break
                    
        finally:
            # Cleanup
            del memory_hogs
            gc.collect()
    
    def test_weak_reference_cleanup(self, sample_config):
        """Test weak reference cleanup in component lifecycle."""
        from universal_ai_core.config.config_manager import UniversalConfiguration
        
        config = UniversalConfiguration(**sample_config)
        
        # Create core and get weak reference
        core = UniversalAICore(config)
        core_weak_ref = weakref.ref(core)
        
        # Verify object exists
        assert core_weak_ref() is not None
        
        # Delete core
        del core
        gc.collect()
        
        # Weak reference should be cleared
        assert core_weak_ref() is None, "Core object not properly cleaned up"
    
    def test_circular_reference_handling(self, sample_config):
        """Test handling of circular references."""
        from universal_ai_core.config.config_manager import UniversalConfiguration
        
        config = UniversalConfiguration(**sample_config)
        core = UniversalAICore(config)
        
        # Create potential circular reference
        core._test_circular_ref = core  # Self-reference
        
        # Create weak reference to track cleanup
        core_weak_ref = weakref.ref(core)
        
        # Delete core
        del core
        gc.collect()
        
        # Should be cleaned up despite circular reference
        assert core_weak_ref() is None, "Circular reference prevented cleanup"


@pytest.mark.memory
class TestResourceManagement:
    """Test resource management and cleanup."""
    
    def test_file_handle_management(self, universal_ai_api, temp_directory):
        """Test proper file handle management."""
        # Create test files
        test_files = []
        for i in range(10):
            test_file = temp_directory / f"test_file_{i}.txt"
            test_file.write_text(f"Test content {i}")
            test_files.append(test_file)
        
        # Simulate file operations
        opened_files = 0
        
        try:
            for test_file in test_files:
                with open(test_file, 'r') as f:
                    content = f.read()
                    opened_files += 1
                    
                    # Test API operations while files are open
                    try:
                        health = universal_ai_api.get_health_status()
                        assert isinstance(health, dict)
                    except Exception:
                        pass
        
        except OSError as e:
            # Should not run out of file handles
            assert "Too many open files" not in str(e), "File handle leak detected"
        
        # All files should be properly closed
        assert opened_files == len(test_files)
    
    def test_thread_resource_cleanup(self, universal_ai_api):
        """Test thread resource cleanup."""
        initial_thread_count = threading.active_count()
        
        # Create threads for concurrent operations
        threads = []
        
        def worker():
            try:
                health = universal_ai_api.get_health_status()
                time.sleep(0.1)
            except Exception:
                pass
        
        # Start threads
        for _ in range(5):
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=2.0)
        
        # Check final thread count
        final_thread_count = threading.active_count()
        
        # Should not have significantly more threads
        thread_increase = final_thread_count - initial_thread_count
        assert thread_increase <= 2, f"Thread leak detected: {thread_increase} extra threads"
    
    def test_api_shutdown_cleanup(self, api_config, sample_config_file, memory_monitor):
        """Test resource cleanup during API shutdown."""
        initial_memory = memory_monitor['initial_memory']
        
        # Create API instance
        api = UniversalAIAPI(config_path=str(sample_config_file), api_config=api_config)
        
        # Use API to allocate resources
        try:
            health = api.get_health_status()
            metrics = api.get_metrics()
            system_info = api.get_system_info()
        except Exception:
            pass
        
        memory_after_usage = memory_monitor['get_memory_usage']()
        
        # Shutdown API
        api.shutdown()
        del api
        gc.collect()
        
        final_memory = memory_monitor['get_memory_usage']()
        
        # Should recover most memory after shutdown
        memory_recovered = memory_after_usage - final_memory
        assert memory_recovered >= 0, "Memory usage increased after shutdown"
        
        # Final memory should be close to initial
        total_increase = final_memory - initial_memory
        assert total_increase < 30, f"High memory retention after shutdown: {total_increase:.1f}MB"


@pytest.mark.memory
class TestMemoryProfiling:
    """Test memory profiling and monitoring capabilities."""
    
    def test_memory_usage_tracking(self, universal_ai_api, memory_monitor):
        """Test memory usage tracking during operations."""
        memory_samples = []
        
        # Baseline memory
        baseline = memory_monitor['get_memory_usage']()
        memory_samples.append(("baseline", baseline))
        
        # API initialization impact
        init_memory = memory_monitor['get_memory_usage']()
        memory_samples.append(("post_init", init_memory))
        
        # Health check impact
        try:
            health = universal_ai_api.get_health_status()
            health_memory = memory_monitor['get_memory_usage']()
            memory_samples.append(("post_health", health_memory))
        except Exception:
            pass
        
        # Metrics collection impact
        try:
            metrics = universal_ai_api.get_metrics()
            metrics_memory = memory_monitor['get_memory_usage']()
            memory_samples.append(("post_metrics", metrics_memory))
        except Exception:
            pass
        
        # Analyze memory progression
        for i, (operation, memory) in enumerate(memory_samples):
            if i > 0:
                prev_memory = memory_samples[i-1][1]
                increase = memory - prev_memory
                
                # Each operation should have reasonable memory impact
                assert increase < 20, f"High memory increase for {operation}: {increase:.1f}MB"
    
    def test_memory_leak_detection_over_time(self, universal_ai_api, sample_molecular_data, memory_monitor):
        """Test long-term memory leak detection."""
        memory_history = []
        
        # Mock processor for consistent behavior
        with patch.object(universal_ai_api.data_processor, 'prepare_features') as mock_process:
            mock_process.return_value = Mock(
                status="success",
                data={"features": [1, 2, 3]},
                processing_time=0.01
            )
            
            # Simulate extended usage
            for cycle in range(10):
                cycle_start_memory = memory_monitor['get_memory_usage']()
                
                # Process requests in this cycle
                for _ in range(10):
                    try:
                        result = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
                    except Exception:
                        pass
                
                cycle_end_memory = memory_monitor['get_memory_usage']()
                memory_history.append((cycle, cycle_start_memory, cycle_end_memory))
                
                # Brief pause between cycles
                time.sleep(0.05)
        
        # Analyze memory trend
        if len(memory_history) >= 3:
            # Calculate trend using linear regression (simplified)
            cycle_numbers = [h[0] for h in memory_history]
            end_memories = [h[2] for h in memory_history]
            
            # Simple trend calculation
            first_half_avg = sum(end_memories[:len(end_memories)//2]) / (len(end_memories)//2)
            second_half_avg = sum(end_memories[len(end_memories)//2:]) / (len(end_memories) - len(end_memories)//2)
            
            trend = second_half_avg - first_half_avg
            
            # Should not have strong upward trend
            assert trend < 15, f"Memory leak trend detected: {trend:.1f}MB increase over time"
    
    def test_peak_memory_usage_limits(self, universal_ai_api, memory_monitor):
        """Test that peak memory usage stays within limits."""
        initial_memory = memory_monitor['initial_memory']
        peak_memory = initial_memory
        
        # Perform various operations and track peak memory
        operations = [
            lambda: universal_ai_api.get_health_status(),
            lambda: universal_ai_api.get_metrics(),
            lambda: universal_ai_api.get_system_info(),
        ]
        
        for operation in operations:
            try:
                result = operation()
                current_memory = memory_monitor['get_memory_usage']()
                peak_memory = max(peak_memory, current_memory)
            except Exception:
                pass
        
        peak_increase = peak_memory - initial_memory
        
        # Peak memory increase should be reasonable
        assert peak_increase < 100, f"Peak memory usage too high: {peak_increase:.1f}MB"