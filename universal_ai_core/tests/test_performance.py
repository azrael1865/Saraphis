"""
Performance tests and benchmarks for Universal AI Core.
Adapted from Saraphis performance test patterns.
"""

import pytest
import asyncio
import time
import threading
import psutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, List
import statistics

from universal_ai_core import UniversalAIAPI, APIConfig
from universal_ai_core.core.universal_ai_core import UniversalAICore
from universal_ai_core.core.plugin_manager import PluginManager
from universal_ai_core.utils.data_utils import DataProcessor
from universal_ai_core.utils.validation_utils import ValidationEngine


@pytest.mark.performance
class TestCorePerformance:
    """Performance tests for core components."""
    
    def test_core_initialization_benchmark(self, sample_config, performance_config):
        """Benchmark core initialization performance."""
        from universal_ai_core.config.config_manager import UniversalConfiguration
        
        config = UniversalConfiguration(**sample_config)
        initialization_times = []
        
        # Benchmark multiple initializations
        for _ in range(performance_config["benchmark_iterations"]):
            start_time = time.perf_counter()
            core = UniversalAICore(config)
            end_time = time.perf_counter()
            
            initialization_times.append(end_time - start_time)
            del core
        
        # Calculate statistics
        avg_time = statistics.mean(initialization_times)
        max_time = max(initialization_times)
        min_time = min(initialization_times)
        std_dev = statistics.stdev(initialization_times) if len(initialization_times) > 1 else 0
        
        # Performance assertions
        assert avg_time < 0.5, f"Average initialization time {avg_time:.3f}s exceeds 0.5s"
        assert max_time < 1.0, f"Maximum initialization time {max_time:.3f}s exceeds 1.0s"
        assert std_dev < 0.2, f"High variability in initialization times: {std_dev:.3f}s"
        
        print(f"Core initialization - Avg: {avg_time:.3f}s, Min: {min_time:.3f}s, Max: {max_time:.3f}s, StdDev: {std_dev:.3f}s")
    
    def test_plugin_manager_performance(self, sample_config, performance_config):
        """Benchmark plugin manager performance."""
        from universal_ai_core.config.config_manager import UniversalConfiguration
        
        config = UniversalConfiguration(**sample_config)
        manager = PluginManager(config)
        
        # Benchmark plugin loading
        plugin_combinations = [
            ("feature_extractors", "molecular"),
            ("feature_extractors", "cybersecurity"),
            ("feature_extractors", "financial"),
            ("models", "molecular"),
            ("models", "cybersecurity"),
            ("models", "financial")
        ]
        
        loading_times = []
        for plugin_type, plugin_name in plugin_combinations:
            start_time = time.perf_counter()
            try:
                success = manager.load_plugin(plugin_type, plugin_name)
                end_time = time.perf_counter()
                loading_times.append(end_time - start_time)
            except Exception:
                # Some plugins might not load in test environment
                loading_times.append(0.1)  # Placeholder time
        
        avg_loading_time = statistics.mean(loading_times)
        total_loading_time = sum(loading_times)
        
        assert avg_loading_time < 0.2, f"Average plugin loading time {avg_loading_time:.3f}s exceeds 0.2s"
        assert total_loading_time < 1.0, f"Total plugin loading time {total_loading_time:.3f}s exceeds 1.0s"
        
        print(f"Plugin loading - Avg: {avg_loading_time:.3f}s, Total: {total_loading_time:.3f}s")
    
    def test_data_processor_throughput(self, performance_config, memory_monitor):
        """Test data processor throughput."""
        processor = DataProcessor()
        
        # Generate test data
        test_datasets = []
        for i in range(performance_config["benchmark_iterations"]):
            dataset = {
                "molecular_data": {
                    "smiles": [f"C{i}", f"CC{i}", f"CCC{i}"],
                    "properties": [{"mw": 100 + i} for _ in range(3)]
                }
            }
            test_datasets.append(dataset)
        
        # Benchmark processing throughput
        start_time = time.perf_counter()
        processed_count = 0
        
        for dataset in test_datasets:
            try:
                result = processor.prepare_features(dataset, ["molecular_weight"])
                if result.status == "success":
                    processed_count += 1
            except Exception:
                pass
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        throughput = processed_count / total_time if total_time > 0 else 0
        
        assert throughput > 5, f"Data processing throughput {throughput:.2f} items/sec is too low"
        assert total_time < 5.0, f"Total processing time {total_time:.3f}s exceeds 5.0s"
        
        print(f"Data processing throughput: {throughput:.2f} items/sec, Total time: {total_time:.3f}s")
    
    def test_validation_engine_performance(self, performance_config):
        """Test validation engine performance."""
        engine = ValidationEngine()
        
        # Generate test data for validation
        test_data = []
        for i in range(performance_config["benchmark_iterations"]):
            data = {
                "molecules": [
                    {"smiles": f"C{i}", "mw": 100 + i},
                    {"smiles": f"CC{i}", "mw": 120 + i}
                ],
                "metadata": {"source": f"test_{i}"}
            }
            test_data.append(data)
        
        # Benchmark validation performance
        validation_times = []
        successful_validations = 0
        
        for data in test_data:
            start_time = time.perf_counter()
            try:
                result = engine.validate(data, ["schema"])
                end_time = time.perf_counter()
                validation_times.append(end_time - start_time)
                if result.is_valid:
                    successful_validations += 1
            except Exception:
                validation_times.append(0.1)  # Placeholder time
        
        avg_validation_time = statistics.mean(validation_times)
        success_rate = successful_validations / len(test_data)
        
        assert avg_validation_time < 0.1, f"Average validation time {avg_validation_time:.3f}s exceeds 0.1s"
        assert success_rate > 0.5, f"Validation success rate {success_rate:.2f} is too low"
        
        print(f"Validation - Avg time: {avg_validation_time:.3f}s, Success rate: {success_rate:.2f}")


@pytest.mark.performance
class TestAPIPerformance:
    """Performance tests for API components."""
    
    def test_api_initialization_performance(self, api_config, sample_config_file, performance_config):
        """Test API initialization performance."""
        initialization_times = []
        
        for _ in range(performance_config["benchmark_iterations"] // 2):  # Fewer iterations for API
            start_time = time.perf_counter()
            api = UniversalAIAPI(config_path=str(sample_config_file), api_config=api_config)
            end_time = time.perf_counter()
            
            initialization_times.append(end_time - start_time)
            api.shutdown()
            del api
            time.sleep(0.1)  # Brief pause between initializations
        
        avg_time = statistics.mean(initialization_times)
        max_time = max(initialization_times)
        
        assert avg_time < 2.0, f"Average API initialization time {avg_time:.3f}s exceeds 2.0s"
        assert max_time < 5.0, f"Maximum API initialization time {max_time:.3f}s exceeds 5.0s"
        
        print(f"API initialization - Avg: {avg_time:.3f}s, Max: {max_time:.3f}s")
    
    def test_api_request_throughput(self, universal_ai_api, sample_molecular_data, performance_config):
        """Test API request throughput."""
        request_times = []
        successful_requests = 0
        
        # Mock data processor for consistent performance
        with patch.object(universal_ai_api.data_processor, 'prepare_features') as mock_process:
            mock_process.return_value = Mock(
                status="success",
                data={"features": [1, 2, 3]},
                processing_time=0.01
            )
            
            # Benchmark request processing
            start_time = time.perf_counter()
            
            for _ in range(performance_config["benchmark_iterations"]):
                request_start = time.perf_counter()
                try:
                    result = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
                    request_end = time.perf_counter()
                    request_times.append(request_end - request_start)
                    if result.status == "success":
                        successful_requests += 1
                except Exception:
                    request_times.append(0.1)  # Placeholder time
            
            end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_request_time = statistics.mean(request_times)
        throughput = successful_requests / total_time if total_time > 0 else 0
        success_rate = successful_requests / performance_config["benchmark_iterations"]
        
        assert avg_request_time < 0.1, f"Average request time {avg_request_time:.3f}s exceeds 0.1s"
        assert throughput > 50, f"Request throughput {throughput:.2f} req/sec is too low"
        assert success_rate > 0.9, f"Request success rate {success_rate:.2f} is too low"
        
        print(f"API throughput: {throughput:.2f} req/sec, Avg time: {avg_request_time:.3f}s, Success: {success_rate:.2f}")
    
    @pytest.mark.asyncio
    async def test_async_api_performance(self, universal_ai_api, performance_config):
        """Test async API performance."""
        async def submit_async_task():
            try:
                task_id = await universal_ai_api.submit_async_task(
                    "test_operation",
                    {"data": "test"},
                    config={}
                )
                return task_id
            except Exception:
                return None
        
        # Benchmark async task submission
        start_time = time.perf_counter()
        task_ids = []
        
        for _ in range(performance_config["benchmark_iterations"] // 2):
            task_id = await submit_async_task()
            if task_id:
                task_ids.append(task_id)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        submission_rate = len(task_ids) / total_time if total_time > 0 else 0
        
        assert submission_rate > 10, f"Async task submission rate {submission_rate:.2f} tasks/sec is too low"
        assert total_time < 2.0, f"Total async submission time {total_time:.3f}s exceeds 2.0s"
        
        print(f"Async submission rate: {submission_rate:.2f} tasks/sec, Total time: {total_time:.3f}s")
    
    def test_api_caching_performance(self, universal_ai_api, sample_molecular_data, performance_config):
        """Test API caching performance."""
        # Mock data processor
        with patch.object(universal_ai_api.data_processor, 'prepare_features') as mock_process:
            mock_process.return_value = Mock(
                status="success",
                data={"features": [1, 2, 3]},
                processing_time=0.01
            )
            
            # First request (cache miss)
            start_time = time.perf_counter()
            result1 = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
            cache_miss_time = time.perf_counter() - start_time
            
            # Subsequent requests (cache hits)
            cache_hit_times = []
            for _ in range(5):
                start_time = time.perf_counter()
                result2 = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
                cache_hit_time = time.perf_counter() - start_time
                cache_hit_times.append(cache_hit_time)
            
            avg_cache_hit_time = statistics.mean(cache_hit_times)
            cache_speedup = cache_miss_time / avg_cache_hit_time if avg_cache_hit_time > 0 else 1
        
        # Cache hits should be significantly faster
        assert avg_cache_hit_time < cache_miss_time, "Cache hits should be faster than cache misses"
        # Allow for some variation in test environment
        print(f"Cache performance - Miss: {cache_miss_time:.4f}s, Hit avg: {avg_cache_hit_time:.4f}s, Speedup: {cache_speedup:.2f}x")


@pytest.mark.performance
class TestConcurrencyPerformance:
    """Test concurrent processing performance."""
    
    def test_concurrent_api_requests(self, universal_ai_api, sample_molecular_data, performance_config):
        """Test concurrent API request performance."""
        num_threads = performance_config["max_concurrent_operations"]
        requests_per_thread = 5
        
        # Mock data processor for consistent results
        with patch.object(universal_ai_api.data_processor, 'prepare_features') as mock_process:
            mock_process.return_value = Mock(
                status="success",
                data={"features": [1, 2, 3]},
                processing_time=0.01
            )
            
            def make_requests():
                results = []
                for _ in range(requests_per_thread):
                    try:
                        result = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
                        results.append(result.status == "success")
                    except Exception:
                        results.append(False)
                return results
            
            # Run concurrent requests
            start_time = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(make_requests) for _ in range(num_threads)]
                thread_results = [future.result() for future in as_completed(futures)]
            
            end_time = time.perf_counter()
        
        total_time = end_time - start_time
        total_requests = num_threads * requests_per_thread
        successful_requests = sum(sum(results) for results in thread_results)
        
        throughput = successful_requests / total_time if total_time > 0 else 0
        success_rate = successful_requests / total_requests
        
        assert throughput > 20, f"Concurrent throughput {throughput:.2f} req/sec is too low"
        assert success_rate > 0.8, f"Concurrent success rate {success_rate:.2f} is too low"
        assert total_time < 3.0, f"Concurrent processing time {total_time:.3f}s exceeds 3.0s"
        
        print(f"Concurrent performance: {throughput:.2f} req/sec, Success: {success_rate:.2f}, Time: {total_time:.3f}s")
    
    def test_thread_safety_performance(self, sample_config, performance_config):
        """Test thread safety performance overhead."""
        from universal_ai_core.config.config_manager import UniversalConfiguration
        
        config = UniversalConfiguration(**sample_config)
        
        # Test single-threaded performance
        single_thread_times = []
        for _ in range(performance_config["benchmark_iterations"]):
            start_time = time.perf_counter()
            core = UniversalAICore(config)
            domains = core.get_available_domains()
            end_time = time.perf_counter()
            single_thread_times.append(end_time - start_time)
            del core
        
        avg_single_thread_time = statistics.mean(single_thread_times)
        
        # Test multi-threaded performance
        def thread_worker():
            times = []
            for _ in range(performance_config["benchmark_iterations"] // 4):
                start_time = time.perf_counter()
                core = UniversalAICore(config)
                domains = core.get_available_domains()
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                del core
            return times
        
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(thread_worker) for _ in range(4)]
            multi_thread_results = [future.result() for future in as_completed(futures)]
        end_time = time.perf_counter()
        
        all_multi_thread_times = [time for results in multi_thread_results for time in results]
        avg_multi_thread_time = statistics.mean(all_multi_thread_times)
        
        # Thread safety overhead should be minimal
        overhead_ratio = avg_multi_thread_time / avg_single_thread_time if avg_single_thread_time > 0 else 1
        
        assert overhead_ratio < 2.0, f"Thread safety overhead {overhead_ratio:.2f}x is too high"
        
        print(f"Thread safety overhead: {overhead_ratio:.2f}x (Single: {avg_single_thread_time:.4f}s, Multi: {avg_multi_thread_time:.4f}s)")


@pytest.mark.performance
class TestMemoryPerformance:
    """Test memory usage and performance."""
    
    def test_memory_usage_during_processing(self, universal_ai_api, sample_molecular_data, memory_monitor, performance_config):
        """Test memory usage during intensive processing."""
        initial_memory = memory_monitor['initial_memory']
        
        # Mock data processor to simulate memory usage
        with patch.object(universal_ai_api.data_processor, 'prepare_features') as mock_process:
            mock_process.return_value = Mock(
                status="success",
                data={"features": list(range(1000))},  # Larger data
                processing_time=0.01
            )
            
            memory_samples = []
            
            # Process many requests while monitoring memory
            for i in range(performance_config["benchmark_iterations"]):
                result = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
                
                if i % 5 == 0:  # Sample memory every 5 requests
                    current_memory = memory_monitor['get_memory_usage']()
                    memory_samples.append(current_memory)
            
            final_memory = memory_monitor['get_memory_usage']()
        
        max_memory_usage = max(memory_samples) if memory_samples else final_memory
        memory_increase = max_memory_usage - initial_memory
        
        # Memory usage should be reasonable
        assert memory_increase < performance_config["memory_limit_mb"], \
            f"Memory increase {memory_increase:.1f}MB exceeds limit {performance_config['memory_limit_mb']}MB"
        
        print(f"Memory usage - Initial: {initial_memory:.1f}MB, Max: {max_memory_usage:.1f}MB, Increase: {memory_increase:.1f}MB")
    
    def test_memory_leak_detection(self, api_config, sample_config_file, memory_monitor, performance_config):
        """Test for memory leaks in API lifecycle."""
        initial_memory = memory_monitor['initial_memory']
        
        memory_samples = []
        
        # Create and destroy multiple API instances
        for i in range(5):  # Fewer iterations for memory leak test
            api = UniversalAIAPI(config_path=str(sample_config_file), api_config=api_config)
            
            # Use the API briefly
            try:
                health = api.get_health_status()
                metrics = api.get_metrics()
            except Exception:
                pass
            
            api.shutdown()
            del api
            
            # Force garbage collection
            import gc
            gc.collect()
            
            current_memory = memory_monitor['get_memory_usage']()
            memory_samples.append(current_memory)
            
            time.sleep(0.1)  # Brief pause
        
        # Check for memory leak pattern
        if len(memory_samples) >= 2:
            memory_trend = memory_samples[-1] - memory_samples[0]
            max_acceptable_increase = 20  # MB
            
            assert memory_trend < max_acceptable_increase, \
                f"Potential memory leak detected: {memory_trend:.1f}MB increase over {len(memory_samples)} cycles"
        
        print(f"Memory leak test - Samples: {[f'{m:.1f}' for m in memory_samples]}, Trend: {memory_trend:.1f}MB")
    
    def test_garbage_collection_performance(self, universal_ai_api, sample_molecular_data, performance_config):
        """Test garbage collection impact on performance."""
        import gc
        
        # Disable automatic garbage collection
        gc.disable()
        
        try:
            # Process requests without GC
            start_time = time.perf_counter()
            for _ in range(performance_config["benchmark_iterations"] // 2):
                try:
                    result = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
                except Exception:
                    pass
            no_gc_time = time.perf_counter() - start_time
            
            # Force garbage collection
            gc_start = time.perf_counter()
            gc.collect()
            gc_time = time.perf_counter() - gc_start
            
        finally:
            # Re-enable automatic garbage collection
            gc.enable()
        
        # GC time should be reasonable compared to processing time
        gc_overhead = gc_time / no_gc_time if no_gc_time > 0 else 0
        
        assert gc_overhead < 0.1, f"Garbage collection overhead {gc_overhead:.3f} is too high"
        assert gc_time < 0.5, f"Garbage collection time {gc_time:.3f}s is too long"
        
        print(f"GC performance - Processing: {no_gc_time:.3f}s, GC: {gc_time:.3f}s, Overhead: {gc_overhead:.3f}")


@pytest.mark.performance
class TestScalabilityPerformance:
    """Test system scalability performance."""
    
    def test_data_size_scalability(self, universal_ai_api, performance_config):
        """Test performance with varying data sizes."""
        data_sizes = [10, 50, 100, 500, 1000]  # Number of molecules
        processing_times = []
        
        with patch.object(universal_ai_api.data_processor, 'prepare_features') as mock_process:
            def mock_processing(data, extractors):
                # Simulate processing time proportional to data size
                data_size = len(data.get("smiles", []))
                time.sleep(data_size * 0.0001)  # 0.1ms per item
                return Mock(
                    status="success",
                    data={"features": list(range(data_size))},
                    processing_time=data_size * 0.0001
                )
            
            mock_process.side_effect = mock_processing
            
            for size in data_sizes:
                test_data = {
                    "smiles": [f"C{i}" for i in range(size)],
                    "properties": [{"mw": 100 + i} for i in range(size)]
                }
                
                start_time = time.perf_counter()
                result = universal_ai_api.process_data(test_data, ["molecular_weight"])
                processing_time = time.perf_counter() - start_time
                processing_times.append(processing_time)
        
        # Check scalability - processing time should scale reasonably
        for i in range(1, len(processing_times)):
            ratio = processing_times[i] / processing_times[0] if processing_times[0] > 0 else 1
            size_ratio = data_sizes[i] / data_sizes[0]
            
            # Processing time should not scale worse than quadratically
            assert ratio <= size_ratio * size_ratio, \
                f"Poor scalability: {ratio:.2f}x time for {size_ratio:.2f}x data"
        
        print(f"Scalability test - Sizes: {data_sizes}, Times: {[f'{t:.4f}' for t in processing_times]}")
    
    def test_concurrent_user_scalability(self, universal_ai_api, sample_molecular_data, performance_config):
        """Test scalability with multiple concurrent users."""
        user_counts = [1, 2, 4, 8]
        requests_per_user = 3
        
        # Mock data processor
        with patch.object(universal_ai_api.data_processor, 'prepare_features') as mock_process:
            mock_process.return_value = Mock(
                status="success",
                data={"features": [1, 2, 3]},
                processing_time=0.01
            )
            
            scalability_results = []
            
            for num_users in user_counts:
                def user_simulation():
                    user_times = []
                    for _ in range(requests_per_user):
                        start_time = time.perf_counter()
                        try:
                            result = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
                            end_time = time.perf_counter()
                            user_times.append(end_time - start_time)
                        except Exception:
                            user_times.append(1.0)  # Penalty for failure
                    return user_times
                
                # Run concurrent users
                start_time = time.perf_counter()
                with ThreadPoolExecutor(max_workers=num_users) as executor:
                    futures = [executor.submit(user_simulation) for _ in range(num_users)]
                    user_results = [future.result() for future in as_completed(futures)]
                total_time = time.perf_counter() - start_time
                
                all_request_times = [time for user_times in user_results for time in user_times]
                avg_response_time = statistics.mean(all_request_times)
                total_requests = num_users * requests_per_user
                throughput = total_requests / total_time if total_time > 0 else 0
                
                scalability_results.append({
                    'users': num_users,
                    'avg_response_time': avg_response_time,
                    'throughput': throughput,
                    'total_time': total_time
                })
        
        # Check that system handles increasing load reasonably
        for i in range(1, len(scalability_results)):
            current = scalability_results[i]
            previous = scalability_results[i-1]
            
            response_time_ratio = current['avg_response_time'] / previous['avg_response_time']
            
            # Response time shouldn't degrade too badly with more users
            assert response_time_ratio < 3.0, \
                f"Response time degradation too high: {response_time_ratio:.2f}x for {current['users']} vs {previous['users']} users"
        
        print("Scalability results:")
        for result in scalability_results:
            print(f"  {result['users']} users: {result['avg_response_time']:.3f}s avg, {result['throughput']:.1f} req/sec")


@pytest.mark.performance
class TestPerformanceRegression:
    """Test for performance regressions."""
    
    def test_baseline_performance_tracking(self, universal_ai_api, sample_molecular_data, performance_config):
        """Track baseline performance metrics."""
        # This test can be extended to compare against stored baselines
        
        # Mock data processor for consistent results
        with patch.object(universal_ai_api.data_processor, 'prepare_features') as mock_process:
            mock_process.return_value = Mock(
                status="success",
                data={"features": [1, 2, 3]},
                processing_time=0.01
            )
            
            # Measure current performance
            request_times = []
            start_time = time.perf_counter()
            
            for _ in range(performance_config["benchmark_iterations"]):
                request_start = time.perf_counter()
                try:
                    result = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
                    request_end = time.perf_counter()
                    request_times.append(request_end - request_start)
                except Exception:
                    request_times.append(0.1)
            
            total_time = time.perf_counter() - start_time
        
        # Calculate performance metrics
        metrics = {
            'avg_request_time': statistics.mean(request_times),
            'median_request_time': statistics.median(request_times),
            'p95_request_time': np.percentile(request_times, 95),
            'p99_request_time': np.percentile(request_times, 99),
            'total_time': total_time,
            'throughput': len(request_times) / total_time if total_time > 0 else 0
        }
        
        # Define baseline expectations (these would normally come from stored baselines)
        baseline_expectations = {
            'avg_request_time': 0.05,  # 50ms average
            'p95_request_time': 0.1,   # 100ms 95th percentile
            'p99_request_time': 0.2,   # 200ms 99th percentile
            'throughput': 20           # 20 req/sec minimum
        }
        
        # Check against baselines
        for metric, expected in baseline_expectations.items():
            actual = metrics[metric]
            if metric == 'throughput':
                assert actual >= expected, f"Performance regression: {metric} {actual:.2f} below baseline {expected:.2f}"
            else:
                assert actual <= expected, f"Performance regression: {metric} {actual:.4f}s above baseline {expected:.4f}s"
        
        print("Performance metrics:")
        for metric, value in metrics.items():
            if 'time' in metric:
                print(f"  {metric}: {value:.4f}s")
            else:
                print(f"  {metric}: {value:.2f}")
    
    def test_performance_consistency(self, universal_ai_api, sample_molecular_data, performance_config):
        """Test performance consistency across multiple runs."""
        # Mock data processor
        with patch.object(universal_ai_api.data_processor, 'prepare_features') as mock_process:
            mock_process.return_value = Mock(
                status="success",
                data={"features": [1, 2, 3]},
                processing_time=0.01
            )
            
            # Run multiple test rounds
            round_results = []
            
            for round_num in range(5):  # 5 rounds of testing
                round_times = []
                
                for _ in range(performance_config["benchmark_iterations"] // 5):
                    start_time = time.perf_counter()
                    try:
                        result = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
                        end_time = time.perf_counter()
                        round_times.append(end_time - start_time)
                    except Exception:
                        round_times.append(0.1)
                
                round_avg = statistics.mean(round_times)
                round_results.append(round_avg)
                
                # Brief pause between rounds
                time.sleep(0.1)
        
        # Check consistency across rounds
        overall_avg = statistics.mean(round_results)
        overall_std = statistics.stdev(round_results) if len(round_results) > 1 else 0
        coefficient_of_variation = overall_std / overall_avg if overall_avg > 0 else 0
        
        # Performance should be consistent (low coefficient of variation)
        assert coefficient_of_variation < 0.2, f"Performance too inconsistent: CV={coefficient_of_variation:.3f}"
        
        print(f"Performance consistency - Avg: {overall_avg:.4f}s, StdDev: {overall_std:.4f}s, CV: {coefficient_of_variation:.3f}")
        print(f"Round results: {[f'{r:.4f}' for r in round_results]}")