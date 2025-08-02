# Performance Tuning Guide

## Overview

This guide provides comprehensive performance optimization strategies for Universal AI Core, covering caching, parallel processing, memory management, and system tuning adapted from Saraphis performance patterns.

## Performance Monitoring

### 1. Built-in Metrics Collection

```python
from universal_ai_core import create_api, APIConfig
import time

# Enable comprehensive monitoring
config = APIConfig(
    enable_monitoring=True,
    max_workers=8,
    enable_caching=True,
    cache_size=10000
)

api = create_api(api_config=config)

# Collect baseline metrics
baseline_metrics = api.get_metrics()
print("Baseline Metrics:")
print(f"  Memory usage: {baseline_metrics.get('system.memory_percent', 0):.1f}%")
print(f"  Active workers: {baseline_metrics.get('active_workers', 0)}")
print(f"  Cache hit rate: {baseline_metrics.get('cache_hit_rate', 0):.2f}")

# Perform operations and measure
start_time = time.time()
test_data = {"molecules": [{"smiles": f"C{i}"} for i in range(100)]}
result = api.process_data(test_data, ["molecular_descriptors"])
processing_time = time.time() - start_time

# Get post-processing metrics
final_metrics = api.get_metrics()
print(f"\nProcessing Results:")
print(f"  Processing time: {processing_time:.2f}s")
print(f"  Throughput: {len(test_data['molecules'])/processing_time:.1f} molecules/sec")
print(f"  Memory increase: {final_metrics.get('system.memory_percent', 0) - baseline_metrics.get('system.memory_percent', 0):.1f}%")

api.shutdown()
```

### 2. Custom Performance Monitoring

```python
import psutil
import time
import threading
from collections import deque

class PerformanceMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.metrics_history = {
            'cpu_percent': deque(maxlen=100),
            'memory_percent': deque(maxlen=100),
            'disk_io': deque(maxlen=100),
            'network_io': deque(maxlen=100)
        }
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.metrics_history['cpu_percent'].append(cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics_history['memory_percent'].append(memory.percent)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.metrics_history['disk_io'].append({
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes
                })
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                self.metrics_history['network_io'].append({
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv
                })
            
            time.sleep(self.interval)
    
    def get_stats(self):
        """Get current statistics."""
        stats = {}
        
        for metric, history in self.metrics_history.items():
            if history:
                if metric in ['cpu_percent', 'memory_percent']:
                    stats[metric] = {
                        'current': history[-1],
                        'average': sum(history) / len(history),
                        'max': max(history),
                        'min': min(history)
                    }
                else:
                    # For I/O metrics, calculate rates
                    if len(history) >= 2:
                        current = history[-1]
                        previous = history[-2]
                        stats[metric] = current
        
        return stats

# Usage example
monitor = PerformanceMonitor(interval=0.5)
monitor.start_monitoring()

# Run your workload
api = create_api()
# ... perform operations ...

# Get performance statistics
stats = monitor.get_stats()
print("Performance Statistics:")
for metric, data in stats.items():
    if isinstance(data, dict) and 'current' in data:
        print(f"  {metric}: {data['current']:.1f} (avg: {data['average']:.1f})")

monitor.stop_monitoring()
api.shutdown()
```

## Caching Optimization

### 1. Intelligent Caching Configuration

```python
from universal_ai_core import create_api, APIConfig

# Optimize caching for different workloads
def create_cache_optimized_api(workload_type="balanced"):
    """Create API optimized for specific workload types."""
    
    cache_configs = {
        "memory_constrained": APIConfig(
            enable_caching=True,
            cache_size=1000,  # Smaller cache
            cache_ttl_hours=6,  # Shorter TTL
            max_workers=2
        ),
        "high_throughput": APIConfig(
            enable_caching=True,
            cache_size=50000,  # Large cache
            cache_ttl_hours=48,  # Longer TTL
            max_workers=16
        ),
        "balanced": APIConfig(
            enable_caching=True,
            cache_size=10000,
            cache_ttl_hours=24,
            max_workers=8
        ),
        "no_cache": APIConfig(
            enable_caching=False,
            max_workers=12  # More workers to compensate
        )
    }
    
    config = cache_configs.get(workload_type, cache_configs["balanced"])
    return create_api(api_config=config)

# Test different cache configurations
def benchmark_cache_performance():
    """Benchmark different cache configurations."""
    
    test_data = {"molecules": [{"smiles": "CCO"}, {"smiles": "CCN"}, {"smiles": "CCC"}]}
    
    for workload_type in ["no_cache", "memory_constrained", "balanced", "high_throughput"]:
        print(f"\nTesting {workload_type} configuration:")
        
        api = create_cache_optimized_api(workload_type)
        
        # First run (cold cache)
        start_time = time.time()
        result1 = api.process_data(test_data, ["molecular_descriptors"], use_cache=True)
        cold_time = time.time() - start_time
        
        # Second run (warm cache)
        start_time = time.time()
        result2 = api.process_data(test_data, ["molecular_descriptors"], use_cache=True)
        warm_time = time.time() - start_time
        
        if api.cache:
            cache_stats = api.cache.get_stats()
            hit_rate = cache_stats.get('hit_rate', 0)
            cache_size = cache_stats.get('size', 0)
        else:
            hit_rate = 0
            cache_size = 0
        
        print(f"  Cold run: {cold_time:.3f}s")
        print(f"  Warm run: {warm_time:.3f}s")
        print(f"  Speedup: {cold_time/warm_time:.1f}x" if warm_time > 0 else "  Speedup: N/A")
        print(f"  Cache hit rate: {hit_rate:.2f}")
        print(f"  Cache size: {cache_size}")
        
        api.shutdown()

benchmark_cache_performance()
```

### 2. Cache Warmup Strategies

```python
def warm_cache_for_domain(api, domain_type="molecular"):
    """Pre-warm cache with common operations."""
    
    warmup_data = {
        "molecular": [
            {"molecules": [{"smiles": "CCO"}]},  # Ethanol
            {"molecules": [{"smiles": "CCN"}]},  # Ethylamine
            {"molecules": [{"smiles": "CCC"}]},  # Propane
            {"molecules": [{"smiles": "CCCC"}]}, # Butane
        ],
        "financial": [
            {"ohlcv": [{"open": 100, "high": 105, "low": 98, "close": 103}]},
            {"returns": [0.01, 0.02, -0.01, 0.03]},
        ],
        "cybersecurity": [
            {"network_traffic": [{"src_ip": "192.168.1.1", "dst_ip": "10.0.0.1"}]},
            {"logs": [{"level": "INFO", "message": "User login successful"}]},
        ]
    }
    
    extractors = {
        "molecular": ["molecular_descriptors", "fingerprints"],
        "financial": ["technical_indicators", "risk_metrics"],
        "cybersecurity": ["network_features", "behavioral_patterns"]
    }
    
    data_samples = warmup_data.get(domain_type, [])
    extractor_list = extractors.get(domain_type, [])
    
    print(f"Warming cache for {domain_type} domain...")
    
    for i, data in enumerate(data_samples):
        for extractor in extractor_list:
            try:
                result = api.process_data(data, [extractor], use_cache=True)
                print(f"  Cached {domain_type} sample {i+1} with {extractor}")
            except Exception as e:
                print(f"  Failed to cache {domain_type} sample {i+1}: {e}")
    
    if api.cache:
        stats = api.cache.get_stats()
        print(f"Cache warmup complete. Cache size: {stats.get('size', 0)}")

# Usage
api = create_api(enable_caching=True, cache_size=10000)

# Warm cache for all domains
for domain in ["molecular", "financial", "cybersecurity"]:
    warm_cache_for_domain(api, domain)

api.shutdown()
```

## Parallel Processing Optimization

### 1. Worker Pool Configuration

```python
import concurrent.futures
import time
from universal_ai_core import create_api

def optimize_worker_count():
    """Find optimal worker count for your system."""
    
    test_data = {"molecules": [{"smiles": f"C{i}"} for i in range(50)]}
    worker_counts = [1, 2, 4, 8, 16, 32]
    
    results = {}
    
    for worker_count in worker_counts:
        print(f"Testing with {worker_count} workers...")
        
        api = create_api(max_workers=worker_count, enable_caching=False)
        
        start_time = time.time()
        try:
            result = api.process_data(test_data, ["molecular_descriptors"])
            processing_time = time.time() - start_time
            
            if result.status == "success":
                throughput = len(test_data["molecules"]) / processing_time
                results[worker_count] = {
                    'time': processing_time,
                    'throughput': throughput,
                    'status': 'success'
                }
            else:
                results[worker_count] = {'status': 'failed'}
                
        except Exception as e:
            results[worker_count] = {'status': 'error', 'error': str(e)}
        
        api.shutdown()
        time.sleep(1)  # Brief pause between tests
    
    # Find optimal worker count
    successful_results = {k: v for k, v in results.items() if v.get('status') == 'success'}
    
    if successful_results:
        optimal_workers = max(successful_results.keys(), 
                            key=lambda k: successful_results[k]['throughput'])
        
        print("\nWorker Count Optimization Results:")
        for workers, result in results.items():
            if result.get('status') == 'success':
                print(f"  {workers:2d} workers: {result['time']:.2f}s ({result['throughput']:.1f} mol/s)")
            else:
                print(f"  {workers:2d} workers: {result['status']}")
        
        print(f"\nOptimal worker count: {optimal_workers}")
        return optimal_workers
    else:
        print("No successful results found")
        return 4  # Default fallback

optimal_workers = optimize_worker_count()
```

### 2. Batch Processing Optimization

```python
def optimize_batch_size(api, total_items=1000):
    """Find optimal batch size for processing."""
    
    batch_sizes = [10, 25, 50, 100, 200, 500]
    results = {}
    
    base_data = [{"smiles": f"C{i}"} for i in range(total_items)]
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        start_time = time.time()
        total_processed = 0
        
        try:
            # Process in batches
            for i in range(0, len(base_data), batch_size):
                batch = base_data[i:i+batch_size]
                batch_data = {"molecules": batch}
                
                result = api.process_data(batch_data, ["molecular_descriptors"])
                
                if result.status == "success":
                    total_processed += len(batch)
                else:
                    break
            
            processing_time = time.time() - start_time
            
            if total_processed > 0:
                throughput = total_processed / processing_time
                results[batch_size] = {
                    'time': processing_time,
                    'throughput': throughput,
                    'processed': total_processed,
                    'status': 'success'
                }
            else:
                results[batch_size] = {'status': 'failed'}
                
        except Exception as e:
            results[batch_size] = {'status': 'error', 'error': str(e)}
    
    # Find optimal batch size
    successful_results = {k: v for k, v in results.items() if v.get('status') == 'success'}
    
    if successful_results:
        optimal_batch = max(successful_results.keys(), 
                           key=lambda k: successful_results[k]['throughput'])
        
        print("\nBatch Size Optimization Results:")
        for batch_size, result in results.items():
            if result.get('status') == 'success':
                print(f"  Batch {batch_size:3d}: {result['time']:.2f}s ({result['throughput']:.1f} mol/s)")
            else:
                print(f"  Batch {batch_size:3d}: {result['status']}")
        
        print(f"\nOptimal batch size: {optimal_batch}")
        return optimal_batch
    else:
        print("No successful results found")
        return 100  # Default fallback

# Usage
api = create_api(max_workers=optimal_workers, enable_caching=True)
optimal_batch = optimize_batch_size(api)
api.shutdown()
```

### 3. Async Processing Patterns

```python
import asyncio
from universal_ai_core import create_api

async def process_large_dataset_async(api, dataset, batch_size=100):
    """Process large dataset using async patterns."""
    
    async def process_batch_async(batch_data, batch_id):
        """Process a single batch asynchronously."""
        try:
            # Submit async task
            task_id = await api.submit_async_task(
                "batch_processing",
                batch_data,
                priority=5
            )
            
            # Wait for result
            result = api.get_task_result(task_id)
            while result is None or not result.is_completed:
                await asyncio.sleep(0.1)
                result = api.get_task_result(task_id)
            
            return {
                'batch_id': batch_id,
                'status': 'success' if result.is_successful else 'failed',
                'result': result.result if result.is_successful else None,
                'error': result.error_message if not result.is_successful else None
            }
            
        except Exception as e:
            return {
                'batch_id': batch_id,
                'status': 'error',
                'error': str(e)
            }
    
    # Split dataset into batches
    batches = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        batches.append({"molecules": batch, "batch_id": i // batch_size})
    
    print(f"Processing {len(dataset)} items in {len(batches)} batches...")
    
    # Process all batches concurrently
    start_time = time.time()
    
    tasks = [
        process_batch_async(batch, batch['batch_id']) 
        for batch in batches
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processing_time = time.time() - start_time
    
    # Analyze results
    successful = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'success')
    failed = len(results) - successful
    
    print(f"Async processing completed in {processing_time:.2f}s")
    print(f"Successful batches: {successful}/{len(batches)}")
    print(f"Failed batches: {failed}")
    print(f"Throughput: {len(dataset)/processing_time:.1f} items/sec")
    
    return results

# Usage example
async def main():
    api = create_api(max_workers=8, enable_caching=True)
    
    # Large dataset
    large_dataset = [{"smiles": f"C{i}"} for i in range(500)]
    
    # Process asynchronously
    results = await process_large_dataset_async(api, large_dataset, batch_size=50)
    
    api.shutdown()

# Run async processing
# asyncio.run(main())
```

## Memory Optimization

### 1. Memory Usage Monitoring

```python
import psutil
import gc
from universal_ai_core import create_api

class MemoryOptimizer:
    def __init__(self, api):
        self.api = api
        self.process = psutil.Process()
        self.baseline_memory = self.get_memory_usage()
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def monitor_operation(self, operation_name, operation_func):
        """Monitor memory usage during an operation."""
        print(f"\nMonitoring memory for: {operation_name}")
        
        # Before operation
        before_memory = self.get_memory_usage()
        before_gc = len(gc.get_objects())
        
        # Run operation
        start_time = time.time()
        result = operation_func()
        operation_time = time.time() - start_time
        
        # After operation (before GC)
        after_memory = self.get_memory_usage()
        after_gc = len(gc.get_objects())
        
        # Force garbage collection
        gc.collect()
        final_memory = self.get_memory_usage()
        
        # Report results
        memory_increase = after_memory - before_memory
        memory_recovered = after_memory - final_memory
        object_increase = after_gc - before_gc
        
        print(f"  Operation time: {operation_time:.2f}s")
        print(f"  Memory before: {before_memory:.1f} MB")
        print(f"  Memory after: {after_memory:.1f} MB (+{memory_increase:.1f} MB)")
        print(f"  Memory final: {final_memory:.1f} MB (recovered {memory_recovered:.1f} MB)")
        print(f"  Object count increase: {object_increase}")
        print(f"  Memory efficiency: {memory_recovered/memory_increase*100:.1f}% recovered" if memory_increase > 0 else "  No memory increase")
        
        return {
            'operation_time': operation_time,
            'memory_increase': memory_increase,
            'memory_recovered': memory_recovered,
            'object_increase': object_increase,
            'result': result
        }

# Usage
api = create_api(enable_caching=True)
optimizer = MemoryOptimizer(api)

# Test different operations
def test_small_dataset():
    data = {"molecules": [{"smiles": "CCO"}]}
    return api.process_data(data, ["molecular_descriptors"])

def test_medium_dataset():
    data = {"molecules": [{"smiles": f"C{i}"} for i in range(100)]}
    return api.process_data(data, ["molecular_descriptors"])

def test_large_dataset():
    data = {"molecules": [{"smiles": f"C{i}"} for i in range(1000)]}
    return api.process_data(data, ["molecular_descriptors"])

# Monitor operations
optimizer.monitor_operation("Small Dataset (1 molecule)", test_small_dataset)
optimizer.monitor_operation("Medium Dataset (100 molecules)", test_medium_dataset)
optimizer.monitor_operation("Large Dataset (1000 molecules)", test_large_dataset)

api.shutdown()
```

### 2. Memory-Efficient Processing

```python
def process_with_memory_constraints(api, dataset, max_memory_mb=500):
    """Process dataset with memory constraints."""
    
    def get_current_memory():
        return psutil.Process().memory_info().rss / 1024 / 1024
    
    initial_memory = get_current_memory()
    batch_size = 100  # Start with reasonable batch size
    min_batch_size = 10
    results = []
    
    i = 0
    while i < len(dataset):
        current_memory = get_current_memory()
        memory_usage = current_memory - initial_memory
        
        # Adjust batch size based on memory usage
        if memory_usage > max_memory_mb * 0.8:  # 80% of limit
            batch_size = max(min_batch_size, batch_size // 2)
            print(f"Reducing batch size to {batch_size} (memory: {memory_usage:.1f} MB)")
            
            # Force garbage collection
            gc.collect()
            
        elif memory_usage < max_memory_mb * 0.4:  # 40% of limit
            batch_size = min(200, batch_size * 2)
            print(f"Increasing batch size to {batch_size} (memory: {memory_usage:.1f} MB)")
        
        # Process batch
        batch = dataset[i:i+batch_size]
        batch_data = {"molecules": batch}
        
        try:
            result = api.process_data(batch_data, ["molecular_descriptors"])
            if result.status == "success":
                results.extend(result.data.get("features", []))
                print(f"Processed batch {i//batch_size + 1}: {len(batch)} items")
            else:
                print(f"Batch processing failed: {result.error_message}")
        
        except MemoryError:
            print("Memory error encountered, reducing batch size")
            batch_size = max(min_batch_size, batch_size // 4)
            continue
        
        i += batch_size
        
        # Periodic cleanup
        if i % (batch_size * 5) == 0:
            gc.collect()
    
    final_memory = get_current_memory()
    print(f"Processing complete. Memory usage: {final_memory - initial_memory:.1f} MB")
    
    return results

# Usage
api = create_api(max_workers=4, enable_caching=False)  # Disable cache to reduce memory

large_dataset = [{"smiles": f"C{i}"} for i in range(2000)]
results = process_with_memory_constraints(api, large_dataset, max_memory_mb=300)

api.shutdown()
```

## System-Level Optimization

### 1. CPU Optimization

```python
import os
import threading
from concurrent.futures import ThreadPoolExecutor

def optimize_cpu_usage():
    """Optimize CPU usage based on system capabilities."""
    
    # Get system information
    cpu_count = os.cpu_count()
    print(f"System CPU count: {cpu_count}")
    
    # Test different thread configurations
    thread_configs = [
        cpu_count // 2,      # Conservative
        cpu_count,           # Equal to CPU count
        cpu_count * 2,       # Oversubscription
        min(16, cpu_count * 3)  # High oversubscription
    ]
    
    test_data = {"molecules": [{"smiles": f"C{i}"} for i in range(200)]}
    
    results = {}
    
    for thread_count in thread_configs:
        print(f"Testing with {thread_count} threads...")
        
        api = create_api(max_workers=thread_count, enable_caching=False)
        
        start_time = time.time()
        
        # Monitor CPU usage during processing
        cpu_percentages = []
        
        def monitor_cpu():
            for _ in range(10):  # Monitor for 10 seconds max
                cpu_percentages.append(psutil.cpu_percent(interval=1))
        
        monitor_thread = threading.Thread(target=monitor_cpu, daemon=True)
        monitor_thread.start()
        
        try:
            result = api.process_data(test_data, ["molecular_descriptors"])
            processing_time = time.time() - start_time
            
            monitor_thread.join(timeout=1)
            
            avg_cpu = sum(cpu_percentages) / len(cpu_percentages) if cpu_percentages else 0
            
            results[thread_count] = {
                'time': processing_time,
                'avg_cpu': avg_cpu,
                'max_cpu': max(cpu_percentages) if cpu_percentages else 0,
                'status': result.status
            }
            
        except Exception as e:
            results[thread_count] = {'status': 'error', 'error': str(e)}
        
        api.shutdown()
        time.sleep(2)  # Cool down between tests
    
    # Find optimal configuration
    successful_results = {k: v for k, v in results.items() if v.get('status') == 'success'}
    
    if successful_results:
        # Optimize for throughput with reasonable CPU usage
        optimal_threads = min(successful_results.keys(), 
                             key=lambda k: successful_results[k]['time'])
        
        print("\nCPU Optimization Results:")
        for threads, result in results.items():
            if result.get('status') == 'success':
                print(f"  {threads:2d} threads: {result['time']:.2f}s (CPU: {result['avg_cpu']:.1f}%)")
            else:
                print(f"  {threads:2d} threads: {result['status']}")
        
        print(f"\nOptimal thread count: {optimal_threads}")
        return optimal_threads
    else:
        return cpu_count

optimal_threads = optimize_cpu_usage()
```

### 2. I/O Optimization

```python
import aiofiles
import asyncio
from pathlib import Path

async def optimize_file_io(api):
    """Optimize file I/O operations."""
    
    # Create test files
    test_dir = Path("test_io")
    test_dir.mkdir(exist_ok=True)
    
    # Generate test data files
    file_count = 100
    files_per_batch = 10
    
    print("Creating test files...")
    for i in range(file_count):
        file_path = test_dir / f"molecules_{i}.txt"
        with open(file_path, 'w') as f:
            f.write(f"CCO\nCCN\nCCC\n")  # Simple test molecules
    
    # Test synchronous file reading
    def sync_read_files():
        start_time = time.time()
        all_molecules = []
        
        for i in range(file_count):
            file_path = test_dir / f"molecules_{i}.txt"
            with open(file_path, 'r') as f:
                molecules = [line.strip() for line in f if line.strip()]
                all_molecules.extend(molecules)
        
        return time.time() - start_time, len(all_molecules)
    
    # Test asynchronous file reading
    async def async_read_files():
        start_time = time.time()
        all_molecules = []
        
        async def read_file(file_path):
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                return [line.strip() for line in content.split('\n') if line.strip()]
        
        # Process files in batches
        for i in range(0, file_count, files_per_batch):
            batch_files = [
                test_dir / f"molecules_{j}.txt" 
                for j in range(i, min(i + files_per_batch, file_count))
            ]
            
            batch_results = await asyncio.gather(*[read_file(f) for f in batch_files])
            
            for molecules in batch_results:
                all_molecules.extend(molecules)
        
        return time.time() - start_time, len(all_molecules)
    
    # Run benchmarks
    print("Testing synchronous file I/O...")
    sync_time, sync_count = sync_read_files()
    
    print("Testing asynchronous file I/O...")
    async_time, async_count = await async_read_files()
    
    print(f"\nFile I/O Optimization Results:")
    print(f"  Synchronous:  {sync_time:.2f}s ({sync_count} molecules)")
    print(f"  Asynchronous: {async_time:.2f}s ({async_count} molecules)")
    print(f"  Speedup: {sync_time/async_time:.1f}x")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    
    return "async" if async_time < sync_time else "sync"

# Run I/O optimization
api = create_api()
# optimal_io = asyncio.run(optimize_file_io(api))
api.shutdown()
```

## Configuration Tuning

### 1. Production Configuration Template

```yaml
# config/production_optimized.yaml
core:
  max_workers: 16  # Adjust based on CPU cores
  enable_monitoring: true
  debug_mode: false
  log_level: "INFO"

api:
  enable_caching: true
  cache_size: 50000  # Large cache for production
  cache_ttl_hours: 48
  enable_safety_checks: true
  rate_limit_requests_per_minute: 2000
  request_timeout: 300.0
  batch_size: 200  # Optimized batch size
  max_queue_size: 2000

plugins:
  feature_extractors:
    molecular:
      enabled: true
      rdkit_enabled: true
      batch_processing: true
      cache_descriptors: true
    cybersecurity:
      enabled: true
      parallel_processing: true
      threat_detection_level: "high"
    financial:
      enabled: true
      technical_indicators_cache: true
      risk_calculation_precision: "high"

performance:
  memory_optimization: true
  gc_threshold: 0.8  # Trigger GC at 80% memory usage
  batch_size_auto_adjust: true
  worker_scaling: "auto"  # Auto-scale workers based on load

monitoring:
  enable_metrics_collection: true
  metrics_retention_hours: 168  # 1 week
  performance_logging: true
  alert_thresholds:
    memory_percent: 85
    response_time_ms: 5000
    error_rate_percent: 5
```

### 2. Environment-Specific Optimization

```python
def create_optimized_api_for_environment(environment="production"):
    """Create API optimized for specific environments."""
    
    optimizations = {
        "development": {
            "max_workers": 2,
            "enable_caching": True,
            "cache_size": 1000,
            "debug_mode": True,
            "enable_monitoring": True,
            "batch_size": 50
        },
        "testing": {
            "max_workers": 1,
            "enable_caching": False,
            "debug_mode": True,
            "enable_monitoring": False,
            "batch_size": 10
        },
        "staging": {
            "max_workers": 4,
            "enable_caching": True,
            "cache_size": 5000,
            "debug_mode": False,
            "enable_monitoring": True,
            "batch_size": 100
        },
        "production": {
            "max_workers": optimal_threads,
            "enable_caching": True,
            "cache_size": 50000,
            "debug_mode": False,
            "enable_monitoring": True,
            "batch_size": optimal_batch
        }
    }
    
    config_params = optimizations.get(environment, optimizations["production"])
    
    config = APIConfig(**config_params)
    return create_api(api_config=config)

# Create environment-specific APIs
dev_api = create_optimized_api_for_environment("development")
prod_api = create_optimized_api_for_environment("production")
```

## Performance Testing and Benchmarking

### 1. Comprehensive Performance Test Suite

```python
class PerformanceBenchmark:
    def __init__(self, api):
        self.api = api
        self.results = {}
    
    def benchmark_throughput(self, dataset_sizes=[10, 50, 100, 500]):
        """Benchmark throughput with different dataset sizes."""
        
        print("Benchmarking throughput...")
        throughput_results = {}
        
        for size in dataset_sizes:
            test_data = {"molecules": [{"smiles": f"C{i}"} for i in range(size)]}
            
            start_time = time.time()
            result = self.api.process_data(test_data, ["molecular_descriptors"])
            processing_time = time.time() - start_time
            
            if result.status == "success":
                throughput = size / processing_time
                throughput_results[size] = {
                    'time': processing_time,
                    'throughput': throughput,
                    'molecules_per_second': throughput
                }
                print(f"  {size:3d} molecules: {processing_time:.2f}s ({throughput:.1f} mol/s)")
            else:
                throughput_results[size] = {'status': 'failed'}
        
        self.results['throughput'] = throughput_results
        return throughput_results
    
    def benchmark_latency(self, iterations=100):
        """Benchmark latency for single requests."""
        
        print("Benchmarking latency...")
        test_data = {"molecules": [{"smiles": "CCO"}]}
        
        latencies = []
        
        for i in range(iterations):
            start_time = time.time()
            result = self.api.process_data(test_data, ["molecular_descriptors"])
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            if result.status == "success":
                latencies.append(latency)
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
            
            latency_results = {
                'average_ms': avg_latency,
                'min_ms': min_latency,
                'max_ms': max_latency,
                'p95_ms': p95_latency,
                'iterations': len(latencies)
            }
            
            print(f"  Average latency: {avg_latency:.1f}ms")
            print(f"  Min latency: {min_latency:.1f}ms")
            print(f"  Max latency: {max_latency:.1f}ms")
            print(f"  P95 latency: {p95_latency:.1f}ms")
            
            self.results['latency'] = latency_results
            return latency_results
    
    def benchmark_memory_scaling(self, dataset_sizes=[100, 500, 1000, 2000]):
        """Benchmark memory usage scaling."""
        
        print("Benchmarking memory scaling...")
        memory_results = {}
        
        for size in dataset_sizes:
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            test_data = {"molecules": [{"smiles": f"C{i}"} for i in range(size)]}
            result = self.api.process_data(test_data, ["molecular_descriptors"])
            
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = peak_memory - initial_memory
            
            # Force garbage collection
            gc.collect()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_recovered = peak_memory - final_memory
            
            memory_results[size] = {
                'memory_increase_mb': memory_increase,
                'memory_recovered_mb': memory_recovered,
                'memory_per_molecule_kb': (memory_increase * 1024) / size if size > 0 else 0,
                'status': result.status
            }
            
            print(f"  {size:4d} molecules: +{memory_increase:.1f}MB (recovered {memory_recovered:.1f}MB)")
        
        self.results['memory_scaling'] = memory_results
        return memory_results
    
    def run_full_benchmark(self):
        """Run complete performance benchmark."""
        
        print("Starting comprehensive performance benchmark...\n")
        
        # Run all benchmarks
        self.benchmark_throughput()
        print()
        self.benchmark_latency()
        print()
        self.benchmark_memory_scaling()
        
        print("\n=== Performance Benchmark Summary ===")
        
        # Summarize results
        if 'throughput' in self.results:
            max_throughput = max(
                [r['throughput'] for r in self.results['throughput'].values() 
                 if isinstance(r, dict) and 'throughput' in r]
            )
            print(f"Maximum throughput: {max_throughput:.1f} molecules/second")
        
        if 'latency' in self.results:
            avg_latency = self.results['latency']['average_ms']
            print(f"Average latency: {avg_latency:.1f}ms")
        
        if 'memory_scaling' in self.results:
            memory_per_mol = [
                r['memory_per_molecule_kb'] for r in self.results['memory_scaling'].values()
                if isinstance(r, dict) and 'memory_per_molecule_kb' in r
            ]
            if memory_per_mol:
                avg_memory_per_mol = sum(memory_per_mol) / len(memory_per_mol)
                print(f"Average memory per molecule: {avg_memory_per_mol:.1f}KB")
        
        return self.results

# Run comprehensive benchmark
api = create_api(max_workers=optimal_threads, enable_caching=True, cache_size=10000)
benchmark = PerformanceBenchmark(api)
results = benchmark.run_full_benchmark()
api.shutdown()
```

## Conclusion

This performance tuning guide provides comprehensive strategies for optimizing Universal AI Core across different dimensions:

1. **Monitoring**: Continuous performance monitoring and metrics collection
2. **Caching**: Intelligent caching strategies for different workloads
3. **Parallelization**: Worker pool and batch processing optimization
4. **Memory**: Memory usage monitoring and optimization techniques
5. **System**: CPU and I/O optimization strategies
6. **Configuration**: Environment-specific tuning parameters
7. **Benchmarking**: Comprehensive performance testing methodologies

Implement these optimizations incrementally, measuring performance impact at each step to achieve optimal system performance for your specific use case and environment.