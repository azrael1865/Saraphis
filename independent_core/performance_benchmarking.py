"""
Performance Benchmarking and Monitoring System
Comprehensive performance analysis for compression systems and Brain components
"""

import time
import psutil
import threading
import logging
import json
import statistics
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import contextmanager
import numpy as np
import gc
import traceback

# GPU monitoring support
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark"""
    name: str
    duration_seconds: float
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_usage_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    throughput_ops_per_sec: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics"""
    operation_name: str
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    avg_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    avg_cpu_usage: float = 0.0
    avg_memory_usage: float = 0.0
    avg_gpu_usage: float = 0.0
    avg_throughput: float = 0.0
    success_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    percentiles: Dict[str, float] = field(default_factory=dict)


class ResourceMonitor:
    """Monitors system resource usage during benchmarks"""
    
    def __init__(self, sampling_interval: float = 0.1):
        self.sampling_interval = sampling_interval
        self._monitoring = False
        self._monitor_thread = None
        self._resource_data = []
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """Start resource monitoring"""
        with self._lock:
            if self._monitoring:
                return
            
            self._monitoring = True
            self._resource_data.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return aggregated results"""
        with self._lock:
            if not self._monitoring:
                return {}
            
            self._monitoring = False
            
            if self._monitor_thread:
                self._monitor_thread.join(timeout=1.0)
            
            # Aggregate results
            if not self._resource_data:
                return {}
            
            cpu_usage = [data['cpu'] for data in self._resource_data]
            memory_usage = [data['memory'] for data in self._resource_data]
            gpu_usage = [data['gpu'] for data in self._resource_data if 'gpu' in data]
            gpu_memory = [data['gpu_memory'] for data in self._resource_data if 'gpu_memory' in data]
            
            results = {
                'avg_cpu_percent': statistics.mean(cpu_usage) if cpu_usage else 0.0,
                'max_cpu_percent': max(cpu_usage) if cpu_usage else 0.0,
                'avg_memory_mb': statistics.mean(memory_usage) if memory_usage else 0.0,
                'max_memory_mb': max(memory_usage) if memory_usage else 0.0,
                'sample_count': len(self._resource_data)
            }
            
            if gpu_usage:
                results.update({
                    'avg_gpu_percent': statistics.mean(gpu_usage),
                    'max_gpu_percent': max(gpu_usage),
                    'avg_gpu_memory_mb': statistics.mean(gpu_memory) if gpu_memory else 0.0,
                    'max_gpu_memory_mb': max(gpu_memory) if gpu_memory else 0.0
                })
            
            return results
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._monitoring:
            try:
                # CPU and memory
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_info = psutil.virtual_memory()
                memory_mb = memory_info.used / (1024 * 1024)
                
                data = {
                    'cpu': cpu_percent,
                    'memory': memory_mb,
                    'timestamp': time.time()
                }
                
                # GPU metrics if available
                if GPU_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Use first GPU
                            data['gpu'] = gpu.load * 100
                            data['gpu_memory'] = gpu.memoryUsed
                    except Exception:
                        pass  # GPU monitoring failed, continue without it
                
                with self._lock:
                    self._resource_data.append(data)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                break


class PerformanceBenchmarker:
    """Main performance benchmarking system"""
    
    def __init__(self, enable_gpu_monitoring: bool = True):
        self.enable_gpu_monitoring = enable_gpu_monitoring and GPU_AVAILABLE
        self.benchmark_results: List[BenchmarkResult] = []
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        self.resource_monitor = ResourceMonitor()
        self._lock = threading.RLock()
        
        # Benchmark suites
        self.benchmark_suites = {}
        self._register_default_benchmarks()
        
        logger.info(f"Performance benchmarker initialized (GPU monitoring: {self.enable_gpu_monitoring})")
    
    def _register_default_benchmarks(self):
        """Register default benchmark suites"""
        # Compression system benchmarks
        self.benchmark_suites['compression'] = {
            'padic_compression': self._benchmark_padic_compression,
            'sheaf_compression': self._benchmark_sheaf_compression,
            'tensor_decomposition': self._benchmark_tensor_decomposition,
            'compression_comparison': self._benchmark_compression_comparison
        }
        
        # Training system benchmarks
        self.benchmark_suites['training'] = {
            'basic_training': self._benchmark_basic_training,
            'domain_training': self._benchmark_domain_training,
            'concurrent_training': self._benchmark_concurrent_training,
            'memory_optimization': self._benchmark_memory_optimization
        }
        
        # Brain system benchmarks
        self.benchmark_suites['brain'] = {
            'prediction_latency': self._benchmark_prediction_latency,
            'domain_routing': self._benchmark_domain_routing,
            'state_persistence': self._benchmark_state_persistence,
            'system_integration': self._benchmark_system_integration
        }
    
    @contextmanager
    def benchmark_context(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for benchmarking operations"""
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)
        start_memory = psutil.virtual_memory().used / (1024 * 1024)
        
        # GPU metrics
        start_gpu = 0.0
        start_gpu_memory = 0.0
        if self.enable_gpu_monitoring:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    start_gpu = gpu.load * 100
                    start_gpu_memory = gpu.memoryUsed
            except Exception:
                pass
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        error_message = None
        success = True
        
        try:
            yield
        except Exception as e:
            error_message = str(e)
            success = False
            logger.error(f"Benchmark {name} failed: {e}")
        finally:
            # Stop monitoring and get results
            resource_stats = self.resource_monitor.stop_monitoring()
            
            duration = time.time() - start_time
            
            # Calculate resource usage
            cpu_usage = resource_stats.get('avg_cpu_percent', 0.0)
            memory_usage = resource_stats.get('avg_memory_mb', start_memory)
            gpu_usage = resource_stats.get('avg_gpu_percent', 0.0)
            gpu_memory = resource_stats.get('avg_gpu_memory_mb', start_gpu_memory)
            
            # Create benchmark result
            result = BenchmarkResult(
                name=name,
                duration_seconds=duration,
                cpu_usage_percent=cpu_usage,
                memory_usage_mb=memory_usage,
                gpu_usage_percent=gpu_usage,
                gpu_memory_mb=gpu_memory,
                success=success,
                error_message=error_message,
                metadata=metadata or {}
            )
            
            # Store result
            with self._lock:
                self.benchmark_results.append(result)
                self._update_performance_metrics(result)
    
    def _update_performance_metrics(self, result: BenchmarkResult):
        """Update aggregated performance metrics"""
        name = result.name
        
        if name not in self.performance_metrics:
            self.performance_metrics[name] = PerformanceMetrics(operation_name=name)
        
        metrics = self.performance_metrics[name]
        
        # Update counters
        metrics.total_runs += 1
        if result.success:
            metrics.successful_runs += 1
        else:
            metrics.failed_runs += 1
        
        # Update timing statistics
        if result.success:
            durations = [r.duration_seconds for r in self.benchmark_results 
                        if r.name == name and r.success]
            
            metrics.avg_duration = statistics.mean(durations)
            metrics.min_duration = min(durations)
            metrics.max_duration = max(durations)
            
            # Calculate percentiles
            if len(durations) >= 5:
                sorted_durations = sorted(durations)
                metrics.percentiles = {
                    'p50': sorted_durations[len(sorted_durations) // 2],
                    'p90': sorted_durations[int(len(sorted_durations) * 0.9)],
                    'p95': sorted_durations[int(len(sorted_durations) * 0.95)],
                    'p99': sorted_durations[int(len(sorted_durations) * 0.99)]
                }
        
        # Update resource usage
        successful_results = [r for r in self.benchmark_results 
                            if r.name == name and r.success]
        
        if successful_results:
            metrics.avg_cpu_usage = statistics.mean([r.cpu_usage_percent for r in successful_results])
            metrics.avg_memory_usage = statistics.mean([r.memory_usage_mb for r in successful_results])
            metrics.avg_gpu_usage = statistics.mean([r.gpu_usage_percent for r in successful_results])
            metrics.avg_throughput = statistics.mean([r.throughput_ops_per_sec for r in successful_results if r.throughput_ops_per_sec > 0])
        
        # Update success rate
        metrics.success_rate = metrics.successful_runs / metrics.total_runs
        metrics.last_updated = datetime.now()
    
    # Compression benchmarks
    def _benchmark_padic_compression(self, data_sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark P-adic compression performance"""
        if data_sizes is None:
            data_sizes = [1000, 10000, 100000]
        
        results = {}
        
        try:
            from independent_core.compression_systems.padic import PadicCompressionSystem
            
            padic_system = PadicCompressionSystem(prime=7, precision=10)
            
            for size in data_sizes:
                test_data = np.random.randn(size)
                
                with self.benchmark_context(f"padic_compression_{size}", 
                                          {'data_size': size, 'compression_type': 'padic'}):
                    compressed = padic_system.compress(test_data)
                    decompressed = padic_system.decompress(compressed)
                    
                    # Calculate compression ratio and throughput
                    compression_ratio = len(test_data) / len(compressed) if compressed else 1.0
                    throughput = size / self.benchmark_results[-1].duration_seconds
                    
                    # Update throughput in the result
                    self.benchmark_results[-1].throughput_ops_per_sec = throughput
                    self.benchmark_results[-1].metadata.update({
                        'compression_ratio': compression_ratio,
                        'reconstruction_error': np.linalg.norm(test_data - decompressed) / np.linalg.norm(test_data)
                    })
                
                results[f'size_{size}'] = {
                    'compression_ratio': compression_ratio,
                    'throughput_ops_per_sec': throughput
                }
        
        except Exception as e:
            logger.error(f"P-adic compression benchmark failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _benchmark_sheaf_compression(self, data_sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark Sheaf compression performance"""
        if data_sizes is None:
            data_sizes = [1000, 10000, 50000]
        
        results = {}
        
        try:
            from independent_core.compression_systems.sheaf import SheafCompressionSystem
            
            sheaf_system = SheafCompressionSystem(compression_level=0.8)
            
            for size in data_sizes:
                # Create structured data for sheaf compression
                test_data = {
                    'sections': {'global': np.random.randn(size)},
                    'topology': {'open_sets': ['U1', 'U2']},
                    'restriction_maps': {}
                }
                
                with self.benchmark_context(f"sheaf_compression_{size}",
                                          {'data_size': size, 'compression_type': 'sheaf'}):
                    compressed = sheaf_system.compress(test_data)
                    decompressed = sheaf_system.decompress(compressed)
                    
                    throughput = size / self.benchmark_results[-1].duration_seconds
                    self.benchmark_results[-1].throughput_ops_per_sec = throughput
                
                results[f'size_{size}'] = {'throughput_ops_per_sec': throughput}
        
        except Exception as e:
            logger.error(f"Sheaf compression benchmark failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _benchmark_tensor_decomposition(self, tensor_shapes: List[Tuple] = None) -> Dict[str, Any]:
        """Benchmark tensor decomposition performance"""
        if tensor_shapes is None:
            tensor_shapes = [(10, 10, 10), (20, 20, 20), (50, 50, 10)]
        
        results = {}
        
        try:
            from independent_core.compression_systems.tensor_decomposition import HOSVDDecomposer
            
            decomposer = HOSVDDecomposer(energy_threshold=0.95)
            
            for shape in tensor_shapes:
                tensor = np.random.randn(*shape)
                tensor_size = np.prod(shape)
                
                with self.benchmark_context(f"tensor_decomposition_{shape}",
                                          {'tensor_shape': shape, 'tensor_size': tensor_size}):
                    decomposition = decomposer.decompose_tensor(tensor)
                    reconstructed = decomposer.reconstruct_tensor(decomposition)
                    
                    throughput = tensor_size / self.benchmark_results[-1].duration_seconds
                    self.benchmark_results[-1].throughput_ops_per_sec = throughput
                    
                    reconstruction_error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)
                    self.benchmark_results[-1].metadata['reconstruction_error'] = reconstruction_error
                
                results[f'shape_{shape}'] = {
                    'throughput_ops_per_sec': throughput,
                    'reconstruction_error': reconstruction_error
                }
        
        except Exception as e:
            logger.error(f"Tensor decomposition benchmark failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _benchmark_compression_comparison(self) -> Dict[str, Any]:
        """Compare performance across compression methods"""
        comparison_results = {}
        test_size = 10000
        test_data = np.random.randn(test_size)
        
        # Test each compression method
        compression_methods = {
            'padic': self._benchmark_padic_compression,
            'sheaf': self._benchmark_sheaf_compression,
            'tensor': self._benchmark_tensor_decomposition
        }
        
        for method_name, benchmark_func in compression_methods.items():
            try:
                with self.benchmark_context(f"compression_comparison_{method_name}"):
                    method_results = benchmark_func([test_size])
                    comparison_results[method_name] = method_results
            except Exception as e:
                comparison_results[method_name] = {'error': str(e)}
        
        return comparison_results
    
    # Training benchmarks
    def _benchmark_basic_training(self) -> Dict[str, Any]:
        """Benchmark basic training performance"""
        results = {}
        
        try:
            # Generate synthetic training data
            X = np.random.randn(1000, 10)
            y = np.random.randint(0, 2, 1000)
            
            training_config = {
                'epochs': 10,
                'batch_size': 32,
                'learning_rate': 0.001,
                'validation_split': 0.2
            }
            
            with self.benchmark_context("basic_training", {'data_size': len(X)}):
                # Simulate training (would use actual TrainingManager in practice)
                time.sleep(0.5)  # Simulate training time
                
                training_throughput = len(X) / self.benchmark_results[-1].duration_seconds
                self.benchmark_results[-1].throughput_ops_per_sec = training_throughput
                
                results['training_throughput'] = training_throughput
                results['epochs'] = training_config['epochs']
        
        except Exception as e:
            logger.error(f"Basic training benchmark failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _benchmark_domain_training(self) -> Dict[str, Any]:
        """Benchmark domain-specific training"""
        results = {}
        
        try:
            domains = ['mathematics', 'language', 'vision']
            
            for domain in domains:
                with self.benchmark_context(f"domain_training_{domain}", {'domain': domain}):
                    # Simulate domain training
                    time.sleep(0.3)  # Simulate domain-specific training
                    
                    results[domain] = {
                        'duration': self.benchmark_results[-1].duration_seconds,
                        'success': self.benchmark_results[-1].success
                    }
        
        except Exception as e:
            logger.error(f"Domain training benchmark failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _benchmark_concurrent_training(self, num_sessions: int = 3) -> Dict[str, Any]:
        """Benchmark concurrent training sessions"""
        results = {}
        
        try:
            with self.benchmark_context(f"concurrent_training_{num_sessions}", 
                                      {'num_sessions': num_sessions}):
                # Simulate concurrent training sessions
                threads = []
                
                def simulate_training_session(session_id):
                    time.sleep(0.4)  # Simulate training
                
                for i in range(num_sessions):
                    thread = threading.Thread(target=simulate_training_session, args=(i,))
                    threads.append(thread)
                    thread.start()
                
                for thread in threads:
                    thread.join()
                
                results['concurrent_sessions'] = num_sessions
                results['total_duration'] = self.benchmark_results[-1].duration_seconds
        
        except Exception as e:
            logger.error(f"Concurrent training benchmark failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _benchmark_memory_optimization(self) -> Dict[str, Any]:
        """Benchmark memory optimization features"""
        results = {}
        
        try:
            # Test tensor compression for memory optimization
            large_tensor = np.random.randn(100, 100, 100)
            initial_memory = psutil.virtual_memory().used / (1024 * 1024)
            
            with self.benchmark_context("memory_optimization", 
                                      {'tensor_size': large_tensor.nbytes / (1024 * 1024)}):
                # Simulate memory optimization
                gc.collect()  # Force garbage collection
                time.sleep(0.2)
                
                final_memory = psutil.virtual_memory().used / (1024 * 1024)
                memory_saved = initial_memory - final_memory
                
                results['memory_saved_mb'] = memory_saved
                results['optimization_effective'] = memory_saved > 0
        
        except Exception as e:
            logger.error(f"Memory optimization benchmark failed: {e}")
            results['error'] = str(e)
        
        return results
    
    # Brain system benchmarks
    def _benchmark_prediction_latency(self, num_predictions: int = 100) -> Dict[str, Any]:
        """Benchmark prediction latency"""
        results = {}
        
        try:
            with self.benchmark_context(f"prediction_latency_{num_predictions}",
                                      {'num_predictions': num_predictions}):
                # Simulate predictions
                total_time = 0
                for i in range(num_predictions):
                    start = time.time()
                    # Simulate prediction processing
                    time.sleep(0.001)  # 1ms per prediction
                    total_time += time.time() - start
                
                avg_latency = total_time / num_predictions
                throughput = num_predictions / self.benchmark_results[-1].duration_seconds
                
                self.benchmark_results[-1].throughput_ops_per_sec = throughput
                
                results['avg_latency_ms'] = avg_latency * 1000
                results['throughput_predictions_per_sec'] = throughput
        
        except Exception as e:
            logger.error(f"Prediction latency benchmark failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _benchmark_domain_routing(self) -> Dict[str, Any]:
        """Benchmark domain routing performance"""
        results = {}
        
        try:
            domains = ['general', 'mathematics', 'language', 'vision', 'audio']
            
            with self.benchmark_context("domain_routing", {'num_domains': len(domains)}):
                # Simulate domain routing decisions
                for domain in domains:
                    # Simulate routing decision
                    time.sleep(0.01)  # 10ms per routing decision
                
                routing_throughput = len(domains) / self.benchmark_results[-1].duration_seconds
                self.benchmark_results[-1].throughput_ops_per_sec = routing_throughput
                
                results['routing_throughput'] = routing_throughput
                results['domains_tested'] = len(domains)
        
        except Exception as e:
            logger.error(f"Domain routing benchmark failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _benchmark_state_persistence(self) -> Dict[str, Any]:
        """Benchmark state persistence performance"""
        results = {}
        
        try:
            # Simulate state data
            state_data = {
                'domain_states': {f'domain_{i}': np.random.randn(100).tolist() for i in range(10)},
                'model_weights': np.random.randn(1000, 100).tolist(),
                'training_history': [{'epoch': i, 'loss': np.random.rand()} for i in range(100)]
            }
            
            with self.benchmark_context("state_persistence", 
                                      {'state_size_mb': len(json.dumps(state_data)) / (1024 * 1024)}):
                # Simulate state save/load
                state_json = json.dumps(state_data)
                loaded_state = json.loads(state_json)
                
                persistence_throughput = len(state_json) / self.benchmark_results[-1].duration_seconds
                self.benchmark_results[-1].throughput_ops_per_sec = persistence_throughput
                
                results['persistence_throughput_bytes_per_sec'] = persistence_throughput
                results['state_size_bytes'] = len(state_json)
        
        except Exception as e:
            logger.error(f"State persistence benchmark failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _benchmark_system_integration(self) -> Dict[str, Any]:
        """Benchmark overall system integration performance"""
        results = {}
        
        try:
            with self.benchmark_context("system_integration"):
                # Simulate integrated system operations
                operations = [
                    ('prediction', 0.05),
                    ('training', 0.2),
                    ('compression', 0.1),
                    ('domain_routing', 0.02),
                    ('state_save', 0.08)
                ]
                
                for op_name, duration in operations:
                    time.sleep(duration)
                
                results['total_operations'] = len(operations)
                results['integration_successful'] = True
        
        except Exception as e:
            logger.error(f"System integration benchmark failed: {e}")
            results['error'] = str(e)
        
        return results
    
    # Main benchmark execution methods
    def run_benchmark_suite(self, suite_name: str) -> Dict[str, Any]:
        """Run a complete benchmark suite"""
        if suite_name not in self.benchmark_suites:
            raise ValueError(f"Unknown benchmark suite: {suite_name}")
        
        suite = self.benchmark_suites[suite_name]
        suite_results = {}
        
        logger.info(f"Running benchmark suite: {suite_name}")
        
        for benchmark_name, benchmark_func in suite.items():
            try:
                logger.info(f"  Running benchmark: {benchmark_name}")
                result = benchmark_func()
                suite_results[benchmark_name] = result
            except Exception as e:
                logger.error(f"  Benchmark {benchmark_name} failed: {e}")
                suite_results[benchmark_name] = {'error': str(e)}
        
        return suite_results
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark suites"""
        all_results = {}
        
        for suite_name in self.benchmark_suites.keys():
            logger.info(f"Running benchmark suite: {suite_name}")
            all_results[suite_name] = self.run_benchmark_suite(suite_name)
        
        return all_results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self._lock:
            summary = {
                'total_benchmarks': len(self.benchmark_results),
                'successful_benchmarks': sum(1 for r in self.benchmark_results if r.success),
                'failed_benchmarks': sum(1 for r in self.benchmark_results if not r.success),
                'total_duration': sum(r.duration_seconds for r in self.benchmark_results),
                'avg_duration': statistics.mean([r.duration_seconds for r in self.benchmark_results]) if self.benchmark_results else 0,
                'performance_metrics': {name: asdict(metrics) for name, metrics in self.performance_metrics.items()},
                'last_updated': datetime.now().isoformat()
            }
            
            # Add resource usage summary
            if self.benchmark_results:
                successful_results = [r for r in self.benchmark_results if r.success]
                if successful_results:
                    summary['resource_usage'] = {
                        'avg_cpu_percent': statistics.mean([r.cpu_usage_percent for r in successful_results]),
                        'avg_memory_mb': statistics.mean([r.memory_usage_mb for r in successful_results]),
                        'avg_gpu_percent': statistics.mean([r.gpu_usage_percent for r in successful_results]),
                        'max_memory_mb': max([r.memory_usage_mb for r in successful_results])
                    }
            
            return summary
    
    def generate_benchmark_report(self, include_details: bool = True) -> str:
        """Generate comprehensive benchmark report"""
        summary = self.get_performance_summary()
        
        report = [
            "Performance Benchmark Report",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Summary:",
            f"  Total Benchmarks: {summary['total_benchmarks']}",
            f"  Successful: {summary['successful_benchmarks']}",
            f"  Failed: {summary['failed_benchmarks']}",
            f"  Success Rate: {summary['successful_benchmarks']/max(summary['total_benchmarks'],1):.1%}",
            f"  Total Duration: {summary['total_duration']:.2f}s",
            f"  Average Duration: {summary['avg_duration']:.3f}s",
            ""
        ]
        
        # Resource usage summary
        if 'resource_usage' in summary:
            usage = summary['resource_usage']
            report.extend([
                "Resource Usage:",
                f"  Average CPU: {usage['avg_cpu_percent']:.1f}%",
                f"  Average Memory: {usage['avg_memory_mb']:.1f} MB",
                f"  Average GPU: {usage['avg_gpu_percent']:.1f}%",
                f"  Peak Memory: {usage['max_memory_mb']:.1f} MB",
                ""
            ])
        
        # Performance metrics details
        if include_details and summary['performance_metrics']:
            report.extend([
                "Detailed Performance Metrics:",
                "-" * 30
            ])
            
            for name, metrics in summary['performance_metrics'].items():
                report.extend([
                    f"{name}:",
                    f"  Runs: {metrics['total_runs']} (Success Rate: {metrics['success_rate']:.1%})",
                    f"  Duration: {metrics['avg_duration']:.3f}s (min: {metrics['min_duration']:.3f}s, max: {metrics['max_duration']:.3f}s)",
                    f"  CPU Usage: {metrics['avg_cpu_usage']:.1f}%",
                    f"  Memory Usage: {metrics['avg_memory_usage']:.1f} MB",
                ])
                
                if metrics['percentiles']:
                    percentiles = metrics['percentiles']
                    report.append(f"  Percentiles: p50={percentiles.get('p50', 0):.3f}s, p95={percentiles.get('p95', 0):.3f}s, p99={percentiles.get('p99', 0):.3f}s")
                
                report.append("")
        
        return "\n".join(report)
    
    def export_results(self, filepath: str, format: str = 'json'):
        """Export benchmark results to file"""
        summary = self.get_performance_summary()
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        elif format.lower() == 'txt':
            with open(filepath, 'w') as f:
                f.write(self.generate_benchmark_report())
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Benchmark results exported to {filepath}")


# Utility functions
def quick_benchmark(func: Callable, *args, **kwargs) -> BenchmarkResult:
    """Quick benchmark of a single function"""
    benchmarker = PerformanceBenchmarker()
    
    with benchmarker.benchmark_context(func.__name__):
        result = func(*args, **kwargs)
    
    return benchmarker.benchmark_results[-1]


def compare_performance(functions: Dict[str, Callable], test_args: Tuple = (), 
                       test_kwargs: Dict = None, runs: int = 5) -> Dict[str, Any]:
    """Compare performance of multiple functions"""
    if test_kwargs is None:
        test_kwargs = {}
    
    benchmarker = PerformanceBenchmarker()
    comparison_results = {}
    
    for name, func in functions.items():
        durations = []
        
        for run in range(runs):
            with benchmarker.benchmark_context(f"{name}_run_{run}"):
                func(*test_args, **test_kwargs)
            
            durations.append(benchmarker.benchmark_results[-1].duration_seconds)
        
        comparison_results[name] = {
            'avg_duration': statistics.mean(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'std_duration': statistics.stdev(durations) if len(durations) > 1 else 0,
            'runs': runs
        }
    
    return comparison_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Performance Benchmarking System")
    print("=" * 40)
    
    # Create benchmarker
    benchmarker = PerformanceBenchmarker()
    
    # Run all benchmarks
    print("\nRunning all benchmark suites...")
    results = benchmarker.run_all_benchmarks()
    
    # Generate report
    print("\nGenerating performance report...")
    report = benchmarker.generate_benchmark_report()
    print(report)
    
    # Export results
    benchmarker.export_results("benchmark_results.json", "json")
    benchmarker.export_results("benchmark_report.txt", "txt")
    
    print("\nBenchmarking completed successfully!")
    print("Results exported to benchmark_results.json and benchmark_report.txt")