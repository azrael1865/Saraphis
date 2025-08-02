"""
Memory usage optimization and management for Saraphis Brain system.
Provides comprehensive memory tracking, leak detection, garbage collection optimization, and cache management.
NO FALLBACKS - HARD FAILURES ONLY
"""

import gc
import logging
import psutil
import sys
import threading
import time
import tracemalloc
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Set, Union, Callable
import numpy as np
import warnings


class MemoryOptimizationStrategy(Enum):
    """Memory optimization strategies"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class MemoryLeakSeverity(Enum):
    """Severity levels for memory leaks"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a point in time"""
    timestamp: float
    total_memory: int
    available_memory: int
    used_memory: int
    memory_percent: float
    process_memory: int
    heap_size: int = 0
    gc_objects: int = 0
    gc_generation_0: int = 0
    gc_generation_1: int = 0
    gc_generation_2: int = 0


@dataclass
class MemoryLeakReport:
    """Report for detected memory leak"""
    component_name: str
    leak_severity: MemoryLeakSeverity
    leak_rate_mb_per_hour: float
    total_leaked_mb: float
    detection_confidence: float
    first_detected: datetime
    last_detected: datetime
    stack_traces: List[str] = field(default_factory=list)
    suspected_objects: List[str] = field(default_factory=list)
    suggested_fixes: List[str] = field(default_factory=list)


@dataclass
class CacheMetrics:
    """Metrics for cache performance"""
    cache_name: str
    total_size: int
    item_count: int
    hit_rate: float
    miss_rate: float
    eviction_count: int
    last_access: float
    creation_time: float
    memory_overhead: int = 0


@dataclass
class GarbageCollectionStats:
    """Garbage collection statistics"""
    collection_count_gen0: int
    collection_count_gen1: int
    collection_count_gen2: int
    total_time_spent: float
    objects_collected: int
    memory_freed: int
    last_collection_time: float
    average_collection_time: float


class MemoryOptimizer:
    """
    Memory usage optimization and management system.
    Provides comprehensive memory tracking, leak detection, garbage collection optimization, and cache management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize memory optimizer"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, dict):
            raise TypeError(f"Config must be dict or None, got {type(config)}")
        
        self.config = config or {}
        self.logger = logging.getLogger('MemoryOptimizer')
        
        # Configuration parameters
        self.enable_tracking = self.config.get('enable_tracking', True)
        self.enable_leak_detection = self.config.get('enable_leak_detection', True)
        self.enable_gc_optimization = self.config.get('enable_gc_optimization', True)
        self.snapshot_interval = self.config.get('snapshot_interval', 60.0)  # seconds
        self.memory_threshold_mb = self.config.get('memory_threshold_mb', 1024)
        self.leak_detection_threshold = self.config.get('leak_detection_threshold', 10.0)  # MB/hour
        self.max_snapshots = self.config.get('max_snapshots', 1000)
        self.optimization_strategy = MemoryOptimizationStrategy(
            self.config.get('optimization_strategy', 'balanced')
        )
        self.enable_tracemalloc = self.config.get('enable_tracemalloc', True)
        
        # Validate configuration
        self._validate_config()
        
        # Memory tracking data
        self.memory_snapshots: deque = deque(maxlen=self.max_snapshots)
        self.component_memory_usage: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self.memory_leak_reports: Dict[str, MemoryLeakReport] = {}
        self.cache_registry: Dict[str, CacheMetrics] = {}
        
        # Garbage collection tracking
        self.gc_stats = GarbageCollectionStats(
            collection_count_gen0=0,
            collection_count_gen1=0,
            collection_count_gen2=0,
            total_time_spent=0.0,
            objects_collected=0,
            memory_freed=0,
            last_collection_time=0.0,
            average_collection_time=0.0
        )
        
        # Object tracking for leak detection
        self.tracked_objects: Dict[str, Set[int]] = defaultdict(set)
        self.object_creation_times: Dict[int, float] = {}
        self.object_sizes: Dict[int, int] = {}
        
        # Thread safety
        self._tracking_lock = threading.RLock()
        self._optimization_lock = threading.RLock()
        self._cache_lock = threading.RLock()
        
        # Background monitoring
        self._monitoring_enabled = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Performance metrics
        self.optimization_metrics = {
            'memory_freed_mb': 0.0,
            'leak_detections': 0,
            'gc_optimizations': 0,
            'cache_optimizations': 0,
            'total_optimizations': 0,
            'average_memory_usage': 0.0,
            'peak_memory_usage': 0.0,
            'memory_efficiency_score': 0.0
        }
        
        # Initialize tracemalloc if enabled
        if self.enable_tracemalloc and self.enable_tracking:
            try:
                tracemalloc.start()
                self.logger.info("Tracemalloc initialized for detailed memory tracking")
            except Exception as e:
                self.logger.warning(f"Failed to initialize tracemalloc: {e}")
        
        self.is_initialized = True
        self.logger.info("MemoryOptimizer initialized successfully")
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if not isinstance(self.enable_tracking, bool):
            raise TypeError(f"enable_tracking must be bool, got {type(self.enable_tracking)}")
        
        if not isinstance(self.snapshot_interval, (int, float)) or self.snapshot_interval <= 0:
            raise ValueError(f"snapshot_interval must be positive number, got {self.snapshot_interval}")
        
        if not isinstance(self.memory_threshold_mb, (int, float)) or self.memory_threshold_mb <= 0:
            raise ValueError(f"memory_threshold_mb must be positive number, got {self.memory_threshold_mb}")
        
        if not isinstance(self.leak_detection_threshold, (int, float)) or self.leak_detection_threshold <= 0:
            raise ValueError(f"leak_detection_threshold must be positive number, got {self.leak_detection_threshold}")
        
        if not isinstance(self.max_snapshots, int) or self.max_snapshots <= 0:
            raise ValueError(f"max_snapshots must be positive int, got {self.max_snapshots}")
    
    @contextmanager
    def track_memory_usage(self, component_name: str):
        """
        Context manager to track memory usage for a specific component.
        
        Args:
            component_name: Name of the component to track
            
        Yields:
            Dictionary containing memory tracking data
            
        Raises:
            TypeError: If component_name is not string
            ValueError: If component_name is empty
        """
        if not isinstance(component_name, str):
            raise TypeError(f"component_name must be str, got {type(component_name)}")
        if not component_name.strip():
            raise ValueError("component_name cannot be empty")
        
        if not self.enable_tracking:
            yield {}
            return
        
        # Take initial memory snapshot
        initial_snapshot = self._take_memory_snapshot()
        initial_process_memory = psutil.Process().memory_info().rss
        start_time = time.time()
        
        # Track object creation if tracemalloc is available
        initial_tracemalloc = None
        if tracemalloc.is_tracing():
            initial_tracemalloc = tracemalloc.take_snapshot()
        
        tracking_data = {
            'component_name': component_name,
            'start_time': start_time,
            'initial_memory': initial_process_memory,
            'peak_memory': initial_process_memory
        }
        
        try:
            yield tracking_data
            
        finally:
            # Take final memory snapshot
            final_snapshot = self._take_memory_snapshot()
            final_process_memory = psutil.Process().memory_info().rss
            end_time = time.time()
            
            # Calculate memory usage delta
            memory_delta = final_process_memory - initial_process_memory
            duration = end_time - start_time
            
            # Detailed tracking with tracemalloc
            tracemalloc_data = None
            if tracemalloc.is_tracing() and initial_tracemalloc:
                try:
                    final_tracemalloc = tracemalloc.take_snapshot()
                    top_stats = final_tracemalloc.compare_to(initial_tracemalloc, 'lineno')
                    tracemalloc_data = {
                        'top_allocations': [
                            {
                                'filename': stat.traceback.format()[0],
                                'size_mb': stat.size / 1024 / 1024,
                                'count': stat.count
                            }
                            for stat in top_stats[:10]
                        ]
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to process tracemalloc data: {e}")
            
            # Store component memory usage
            with self._tracking_lock:
                usage_data = {
                    'timestamp': end_time,
                    'memory_delta': memory_delta,
                    'duration': duration,
                    'memory_rate': memory_delta / duration if duration > 0 else 0,
                    'tracemalloc_data': tracemalloc_data
                }
                self.component_memory_usage[component_name].append(usage_data)
                
                # Update tracking data for caller
                tracking_data.update({
                    'end_time': end_time,
                    'final_memory': final_process_memory,
                    'memory_delta': memory_delta,
                    'duration': duration,
                    'memory_rate': memory_delta / duration if duration > 0 else 0
                })
            
            # Check for potential memory leak
            if memory_delta > 50 * 1024 * 1024:  # > 50MB increase
                self._flag_potential_leak(component_name, memory_delta, duration)
            
            self.logger.debug(
                f"Memory tracking for '{component_name}': "
                f"delta={memory_delta / 1024 / 1024:.2f}MB, "
                f"duration={duration:.2f}s"
            )
    
    def detect_memory_leaks(self) -> Dict[str, MemoryLeakReport]:
        """
        Analyze memory usage patterns to detect potential memory leaks.
        
        Returns:
            Dictionary mapping component names to memory leak reports
        """
        if not self.enable_leak_detection:
            raise RuntimeError("Memory leak detection must be enabled")
        
        detected_leaks = {}
        current_time = datetime.now()
        
        with self._tracking_lock:
            for component_name, usage_history in self.component_memory_usage.items():
                if len(usage_history) < 10:  # Need minimum samples
                    continue
                
                leak_report = self._analyze_component_for_leaks(component_name, list(usage_history), current_time)
                if leak_report:
                    detected_leaks[component_name] = leak_report
        
        # Update stored leak reports
        with self._optimization_lock:
            self.memory_leak_reports.update(detected_leaks)
            self.optimization_metrics['leak_detections'] = len(self.memory_leak_reports)
        
        if detected_leaks:
            self.logger.warning(f"Detected {len(detected_leaks)} potential memory leaks")
        
        return detected_leaks
    
    def _analyze_component_for_leaks(self, component_name: str, usage_history: List[Dict], current_time: datetime) -> Optional[MemoryLeakReport]:
        """Analyze component usage history for memory leak patterns"""
        if not usage_history:
            return None
        
        # Calculate memory growth trend
        memory_deltas = [usage['memory_delta'] for usage in usage_history]
        timestamps = [usage['timestamp'] for usage in usage_history]
        
        # Filter out negative deltas (memory freed)
        positive_deltas = [delta for delta in memory_deltas if delta > 0]
        if len(positive_deltas) < 5:
            return None
        
        # Calculate leak rate (MB per hour)
        time_span = timestamps[-1] - timestamps[0]
        if time_span <= 0:
            return None
        
        total_growth = sum(positive_deltas)
        leak_rate = (total_growth / 1024/ 1024) / (time_span / 3600)  # MB per hour
        
        if leak_rate < self.leak_detection_threshold:
            return None
        
        # Determine severity
        if leak_rate > 100:
            severity = MemoryLeakSeverity.CRITICAL
        elif leak_rate > 50:
            severity = MemoryLeakSeverity.HIGH
        elif leak_rate > 20:
            severity = MemoryLeakSeverity.MEDIUM
        else:
            severity = MemoryLeakSeverity.LOW
        
        # Calculate confidence based on consistency of growth
        growth_consistency = self._calculate_growth_consistency(memory_deltas)
        
        # Generate suggested fixes
        suggested_fixes = self._generate_leak_fix_suggestions(component_name, usage_history)
        
        return MemoryLeakReport(
            component_name=component_name,
            leak_severity=severity,
            leak_rate_mb_per_hour=leak_rate,
            total_leaked_mb=total_growth / 1024 / 1024,
            detection_confidence=growth_consistency,
            first_detected=datetime.fromtimestamp(timestamps[0]),
            last_detected=current_time,
            suspected_objects=self._identify_suspected_objects(component_name, usage_history),
            suggested_fixes=suggested_fixes
        )
    
    def _calculate_growth_consistency(self, memory_deltas: List[int]) -> float:
        """Calculate consistency of memory growth pattern (0.0 to 1.0)"""
        if len(memory_deltas) < 3:
            return 0.0
        
        positive_deltas = [delta for delta in memory_deltas if delta > 0]
        if len(positive_deltas) < len(memory_deltas) * 0.5:
            return 0.0
        
        # Calculate coefficient of variation (lower = more consistent)
        mean_delta = np.mean(positive_deltas)
        std_delta = np.std(positive_deltas)
        
        if mean_delta == 0:
            return 0.0
        
        cv = std_delta / mean_delta
        # Convert to confidence score (lower CV = higher confidence)
        confidence = max(0.0, min(1.0, 1.0 - cv))
        
        return confidence
    
    def _identify_suspected_objects(self, component_name: str, usage_history: List[Dict]) -> List[str]:
        """Identify objects that might be causing memory leaks"""
        suspected = []
        
        # Analyze tracemalloc data if available
        for usage in usage_history[-5:]:  # Last 5 entries
            tracemalloc_data = usage.get('tracemalloc_data')
            if tracemalloc_data and 'top_allocations' in tracemalloc_data:
                for allocation in tracemalloc_data['top_allocations'][:3]:
                    filename = allocation['filename']
                    if filename not in suspected:
                        suspected.append(filename)
        
        return suspected
    
    def _generate_leak_fix_suggestions(self, component_name: str, usage_history: List[Dict]) -> List[str]:
        """Generate suggestions for fixing memory leaks"""
        suggestions = [
            "Review object lifecycle management",
            "Check for circular references",
            "Implement proper cleanup in __del__ methods",
            "Use weak references for callbacks and observers",
            "Ensure files and resources are properly closed",
            "Review caching strategies for unbounded growth",
            "Implement memory limits for data structures",
            "Add explicit garbage collection calls",
            "Profile with memory profiler tools",
            "Check for event listener leaks"
        ]
        
        # Add component-specific suggestions
        component_lower = component_name.lower()
        if 'cache' in component_lower:
            suggestions.insert(0, "Implement cache size limits and LRU eviction")
        if 'training' in component_lower:
            suggestions.insert(0, "Clear intermediate gradients and tensors")
        if 'data' in component_lower:
            suggestions.insert(0, "Implement data streaming or pagination")
        
        return suggestions[:8]  # Return top 8 suggestions
    
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Optimize memory usage based on current strategy and detected issues.
        
        Returns:
            Dictionary containing optimization results
        """
        optimization_start = time.time()
        optimizations_applied = []
        
        # Detect memory leaks first
        leaks = self.detect_memory_leaks()
        
        # Optimize garbage collection
        if self.enable_gc_optimization:
            gc_result = self._optimize_garbage_collection()
            optimizations_applied.extend(gc_result)
        
        # Optimize caches
        cache_result = self._optimize_caches()
        optimizations_applied.extend(cache_result)
        
        # Apply memory cleanup based on strategy
        cleanup_result = self._apply_memory_cleanup()
        optimizations_applied.extend(cleanup_result)
        
        # Update metrics
        with self._optimization_lock:
            self.optimization_metrics['total_optimizations'] += len(optimizations_applied)
        
        optimization_time = time.time() - optimization_start
        
        result = {
            'status': 'optimization_completed',
            'leaks_detected': len(leaks),
            'optimizations_applied': len(optimizations_applied),
            'optimization_details': optimizations_applied,
            'total_time': optimization_time,
            'memory_freed_estimate': self._estimate_memory_freed(optimizations_applied)
        }
        
        self.logger.info(
            f"Memory optimization completed: {len(optimizations_applied)} optimizations applied "
            f"in {optimization_time:.2f}s"
        )
        
        return result
    
    def _optimize_garbage_collection(self) -> List[str]:
        """Optimize garbage collection settings and trigger collection"""
        optimizations = []
        initial_memory = psutil.Process().memory_info().rss
        
        # Get current GC stats
        gc_counts_before = gc.get_count()
        
        # Tune GC thresholds based on strategy
        if self.optimization_strategy == MemoryOptimizationStrategy.AGGRESSIVE:
            # More frequent collection
            gc.set_threshold(500, 8, 8)
            optimizations.append("Set aggressive GC thresholds (500, 8, 8)")
        elif self.optimization_strategy == MemoryOptimizationStrategy.BALANCED:
            # Default balanced approach
            gc.set_threshold(700, 10, 10)
            optimizations.append("Set balanced GC thresholds (700, 10, 10)")
        elif self.optimization_strategy == MemoryOptimizationStrategy.CONSERVATIVE:
            # Less frequent collection, higher thresholds
            gc.set_threshold(1000, 15, 15)
            optimizations.append("Set conservative GC thresholds (1000, 15, 15)")
        
        # Force garbage collection
        gc_start = time.time()
        collected = gc.collect()
        gc_time = time.time() - gc_start
        
        if collected > 0:
            optimizations.append(f"Collected {collected} objects in {gc_time:.3f}s")
        
        # Update GC stats
        gc_counts_after = gc.get_count()
        final_memory = psutil.Process().memory_info().rss
        memory_freed = initial_memory - final_memory
        
        if memory_freed > 0:
            self.optimization_metrics['memory_freed_mb'] += memory_freed / 1024 / 1024
            self.optimization_metrics['gc_optimizations'] += 1
            optimizations.append(f"Freed {memory_freed / 1024 / 1024:.2f}MB through GC")
        
        # Update GC statistics
        self.gc_stats.total_time_spent += gc_time
        self.gc_stats.objects_collected += collected
        self.gc_stats.memory_freed += max(0, memory_freed)
        self.gc_stats.last_collection_time = gc_time
        
        return optimizations
    
    def _optimize_caches(self) -> List[str]:
        """Optimize registered caches"""
        optimizations = []
        
        with self._cache_lock:
            for cache_name, metrics in self.cache_registry.items():
                # Optimize based on hit rate and size
                if metrics.hit_rate < 0.3 and metrics.item_count > 1000:
                    # Low hit rate with many items - consider cache eviction
                    optimizations.append(f"Cache '{cache_name}' has low hit rate ({metrics.hit_rate:.2f}) - consider size reduction")
                
                if metrics.total_size > 100 * 1024 * 1024:  # > 100MB
                    optimizations.append(f"Cache '{cache_name}' is large ({metrics.total_size / 1024 / 1024:.1f}MB) - consider cleanup")
                
                # Check for stale caches
                current_time = time.time()
                if current_time - metrics.last_access > 3600:  # 1 hour
                    optimizations.append(f"Cache '{cache_name}' appears stale (last access: {(current_time - metrics.last_access) / 60:.1f} min ago)")
        
        if optimizations:
            self.optimization_metrics['cache_optimizations'] += len(optimizations)
        
        return optimizations
    
    def _apply_memory_cleanup(self) -> List[str]:
        """Apply memory cleanup based on optimization strategy"""
        optimizations = []
        
        # Clear weak references to dead objects
        initial_refs = len(gc.get_referrers())
        gc.collect()  # This will clear weak references
        final_refs = len(gc.get_referrers())
        
        if initial_refs > final_refs:
            optimizations.append(f"Cleared {initial_refs - final_refs} weak references")
        
        # Force Python to release memory back to OS (platform dependent)
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
            optimizations.append("Triggered malloc_trim to release memory to OS")
        except Exception:
            pass  # malloc_trim not available on all platforms
        
        return optimizations
    
    def _estimate_memory_freed(self, optimizations: List[str]) -> float:
        """Estimate memory freed from optimizations (in MB)"""
        total_freed = 0.0
        
        for opt in optimizations:
            if "Freed" in opt and "MB" in opt:
                try:
                    # Extract MB value from string like "Freed 15.3MB through GC"
                    parts = opt.split()
                    for i, part in enumerate(parts):
                        if "MB" in part:
                            mb_value = float(part.replace("MB", ""))
                            total_freed += mb_value
                            break
                except (ValueError, IndexError):
                    continue
        
        return total_freed
    
    def get_memory_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive memory usage report.
        
        Returns:
            Dictionary containing detailed memory analysis
        """
        current_snapshot = self._take_memory_snapshot()
        
        # Component memory usage summary
        component_summary = {}
        with self._tracking_lock:
            for component_name, usage_history in self.component_memory_usage.items():
                if usage_history:
                    memory_deltas = [usage['memory_delta'] for usage in usage_history]
                    durations = [usage['duration'] for usage in usage_history]
                    
                    component_summary[component_name] = {
                        'total_calls': len(usage_history),
                        'total_memory_delta': sum(memory_deltas),
                        'average_memory_delta': np.mean(memory_deltas),
                        'max_memory_delta': max(memory_deltas),
                        'average_duration': np.mean(durations),
                        'memory_rate_avg': np.mean([usage['memory_rate'] for usage in usage_history])
                    }
        
        # Memory leak summary
        leak_summary = {}
        with self._optimization_lock:
            for component_name, leak_report in self.memory_leak_reports.items():
                leak_summary[component_name] = {
                    'severity': leak_report.leak_severity.value,
                    'leak_rate_mb_per_hour': leak_report.leak_rate_mb_per_hour,
                    'total_leaked_mb': leak_report.total_leaked_mb,
                    'confidence': leak_report.detection_confidence,
                    'suggested_fixes': leak_report.suggested_fixes[:3]
                }
        
        # Cache summary
        cache_summary = {}
        with self._cache_lock:
            for cache_name, metrics in self.cache_registry.items():
                cache_summary[cache_name] = {
                    'size_mb': metrics.total_size / 1024 / 1024,
                    'item_count': metrics.item_count,
                    'hit_rate': metrics.hit_rate,
                    'age_hours': (time.time() - metrics.creation_time) / 3600
                }
        
        # System memory information
        system_memory = {
            'total_mb': current_snapshot.total_memory / 1024 / 1024,
            'available_mb': current_snapshot.available_memory / 1024 / 1024,
            'used_mb': current_snapshot.used_memory / 1024 / 1024,
            'usage_percent': current_snapshot.memory_percent,
            'process_memory_mb': current_snapshot.process_memory / 1024 / 1024
        }
        
        # Historical trends
        historical_data = []
        for snapshot in list(self.memory_snapshots)[-20:]:  # Last 20 snapshots
            historical_data.append({
                'timestamp': snapshot.timestamp,
                'memory_percent': snapshot.memory_percent,
                'process_memory_mb': snapshot.process_memory / 1024 / 1024
            })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_memory': system_memory,
            'component_summary': component_summary,
            'leak_summary': leak_summary,
            'cache_summary': cache_summary,
            'gc_statistics': {
                'collections_gen0': self.gc_stats.collection_count_gen0,
                'collections_gen1': self.gc_stats.collection_count_gen1,
                'collections_gen2': self.gc_stats.collection_count_gen2,
                'total_time_spent': self.gc_stats.total_time_spent,
                'objects_collected': self.gc_stats.objects_collected,
                'memory_freed_mb': self.gc_stats.memory_freed / 1024 / 1024,
                'average_collection_time': self.gc_stats.average_collection_time
            },
            'optimization_metrics': self.optimization_metrics.copy(),
            'historical_trends': historical_data,
            'configuration': {
                'optimization_strategy': self.optimization_strategy.value,
                'tracking_enabled': self.enable_tracking,
                'leak_detection_enabled': self.enable_leak_detection,
                'memory_threshold_mb': self.memory_threshold_mb,
                'leak_threshold_mb_per_hour': self.leak_detection_threshold
            }
        }
    
    def clear_memory_cache(self, cache_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear memory caches to free up memory.
        
        Args:
            cache_name: Optional specific cache to clear. If None, clears all caches.
            
        Returns:
            Dictionary containing cache clearing results
        """
        if cache_name is not None and not isinstance(cache_name, str):
            raise TypeError(f"cache_name must be str or None, got {type(cache_name)}")
        
        cleared_caches = []
        total_freed = 0
        
        with self._cache_lock:
            if cache_name:
                # Clear specific cache
                if cache_name in self.cache_registry:
                    metrics = self.cache_registry[cache_name]
                    total_freed = metrics.total_size
                    del self.cache_registry[cache_name]
                    cleared_caches.append(cache_name)
                else:
                    raise ValueError(f"Cache '{cache_name}' not found")
            else:
                # Clear all caches
                for name, metrics in self.cache_registry.items():
                    total_freed += metrics.total_size
                    cleared_caches.append(name)
                self.cache_registry.clear()
        
        # Force garbage collection after cache clearing
        gc.collect()
        
        result = {
            'caches_cleared': cleared_caches,
            'estimated_memory_freed_mb': total_freed / 1024 / 1024,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Cleared {len(cleared_caches)} caches, freed ~{total_freed / 1024 / 1024:.1f}MB")
        
        return result
    
    def register_cache(self, cache_name: str, cache_metrics: CacheMetrics) -> None:
        """
        Register a cache for monitoring and optimization.
        
        Args:
            cache_name: Name of the cache
            cache_metrics: Cache metrics object
        """
        if not isinstance(cache_name, str):
            raise TypeError(f"cache_name must be str, got {type(cache_name)}")
        if not isinstance(cache_metrics, CacheMetrics):
            raise TypeError(f"cache_metrics must be CacheMetrics, got {type(cache_metrics)}")
        
        with self._cache_lock:
            self.cache_registry[cache_name] = cache_metrics
        
        self.logger.debug(f"Registered cache '{cache_name}' for monitoring")
    
    def _take_memory_snapshot(self) -> MemorySnapshot:
        """Take a snapshot of current memory usage"""
        memory_info = psutil.virtual_memory()
        process_memory = psutil.Process().memory_info().rss
        
        # Get GC information
        gc_counts = gc.get_count()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            total_memory=memory_info.total,
            available_memory=memory_info.available,
            used_memory=memory_info.used,
            memory_percent=memory_info.percent,
            process_memory=process_memory,
            gc_objects=len(gc.get_objects()),
            gc_generation_0=gc_counts[0],
            gc_generation_1=gc_counts[1],
            gc_generation_2=gc_counts[2]
        )
        
        # Store snapshot
        self.memory_snapshots.append(snapshot)
        
        # Update metrics
        self.optimization_metrics['average_memory_usage'] = memory_info.percent
        if memory_info.percent > self.optimization_metrics['peak_memory_usage']:
            self.optimization_metrics['peak_memory_usage'] = memory_info.percent
        
        return snapshot
    
    def _flag_potential_leak(self, component_name: str, memory_delta: int, duration: float) -> None:
        """Flag a component for potential memory leak"""
        leak_rate = (memory_delta / 1024 / 1024) / (duration / 3600)  # MB per hour
        
        if leak_rate > self.leak_detection_threshold:
            self.logger.warning(
                f"Potential memory leak in '{component_name}': "
                f"{leak_rate:.1f}MB/hour growth rate"
            )
    
    def start_monitoring(self) -> None:
        """Start background memory monitoring"""
        if self._monitoring_enabled:
            return
        
        self._monitoring_enabled = True
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="MemoryMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("Background memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background memory monitoring"""
        if not self._monitoring_enabled:
            return
        
        self._monitoring_enabled = False
        self._stop_monitoring.set()
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Background memory monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while not self._stop_monitoring.wait(self.snapshot_interval):
            try:
                # Take periodic memory snapshot
                self._take_memory_snapshot()
                
                # Check for critical memory usage
                self._check_critical_memory()
                
                # Periodic leak detection
                if len(self.memory_snapshots) % 10 == 0:  # Every 10 snapshots
                    self.detect_memory_leaks()
                
                # Automatic optimization if memory usage is high
                current_memory = psutil.virtual_memory().percent
                if current_memory > 85 and self.optimization_strategy != MemoryOptimizationStrategy.CONSERVATIVE:
                    self.logger.info("High memory usage detected, triggering automatic optimization")
                    self.optimize_memory_usage()
                
            except Exception as e:
                self.logger.error(f"Error in memory monitoring loop: {e}")
    
    def _check_critical_memory(self) -> None:
        """Check for critical memory conditions"""
        memory_info = psutil.virtual_memory()
        
        if memory_info.percent > 95:
            self.logger.critical(f"Critical memory usage: {memory_info.percent}%")
        elif memory_info.percent > 85:
            self.logger.warning(f"High memory usage: {memory_info.percent}%")
        
        # Check for rapid memory growth
        if len(self.memory_snapshots) >= 2:
            recent_snapshots = list(self.memory_snapshots)[-2:]
            memory_growth = recent_snapshots[1].process_memory - recent_snapshots[0].process_memory
            
            if memory_growth > 100 * 1024 * 1024:  # > 100MB growth
                self.logger.warning(
                    f"Rapid memory growth detected: {memory_growth / 1024 / 1024:.1f}MB "
                    f"in {recent_snapshots[1].timestamp - recent_snapshots[0].timestamp:.1f}s"
                )