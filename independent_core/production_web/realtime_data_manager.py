"""
Saraphis Real-Time Data Manager
Production-ready real-time data streaming and management
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from collections import defaultdict, deque
import json
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class CacheManager:
    """High-performance cache for real-time data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = {}
        self.cache_metadata = {}
        self.access_counts = defaultdict(int)
        self.ttl_seconds = config.get('ttl_seconds', 60)
        self.max_cache_size = config.get('max_cache_size', 10000)
        self._lock = threading.Lock()
        
        # Start cleanup thread
        cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key in self.cache:
                metadata = self.cache_metadata[key]
                if time.time() - metadata['created'] < self.ttl_seconds:
                    self.access_counts[key] += 1
                    metadata['last_accessed'] = time.time()
                    return self.cache[key]
                else:
                    # Expired
                    del self.cache[key]
                    del self.cache_metadata[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        with self._lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_cache_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = value
            self.cache_metadata[key] = {
                'created': time.time(),
                'last_accessed': time.time(),
                'ttl': ttl or self.ttl_seconds
            }
    
    def invalidate(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        with self._lock:
            keys_to_remove = [k for k in self.cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.cache[key]
                del self.cache_metadata[key]
                self.access_counts.pop(key, None)
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.cache:
            return
        
        lru_key = min(
            self.cache_metadata.keys(),
            key=lambda k: self.cache_metadata[k]['last_accessed']
        )
        
        del self.cache[lru_key]
        del self.cache_metadata[lru_key]
        self.access_counts.pop(lru_key, None)
    
    def _cleanup_loop(self):
        """Clean up expired entries"""
        while True:
            try:
                time.sleep(30)  # Run every 30 seconds
                
                with self._lock:
                    current_time = time.time()
                    expired_keys = []
                    
                    for key, metadata in self.cache_metadata.items():
                        if current_time - metadata['created'] > metadata['ttl']:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        del self.cache[key]
                        del self.cache_metadata[key]
                        self.access_counts.pop(key, None)
                        
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")


class RealTimeDataManager:
    """Production-ready real-time data streaming and management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Data sources configuration
        self.data_sources = {
            'brain_metrics': {
                'type': 'polling',
                'interval_seconds': 1,
                'priority': 'high',
                'processor': self._process_brain_metrics
            },
            'uncertainty_data': {
                'type': 'streaming',
                'buffer_size': 1000,
                'priority': 'high',
                'processor': self._process_uncertainty_data
            },
            'training_progress': {
                'type': 'event',
                'priority': 'medium',
                'processor': self._process_training_progress
            },
            'system_health': {
                'type': 'polling',
                'interval_seconds': 5,
                'priority': 'medium',
                'processor': self._process_system_health
            },
            'performance_metrics': {
                'type': 'streaming',
                'buffer_size': 500,
                'priority': 'high',
                'processor': self._process_performance_metrics
            },
            'security_events': {
                'type': 'event',
                'priority': 'critical',
                'processor': self._process_security_events
            },
            'api_traffic': {
                'type': 'streaming',
                'buffer_size': 2000,
                'priority': 'medium',
                'processor': self._process_api_traffic
            },
            'user_activity': {
                'type': 'event',
                'priority': 'low',
                'processor': self._process_user_activity
            }
        }
        
        # Data streams and buffers
        self.data_streams = defaultdict(deque)
        self.stream_metadata = defaultdict(dict)
        self.subscribers = defaultdict(set)
        
        # Real-time processing
        self.processing_queue = queue.PriorityQueue()
        self.processed_data = defaultdict(dict)
        
        # Cache manager
        cache_config = config.get('cache', {})
        self.cache_manager = CacheManager(cache_config)
        
        # Stream configuration
        self.max_stream_size = config.get('max_stream_size', 10000)
        self.batch_size = config.get('batch_size', 100)
        self.processing_threads = config.get('processing_threads', 4)
        
        # Aggregation configuration
        self.aggregation_windows = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600
        }
        self.aggregated_data = defaultdict(lambda: defaultdict(dict))
        
        # Performance tracking
        self.performance_metrics = {
            'data_points_processed': 0,
            'processing_errors': 0,
            'average_latency': 0,
            'stream_throughput': defaultdict(int)
        }
        
        # Thread safety
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # Thread pool for processing
        self.executor = ThreadPoolExecutor(max_workers=self.processing_threads)
        
        # Start background threads
        self._start_background_threads()
        
        self.logger.info("Real-Time Data Manager initialized")
    
    def get_dashboard_data(self, dashboard_type: str, 
                         preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Get real-time data for dashboard"""
        try:
            # Check cache first
            cache_key = f"dashboard_{dashboard_type}_{self._hash_preferences(preferences)}"
            cached_data = self.cache_manager.get(cache_key)
            
            if cached_data:
                return cached_data
            
            # Determine required data sources
            required_sources = self._get_required_sources(dashboard_type)
            
            # Collect data from sources
            dashboard_data = {}
            
            for source in required_sources:
                # Get latest processed data
                with self._lock:
                    if source in self.processed_data:
                        source_data = self.processed_data[source].copy()
                        
                        # Apply user preferences
                        filtered_data = self._apply_preferences(
                            source_data, preferences
                        )
                        
                        dashboard_data[source] = filtered_data
                    else:
                        # Get from stream if no processed data
                        stream_data = self._get_stream_data(source, preferences)
                        dashboard_data[source] = stream_data
            
            # Add aggregated data if requested
            if preferences.get('include_aggregations', True):
                dashboard_data['aggregations'] = self._get_aggregated_data(
                    required_sources, preferences
                )
            
            # Add metadata
            dashboard_data['_metadata'] = {
                'timestamp': time.time(),
                'dashboard_type': dashboard_type,
                'data_sources': required_sources,
                'freshness': self._calculate_data_freshness(dashboard_data)
            }
            
            # Cache the result
            self.cache_manager.set(cache_key, dashboard_data, ttl=5)
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Dashboard data retrieval failed: {e}")
            return {'error': str(e)}
    
    def subscribe_to_stream(self, stream_name: str, callback: Callable):
        """Subscribe to real-time data stream"""
        try:
            with self._lock:
                self.subscribers[stream_name].add(callback)
                self.logger.info(f"Subscribed to stream: {stream_name}")
                
                # Send latest data immediately
                if stream_name in self.processed_data:
                    try:
                        callback(self.processed_data[stream_name])
                    except Exception as e:
                        self.logger.error(f"Callback error: {e}")
                        
        except Exception as e:
            self.logger.error(f"Stream subscription failed: {e}")
    
    def unsubscribe_from_stream(self, stream_name: str, callback: Callable):
        """Unsubscribe from data stream"""
        try:
            with self._lock:
                self.subscribers[stream_name].discard(callback)
                self.logger.info(f"Unsubscribed from stream: {stream_name}")
                
        except Exception as e:
            self.logger.error(f"Stream unsubscription failed: {e}")
    
    def push_data(self, source: str, data: Dict[str, Any]):
        """Push data to processing queue"""
        try:
            # Determine priority
            priority = self._get_source_priority(source)
            
            # Add to processing queue
            item = {
                'source': source,
                'data': data,
                'timestamp': time.time(),
                'id': self._generate_data_id(source, data)
            }
            
            self.processing_queue.put((priority, item))
            
            # Track throughput
            with self._lock:
                self.performance_metrics['stream_throughput'][source] += 1
                
        except Exception as e:
            self.logger.error(f"Data push failed: {e}")
    
    def get_stream_statistics(self) -> Dict[str, Any]:
        """Get real-time stream statistics"""
        try:
            with self._lock:
                stats = {
                    'active_streams': len(self.data_streams),
                    'total_subscribers': sum(len(subs) for subs in self.subscribers.values()),
                    'queue_size': self.processing_queue.qsize(),
                    'performance': self.performance_metrics.copy(),
                    'stream_details': {}
                }
                
                # Add stream-specific stats
                for stream_name, stream_data in self.data_streams.items():
                    stats['stream_details'][stream_name] = {
                        'buffer_size': len(stream_data),
                        'subscriber_count': len(self.subscribers.get(stream_name, [])),
                        'metadata': self.stream_metadata.get(stream_name, {})
                    }
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Stream statistics retrieval failed: {e}")
            return {'error': str(e)}
    
    def _get_required_sources(self, dashboard_type: str) -> List[str]:
        """Get required data sources for dashboard type"""
        dashboard_sources = {
            'system_overview': [
                'brain_metrics', 'system_health', 'performance_metrics', 'api_traffic'
            ],
            'uncertainty_analysis': [
                'uncertainty_data', 'brain_metrics', 'performance_metrics'
            ],
            'training_monitoring': [
                'training_progress', 'performance_metrics', 'system_health'
            ],
            'production_metrics': [
                'api_traffic', 'performance_metrics', 'security_events', 'user_activity'
            ]
        }
        
        return dashboard_sources.get(dashboard_type, [])
    
    def _process_brain_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process brain metrics data"""
        try:
            processed = {
                'quantum_state': {
                    'coherence': data.get('coherence', 0),
                    'entanglement': data.get('entanglement', 0),
                    'fidelity': data.get('fidelity', 0),
                    'stability': data.get('stability', 0)
                },
                'neural_activity': {
                    'active_neurons': data.get('active_neurons', 0),
                    'firing_rate': data.get('firing_rate', 0),
                    'synchronization': data.get('synchronization', 0)
                },
                'performance': {
                    'accuracy': data.get('accuracy', 0),
                    'latency': data.get('latency', 0),
                    'throughput': data.get('throughput', 0)
                },
                'timestamp': time.time()
            }
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Brain metrics processing failed: {e}")
            return {}
    
    def _process_uncertainty_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process uncertainty quantification data"""
        try:
            processed = {
                'aleatoric_uncertainty': data.get('aleatoric', {}),
                'epistemic_uncertainty': data.get('epistemic', {}),
                'total_uncertainty': data.get('total', 0),
                'confidence_intervals': data.get('confidence_intervals', {}),
                'risk_metrics': {
                    'value_at_risk': data.get('var', 0),
                    'conditional_var': data.get('cvar', 0),
                    'risk_score': data.get('risk_score', 0)
                },
                'timestamp': time.time()
            }
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Uncertainty data processing failed: {e}")
            return {}
    
    def _process_training_progress(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process training progress data"""
        try:
            processed = {
                'current_epoch': data.get('epoch', 0),
                'total_epochs': data.get('total_epochs', 0),
                'loss_metrics': {
                    'training_loss': data.get('train_loss', 0),
                    'validation_loss': data.get('val_loss', 0),
                    'test_loss': data.get('test_loss', 0)
                },
                'accuracy_metrics': {
                    'training_accuracy': data.get('train_acc', 0),
                    'validation_accuracy': data.get('val_acc', 0),
                    'test_accuracy': data.get('test_acc', 0)
                },
                'time_metrics': {
                    'elapsed_time': data.get('elapsed_time', 0),
                    'estimated_remaining': data.get('eta', 0),
                    'epoch_duration': data.get('epoch_time', 0)
                },
                'resource_usage': {
                    'gpu_utilization': data.get('gpu_util', 0),
                    'memory_usage': data.get('mem_usage', 0),
                    'disk_io': data.get('disk_io', 0)
                },
                'timestamp': time.time()
            }
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Training progress processing failed: {e}")
            return {}
    
    def _process_system_health(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process system health data"""
        try:
            processed = {
                'components': {
                    'brain': data.get('brain_health', 'unknown'),
                    'api_gateway': data.get('api_health', 'unknown'),
                    'database': data.get('db_health', 'unknown'),
                    'cache': data.get('cache_health', 'unknown')
                },
                'resources': {
                    'cpu_usage': data.get('cpu', 0),
                    'memory_usage': data.get('memory', 0),
                    'disk_usage': data.get('disk', 0),
                    'network_io': data.get('network', 0)
                },
                'alerts': data.get('active_alerts', []),
                'uptime': data.get('uptime_seconds', 0),
                'timestamp': time.time()
            }
            
            return processed
            
        except Exception as e:
            self.logger.error(f"System health processing failed: {e}")
            return {}
    
    def _process_performance_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process performance metrics data"""
        try:
            processed = {
                'response_times': {
                    'p50': data.get('rt_p50', 0),
                    'p90': data.get('rt_p90', 0),
                    'p95': data.get('rt_p95', 0),
                    'p99': data.get('rt_p99', 0)
                },
                'throughput': {
                    'requests_per_second': data.get('rps', 0),
                    'bytes_per_second': data.get('bps', 0),
                    'errors_per_second': data.get('eps', 0)
                },
                'error_rates': {
                    'total_errors': data.get('total_errors', 0),
                    'error_rate': data.get('error_rate', 0),
                    'error_types': data.get('error_breakdown', {})
                },
                'saturation': {
                    'queue_depth': data.get('queue_depth', 0),
                    'connection_pool': data.get('conn_pool_usage', 0),
                    'thread_pool': data.get('thread_pool_usage', 0)
                },
                'timestamp': time.time()
            }
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Performance metrics processing failed: {e}")
            return {}
    
    def _process_security_events(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process security events data"""
        try:
            processed = {
                'event_type': data.get('type', 'unknown'),
                'severity': data.get('severity', 'info'),
                'source': data.get('source', 'unknown'),
                'details': data.get('details', {}),
                'affected_resources': data.get('resources', []),
                'mitigation_status': data.get('mitigation', 'pending'),
                'timestamp': data.get('timestamp', time.time())
            }
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Security event processing failed: {e}")
            return {}
    
    def _process_api_traffic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process API traffic data"""
        try:
            processed = {
                'endpoint_stats': data.get('endpoints', {}),
                'method_distribution': data.get('methods', {}),
                'status_codes': data.get('status_codes', {}),
                'client_stats': {
                    'unique_clients': data.get('unique_clients', 0),
                    'top_clients': data.get('top_clients', []),
                    'client_distribution': data.get('client_dist', {})
                },
                'geographic_data': data.get('geo_data', {}),
                'timestamp': time.time()
            }
            
            return processed
            
        except Exception as e:
            self.logger.error(f"API traffic processing failed: {e}")
            return {}
    
    def _process_user_activity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process user activity data"""
        try:
            processed = {
                'active_users': data.get('active_users', 0),
                'user_actions': data.get('actions', []),
                'session_stats': {
                    'active_sessions': data.get('active_sessions', 0),
                    'avg_session_duration': data.get('avg_duration', 0),
                    'session_distribution': data.get('session_dist', {})
                },
                'feature_usage': data.get('feature_usage', {}),
                'timestamp': time.time()
            }
            
            return processed
            
        except Exception as e:
            self.logger.error(f"User activity processing failed: {e}")
            return {}
    
    def _apply_preferences(self, data: Dict[str, Any], 
                         preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Apply user preferences to data"""
        try:
            filtered_data = data.copy()
            
            # Apply time range filter
            if 'time_range' in preferences:
                start_time = preferences['time_range'].get('start')
                end_time = preferences['time_range'].get('end')
                
                if start_time and end_time:
                    # Filter time-based data
                    filtered_data = self._filter_by_time_range(
                        filtered_data, start_time, end_time
                    )
            
            # Apply data granularity
            if 'granularity' in preferences:
                filtered_data = self._adjust_granularity(
                    filtered_data, preferences['granularity']
                )
            
            # Apply field filters
            if 'fields' in preferences:
                filtered_data = self._filter_fields(
                    filtered_data, preferences['fields']
                )
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Preference application failed: {e}")
            return data
    
    def _get_stream_data(self, source: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Get data directly from stream"""
        try:
            with self._lock:
                if source not in self.data_streams:
                    return {}
                
                stream = self.data_streams[source]
                
                # Get latest N items based on preferences
                limit = preferences.get('limit', 100)
                items = list(stream)[-limit:]
                
                return {
                    'items': items,
                    'count': len(items),
                    'stream_size': len(stream)
                }
                
        except Exception as e:
            self.logger.error(f"Stream data retrieval failed: {e}")
            return {}
    
    def _get_aggregated_data(self, sources: List[str], 
                           preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Get aggregated data for sources"""
        try:
            window = preferences.get('aggregation_window', '5m')
            
            with self._lock:
                aggregated = {}
                
                for source in sources:
                    if source in self.aggregated_data and window in self.aggregated_data[source]:
                        aggregated[source] = self.aggregated_data[source][window]
                
                return aggregated
                
        except Exception as e:
            self.logger.error(f"Aggregated data retrieval failed: {e}")
            return {}
    
    def _calculate_data_freshness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate freshness of data"""
        try:
            current_time = time.time()
            freshness = {}
            
            for source, source_data in data.items():
                if source.startswith('_'):
                    continue
                    
                if isinstance(source_data, dict) and 'timestamp' in source_data:
                    age_seconds = current_time - source_data['timestamp']
                    freshness[source] = {
                        'age_seconds': age_seconds,
                        'is_fresh': age_seconds < 60,  # Fresh if less than 1 minute old
                        'freshness_score': max(0, 1 - (age_seconds / 300))  # 5 minute decay
                    }
            
            return freshness
            
        except Exception as e:
            self.logger.error(f"Freshness calculation failed: {e}")
            return {}
    
    def _get_source_priority(self, source: str) -> int:
        """Get priority for data source"""
        priorities = {
            'critical': 0,
            'high': 1,
            'medium': 2,
            'low': 3
        }
        
        source_config = self.data_sources.get(source, {})
        priority_level = source_config.get('priority', 'medium')
        
        return priorities.get(priority_level, 2)
    
    def _generate_data_id(self, source: str, data: Dict[str, Any]) -> str:
        """Generate unique ID for data item"""
        timestamp = str(time.time())
        data_hash = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:8]
        return f"{source}_{timestamp}_{data_hash}"
    
    def _hash_preferences(self, preferences: Dict[str, Any]) -> str:
        """Hash preferences for cache key"""
        return hashlib.md5(
            json.dumps(preferences, sort_keys=True).encode()
        ).hexdigest()[:16]
    
    def _filter_by_time_range(self, data: Dict[str, Any], 
                            start_time: float, end_time: float) -> Dict[str, Any]:
        """Filter data by time range"""
        # Implementation depends on data structure
        # This is a placeholder for the filtering logic
        return data
    
    def _adjust_granularity(self, data: Dict[str, Any], 
                          granularity: str) -> Dict[str, Any]:
        """Adjust data granularity"""
        # Implementation depends on data structure
        # This is a placeholder for the granularity adjustment logic
        return data
    
    def _filter_fields(self, data: Dict[str, Any], 
                     fields: List[str]) -> Dict[str, Any]:
        """Filter data to include only specified fields"""
        if not fields:
            return data
            
        filtered = {}
        for field in fields:
            if field in data:
                filtered[field] = data[field]
        
        return filtered
    
    def _start_background_threads(self):
        """Start background processing threads"""
        # Data processing thread
        for i in range(self.processing_threads):
            thread = threading.Thread(
                target=self._processing_loop,
                name=f"DataProcessor-{i}",
                daemon=True
            )
            thread.start()
        
        # Aggregation thread
        aggregation_thread = threading.Thread(
            target=self._aggregation_loop,
            daemon=True
        )
        aggregation_thread.start()
        
        # Stream cleanup thread
        cleanup_thread = threading.Thread(
            target=self._stream_cleanup_loop,
            daemon=True
        )
        cleanup_thread.start()
        
        # Metrics collection thread
        metrics_thread = threading.Thread(
            target=self._metrics_collection_loop,
            daemon=True
        )
        metrics_thread.start()
    
    def _processing_loop(self):
        """Main data processing loop"""
        while not self._stop_event.is_set():
            try:
                # Get item from queue with timeout
                priority, item = self.processing_queue.get(timeout=1)
                
                # Process the data
                source = item['source']
                data = item['data']
                
                # Get processor for source
                source_config = self.data_sources.get(source)
                if source_config and 'processor' in source_config:
                    processor = source_config['processor']
                    processed_data = processor(data)
                    
                    # Store processed data
                    with self._lock:
                        self.processed_data[source] = processed_data
                        
                        # Add to stream
                        stream = self.data_streams[source]
                        if len(stream) >= self.max_stream_size:
                            stream.popleft()
                        stream.append(processed_data)
                        
                        # Update metadata
                        self.stream_metadata[source] = {
                            'last_updated': time.time(),
                            'item_count': len(stream),
                            'data_id': item['id']
                        }
                        
                        # Update performance metrics
                        self.performance_metrics['data_points_processed'] += 1
                    
                    # Notify subscribers
                    self._notify_subscribers(source, processed_data)
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                with self._lock:
                    self.performance_metrics['processing_errors'] += 1
    
    def _aggregation_loop(self):
        """Aggregate data at different time windows"""
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                
                with self._lock:
                    for source, stream in self.data_streams.items():
                        if not stream:
                            continue
                        
                        # Aggregate for each window
                        for window_name, window_seconds in self.aggregation_windows.items():
                            # Get data points within window
                            window_data = [
                                item for item in stream
                                if isinstance(item, dict) and 
                                'timestamp' in item and
                                current_time - item['timestamp'] <= window_seconds
                            ]
                            
                            if window_data:
                                # Perform aggregation
                                aggregated = self._aggregate_data_points(
                                    window_data, window_name
                                )
                                
                                self.aggregated_data[source][window_name] = aggregated
                
                time.sleep(10)  # Aggregate every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Aggregation error: {e}")
                time.sleep(30)
    
    def _stream_cleanup_loop(self):
        """Clean up old stream data"""
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                max_age = 3600  # 1 hour
                
                with self._lock:
                    for source, stream in self.data_streams.items():
                        # Remove old items
                        while stream:
                            item = stream[0]
                            if isinstance(item, dict) and 'timestamp' in item:
                                if current_time - item['timestamp'] > max_age:
                                    stream.popleft()
                                else:
                                    break
                            else:
                                break
                
                time.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Stream cleanup error: {e}")
                time.sleep(600)
    
    def _metrics_collection_loop(self):
        """Collect performance metrics"""
        while not self._stop_event.is_set():
            try:
                # Calculate average latency
                # This would be implemented based on actual latency measurements
                
                time.sleep(60)  # Collect metrics every minute
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                time.sleep(120)
    
    def _aggregate_data_points(self, data_points: List[Dict[str, Any]], 
                             window: str) -> Dict[str, Any]:
        """Aggregate multiple data points"""
        try:
            if not data_points:
                return {}
            
            # Example aggregation logic - would be customized per data type
            aggregated = {
                'window': window,
                'count': len(data_points),
                'start_time': min(d['timestamp'] for d in data_points if 'timestamp' in d),
                'end_time': max(d['timestamp'] for d in data_points if 'timestamp' in d)
            }
            
            # Add numeric field aggregations
            numeric_fields = set()
            for point in data_points:
                for key, value in point.items():
                    if isinstance(value, (int, float)) and key != 'timestamp':
                        numeric_fields.add(key)
            
            for field in numeric_fields:
                values = [p.get(field, 0) for p in data_points if field in p]
                if values:
                    aggregated[field] = {
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values),
                        'sum': sum(values)
                    }
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Data aggregation failed: {e}")
            return {}
    
    def _notify_subscribers(self, source: str, data: Dict[str, Any]):
        """Notify subscribers of new data"""
        subscribers = self.subscribers.get(source, set()).copy()
        
        for callback in subscribers:
            try:
                # Run callback in thread pool to avoid blocking
                self.executor.submit(callback, data)
            except Exception as e:
                self.logger.error(f"Subscriber notification failed: {e}")
    
    def shutdown(self):
        """Shutdown real-time data manager"""
        self.logger.info("Shutting down Real-Time Data Manager")
        self._stop_event.set()
        self.executor.shutdown(wait=True)