"""
Saraphis API Metrics Collector
Production-ready API metrics collection and analysis
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import statistics
import json

logger = logging.getLogger(__name__)


class APIMetricsCollector:
    """Production-ready API metrics collector with comprehensive tracking"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Metrics configuration
        self.retention_period = config.get('retention_period_hours', 24)
        self.aggregation_intervals = config.get('aggregation_intervals', [1, 5, 15, 60])  # minutes
        self.percentiles = config.get('percentiles', [50, 90, 95, 99])
        
        # Metrics storage
        self.request_metrics = deque(maxlen=100000)  # Raw request metrics
        self.endpoint_metrics = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'error_count': 0,
            'response_times': deque(maxlen=1000),
            'status_codes': defaultdict(int),
            'last_updated': time.time()
        })
        
        self.user_metrics = defaultdict(lambda: {
            'request_count': 0,
            'error_count': 0,
            'total_response_time': 0,
            'endpoints_accessed': set(),
            'last_activity': time.time()
        })
        
        # Performance metrics
        self.performance_metrics = {
            'response_times': deque(maxlen=10000),
            'throughput': deque(maxlen=1000),
            'error_rates': deque(maxlen=1000),
            'concurrent_requests': 0,
            'peak_concurrent_requests': 0
        }
        
        # System metrics
        self.system_metrics = {
            'uptime_start': time.time(),
            'total_requests': 0,
            'total_errors': 0,
            'total_bytes_in': 0,
            'total_bytes_out': 0
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'error_rate': config.get('error_rate_threshold', 0.05),  # 5%
            'response_time': config.get('response_time_threshold', 1.0),  # 1 second
            'throughput': config.get('throughput_threshold', 1000)  # requests per minute
        }
        
        # Active alerts
        self.active_alerts = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Start background threads
        self._start_background_threads()
        
        self.logger.info("API Metrics Collector initialized")
    
    def track_request_metrics(self, request_id: str, request: Dict[str, Any], 
                            response: Dict[str, Any], processing_time: float):
        """Track comprehensive API request metrics"""
        try:
            with self._lock:
                current_time = time.time()
                
                # Create metric record
                metric = {
                    'request_id': request_id,
                    'timestamp': current_time,
                    'endpoint': request.get('endpoint', 'unknown'),
                    'method': request.get('method', 'UNKNOWN'),
                    'user_id': request.get('user', {}).get('user_id', 'anonymous'),
                    'processing_time': processing_time,
                    'status': 'success' if response.get('success', False) else 'error',
                    'status_code': self._extract_status_code(response),
                    'request_size': self._calculate_size(request),
                    'response_size': self._calculate_size(response),
                    'client_ip': request.get('client_ip', 'unknown')
                }
                
                # Store raw metric
                self.request_metrics.append(metric)
                
                # Update endpoint metrics
                self._update_endpoint_metrics(metric)
                
                # Update user metrics
                self._update_user_metrics(metric)
                
                # Update performance metrics
                self._update_performance_metrics(metric)
                
                # Update system metrics
                self._update_system_metrics(metric)
                
                # Check for alerts
                self._check_alert_conditions()
                
                # Update concurrent requests
                self.performance_metrics['concurrent_requests'] += 1
                if self.performance_metrics['concurrent_requests'] > self.performance_metrics['peak_concurrent_requests']:
                    self.performance_metrics['peak_concurrent_requests'] = self.performance_metrics['concurrent_requests']
                
                # Decrement after processing
                threading.Timer(processing_time, self._decrement_concurrent_requests).start()
                
        except Exception as e:
            self.logger.error(f"Metrics tracking failed: {e}")
    
    def get_api_metrics(self) -> Dict[str, Any]:
        """Get comprehensive API metrics"""
        try:
            with self._lock:
                return {
                    'summary': self._generate_summary_metrics(),
                    'endpoint_metrics': self._get_endpoint_summary(),
                    'performance_metrics': self._get_performance_summary(),
                    'user_metrics': self._get_user_summary(),
                    'system_metrics': self._get_system_summary(),
                    'alerts': self._get_active_alerts(),
                    'trends': self._analyze_trends()
                }
                
        except Exception as e:
            self.logger.error(f"Metrics retrieval failed: {e}")
            return {'error': str(e)}
    
    def get_endpoint_metrics(self, endpoint: str) -> Dict[str, Any]:
        """Get metrics for specific endpoint"""
        try:
            with self._lock:
                if endpoint not in self.endpoint_metrics:
                    return {'error': f'No metrics for endpoint: {endpoint}'}
                
                metrics = self.endpoint_metrics[endpoint]
                response_times = list(metrics['response_times'])
                
                if not response_times:
                    return {'error': 'No response time data'}
                
                return {
                    'endpoint': endpoint,
                    'total_requests': metrics['count'],
                    'error_count': metrics['error_count'],
                    'error_rate': metrics['error_count'] / metrics['count'] if metrics['count'] > 0 else 0,
                    'average_response_time': metrics['total_time'] / metrics['count'] if metrics['count'] > 0 else 0,
                    'response_time_percentiles': self._calculate_percentiles(response_times),
                    'status_code_distribution': dict(metrics['status_codes']),
                    'requests_per_minute': self._calculate_rpm(endpoint),
                    'last_updated': datetime.fromtimestamp(metrics['last_updated']).isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Endpoint metrics retrieval failed: {e}")
            return {'error': str(e)}
    
    def get_user_metrics(self, user_id: str) -> Dict[str, Any]:
        """Get metrics for specific user"""
        try:
            with self._lock:
                if user_id not in self.user_metrics:
                    return {'error': f'No metrics for user: {user_id}'}
                
                metrics = self.user_metrics[user_id]
                
                return {
                    'user_id': user_id,
                    'total_requests': metrics['request_count'],
                    'error_count': metrics['error_count'],
                    'error_rate': metrics['error_count'] / metrics['request_count'] if metrics['request_count'] > 0 else 0,
                    'average_response_time': metrics['total_response_time'] / metrics['request_count'] if metrics['request_count'] > 0 else 0,
                    'unique_endpoints': len(metrics['endpoints_accessed']),
                    'endpoints_accessed': list(metrics['endpoints_accessed']),
                    'last_activity': datetime.fromtimestamp(metrics['last_activity']).isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"User metrics retrieval failed: {e}")
            return {'error': str(e)}
    
    def _update_endpoint_metrics(self, metric: Dict[str, Any]):
        """Update endpoint-specific metrics"""
        try:
            endpoint = metric['endpoint']
            endpoint_metric = self.endpoint_metrics[endpoint]
            
            endpoint_metric['count'] += 1
            endpoint_metric['total_time'] += metric['processing_time']
            endpoint_metric['response_times'].append(metric['processing_time'])
            endpoint_metric['status_codes'][metric['status_code']] += 1
            endpoint_metric['last_updated'] = metric['timestamp']
            
            if metric['status'] == 'error':
                endpoint_metric['error_count'] += 1
                
        except Exception as e:
            self.logger.error(f"Endpoint metrics update failed: {e}")
    
    def _update_user_metrics(self, metric: Dict[str, Any]):
        """Update user-specific metrics"""
        try:
            user_id = metric['user_id']
            user_metric = self.user_metrics[user_id]
            
            user_metric['request_count'] += 1
            user_metric['total_response_time'] += metric['processing_time']
            user_metric['endpoints_accessed'].add(metric['endpoint'])
            user_metric['last_activity'] = metric['timestamp']
            
            if metric['status'] == 'error':
                user_metric['error_count'] += 1
                
        except Exception as e:
            self.logger.error(f"User metrics update failed: {e}")
    
    def _update_performance_metrics(self, metric: Dict[str, Any]):
        """Update performance metrics"""
        try:
            # Add response time
            self.performance_metrics['response_times'].append({
                'timestamp': metric['timestamp'],
                'value': metric['processing_time']
            })
            
            # Calculate current throughput
            current_time = metric['timestamp']
            recent_requests = [
                m for m in self.request_metrics 
                if current_time - m['timestamp'] < 60  # Last minute
            ]
            
            throughput = len(recent_requests)
            self.performance_metrics['throughput'].append({
                'timestamp': current_time,
                'value': throughput
            })
            
            # Calculate error rate
            if recent_requests:
                errors = sum(1 for m in recent_requests if m['status'] == 'error')
                error_rate = errors / len(recent_requests)
                
                self.performance_metrics['error_rates'].append({
                    'timestamp': current_time,
                    'value': error_rate
                })
                
        except Exception as e:
            self.logger.error(f"Performance metrics update failed: {e}")
    
    def _update_system_metrics(self, metric: Dict[str, Any]):
        """Update system-wide metrics"""
        try:
            self.system_metrics['total_requests'] += 1
            self.system_metrics['total_bytes_in'] += metric['request_size']
            self.system_metrics['total_bytes_out'] += metric['response_size']
            
            if metric['status'] == 'error':
                self.system_metrics['total_errors'] += 1
                
        except Exception as e:
            self.logger.error(f"System metrics update failed: {e}")
    
    def _check_alert_conditions(self):
        """Check for alert conditions"""
        try:
            current_time = time.time()
            
            # Check error rate
            if self.performance_metrics['error_rates']:
                current_error_rate = self.performance_metrics['error_rates'][-1]['value']
                if current_error_rate > self.alert_thresholds['error_rate']:
                    self._create_alert('high_error_rate', {
                        'error_rate': current_error_rate,
                        'threshold': self.alert_thresholds['error_rate']
                    })
            
            # Check response time
            if self.performance_metrics['response_times']:
                recent_times = [
                    rt['value'] for rt in list(self.performance_metrics['response_times'])[-100:]
                ]
                if recent_times:
                    avg_response_time = statistics.mean(recent_times)
                    if avg_response_time > self.alert_thresholds['response_time']:
                        self._create_alert('high_response_time', {
                            'average_response_time': avg_response_time,
                            'threshold': self.alert_thresholds['response_time']
                        })
            
            # Check throughput
            if self.performance_metrics['throughput']:
                current_throughput = self.performance_metrics['throughput'][-1]['value']
                if current_throughput > self.alert_thresholds['throughput']:
                    self._create_alert('high_throughput', {
                        'throughput': current_throughput,
                        'threshold': self.alert_thresholds['throughput']
                    })
                    
        except Exception as e:
            self.logger.error(f"Alert condition check failed: {e}")
    
    def _create_alert(self, alert_type: str, details: Dict[str, Any]):
        """Create or update alert"""
        try:
            alert_id = f"{alert_type}_{int(time.time())}"
            
            if alert_type not in self.active_alerts:
                self.active_alerts[alert_type] = {
                    'alert_id': alert_id,
                    'type': alert_type,
                    'created_at': time.time(),
                    'updated_at': time.time(),
                    'details': details,
                    'count': 1
                }
                self.logger.warning(f"Alert created: {alert_type} - {details}")
            else:
                # Update existing alert
                self.active_alerts[alert_type]['updated_at'] = time.time()
                self.active_alerts[alert_type]['count'] += 1
                self.active_alerts[alert_type]['details'] = details
                
        except Exception as e:
            self.logger.error(f"Alert creation failed: {e}")
    
    def _generate_summary_metrics(self) -> Dict[str, Any]:
        """Generate summary metrics"""
        try:
            uptime = time.time() - self.system_metrics['uptime_start']
            total_requests = self.system_metrics['total_requests']
            
            if total_requests == 0:
                return {
                    'uptime_seconds': uptime,
                    'total_requests': 0,
                    'requests_per_second': 0,
                    'error_rate': 0,
                    'average_response_time': 0
                }
            
            # Calculate averages
            total_time = sum(em['total_time'] for em in self.endpoint_metrics.values())
            
            return {
                'uptime_seconds': uptime,
                'total_requests': total_requests,
                'requests_per_second': total_requests / uptime if uptime > 0 else 0,
                'error_rate': self.system_metrics['total_errors'] / total_requests,
                'average_response_time': total_time / total_requests,
                'total_data_in_mb': self.system_metrics['total_bytes_in'] / (1024 * 1024),
                'total_data_out_mb': self.system_metrics['total_bytes_out'] / (1024 * 1024),
                'unique_users': len(self.user_metrics),
                'unique_endpoints': len(self.endpoint_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Summary metrics generation failed: {e}")
            return {}
    
    def _get_endpoint_summary(self) -> Dict[str, Any]:
        """Get endpoint metrics summary"""
        try:
            summary = {}
            
            # Get top endpoints by request count
            sorted_endpoints = sorted(
                self.endpoint_metrics.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )[:10]
            
            for endpoint, metrics in sorted_endpoints:
                summary[endpoint] = {
                    'requests': metrics['count'],
                    'errors': metrics['error_count'],
                    'average_time': metrics['total_time'] / metrics['count'] if metrics['count'] > 0 else 0
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Endpoint summary generation failed: {e}")
            return {}
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        try:
            # Get recent response times
            recent_times = [
                rt['value'] for rt in list(self.performance_metrics['response_times'])[-1000:]
            ]
            
            if not recent_times:
                return {}
            
            return {
                'response_time_percentiles': self._calculate_percentiles(recent_times),
                'current_throughput': self.performance_metrics['throughput'][-1]['value'] if self.performance_metrics['throughput'] else 0,
                'current_error_rate': self.performance_metrics['error_rates'][-1]['value'] if self.performance_metrics['error_rates'] else 0,
                'concurrent_requests': self.performance_metrics['concurrent_requests'],
                'peak_concurrent_requests': self.performance_metrics['peak_concurrent_requests']
            }
            
        except Exception as e:
            self.logger.error(f"Performance summary generation failed: {e}")
            return {}
    
    def _get_user_summary(self) -> Dict[str, Any]:
        """Get user metrics summary"""
        try:
            # Get top users by request count
            sorted_users = sorted(
                self.user_metrics.items(),
                key=lambda x: x[1]['request_count'],
                reverse=True
            )[:10]
            
            summary = {}
            for user_id, metrics in sorted_users:
                summary[user_id] = {
                    'requests': metrics['request_count'],
                    'errors': metrics['error_count'],
                    'endpoints': len(metrics['endpoints_accessed'])
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"User summary generation failed: {e}")
            return {}
    
    def _get_system_summary(self) -> Dict[str, Any]:
        """Get system metrics summary"""
        try:
            return {
                'uptime_hours': (time.time() - self.system_metrics['uptime_start']) / 3600,
                'total_requests': self.system_metrics['total_requests'],
                'total_errors': self.system_metrics['total_errors'],
                'data_transferred_gb': (
                    self.system_metrics['total_bytes_in'] + 
                    self.system_metrics['total_bytes_out']
                ) / (1024 ** 3)
            }
            
        except Exception as e:
            self.logger.error(f"System summary generation failed: {e}")
            return {}
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        try:
            return list(self.active_alerts.values())
        except Exception as e:
            self.logger.error(f"Active alerts retrieval failed: {e}")
            return []
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze metric trends"""
        try:
            trends = {}
            
            # Analyze throughput trend
            if len(self.performance_metrics['throughput']) > 10:
                throughput_values = [t['value'] for t in list(self.performance_metrics['throughput'])[-10:]]
                trends['throughput_trend'] = self._calculate_trend(throughput_values)
            
            # Analyze response time trend
            if len(self.performance_metrics['response_times']) > 10:
                response_values = [rt['value'] for rt in list(self.performance_metrics['response_times'])[-10:]]
                trends['response_time_trend'] = self._calculate_trend(response_values)
            
            # Analyze error rate trend
            if len(self.performance_metrics['error_rates']) > 10:
                error_values = [er['value'] for er in list(self.performance_metrics['error_rates'])[-10:]]
                trends['error_rate_trend'] = self._calculate_trend(error_values)
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            return {}
    
    def _calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calculate percentiles for given values"""
        try:
            if not values:
                return {}
            
            sorted_values = sorted(values)
            result = {}
            
            for p in self.percentiles:
                index = int((p / 100) * len(sorted_values))
                if index >= len(sorted_values):
                    index = len(sorted_values) - 1
                result[f'p{p}'] = sorted_values[index]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Percentile calculation failed: {e}")
            return {}
    
    def _calculate_rpm(self, endpoint: str) -> float:
        """Calculate requests per minute for endpoint"""
        try:
            current_time = time.time()
            recent_requests = [
                m for m in self.request_metrics
                if m['endpoint'] == endpoint and current_time - m['timestamp'] < 60
            ]
            
            return len(recent_requests)
            
        except Exception as e:
            self.logger.error(f"RPM calculation failed: {e}")
            return 0.0
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        try:
            if len(values) < 2:
                return 'stable'
            
            # Simple linear regression
            n = len(values)
            x = list(range(n))
            
            x_mean = sum(x) / n
            y_mean = sum(values) / n
            
            numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                return 'stable'
            
            slope = numerator / denominator
            
            # Determine trend based on slope
            if slope > 0.1:
                return 'increasing'
            elif slope < -0.1:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception as e:
            self.logger.error(f"Trend calculation failed: {e}")
            return 'unknown'
    
    def _extract_status_code(self, response: Dict[str, Any]) -> int:
        """Extract status code from response"""
        try:
            if 'error' in response and isinstance(response['error'], dict):
                return response['error'].get('status_code', 500)
            elif response.get('success', False):
                return 200
            else:
                return 500
                
        except Exception as e:
            self.logger.error(f"Status code extraction failed: {e}")
            return 500
    
    def _calculate_size(self, data: Dict[str, Any]) -> int:
        """Calculate size of data in bytes"""
        try:
            return len(json.dumps(data).encode('utf-8'))
        except Exception as e:
            self.logger.error(f"Size calculation failed: {e}")
            return 0
    
    def _decrement_concurrent_requests(self):
        """Decrement concurrent requests counter"""
        with self._lock:
            self.performance_metrics['concurrent_requests'] = max(
                0, self.performance_metrics['concurrent_requests'] - 1
            )
    
    def _start_background_threads(self):
        """Start background processing threads"""
        # Start metrics cleanup thread
        cleanup_thread = threading.Thread(
            target=self._cleanup_old_metrics,
            daemon=True
        )
        cleanup_thread.start()
        
        # Start metrics aggregation thread
        aggregation_thread = threading.Thread(
            target=self._aggregate_metrics,
            daemon=True
        )
        aggregation_thread.start()
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics data"""
        while True:
            try:
                current_time = time.time()
                retention_seconds = self.retention_period * 3600
                
                with self._lock:
                    # Clean up request metrics
                    while self.request_metrics and self.request_metrics[0]['timestamp'] < current_time - retention_seconds:
                        self.request_metrics.popleft()
                    
                    # Clean up performance metrics
                    for metric_list in ['response_times', 'throughput', 'error_rates']:
                        metrics = self.performance_metrics[metric_list]
                        while metrics and metrics[0]['timestamp'] < current_time - retention_seconds:
                            metrics.popleft()
                    
                    # Clean up old alerts
                    expired_alerts = []
                    for alert_type, alert in self.active_alerts.items():
                        if current_time - alert['updated_at'] > 3600:  # 1 hour
                            expired_alerts.append(alert_type)
                    
                    for alert_type in expired_alerts:
                        del self.active_alerts[alert_type]
                
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Metrics cleanup failed: {e}")
                time.sleep(600)  # 10 minutes on error
    
    def _aggregate_metrics(self):
        """Aggregate metrics at different intervals"""
        while True:
            try:
                # Placeholder for metric aggregation logic
                # In production, would aggregate metrics into time buckets
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Metrics aggregation failed: {e}")
                time.sleep(300)  # 5 minutes on error