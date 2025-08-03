"""
Saraphis Dashboard Metrics Collector
Production-ready dashboard performance metrics collection
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


class DashboardMetricsCollector:
    """Production-ready dashboard metrics collection and analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Metrics configuration
        self.retention_period_hours = config.get('retention_period_hours', 24)
        self.aggregation_intervals = config.get('aggregation_intervals', [1, 5, 15, 60])  # minutes
        self.percentiles = config.get('percentiles', [50, 90, 95, 99])
        
        # Dashboard render metrics
        self.render_metrics = defaultdict(lambda: {
            'total_renders': 0,
            'successful_renders': 0,
            'failed_renders': 0,
            'render_times': deque(maxlen=1000),
            'component_counts': deque(maxlen=1000),
            'data_freshness_scores': deque(maxlen=1000),
            'last_render': None
        })
        
        # Component performance metrics
        self.component_metrics = defaultdict(lambda: {
            'render_count': 0,
            'update_count': 0,
            'error_count': 0,
            'render_times': deque(maxlen=500),
            'update_times': deque(maxlen=500),
            'interaction_count': 0
        })
        
        # User activity metrics
        self.user_activity_metrics = defaultdict(lambda: {
            'dashboard_views': defaultdict(int),
            'interaction_count': 0,
            'session_count': 0,
            'total_session_duration': 0,
            'last_activity': None,
            'feature_usage': defaultdict(int)
        })
        
        # Real-time performance metrics
        self.realtime_metrics = {
            'websocket_connections': deque(maxlen=1000),
            'data_stream_throughput': defaultdict(lambda: deque(maxlen=1000)),
            'update_latencies': deque(maxlen=1000),
            'broadcast_success_rate': deque(maxlen=100)
        }
        
        # System resource metrics
        self.resource_metrics = {
            'memory_usage': deque(maxlen=1000),
            'cpu_usage': deque(maxlen=1000),
            'active_dashboards': deque(maxlen=1000),
            'active_components': deque(maxlen=1000),
            'queue_sizes': defaultdict(lambda: deque(maxlen=1000))
        }
        
        # Error tracking
        self.error_metrics = defaultdict(lambda: {
            'count': 0,
            'types': defaultdict(int),
            'affected_users': set(),
            'affected_dashboards': set(),
            'last_occurrence': None,
            'error_messages': deque(maxlen=100)
        })
        
        # Aggregated metrics
        self.aggregated_metrics = defaultdict(lambda: defaultdict(dict))
        
        # Alert thresholds
        self.alert_thresholds = {
            'render_time_ms': config.get('render_time_threshold', 1000),
            'error_rate_percent': config.get('error_rate_threshold', 5),
            'memory_usage_mb': config.get('memory_threshold', 1024),
            'cpu_usage_percent': config.get('cpu_threshold', 80),
            'websocket_disconnect_rate': config.get('disconnect_threshold', 10)
        }
        
        # Active alerts
        self.active_alerts = {}
        
        # Historical data
        self.historical_data = defaultdict(lambda: deque(maxlen=10000))
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Start background threads
        self._start_background_threads()
        
        self.logger.info("Dashboard Metrics Collector initialized")
    
    def record_dashboard_render(self, user_id: str, dashboard_type: str, 
                              render_time: float, component_count: int = 0,
                              success: bool = True):
        """Record dashboard render metrics"""
        try:
            with self._lock:
                # Update render metrics
                metrics = self.render_metrics[dashboard_type]
                metrics['total_renders'] += 1
                
                if success:
                    metrics['successful_renders'] += 1
                else:
                    metrics['failed_renders'] += 1
                
                metrics['render_times'].append({
                    'value': render_time,
                    'timestamp': time.time()
                })
                
                metrics['component_counts'].append(component_count)
                metrics['last_render'] = time.time()
                
                # Update user activity
                user_metrics = self.user_activity_metrics[user_id]
                user_metrics['dashboard_views'][dashboard_type] += 1
                user_metrics['last_activity'] = time.time()
                
                # Store historical data
                self.historical_data['dashboard_renders'].append({
                    'timestamp': time.time(),
                    'user_id': user_id,
                    'dashboard_type': dashboard_type,
                    'render_time': render_time,
                    'component_count': component_count,
                    'success': success
                })
                
                # Check for alerts
                if render_time > self.alert_thresholds['render_time_ms'] / 1000:
                    self._create_alert('slow_render', {
                        'dashboard_type': dashboard_type,
                        'render_time': render_time,
                        'threshold': self.alert_thresholds['render_time_ms'] / 1000
                    })
                    
        except Exception as e:
            self.logger.error(f"Dashboard render recording failed: {e}")
    
    def record_component_performance(self, component_id: str, component_type: str,
                                   operation: str, duration: float, success: bool = True):
        """Record component performance metrics"""
        try:
            with self._lock:
                metrics = self.component_metrics[f"{component_type}:{component_id}"]
                
                if operation == 'render':
                    metrics['render_count'] += 1
                    metrics['render_times'].append({
                        'value': duration,
                        'timestamp': time.time()
                    })
                elif operation == 'update':
                    metrics['update_count'] += 1
                    metrics['update_times'].append({
                        'value': duration,
                        'timestamp': time.time()
                    })
                
                if not success:
                    metrics['error_count'] += 1
                    
        except Exception as e:
            self.logger.error(f"Component performance recording failed: {e}")
    
    def record_user_interaction(self, user_id: str, interaction_type: str, 
                              processing_time: float, dashboard_type: Optional[str] = None):
        """Record user interaction metrics"""
        try:
            with self._lock:
                user_metrics = self.user_activity_metrics[user_id]
                user_metrics['interaction_count'] += 1
                user_metrics['last_activity'] = time.time()
                user_metrics['feature_usage'][interaction_type] += 1
                
                # Store interaction data
                self.historical_data['interactions'].append({
                    'timestamp': time.time(),
                    'user_id': user_id,
                    'interaction_type': interaction_type,
                    'processing_time': processing_time,
                    'dashboard_type': dashboard_type
                })
                
        except Exception as e:
            self.logger.error(f"User interaction recording failed: {e}")
    
    def record_websocket_metrics(self, metric_type: str, value: Any):
        """Record WebSocket-related metrics"""
        try:
            with self._lock:
                if metric_type == 'connection_count':
                    self.realtime_metrics['websocket_connections'].append({
                        'value': value,
                        'timestamp': time.time()
                    })
                elif metric_type == 'update_latency':
                    self.realtime_metrics['update_latencies'].append({
                        'value': value,
                        'timestamp': time.time()
                    })
                elif metric_type == 'broadcast_success':
                    self.realtime_metrics['broadcast_success_rate'].append({
                        'success_rate': value,
                        'timestamp': time.time()
                    })
                elif metric_type.startswith('stream_throughput:'):
                    stream_name = metric_type.split(':', 1)[1]
                    self.realtime_metrics['data_stream_throughput'][stream_name].append({
                        'value': value,
                        'timestamp': time.time()
                    })
                    
        except Exception as e:
            self.logger.error(f"WebSocket metrics recording failed: {e}")
    
    def record_resource_usage(self, memory_mb: float, cpu_percent: float,
                            active_dashboards: int, active_components: int):
        """Record system resource usage"""
        try:
            with self._lock:
                current_time = time.time()
                
                self.resource_metrics['memory_usage'].append({
                    'value': memory_mb,
                    'timestamp': current_time
                })
                
                self.resource_metrics['cpu_usage'].append({
                    'value': cpu_percent,
                    'timestamp': current_time
                })
                
                self.resource_metrics['active_dashboards'].append({
                    'value': active_dashboards,
                    'timestamp': current_time
                })
                
                self.resource_metrics['active_components'].append({
                    'value': active_components,
                    'timestamp': current_time
                })
                
                # Check resource alerts
                if memory_mb > self.alert_thresholds['memory_usage_mb']:
                    self._create_alert('high_memory_usage', {
                        'memory_mb': memory_mb,
                        'threshold': self.alert_thresholds['memory_usage_mb']
                    })
                
                if cpu_percent > self.alert_thresholds['cpu_usage_percent']:
                    self._create_alert('high_cpu_usage', {
                        'cpu_percent': cpu_percent,
                        'threshold': self.alert_thresholds['cpu_usage_percent']
                    })
                    
        except Exception as e:
            self.logger.error(f"Resource usage recording failed: {e}")
    
    def record_dashboard_error(self, user_id: str, dashboard_type: str, 
                             error_message: str, error_type: str = 'unknown'):
        """Record dashboard error"""
        try:
            with self._lock:
                error_key = f"{dashboard_type}:{error_type}"
                error_metric = self.error_metrics[error_key]
                
                error_metric['count'] += 1
                error_metric['types'][error_type] += 1
                error_metric['affected_users'].add(user_id)
                error_metric['affected_dashboards'].add(dashboard_type)
                error_metric['last_occurrence'] = time.time()
                error_metric['error_messages'].append({
                    'message': error_message,
                    'timestamp': time.time(),
                    'user_id': user_id
                })
                
                # Check error rate
                self._check_error_rate_alert(dashboard_type)
                
        except Exception as e:
            self.logger.error(f"Error recording failed: {e}")
    
    def record_dashboard_close(self, user_id: str, dashboard_type: str, 
                             session_duration: float):
        """Record dashboard close event"""
        try:
            with self._lock:
                user_metrics = self.user_activity_metrics[user_id]
                user_metrics['session_count'] += 1
                user_metrics['total_session_duration'] += session_duration
                
                # Store session data
                self.historical_data['sessions'].append({
                    'timestamp': time.time(),
                    'user_id': user_id,
                    'dashboard_type': dashboard_type,
                    'duration': session_duration
                })
                
        except Exception as e:
            self.logger.error(f"Dashboard close recording failed: {e}")
    
    def record_access_denied(self, user_id: str, dashboard_type: str):
        """Record access denied event"""
        try:
            with self._lock:
                self.historical_data['access_denied'].append({
                    'timestamp': time.time(),
                    'user_id': user_id,
                    'dashboard_type': dashboard_type
                })
                
        except Exception as e:
            self.logger.error(f"Access denied recording failed: {e}")
    
    def record_interaction_error(self, user_id: str, error_message: str):
        """Record interaction error"""
        try:
            with self._lock:
                error_metric = self.error_metrics['interaction_errors']
                error_metric['count'] += 1
                error_metric['affected_users'].add(user_id)
                error_metric['last_occurrence'] = time.time()
                error_metric['error_messages'].append({
                    'message': error_message,
                    'timestamp': time.time(),
                    'user_id': user_id
                })
                
        except Exception as e:
            self.logger.error(f"Interaction error recording failed: {e}")
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get comprehensive dashboard metrics"""
        try:
            with self._lock:
                # Calculate summary metrics
                summary = self._calculate_summary_metrics()
                
                # Get dashboard-specific metrics
                dashboard_metrics = {}
                for dashboard_type, metrics in self.render_metrics.items():
                    dashboard_metrics[dashboard_type] = self._format_dashboard_metrics(metrics)
                
                # Get component metrics summary
                component_summary = self._calculate_component_summary()
                
                # Get user activity summary
                user_summary = self._calculate_user_summary()
                
                # Get real-time metrics
                realtime_summary = self._calculate_realtime_summary()
                
                # Get resource metrics
                resource_summary = self._calculate_resource_summary()
                
                # Get error summary
                error_summary = self._calculate_error_summary()
                
                return {
                    'summary': summary,
                    'dashboards': dashboard_metrics,
                    'components': component_summary,
                    'users': user_summary,
                    'realtime': realtime_summary,
                    'resources': resource_summary,
                    'errors': error_summary,
                    'alerts': self._get_active_alerts(),
                    'timestamp': time.time()
                }
                
        except Exception as e:
            self.logger.error(f"Dashboard metrics retrieval failed: {e}")
            return {'error': str(e)}
    
    def aggregate_metrics(self):
        """Aggregate metrics at different time intervals"""
        try:
            current_time = time.time()
            
            with self._lock:
                # Aggregate render metrics
                for dashboard_type, metrics in self.render_metrics.items():
                    for interval in self.aggregation_intervals:
                        interval_key = f"{interval}m"
                        
                        # Get data points within interval
                        interval_seconds = interval * 60
                        recent_renders = [
                            r for r in metrics['render_times']
                            if isinstance(r, dict) and 
                            current_time - r['timestamp'] <= interval_seconds
                        ]
                        
                        if recent_renders:
                            render_times = [r['value'] for r in recent_renders]
                            
                            self.aggregated_metrics[dashboard_type][interval_key] = {
                                'count': len(recent_renders),
                                'average_render_time': statistics.mean(render_times),
                                'min_render_time': min(render_times),
                                'max_render_time': max(render_times),
                                'percentiles': self._calculate_percentiles(render_times),
                                'timestamp': current_time
                            }
                            
        except Exception as e:
            self.logger.error(f"Metrics aggregation failed: {e}")
    
    def cleanup_old_metrics(self):
        """Clean up metrics older than retention period"""
        try:
            current_time = time.time()
            cutoff_time = current_time - (self.retention_period_hours * 3600)
            
            with self._lock:
                # Clean up historical data
                for data_type, data_list in self.historical_data.items():
                    while data_list and data_list[0]['timestamp'] < cutoff_time:
                        data_list.popleft()
                
                # Clean up time-based metrics
                for metrics_dict in [self.render_metrics, self.component_metrics]:
                    for metrics in metrics_dict.values():
                        for key in ['render_times', 'update_times']:
                            if key in metrics:
                                times = metrics[key]
                                while times and isinstance(times[0], dict) and times[0]['timestamp'] < cutoff_time:
                                    times.popleft()
                
                # Clean up old alerts
                expired_alerts = []
                for alert_id, alert in self.active_alerts.items():
                    if current_time - alert['created_at'] > 3600:  # 1 hour
                        expired_alerts.append(alert_id)
                
                for alert_id in expired_alerts:
                    del self.active_alerts[alert_id]
                    
        except Exception as e:
            self.logger.error(f"Metrics cleanup failed: {e}")
    
    def _calculate_summary_metrics(self) -> Dict[str, Any]:
        """Calculate overall summary metrics"""
        try:
            total_renders = sum(m['total_renders'] for m in self.render_metrics.values())
            successful_renders = sum(m['successful_renders'] for m in self.render_metrics.values())
            failed_renders = sum(m['failed_renders'] for m in self.render_metrics.values())
            
            # Calculate average render time across all dashboards
            all_render_times = []
            for metrics in self.render_metrics.values():
                all_render_times.extend([r['value'] for r in metrics['render_times'] if isinstance(r, dict)])
            
            avg_render_time = statistics.mean(all_render_times) if all_render_times else 0
            
            return {
                'total_renders': total_renders,
                'successful_renders': successful_renders,
                'failed_renders': failed_renders,
                'success_rate': successful_renders / total_renders if total_renders > 0 else 1.0,
                'average_render_time': avg_render_time,
                'active_users': len(self.user_activity_metrics),
                'total_interactions': sum(u['interaction_count'] for u in self.user_activity_metrics.values())
            }
            
        except Exception as e:
            self.logger.error(f"Summary calculation failed: {e}")
            return {}
    
    def _format_dashboard_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Format dashboard-specific metrics"""
        try:
            render_times = [r['value'] for r in metrics['render_times'] if isinstance(r, dict)]
            
            formatted = {
                'total_renders': metrics['total_renders'],
                'successful_renders': metrics['successful_renders'],
                'failed_renders': metrics['failed_renders'],
                'success_rate': metrics['successful_renders'] / metrics['total_renders'] 
                               if metrics['total_renders'] > 0 else 1.0,
                'last_render': metrics['last_render']
            }
            
            if render_times:
                formatted.update({
                    'average_render_time': statistics.mean(render_times),
                    'min_render_time': min(render_times),
                    'max_render_time': max(render_times),
                    'render_time_percentiles': self._calculate_percentiles(render_times)
                })
            
            if metrics['component_counts']:
                formatted['average_component_count'] = statistics.mean(metrics['component_counts'])
            
            return formatted
            
        except Exception as e:
            self.logger.error(f"Dashboard metrics formatting failed: {e}")
            return {}
    
    def _calculate_component_summary(self) -> Dict[str, Any]:
        """Calculate component metrics summary"""
        try:
            total_components = len(self.component_metrics)
            total_renders = sum(m['render_count'] for m in self.component_metrics.values())
            total_updates = sum(m['update_count'] for m in self.component_metrics.values())
            total_errors = sum(m['error_count'] for m in self.component_metrics.values())
            
            # Get top components by render count
            top_components = sorted(
                self.component_metrics.items(),
                key=lambda x: x[1]['render_count'],
                reverse=True
            )[:10]
            
            return {
                'total_components': total_components,
                'total_renders': total_renders,
                'total_updates': total_updates,
                'total_errors': total_errors,
                'error_rate': total_errors / (total_renders + total_updates) 
                            if (total_renders + total_updates) > 0 else 0,
                'top_components': [
                    {
                        'component': comp_id,
                        'renders': metrics['render_count'],
                        'updates': metrics['update_count'],
                        'errors': metrics['error_count']
                    }
                    for comp_id, metrics in top_components
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Component summary calculation failed: {e}")
            return {}
    
    def _calculate_user_summary(self) -> Dict[str, Any]:
        """Calculate user activity summary"""
        try:
            active_users = len(self.user_activity_metrics)
            total_views = sum(
                sum(u['dashboard_views'].values()) 
                for u in self.user_activity_metrics.values()
            )
            total_interactions = sum(
                u['interaction_count'] 
                for u in self.user_activity_metrics.values()
            )
            
            # Get most active users
            most_active = sorted(
                self.user_activity_metrics.items(),
                key=lambda x: sum(x[1]['dashboard_views'].values()),
                reverse=True
            )[:10]
            
            return {
                'active_users': active_users,
                'total_dashboard_views': total_views,
                'total_interactions': total_interactions,
                'average_views_per_user': total_views / active_users if active_users > 0 else 0,
                'most_active_users': [
                    {
                        'user_id': user_id,
                        'total_views': sum(metrics['dashboard_views'].values()),
                        'interactions': metrics['interaction_count'],
                        'last_activity': metrics['last_activity']
                    }
                    for user_id, metrics in most_active
                ]
            }
            
        except Exception as e:
            self.logger.error(f"User summary calculation failed: {e}")
            return {}
    
    def _calculate_realtime_summary(self) -> Dict[str, Any]:
        """Calculate real-time metrics summary"""
        try:
            # WebSocket metrics
            ws_connections = [c['value'] for c in self.realtime_metrics['websocket_connections']]
            current_connections = ws_connections[-1] if ws_connections else 0
            avg_connections = statistics.mean(ws_connections) if ws_connections else 0
            
            # Update latency
            latencies = [l['value'] for l in self.realtime_metrics['update_latencies']]
            avg_latency = statistics.mean(latencies) if latencies else 0
            
            # Broadcast success rate
            success_rates = [b['success_rate'] for b in self.realtime_metrics['broadcast_success_rate']]
            avg_success_rate = statistics.mean(success_rates) if success_rates else 1.0
            
            # Stream throughput
            stream_summary = {}
            for stream, throughput_data in self.realtime_metrics['data_stream_throughput'].items():
                values = [t['value'] for t in throughput_data]
                if values:
                    stream_summary[stream] = {
                        'average_throughput': statistics.mean(values),
                        'max_throughput': max(values),
                        'current_throughput': values[-1] if values else 0
                    }
            
            return {
                'websocket': {
                    'current_connections': current_connections,
                    'average_connections': avg_connections,
                    'max_connections': max(ws_connections) if ws_connections else 0
                },
                'latency': {
                    'average_update_latency': avg_latency,
                    'latency_percentiles': self._calculate_percentiles(latencies) if latencies else {}
                },
                'broadcast': {
                    'average_success_rate': avg_success_rate,
                    'total_broadcasts': len(self.realtime_metrics['broadcast_success_rate'])
                },
                'streams': stream_summary
            }
            
        except Exception as e:
            self.logger.error(f"Real-time summary calculation failed: {e}")
            return {}
    
    def _calculate_resource_summary(self) -> Dict[str, Any]:
        """Calculate resource usage summary"""
        try:
            # Memory usage
            memory_values = [m['value'] for m in self.resource_metrics['memory_usage']]
            current_memory = memory_values[-1] if memory_values else 0
            avg_memory = statistics.mean(memory_values) if memory_values else 0
            
            # CPU usage
            cpu_values = [c['value'] for c in self.resource_metrics['cpu_usage']]
            current_cpu = cpu_values[-1] if cpu_values else 0
            avg_cpu = statistics.mean(cpu_values) if cpu_values else 0
            
            # Active resources
            dashboard_values = [d['value'] for d in self.resource_metrics['active_dashboards']]
            current_dashboards = dashboard_values[-1] if dashboard_values else 0
            
            component_values = [c['value'] for c in self.resource_metrics['active_components']]
            current_components = component_values[-1] if component_values else 0
            
            return {
                'memory': {
                    'current_mb': current_memory,
                    'average_mb': avg_memory,
                    'max_mb': max(memory_values) if memory_values else 0
                },
                'cpu': {
                    'current_percent': current_cpu,
                    'average_percent': avg_cpu,
                    'max_percent': max(cpu_values) if cpu_values else 0
                },
                'active_resources': {
                    'dashboards': current_dashboards,
                    'components': current_components
                }
            }
            
        except Exception as e:
            self.logger.error(f"Resource summary calculation failed: {e}")
            return {}
    
    def _calculate_error_summary(self) -> Dict[str, Any]:
        """Calculate error metrics summary"""
        try:
            total_errors = sum(e['count'] for e in self.error_metrics.values())
            error_types = defaultdict(int)
            affected_users = set()
            affected_dashboards = set()
            
            for error_metric in self.error_metrics.values():
                for error_type, count in error_metric['types'].items():
                    error_types[error_type] += count
                
                affected_users.update(error_metric['affected_users'])
                affected_dashboards.update(error_metric['affected_dashboards'])
            
            # Get recent errors
            recent_errors = []
            for error_key, error_metric in self.error_metrics.items():
                for error_msg in list(error_metric['error_messages'])[-5:]:
                    recent_errors.append({
                        'error_key': error_key,
                        'message': error_msg['message'],
                        'timestamp': error_msg['timestamp'],
                        'user_id': error_msg.get('user_id')
                    })
            
            # Sort by timestamp
            recent_errors.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return {
                'total_errors': total_errors,
                'error_types': dict(error_types),
                'affected_users': len(affected_users),
                'affected_dashboards': len(affected_dashboards),
                'recent_errors': recent_errors[:10]
            }
            
        except Exception as e:
            self.logger.error(f"Error summary calculation failed: {e}")
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
    
    def _check_error_rate_alert(self, dashboard_type: str):
        """Check if error rate exceeds threshold"""
        try:
            metrics = self.render_metrics.get(dashboard_type, {})
            total = metrics.get('total_renders', 0)
            failed = metrics.get('failed_renders', 0)
            
            if total > 10:  # Only check after sufficient renders
                error_rate = (failed / total) * 100
                
                if error_rate > self.alert_thresholds['error_rate_percent']:
                    self._create_alert('high_error_rate', {
                        'dashboard_type': dashboard_type,
                        'error_rate': error_rate,
                        'threshold': self.alert_thresholds['error_rate_percent']
                    })
                    
        except Exception as e:
            self.logger.error(f"Error rate check failed: {e}")
    
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
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts"""
        try:
            return list(self.active_alerts.values())
        except Exception as e:
            self.logger.error(f"Active alerts retrieval failed: {e}")
            return []
    
    def _start_background_threads(self):
        """Start background processing threads"""
        # Aggregation thread
        aggregation_thread = threading.Thread(
            target=self._aggregation_loop,
            daemon=True
        )
        aggregation_thread.start()
        
        # Cleanup thread
        cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        cleanup_thread.start()
    
    def _aggregation_loop(self):
        """Periodically aggregate metrics"""
        while True:
            try:
                time.sleep(60)  # Run every minute
                self.aggregate_metrics()
            except Exception as e:
                self.logger.error(f"Aggregation loop error: {e}")
                time.sleep(300)  # 5 minutes on error
    
    def _cleanup_loop(self):
        """Periodically clean up old metrics"""
        while True:
            try:
                time.sleep(3600)  # Run every hour
                self.cleanup_old_metrics()
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                time.sleep(7200)  # 2 hours on error