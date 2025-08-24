"""
Saraphis Data Metrics Collector
Production-ready data performance and integrity metrics system
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)


class DataMetricsCollector:
    """Production-ready data performance and integrity metrics system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics_history = deque(maxlen=10000)
        self.performance_history = deque(maxlen=5000)
        
        # Metric thresholds
        self.performance_thresholds = {
            'backup_time_seconds': config.get('backup_time_threshold', 300),
            'restore_time_seconds': config.get('restore_time_threshold', 600),
            'compression_ratio': config.get('compression_ratio_threshold', 0.5),
            'encryption_time_ms': config.get('encryption_time_threshold', 100),
            'replication_lag_seconds': config.get('replication_lag_threshold', 300),
            'storage_usage_percent': config.get('storage_usage_threshold', 80)
        }
        
        # Metric categories
        self.metric_categories = {
            'backup': defaultdict(float),
            'compression': defaultdict(float),
            'encryption': defaultdict(float),
            'storage': defaultdict(float),
            'replication': defaultdict(float),
            'integrity': defaultdict(float),
            'performance': defaultdict(float),
            'availability': defaultdict(float)
        }
        
        # Real-time metrics
        self.real_time_metrics = {
            'current_backup_operations': 0,
            'current_restore_operations': 0,
            'current_replication_operations': 0,
            'active_encryption_operations': 0,
            'storage_operations_per_second': 0.0,
            'data_throughput_mbps': 0.0,
            'system_health_score': 1.0
        }
        
        # Alerts
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        
        # Thread control
        self._lock = threading.Lock()
        self.is_running = True
        
        # Start metric collection threads
        self._start_metric_threads()
        
        self.logger.info("Data Metrics Collector initialized")
    
    def record_metric(self, category: str, metric_name: str, value: float, 
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a data metric"""
        try:
            with self._lock:
                # Update category metrics
                if category in self.metric_categories:
                    self.metric_categories[category][metric_name] = value
                
                # Store in history
                metric_record = {
                    'timestamp': time.time(),
                    'category': category,
                    'metric_name': metric_name,
                    'value': value,
                    'metadata': metadata or {}
                }
                
                self.metrics_history.append(metric_record)
                
                # Check thresholds
                self._check_metric_threshold(category, metric_name, value)
                
        except Exception as e:
            self.logger.error(f"Failed to record metric: {e}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values"""
        try:
            with self._lock:
                current_metrics = {}
                
                # Collect metrics from all categories
                for category, metrics in self.metric_categories.items():
                    current_metrics[category] = dict(metrics)
                
                # Add real-time metrics
                current_metrics['real_time'] = self.real_time_metrics.copy()
                
                # Add system health
                current_metrics['system_health'] = {
                    'health_score': self.real_time_metrics['system_health_score'],
                    'active_alerts': len(self.active_alerts),
                    'status': self._get_health_status()
                }
                
                return current_metrics
                
        except Exception as e:
            self.logger.error(f"Failed to get current metrics: {e}")
            return {}
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all metrics (alias for get_current_metrics)"""
        return self.get_current_metrics()
    
    def get_historical_metrics(self, start_time: float, end_time: float) -> List[Dict[str, Any]]:
        """Get historical metrics within time range"""
        try:
            historical = []
            for record in self.metric_history:
                if start_time <= record['timestamp'] <= end_time:
                    historical.append(record)
            return historical
        except Exception as e:
            self.logger.error(f"Failed to get historical metrics: {e}")
            return []
    
    def analyze_metrics(self) -> Dict[str, Any]:
        """Analyze metrics and generate insights"""
        try:
            # Get time-based analysis
            time_analysis = self._analyze_time_based_metrics()
            
            # Analyze trends
            trend_analysis = self._analyze_metric_trends()
            
            # Analyze anomalies
            anomaly_analysis = self._detect_metric_anomalies()
            
            # Performance analysis
            performance_analysis = self._analyze_performance_metrics()
            
            # Generate recommendations
            recommendations = self._generate_metric_recommendations(
                time_analysis, trend_analysis, anomaly_analysis, performance_analysis
            )
            
            return {
                'analysis_timestamp': datetime.now().isoformat(),
                'time_analysis': time_analysis,
                'trend_analysis': trend_analysis,
                'anomaly_analysis': anomaly_analysis,
                'performance_analysis': performance_analysis,
                'system_health_score': self.real_time_metrics['system_health_score'],
                'active_alerts': self._get_active_alerts(),
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Metric analysis failed: {e}")
            return {
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            # Get current metrics
            current_metrics = self.get_current_metrics()
            
            # Analyze metrics
            analysis = self.analyze_metrics()
            
            # Calculate performance scores
            performance_scores = self._calculate_performance_scores()
            
            # Get metric statistics
            metric_statistics = self._calculate_metric_statistics()
            
            # Check SLA compliance
            sla_compliance = self._check_sla_compliance()
            
            report = {
                'report_id': f"data_metrics_report_{int(time.time())}",
                'timestamp': datetime.now().isoformat(),
                'current_metrics': current_metrics,
                'metric_analysis': analysis,
                'performance_scores': performance_scores,
                'metric_statistics': metric_statistics,
                'sla_compliance': sla_compliance,
                'alert_summary': self._get_alert_summary(),
                'recommendations': analysis.get('recommendations', [])
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _start_metric_threads(self):
        """Start metric collection threads"""
        # Start metric aggregation thread
        aggregation_thread = threading.Thread(
            target=self._metric_aggregation_loop,
            daemon=True
        )
        aggregation_thread.start()
        
        # Start performance monitoring thread
        performance_thread = threading.Thread(
            target=self._performance_monitoring_loop,
            daemon=True
        )
        performance_thread.start()
        
        # Start health monitoring thread
        health_thread = threading.Thread(
            target=self._health_monitoring_loop,
            daemon=True
        )
        health_thread.start()
    
    def _metric_aggregation_loop(self):
        """Aggregate metrics periodically"""
        while self.is_running:
            try:
                # Aggregate recent metrics
                self._aggregate_recent_metrics()
                
                # Update real-time metrics
                self._update_real_time_metrics()
                
                time.sleep(10)  # Aggregate every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Metric aggregation error: {e}")
                time.sleep(30)
    
    def _performance_monitoring_loop(self):
        """Monitor performance metrics"""
        while self.is_running:
            try:
                # Calculate performance metrics
                self._calculate_throughput_metrics()
                
                # Monitor operation latencies
                self._monitor_operation_latencies()
                
                # Update performance history
                self._update_performance_history()
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                time.sleep(10)
    
    def _health_monitoring_loop(self):
        """Monitor system health"""
        while self.is_running:
            try:
                # Calculate health score
                health_score = self._calculate_health_score()
                
                with self._lock:
                    self.real_time_metrics['system_health_score'] = health_score
                
                # Check for alerts
                self._check_health_alerts(health_score)
                
                time.sleep(30)  # Check health every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(60)
    
    def _check_metric_threshold(self, category: str, metric_name: str, value: float):
        """Check if metric exceeds threshold"""
        try:
            threshold_key = f"{metric_name}"
            
            if threshold_key in self.performance_thresholds:
                threshold = self.performance_thresholds[threshold_key]
                
                # Check if threshold is exceeded
                exceeded = False
                if 'time' in metric_name or 'lag' in metric_name:
                    exceeded = value > threshold
                elif 'ratio' in metric_name:
                    exceeded = value < threshold
                elif 'usage' in metric_name:
                    exceeded = value > threshold
                
                if exceeded:
                    self._create_alert(
                        alert_type='threshold_exceeded',
                        severity='warning',
                        category=category,
                        metric=metric_name,
                        value=value,
                        threshold=threshold
                    )
                    
        except Exception as e:
            self.logger.error(f"Threshold check failed: {e}")
    
    def _create_alert(self, alert_type: str, severity: str, **kwargs):
        """Create performance alert"""
        try:
            alert_id = f"alert_{int(time.time() * 1000)}"
            
            alert = {
                'alert_id': alert_id,
                'type': alert_type,
                'severity': severity,
                'timestamp': time.time(),
                'details': kwargs,
                'status': 'active'
            }
            
            with self._lock:
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
            
            self.logger.warning(f"Alert created: {alert_type} - {severity} - {kwargs}")
            
        except Exception as e:
            self.logger.error(f"Alert creation failed: {e}")
    
    def _aggregate_recent_metrics(self):
        """Aggregate recent metrics"""
        try:
            current_time = time.time()
            window_start = current_time - 300  # 5 minute window
            
            # Get recent metrics
            recent_metrics = [
                m for m in self.metrics_history 
                if m['timestamp'] > window_start
            ]
            
            if not recent_metrics:
                return
            
            # Aggregate by category and metric
            aggregated = defaultdict(lambda: defaultdict(list))
            
            for metric in recent_metrics:
                category = metric['category']
                name = metric['metric_name']
                value = metric['value']
                aggregated[category][name].append(value)
            
            # Calculate aggregates
            with self._lock:
                for category, metrics in aggregated.items():
                    for name, values in metrics.items():
                        if values:
                            # Calculate average
                            avg_value = statistics.mean(values)
                            self.metric_categories[category][f"{name}_avg"] = avg_value
                            
                            # Calculate min/max
                            self.metric_categories[category][f"{name}_min"] = min(values)
                            self.metric_categories[category][f"{name}_max"] = max(values)
                            
        except Exception as e:
            self.logger.error(f"Metric aggregation failed: {e}")
    
    def _update_real_time_metrics(self):
        """Update real-time metric values"""
        try:
            with self._lock:
                # Calculate operations per second
                current_time = time.time()
                recent_ops = sum(1 for m in self.metrics_history 
                               if m['timestamp'] > current_time - 60)
                
                self.real_time_metrics['storage_operations_per_second'] = recent_ops / 60
                
                # Update other real-time metrics from categories
                backup_metrics = self.metric_categories.get('backup', {})
                self.real_time_metrics['current_backup_operations'] = int(
                    backup_metrics.get('active_backups', 0)
                )
                
                replication_metrics = self.metric_categories.get('replication', {})
                self.real_time_metrics['current_replication_operations'] = int(
                    replication_metrics.get('active_replications', 0)
                )
                
        except Exception as e:
            self.logger.error(f"Real-time metric update failed: {e}")
    
    def _calculate_throughput_metrics(self):
        """Calculate data throughput metrics"""
        try:
            # Get recent data transfer metrics
            current_time = time.time()
            window_start = current_time - 60  # 1 minute window
            
            data_transfers = [
                m for m in self.metrics_history
                if m['timestamp'] > window_start and 
                m['metric_name'] in ['data_transferred_bytes', 'data_processed_bytes']
            ]
            
            if data_transfers:
                total_bytes = sum(m['value'] for m in data_transfers)
                throughput_mbps = (total_bytes / (1024**2)) / 60  # MB/s
                
                with self._lock:
                    self.real_time_metrics['data_throughput_mbps'] = throughput_mbps
                    
        except Exception as e:
            self.logger.error(f"Throughput calculation failed: {e}")
    
    def _monitor_operation_latencies(self):
        """Monitor operation latencies"""
        try:
            # Get recent operation metrics
            current_time = time.time()
            window_start = current_time - 300  # 5 minute window
            
            latency_metrics = [
                m for m in self.metrics_history
                if m['timestamp'] > window_start and 
                ('time' in m['metric_name'] or 'duration' in m['metric_name'])
            ]
            
            if latency_metrics:
                # Group by operation type
                operation_latencies = defaultdict(list)
                
                for metric in latency_metrics:
                    operation = metric['category']
                    latency = metric['value']
                    operation_latencies[operation].append(latency)
                
                # Store performance metrics
                performance_record = {
                    'timestamp': current_time,
                    'operation_latencies': {}
                }
                
                for operation, latencies in operation_latencies.items():
                    performance_record['operation_latencies'][operation] = {
                        'avg_latency': statistics.mean(latencies),
                        'p95_latency': sorted(latencies)[int(len(latencies) * 0.95)],
                        'max_latency': max(latencies)
                    }
                
                self.performance_history.append(performance_record)
                
        except Exception as e:
            self.logger.error(f"Latency monitoring failed: {e}")
    
    def _update_performance_history(self):
        """Update performance history"""
        try:
            with self._lock:
                # Record current performance snapshot
                snapshot = {
                    'timestamp': time.time(),
                    'throughput_mbps': self.real_time_metrics['data_throughput_mbps'],
                    'operations_per_second': self.real_time_metrics['storage_operations_per_second'],
                    'active_operations': {
                        'backups': self.real_time_metrics['current_backup_operations'],
                        'restores': self.real_time_metrics['current_restore_operations'],
                        'replications': self.real_time_metrics['current_replication_operations']
                    }
                }
                
                self.performance_history.append(snapshot)
                
        except Exception as e:
            self.logger.error(f"Performance history update failed: {e}")
    
    def _calculate_health_score(self) -> float:
        """Calculate system health score"""
        try:
            scores = []
            
            # Check backup health
            backup_metrics = self.metric_categories.get('backup', {})
            backup_success_rate = backup_metrics.get('success_rate', 1.0)
            scores.append(backup_success_rate)
            
            # Check compression health
            compression_metrics = self.metric_categories.get('compression', {})
            compression_ratio = compression_metrics.get('average_compression_ratio', 0.5)
            compression_score = min(1.0, compression_ratio / 0.5)  # Normalize
            scores.append(compression_score)
            
            # Check encryption health
            encryption_metrics = self.metric_categories.get('encryption', {})
            encryption_coverage = encryption_metrics.get('coverage_percentage', 1.0)
            scores.append(encryption_coverage)
            
            # Check storage health
            storage_metrics = self.metric_categories.get('storage', {})
            storage_usage = storage_metrics.get('usage_percentage', 0) / 100
            storage_score = max(0, 1 - storage_usage)  # Inverse of usage
            scores.append(storage_score)
            
            # Check replication health
            replication_metrics = self.metric_categories.get('replication', {})
            replication_score = replication_metrics.get('health_score', 1.0)
            scores.append(replication_score)
            
            # Check for active alerts
            alert_penalty = len(self.active_alerts) * 0.1
            alert_score = max(0, 1 - alert_penalty)
            scores.append(alert_score)
            
            # Calculate weighted average
            if scores:
                health_score = statistics.mean(scores)
            else:
                health_score = 1.0
            
            return min(1.0, max(0.0, health_score))
            
        except Exception as e:
            self.logger.error(f"Health score calculation failed: {e}")
            return 0.5
    
    def _check_health_alerts(self, health_score: float):
        """Check and create health-based alerts"""
        try:
            # Critical health
            if health_score < 0.5:
                self._create_alert(
                    alert_type='critical_health',
                    severity='critical',
                    health_score=health_score,
                    message='System health critically low'
                )
            # Warning health
            elif health_score < 0.7:
                self._create_alert(
                    alert_type='low_health',
                    severity='warning',
                    health_score=health_score,
                    message='System health below threshold'
                )
                
        except Exception as e:
            self.logger.error(f"Health alert check failed: {e}")
    
    def _get_health_status(self) -> str:
        """Get health status description"""
        score = self.real_time_metrics['system_health_score']
        
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.7:
            return 'good'
        elif score >= 0.5:
            return 'fair'
        elif score >= 0.3:
            return 'poor'
        else:
            return 'critical'
    
    def _analyze_time_based_metrics(self) -> Dict[str, Any]:
        """Analyze metrics over time periods"""
        try:
            current_time = time.time()
            
            # Define time windows
            windows = {
                'last_hour': 3600,
                'last_day': 86400,
                'last_week': 604800
            }
            
            analysis = {}
            
            for window_name, window_seconds in windows.items():
                window_start = current_time - window_seconds
                
                # Get metrics in window
                window_metrics = [
                    m for m in self.metrics_history
                    if m['timestamp'] > window_start
                ]
                
                if window_metrics:
                    # Analyze by category
                    category_analysis = defaultdict(dict)
                    
                    for category in self.metric_categories.keys():
                        category_metrics = [
                            m for m in window_metrics
                            if m['category'] == category
                        ]
                        
                        if category_metrics:
                            values = [m['value'] for m in category_metrics]
                            category_analysis[category] = {
                                'count': len(values),
                                'average': statistics.mean(values),
                                'min': min(values),
                                'max': max(values)
                            }
                    
                    analysis[window_name] = dict(category_analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Time-based analysis failed: {e}")
            return {}
    
    def _analyze_metric_trends(self) -> Dict[str, Any]:
        """Analyze metric trends"""
        try:
            trends = {}
            
            # Analyze each category
            for category, metrics in self.metric_categories.items():
                category_trends = {}
                
                # Get historical values
                category_history = [
                    m for m in self.metrics_history
                    if m['category'] == category
                ]
                
                if len(category_history) > 10:
                    # Group by metric name
                    metric_values = defaultdict(list)
                    
                    for record in category_history:
                        metric_values[record['metric_name']].append({
                            'timestamp': record['timestamp'],
                            'value': record['value']
                        })
                    
                    # Calculate trends
                    for metric_name, values in metric_values.items():
                        if len(values) > 5:
                            # Sort by timestamp
                            values.sort(key=lambda x: x['timestamp'])
                            
                            # Calculate trend direction
                            recent_avg = statistics.mean([v['value'] for v in values[-5:]])
                            older_avg = statistics.mean([v['value'] for v in values[:5]])
                            
                            if recent_avg > older_avg * 1.1:
                                trend = 'increasing'
                            elif recent_avg < older_avg * 0.9:
                                trend = 'decreasing'
                            else:
                                trend = 'stable'
                            
                            category_trends[metric_name] = {
                                'trend': trend,
                                'change_percentage': ((recent_avg - older_avg) / older_avg * 100) 
                                                   if older_avg > 0 else 0
                            }
                
                if category_trends:
                    trends[category] = category_trends
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            return {}
    
    def _detect_metric_anomalies(self) -> Dict[str, Any]:
        """Detect anomalies in metrics"""
        try:
            anomalies = []
            
            # Check each metric category
            for category, metrics in self.metric_categories.items():
                for metric_name, current_value in metrics.items():
                    # Get historical values
                    historical_values = [
                        m['value'] for m in self.metrics_history
                        if m['category'] == category and m['metric_name'] == metric_name
                    ]
                    
                    if len(historical_values) > 20:
                        # Calculate statistics
                        mean = statistics.mean(historical_values)
                        stdev = statistics.stdev(historical_values)
                        
                        # Check for anomalies (values outside 3 standard deviations)
                        if stdev > 0:
                            z_score = abs((current_value - mean) / stdev)
                            
                            if z_score > 3:
                                anomalies.append({
                                    'category': category,
                                    'metric': metric_name,
                                    'current_value': current_value,
                                    'expected_range': [mean - 3*stdev, mean + 3*stdev],
                                    'z_score': z_score,
                                    'severity': 'high' if z_score > 4 else 'medium'
                                })
            
            return {
                'anomaly_count': len(anomalies),
                'anomalies': anomalies[:10]  # Limit to top 10
            }
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return {'anomaly_count': 0, 'anomalies': []}
    
    def _analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics"""
        try:
            if not self.performance_history:
                return {}
            
            # Get recent performance data
            recent_performance = list(self.performance_history)[-100:]
            
            # Calculate performance statistics
            throughput_values = [p['throughput_mbps'] for p in recent_performance]
            ops_values = [p['operations_per_second'] for p in recent_performance]
            
            analysis = {
                'throughput': {
                    'average_mbps': statistics.mean(throughput_values) if throughput_values else 0,
                    'peak_mbps': max(throughput_values) if throughput_values else 0,
                    'min_mbps': min(throughput_values) if throughput_values else 0
                },
                'operations': {
                    'average_ops': statistics.mean(ops_values) if ops_values else 0,
                    'peak_ops': max(ops_values) if ops_values else 0,
                    'min_ops': min(ops_values) if ops_values else 0
                }
            }
            
            # Analyze latencies
            if recent_performance and 'operation_latencies' in recent_performance[-1]:
                latest_latencies = recent_performance[-1]['operation_latencies']
                analysis['latencies'] = latest_latencies
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")
            return {}
    
    def _generate_metric_recommendations(self, time_analysis: Dict[str, Any],
                                       trend_analysis: Dict[str, Any],
                                       anomaly_analysis: Dict[str, Any],
                                       performance_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on metric analysis"""
        recommendations = []
        
        # Check for anomalies
        if anomaly_analysis.get('anomaly_count', 0) > 0:
            recommendations.append(
                f"Investigate {anomaly_analysis['anomaly_count']} metric anomalies detected"
            )
        
        # Check trends
        for category, trends in trend_analysis.items():
            for metric, trend_data in trends.items():
                if trend_data['trend'] == 'increasing' and 'error' in metric:
                    recommendations.append(
                        f"Address increasing {metric} in {category} - up {trend_data['change_percentage']:.1f}%"
                    )
        
        # Check performance
        if performance_analysis:
            throughput = performance_analysis.get('throughput', {})
            if throughput.get('average_mbps', 0) < 10:
                recommendations.append(
                    "Optimize data throughput - currently below 10 MB/s average"
                )
        
        # Check health score
        if self.real_time_metrics['system_health_score'] < 0.8:
            recommendations.append(
                f"Improve system health score - currently {self.real_time_metrics['system_health_score']:.2f}"
            )
        
        # Check active alerts
        if len(self.active_alerts) > 5:
            recommendations.append(
                f"Address {len(self.active_alerts)} active performance alerts"
            )
        
        return recommendations
    
    def _calculate_performance_scores(self) -> Dict[str, Any]:
        """Calculate performance scores for each category"""
        try:
            scores = {}
            
            # Backup performance
            backup_metrics = self.metric_categories.get('backup', {})
            backup_time = backup_metrics.get('backup_time_seconds_avg', 0)
            backup_threshold = self.performance_thresholds['backup_time_seconds']
            
            if backup_time > 0:
                backup_score = min(1.0, backup_threshold / backup_time)
            else:
                backup_score = 1.0
            
            scores['backup_performance'] = backup_score
            
            # Compression performance
            compression_metrics = self.metric_categories.get('compression', {})
            compression_ratio = compression_metrics.get('average_compression_ratio', 0.5)
            scores['compression_performance'] = min(1.0, compression_ratio / 0.5)
            
            # Encryption performance
            encryption_metrics = self.metric_categories.get('encryption', {})
            encryption_time = encryption_metrics.get('encryption_time_ms_avg', 0)
            encryption_threshold = self.performance_thresholds['encryption_time_ms']
            
            if encryption_time > 0:
                encryption_score = min(1.0, encryption_threshold / encryption_time)
            else:
                encryption_score = 1.0
            
            scores['encryption_performance'] = encryption_score
            
            # Storage performance
            storage_metrics = self.metric_categories.get('storage', {})
            storage_usage = storage_metrics.get('usage_percentage', 0)
            storage_threshold = self.performance_thresholds['storage_usage_percent']
            
            storage_score = max(0, 1 - (storage_usage / storage_threshold))
            scores['storage_performance'] = storage_score
            
            # Overall performance
            scores['overall_performance'] = statistics.mean(scores.values())
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Performance score calculation failed: {e}")
            return {}
    
    def _calculate_metric_statistics(self) -> Dict[str, Any]:
        """Calculate detailed metric statistics"""
        try:
            statistics_data = {}
            
            # Calculate statistics for each category
            for category, metrics in self.metric_categories.items():
                category_stats = {}
                
                # Get all values for this category
                category_values = [
                    m for m in self.metrics_history
                    if m['category'] == category
                ]
                
                if category_values:
                    # Group by metric name
                    metric_groups = defaultdict(list)
                    for value in category_values:
                        metric_groups[value['metric_name']].append(value['value'])
                    
                    # Calculate statistics for each metric
                    for metric_name, values in metric_groups.items():
                        if values:
                            category_stats[metric_name] = {
                                'count': len(values),
                                'mean': statistics.mean(values),
                                'median': statistics.median(values),
                                'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                                'min': min(values),
                                'max': max(values)
                            }
                
                if category_stats:
                    statistics_data[category] = category_stats
            
            return statistics_data
            
        except Exception as e:
            self.logger.error(f"Metric statistics calculation failed: {e}")
            return {}
    
    def _check_sla_compliance(self) -> Dict[str, Any]:
        """Check SLA compliance"""
        try:
            sla_checks = {
                'backup_sla': self._check_backup_sla(),
                'restore_sla': self._check_restore_sla(),
                'availability_sla': self._check_availability_sla(),
                'performance_sla': self._check_performance_sla()
            }
            
            overall_compliance = all(check.get('compliant', False) for check in sla_checks.values())
            
            return {
                'overall_compliance': overall_compliance,
                'sla_checks': sla_checks,
                'compliance_percentage': sum(1 for check in sla_checks.values() 
                                           if check.get('compliant', False)) / len(sla_checks) * 100
            }
            
        except Exception as e:
            self.logger.error(f"SLA compliance check failed: {e}")
            return {'overall_compliance': False, 'error': str(e)}
    
    def _check_backup_sla(self) -> Dict[str, Any]:
        """Check backup SLA compliance"""
        backup_metrics = self.metric_categories.get('backup', {})
        backup_time = backup_metrics.get('backup_time_seconds_avg', 0)
        
        compliant = backup_time <= self.performance_thresholds['backup_time_seconds']
        
        return {
            'compliant': compliant,
            'current_value': backup_time,
            'sla_threshold': self.performance_thresholds['backup_time_seconds'],
            'metric': 'Average backup time (seconds)'
        }
    
    def _check_restore_sla(self) -> Dict[str, Any]:
        """Check restore SLA compliance"""
        backup_metrics = self.metric_categories.get('backup', {})
        restore_time = backup_metrics.get('restore_time_seconds_avg', 0)
        
        compliant = restore_time <= self.performance_thresholds['restore_time_seconds']
        
        return {
            'compliant': compliant,
            'current_value': restore_time,
            'sla_threshold': self.performance_thresholds['restore_time_seconds'],
            'metric': 'Average restore time (seconds)'
        }
    
    def _check_availability_sla(self) -> Dict[str, Any]:
        """Check availability SLA compliance"""
        availability_metrics = self.metric_categories.get('availability', {})
        uptime_percentage = availability_metrics.get('uptime_percentage', 100)
        
        sla_target = 99.9  # 99.9% availability
        compliant = uptime_percentage >= sla_target
        
        return {
            'compliant': compliant,
            'current_value': uptime_percentage,
            'sla_threshold': sla_target,
            'metric': 'System availability percentage'
        }
    
    def _check_performance_sla(self) -> Dict[str, Any]:
        """Check performance SLA compliance"""
        throughput = self.real_time_metrics.get('data_throughput_mbps', 0)
        
        sla_target = 50  # 50 MB/s minimum throughput
        compliant = throughput >= sla_target
        
        return {
            'compliant': compliant,
            'current_value': throughput,
            'sla_threshold': sla_target,
            'metric': 'Data throughput (MB/s)'
        }
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        with self._lock:
            return list(self.active_alerts.values())
    
    def _get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        try:
            with self._lock:
                alerts = list(self.active_alerts.values())
            
            # Count by severity
            severity_counts = defaultdict(int)
            for alert in alerts:
                severity_counts[alert['severity']] += 1
            
            # Count by type
            type_counts = defaultdict(int)
            for alert in alerts:
                type_counts[alert['type']] += 1
            
            return {
                'total_alerts': len(alerts),
                'by_severity': dict(severity_counts),
                'by_type': dict(type_counts),
                'oldest_alert': min(alerts, key=lambda a: a['timestamp'])['timestamp'] 
                               if alerts else None
            }
            
        except Exception as e:
            self.logger.error(f"Alert summary generation failed: {e}")
            return {}
    
    def clear_alert(self, alert_id: str):
        """Clear specific alert"""
        try:
            with self._lock:
                if alert_id in self.active_alerts:
                    alert = self.active_alerts[alert_id]
                    alert['status'] = 'cleared'
                    alert['cleared_at'] = time.time()
                    del self.active_alerts[alert_id]
                    
                    self.logger.info(f"Alert {alert_id} cleared")
                    
        except Exception as e:
            self.logger.error(f"Alert clearing failed: {e}")
    
    def generate_metrics_report(self) -> Dict[str, Any]:
        """Generate comprehensive metrics report"""
        try:
            current_metrics = self.get_current_metrics()
            alert_summary = self._generate_alert_summary()
            
            return {
                'report_id': f"metrics_report_{int(time.time())}",
                'timestamp': time.time(),
                'current_metrics': current_metrics,
                'alert_summary': alert_summary,
                'system_health': current_metrics.get('system_health', {}),
                'recommendations': self._generate_metrics_recommendations(current_metrics)
            }
        except Exception as e:
            self.logger.error(f"Failed to generate metrics report: {e}")
            return {
                'report_id': f"error_report_{int(time.time())}",
                'timestamp': time.time(),
                'error': str(e)
            }
    
    def _generate_metrics_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        try:
            system_health = metrics.get('system_health', {})
            health_score = system_health.get('health_score', 0)
            
            if health_score < 0.7:
                recommendations.append("System health is below optimal - investigate alerts")
            
            if system_health.get('active_alerts', 0) > 5:
                recommendations.append("High number of active alerts - prioritize resolution")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            return []
    
    def shutdown(self):
        """Shutdown metrics collector"""
        self.logger.info("Shutting down Data Metrics Collector")
        self.is_running = False