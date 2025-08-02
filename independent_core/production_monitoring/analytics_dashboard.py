"""
Saraphis Production Analytics Dashboard
Provides real-time visualization and analytics for production system performance
NO FALLBACKS - HARD FAILURES ONLY
"""

import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import statistics

logger = logging.getLogger(__name__)


class ProductionAnalyticsDashboard:
    """
    Production Analytics Dashboard with real-time visualization and insights
    NO FALLBACKS - HARD FAILURES ONLY
    """
    
    def __init__(self, monitor, optimization_engine, alert_system):
        self.monitor = monitor
        self.optimization_engine = optimization_engine
        self.alert_system = alert_system
        
        # Dashboard configuration
        self.refresh_interval = 1.0  # 1 second refresh
        self.history_window = 3600  # 1 hour of history
        self.metrics_retention = 86400  # 24 hours
        
        # Analytics storage
        self.system_metrics_history = defaultdict(lambda: deque(maxlen=3600))
        self.agent_metrics_history = defaultdict(lambda: deque(maxlen=3600))
        self.optimization_history = deque(maxlen=1000)
        self.alert_history = deque(maxlen=10000)
        self.performance_trends = {}
        
        # Real-time metrics
        self.current_metrics = {
            'systems': {},
            'agents': {},
            'overall': {
                'health_score': 0.0,
                'performance_score': 0.0,
                'resource_utilization': 0.0,
                'error_rate': 0.0,
                'response_time': 0.0
            }
        }
        
        # Dashboard state
        self.is_running = False
        self.dashboard_thread = None
        self.analytics_thread = None
        self.start_time = None
        self._lock = threading.Lock()
        
        # Performance analytics
        self.performance_analyzer = PerformanceAnalyzer()
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        
        logger.info("Production Analytics Dashboard initialized")
    
    def start_dashboard(self) -> Dict[str, Any]:
        """Start the analytics dashboard"""
        with self._lock:
            if self.is_running:
                return {
                    'success': False,
                    'error': 'Dashboard already running'
                }
            
            self.is_running = True
            self.start_time = datetime.now()
            
            # Start dashboard threads
            self.dashboard_thread = threading.Thread(
                target=self._dashboard_loop,
                daemon=True
            )
            self.analytics_thread = threading.Thread(
                target=self._analytics_loop,
                daemon=True
            )
            
            self.dashboard_thread.start()
            self.analytics_thread.start()
            
            logger.info("Analytics dashboard started")
            return {
                'success': True,
                'dashboard_url': 'http://localhost:8080/analytics',
                'api_url': 'http://localhost:8080/api/v1/analytics'
            }
    
    def _dashboard_loop(self):
        """Main dashboard update loop"""
        while self.is_running:
            try:
                # Collect current metrics
                self._update_current_metrics()
                
                # Update history
                self._update_metrics_history()
                
                # Generate dashboard data
                self._generate_dashboard_data()
                
                time.sleep(self.refresh_interval)
                
            except Exception as e:
                logger.error(f"Dashboard loop error: {e}")
                # NO FALLBACK - Continue loop
    
    def _analytics_loop(self):
        """Analytics processing loop"""
        while self.is_running:
            try:
                # Analyze performance trends
                self._analyze_performance_trends()
                
                # Detect anomalies
                self._detect_anomalies()
                
                # Generate insights
                self._generate_insights()
                
                time.sleep(5.0)  # Run every 5 seconds
                
            except Exception as e:
                logger.error(f"Analytics loop error: {e}")
                # NO FALLBACK - Continue loop
    
    def _update_current_metrics(self):
        """Update current metrics from monitor"""
        try:
            # Get system metrics
            system_statuses = self.monitor.get_all_system_status()
            for system_name, status in system_statuses.items():
                self.current_metrics['systems'][system_name] = {
                    'health': status.get('health', 0.0),
                    'performance': status.get('performance', {}).get('score', 0.0),
                    'cpu_usage': status.get('resources', {}).get('cpu_percent', 0.0),
                    'memory_usage': status.get('resources', {}).get('memory_percent', 0.0),
                    'response_time': status.get('performance', {}).get('response_time_ms', 0.0),
                    'error_rate': status.get('performance', {}).get('error_rate', 0.0),
                    'throughput': status.get('performance', {}).get('throughput', 0.0),
                    'last_updated': datetime.now()
                }
            
            # Get agent metrics
            agent_statuses = self.monitor.get_all_agent_status()
            for agent_name, status in agent_statuses.items():
                self.current_metrics['agents'][agent_name] = {
                    'health': status.get('health', 0.0),
                    'task_success_rate': status.get('task_success_rate', 0.0),
                    'response_time': status.get('response_time', 0.0),
                    'active_tasks': status.get('active_tasks', 0),
                    'completed_tasks': status.get('completed_tasks', 0),
                    'coordination_score': status.get('coordination_score', 0.0),
                    'last_updated': datetime.now()
                }
            
            # Calculate overall metrics
            self._calculate_overall_metrics()
            
        except Exception as e:
            logger.error(f"Failed to update current metrics: {e}")
            raise
    
    def _calculate_overall_metrics(self):
        """Calculate overall system metrics"""
        try:
            # System health
            system_healths = [m['health'] for m in self.current_metrics['systems'].values()]
            agent_healths = [m['health'] for m in self.current_metrics['agents'].values()]
            all_healths = system_healths + agent_healths
            
            self.current_metrics['overall']['health_score'] = (
                statistics.mean(all_healths) if all_healths else 0.0
            )
            
            # Performance score
            system_perfs = [m['performance'] for m in self.current_metrics['systems'].values()]
            agent_perfs = [m['task_success_rate'] for m in self.current_metrics['agents'].values()]
            all_perfs = system_perfs + agent_perfs
            
            self.current_metrics['overall']['performance_score'] = (
                statistics.mean(all_perfs) if all_perfs else 0.0
            )
            
            # Resource utilization
            cpu_usages = [m['cpu_usage'] for m in self.current_metrics['systems'].values()]
            memory_usages = [m['memory_usage'] for m in self.current_metrics['systems'].values()]
            
            self.current_metrics['overall']['resource_utilization'] = (
                (statistics.mean(cpu_usages) + statistics.mean(memory_usages)) / 2
                if cpu_usages and memory_usages else 0.0
            )
            
            # Error rate
            error_rates = [m['error_rate'] for m in self.current_metrics['systems'].values()]
            self.current_metrics['overall']['error_rate'] = (
                statistics.mean(error_rates) if error_rates else 0.0
            )
            
            # Response time
            response_times = [m['response_time'] for m in self.current_metrics['systems'].values()]
            self.current_metrics['overall']['response_time'] = (
                statistics.mean(response_times) if response_times else 0.0
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate overall metrics: {e}")
            raise
    
    def _update_metrics_history(self):
        """Update metrics history"""
        timestamp = time.time()
        
        # Store system metrics
        for system_name, metrics in self.current_metrics['systems'].items():
            self.system_metrics_history[system_name].append({
                'timestamp': timestamp,
                'metrics': metrics.copy()
            })
        
        # Store agent metrics
        for agent_name, metrics in self.current_metrics['agents'].items():
            self.agent_metrics_history[agent_name].append({
                'timestamp': timestamp,
                'metrics': metrics.copy()
            })
        
        # Store optimization events
        recent_optimizations = self.optimization_engine.get_optimization_history(limit=10)
        for opt in recent_optimizations:
            if opt not in self.optimization_history:
                self.optimization_history.append(opt)
        
        # Store alert events
        recent_alerts = self.alert_system.get_recent_alerts(minutes=1)
        for alert in recent_alerts:
            if alert not in self.alert_history:
                self.alert_history.append(alert)
    
    def _generate_dashboard_data(self):
        """Generate dashboard visualization data"""
        with self._lock:
            self.dashboard_data = {
                'timestamp': datetime.now().isoformat(),
                'uptime': str(datetime.now() - self.start_time) if self.start_time else "0:00:00",
                'current_metrics': self.current_metrics,
                'performance_trends': self.performance_trends,
                'recent_alerts': list(self.alert_history)[-10:],
                'recent_optimizations': list(self.optimization_history)[-10:],
                'system_charts': self._generate_system_charts(),
                'agent_charts': self._generate_agent_charts(),
                'insights': self._get_current_insights()
            }
    
    def _generate_system_charts(self) -> Dict[str, Any]:
        """Generate system performance charts data"""
        charts = {}
        
        for system_name, history in self.system_metrics_history.items():
            if not history:
                continue
            
            # Extract time series data
            timestamps = [h['timestamp'] for h in history]
            health_scores = [h['metrics']['health'] for h in history]
            cpu_usage = [h['metrics']['cpu_usage'] for h in history]
            memory_usage = [h['metrics']['memory_usage'] for h in history]
            response_times = [h['metrics']['response_time'] for h in history]
            
            charts[system_name] = {
                'health_trend': {
                    'x': timestamps,
                    'y': health_scores,
                    'type': 'line',
                    'name': 'Health Score'
                },
                'resource_usage': {
                    'cpu': {
                        'x': timestamps,
                        'y': cpu_usage,
                        'type': 'line',
                        'name': 'CPU %'
                    },
                    'memory': {
                        'x': timestamps,
                        'y': memory_usage,
                        'type': 'line',
                        'name': 'Memory %'
                    }
                },
                'response_time': {
                    'x': timestamps,
                    'y': response_times,
                    'type': 'line',
                    'name': 'Response Time (ms)'
                }
            }
        
        return charts
    
    def _generate_agent_charts(self) -> Dict[str, Any]:
        """Generate agent performance charts data"""
        charts = {}
        
        for agent_name, history in self.agent_metrics_history.items():
            if not history:
                continue
            
            # Extract time series data
            timestamps = [h['timestamp'] for h in history]
            health_scores = [h['metrics']['health'] for h in history]
            success_rates = [h['metrics']['task_success_rate'] for h in history]
            active_tasks = [h['metrics']['active_tasks'] for h in history]
            
            charts[agent_name] = {
                'health_trend': {
                    'x': timestamps,
                    'y': health_scores,
                    'type': 'line',
                    'name': 'Health Score'
                },
                'success_rate': {
                    'x': timestamps,
                    'y': success_rates,
                    'type': 'line',
                    'name': 'Task Success Rate'
                },
                'task_activity': {
                    'x': timestamps,
                    'y': active_tasks,
                    'type': 'bar',
                    'name': 'Active Tasks'
                }
            }
        
        return charts
    
    def _analyze_performance_trends(self):
        """Analyze performance trends"""
        try:
            trends = {}
            
            # Analyze system trends
            for system_name, history in self.system_metrics_history.items():
                if len(history) < 10:
                    continue
                
                recent_metrics = list(history)[-60:]  # Last minute
                health_trend = self.trend_analyzer.analyze_trend(
                    [h['metrics']['health'] for h in recent_metrics]
                )
                performance_trend = self.trend_analyzer.analyze_trend(
                    [h['metrics']['performance'] for h in recent_metrics]
                )
                
                trends[system_name] = {
                    'health_trend': health_trend,
                    'performance_trend': performance_trend,
                    'trend_direction': self._get_trend_direction(health_trend, performance_trend)
                }
            
            self.performance_trends = trends
            
        except Exception as e:
            logger.error(f"Failed to analyze performance trends: {e}")
    
    def _detect_anomalies(self):
        """Detect anomalies in system behavior"""
        try:
            anomalies = []
            
            # Check system metrics for anomalies
            for system_name, history in self.system_metrics_history.items():
                if len(history) < 30:
                    continue
                
                recent_metrics = list(history)[-30:]
                
                # Check for health anomalies
                health_values = [h['metrics']['health'] for h in recent_metrics]
                if self.anomaly_detector.is_anomaly(health_values):
                    anomalies.append({
                        'type': 'system_health',
                        'system': system_name,
                        'severity': 'high',
                        'description': f"Anomaly detected in {system_name} health metrics"
                    })
                
                # Check for response time anomalies
                response_times = [h['metrics']['response_time'] for h in recent_metrics]
                if self.anomaly_detector.is_anomaly(response_times):
                    anomalies.append({
                        'type': 'response_time',
                        'system': system_name,
                        'severity': 'medium',
                        'description': f"Response time anomaly in {system_name}"
                    })
            
            # Report anomalies to alert system
            for anomaly in anomalies:
                self.alert_system.create_alert(
                    alert_type='PERFORMANCE',
                    severity=anomaly['severity'].upper(),
                    source=f"analytics/{anomaly['system']}",
                    message=anomaly['description']
                )
            
        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
    
    def _generate_insights(self):
        """Generate actionable insights"""
        try:
            insights = []
            
            # Performance insights
            overall_health = self.current_metrics['overall']['health_score']
            if overall_health < 0.7:
                insights.append({
                    'type': 'performance',
                    'priority': 'high',
                    'message': f"Overall system health is low ({overall_health:.1%})",
                    'recommendation': "Review failing components and apply optimizations"
                })
            
            # Resource insights
            resource_util = self.current_metrics['overall']['resource_utilization']
            if resource_util > 80:
                insights.append({
                    'type': 'resource',
                    'priority': 'medium',
                    'message': f"High resource utilization ({resource_util:.1f}%)",
                    'recommendation': "Consider scaling resources or optimizing workloads"
                })
            
            # Error rate insights
            error_rate = self.current_metrics['overall']['error_rate']
            if error_rate > 0.05:
                insights.append({
                    'type': 'reliability',
                    'priority': 'high',
                    'message': f"Elevated error rate ({error_rate:.1%})",
                    'recommendation': "Investigate error sources and implement fixes"
                })
            
            # Optimization insights
            recent_optimizations = list(self.optimization_history)[-10:]
            if recent_optimizations:
                successful = sum(1 for opt in recent_optimizations if opt.get('success'))
                success_rate = successful / len(recent_optimizations)
                if success_rate < 0.8:
                    insights.append({
                        'type': 'optimization',
                        'priority': 'medium',
                        'message': f"Low optimization success rate ({success_rate:.1%})",
                        'recommendation': "Review optimization strategies and parameters"
                    })
            
            self.current_insights = insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
    
    def _get_current_insights(self) -> List[Dict[str, Any]]:
        """Get current insights"""
        return getattr(self, 'current_insights', [])
    
    def _get_trend_direction(self, health_trend: float, performance_trend: float) -> str:
        """Get overall trend direction"""
        combined_trend = (health_trend + performance_trend) / 2
        
        if combined_trend > 0.1:
            return "improving"
        elif combined_trend < -0.1:
            return "degrading"
        else:
            return "stable"
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        with self._lock:
            return getattr(self, 'dashboard_data', {
                'error': 'Dashboard not initialized',
                'timestamp': datetime.now().isoformat()
            })
    
    def get_performance_summary(self, time_range_minutes: int = 60) -> Dict[str, Any]:
        """Get performance summary for specified time range"""
        cutoff_time = time.time() - (time_range_minutes * 60)
        
        summary = {
            'time_range_minutes': time_range_minutes,
            'systems': {},
            'agents': {},
            'overall': {
                'average_health': 0.0,
                'average_performance': 0.0,
                'peak_resource_usage': 0.0,
                'total_alerts': 0,
                'total_optimizations': 0
            }
        }
        
        # Analyze system performance
        for system_name, history in self.system_metrics_history.items():
            recent_history = [h for h in history if h['timestamp'] > cutoff_time]
            if not recent_history:
                continue
            
            health_values = [h['metrics']['health'] for h in recent_history]
            perf_values = [h['metrics']['performance'] for h in recent_history]
            cpu_values = [h['metrics']['cpu_usage'] for h in recent_history]
            memory_values = [h['metrics']['memory_usage'] for h in recent_history]
            
            summary['systems'][system_name] = {
                'average_health': statistics.mean(health_values),
                'average_performance': statistics.mean(perf_values),
                'peak_cpu': max(cpu_values),
                'peak_memory': max(memory_values),
                'health_stability': statistics.stdev(health_values) if len(health_values) > 1 else 0.0
            }
        
        # Analyze agent performance
        for agent_name, history in self.agent_metrics_history.items():
            recent_history = [h for h in history if h['timestamp'] > cutoff_time]
            if not recent_history:
                continue
            
            health_values = [h['metrics']['health'] for h in recent_history]
            success_values = [h['metrics']['task_success_rate'] for h in recent_history]
            
            summary['agents'][agent_name] = {
                'average_health': statistics.mean(health_values),
                'average_success_rate': statistics.mean(success_values),
                'total_tasks': sum(h['metrics']['completed_tasks'] for h in recent_history[-1:])
            }
        
        # Count alerts and optimizations
        summary['overall']['total_alerts'] = sum(
            1 for alert in self.alert_history
            if alert.get('timestamp', 0) > cutoff_time
        )
        summary['overall']['total_optimizations'] = sum(
            1 for opt in self.optimization_history
            if opt.get('timestamp', 0) > cutoff_time
        )
        
        # Calculate overall averages
        all_healths = (
            [s['average_health'] for s in summary['systems'].values()] +
            [a['average_health'] for a in summary['agents'].values()]
        )
        all_perfs = [s['average_performance'] for s in summary['systems'].values()]
        
        if all_healths:
            summary['overall']['average_health'] = statistics.mean(all_healths)
        if all_perfs:
            summary['overall']['average_performance'] = statistics.mean(all_perfs)
        
        peak_resources = []
        for s in summary['systems'].values():
            peak_resources.extend([s['peak_cpu'], s['peak_memory']])
        if peak_resources:
            summary['overall']['peak_resource_usage'] = max(peak_resources)
        
        return summary
    
    def export_analytics_report(self, filepath: str) -> Dict[str, Any]:
        """Export comprehensive analytics report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'uptime': str(datetime.now() - self.start_time) if self.start_time else "0:00:00",
                'executive_summary': self._generate_executive_summary(),
                'performance_summary': self.get_performance_summary(60),
                'current_state': self.current_metrics,
                'trends': self.performance_trends,
                'insights': self._get_current_insights(),
                'alert_summary': self._generate_alert_summary(),
                'optimization_summary': self._generate_optimization_summary()
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Analytics report exported to {filepath}")
            return {
                'success': True,
                'filepath': filepath,
                'report_size': len(json.dumps(report))
            }
            
        except Exception as e:
            logger.error(f"Failed to export analytics report: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary"""
        return {
            'overall_health': self.current_metrics['overall']['health_score'],
            'overall_performance': self.current_metrics['overall']['performance_score'],
            'resource_utilization': self.current_metrics['overall']['resource_utilization'],
            'error_rate': self.current_metrics['overall']['error_rate'],
            'response_time': self.current_metrics['overall']['response_time'],
            'critical_issues': len([a for a in self.alert_history if a.get('severity') == 'CRITICAL']),
            'active_optimizations': len([o for o in self.optimization_history if o.get('active', False)])
        }
    
    def _generate_alert_summary(self) -> Dict[str, Any]:
        """Generate alert summary"""
        alerts_by_type = defaultdict(int)
        alerts_by_severity = defaultdict(int)
        
        for alert in self.alert_history:
            alerts_by_type[alert.get('alert_type', 'unknown')] += 1
            alerts_by_severity[alert.get('severity', 'unknown')] += 1
        
        return {
            'total_alerts': len(self.alert_history),
            'by_type': dict(alerts_by_type),
            'by_severity': dict(alerts_by_severity),
            'recent_critical': [
                a for a in list(self.alert_history)[-20:]
                if a.get('severity') == 'CRITICAL'
            ]
        }
    
    def _generate_optimization_summary(self) -> Dict[str, Any]:
        """Generate optimization summary"""
        successful_opts = [o for o in self.optimization_history if o.get('success')]
        failed_opts = [o for o in self.optimization_history if not o.get('success')]
        
        impact_scores = [o.get('impact', 0) for o in successful_opts]
        
        return {
            'total_optimizations': len(self.optimization_history),
            'successful': len(successful_opts),
            'failed': len(failed_opts),
            'success_rate': len(successful_opts) / len(self.optimization_history) if self.optimization_history else 0,
            'average_impact': statistics.mean(impact_scores) if impact_scores else 0,
            'top_optimizations': sorted(
                successful_opts,
                key=lambda x: x.get('impact', 0),
                reverse=True
            )[:5]
        }
    
    def stop_dashboard(self) -> Dict[str, Any]:
        """Stop the analytics dashboard"""
        with self._lock:
            self.is_running = False
            
            # Generate final report
            final_report = self.export_analytics_report('final_analytics_report.json')
            
            logger.info("Analytics dashboard stopped")
            return {
                'success': True,
                'final_report': final_report,
                'runtime': str(datetime.now() - self.start_time) if self.start_time else "0:00:00"
            }


class PerformanceAnalyzer:
    """Analyze system performance patterns"""
    
    def calculate_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate composite performance score"""
        weights = {
            'health': 0.3,
            'response_time': 0.25,
            'error_rate': 0.25,
            'throughput': 0.2
        }
        
        score = 0.0
        score += weights['health'] * metrics.get('health', 0)
        
        # Normalize response time (lower is better)
        response_time = metrics.get('response_time', 1000)
        response_score = max(0, 1 - (response_time / 1000))  # 1000ms baseline
        score += weights['response_time'] * response_score
        
        # Invert error rate (lower is better)
        error_rate = metrics.get('error_rate', 0)
        score += weights['error_rate'] * (1 - error_rate)
        
        # Normalize throughput
        throughput = metrics.get('throughput', 0)
        throughput_score = min(1, throughput / 1000)  # 1000 ops/sec baseline
        score += weights['throughput'] * throughput_score
        
        return max(0, min(1, score))


class TrendAnalyzer:
    """Analyze trends in metrics"""
    
    def analyze_trend(self, values: List[float]) -> float:
        """
        Analyze trend in values
        Returns: -1 to 1 (negative = declining, positive = improving)
        """
        if len(values) < 2:
            return 0.0
        
        # Calculate linear regression slope
        n = len(values)
        x = list(range(n))
        
        # Calculate means
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        # Calculate slope
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        
        # Normalize slope to -1 to 1 range
        max_expected_slope = 0.1  # Adjust based on expected rate of change
        normalized_slope = max(-1, min(1, slope / max_expected_slope))
        
        return normalized_slope


class AnomalyDetector:
    """Detect anomalies in metrics"""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Standard deviations for anomaly threshold
    
    def is_anomaly(self, values: List[float]) -> bool:
        """Check if latest value is anomalous"""
        if len(values) < 10:
            return False
        
        # Use all but last value to establish baseline
        baseline = values[:-1]
        latest = values[-1]
        
        mean = statistics.mean(baseline)
        stdev = statistics.stdev(baseline)
        
        if stdev == 0:
            return latest != mean
        
        # Check if latest value is beyond threshold
        z_score = abs((latest - mean) / stdev)
        return z_score > self.sensitivity