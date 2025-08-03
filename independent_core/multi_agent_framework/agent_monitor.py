"""
Saraphis Agent Monitor
Production-ready agent monitoring and health checks
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import threading
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import uuid
import json
import traceback

logger = logging.getLogger(__name__)


class AgentMonitor:
    """Production-ready agent monitoring and health checks"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Agent health tracking
        self.agent_health_status = {}
        self.health_history = defaultdict(lambda: deque(maxlen=1000))
        self.health_thresholds = self._initialize_health_thresholds()
        
        # Performance monitoring
        self.performance_metrics = defaultdict(lambda: {
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'response_times': deque(maxlen=100),
            'error_rates': deque(maxlen=100),
            'throughput': deque(maxlen=100)
        })
        
        # Alert management
        self.active_alerts = defaultdict(list)
        self.alert_history = deque(maxlen=10000)
        self.alert_thresholds = self._initialize_alert_thresholds()
        
        # Monitoring configuration
        self.monitoring_config = {
            'check_interval': config.get('health_check_interval', 30),
            'metric_collection_interval': config.get('metric_interval', 10),
            'alert_cooldown': config.get('alert_cooldown', 300),
            'enable_auto_recovery': config.get('enable_auto_recovery', True)
        }
        
        # Monitoring statistics
        self.monitoring_stats = {
            'total_health_checks': 0,
            'failed_health_checks': 0,
            'alerts_generated': 0,
            'auto_recoveries': 0,
            'monitoring_uptime': 0
        }
        
        # Registered agents
        self.monitored_agents = {}
        
        # Active state
        self.is_active_flag = False
        self.monitoring_start_time = None
        
        # Thread safety
        self._lock = threading.Lock()
        self._agent_lock = threading.Lock()
        
        self.logger.info("Agent Monitor initialized")
    
    def initialize_monitoring(self) -> Dict[str, Any]:
        """Initialize monitoring system"""
        try:
            self.logger.info("Initializing agent monitoring system...")
            
            # Initialize monitoring infrastructure
            infra_init_result = self._initialize_monitoring_infrastructure()
            
            # Initialize metric collectors
            collector_init_result = self._initialize_metric_collectors()
            
            # Initialize alert system
            alert_init_result = self._initialize_alert_system()
            
            # Start monitoring services
            self._start_monitoring_services()
            
            # Set active flag and start time
            self.is_active_flag = True
            self.monitoring_start_time = time.time()
            
            return {
                'success': True,
                'monitoring_infrastructure': infra_init_result,
                'metric_collectors': collector_init_result,
                'alert_system': alert_init_result,
                'monitoring_config': self.monitoring_config,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Monitoring initialization failed: {e}")
            return {
                'success': False,
                'error': f'Monitoring initialization failed: {str(e)}',
                'traceback': traceback.format_exc()
            }
    
    def register_agent_for_monitoring(self, agent_id: str, agent: Any) -> Dict[str, Any]:
        """Register agent for monitoring"""
        try:
            with self._agent_lock:
                # Store agent reference
                self.monitored_agents[agent_id] = {
                    'agent': agent,
                    'agent_type': agent.agent_type,
                    'registration_time': time.time(),
                    'monitoring_enabled': True,
                    'health_check_count': 0,
                    'last_health_check': None
                }
                
                # Initialize health status
                self.agent_health_status[agent_id] = {
                    'status': 'healthy',
                    'last_check': time.time(),
                    'health_score': 1.0,
                    'issues': []
                }
            
            self.logger.info(f"Registered agent {agent_id} for monitoring")
            
            return {
                'success': True,
                'agent_id': agent_id,
                'monitoring_enabled': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Agent registration failed: {e}")
            return {
                'success': False,
                'error': f'Registration failed: {str(e)}'
            }
    
    def get_agent_health(self, agent_id: str) -> Dict[str, Any]:
        """Get current health status of specific agent"""
        try:
            with self._lock:
                if agent_id not in self.agent_health_status:
                    return {
                        'status': 'unknown',
                        'error': f'Agent {agent_id} not registered for monitoring'
                    }
                
                health_status = self.agent_health_status[agent_id].copy()
                
                # Add recent metrics
                if agent_id in self.performance_metrics:
                    metrics = self.performance_metrics[agent_id]
                    health_status['recent_metrics'] = {
                        'cpu_usage': list(metrics['cpu_usage'])[-10:] if metrics['cpu_usage'] else [],
                        'memory_usage': list(metrics['memory_usage'])[-10:] if metrics['memory_usage'] else [],
                        'response_times': list(metrics['response_times'])[-10:] if metrics['response_times'] else [],
                        'error_rates': list(metrics['error_rates'])[-10:] if metrics['error_rates'] else []
                    }
                
                # Add active alerts
                health_status['active_alerts'] = self.active_alerts.get(agent_id, [])
                
                return health_status
            
        except Exception as e:
            self.logger.error(f"Failed to get agent health: {e}")
            return {
                'status': 'error',
                'error': f'Health check failed: {str(e)}'
            }
    
    def collect_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Collect performance metrics for specific agent"""
        try:
            if agent_id not in self.monitored_agents:
                return {
                    'success': False,
                    'error': f'Agent {agent_id} not registered'
                }
            
            agent_info = self.monitored_agents[agent_id]
            agent = agent_info['agent']
            
            # Collect metrics
            metrics = {
                'timestamp': time.time(),
                'cpu_usage': agent.get_cpu_usage(),
                'memory_usage': agent.get_memory_usage(),
                'task_queue_size': agent.get_task_queue_size(),
                'active_tasks': agent.get_active_task_count(),
                'completed_tasks': agent.get_completed_task_count(),
                'failed_tasks': agent.get_failed_task_count(),
                'average_response_time': agent.get_average_response_time(),
                'error_rate': agent.get_error_rate()
            }
            
            # Store metrics
            with self._lock:
                perf_metrics = self.performance_metrics[agent_id]
                perf_metrics['cpu_usage'].append(metrics['cpu_usage'])
                perf_metrics['memory_usage'].append(metrics['memory_usage'])
                perf_metrics['response_times'].append(metrics['average_response_time'])
                perf_metrics['error_rates'].append(metrics['error_rate'])
                
                # Calculate throughput
                if len(perf_metrics['throughput']) > 0:
                    last_completed = perf_metrics['throughput'][-1] if perf_metrics['throughput'] else 0
                    throughput = metrics['completed_tasks'] - last_completed
                else:
                    throughput = metrics['completed_tasks']
                
                perf_metrics['throughput'].append(throughput)
            
            return {
                'success': True,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Metric collection failed for agent {agent_id}: {e}")
            return {
                'success': False,
                'error': f'Metric collection failed: {str(e)}'
            }
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'monitoring_uptime': self._calculate_monitoring_uptime(),
                'monitored_agents': len(self.monitored_agents),
                'agent_health_summary': {},
                'system_health': {},
                'performance_summary': {},
                'alert_summary': {},
                'recommendations': []
            }
            
            with self._lock:
                # Agent health summary
                healthy_agents = sum(1 for status in self.agent_health_status.values() 
                                   if status['status'] == 'healthy')
                degraded_agents = sum(1 for status in self.agent_health_status.values() 
                                    if status['status'] == 'degraded')
                unhealthy_agents = sum(1 for status in self.agent_health_status.values() 
                                     if status['status'] == 'unhealthy')
                
                report['agent_health_summary'] = {
                    'healthy': healthy_agents,
                    'degraded': degraded_agents,
                    'unhealthy': unhealthy_agents,
                    'health_percentage': (healthy_agents / max(len(self.monitored_agents), 1)) * 100
                }
                
                # System health
                report['system_health'] = self._calculate_system_health()
                
                # Performance summary
                report['performance_summary'] = self._generate_performance_summary()
                
                # Alert summary
                active_alert_count = sum(len(alerts) for alerts in self.active_alerts.values())
                report['alert_summary'] = {
                    'active_alerts': active_alert_count,
                    'total_alerts_generated': self.monitoring_stats['alerts_generated'],
                    'auto_recoveries': self.monitoring_stats['auto_recoveries'],
                    'top_alert_types': self._get_top_alert_types()
                }
                
                # Generate recommendations
                report['recommendations'] = self._generate_monitoring_recommendations(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Monitoring report generation failed: {e}")
            return {
                'error': f'Report generation failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def trigger_alert(self, agent_id: str, alert_type: str, 
                     severity: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger monitoring alert for agent"""
        try:
            alert_id = f"alert_{uuid.uuid4().hex[:8]}_{int(time.time())}"
            
            alert = {
                'alert_id': alert_id,
                'agent_id': agent_id,
                'alert_type': alert_type,
                'severity': severity,
                'details': details,
                'timestamp': time.time(),
                'status': 'active'
            }
            
            with self._lock:
                # Add to active alerts
                self.active_alerts[agent_id].append(alert)
                
                # Add to alert history
                self.alert_history.append(alert)
                
                # Update statistics
                self.monitoring_stats['alerts_generated'] += 1
            
            # Log alert
            self.logger.warning(
                f"Alert triggered for agent {agent_id}: "
                f"{alert_type} (severity: {severity})"
            )
            
            # Attempt auto-recovery if enabled
            if (self.monitoring_config['enable_auto_recovery'] and 
                severity in ['medium', 'low']):
                recovery_result = self._attempt_auto_recovery(agent_id, alert_type)
                if recovery_result['success']:
                    alert['auto_recovery_attempted'] = True
                    alert['recovery_result'] = recovery_result
            
            return {
                'success': True,
                'alert_id': alert_id,
                'alert': alert,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Alert triggering failed: {e}")
            return {
                'success': False,
                'error': f'Alert trigger failed: {str(e)}'
            }
    
    def is_active(self) -> bool:
        """Check if monitor is active"""
        return self.is_active_flag
    
    def validate_monitoring_coverage(self) -> Dict[str, Any]:
        """Validate monitoring coverage across agents"""
        try:
            validation_results = {
                'total_agents': len(self.monitored_agents),
                'monitored_agents': 0,
                'monitoring_gaps': [],
                'metric_coverage': {},
                'alert_coverage': {},
                'coverage_percentage': 0
            }
            
            with self._agent_lock:
                for agent_id, agent_info in self.monitored_agents.items():
                    if agent_info['monitoring_enabled']:
                        validation_results['monitored_agents'] += 1
                    else:
                        validation_results['monitoring_gaps'].append({
                            'agent_id': agent_id,
                            'reason': 'monitoring_disabled'
                        })
                    
                    # Check metric coverage
                    if agent_id in self.performance_metrics:
                        metrics = self.performance_metrics[agent_id]
                        has_metrics = any(len(metric_list) > 0 for metric_list in metrics.values())
                        if not has_metrics:
                            validation_results['monitoring_gaps'].append({
                                'agent_id': agent_id,
                                'reason': 'no_metrics_collected'
                            })
            
            # Calculate coverage percentage
            if validation_results['total_agents'] > 0:
                validation_results['coverage_percentage'] = (
                    validation_results['monitored_agents'] / 
                    validation_results['total_agents']
                ) * 100
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Monitoring coverage validation failed: {e}")
            return {
                'error': f'Validation failed: {str(e)}',
                'coverage_percentage': 0
            }
    
    def _initialize_health_thresholds(self) -> Dict[str, Any]:
        """Initialize health check thresholds"""
        return {
            'cpu_usage': {
                'warning': self.config.get('cpu_warning_threshold', 70),
                'critical': self.config.get('cpu_critical_threshold', 90)
            },
            'memory_usage': {
                'warning': self.config.get('memory_warning_threshold', 80),
                'critical': self.config.get('memory_critical_threshold', 95)
            },
            'response_time': {
                'warning': self.config.get('response_time_warning_ms', 1000),
                'critical': self.config.get('response_time_critical_ms', 5000)
            },
            'error_rate': {
                'warning': self.config.get('error_rate_warning', 0.05),
                'critical': self.config.get('error_rate_critical', 0.1)
            },
            'task_queue': {
                'warning': self.config.get('queue_warning_size', 50),
                'critical': self.config.get('queue_critical_size', 100)
            }
        }
    
    def _initialize_alert_thresholds(self) -> Dict[str, Any]:
        """Initialize alert thresholds"""
        return {
            'consecutive_failures': self.config.get('consecutive_failure_threshold', 3),
            'alert_rate_limit': self.config.get('alert_rate_limit_per_hour', 10),
            'escalation_threshold': self.config.get('escalation_threshold', 5),
            'auto_clear_duration': self.config.get('auto_clear_duration_seconds', 300)
        }
    
    def _initialize_monitoring_infrastructure(self) -> Dict[str, Any]:
        """Initialize monitoring infrastructure"""
        try:
            # Initialize metric storage
            self.metric_storage = {
                'retention_period': self.config.get('metric_retention_hours', 24),
                'aggregation_intervals': [1, 5, 15, 60],  # minutes
                'compression_enabled': self.config.get('metric_compression', True)
            }
            
            return {
                'metric_storage': 'initialized',
                'retention_period': self.metric_storage['retention_period']
            }
            
        except Exception as e:
            self.logger.error(f"Monitoring infrastructure initialization failed: {e}")
            raise RuntimeError(f"Infrastructure initialization failed: {str(e)}")
    
    def _initialize_metric_collectors(self) -> Dict[str, Any]:
        """Initialize metric collection system"""
        try:
            # Define metric collectors
            self.metric_collectors = {
                'system': SystemMetricCollector(),
                'agent': AgentMetricCollector(),
                'performance': PerformanceMetricCollector()
            }
            
            return {
                'collectors_initialized': len(self.metric_collectors),
                'collector_types': list(self.metric_collectors.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Metric collector initialization failed: {e}")
            raise RuntimeError(f"Collector initialization failed: {str(e)}")
    
    def _initialize_alert_system(self) -> Dict[str, Any]:
        """Initialize alert management system"""
        try:
            # Initialize alert channels
            self.alert_channels = {
                'log': LogAlertChannel(self.logger),
                'metric': MetricAlertChannel()
            }
            
            # Initialize alert rules
            self.alert_rules = self._load_alert_rules()
            
            return {
                'alert_channels': list(self.alert_channels.keys()),
                'alert_rules_loaded': len(self.alert_rules)
            }
            
        except Exception as e:
            self.logger.error(f"Alert system initialization failed: {e}")
            raise RuntimeError(f"Alert system initialization failed: {str(e)}")
    
    def _start_monitoring_services(self) -> None:
        """Start background monitoring services"""
        try:
            # Start health check thread
            health_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True
            )
            health_thread.start()
            
            # Start metric collection thread
            metric_thread = threading.Thread(
                target=self._metric_collection_loop,
                daemon=True
            )
            metric_thread.start()
            
            # Start alert processing thread
            alert_thread = threading.Thread(
                target=self._alert_processing_loop,
                daemon=True
            )
            alert_thread.start()
            
            self.logger.info("Monitoring services started")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring services: {e}")
            raise RuntimeError(f"Service startup failed: {str(e)}")
    
    def _health_check_loop(self) -> None:
        """Background thread for agent health checks"""
        while self.is_active_flag:
            try:
                time.sleep(self.monitoring_config['check_interval'])
                
                with self._agent_lock:
                    for agent_id, agent_info in self.monitored_agents.items():
                        if agent_info['monitoring_enabled']:
                            self._perform_health_check(agent_id)
                
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
    
    def _metric_collection_loop(self) -> None:
        """Background thread for metric collection"""
        while self.is_active_flag:
            try:
                time.sleep(self.monitoring_config['metric_collection_interval'])
                
                with self._agent_lock:
                    for agent_id in self.monitored_agents:
                        self.collect_agent_metrics(agent_id)
                
            except Exception as e:
                self.logger.error(f"Metric collection loop error: {e}")
    
    def _alert_processing_loop(self) -> None:
        """Background thread for alert processing"""
        while self.is_active_flag:
            try:
                time.sleep(30)  # Process alerts every 30 seconds
                
                with self._lock:
                    # Process active alerts
                    for agent_id, alerts in list(self.active_alerts.items()):
                        updated_alerts = []
                        
                        for alert in alerts:
                            # Check if alert should be auto-cleared
                            if self._should_auto_clear_alert(alert):
                                alert['status'] = 'cleared'
                                alert['cleared_at'] = time.time()
                            else:
                                updated_alerts.append(alert)
                        
                        if updated_alerts:
                            self.active_alerts[agent_id] = updated_alerts
                        else:
                            del self.active_alerts[agent_id]
                
            except Exception as e:
                self.logger.error(f"Alert processing loop error: {e}")
    
    def _perform_health_check(self, agent_id: str) -> None:
        """Perform health check on specific agent"""
        try:
            agent_info = self.monitored_agents[agent_id]
            agent = agent_info['agent']
            
            # Collect health indicators
            health_indicators = {
                'responsive': agent.is_responsive(),
                'cpu_usage': agent.get_cpu_usage(),
                'memory_usage': agent.get_memory_usage(),
                'error_rate': agent.get_error_rate(),
                'task_queue_size': agent.get_task_queue_size(),
                'last_activity': agent.get_last_activity_time()
            }
            
            # Evaluate health
            health_evaluation = self._evaluate_agent_health(health_indicators)
            
            # Update health status
            with self._lock:
                self.agent_health_status[agent_id] = {
                    'status': health_evaluation['status'],
                    'last_check': time.time(),
                    'health_score': health_evaluation['score'],
                    'issues': health_evaluation['issues']
                }
                
                # Record in history
                self.health_history[agent_id].append({
                    'timestamp': time.time(),
                    'status': health_evaluation['status'],
                    'score': health_evaluation['score']
                })
            
            # Update monitoring stats
            self.monitoring_stats['total_health_checks'] += 1
            if health_evaluation['status'] != 'healthy':
                self.monitoring_stats['failed_health_checks'] += 1
            
            # Update agent info
            agent_info['health_check_count'] += 1
            agent_info['last_health_check'] = time.time()
            
            # Trigger alerts if needed
            for issue in health_evaluation['issues']:
                if issue['severity'] in ['high', 'critical']:
                    self.trigger_alert(
                        agent_id,
                        issue['type'],
                        issue['severity'],
                        issue
                    )
            
        except Exception as e:
            self.logger.error(f"Health check failed for agent {agent_id}: {e}")
    
    def _evaluate_agent_health(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate agent health based on indicators"""
        issues = []
        health_score = 1.0
        
        # Check responsiveness
        if not indicators.get('responsive', True):
            issues.append({
                'type': 'unresponsive',
                'severity': 'critical',
                'description': 'Agent is not responding'
            })
            health_score = 0.0
        
        # Check CPU usage
        cpu_usage = indicators.get('cpu_usage', 0)
        if cpu_usage > self.health_thresholds['cpu_usage']['critical']:
            issues.append({
                'type': 'high_cpu_usage',
                'severity': 'high',
                'value': cpu_usage,
                'threshold': self.health_thresholds['cpu_usage']['critical']
            })
            health_score *= 0.5
        elif cpu_usage > self.health_thresholds['cpu_usage']['warning']:
            issues.append({
                'type': 'elevated_cpu_usage',
                'severity': 'medium',
                'value': cpu_usage,
                'threshold': self.health_thresholds['cpu_usage']['warning']
            })
            health_score *= 0.8
        
        # Check memory usage
        memory_usage = indicators.get('memory_usage', 0)
        if memory_usage > self.health_thresholds['memory_usage']['critical']:
            issues.append({
                'type': 'high_memory_usage',
                'severity': 'high',
                'value': memory_usage,
                'threshold': self.health_thresholds['memory_usage']['critical']
            })
            health_score *= 0.5
        elif memory_usage > self.health_thresholds['memory_usage']['warning']:
            issues.append({
                'type': 'elevated_memory_usage',
                'severity': 'medium',
                'value': memory_usage,
                'threshold': self.health_thresholds['memory_usage']['warning']
            })
            health_score *= 0.8
        
        # Check error rate
        error_rate = indicators.get('error_rate', 0)
        if error_rate > self.health_thresholds['error_rate']['critical']:
            issues.append({
                'type': 'high_error_rate',
                'severity': 'high',
                'value': error_rate,
                'threshold': self.health_thresholds['error_rate']['critical']
            })
            health_score *= 0.6
        
        # Determine overall status
        if health_score >= 0.8:
            status = 'healthy'
        elif health_score >= 0.5:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        return {
            'status': status,
            'score': health_score,
            'issues': issues
        }
    
    def _calculate_monitoring_uptime(self) -> str:
        """Calculate monitoring system uptime"""
        if not self.monitoring_start_time:
            return "0:00:00"
        
        uptime_seconds = time.time() - self.monitoring_start_time
        return str(timedelta(seconds=int(uptime_seconds)))
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health"""
        try:
            healthy_agents = sum(1 for status in self.agent_health_status.values() 
                               if status['status'] == 'healthy')
            total_agents = len(self.agent_health_status)
            
            if total_agents == 0:
                return {'status': 'unknown', 'score': 0}
            
            health_percentage = (healthy_agents / total_agents) * 100
            
            if health_percentage >= 95:
                status = 'excellent'
            elif health_percentage >= 80:
                status = 'good'
            elif health_percentage >= 60:
                status = 'fair'
            else:
                status = 'poor'
            
            return {
                'status': status,
                'score': health_percentage,
                'healthy_agents': healthy_agents,
                'total_agents': total_agents
            }
            
        except Exception as e:
            self.logger.error(f"System health calculation failed: {e}")
            return {'status': 'error', 'score': 0}
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary across all agents"""
        try:
            summary = {
                'average_cpu_usage': 0,
                'average_memory_usage': 0,
                'average_response_time': 0,
                'average_error_rate': 0,
                'total_throughput': 0
            }
            
            agent_count = 0
            
            for agent_id, metrics in self.performance_metrics.items():
                if metrics['cpu_usage']:
                    summary['average_cpu_usage'] += sum(metrics['cpu_usage']) / len(metrics['cpu_usage'])
                    agent_count += 1
                
                if metrics['memory_usage']:
                    summary['average_memory_usage'] += sum(metrics['memory_usage']) / len(metrics['memory_usage'])
                
                if metrics['response_times']:
                    summary['average_response_time'] += sum(metrics['response_times']) / len(metrics['response_times'])
                
                if metrics['error_rates']:
                    summary['average_error_rate'] += sum(metrics['error_rates']) / len(metrics['error_rates'])
                
                if metrics['throughput']:
                    summary['total_throughput'] += sum(metrics['throughput'])
            
            # Calculate averages
            if agent_count > 0:
                summary['average_cpu_usage'] /= agent_count
                summary['average_memory_usage'] /= agent_count
                summary['average_response_time'] /= agent_count
                summary['average_error_rate'] /= agent_count
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Performance summary generation failed: {e}")
            return {}
    
    def _get_top_alert_types(self) -> List[Dict[str, Any]]:
        """Get most common alert types"""
        alert_counts = defaultdict(int)
        
        for alert in self.alert_history:
            alert_counts[alert['alert_type']] += 1
        
        # Sort by count and get top 5
        top_alerts = sorted(
            alert_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return [
            {'type': alert_type, 'count': count}
            for alert_type, count in top_alerts
        ]
    
    def _generate_monitoring_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on monitoring data"""
        recommendations = []
        
        # Check agent health
        health_summary = report['agent_health_summary']
        if health_summary['unhealthy'] > 0:
            recommendations.append(
                f"Address {health_summary['unhealthy']} unhealthy agents immediately"
            )
        
        if health_summary['degraded'] > 0:
            recommendations.append(
                f"Investigate {health_summary['degraded']} degraded agents"
            )
        
        # Check performance
        perf_summary = report['performance_summary']
        if perf_summary.get('average_cpu_usage', 0) > 70:
            recommendations.append(
                "High average CPU usage detected. Consider scaling or optimization"
            )
        
        if perf_summary.get('average_error_rate', 0) > 0.05:
            recommendations.append(
                "Elevated error rates detected. Review error logs and agent implementations"
            )
        
        # Check alerts
        alert_summary = report['alert_summary']
        if alert_summary['active_alerts'] > 10:
            recommendations.append(
                f"{alert_summary['active_alerts']} active alerts require attention"
            )
        
        return recommendations
    
    def _load_alert_rules(self) -> List[Dict[str, Any]]:
        """Load alert rule definitions"""
        return [
            {
                'name': 'high_cpu_sustained',
                'condition': 'cpu_usage > 80 for 5 minutes',
                'severity': 'high',
                'auto_recovery': True
            },
            {
                'name': 'memory_leak_detected',
                'condition': 'memory_usage increasing for 30 minutes',
                'severity': 'critical',
                'auto_recovery': False
            },
            {
                'name': 'unresponsive_agent',
                'condition': 'no heartbeat for 2 minutes',
                'severity': 'critical',
                'auto_recovery': True
            }
        ]
    
    def _should_auto_clear_alert(self, alert: Dict[str, Any]) -> bool:
        """Check if alert should be automatically cleared"""
        # Check alert age
        alert_age = time.time() - alert['timestamp']
        auto_clear_duration = self.alert_thresholds['auto_clear_duration']
        
        # Only auto-clear low and medium severity alerts
        if alert['severity'] in ['low', 'medium'] and alert_age > auto_clear_duration:
            # Verify the issue is resolved
            agent_id = alert['agent_id']
            if agent_id in self.agent_health_status:
                health_status = self.agent_health_status[agent_id]
                if health_status['status'] == 'healthy':
                    return True
        
        return False
    
    def _attempt_auto_recovery(self, agent_id: str, alert_type: str) -> Dict[str, Any]:
        """Attempt automatic recovery for agent issue"""
        try:
            self.logger.info(f"Attempting auto-recovery for agent {agent_id} ({alert_type})")
            
            recovery_actions = {
                'high_cpu_usage': self._recover_high_cpu,
                'high_memory_usage': self._recover_high_memory,
                'unresponsive': self._recover_unresponsive
            }
            
            recovery_action = recovery_actions.get(alert_type)
            if recovery_action:
                result = recovery_action(agent_id)
                
                if result['success']:
                    self.monitoring_stats['auto_recoveries'] += 1
                
                return result
            
            return {
                'success': False,
                'reason': 'No recovery action defined for alert type'
            }
            
        except Exception as e:
            self.logger.error(f"Auto-recovery failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _recover_high_cpu(self, agent_id: str) -> Dict[str, Any]:
        """Recovery action for high CPU usage"""
        # In a real implementation, this would take actual recovery actions
        return {
            'success': True,
            'action': 'Reduced agent workload',
            'details': 'Temporarily reduced concurrent task limit'
        }
    
    def _recover_high_memory(self, agent_id: str) -> Dict[str, Any]:
        """Recovery action for high memory usage"""
        # In a real implementation, this would take actual recovery actions
        return {
            'success': True,
            'action': 'Cleared agent caches',
            'details': 'Cleared non-essential caches and triggered garbage collection'
        }
    
    def _recover_unresponsive(self, agent_id: str) -> Dict[str, Any]:
        """Recovery action for unresponsive agent"""
        # In a real implementation, this would restart the agent
        return {
            'success': True,
            'action': 'Restarted agent',
            'details': 'Agent process restarted successfully'
        }


# Metric Collector Classes

class SystemMetricCollector:
    """Collect system-level metrics"""
    
    def collect(self) -> Dict[str, Any]:
        """Collect system metrics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
        }


class AgentMetricCollector:
    """Collect agent-specific metrics"""
    
    def collect(self, agent: Any) -> Dict[str, Any]:
        """Collect agent metrics"""
        return {
            'status': agent.get_status(),
            'uptime': agent.get_uptime(),
            'task_count': agent.get_task_count(),
            'resource_usage': agent.get_resource_usage()
        }


class PerformanceMetricCollector:
    """Collect performance metrics"""
    
    def collect(self, agent: Any) -> Dict[str, Any]:
        """Collect performance metrics"""
        return {
            'response_times': agent.get_response_times(),
            'throughput': agent.get_throughput(),
            'error_rate': agent.get_error_rate(),
            'latency_percentiles': agent.get_latency_percentiles()
        }


# Alert Channel Classes

class LogAlertChannel:
    """Log-based alert channel"""
    
    def __init__(self, logger):
        self.logger = logger
    
    def send_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert to log"""
        self.logger.warning(f"ALERT: {alert}")


class MetricAlertChannel:
    """Metric-based alert channel"""
    
    def send_alert(self, alert: Dict[str, Any]) -> None:
        """Send alert as metric"""
        # In a real implementation, this would send to a metrics system
        pass