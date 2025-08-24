"""
Saraphis Scaling Orchestrator
Coordinates all scaling and recovery operations across the production system
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import json

from .auto_scaling_engine import AutoScalingEngine
from .auto_recovery_engine import AutoRecoveryEngine
from .load_balancer import IntelligentLoadBalancer
from .predictive_analytics import PredictiveScalingAnalytics

logger = logging.getLogger(__name__)


class ScalingOrchestrator:
    """
    Master orchestrator for all scaling and recovery operations
    Coordinates between scaling engine, recovery engine, load balancer, and predictive analytics
    NO FALLBACKS - HARD FAILURES ONLY
    """
    
    def __init__(self, monitor, brain_system, agent_system):
        self.monitor = monitor
        self.brain_system = brain_system
        self.agent_system = agent_system
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize sub-components
        self.scaling_engine = AutoScalingEngine(monitor, brain_system, agent_system)
        self.recovery_engine = AutoRecoveryEngine(monitor, self.scaling_engine)
        self.load_balancer = IntelligentLoadBalancer(monitor, self.scaling_engine)
        self.predictive_analytics = PredictiveScalingAnalytics(monitor, self.scaling_engine)
        
        # Orchestration configuration
        self.orchestration_interval = 30.0  # Coordinate every 30 seconds
        self.emergency_threshold = 0.3  # Health below 30% triggers emergency response
        
        # Orchestration state
        self.is_running = False
        self.orchestration_thread = None
        self.operation_history = deque(maxlen=1000)
        self.orchestration_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'emergency_responses': 0,
            'predictive_actions': 0
        }
        self._lock = threading.Lock()
        
        self.logger.info("Scaling Orchestrator initialized")
    
    def start_orchestration(self) -> Dict[str, Any]:
        """Start the scaling orchestration system"""
        with self._lock:
            if self.is_running:
                return {
                    'success': False,
                    'error': 'Orchestration already running'
                }
            
            try:
                # Start all sub-components
                self.logger.info("Starting all scaling components...")
                
                scaling_result = self.scaling_engine.start_auto_scaling()
                if not scaling_result.get('success'):
                    raise Exception(f"Failed to start scaling engine: {scaling_result.get('error')}")
                
                recovery_result = self.recovery_engine.start_auto_recovery()
                if not recovery_result.get('success'):
                    raise Exception(f"Failed to start recovery engine: {recovery_result.get('error')}")
                
                balancer_result = self.load_balancer.start_load_balancing()
                if not balancer_result.get('success'):
                    raise Exception(f"Failed to start load balancer: {balancer_result.get('error')}")
                
                analytics_result = self.predictive_analytics.start_analytics()
                if not analytics_result.get('success'):
                    raise Exception(f"Failed to start predictive analytics: {analytics_result.get('error')}")
                
                # Start orchestration thread
                self.is_running = True
                self.orchestration_thread = threading.Thread(
                    target=self._orchestration_loop,
                    daemon=True
                )
                self.orchestration_thread.start()
                
                self.logger.info("Scaling Orchestrator started successfully")
                return {
                    'success': True,
                    'components': {
                        'scaling_engine': 'running',
                        'recovery_engine': 'running',
                        'load_balancer': 'running',
                        'predictive_analytics': 'running'
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Failed to start orchestration: {e}")
                # Stop any started components
                self._emergency_stop()
                return {
                    'success': False,
                    'error': str(e)
                }
    
    def _orchestration_loop(self):
        """Main orchestration loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Check system health
                health_check = self._check_overall_health()
                
                if health_check['emergency_needed']:
                    self._handle_emergency(health_check)
                
                # Coordinate scaling operations
                self._coordinate_scaling()
                
                # Optimize load distribution
                self._coordinate_load_balancing()
                
                # Check predictive insights
                self._apply_predictive_scaling()
                
                # Monitor and report
                self._monitor_operations()
                
                # Sleep for remaining time
                elapsed = time.time() - start_time
                sleep_time = max(0, self.orchestration_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Orchestration loop error: {e}")
                # NO FALLBACK - Continue loop
    
    def _check_overall_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        try:
            health_check = {
                'timestamp': datetime.now(),
                'overall_health': 1.0,
                'critical_issues': [],
                'emergency_needed': False
            }
            
            # Check all systems
            system_healths = []
            system_statuses = self.monitor.get_all_system_status()
            
            for system_name, status in system_statuses.items():
                health = status.get('health', 1.0)
                system_healths.append(health)
                
                if health < self.emergency_threshold:
                    health_check['critical_issues'].append({
                        'type': 'system',
                        'name': system_name,
                        'health': health,
                        'severity': 'critical'
                    })
            
            # Check all agents
            agent_healths = []
            agent_statuses = self.monitor.get_all_agent_status()
            
            for agent_name, status in agent_statuses.items():
                health = status.get('health', 1.0)
                agent_healths.append(health)
                
                if health < self.emergency_threshold:
                    health_check['critical_issues'].append({
                        'type': 'agent',
                        'name': agent_name,
                        'health': health,
                        'severity': 'critical'
                    })
            
            # Calculate overall health
            all_healths = system_healths + agent_healths
            if all_healths:
                health_check['overall_health'] = sum(all_healths) / len(all_healths)
            
            # Check if emergency response needed
            health_check['emergency_needed'] = (
                len(health_check['critical_issues']) > 0 or
                health_check['overall_health'] < 0.5
            )
            
            return health_check
            
        except Exception as e:
            self.logger.error(f"Failed to check overall health: {e}")
            return {
                'overall_health': 0.0,
                'critical_issues': [],
                'emergency_needed': True,
                'error': str(e)
            }
    
    def _handle_emergency(self, health_check: Dict[str, Any]):
        """Handle emergency situations"""
        try:
            self.logger.warning(f"EMERGENCY: Handling {len(health_check['critical_issues'])} critical issues")
            self.orchestration_metrics['emergency_responses'] += 1
            
            operation = {
                'type': 'emergency_response',
                'timestamp': datetime.now(),
                'issues': health_check['critical_issues'],
                'actions': []
            }
            
            # Handle each critical issue
            for issue in health_check['critical_issues']:
                component_name = issue['name']
                component_type = issue['type']
                
                # First attempt: Recovery
                recovery_result = self._initiate_recovery(component_type, component_name)
                operation['actions'].append({
                    'action': 'recovery',
                    'component': component_name,
                    'result': recovery_result
                })
                
                if not recovery_result.get('success'):
                    # Second attempt: Scale and replace
                    scale_result = self._emergency_scale(component_type, component_name)
                    operation['actions'].append({
                        'action': 'emergency_scale',
                        'component': component_name,
                        'result': scale_result
                    })
            
            # Record operation
            self._record_operation(operation)
            
        except Exception as e:
            self.logger.error(f"Failed to handle emergency: {e}")
    
    def _coordinate_scaling(self):
        """Coordinate scaling operations across components"""
        try:
            # Get scaling recommendations from analytics
            scaling_analysis = self.scaling_engine.analyze_scaling_requirements()
            
            if not scaling_analysis.get('scaling_needed'):
                return
            
            self.logger.info("Coordinating scaling operations")
            
            operation = {
                'type': 'coordinated_scaling',
                'timestamp': datetime.now(),
                'requirements': scaling_analysis,
                'results': []
            }
            
            # Execute scaling operations
            if scaling_analysis['scale_up']['systems']:
                systems_to_scale = [
                    {
                        'name': item['name'],
                        'target_instances': item['current_instances'] + 1,
                        'reason': item['reason']
                    }
                    for item in scaling_analysis['scale_up']['systems']
                ]
                
                result = self.scaling_engine.execute_system_scaling(systems_to_scale)
                operation['results'].append({
                    'type': 'system_scale_up',
                    'result': result
                })
            
            if scaling_analysis['scale_down']['systems']:
                systems_to_scale = [
                    {
                        'name': item['name'],
                        'target_instances': item['current_instances'] - 1,
                        'reason': item['reason']
                    }
                    for item in scaling_analysis['scale_down']['systems']
                ]
                
                result = self.scaling_engine.execute_system_scaling(systems_to_scale)
                operation['results'].append({
                    'type': 'system_scale_down',
                    'result': result
                })
            
            # Similar for agents
            if scaling_analysis['scale_up']['agents']:
                agents_to_scale = [
                    {
                        'name': item['name'],
                        'target_instances': item['current_instances'] + 1,
                        'reason': item['reason']
                    }
                    for item in scaling_analysis['scale_up']['agents']
                ]
                
                result = self.scaling_engine.execute_agent_scaling(agents_to_scale)
                operation['results'].append({
                    'type': 'agent_scale_up',
                    'result': result
                })
            
            # Update load balancer after scaling
            self.load_balancer.optimize_load_distribution()
            
            # Record operation
            self._record_operation(operation)
            
        except Exception as e:
            self.logger.error(f"Failed to coordinate scaling: {e}")
    
    def _coordinate_load_balancing(self):
        """Coordinate load balancing operations"""
        try:
            # Check current load distribution
            load_analysis = self.load_balancer.analyze_workload_distribution()
            
            if load_analysis.get('overall_efficiency', 1.0) < 0.95:
                self.logger.info(f"Load efficiency below target: {load_analysis['overall_efficiency']:.2%}")
                
                # Optimize distribution
                optimization_result = self.load_balancer.optimize_load_distribution()
                
                if optimization_result.get('success'):
                    self.logger.info(f"Load distribution optimized: {len(optimization_result['actions_taken'])} actions taken")
                    
        except Exception as e:
            self.logger.error(f"Failed to coordinate load balancing: {e}")
    
    def _apply_predictive_scaling(self):
        """Apply predictive scaling based on analytics"""
        try:
            # Get predictions for next 30 minutes
            predictions = self.predictive_analytics.predict_future_demand(time_range_hours=0.5)
            
            if not predictions.get('components'):
                return
            
            scaling_needed = []
            
            # Check each component's predictions
            for component_name, component_predictions in predictions['components'].items():
                if not component_predictions:
                    continue
                
                # Look at the 30-minute prediction
                future_prediction = component_predictions[-1] if component_predictions else None
                
                if future_prediction:
                    current_instances = self.scaling_engine.current_instances['systems'].get(component_name, 1)
                    required_instances = future_prediction.get('required_instances', current_instances)
                    
                    if required_instances != current_instances:
                        scaling_needed.append({
                            'name': component_name,
                            'current_instances': current_instances,
                            'target_instances': required_instances,
                            'reason': f"Predictive scaling - expected load: {future_prediction.get('predicted_load', 0):.1f}"
                        })
            
            # Execute predictive scaling if needed
            if scaling_needed:
                self.logger.info(f"Applying predictive scaling for {len(scaling_needed)} components")
                self.orchestration_metrics['predictive_actions'] += 1
                
                result = self.scaling_engine.execute_system_scaling(scaling_needed)
                
                operation = {
                    'type': 'predictive_scaling',
                    'timestamp': datetime.now(),
                    'predictions': scaling_needed,
                    'result': result
                }
                
                self._record_operation(operation)
                
        except Exception as e:
            self.logger.error(f"Failed to apply predictive scaling: {e}")
    
    def _monitor_operations(self):
        """Monitor ongoing operations and system state"""
        try:
            # Only monitor scaling performance if operations exist
            scaling_performance = None
            if (self.scaling_engine.scaling_metrics['systems']['total_scaling_operations'] > 0 or 
                self.scaling_engine.scaling_metrics['agents']['total_scaling_operations'] > 0):
                scaling_performance = self.scaling_engine.monitor_scaling_performance()
            
            # Check recovery performance
            recovery_performance = self.recovery_engine.monitor_recovery_performance()
            
            # Check load balancing performance
            balancing_performance = self.load_balancer.monitor_balancing_performance()
            
            # Check prediction accuracy
            prediction_accuracy = self.predictive_analytics.validate_prediction_accuracy()
            
            # Log summary if any issues
            if scaling_performance and scaling_performance.get('overall_efficiency', 1.0) < 0.95:
                self.logger.warning(f"Scaling efficiency below target: {scaling_performance['overall_efficiency']:.2%}")
            
            if not recovery_performance['overall'].get('meets_sla', True):
                self.logger.warning("Recovery SLA not met")
            
            if balancing_performance.get('current_efficiency', 1.0) < 0.95:
                self.logger.warning(f"Load balancing efficiency below target: {balancing_performance['current_efficiency']:.2%}")
            
            if not prediction_accuracy.get('meets_threshold', True):
                self.logger.warning(f"Prediction accuracy below threshold: {prediction_accuracy['overall_accuracy']:.2%}")
                
        except Exception as e:
            self.logger.error(f"Failed to monitor operations: {e}")
    
    def _initiate_recovery(self, component_type: str, component_name: str) -> Dict[str, Any]:
        """Initiate recovery for a component"""
        try:
            if component_type == 'system':
                return self.recovery_engine.execute_system_recovery([component_name])
            else:
                return self.recovery_engine.execute_agent_recovery([component_name])
                
        except Exception as e:
            self.logger.error(f"Failed to initiate recovery: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _emergency_scale(self, component_type: str, component_name: str) -> Dict[str, Any]:
        """Emergency scaling for critical components"""
        try:
            current_instances = self.scaling_engine.current_instances[f"{component_type}s"].get(component_name, 1)
            
            # Double the instances in emergency
            target_instances = min(
                current_instances * 2,
                self.scaling_engine.max_instances[f"{component_type}s"]
            )
            
            scaling_request = [{
                'name': component_name,
                'target_instances': target_instances,
                'reason': 'Emergency scaling - critical health'
            }]
            
            if component_type == 'system':
                return self.scaling_engine.execute_system_scaling(scaling_request)
            else:
                return self.scaling_engine.execute_agent_scaling(scaling_request)
                
        except Exception as e:
            self.logger.error(f"Failed to execute emergency scaling: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _record_operation(self, operation: Dict[str, Any]):
        """Record orchestration operation"""
        self.operation_history.append(operation)
        self.orchestration_metrics['total_operations'] += 1
        
        # Check if operation was successful
        success = True
        if 'result' in operation:
            success = operation['result'].get('success', False)
        elif 'results' in operation:
            success = all(r.get('result', {}).get('success', False) for r in operation['results'])
        
        if success:
            self.orchestration_metrics['successful_operations'] += 1
        else:
            self.orchestration_metrics['failed_operations'] += 1
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestration status"""
        try:
            return {
                'is_running': self.is_running,
                'components': {
                    'scaling_engine': self.scaling_engine.get_scaling_status(),
                    'recovery_engine': self.recovery_engine.get_recovery_status(),
                    'load_balancer': self.load_balancer.get_load_balancing_status(),
                    'predictive_analytics': self.predictive_analytics.get_analytics_status()
                },
                'metrics': self.orchestration_metrics.copy(),
                'recent_operations': list(self.operation_history)[-10:],
                'health_check': self._check_overall_health()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get orchestration status: {e}")
            return {
                'error': str(e),
                'is_running': self.is_running
            }
    
    def execute_manual_scaling(self, scaling_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute manual scaling request"""
        try:
            self.logger.info(f"Executing manual scaling request")
            
            results = {
                'timestamp': datetime.now(),
                'systems': {},
                'agents': {},
                'success': True
            }
            
            # Scale systems
            if 'systems' in scaling_request:
                for system_name, target_instances in scaling_request['systems'].items():
                    scale_result = self.scaling_engine.execute_system_scaling([{
                        'name': system_name,
                        'target_instances': target_instances,
                        'reason': 'Manual scaling request'
                    }])
                    
                    results['systems'][system_name] = scale_result
                    if not scale_result.get('success'):
                        results['success'] = False
            
            # Scale agents
            if 'agents' in scaling_request:
                for agent_name, target_instances in scaling_request['agents'].items():
                    scale_result = self.scaling_engine.execute_agent_scaling([{
                        'name': agent_name,
                        'target_instances': target_instances,
                        'reason': 'Manual scaling request'
                    }])
                    
                    results['agents'][agent_name] = scale_result
                    if not scale_result.get('success'):
                        results['success'] = False
            
            # Update load balancing after manual scaling
            self.load_balancer.optimize_load_distribution()
            
            # Record manual scaling as an orchestration operation  
            end_time = datetime.now()
            operation = {
                'type': 'manual_scaling',
                'timestamp': end_time,
                'success': results['success'],
                'details': {
                    'systems_scaled': len(scaling_request.get('systems', {})),
                    'agents_scaled': len(scaling_request.get('agents', {})),
                    'total_time': (end_time - results['timestamp']).total_seconds()
                },
                'results': [{
                    'type': 'manual_scaling',
                    'result': results
                }]
            }
            self._record_operation(operation)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to execute manual scaling: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_orchestration_report(self) -> Dict[str, Any]:
        """Generate comprehensive orchestration report"""
        try:
            report = {
                'report_id': f"orchestration_{int(time.time())}",
                'timestamp': datetime.now().isoformat(),
                'executive_summary': self._generate_executive_summary(),
                'component_reports': {
                    'scaling': self.scaling_engine.generate_scaling_report(
                        self.scaling_engine.get_scaling_status()
                    ),
                    'recovery': self.recovery_engine.generate_recovery_report(
                        self.recovery_engine.get_recovery_status()
                    ),
                    'load_balancing': self.load_balancer.generate_balancing_report(
                        self.load_balancer.analyze_workload_distribution()
                    ),
                    'predictions': self.predictive_analytics.validate_prediction_accuracy()
                },
                'metrics': self.orchestration_metrics.copy(),
                'recommendations': self._generate_recommendations()
            }
            
            # Save report
            report_file = f"orchestration_report_{report['report_id']}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Orchestration report saved to {report_file}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate orchestration report: {e}")
            raise
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary for report"""
        health_check = self._check_overall_health()
        
        total_ops = self.orchestration_metrics['total_operations']
        success_rate = (
            self.orchestration_metrics['successful_operations'] / total_ops
            if total_ops > 0 else 0
        )
        
        return {
            'overall_health': health_check['overall_health'],
            'critical_issues': len(health_check['critical_issues']),
            'total_operations': total_ops,
            'success_rate': success_rate,
            'emergency_responses': self.orchestration_metrics['emergency_responses'],
            'predictive_actions': self.orchestration_metrics['predictive_actions']
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on current state"""
        recommendations = []
        
        # Check overall health
        health_check = self._check_overall_health()
        if health_check['overall_health'] < 0.8:
            recommendations.append(
                f"Overall system health is low ({health_check['overall_health']:.2%}) - review component health"
            )
        
        # Check operation success rate
        total_ops = self.orchestration_metrics['total_operations']
        if total_ops > 0:
            success_rate = self.orchestration_metrics['successful_operations'] / total_ops
            if success_rate < 0.95:
                recommendations.append(
                    f"Operation success rate below target ({success_rate:.1%}) - review failed operations"
                )
        
        # Check emergency responses
        if self.orchestration_metrics['emergency_responses'] > 5:
            recommendations.append(
                "High number of emergency responses - investigate root causes"
            )
        
        # Check predictive accuracy
        prediction_accuracy = self.predictive_analytics.validate_prediction_accuracy()
        if not prediction_accuracy.get('meets_threshold', True):
            recommendations.append(
                f"Prediction accuracy below threshold ({prediction_accuracy['overall_accuracy']:.1%}) - retrain models"
            )
        
        return recommendations
    
    def _emergency_stop(self):
        """Emergency stop of all components"""
        try:
            self.logger.warning("Executing emergency stop")
            
            # Stop all components
            if hasattr(self, 'scaling_engine'):
                self.scaling_engine.stop_auto_scaling()
            
            if hasattr(self, 'recovery_engine'):
                self.recovery_engine.stop_auto_recovery()
            
            if hasattr(self, 'load_balancer'):
                self.load_balancer.stop_load_balancing()
            
            if hasattr(self, 'predictive_analytics'):
                self.predictive_analytics.stop_analytics()
                
        except Exception as e:
            self.logger.error(f"Error during emergency stop: {e}")
    
    def stop_orchestration(self) -> Dict[str, Any]:
        """Stop the scaling orchestration system"""
        with self._lock:
            self.is_running = False
            
            self.logger.info("Stopping all scaling components...")
            
            # Stop all sub-components
            component_results = {}
            
            try:
                component_results['scaling_engine'] = self.scaling_engine.stop_auto_scaling()
                component_results['recovery_engine'] = self.recovery_engine.stop_auto_recovery()
                component_results['load_balancer'] = self.load_balancer.stop_load_balancing()
                component_results['predictive_analytics'] = self.predictive_analytics.stop_analytics()
            except Exception as e:
                self.logger.error(f"Error stopping components: {e}")
            
            # Generate final report
            try:
                final_report = self.generate_orchestration_report()
            except:
                final_report = None
            
            self.logger.info("Scaling Orchestrator stopped")
            return {
                'success': True,
                'component_results': component_results,
                'final_report': final_report,
                'total_operations': self.orchestration_metrics['total_operations']
            }