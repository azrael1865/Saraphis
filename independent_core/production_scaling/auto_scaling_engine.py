"""
Saraphis Auto-Scaling Engine
Intelligent auto-scaling for all 11 systems and 8 agents based on demand
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import statistics
import math

logger = logging.getLogger(__name__)


class AutoScalingEngine:
    """
    Intelligent auto-scaling engine for Saraphis production
    Scales within 10 seconds decision time
    NO FALLBACKS - HARD FAILURES ONLY
    """
    
    def __init__(self, monitor, brain_system, agent_system):
        self.monitor = monitor
        self.brain_system = brain_system
        self.agent_system = agent_system
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Scaling configuration
        self.max_decision_time = 10.0  # 10 seconds max
        self.scale_check_interval = 5.0  # Check every 5 seconds
        self.min_instances = {
            'systems': 1,
            'agents': 1
        }
        self.max_instances = {
            'systems': 10,
            'agents': 20
        }
        
        # Scaling thresholds
        self.scale_up_thresholds = {
            'cpu_percent': 80,
            'memory_percent': 85,
            'response_time_ms': 500,
            'error_rate': 0.05,
            'queue_length': 100
        }
        self.scale_down_thresholds = {
            'cpu_percent': 30,
            'memory_percent': 40,
            'response_time_ms': 100,
            'error_rate': 0.01,
            'queue_length': 10
        }
        
        # Scaling state
        self.current_instances = {
            'systems': defaultdict(lambda: 1),
            'agents': defaultdict(lambda: 1)
        }
        self.scaling_history = deque(maxlen=10000)
        self.scaling_metrics = defaultdict(lambda: {
            'total_scaling_operations': 0,
            'successful_scaling': 0,
            'failed_scaling': 0,
            'average_scaling_time': 0.0,
            'total_scale_ups': 0,
            'total_scale_downs': 0
        })
        
        # Predictive scaling
        self.workload_history = defaultdict(lambda: deque(maxlen=1440))  # 24 hours at 1min intervals
        self.scaling_predictions = {}
        
        # System state
        self.is_running = False
        self.scaling_thread = None
        self.executor = ThreadPoolExecutor(max_workers=5)
        self._lock = threading.Lock()
        
        self.logger.info("Auto-Scaling Engine initialized")
    
    def start_auto_scaling(self) -> Dict[str, Any]:
        """Start the auto-scaling engine"""
        with self._lock:
            if self.is_running:
                return {
                    'success': False,
                    'error': 'Auto-scaling already running'
                }
            
            self.is_running = True
            self.scaling_thread = threading.Thread(
                target=self._scaling_loop,
                daemon=True
            )
            self.scaling_thread.start()
            
            self.logger.info("Auto-Scaling Engine started")
            return {
                'success': True,
                'check_interval': self.scale_check_interval,
                'max_decision_time': self.max_decision_time
            }
    
    def _scaling_loop(self):
        """Main scaling loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Analyze scaling requirements
                requirements = self.analyze_scaling_requirements()
                
                if requirements.get('scaling_needed'):
                    # Make scaling decision within time limit
                    future = self.executor.submit(self._execute_scaling, requirements)
                    
                    try:
                        result = future.result(timeout=self.max_decision_time)
                        decision_time = time.time() - start_time
                        
                        if decision_time > self.max_decision_time:
                            self.logger.error(f"Scaling decision exceeded time limit: {decision_time:.2f}s")
                            
                    except TimeoutError:
                        self.logger.error("Scaling decision timeout - HARD FAILURE")
                        raise
                
                # Update workload history
                self._update_workload_history()
                
                time.sleep(self.scale_check_interval)
                
            except Exception as e:
                self.logger.error(f"Scaling loop HARD FAILURE: {e}")
                # HARD FAILURE - NO FALLBACKS
                raise
    
    def analyze_scaling_requirements(self) -> Dict[str, Any]:
        """Analyze current scaling requirements"""
        try:
            requirements = {
                'timestamp': time.time(),
                'scaling_needed': False,
                'scale_up': defaultdict(list),
                'scale_down': defaultdict(list),
                'current_load': {}
            }
            
            # Analyze system requirements
            system_statuses = self.monitor.get_all_system_status()
            for system_name, status in system_statuses.items():
                metrics = self._extract_scaling_metrics(status)
                requirements['current_load'][system_name] = metrics
                
                # Check if scaling is needed
                if self._needs_scale_up(metrics):
                    requirements['scale_up']['systems'].append({
                        'name': system_name,
                        'current_instances': self.current_instances['systems'][system_name],
                        'metrics': metrics,
                        'reason': self._get_scale_up_reason(metrics)
                    })
                    requirements['scaling_needed'] = True
                    
                elif self._needs_scale_down(metrics):
                    if self.current_instances['systems'][system_name] > self.min_instances['systems']:
                        requirements['scale_down']['systems'].append({
                            'name': system_name,
                            'current_instances': self.current_instances['systems'][system_name],
                            'metrics': metrics,
                            'reason': self._get_scale_down_reason(metrics)
                        })
                        requirements['scaling_needed'] = True
            
            # Analyze agent requirements
            agent_statuses = self.monitor.get_all_agent_status()
            for agent_name, status in agent_statuses.items():
                metrics = self._extract_agent_metrics(status)
                requirements['current_load'][agent_name] = metrics
                
                # Check if scaling is needed
                if self._needs_agent_scale_up(metrics):
                    requirements['scale_up']['agents'].append({
                        'name': agent_name,
                        'current_instances': self.current_instances['agents'][agent_name],
                        'metrics': metrics,
                        'reason': 'High task load or response time'
                    })
                    requirements['scaling_needed'] = True
                    
                elif self._needs_agent_scale_down(metrics):
                    if self.current_instances['agents'][agent_name] > self.min_instances['agents']:
                        requirements['scale_down']['agents'].append({
                            'name': agent_name,
                            'current_instances': self.current_instances['agents'][agent_name],
                            'metrics': metrics,
                            'reason': 'Low task load'
                        })
                        requirements['scaling_needed'] = True
            
            return requirements
            
        except Exception as e:
            self.logger.error(f"Failed to analyze scaling requirements: {e}")
            raise
    
    def _extract_scaling_metrics(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """Extract scaling-relevant metrics from status"""
        performance = status.get('performance', {})
        resources = status.get('resources', {})
        
        return {
            'cpu_percent': resources.get('cpu_percent', 0),
            'memory_percent': resources.get('memory_percent', 0),
            'response_time_ms': performance.get('response_time_ms', 0),
            'error_rate': performance.get('error_rate', 0),
            'throughput': performance.get('throughput', 0),
            'health_score': status.get('health', 1.0)
        }
    
    def _extract_agent_metrics(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """Extract agent-specific metrics"""
        return {
            'active_tasks': status.get('active_tasks', 0),
            'completed_tasks': status.get('completed_tasks', 0),
            'task_success_rate': status.get('task_success_rate', 1.0),
            'response_time': status.get('response_time', 0),
            'health_score': status.get('health', 1.0)
        }
    
    def _needs_scale_up(self, metrics: Dict[str, Any]) -> bool:
        """Check if system needs scaling up"""
        return (
            metrics['cpu_percent'] > self.scale_up_thresholds['cpu_percent'] or
            metrics['memory_percent'] > self.scale_up_thresholds['memory_percent'] or
            metrics['response_time_ms'] > self.scale_up_thresholds['response_time_ms'] or
            metrics['error_rate'] > self.scale_up_thresholds['error_rate']
        )
    
    def _needs_scale_down(self, metrics: Dict[str, Any]) -> bool:
        """Check if system can scale down"""
        return (
            metrics['cpu_percent'] < self.scale_down_thresholds['cpu_percent'] and
            metrics['memory_percent'] < self.scale_down_thresholds['memory_percent'] and
            metrics['response_time_ms'] < self.scale_down_thresholds['response_time_ms'] and
            metrics['error_rate'] < self.scale_down_thresholds['error_rate']
        )
    
    def _needs_agent_scale_up(self, metrics: Dict[str, Any]) -> bool:
        """Check if agent needs scaling up"""
        return (
            metrics['active_tasks'] > 10 or
            metrics['response_time'] > 200 or
            metrics['task_success_rate'] < 0.95
        )
    
    def _needs_agent_scale_down(self, metrics: Dict[str, Any]) -> bool:
        """Check if agent can scale down"""
        return (
            metrics['active_tasks'] < 2 and
            metrics['response_time'] < 50 and
            metrics['task_success_rate'] > 0.99
        )
    
    def _get_scale_up_reason(self, metrics: Dict[str, Any]) -> str:
        """Get reason for scaling up"""
        reasons = []
        if metrics['cpu_percent'] > self.scale_up_thresholds['cpu_percent']:
            reasons.append(f"High CPU: {metrics['cpu_percent']:.1f}%")
        if metrics['memory_percent'] > self.scale_up_thresholds['memory_percent']:
            reasons.append(f"High Memory: {metrics['memory_percent']:.1f}%")
        if metrics['response_time_ms'] > self.scale_up_thresholds['response_time_ms']:
            reasons.append(f"High Response Time: {metrics['response_time_ms']:.0f}ms")
        if metrics['error_rate'] > self.scale_up_thresholds['error_rate']:
            reasons.append(f"High Error Rate: {metrics['error_rate']:.1%}")
        return ", ".join(reasons)
    
    def _get_scale_down_reason(self, metrics: Dict[str, Any]) -> str:
        """Get reason for scaling down"""
        return f"Low utilization - CPU: {metrics['cpu_percent']:.1f}%, Memory: {metrics['memory_percent']:.1f}%"
    
    def _execute_scaling(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scaling operations"""
        start_time = time.time()
        results = {
            'success': True,
            'timestamp': datetime.now(),
            'operations': [],
            'errors': []
        }
        
        try:
            # Execute scale-up operations
            for component_type in ['systems', 'agents']:
                for item in requirements['scale_up'][component_type]:
                    result = self._scale_up_component(component_type, item)
                    results['operations'].append(result)
                    if not result['success']:
                        results['success'] = False
                        results['errors'].append(result['error'])
            
            # Execute scale-down operations
            for component_type in ['systems', 'agents']:
                for item in requirements['scale_down'][component_type]:
                    result = self._scale_down_component(component_type, item)
                    results['operations'].append(result)
                    if not result['success']:
                        results['success'] = False
                        results['errors'].append(result['error'])
            
            # Record scaling operation
            scaling_time = time.time() - start_time
            self._record_scaling_operation(requirements, results, scaling_time)
            
            # Validate scaling impact
            if results['success']:
                validation = self.validate_scaling_impact(results['operations'])
                results['validation'] = validation
            
            return results
            
        except Exception as e:
            self.logger.error(f"HARD FAILURE in scaling execution: {e}")
            # NO FALLBACKS - HARD FAILURES ONLY
            raise
    
    def _scale_up_component(self, component_type: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """Scale up a single component"""
        try:
            name = item['name']
            current = item['current_instances']
            max_allowed = self.max_instances[component_type]
            
            if current >= max_allowed:
                return {
                    'success': False,
                    'component': name,
                    'type': component_type,
                    'action': 'scale_up',
                    'error': f'Already at maximum instances: {max_allowed}'
                }
            
            # Calculate new instance count
            new_count = min(current + 1, max_allowed)
            
            self.logger.info(f"Scaling up {component_type} {name}: {current} -> {new_count}")
            
            # Simulate scaling operation
            # In production, this would create new instances
            time.sleep(2.0)  # Simulate scaling time
            
            # Update instance count
            self.current_instances[component_type][name] = new_count
            
            return {
                'success': True,
                'component': name,
                'type': component_type,
                'action': 'scale_up',
                'previous_instances': current,
                'new_instances': new_count,
                'reason': item['reason']
            }
            
        except Exception as e:
            self.logger.error(f"HARD FAILURE scaling up {item['name']}: {e}")
            # NO FALLBACKS - HARD FAILURES ONLY
            raise
    
    def _scale_down_component(self, component_type: str, item: Dict[str, Any]) -> Dict[str, Any]:
        """Scale down a single component"""
        try:
            name = item['name']
            current = item['current_instances']
            min_allowed = self.min_instances[component_type]
            
            if current <= min_allowed:
                return {
                    'success': False,
                    'component': name,
                    'type': component_type,
                    'action': 'scale_down',
                    'error': f'Already at minimum instances: {min_allowed}'
                }
            
            # Calculate new instance count
            new_count = max(current - 1, min_allowed)
            
            self.logger.info(f"Scaling down {component_type} {name}: {current} -> {new_count}")
            
            # Simulate scaling operation
            # In production, this would terminate instances gracefully
            time.sleep(1.0)  # Simulate scaling time
            
            # Update instance count
            self.current_instances[component_type][name] = new_count
            
            return {
                'success': True,
                'component': name,
                'type': component_type,
                'action': 'scale_down',
                'previous_instances': current,
                'new_instances': new_count,
                'reason': item['reason']
            }
            
        except Exception as e:
            self.logger.error(f"HARD FAILURE scaling down {item['name']}: {e}")
            # NO FALLBACKS - HARD FAILURES ONLY
            raise
    
    def predict_scaling_needs(self, time_horizon_minutes: int = 30) -> Dict[str, Any]:
        """Predict future scaling requirements"""
        try:
            predictions = {
                'timestamp': datetime.now(),
                'time_horizon_minutes': time_horizon_minutes,
                'predictions': defaultdict(dict)
            }
            
            # Analyze historical patterns for each component
            for component_type in ['systems', 'agents']:
                for name, history in self.workload_history.items():
                    if len(history) < 60:  # Need at least 1 hour of data
                        # HARD FAILURE - insufficient data for predictions
                        raise ValueError(f"Insufficient workload history for {name}: {len(history)} < 60 required")
                    
                    # Extract workload pattern
                    recent_loads = [h['load'] for h in list(history)[-60:]]
                    trend = self._calculate_trend(recent_loads)
                    
                    # Predict future load
                    current_load = recent_loads[-1] if recent_loads else 0
                    predicted_load = current_load + (trend * time_horizon_minutes)
                    
                    # Determine if scaling will be needed
                    current_instances = self.current_instances[component_type].get(name, 1)
                    predicted_instances = self._calculate_required_instances(
                        predicted_load,
                        component_type
                    )
                    
                    if predicted_instances != current_instances:
                        predictions['predictions'][name] = {
                            'type': component_type,
                            'current_instances': current_instances,
                            'predicted_instances': predicted_instances,
                            'current_load': current_load,
                            'predicted_load': predicted_load,
                            'confidence': self._calculate_prediction_confidence(history),
                            'action': 'scale_up' if predicted_instances > current_instances else 'scale_down'
                        }
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Failed to predict scaling needs: {e}")
            raise
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values (change per minute)"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _calculate_required_instances(self, load: float, component_type: str) -> int:
        """Calculate required instances based on load"""
        # Simple calculation: 1 instance per 100 units of load
        base_capacity = 100.0
        required = math.ceil(load / base_capacity)
        
        # Apply limits
        return max(
            self.min_instances[component_type],
            min(required, self.max_instances[component_type])
        )
    
    def _calculate_prediction_confidence(self, history: deque) -> float:
        """Calculate confidence in prediction based on historical accuracy"""
        if len(history) < 100:
            return 0.5  # Low confidence with limited data
        
        # Calculate variance in recent data
        recent_loads = [h['load'] for h in list(history)[-30:]]
        if not recent_loads:
            return 0.5
        
        variance = statistics.variance(recent_loads) if len(recent_loads) > 1 else 0
        
        # Lower variance = higher confidence
        max_variance = 1000  # Adjust based on typical load values
        confidence = max(0.3, 1.0 - (variance / max_variance))
        
        return min(0.95, confidence)
    
    def execute_system_scaling(self, systems_to_scale: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute scaling for specified systems"""
        start_time = time.time()
        results = {
            'success': True,
            'scaled_systems': [],
            'failed_systems': [],
            'total_time': 0.0
        }
        
        try:
            for system in systems_to_scale:
                system_name = system['name']
                target_instances = system['target_instances']
                
                # Validate target instances - HARD FAILURE for invalid targets
                if target_instances < self.min_instances['systems']:
                    raise ValueError(f'HARD FAILURE: {system_name} target instances {target_instances} below minimum {self.min_instances["systems"]} - NO FALLBACKS')
                    
                if target_instances > self.max_instances['systems']:
                    raise ValueError(f'HARD FAILURE: {system_name} target instances {target_instances} above maximum {self.max_instances["systems"]} - NO FALLBACKS')
                
                # Execute scaling
                current = self.current_instances['systems'][system_name]
                
                if target_instances > current:
                    # Scale up
                    for _ in range(target_instances - current):
                        scale_result = self._scale_up_component('systems', {
                            'name': system_name,
                            'current_instances': self.current_instances['systems'][system_name],
                            'reason': system.get('reason', 'Manual scaling')
                        })
                        if not scale_result['success']:
                            results['failed_systems'].append(scale_result)
                            results['success'] = False
                            break
                    else:
                        results['scaled_systems'].append({
                            'name': system_name,
                            'previous': current,
                            'new': target_instances
                        })
                        
                elif target_instances < current:
                    # Scale down
                    for _ in range(current - target_instances):
                        scale_result = self._scale_down_component('systems', {
                            'name': system_name,
                            'current_instances': self.current_instances['systems'][system_name],
                            'reason': system.get('reason', 'Manual scaling')
                        })
                        if not scale_result['success']:
                            results['failed_systems'].append(scale_result)
                            results['success'] = False
                            break
                    else:
                        results['scaled_systems'].append({
                            'name': system_name,
                            'previous': current,
                            'new': target_instances
                        })
            
            results['total_time'] = time.time() - start_time
            return results
            
        except Exception as e:
            self.logger.error(f"HARD FAILURE in system scaling: {e}")
            # NO FALLBACKS - HARD FAILURES ONLY
            raise
    
    def execute_agent_scaling(self, agents_to_scale: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute scaling for specified agents"""
        start_time = time.time()
        results = {
            'success': True,
            'scaled_agents': [],
            'failed_agents': [],
            'total_time': 0.0
        }
        
        try:
            for agent in agents_to_scale:
                agent_name = agent['name']
                target_instances = agent['target_instances']
                
                # Validate target instances - HARD FAILURE for invalid targets
                if target_instances < self.min_instances['agents']:
                    raise ValueError(f'HARD FAILURE: {agent_name} target instances {target_instances} below minimum {self.min_instances["agents"]} - NO FALLBACKS')
                    
                if target_instances > self.max_instances['agents']:
                    raise ValueError(f'HARD FAILURE: {agent_name} target instances {target_instances} above maximum {self.max_instances["agents"]} - NO FALLBACKS')
                
                # Execute scaling
                current = self.current_instances['agents'][agent_name]
                
                if target_instances > current:
                    # Scale up
                    for _ in range(target_instances - current):
                        scale_result = self._scale_up_component('agents', {
                            'name': agent_name,
                            'current_instances': self.current_instances['agents'][agent_name],
                            'reason': agent.get('reason', 'Manual scaling')
                        })
                        if not scale_result['success']:
                            results['failed_agents'].append(scale_result)
                            results['success'] = False
                            break
                    else:
                        results['scaled_agents'].append({
                            'name': agent_name,
                            'previous': current,
                            'new': target_instances
                        })
                        
                elif target_instances < current:
                    # Scale down
                    for _ in range(current - target_instances):
                        scale_result = self._scale_down_component('agents', {
                            'name': agent_name,
                            'current_instances': self.current_instances['agents'][agent_name],
                            'reason': agent.get('reason', 'Manual scaling')
                        })
                        if not scale_result['success']:
                            results['failed_agents'].append(scale_result)
                            results['success'] = False
                            break
                    else:
                        results['scaled_agents'].append({
                            'name': agent_name,
                            'previous': current,
                            'new': target_instances
                        })
            
            results['total_time'] = time.time() - start_time
            return results
            
        except Exception as e:
            self.logger.error(f"HARD FAILURE in agent scaling: {e}")
            # NO FALLBACKS - HARD FAILURES ONLY
            raise
    
    def validate_scaling_impact(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate impact of scaling decisions"""
        try:
            validation = {
                'timestamp': datetime.now(),
                'operations_validated': len(operations),
                'validations': []
            }
            
            # Wait for scaling to take effect
            time.sleep(5.0)
            
            # Validate each operation
            for operation in operations:
                if not operation.get('success'):
                    # HARD FAILURE - failed operations should not be silently skipped
                    raise RuntimeError(f"Cannot validate failed operation: {operation}")
                
                component_name = operation['component']
                component_type = operation['type']
                
                # Get current metrics - HARD FAILURE if monitor doesn't support required methods
                if component_type == 'systems':
                    if not hasattr(self.monitor, 'get_system_status'):
                        raise AttributeError(f"Monitor must implement get_system_status() method - NO FALLBACKS")
                    status = self.monitor.get_system_status(component_name)
                    metrics = self._extract_scaling_metrics(status)
                else:
                    if not hasattr(self.monitor, 'get_agent_status'):
                        raise AttributeError(f"Monitor must implement get_agent_status() method - NO FALLBACKS")
                    status = self.monitor.get_agent_status(component_name)
                    metrics = self._extract_agent_metrics(status)
                
                # Calculate impact
                impact = self._calculate_scaling_impact(operation, metrics)
                
                validation['validations'].append({
                    'component': component_name,
                    'type': component_type,
                    'action': operation['action'],
                    'metrics_after': metrics,
                    'impact': impact,
                    'successful': impact['efficiency'] > 0
                })
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Failed to validate scaling impact: {e}")
            raise
    
    def _calculate_scaling_impact(self, operation: Dict[str, Any], 
                                 metrics_after: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate impact of a scaling operation"""
        # Simulate impact calculation
        # In production, this would compare before/after metrics
        
        if operation['action'] == 'scale_up':
            # Expect improvement in metrics
            expected_improvement = 0.2  # 20% improvement expected
            actual_improvement = 0.15  # Simulated actual improvement
        else:
            # Scale down should maintain acceptable performance
            expected_improvement = -0.05  # Small degradation acceptable
            actual_improvement = -0.03  # Simulated actual impact
        
        efficiency = actual_improvement / expected_improvement if expected_improvement != 0 else 0
        
        return {
            'expected_improvement': expected_improvement,
            'actual_improvement': actual_improvement,
            'efficiency': efficiency,
            'performance_maintained': metrics_after.get('health_score', 0) > 0.8
        }
    
    def optimize_scaling_strategy(self) -> Dict[str, Any]:
        """Optimize scaling strategy based on performance data"""
        try:
            optimization = {
                'timestamp': datetime.now(),
                'current_thresholds': {
                    'scale_up': self.scale_up_thresholds.copy(),
                    'scale_down': self.scale_down_thresholds.copy()
                },
                'optimizations': []
            }
            
            # Analyze scaling history
            if len(self.scaling_history) < 100:
                optimization['message'] = 'Insufficient data for optimization'
                return optimization
            
            # Calculate scaling effectiveness
            recent_operations = list(self.scaling_history)[-100:]
            
            # Analyze scale-up operations
            scale_ups = [op for op in recent_operations if op.get('action') == 'scale_up']
            if scale_ups:
                avg_response_time = statistics.mean(
                    op.get('response_time', 0) for op in scale_ups
                )
                
                if avg_response_time > 8.0:  # Taking too long
                    # Lower thresholds to scale earlier
                    self.scale_up_thresholds['cpu_percent'] = max(70, self.scale_up_thresholds['cpu_percent'] - 5)
                    self.scale_up_thresholds['memory_percent'] = max(75, self.scale_up_thresholds['memory_percent'] - 5)
                    optimization['optimizations'].append({
                        'type': 'threshold_adjustment',
                        'reason': 'Slow scale-up response times',
                        'adjustments': 'Lowered scale-up thresholds'
                    })
            
            # Analyze scale-down operations
            scale_downs = [op for op in recent_operations if op.get('action') == 'scale_down']
            if scale_downs:
                # Check if we're scaling down too aggressively
                scale_down_failures = sum(1 for op in scale_downs if not op.get('success', True))
                if scale_down_failures > len(scale_downs) * 0.1:  # >10% failures
                    # Raise thresholds to be more conservative
                    self.scale_down_thresholds['cpu_percent'] = min(40, self.scale_down_thresholds['cpu_percent'] + 5)
                    self.scale_down_thresholds['memory_percent'] = min(50, self.scale_down_thresholds['memory_percent'] + 5)
                    optimization['optimizations'].append({
                        'type': 'threshold_adjustment',
                        'reason': 'High scale-down failure rate',
                        'adjustments': 'Raised scale-down thresholds'
                    })
            
            optimization['new_thresholds'] = {
                'scale_up': self.scale_up_thresholds.copy(),
                'scale_down': self.scale_down_thresholds.copy()
            }
            
            return optimization
            
        except Exception as e:
            self.logger.error(f"Failed to optimize scaling strategy: {e}")
            raise
    
    def monitor_scaling_performance(self) -> Dict[str, Any]:
        """Monitor scaling performance and efficiency"""
        try:
            performance = {
                'timestamp': datetime.now(),
                'metrics': {},
                'overall_efficiency': 0.0,
                'recommendations': []
            }
            
            # Check if any operations exist at all
            total_operations = (self.scaling_metrics['systems']['total_scaling_operations'] +
                              self.scaling_metrics['agents']['total_scaling_operations'])
            
            if total_operations == 0:
                # HARD FAILURE - no operations to monitor
                raise ValueError("No scaling operations recorded for any component type")
            
            # Calculate metrics for each component type
            for component_type in ['systems', 'agents']:
                metrics = self.scaling_metrics[component_type]
                
                if metrics['total_scaling_operations'] == 0:
                    # Skip component types with no operations
                    continue
                
                success_rate = metrics['successful_scaling'] / metrics['total_scaling_operations']
                scale_up_ratio = metrics['total_scale_ups'] / metrics['total_scaling_operations']
                
                performance['metrics'][component_type] = {
                    'total_operations': metrics['total_scaling_operations'],
                    'success_rate': success_rate,
                    'average_scaling_time': metrics['average_scaling_time'],
                    'scale_up_ratio': scale_up_ratio,
                    'scale_down_ratio': 1 - scale_up_ratio
                }
                
                # Generate recommendations
                if success_rate < 0.95:
                    performance['recommendations'].append(
                        f"Low success rate for {component_type} scaling: {success_rate:.1%}"
                    )
                
                if metrics['average_scaling_time'] > 8.0:
                    performance['recommendations'].append(
                        f"High average scaling time for {component_type}: {metrics['average_scaling_time']:.1f}s"
                    )
                
                if scale_up_ratio > 0.8:
                    performance['recommendations'].append(
                        f"Frequent scale-ups for {component_type} - consider increasing base capacity"
                    )
            
            # Calculate overall efficiency
            all_operations = sum(
                self.scaling_metrics[ct]['total_scaling_operations'] 
                for ct in ['systems', 'agents']
            )
            successful_operations = sum(
                self.scaling_metrics[ct]['successful_scaling'] 
                for ct in ['systems', 'agents']
            )
            
            if all_operations > 0:
                performance['overall_efficiency'] = successful_operations / all_operations
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Failed to monitor scaling performance: {e}")
            raise
    
    def _update_workload_history(self):
        """Update workload history for predictive scaling"""
        try:
            # Get current workload for all components
            system_statuses = self.monitor.get_all_system_status()
            for system_name, status in system_statuses.items():
                metrics = self._extract_scaling_metrics(status)
                # Calculate composite load score
                load = (
                    metrics['cpu_percent'] * 0.3 +
                    metrics['memory_percent'] * 0.3 +
                    (metrics['response_time_ms'] / 10) * 0.2 +  # Normalize to percentage
                    metrics['error_rate'] * 100 * 0.2  # Convert to percentage
                )
                
                self.workload_history[system_name].append({
                    'timestamp': time.time(),
                    'load': load,
                    'metrics': metrics
                })
            
            # Update agent workload
            agent_statuses = self.monitor.get_all_agent_status()
            for agent_name, status in agent_statuses.items():
                metrics = self._extract_agent_metrics(status)
                # Calculate agent load
                load = (
                    metrics['active_tasks'] * 10 +  # Weight active tasks
                    (100 - metrics['task_success_rate'] * 100) +  # Penalty for failures
                    metrics['response_time'] / 10  # Normalize response time
                )
                
                self.workload_history[agent_name].append({
                    'timestamp': time.time(),
                    'load': load,
                    'metrics': metrics
                })
                
        except Exception as e:
            self.logger.error(f"HARD FAILURE updating workload history: {e}")
            # NO FALLBACKS - HARD FAILURES ONLY
            raise
    
    def _record_scaling_operation(self, requirements: Dict[str, Any], 
                                 results: Dict[str, Any], scaling_time: float):
        """Record scaling operation for analytics"""
        record = {
            'timestamp': time.time(),
            'requirements': requirements,
            'results': results,
            'scaling_time': scaling_time,
            'success': results.get('success', False)
        }
        
        self.scaling_history.append(record)
        
        # Update metrics
        for operation in results.get('operations', []):
            if not operation.get('success'):
                # HARD FAILURE - failed operations should not be silently ignored
                raise RuntimeError(f"Cannot record metrics for failed operation: {operation}")
                
            component_type = operation['type']
            metrics = self.scaling_metrics[component_type]
            
            metrics['total_scaling_operations'] += 1
            if operation['success']:
                metrics['successful_scaling'] += 1
            else:
                metrics['failed_scaling'] += 1
            
            if operation['action'] == 'scale_up':
                metrics['total_scale_ups'] += 1
            else:
                metrics['total_scale_downs'] += 1
            
            # Update average scaling time
            total = metrics['total_scaling_operations']
            current_avg = metrics['average_scaling_time']
            metrics['average_scaling_time'] = (
                (current_avg * (total - 1) + scaling_time) / total
            )
    
    def generate_scaling_report(self, scaling_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive scaling report"""
        try:
            report = {
                'report_id': f"scaling_{int(time.time())}",
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_operations': len(scaling_results.get('operations', [])),
                    'successful_operations': sum(
                        1 for op in scaling_results.get('operations', [])
                        if op.get('success')
                    ),
                    'total_time': scaling_results.get('total_time', 0),
                    'success': scaling_results.get('success', False)
                },
                'operations': scaling_results.get('operations', []),
                'current_state': {
                    'system_instances': dict(self.current_instances['systems']),
                    'agent_instances': dict(self.current_instances['agents'])
                },
                'performance_metrics': (
                    self.monitor_scaling_performance() 
                    if (self.scaling_metrics['systems']['total_scaling_operations'] > 0 or 
                        self.scaling_metrics['agents']['total_scaling_operations'] > 0) 
                    else None
                ),
                'recommendations': []
            }
            
            # Add recommendations based on results
            if not scaling_results.get('success'):
                report['recommendations'].append(
                    "Scaling operation failed - review error logs and retry"
                )
            
            if report['summary']['total_time'] > self.max_decision_time:
                report['recommendations'].append(
                    f"Scaling took {report['summary']['total_time']:.1f}s - exceeds {self.max_decision_time}s limit"
                )
            
            # Check instance limits
            for system, count in self.current_instances['systems'].items():
                if count >= self.max_instances['systems'] * 0.8:
                    report['recommendations'].append(
                        f"System {system} approaching max instances ({count}/{self.max_instances['systems']})"
                    )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate scaling report: {e}")
            raise
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status"""
        # Only include performance if operations exist
        performance = None
        if (self.scaling_metrics['systems']['total_scaling_operations'] > 0 or 
            self.scaling_metrics['agents']['total_scaling_operations'] > 0):
            performance = self.monitor_scaling_performance()
        
        return {
            'is_running': self.is_running,
            'current_instances': {
                'systems': dict(self.current_instances['systems']),
                'agents': dict(self.current_instances['agents'])
            },
            'thresholds': {
                'scale_up': self.scale_up_thresholds,
                'scale_down': self.scale_down_thresholds
            },
            'recent_operations': list(self.scaling_history)[-10:],
            'performance': performance
        }
    
    def stop_auto_scaling(self) -> Dict[str, Any]:
        """Stop the auto-scaling engine"""
        with self._lock:
            self.is_running = False
            
            # Shutdown executor (timeout parameter not supported in older Python versions)
            self.executor.shutdown(wait=True)
            
            # Generate final report (only if operations exist)
            final_metrics = None
            if (self.scaling_metrics['systems']['total_scaling_operations'] > 0 or 
                self.scaling_metrics['agents']['total_scaling_operations'] > 0):
                final_metrics = self.monitor_scaling_performance()
            
            self.logger.info("Auto-Scaling Engine stopped")
            return {
                'success': True,
                'final_metrics': final_metrics,
                'total_operations': sum(
                    self.scaling_metrics[ct]['total_scaling_operations']
                    for ct in ['systems', 'agents']
                )
            }


def create_scaling_engine(monitor, brain_system, agent_system) -> Dict[str, Any]:
    """Factory function to create scaling engine with all components"""
    try:
        engine = AutoScalingEngine(monitor, brain_system, agent_system)
        
        return {
            'success': True,
            'engine': engine
        }
        
    except Exception as e:
        logger.error(f"Failed to create scaling engine: {e}")
        return {
            'success': False,
            'error': str(e)
        }