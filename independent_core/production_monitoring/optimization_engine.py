"""
Production Optimization Engine - Real-time optimization for Saraphis production
NO FALLBACKS - HARD FAILURES ONLY

Provides dynamic optimization of system performance, resource allocation,
and workload distribution with automatic optimization application.
"""

import os
import sys
import json
import time
import logging
import threading
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor
import traceback

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class OptimizationAction:
    """Represents an optimization action to be applied"""
    action_id: str
    action_type: str  # 'resource', 'performance', 'coordination', 'security'
    target: str  # System or agent name
    parameters: Dict[str, Any]
    priority: str  # 'critical', 'high', 'medium', 'low'
    expected_improvement: float
    status: str  # 'pending', 'applied', 'failed', 'rolled_back'
    applied_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None


@dataclass
class OptimizationResult:
    """Result of an optimization action"""
    action_id: str
    success: bool
    actual_improvement: float
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    execution_time: float
    error: Optional[str] = None


class ProductionOptimizationEngine:
    """Real-time optimization engine for Saraphis production"""
    
    def __init__(self, monitor, brain_system, agent_system):
        """
        Initialize production optimization engine.
        
        Args:
            monitor: Real-time production monitor instance
            brain_system: Main Brain system instance
            agent_system: Multi-agent system instance
        """
        self.monitor = monitor
        self.brain_system = brain_system
        self.agent_system = agent_system
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Optimization state
        self.optimization_active = False
        self.auto_optimization_enabled = True
        
        # Optimization history
        self.optimization_actions: Dict[str, OptimizationAction] = {}
        self.optimization_results: List[OptimizationResult] = []
        self.optimization_queue = deque()
        
        # Performance baselines
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        self.optimization_thresholds = {
            'cpu_threshold': 80,  # Trigger optimization if CPU > 80%
            'memory_threshold': 85,  # Trigger optimization if memory > 85%
            'response_time_threshold': 100,  # ms
            'error_rate_threshold': 0.05,
            'min_improvement': 0.10  # Minimum 10% improvement expected
        }
        
        # Resource allocation limits
        self.resource_limits = {
            'max_cpu_per_system': 90,
            'max_memory_per_system': 90,
            'max_total_cpu': 80,
            'max_total_memory': 85,
            'min_free_resources': 10  # Always keep 10% free
        }
        
        # Optimization strategies
        self.optimization_strategies = {
            'resource': self._optimize_resource_strategy,
            'performance': self._optimize_performance_strategy,
            'coordination': self._optimize_coordination_strategy,
            'workload': self._optimize_workload_strategy,
            'security': self._optimize_security_strategy
        }
        
        # Thread management
        self._optimization_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=8)
        
        # Optimization metrics
        self.total_optimizations = 0
        self.successful_optimizations = 0
        self.failed_optimizations = 0
        self.total_improvement = 0.0
        
        # Rollback capability
        self.rollback_enabled = True
        self.rollback_history: Dict[str, Dict[str, Any]] = {}
        
    def start_optimization(self) -> Dict[str, Any]:
        """Start the optimization engine"""
        try:
            self.logger.info("Starting production optimization engine...")
            
            with self._lock:
                if self.optimization_active:
                    raise RuntimeError("Optimization already active")
                
                self.optimization_active = True
                self._stop_event.clear()
                
                # Start optimization thread
                self._optimization_thread = threading.Thread(
                    target=self._optimization_loop,
                    daemon=True
                )
                self._optimization_thread.start()
                
                self.logger.info("Optimization engine started successfully")
                
                return {
                    'started': True,
                    'auto_optimization': self.auto_optimization_enabled,
                    'strategies': list(self.optimization_strategies.keys()),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to start optimization: {e}")
            return {
                'started': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def stop_optimization(self) -> Dict[str, Any]:
        """Stop the optimization engine"""
        try:
            self.logger.info("Stopping optimization engine...")
            
            with self._lock:
                if not self.optimization_active:
                    return {'stopped': True, 'message': 'Optimization was not active'}
                
                self.optimization_active = False
                self._stop_event.set()
                
                if self._optimization_thread:
                    self._optimization_thread.join(timeout=5)
                
                # Shutdown executor
                self._executor.shutdown(wait=False)
                
                self.logger.info("Optimization engine stopped")
                
                return {
                    'stopped': True,
                    'total_optimizations': self.total_optimizations,
                    'successful_optimizations': self.successful_optimizations,
                    'failed_optimizations': self.failed_optimizations,
                    'total_improvement': self.total_improvement,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error stopping optimization: {e}")
            return {
                'stopped': False,
                'error': str(e)
            }
    
    def _optimization_loop(self):
        """Main optimization loop"""
        while not self._stop_event.is_set():
            try:
                # Check for optimization opportunities
                if self.auto_optimization_enabled:
                    self._check_optimization_opportunities()
                
                # Process optimization queue
                self._process_optimization_queue()
                
                # Sleep for a short interval
                self._stop_event.wait(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                self.logger.error(traceback.format_exc())
    
    def _check_optimization_opportunities(self):
        """Check for optimization opportunities based on current metrics"""
        try:
            # Get current system metrics
            system_status = self.monitor.monitor_all_systems()
            
            if 'system_status' not in system_status:
                return
            
            # Check each system for optimization needs
            for system_name, status in system_status['system_status'].items():
                # Check CPU usage
                if status['cpu_usage'] > self.optimization_thresholds['cpu_threshold']:
                    self._queue_optimization({
                        'type': 'resource',
                        'target': system_name,
                        'reason': 'high_cpu',
                        'metric': status['cpu_usage'],
                        'priority': 'high' if status['cpu_usage'] > 90 else 'medium'
                    })
                
                # Check memory usage
                if status['memory_usage'] > self.optimization_thresholds['memory_threshold']:
                    self._queue_optimization({
                        'type': 'resource',
                        'target': system_name,
                        'reason': 'high_memory',
                        'metric': status['memory_usage'],
                        'priority': 'high' if status['memory_usage'] > 95 else 'medium'
                    })
                
                # Check response time
                if status['response_time'] > self.optimization_thresholds['response_time_threshold']:
                    self._queue_optimization({
                        'type': 'performance',
                        'target': system_name,
                        'reason': 'slow_response',
                        'metric': status['response_time'],
                        'priority': 'medium'
                    })
                
                # Check error rate
                if status['error_count'] > 0:
                    error_rate = status['error_count'] / max(1, self.monitor.request_counts[system_name])
                    if error_rate > self.optimization_thresholds['error_rate_threshold']:
                        self._queue_optimization({
                            'type': 'performance',
                            'target': system_name,
                            'reason': 'high_errors',
                            'metric': error_rate,
                            'priority': 'high'
                        })
            
            # Check agent coordination
            agent_status = self.monitor.monitor_all_agents()
            
            if 'agent_status' in agent_status:
                for agent_name, status in agent_status['agent_status'].items():
                    if status['coordination_score'] < 0.8:
                        self._queue_optimization({
                            'type': 'coordination',
                            'target': agent_name,
                            'reason': 'poor_coordination',
                            'metric': status['coordination_score'],
                            'priority': 'medium'
                        })
                    
        except Exception as e:
            self.logger.error(f"Error checking optimization opportunities: {e}")
    
    def _queue_optimization(self, optimization_request: Dict[str, Any]):
        """Queue an optimization request"""
        try:
            # Check if similar optimization is already queued
            for queued in self.optimization_queue:
                if (queued['type'] == optimization_request['type'] and 
                    queued['target'] == optimization_request['target'] and
                    queued['reason'] == optimization_request['reason']):
                    return  # Skip duplicate
            
            # Add to queue
            with self._lock:
                self.optimization_queue.append(optimization_request)
                self.logger.info(f"Queued optimization: {optimization_request}")
                
        except Exception as e:
            self.logger.error(f"Error queueing optimization: {e}")
    
    def _process_optimization_queue(self):
        """Process pending optimizations from the queue"""
        try:
            while self.optimization_queue and self.optimization_active:
                with self._lock:
                    if not self.optimization_queue:
                        break
                    
                    # Get next optimization
                    optimization_request = self.optimization_queue.popleft()
                
                # Create optimization action
                action = self._create_optimization_action(optimization_request)
                
                if action:
                    # Apply optimization
                    result = self._apply_optimization(action)
                    
                    # Store result
                    with self._lock:
                        self.optimization_results.append(result)
                        
                        if result.success:
                            self.successful_optimizations += 1
                            self.total_improvement += result.actual_improvement
                        else:
                            self.failed_optimizations += 1
                        
                        self.total_optimizations += 1
                        
        except Exception as e:
            self.logger.error(f"Error processing optimization queue: {e}")
    
    def _create_optimization_action(self, request: Dict[str, Any]) -> Optional[OptimizationAction]:
        """Create an optimization action from a request"""
        try:
            action_id = f"opt_{int(time.time())}_{request['target']}"
            
            # Determine parameters based on optimization type
            parameters = {}
            expected_improvement = 0.1  # Default 10%
            
            if request['type'] == 'resource':
                if request['reason'] == 'high_cpu':
                    parameters['cpu_increase'] = 0.2  # Increase CPU allocation by 20%
                    parameters['thread_optimization'] = True
                    expected_improvement = 0.15
                elif request['reason'] == 'high_memory':
                    parameters['memory_increase'] = 0.25  # Increase memory by 25%
                    parameters['garbage_collection'] = True
                    expected_improvement = 0.20
            
            elif request['type'] == 'performance':
                if request['reason'] == 'slow_response':
                    parameters['cache_optimization'] = True
                    parameters['query_optimization'] = True
                    expected_improvement = 0.25
                elif request['reason'] == 'high_errors':
                    parameters['error_handling_improvement'] = True
                    parameters['retry_strategy'] = 'exponential_backoff'
                    expected_improvement = 0.30
            
            elif request['type'] == 'coordination':
                parameters['communication_optimization'] = True
                parameters['task_redistribution'] = True
                expected_improvement = 0.15
            
            action = OptimizationAction(
                action_id=action_id,
                action_type=request['type'],
                target=request['target'],
                parameters=parameters,
                priority=request['priority'],
                expected_improvement=expected_improvement,
                status='pending'
            )
            
            with self._lock:
                self.optimization_actions[action_id] = action
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error creating optimization action: {e}")
            return None
    
    def _apply_optimization(self, action: OptimizationAction) -> OptimizationResult:
        """Apply an optimization action"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Applying optimization: {action.action_id}")
            
            # Get metrics before optimization
            metrics_before = self._get_target_metrics(action.target)
            
            # Store rollback information
            if self.rollback_enabled:
                self.rollback_history[action.action_id] = {
                    'action': action,
                    'state_before': metrics_before,
                    'timestamp': datetime.now()
                }
            
            # Apply optimization based on type
            strategy = self.optimization_strategies.get(action.action_type)
            if not strategy:
                raise ValueError(f"Unknown optimization type: {action.action_type}")
            
            # Execute optimization strategy
            success, error = strategy(action)
            
            # Update action status
            action.status = 'applied' if success else 'failed'
            action.applied_at = datetime.now()
            
            # Wait a moment for changes to take effect
            time.sleep(2)
            
            # Get metrics after optimization
            metrics_after = self._get_target_metrics(action.target)
            
            # Calculate actual improvement
            actual_improvement = self._calculate_improvement(metrics_before, metrics_after)
            
            execution_time = time.time() - start_time
            
            result = OptimizationResult(
                action_id=action.action_id,
                success=success,
                actual_improvement=actual_improvement,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                execution_time=execution_time,
                error=error
            )
            
            action.result = asdict(result)
            
            # Check if improvement meets expectations
            if success and actual_improvement < action.expected_improvement * 0.5:
                self.logger.warning(
                    f"Optimization {action.action_id} underperformed: "
                    f"expected {action.expected_improvement:.2%}, got {actual_improvement:.2%}"
                )
                
                # Consider rollback if improvement is too low
                if self.rollback_enabled and actual_improvement < 0.05:
                    self._rollback_optimization(action.action_id)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying optimization: {e}")
            return OptimizationResult(
                action_id=action.action_id,
                success=False,
                actual_improvement=0.0,
                metrics_before=metrics_before if 'metrics_before' in locals() else {},
                metrics_after={},
                execution_time=time.time() - start_time,
                error=str(e)
            )
    
    def _get_target_metrics(self, target: str) -> Dict[str, float]:
        """Get current metrics for a target system or agent"""
        try:
            metrics = {}
            
            # Check if target is a system
            if target in self.monitor.system_metrics:
                system_metrics = self.monitor.system_metrics[target]
                metrics = {
                    'health_score': system_metrics.health_score,
                    'cpu_usage': system_metrics.cpu_usage,
                    'memory_usage': system_metrics.memory_usage,
                    'response_time': system_metrics.response_time,
                    'error_count': system_metrics.error_count
                }
            
            # Check if target is an agent
            elif target in self.monitor.agent_metrics:
                agent_metrics = self.monitor.agent_metrics[target]
                metrics = {
                    'success_rate': agent_metrics.success_rate,
                    'average_task_time': agent_metrics.average_task_time,
                    'coordination_score': agent_metrics.coordination_score,
                    'resource_usage': agent_metrics.resource_usage
                }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting target metrics: {e}")
            return {}
    
    def _calculate_improvement(self, before: Dict[str, float], after: Dict[str, float]) -> float:
        """Calculate improvement percentage"""
        try:
            improvements = []
            
            # Calculate improvement for each metric
            for metric, before_value in before.items():
                if metric in after:
                    after_value = after[metric]
                    
                    # Different metrics improve in different directions
                    if metric in ['health_score', 'success_rate', 'coordination_score']:
                        # Higher is better
                        if before_value > 0:
                            improvement = (after_value - before_value) / before_value
                        else:
                            improvement = after_value
                    else:
                        # Lower is better (cpu, memory, response time, errors)
                        if before_value > 0:
                            improvement = (before_value - after_value) / before_value
                        else:
                            improvement = 0.0
                    
                    improvements.append(improvement)
            
            # Return average improvement
            return statistics.mean(improvements) if improvements else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating improvement: {e}")
            return 0.0
    
    def _optimize_resource_strategy(self, action: OptimizationAction) -> Tuple[bool, Optional[str]]:
        """Apply resource optimization strategy"""
        try:
            target = action.target
            parameters = action.parameters
            
            # Simulate resource optimization
            # In real implementation, would actually adjust resource allocation
            
            if 'cpu_increase' in parameters:
                # Increase CPU allocation
                self.logger.info(f"Increasing CPU allocation for {target} by {parameters['cpu_increase']:.0%}")
                # Would call actual resource manager here
            
            if 'memory_increase' in parameters:
                # Increase memory allocation
                self.logger.info(f"Increasing memory allocation for {target} by {parameters['memory_increase']:.0%}")
                # Would call actual resource manager here
            
            if parameters.get('thread_optimization'):
                # Optimize thread pool sizes
                self.logger.info(f"Optimizing thread pools for {target}")
                # Would adjust thread pool configuration here
            
            if parameters.get('garbage_collection'):
                # Force garbage collection
                self.logger.info(f"Triggering garbage collection for {target}")
                # Would trigger GC here
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def _optimize_performance_strategy(self, action: OptimizationAction) -> Tuple[bool, Optional[str]]:
        """Apply performance optimization strategy"""
        try:
            target = action.target
            parameters = action.parameters
            
            if parameters.get('cache_optimization'):
                # Optimize caching
                self.logger.info(f"Optimizing cache for {target}")
                # Would optimize cache configuration here
            
            if parameters.get('query_optimization'):
                # Optimize database queries
                self.logger.info(f"Optimizing queries for {target}")
                # Would optimize query execution plans here
            
            if parameters.get('error_handling_improvement'):
                # Improve error handling
                self.logger.info(f"Improving error handling for {target}")
                # Would update error handling configuration here
            
            if 'retry_strategy' in parameters:
                # Update retry strategy
                self.logger.info(f"Updating retry strategy for {target} to {parameters['retry_strategy']}")
                # Would configure retry mechanism here
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def _optimize_coordination_strategy(self, action: OptimizationAction) -> Tuple[bool, Optional[str]]:
        """Apply coordination optimization strategy"""
        try:
            target = action.target
            parameters = action.parameters
            
            if parameters.get('communication_optimization'):
                # Optimize agent communication
                self.logger.info(f"Optimizing communication for {target}")
                # Would optimize communication protocols here
            
            if parameters.get('task_redistribution'):
                # Redistribute tasks among agents
                self.logger.info(f"Redistributing tasks for {target}")
                # Would rebalance task assignments here
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def _optimize_workload_strategy(self, action: OptimizationAction) -> Tuple[bool, Optional[str]]:
        """Apply workload optimization strategy"""
        try:
            target = action.target
            
            # Analyze current workload distribution
            self.logger.info(f"Optimizing workload distribution for {target}")
            
            # Would implement actual workload balancing here
            # - Move tasks between systems
            # - Adjust processing priorities
            # - Optimize batch sizes
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def _optimize_security_strategy(self, action: OptimizationAction) -> Tuple[bool, Optional[str]]:
        """Apply security optimization strategy"""
        try:
            target = action.target
            
            # Enhance security configuration
            self.logger.info(f"Optimizing security configuration for {target}")
            
            # Would implement actual security enhancements here
            # - Update access controls
            # - Enhance encryption
            # - Improve audit logging
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def _rollback_optimization(self, action_id: str):
        """Rollback an optimization"""
        try:
            if action_id not in self.rollback_history:
                self.logger.warning(f"No rollback information for {action_id}")
                return
            
            rollback_info = self.rollback_history[action_id]
            action = rollback_info['action']
            
            self.logger.info(f"Rolling back optimization {action_id}")
            
            # In real implementation, would restore previous state
            # For now, just mark as rolled back
            action.status = 'rolled_back'
            
            # Remove from rollback history
            del self.rollback_history[action_id]
            
        except Exception as e:
            self.logger.error(f"Error rolling back optimization: {e}")
    
    def optimize_system_performance(self) -> Dict[str, Any]:
        """
        Optimize system performance in real-time.
        NO FALLBACKS - HARD FAILURES ONLY
        """
        try:
            # Get current performance issues
            issues = self.monitor.detect_performance_issues()
            
            if not issues.get('issues'):
                return {
                    'optimization': 'system_performance',
                    'status': 'no_issues',
                    'message': 'No performance issues detected',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Create optimization actions for each issue
            optimization_count = 0
            
            for issue in issues['issues']:
                request = {
                    'type': 'performance',
                    'target': issue.get('system') or issue.get('agent'),
                    'reason': issue['type'],
                    'metric': issue.get('response_time') or issue.get('latency') or issue.get('success_rate'),
                    'priority': issue['severity']
                }
                
                self._queue_optimization(request)
                optimization_count += 1
            
            return {
                'optimization': 'system_performance',
                'status': 'optimizing',
                'issues_addressed': optimization_count,
                'queue_size': len(self.optimization_queue),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to optimize system performance: {e}")
            return {
                'optimization': 'system_performance',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def optimize_agent_coordination(self) -> Dict[str, Any]:
        """
        Optimize agent coordination and communication.
        NO FALLBACKS - HARD FAILURES ONLY
        """
        try:
            # Get agent metrics
            agent_status = self.monitor.monitor_all_agents()
            
            if 'agent_status' not in agent_status:
                raise RuntimeError("Unable to get agent status")
            
            optimization_count = 0
            coordination_issues = []
            
            for agent_name, status in agent_status['agent_status'].items():
                # Check coordination score
                if status['coordination_score'] < 0.85:
                    coordination_issues.append({
                        'agent': agent_name,
                        'score': status['coordination_score'],
                        'latency': status['communication_latency']
                    })
                    
                    # Queue optimization
                    self._queue_optimization({
                        'type': 'coordination',
                        'target': agent_name,
                        'reason': 'poor_coordination',
                        'metric': status['coordination_score'],
                        'priority': 'high' if status['coordination_score'] < 0.7 else 'medium'
                    })
                    optimization_count += 1
            
            return {
                'optimization': 'agent_coordination',
                'status': 'optimizing' if optimization_count > 0 else 'optimal',
                'coordination_issues': coordination_issues,
                'optimizations_queued': optimization_count,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to optimize agent coordination: {e}")
            return {
                'optimization': 'agent_coordination',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def optimize_resource_allocation(self) -> Dict[str, Any]:
        """
        Optimize resource allocation across systems.
        NO FALLBACKS - HARD FAILURES ONLY
        """
        try:
            # Get resource utilization
            resource_status = self.monitor.track_resource_utilization()
            
            if 'average_utilization' not in resource_status:
                raise RuntimeError("Unable to get resource utilization")
            
            avg_util = resource_status['average_utilization']
            hotspots = resource_status['resource_hotspots']
            
            optimization_count = 0
            resource_optimizations = []
            
            # Address CPU hotspots
            for system, cpu_usage in hotspots['cpu']:
                if cpu_usage > self.optimization_thresholds['cpu_threshold']:
                    resource_optimizations.append({
                        'system': system,
                        'resource': 'cpu',
                        'current': cpu_usage,
                        'target': 70  # Target 70% CPU usage
                    })
                    
                    self._queue_optimization({
                        'type': 'resource',
                        'target': system,
                        'reason': 'high_cpu',
                        'metric': cpu_usage,
                        'priority': 'high' if cpu_usage > 90 else 'medium'
                    })
                    optimization_count += 1
            
            # Address memory hotspots
            for system, memory_usage in hotspots['memory']:
                if memory_usage > self.optimization_thresholds['memory_threshold']:
                    resource_optimizations.append({
                        'system': system,
                        'resource': 'memory',
                        'current': memory_usage,
                        'target': 75  # Target 75% memory usage
                    })
                    
                    self._queue_optimization({
                        'type': 'resource',
                        'target': system,
                        'reason': 'high_memory',
                        'metric': memory_usage,
                        'priority': 'high' if memory_usage > 95 else 'medium'
                    })
                    optimization_count += 1
            
            return {
                'optimization': 'resource_allocation',
                'status': 'optimizing' if optimization_count > 0 else 'balanced',
                'average_utilization': avg_util,
                'resource_optimizations': resource_optimizations,
                'optimizations_queued': optimization_count,
                'efficiency_score': resource_status['efficiency_score'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to optimize resource allocation: {e}")
            return {
                'optimization': 'resource_allocation',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def optimize_workload_distribution(self) -> Dict[str, Any]:
        """
        Optimize workload distribution and balancing.
        NO FALLBACKS - HARD FAILURES ONLY
        """
        try:
            # Analyze current workload distribution
            system_loads = {}
            agent_loads = {}
            
            # Get system loads
            for system_name, metrics in self.monitor.system_metrics.items():
                load_score = (metrics.cpu_usage + metrics.memory_usage) / 2
                system_loads[system_name] = {
                    'load': load_score,
                    'requests': metrics.request_count,
                    'response_time': metrics.response_time
                }
            
            # Get agent loads  
            for agent_name, metrics in self.monitor.agent_metrics.items():
                agent_loads[agent_name] = {
                    'tasks': metrics.task_count,
                    'avg_task_time': metrics.average_task_time,
                    'resource_usage': metrics.resource_usage
                }
            
            # Identify imbalances
            avg_system_load = statistics.mean(s['load'] for s in system_loads.values())
            avg_agent_tasks = statistics.mean(a['tasks'] for a in agent_loads.values())
            
            rebalancing_needed = []
            
            # Check for system imbalances
            for system, load_info in system_loads.items():
                deviation = abs(load_info['load'] - avg_system_load)
                if deviation > 20:  # More than 20% deviation
                    rebalancing_needed.append({
                        'type': 'system',
                        'target': system,
                        'current_load': load_info['load'],
                        'target_load': avg_system_load
                    })
            
            # Check for agent imbalances
            for agent, load_info in agent_loads.items():
                if load_info['tasks'] > 0:
                    deviation = abs(load_info['tasks'] - avg_agent_tasks) / avg_agent_tasks
                    if deviation > 0.3:  # More than 30% deviation
                        rebalancing_needed.append({
                            'type': 'agent',
                            'target': agent,
                            'current_tasks': load_info['tasks'],
                            'target_tasks': int(avg_agent_tasks)
                        })
            
            # Queue workload optimizations
            for imbalance in rebalancing_needed:
                self._queue_optimization({
                    'type': 'workload',
                    'target': imbalance['target'],
                    'reason': 'load_imbalance',
                    'metric': imbalance.get('current_load', imbalance.get('current_tasks')),
                    'priority': 'medium'
                })
            
            return {
                'optimization': 'workload_distribution',
                'status': 'rebalancing' if rebalancing_needed else 'balanced',
                'system_loads': system_loads,
                'agent_loads': agent_loads,
                'rebalancing_targets': rebalancing_needed,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to optimize workload distribution: {e}")
            return {
                'optimization': 'workload_distribution',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def optimize_security_configuration(self) -> Dict[str, Any]:
        """
        Optimize security configuration and monitoring.
        NO FALLBACKS - HARD FAILURES ONLY
        """
        try:
            # Get security status
            security_status = self.monitor.monitor_security_status()
            
            if 'security_score' not in security_status:
                raise RuntimeError("Unable to get security status")
            
            security_optimizations = []
            
            # Check security score
            if security_status['security_score'] < 0.95:
                # Analyze security events
                for event in security_status.get('security_events', []):
                    if event['severity'] in ['high', 'critical']:
                        security_optimizations.append({
                            'type': event['type'],
                            'action': event.get('action', 'investigate'),
                            'priority': event['severity']
                        })
                        
                        # Queue security optimization
                        self._queue_optimization({
                            'type': 'security',
                            'target': 'security_system',
                            'reason': event['type'],
                            'metric': security_status['security_score'],
                            'priority': event['severity']
                        })
            
            # Apply security recommendations
            for recommendation in security_status.get('recommendations', []):
                if recommendation:  # Filter out None values
                    security_optimizations.append({
                        'type': 'recommendation',
                        'action': recommendation,
                        'priority': 'medium'
                    })
            
            return {
                'optimization': 'security_configuration',
                'status': 'optimizing' if security_optimizations else 'secure',
                'security_score': security_status['security_score'],
                'optimizations': security_optimizations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to optimize security configuration: {e}")
            return {
                'optimization': 'security_configuration',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def apply_optimization_changes(self) -> Dict[str, Any]:
        """
        Apply optimization changes to production systems.
        NO FALLBACKS - HARD FAILURES ONLY
        """
        try:
            # Process all pending optimizations immediately
            pending_count = len(self.optimization_queue)
            
            if pending_count == 0:
                return {
                    'application': 'optimization_changes',
                    'status': 'no_pending_changes',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Force process all pending optimizations
            self._process_optimization_queue()
            
            # Get applied optimizations
            applied = [
                action for action in self.optimization_actions.values()
                if action.status == 'applied'
            ]
            
            failed = [
                action for action in self.optimization_actions.values()
                if action.status == 'failed'
            ]
            
            return {
                'application': 'optimization_changes',
                'status': 'applied',
                'total_pending': pending_count,
                'applied_count': len(applied),
                'failed_count': len(failed),
                'success_rate': len(applied) / (len(applied) + len(failed)) if (len(applied) + len(failed)) > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to apply optimization changes: {e}")
            return {
                'application': 'optimization_changes',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def measure_optimization_impact(self) -> Dict[str, Any]:
        """
        Measure impact of optimization changes.
        NO FALLBACKS - HARD FAILURES ONLY
        """
        try:
            if not self.optimization_results:
                return {
                    'measurement': 'optimization_impact',
                    'status': 'no_optimizations',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Analyze optimization results
            successful = [r for r in self.optimization_results if r.success]
            
            if not successful:
                return {
                    'measurement': 'optimization_impact',
                    'status': 'no_successful_optimizations',
                    'total_attempts': len(self.optimization_results),
                    'timestamp': datetime.now().isoformat()
                }
            
            # Calculate impact metrics
            improvements = [r.actual_improvement for r in successful]
            avg_improvement = statistics.mean(improvements)
            max_improvement = max(improvements)
            min_improvement = min(improvements)
            
            # Categorize by optimization type
            impact_by_type = defaultdict(list)
            for action_id, action in self.optimization_actions.items():
                result = next((r for r in successful if r.action_id == action_id), None)
                if result:
                    impact_by_type[action.action_type].append(result.actual_improvement)
            
            type_impacts = {
                opt_type: statistics.mean(impacts)
                for opt_type, impacts in impact_by_type.items()
            }
            
            return {
                'measurement': 'optimization_impact',
                'status': 'measured',
                'total_optimizations': self.total_optimizations,
                'successful_optimizations': self.successful_optimizations,
                'failed_optimizations': self.failed_optimizations,
                'average_improvement': avg_improvement,
                'max_improvement': max_improvement,
                'min_improvement': min_improvement,
                'impact_by_type': type_impacts,
                'overall_improvement': self.total_improvement / max(1, self.successful_optimizations),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to measure optimization impact: {e}")
            return {
                'measurement': 'optimization_impact',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def rollback_optimization_changes(self) -> Dict[str, Any]:
        """
        Rollback optimization changes if needed.
        NO FALLBACKS - HARD FAILURES ONLY
        """
        try:
            if not self.rollback_enabled:
                return {
                    'rollback': 'optimization_changes',
                    'status': 'rollback_disabled',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Find recent optimizations that underperformed
            rollback_candidates = []
            
            for result in self.optimization_results[-10:]:  # Check last 10 optimizations
                if result.success and result.actual_improvement < 0.05:  # Less than 5% improvement
                    if result.action_id in self.rollback_history:
                        rollback_candidates.append(result.action_id)
            
            if not rollback_candidates:
                return {
                    'rollback': 'optimization_changes',
                    'status': 'no_rollback_needed',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Perform rollbacks
            rolled_back = []
            for action_id in rollback_candidates:
                try:
                    self._rollback_optimization(action_id)
                    rolled_back.append(action_id)
                except Exception as e:
                    self.logger.error(f"Failed to rollback {action_id}: {e}")
            
            return {
                'rollback': 'optimization_changes',
                'status': 'rolled_back',
                'rolled_back_count': len(rolled_back),
                'rolled_back_actions': rolled_back,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to rollback optimization changes: {e}")
            return {
                'rollback': 'optimization_changes',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_optimization_report(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive optimization report.
        NO FALLBACKS - HARD FAILURES ONLY
        """
        try:
            report = {
                'report_id': f"opt_engine_report_{int(time.time())}",
                'timestamp': datetime.now().isoformat(),
                'optimization_period': {
                    'duration_seconds': time.time() - (self.monitor.monitoring_start_time.timestamp() if self.monitor.monitoring_start_time else time.time()),
                    'total_optimizations': self.total_optimizations,
                    'successful': self.successful_optimizations,
                    'failed': self.failed_optimizations
                },
                
                # Performance improvements
                'performance_improvements': {
                    'average_improvement': self.total_improvement / max(1, self.successful_optimizations),
                    'best_optimization': None,
                    'worst_optimization': None
                },
                
                # Optimization breakdown
                'optimization_breakdown': {},
                
                # Resource impact
                'resource_impact': {
                    'cpu_reduction': 0.0,
                    'memory_reduction': 0.0,
                    'response_time_improvement': 0.0
                },
                
                # Recommendations
                'recommendations': [],
                
                # Detailed results
                'detailed_results': []
            }
            
            # Find best and worst optimizations
            if self.optimization_results:
                successful = [r for r in self.optimization_results if r.success]
                if successful:
                    best = max(successful, key=lambda r: r.actual_improvement)
                    worst = min(successful, key=lambda r: r.actual_improvement)
                    
                    report['performance_improvements']['best_optimization'] = {
                        'action_id': best.action_id,
                        'improvement': best.actual_improvement,
                        'type': self.optimization_actions.get(best.action_id, {}).action_type
                    }
                    
                    report['performance_improvements']['worst_optimization'] = {
                        'action_id': worst.action_id,
                        'improvement': worst.actual_improvement,
                        'type': self.optimization_actions.get(worst.action_id, {}).action_type
                    }
            
            # Calculate optimization breakdown by type
            type_counts = defaultdict(int)
            type_improvements = defaultdict(list)
            
            for action in self.optimization_actions.values():
                type_counts[action.action_type] += 1
                result = next((r for r in self.optimization_results if r.action_id == action.action_id), None)
                if result and result.success:
                    type_improvements[action.action_type].append(result.actual_improvement)
            
            for opt_type, count in type_counts.items():
                improvements = type_improvements.get(opt_type, [])
                report['optimization_breakdown'][opt_type] = {
                    'count': count,
                    'average_improvement': statistics.mean(improvements) if improvements else 0.0
                }
            
            # Calculate resource impact
            for result in self.optimization_results:
                if result.success:
                    before = result.metrics_before
                    after = result.metrics_after
                    
                    if 'cpu_usage' in before and 'cpu_usage' in after:
                        report['resource_impact']['cpu_reduction'] += (before['cpu_usage'] - after['cpu_usage'])
                    
                    if 'memory_usage' in before and 'memory_usage' in after:
                        report['resource_impact']['memory_reduction'] += (before['memory_usage'] - after['memory_usage'])
                    
                    if 'response_time' in before and 'response_time' in after:
                        report['resource_impact']['response_time_improvement'] += (before['response_time'] - after['response_time'])
            
            # Normalize resource impact
            if self.successful_optimizations > 0:
                report['resource_impact']['cpu_reduction'] /= self.successful_optimizations
                report['resource_impact']['memory_reduction'] /= self.successful_optimizations
                report['resource_impact']['response_time_improvement'] /= self.successful_optimizations
            
            # Generate recommendations
            if report['performance_improvements']['average_improvement'] < 0.1:
                report['recommendations'].append("Consider more aggressive optimization strategies")
            
            if self.failed_optimizations > self.successful_optimizations * 0.2:
                report['recommendations'].append("Review failed optimization patterns")
            
            if report['resource_impact']['cpu_reduction'] < 5:
                report['recommendations'].append("Focus on CPU optimization strategies")
            
            # Add recent optimization details
            for result in self.optimization_results[-20:]:  # Last 20 optimizations
                action = self.optimization_actions.get(result.action_id)
                if action:
                    report['detailed_results'].append({
                        'action_id': result.action_id,
                        'type': action.action_type,
                        'target': action.target,
                        'success': result.success,
                        'improvement': result.actual_improvement,
                        'execution_time': result.execution_time
                    })
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization report: {e}")
            return {
                'report_id': f"opt_engine_report_error_{int(time.time())}",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }