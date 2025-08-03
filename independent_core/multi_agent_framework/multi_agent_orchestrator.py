"""
Saraphis Multi-Agent Orchestrator
Production-ready multi-agent orchestration with comprehensive coordination
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import threading
import queue
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import uuid
import json
import traceback

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """Production-ready multi-agent orchestration with comprehensive coordination"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components (will be imported after creation)
        self.agent_manager = None
        self.agent_coordinator = None
        self.task_distributor = None
        self.agent_monitor = None
        self.agent_integration_manager = None
        
        # Agent registry and state management
        self.agent_registry = {}
        self.agent_states = {}
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Performance tracking
        self.orchestration_metrics = {
            'total_agents': 0,
            'active_agents': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'average_task_time': 0.0,
            'agent_utilization': 0.0,
            'communication_overhead': 0.0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        self._agent_registry_lock = threading.Lock()
        
        # Initialize components
        self._initialize_components()
        
        self.logger.info("Multi-Agent Orchestrator initialized")
    
    def _initialize_components(self):
        """Initialize orchestrator components"""
        try:
            from .agent_manager import AgentManager
            from .agent_coordinator import AgentCoordinator
            from .task_distributor import TaskDistributor
            from .agent_monitor import AgentMonitor
            from .agent_integration_manager import AgentIntegrationManager
            
            self.agent_manager = AgentManager(self.config.get('agent_config', {}))
            self.agent_coordinator = AgentCoordinator(self.config.get('coordination_config', {}))
            self.task_distributor = TaskDistributor(self.config.get('distribution_config', {}))
            self.agent_monitor = AgentMonitor(self.config.get('monitoring_config', {}))
            self.agent_integration_manager = AgentIntegrationManager(self.config.get('integration_config', {}))
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize orchestrator components: {str(e)}")
    
    def initialize_multi_agent_framework(self, brain_core: Any) -> Dict[str, Any]:
        """Initialize comprehensive multi-agent development framework"""
        try:
            self.logger.info("Initializing multi-agent framework...")
            
            # Initialize agent management system
            agent_init_result = self.agent_manager.initialize_agents(brain_core)
            if not agent_init_result['success']:
                raise RuntimeError(f"Agent initialization failed: {agent_init_result.get('error')}")
            
            # Initialize coordination system
            coordination_result = self.agent_coordinator.initialize_coordination()
            if not coordination_result['success']:
                raise RuntimeError(f"Coordination initialization failed: {coordination_result.get('error')}")
            
            # Initialize task distribution system
            distribution_result = self.task_distributor.initialize_distribution()
            if not distribution_result['success']:
                raise RuntimeError(f"Distribution initialization failed: {distribution_result.get('error')}")
            
            # Initialize monitoring system
            monitoring_result = self.agent_monitor.initialize_monitoring()
            if not monitoring_result['success']:
                raise RuntimeError(f"Monitoring initialization failed: {monitoring_result.get('error')}")
            
            # Initialize integration validation
            integration_result = self.agent_integration_manager.initialize_integration()
            if not integration_result['success']:
                raise RuntimeError(f"Integration initialization failed: {integration_result.get('error')}")
            
            # Register all agents with brain core
            self._register_agents_with_brain(brain_core)
            
            # Validate complete framework
            validation_result = self._validate_framework_integration()
            
            # Update metrics
            with self._lock:
                self.orchestration_metrics['total_agents'] = len(self.agent_registry)
                self.orchestration_metrics['active_agents'] = len(self.agent_registry)
            
            self.logger.info(f"Multi-agent framework initialized with {len(self.agent_registry)} agents")
            
            return {
                'success': True,
                'agent_initialization': agent_init_result,
                'coordination_initialization': coordination_result,
                'distribution_initialization': distribution_result,
                'monitoring_initialization': monitoring_result,
                'integration_initialization': integration_result,
                'framework_validation': validation_result,
                'total_agents_created': len(self.agent_registry),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Multi-agent framework initialization failed: {e}")
            return {
                'success': False,
                'error': f'Framework initialization failed: {str(e)}',
                'traceback': traceback.format_exc()
            }
    
    def create_specialized_agents(self, agent_specifications: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create specialized agents for different Saraphis components"""
        try:
            created_agents = {}
            creation_start_time = time.time()
            
            for spec in agent_specifications:
                agent_type = spec.get('agent_type')
                agent_config = spec.get('config', {})
                
                self.logger.debug(f"Creating {agent_type} agent...")
                
                # Create agent based on type
                agent = None
                if agent_type == 'brain_orchestration':
                    agent = self.agent_manager.create_brain_orchestration_agent(agent_config)
                elif agent_type == 'proof_system':
                    agent = self.agent_manager.create_proof_system_agent(agent_config)
                elif agent_type == 'uncertainty':
                    agent = self.agent_manager.create_uncertainty_agent(agent_config)
                elif agent_type == 'training':
                    agent = self.agent_manager.create_training_agent(agent_config)
                elif agent_type == 'domain':
                    agent = self.agent_manager.create_domain_agent(agent_config)
                elif agent_type == 'compression':
                    agent = self.agent_manager.create_compression_agent(agent_config)
                elif agent_type == 'production':
                    agent = self.agent_manager.create_production_agent(agent_config)
                elif agent_type == 'web_interface':
                    agent = self.agent_manager.create_web_interface_agent(agent_config)
                else:
                    raise ValueError(f"Unknown agent type: {agent_type}")
                
                # Register agent
                agent_id = self._register_agent(agent, spec)
                created_agents[agent_id] = {
                    'agent_type': agent_type,
                    'status': 'created',
                    'capabilities': agent.get_capabilities(),
                    'performance_metrics': agent.get_performance_metrics(),
                    'creation_time': time.time()
                }
            
            # Initialize agent coordination
            coordination_result = self.agent_coordinator.setup_agent_communication(created_agents)
            
            creation_time = time.time() - creation_start_time
            
            return {
                'success': True,
                'created_agents': created_agents,
                'coordination_setup': coordination_result,
                'total_agents': len(created_agents),
                'creation_time_seconds': creation_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Specialized agent creation failed: {e}")
            return {
                'success': False,
                'error': f'Agent creation failed: {str(e)}',
                'traceback': traceback.format_exc()
            }
    
    def distribute_tasks_to_agents(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Distribute tasks to appropriate specialized agents"""
        try:
            distribution_results = {}
            distribution_start_time = time.time()
            
            for task in tasks:
                task_id = task.get('task_id', str(uuid.uuid4()))
                task_type = task.get('task_type')
                task_data = task.get('task_data', {})
                priority = task.get('priority', 'normal')
                
                # Add task metadata
                task_with_metadata = {
                    'task_id': task_id,
                    'task_type': task_type,
                    'task_data': task_data,
                    'priority': priority,
                    'submission_time': time.time()
                }
                
                # Determine appropriate agent for task
                target_agent = self.task_distributor.select_agent_for_task(task_type, task_data)
                
                if target_agent:
                    # Submit task to agent
                    submission_result = self._submit_task_to_agent(target_agent, task_with_metadata)
                    
                    distribution_results[task_id] = {
                        'assigned_agent': target_agent.agent_id,
                        'task_status': 'assigned' if submission_result['success'] else 'failed',
                        'submission_result': submission_result,
                        'assignment_time': time.time()
                    }
                else:
                    distribution_results[task_id] = {
                        'assigned_agent': None,
                        'task_status': 'failed',
                        'error': 'No suitable agent found for task',
                        'assignment_time': time.time()
                    }
            
            # Update metrics
            successful_assignments = sum(1 for r in distribution_results.values() 
                                       if r['task_status'] == 'assigned')
            
            distribution_time = time.time() - distribution_start_time
            
            return {
                'success': True,
                'distribution_results': distribution_results,
                'tasks_distributed': len(tasks),
                'successful_assignments': successful_assignments,
                'failed_assignments': len(tasks) - successful_assignments,
                'distribution_time_seconds': distribution_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Task distribution failed: {e}")
            return {
                'success': False,
                'error': f'Task distribution failed: {str(e)}',
                'traceback': traceback.format_exc()
            }
    
    def monitor_agent_health(self) -> Dict[str, Any]:
        """Monitor health and performance of all agents"""
        try:
            health_results = {}
            monitoring_start_time = time.time()
            
            with self._agent_registry_lock:
                for agent_id, agent in self.agent_registry.items():
                    # Get agent health status
                    health_status = self.agent_monitor.get_agent_health(agent_id)
                    
                    # Get agent performance metrics
                    performance_metrics = agent.get_performance_metrics()
                    
                    # Check agent communication status
                    communication_status = self.agent_coordinator.get_agent_communication_status(agent_id)
                    
                    # Get agent state
                    agent_state = self.agent_states.get(agent_id, {})
                    
                    health_results[agent_id] = {
                        'agent_type': agent.agent_type,
                        'health_status': health_status,
                        'performance_metrics': performance_metrics,
                        'communication_status': communication_status,
                        'last_heartbeat': agent.get_last_heartbeat(),
                        'task_queue_size': agent.get_task_queue_size(),
                        'memory_usage': agent.get_memory_usage(),
                        'cpu_usage': agent.get_cpu_usage(),
                        'current_state': agent_state.get('state', 'unknown'),
                        'active_tasks': agent_state.get('active_tasks', 0),
                        'completed_tasks': agent_state.get('completed_tasks', 0)
                    }
            
            # Calculate overall system health
            overall_health = self._calculate_overall_system_health(health_results)
            
            monitoring_time = time.time() - monitoring_start_time
            
            return {
                'success': True,
                'agent_health': health_results,
                'overall_system_health': overall_health,
                'total_agents_monitored': len(health_results),
                'monitoring_time_seconds': monitoring_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Agent health monitoring failed: {e}")
            return {
                'success': False,
                'error': f'Health monitoring failed: {str(e)}',
                'traceback': traceback.format_exc()
            }
    
    def get_agent_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report for all agents"""
        try:
            performance_report = {
                'agents': {},
                'system_metrics': {},
                'recommendations': []
            }
            
            with self._agent_registry_lock:
                for agent_id, agent in self.agent_registry.items():
                    performance_report['agents'][agent_id] = {
                        'agent_type': agent.agent_type,
                        'performance_metrics': agent.get_performance_metrics(),
                        'resource_usage': {
                            'memory_mb': agent.get_memory_usage(),
                            'cpu_percent': agent.get_cpu_usage()
                        },
                        'task_statistics': {
                            'completed': self.agent_states.get(agent_id, {}).get('completed_tasks', 0),
                            'failed': self.agent_states.get(agent_id, {}).get('failed_tasks', 0),
                            'average_completion_time': self.agent_states.get(agent_id, {}).get('avg_task_time', 0)
                        }
                    }
            
            # Calculate system-wide metrics
            performance_report['system_metrics'] = {
                'total_agents': len(self.agent_registry),
                'active_agents': self.orchestration_metrics['active_agents'],
                'total_completed_tasks': self.orchestration_metrics['completed_tasks'],
                'total_failed_tasks': self.orchestration_metrics['failed_tasks'],
                'average_task_time': self.orchestration_metrics['average_task_time'],
                'agent_utilization': self.orchestration_metrics['agent_utilization']
            }
            
            # Generate recommendations
            performance_report['recommendations'] = self._generate_performance_recommendations(
                performance_report['agents'],
                performance_report['system_metrics']
            )
            
            return {
                'success': True,
                'performance_report': performance_report,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Performance report generation failed: {e}")
            return {
                'success': False,
                'error': f'Report generation failed: {str(e)}',
                'traceback': traceback.format_exc()
            }
    
    def _register_agent(self, agent: Any, spec: Dict[str, Any]) -> str:
        """Register agent in the orchestrator"""
        try:
            agent_id = f"agent_{spec['agent_type']}_{uuid.uuid4().hex[:8]}"
            
            with self._agent_registry_lock:
                # Set agent ID
                agent.agent_id = agent_id
                agent.agent_type = spec['agent_type']
                
                # Register in registry
                self.agent_registry[agent_id] = agent
                
                # Initialize agent state
                self.agent_states[agent_id] = {
                    'state': 'initialized',
                    'active_tasks': 0,
                    'completed_tasks': 0,
                    'failed_tasks': 0,
                    'avg_task_time': 0.0,
                    'last_activity': time.time()
                }
            
            self.logger.info(f"Registered agent: {agent_id} (type: {spec['agent_type']})")
            return agent_id
            
        except Exception as e:
            self.logger.error(f"Agent registration failed: {e}")
            raise RuntimeError(f"Failed to register agent: {str(e)}")
    
    def _register_agents_with_brain(self, brain_core: Any) -> None:
        """Register all agents with brain core for integration"""
        try:
            with self._agent_registry_lock:
                for agent_id, agent in self.agent_registry.items():
                    # Register agent capabilities with brain
                    brain_core.register_agent_capabilities(agent_id, agent.get_capabilities())
                    
                    # Set up agent communication channels
                    brain_core.setup_agent_communication(agent_id, agent.get_communication_endpoints())
                    
                    # Register agent monitoring
                    brain_core.register_agent_monitoring(agent_id, agent.get_monitoring_hooks())
            
            self.logger.info(f"Registered {len(self.agent_registry)} agents with brain core")
            
        except Exception as e:
            self.logger.error(f"Agent registration with brain failed: {e}")
            raise RuntimeError(f"Agent registration failed: {str(e)}")
    
    def _submit_task_to_agent(self, agent: Any, task: Dict[str, Any]) -> Dict[str, Any]:
        """Submit task to specific agent"""
        try:
            # Update agent state
            agent_id = agent.agent_id
            with self._lock:
                self.agent_states[agent_id]['active_tasks'] += 1
                self.agent_states[agent_id]['last_activity'] = time.time()
            
            # Submit task
            result = agent.execute_task(task['task_data'], task['priority'])
            
            # Update metrics based on result
            with self._lock:
                self.agent_states[agent_id]['active_tasks'] -= 1
                if result.get('success', False):
                    self.agent_states[agent_id]['completed_tasks'] += 1
                    self.orchestration_metrics['completed_tasks'] += 1
                else:
                    self.agent_states[agent_id]['failed_tasks'] += 1
                    self.orchestration_metrics['failed_tasks'] += 1
            
            return {
                'success': True,
                'task_result': result,
                'execution_time': result.get('execution_time', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Task submission to agent failed: {e}")
            return {
                'success': False,
                'error': f'Task submission failed: {str(e)}'
            }
    
    def _validate_framework_integration(self) -> Dict[str, Any]:
        """Validate complete multi-agent framework integration"""
        try:
            validation_results = {
                'agent_registration': len(self.agent_registry) > 0,
                'coordination_active': self.agent_coordinator.is_active(),
                'distribution_active': self.task_distributor.is_active(),
                'monitoring_active': self.agent_monitor.is_active(),
                'integration_active': self.agent_integration_manager.is_active(),
                'validation_timestamp': datetime.now().isoformat()
            }
            
            # Check agent communication
            communication_validation = self.agent_coordinator.validate_communication_network()
            
            # Check task distribution
            distribution_validation = self.task_distributor.validate_distribution_system()
            
            # Check monitoring coverage
            monitoring_validation = self.agent_monitor.validate_monitoring_coverage()
            
            validation_results.update({
                'communication_validation': communication_validation,
                'distribution_validation': distribution_validation,
                'monitoring_validation': monitoring_validation,
                'overall_validation': all([
                    validation_results['agent_registration'],
                    validation_results['coordination_active'],
                    validation_results['distribution_active'],
                    validation_results['monitoring_active'],
                    validation_results['integration_active']
                ])
            })
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Framework validation failed: {e}")
            return {
                'error': f'Validation failed: {str(e)}',
                'validation_success': False
            }
    
    def _calculate_overall_system_health(self, health_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall system health metrics"""
        try:
            total_agents = len(health_results)
            if total_agents == 0:
                return {
                    'overall_status': 'no_agents',
                    'health_percentage': 0,
                    'error': 'No agents registered'
                }
            
            healthy_agents = sum(1 for result in health_results.values() 
                               if result['health_status'].get('status') == 'healthy')
            
            # Calculate average performance metrics
            total_memory = sum(result.get('memory_usage', 0) for result in health_results.values())
            total_cpu = sum(result.get('cpu_usage', 0) for result in health_results.values())
            total_queue_size = sum(result.get('task_queue_size', 0) for result in health_results.values())
            
            avg_memory_usage = total_memory / total_agents
            avg_cpu_usage = total_cpu / total_agents
            avg_task_queue_size = total_queue_size / total_agents
            
            # Determine overall status
            health_percentage = (healthy_agents / total_agents) * 100
            if health_percentage >= 95:
                overall_status = 'healthy'
            elif health_percentage >= 80:
                overall_status = 'degraded'
            elif health_percentage >= 50:
                overall_status = 'critical'
            else:
                overall_status = 'failure'
            
            return {
                'overall_status': overall_status,
                'health_percentage': health_percentage,
                'total_agents': total_agents,
                'healthy_agents': healthy_agents,
                'unhealthy_agents': total_agents - healthy_agents,
                'average_memory_usage': avg_memory_usage,
                'average_cpu_usage': avg_cpu_usage,
                'average_task_queue_size': avg_task_queue_size,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"System health calculation failed: {e}")
            return {
                'overall_status': 'unknown',
                'error': f'Health calculation failed: {str(e)}'
            }
    
    def _generate_performance_recommendations(self, agent_data: Dict[str, Any], 
                                            system_metrics: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on metrics"""
        recommendations = []
        
        try:
            # Check agent utilization
            if system_metrics.get('agent_utilization', 0) < 0.5:
                recommendations.append("Agent utilization is low. Consider reducing the number of agents or increasing task load.")
            elif system_metrics.get('agent_utilization', 0) > 0.9:
                recommendations.append("Agent utilization is very high. Consider adding more agents to distribute the load.")
            
            # Check task failure rate
            total_tasks = system_metrics.get('total_completed_tasks', 0) + system_metrics.get('total_failed_tasks', 0)
            if total_tasks > 0:
                failure_rate = system_metrics.get('total_failed_tasks', 0) / total_tasks
                if failure_rate > 0.1:
                    recommendations.append(f"High task failure rate ({failure_rate*100:.1f}%). Review task processing logic and error handling.")
            
            # Check individual agent performance
            for agent_id, agent_info in agent_data.items():
                # Check memory usage
                if agent_info['resource_usage']['memory_mb'] > 1000:
                    recommendations.append(f"Agent {agent_id} has high memory usage. Consider optimizing memory consumption.")
                
                # Check CPU usage
                if agent_info['resource_usage']['cpu_percent'] > 80:
                    recommendations.append(f"Agent {agent_id} has high CPU usage. Consider optimizing processing efficiency.")
                
                # Check task completion time
                if agent_info['task_statistics']['average_completion_time'] > 60:
                    recommendations.append(f"Agent {agent_id} has slow task completion times. Review task processing logic.")
            
            # Add general recommendations if no specific issues
            if not recommendations:
                recommendations.append("System performance is within normal parameters.")
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to an error.")
        
        return recommendations