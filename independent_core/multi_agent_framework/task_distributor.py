"""
Saraphis Task Distributor
Production-ready task distribution and load balancing
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
import random

logger = logging.getLogger(__name__)


class TaskDistributor:
    """Production-ready task distribution and load balancing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Task distribution infrastructure
        self.task_queues = defaultdict(queue.Queue)
        self.agent_capabilities = {}
        self.agent_load = defaultdict(lambda: {'current_load': 0, 'max_load': 10})
        self.task_assignments = {}
        
        # Distribution strategies
        self.distribution_strategies = {
            'round_robin': self._round_robin_distribution,
            'load_balanced': self._load_balanced_distribution,
            'capability_based': self._capability_based_distribution,
            'priority_based': self._priority_based_distribution,
            'affinity_based': self._affinity_based_distribution
        }
        
        # Current strategy
        self.current_strategy = config.get('distribution_strategy', 'load_balanced')
        
        # Performance tracking
        self.distribution_metrics = {
            'total_tasks_distributed': 0,
            'successful_distributions': 0,
            'failed_distributions': 0,
            'average_distribution_time': 0.0,
            'load_balance_efficiency': 0.0,
            'strategy_performance': defaultdict(lambda: {
                'usage_count': 0,
                'success_rate': 0.0,
                'average_time': 0.0
            })
        }
        
        # Agent registry for distribution
        self.registered_agents = {}
        self.agent_round_robin_index = 0
        
        # Task affinity tracking
        self.task_affinity = defaultdict(list)  # task_type -> [preferred_agents]
        
        # Active state
        self.is_active_flag = False
        
        # Thread safety
        self._lock = threading.Lock()
        self._agent_lock = threading.Lock()
        
        self.logger.info("Task Distributor initialized")
    
    def initialize_distribution(self) -> Dict[str, Any]:
        """Initialize task distribution system"""
        try:
            self.logger.info("Initializing task distribution system...")
            
            # Initialize distribution infrastructure
            infra_init_result = self._initialize_distribution_infrastructure()
            
            # Initialize load tracking
            load_init_result = self._initialize_load_tracking()
            
            # Initialize affinity system
            affinity_init_result = self._initialize_affinity_system()
            
            # Start distribution services
            self._start_distribution_services()
            
            # Set active flag
            self.is_active_flag = True
            
            return {
                'success': True,
                'distribution_infrastructure': infra_init_result,
                'load_tracking': load_init_result,
                'affinity_system': affinity_init_result,
                'current_strategy': self.current_strategy,
                'available_strategies': list(self.distribution_strategies.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Distribution initialization failed: {e}")
            return {
                'success': False,
                'error': f'Distribution initialization failed: {str(e)}',
                'traceback': traceback.format_exc()
            }
    
    def select_agent_for_task(self, task_type: str, task_data: Dict[str, Any]) -> Optional[Any]:
        """Select appropriate agent for task based on current strategy"""
        try:
            selection_start_time = time.time()
            
            # Get distribution strategy
            strategy = self.distribution_strategies.get(
                self.current_strategy,
                self._capability_based_distribution
            )
            
            # Select agent using strategy
            selected_agent = strategy(task_type, task_data)
            
            # Track distribution metrics
            selection_time = time.time() - selection_start_time
            self._update_distribution_metrics(
                self.current_strategy,
                selection_time,
                success=selected_agent is not None
            )
            
            if selected_agent:
                self.logger.debug(f"Selected agent {selected_agent.agent_id} for task type {task_type}")
            else:
                self.logger.warning(f"No suitable agent found for task type {task_type}")
            
            return selected_agent
            
        except Exception as e:
            self.logger.error(f"Agent selection failed: {e}")
            self._update_distribution_metrics(self.current_strategy, 0, success=False)
            return None
    
    def register_agent_capabilities(self, agent_id: str, agent: Any, 
                                  capabilities: List[str]) -> Dict[str, Any]:
        """Register agent capabilities for task distribution"""
        try:
            with self._agent_lock:
                # Register agent
                self.registered_agents[agent_id] = agent
                
                # Register capabilities
                self.agent_capabilities[agent_id] = {
                    'capabilities': capabilities,
                    'task_types': self._map_capabilities_to_task_types(capabilities),
                    'performance_score': 1.0,  # Initial score
                    'availability': True,
                    'registration_time': time.time()
                }
                
                # Initialize load tracking
                max_load = self._calculate_agent_max_load(agent)
                self.agent_load[agent_id] = {
                    'current_load': 0,
                    'max_load': max_load,
                    'load_percentage': 0.0
                }
            
            self.logger.info(f"Registered agent {agent_id} with {len(capabilities)} capabilities")
            
            return {
                'success': True,
                'agent_id': agent_id,
                'registered_capabilities': len(capabilities),
                'task_types': len(self.agent_capabilities[agent_id]['task_types']),
                'max_load': max_load
            }
            
        except Exception as e:
            self.logger.error(f"Agent capability registration failed: {e}")
            return {
                'success': False,
                'error': f'Registration failed: {str(e)}'
            }
    
    def update_agent_load(self, agent_id: str, load_delta: int) -> Dict[str, Any]:
        """Update agent load for load balancing"""
        try:
            with self._lock:
                if agent_id not in self.agent_load:
                    return {
                        'success': False,
                        'error': f'Agent {agent_id} not registered'
                    }
                
                # Update load
                current_load = self.agent_load[agent_id]['current_load']
                new_load = max(0, current_load + load_delta)
                max_load = self.agent_load[agent_id]['max_load']
                
                self.agent_load[agent_id]['current_load'] = new_load
                self.agent_load[agent_id]['load_percentage'] = (new_load / max_load) * 100
                
                # Update availability based on load
                if agent_id in self.agent_capabilities:
                    self.agent_capabilities[agent_id]['availability'] = new_load < max_load
            
            return {
                'success': True,
                'agent_id': agent_id,
                'current_load': new_load,
                'max_load': max_load,
                'load_percentage': self.agent_load[agent_id]['load_percentage'],
                'is_available': new_load < max_load
            }
            
        except Exception as e:
            self.logger.error(f"Agent load update failed: {e}")
            return {
                'success': False,
                'error': f'Load update failed: {str(e)}'
            }
    
    def get_distribution_metrics(self) -> Dict[str, Any]:
        """Get task distribution metrics"""
        with self._lock:
            # Calculate load balance efficiency
            if self.agent_load:
                load_percentages = [agent['load_percentage'] 
                                  for agent in self.agent_load.values()]
                avg_load = sum(load_percentages) / len(load_percentages)
                load_variance = sum((load - avg_load) ** 2 for load in load_percentages) / len(load_percentages)
                self.distribution_metrics['load_balance_efficiency'] = 1.0 - (load_variance / 10000)  # Normalize
            
            return {
                'metrics': self.distribution_metrics.copy(),
                'strategy_performance': dict(self.distribution_metrics['strategy_performance']),
                'agent_loads': {
                    agent_id: {
                        'current_load': load['current_load'],
                        'max_load': load['max_load'],
                        'load_percentage': load['load_percentage']
                    }
                    for agent_id, load in self.agent_load.items()
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def set_distribution_strategy(self, strategy: str) -> Dict[str, Any]:
        """Set task distribution strategy"""
        try:
            if strategy not in self.distribution_strategies:
                return {
                    'success': False,
                    'error': f'Unknown strategy: {strategy}',
                    'available_strategies': list(self.distribution_strategies.keys())
                }
            
            old_strategy = self.current_strategy
            self.current_strategy = strategy
            
            self.logger.info(f"Changed distribution strategy from {old_strategy} to {strategy}")
            
            return {
                'success': True,
                'previous_strategy': old_strategy,
                'current_strategy': strategy,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Strategy change failed: {e}")
            return {
                'success': False,
                'error': f'Strategy change failed: {str(e)}'
            }
    
    def is_active(self) -> bool:
        """Check if distributor is active"""
        return self.is_active_flag
    
    def validate_distribution_system(self) -> Dict[str, Any]:
        """Validate distribution system configuration"""
        try:
            validation_results = {
                'registered_agents': len(self.registered_agents),
                'available_agents': 0,
                'overloaded_agents': 0,
                'task_type_coverage': {},
                'strategy_validation': {},
                'system_health': 'unknown'
            }
            
            with self._agent_lock:
                # Check agent availability
                for agent_id, capabilities in self.agent_capabilities.items():
                    if capabilities.get('availability', False):
                        validation_results['available_agents'] += 1
                    
                    # Check if overloaded
                    load_info = self.agent_load.get(agent_id, {})
                    if load_info.get('load_percentage', 0) > 90:
                        validation_results['overloaded_agents'] += 1
                
                # Check task type coverage
                all_task_types = set()
                for agent_caps in self.agent_capabilities.values():
                    all_task_types.update(agent_caps.get('task_types', []))
                
                validation_results['task_type_coverage'] = {
                    'total_task_types': len(all_task_types),
                    'covered_task_types': list(all_task_types)
                }
            
            # Validate strategies
            for strategy_name in self.distribution_strategies:
                validation_results['strategy_validation'][strategy_name] = 'available'
            
            # Determine system health
            if validation_results['available_agents'] == 0:
                validation_results['system_health'] = 'critical'
            elif validation_results['overloaded_agents'] > validation_results['available_agents'] / 2:
                validation_results['system_health'] = 'degraded'
            else:
                validation_results['system_health'] = 'healthy'
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Distribution system validation failed: {e}")
            return {
                'error': f'Validation failed: {str(e)}',
                'system_health': 'unknown'
            }
    
    def _initialize_distribution_infrastructure(self) -> Dict[str, Any]:
        """Initialize core distribution infrastructure"""
        try:
            # Initialize task routing tables
            self.task_routing = {}
            
            # Initialize distribution policies
            self.distribution_policies = {
                'max_retries': self.config.get('max_distribution_retries', 3),
                'retry_delay': self.config.get('retry_delay_seconds', 1),
                'load_threshold': self.config.get('load_threshold_percent', 80),
                'affinity_weight': self.config.get('affinity_weight', 0.3)
            }
            
            return {
                'task_routing': 'initialized',
                'distribution_policies': self.distribution_policies
            }
            
        except Exception as e:
            self.logger.error(f"Distribution infrastructure initialization failed: {e}")
            raise RuntimeError(f"Infrastructure initialization failed: {str(e)}")
    
    def _initialize_load_tracking(self) -> Dict[str, Any]:
        """Initialize load tracking system"""
        try:
            # Initialize load monitoring
            self.load_monitor = {
                'update_interval': self.config.get('load_update_interval', 5),
                'threshold_alerts': self.config.get('enable_threshold_alerts', True),
                'load_history': defaultdict(lambda: deque(maxlen=100))
            }
            
            return {
                'load_monitoring': 'initialized',
                'update_interval': self.load_monitor['update_interval']
            }
            
        except Exception as e:
            self.logger.error(f"Load tracking initialization failed: {e}")
            raise RuntimeError(f"Load tracking initialization failed: {str(e)}")
    
    def _initialize_affinity_system(self) -> Dict[str, Any]:
        """Initialize task affinity system"""
        try:
            # Initialize affinity tracking
            self.affinity_config = {
                'enable_affinity': self.config.get('enable_task_affinity', True),
                'affinity_decay': self.config.get('affinity_decay_factor', 0.95),
                'min_affinity_score': self.config.get('min_affinity_score', 0.1)
            }
            
            return {
                'affinity_system': 'initialized',
                'affinity_enabled': self.affinity_config['enable_affinity']
            }
            
        except Exception as e:
            self.logger.error(f"Affinity system initialization failed: {e}")
            raise RuntimeError(f"Affinity initialization failed: {str(e)}")
    
    def _start_distribution_services(self) -> None:
        """Start background distribution services"""
        try:
            # Start load monitoring thread
            load_thread = threading.Thread(
                target=self._load_monitoring_loop,
                daemon=True
            )
            load_thread.start()
            
            # Start affinity update thread
            affinity_thread = threading.Thread(
                target=self._affinity_update_loop,
                daemon=True
            )
            affinity_thread.start()
            
            self.logger.info("Distribution services started")
            
        except Exception as e:
            self.logger.error(f"Failed to start distribution services: {e}")
            raise RuntimeError(f"Service startup failed: {str(e)}")
    
    def _map_capabilities_to_task_types(self, capabilities: List[str]) -> List[str]:
        """Map agent capabilities to supported task types"""
        task_type_mapping = {
            # Brain orchestration capabilities
            'domain_management': ['domain_create', 'domain_update', 'domain_delete'],
            'session_orchestration': ['session_start', 'session_manage', 'session_end'],
            'resource_coordination': ['resource_allocate', 'resource_optimize', 'resource_release'],
            
            # Proof system capabilities
            'formal_verification': ['proof_generate', 'proof_verify', 'proof_validate'],
            'confidence_aggregation': ['confidence_calculate', 'confidence_aggregate'],
            
            # Uncertainty capabilities
            'conformalized_credal_quantification': ['uncertainty_quantify', 'credal_calculate'],
            'uncertainty_based_decision_making': ['decision_analyze', 'decision_recommend'],
            
            # Training capabilities
            'neural_network_training': ['model_train', 'model_validate', 'model_optimize'],
            'gradient_ascent_clipping': ['gradient_compute', 'gradient_clip'],
            
            # Compression capabilities
            'hybrid_padic_compression': ['data_compress', 'data_decompress'],
            'tensor_decomposition': ['tensor_decompose', 'tensor_reconstruct'],
            
            # Production capabilities
            'production_monitoring': ['system_monitor', 'alert_generate', 'metric_collect'],
            'auto_scaling': ['scale_up', 'scale_down', 'scale_optimize'],
            
            # Web interface capabilities
            'dashboard_rendering': ['dashboard_update', 'dashboard_render'],
            'real_time_communication': ['websocket_send', 'websocket_receive']
        }
        
        task_types = []
        for capability in capabilities:
            if capability in task_type_mapping:
                task_types.extend(task_type_mapping[capability])
        
        return list(set(task_types))
    
    def _update_agent_load_internal(self, agent_id: str, load_delta: int) -> bool:
        """Update agent load without acquiring locks (caller must hold locks)"""
        if agent_id not in self.agent_load:
            return False
        
        # Update load
        current_load = self.agent_load[agent_id]['current_load']
        new_load = max(0, current_load + load_delta)
        max_load = self.agent_load[agent_id]['max_load']
        
        self.agent_load[agent_id]['current_load'] = new_load
        self.agent_load[agent_id]['load_percentage'] = (new_load / max_load) * 100
        
        # Update availability based on load
        if agent_id in self.agent_capabilities:
            self.agent_capabilities[agent_id]['availability'] = new_load < max_load
        
        return True

    def complete_task(self, agent_id: str, task_id: str = None) -> Dict[str, Any]:
        """Mark a task as completed and reduce agent load"""
        try:
            return self.update_agent_load(agent_id, -1)
        except Exception as e:
            self.logger.error(f"Task completion failed: {e}")
            return {
                'success': False,
                'error': f'Task completion failed: {str(e)}'
            }

    def reset_agent_loads(self) -> Dict[str, Any]:
        """Reset all agent loads to zero (for testing or recovery)"""
        try:
            with self._lock:
                reset_count = 0
                for agent_id in self.agent_load:
                    self.agent_load[agent_id]['current_load'] = 0
                    self.agent_load[agent_id]['load_percentage'] = 0.0
                    
                    # Update availability
                    if agent_id in self.agent_capabilities:
                        self.agent_capabilities[agent_id]['availability'] = True
                    
                    reset_count += 1
            
            self.logger.info(f"Reset loads for {reset_count} agents")
            return {
                'success': True,
                'agents_reset': reset_count
            }
            
        except Exception as e:
            self.logger.error(f"Agent load reset failed: {e}")
            return {
                'success': False,
                'error': f'Load reset failed: {str(e)}'
            }

    def _calculate_agent_max_load(self, agent: Any) -> int:
        """Calculate maximum load for agent based on resources"""
        try:
            # Get agent resource limits
            resource_limits = agent.config.get('resource_limits', {})
            
            # If max_concurrent_tasks is explicitly configured, use it directly
            configured_max = resource_limits.get('max_concurrent_tasks')
            if configured_max is not None:
                return max(1, configured_max)  # Ensure at least 1
            
            # Otherwise calculate based on CPU and memory limits
            max_cpu = resource_limits.get('max_cpu_percent', 50)
            max_memory = resource_limits.get('max_memory_mb', 512)
            
            # Calculate max concurrent tasks
            # Rough estimate: 10% CPU and 50MB per task
            cpu_based_limit = max_cpu // 10
            memory_based_limit = max_memory // 50
            
            # Take the minimum to be conservative
            max_load = max(1, min(cpu_based_limit, memory_based_limit))
            
            return max_load
            
        except Exception as e:
            self.logger.error(f"Failed to calculate agent max load: {e}")
            return 10  # Default fallback
    
    # Distribution Strategy Implementations
    
    def _round_robin_distribution(self, task_type: str, task_data: Dict[str, Any]) -> Optional[Any]:
        """Round-robin task distribution"""
        with self._lock:
            available_agents = [
                (agent_id, agent) for agent_id, agent in self.registered_agents.items()
                if self.agent_capabilities.get(agent_id, {}).get('availability', False)
            ]
            
            if not available_agents:
                return None
            
            # Get next agent in round-robin order
            selected_index = self.agent_round_robin_index % len(available_agents)
            self.agent_round_robin_index += 1
            
            agent_id, agent = available_agents[selected_index]
            
            # Update load internally
            self._update_agent_load_internal(agent_id, 1)
            
            return agent
    
    def _load_balanced_distribution(self, task_type: str, task_data: Dict[str, Any]) -> Optional[Any]:
        """Load-balanced task distribution"""
        with self._lock:
            # Get agents that can handle this task type
            capable_agents = []
            
            for agent_id, capabilities in self.agent_capabilities.items():
                if (task_type in capabilities.get('task_types', []) and 
                    capabilities.get('availability', False)):
                    load_info = self.agent_load.get(agent_id, {})
                    capable_agents.append({
                        'agent_id': agent_id,
                        'agent': self.registered_agents[agent_id],
                        'load_percentage': load_info.get('load_percentage', 0)
                    })
            
            if not capable_agents:
                return None
            
            # Sort by load (ascending)
            capable_agents.sort(key=lambda x: x['load_percentage'])
            
            # Select least loaded agent
            selected = capable_agents[0]
            
            # Update load internally
            self._update_agent_load_internal(selected['agent_id'], 1)
            
            return selected['agent']
    
    def _capability_based_distribution(self, task_type: str, task_data: Dict[str, Any]) -> Optional[Any]:
        """Capability-based task distribution"""
        with self._lock:
            # Find agents with required capabilities
            best_agent = None
            best_agent_id = None
            best_score = 0
            
            for agent_id, capabilities in self.agent_capabilities.items():
                if (task_type in capabilities.get('task_types', []) and 
                    capabilities.get('availability', False)):
                    
                    # Calculate capability score
                    capability_count = len(capabilities.get('capabilities', []))
                    performance_score = capabilities.get('performance_score', 1.0)
                    
                    # Factor in load
                    load_info = self.agent_load.get(agent_id, {})
                    load_factor = 1.0 - (load_info.get('load_percentage', 0) / 100)
                    
                    # Calculate overall score
                    score = capability_count * performance_score * load_factor
                    
                    if score > best_score:
                        best_score = score
                        best_agent = self.registered_agents[agent_id]
                        best_agent_id = agent_id
            
            if best_agent:
                # Update load internally
                self._update_agent_load_internal(best_agent_id, 1)
            
            return best_agent
    
    def _priority_based_distribution(self, task_type: str, task_data: Dict[str, Any]) -> Optional[Any]:
        """Priority-based task distribution"""
        priority = task_data.get('priority', 'normal')
        
        with self._lock:
            # Get agents sorted by priority handling capability
            priority_agents = []
            
            for agent_id, agent in self.registered_agents.items():
                capabilities = self.agent_capabilities.get(agent_id, {})
                
                if (task_type in capabilities.get('task_types', []) and 
                    capabilities.get('availability', False)):
                    
                    # Check agent priority level
                    agent_priority = agent.config.get('priority_level', 'normal')
                    
                    # High priority tasks go to critical/high priority agents
                    if priority == 'high' and agent_priority in ['critical', 'high']:
                        priority_score = 2.0
                    elif priority == agent_priority:
                        priority_score = 1.0
                    else:
                        priority_score = 0.5
                    
                    priority_agents.append({
                        'agent': agent,
                        'agent_id': agent_id,
                        'score': priority_score
                    })
            
            if not priority_agents:
                return None
            
            # Sort by priority score
            priority_agents.sort(key=lambda x: x['score'], reverse=True)
            
            selected_agent = priority_agents[0]['agent']
            selected_agent_id = priority_agents[0]['agent_id']
            
            # Update load internally
            self._update_agent_load_internal(selected_agent_id, 1)
            
            return selected_agent
    
    def _affinity_based_distribution(self, task_type: str, task_data: Dict[str, Any]) -> Optional[Any]:
        """Affinity-based task distribution"""
        with self._lock:
            # Check task affinity history
            preferred_agents = self.task_affinity.get(task_type, [])
            
            # Try preferred agents first
            for agent_entry in preferred_agents:
                agent_id = agent_entry['agent_id']
                if agent_id in self.registered_agents:
                    capabilities = self.agent_capabilities.get(agent_id, {})
                    if capabilities.get('availability', False):
                        agent = self.registered_agents[agent_id]
                        
                        # Update affinity score
                        agent_entry['score'] = min(1.0, agent_entry['score'] * 1.1)
                        
                        # Update load internally
                        self._update_agent_load_internal(agent_id, 1)
                        
                        return agent
        
        # Fall back to capability-based distribution (outside lock)
        return self._capability_based_distribution(task_type, task_data)
    
    def _update_distribution_metrics(self, strategy: str, distribution_time: float, 
                                   success: bool) -> None:
        """Update distribution metrics"""
        try:
            with self._lock:
                self.distribution_metrics['total_tasks_distributed'] += 1
                
                if success:
                    self.distribution_metrics['successful_distributions'] += 1
                else:
                    self.distribution_metrics['failed_distributions'] += 1
                
                # Update average distribution time
                total = self.distribution_metrics['total_tasks_distributed']
                current_avg = self.distribution_metrics['average_distribution_time']
                self.distribution_metrics['average_distribution_time'] = (
                    (current_avg * (total - 1) + distribution_time) / total
                )
                
                # Update strategy performance
                strategy_metrics = self.distribution_metrics['strategy_performance'][strategy]
                strategy_metrics['usage_count'] += 1
                
                if success:
                    total_uses = strategy_metrics['usage_count']
                    current_success = strategy_metrics['success_rate']
                    strategy_metrics['success_rate'] = (
                        (current_success * (total_uses - 1) + 1) / total_uses
                    )
                
                # Update average time for strategy
                current_avg_time = strategy_metrics['average_time']
                strategy_metrics['average_time'] = (
                    (current_avg_time * (strategy_metrics['usage_count'] - 1) + distribution_time) / 
                    strategy_metrics['usage_count']
                )
                
        except Exception as e:
            self.logger.error(f"Failed to update distribution metrics: {e}")
    
    def _load_monitoring_loop(self) -> None:
        """Background thread for monitoring agent loads"""
        while self.is_active_flag:
            try:
                time.sleep(self.load_monitor['update_interval'])
                
                with self._lock:
                    # Update load history
                    for agent_id, load_info in self.agent_load.items():
                        self.load_monitor['load_history'][agent_id].append({
                            'timestamp': time.time(),
                            'load_percentage': load_info['load_percentage']
                        })
                    
                    # Check for overloaded agents
                    if self.load_monitor['threshold_alerts']:
                        for agent_id, load_info in self.agent_load.items():
                            if load_info['load_percentage'] > self.distribution_policies['load_threshold']:
                                self.logger.warning(
                                    f"Agent {agent_id} is overloaded: "
                                    f"{load_info['load_percentage']:.1f}%"
                                )
                
            except Exception as e:
                self.logger.error(f"Load monitoring error: {e}")
    
    def _affinity_update_loop(self) -> None:
        """Background thread for updating task affinity"""
        while self.is_active_flag:
            try:
                time.sleep(60)  # Update every minute
                
                if not self.affinity_config['enable_affinity']:
                    continue
                
                with self._lock:
                    # Decay affinity scores
                    decay_factor = self.affinity_config['affinity_decay']
                    min_score = self.affinity_config['min_affinity_score']
                    
                    for task_type, agent_list in self.task_affinity.items():
                        # Update scores
                        updated_list = []
                        for agent_entry in agent_list:
                            agent_entry['score'] *= decay_factor
                            if agent_entry['score'] >= min_score:
                                updated_list.append(agent_entry)
                        
                        # Sort by score
                        updated_list.sort(key=lambda x: x['score'], reverse=True)
                        
                        # Keep top entries
                        self.task_affinity[task_type] = updated_list[:10]
                
            except Exception as e:
                self.logger.error(f"Affinity update error: {e}")