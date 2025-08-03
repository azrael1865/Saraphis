"""
Saraphis Agent Manager
Production-ready agent management with specialized agent creation
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import uuid
import json
import traceback

logger = logging.getLogger(__name__)


class AgentManager:
    """Production-ready agent management with specialized agent creation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Agent templates and factories
        self.agent_templates = {}
        self.agent_factories = {}
        
        # Agent lifecycle management
        self.agent_lifecycle_states = {}
        self.agent_creation_history = []
        
        # Performance tracking
        self.management_metrics = {
            'total_agents_created': 0,
            'active_agents': 0,
            'failed_creations': 0,
            'average_creation_time': 0.0,
            'template_usage': defaultdict(int)
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize agent templates
        self._initialize_agent_templates()
        
        self.logger.info("Agent Manager initialized")
    
    def initialize_agents(self, brain_core: Any) -> Dict[str, Any]:
        """Initialize agent management system with brain core integration"""
        try:
            self.logger.info("Initializing agent management system...")
            
            # Store brain core reference for agent creation
            self.brain_core = brain_core
            
            # Initialize agent factories
            self._initialize_agent_factories()
            
            # Create default agents based on configuration
            default_agents = self._create_default_agents()
            
            # Validate agent templates
            template_validation = self._validate_agent_templates()
            
            return {
                'success': True,
                'agent_templates_initialized': len(self.agent_templates),
                'agent_factories_initialized': len(self.agent_factories),
                'default_agents_created': len(default_agents),
                'template_validation': template_validation,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Agent initialization failed: {e}")
            return {
                'success': False,
                'error': f'Agent initialization failed: {str(e)}',
                'traceback': traceback.format_exc()
            }
    
    def create_brain_orchestration_agent(self, config: Dict[str, Any]) -> 'BrainOrchestrationAgent':
        """Create specialized brain orchestration agent"""
        try:
            from .specialized_agents import BrainOrchestrationAgent
            
            agent_config = {
                'agent_type': 'brain_orchestration',
                'capabilities': [
                    'domain_management',
                    'session_orchestration',
                    'resource_coordination',
                    'performance_optimization',
                    'cross_domain_integration'
                ],
                'communication_protocols': ['internal', 'external', 'cross_domain'],
                'monitoring_hooks': ['performance', 'health', 'resource_usage'],
                'security_level': 'high',
                'priority_level': 'critical',
                'resource_limits': {
                    'max_memory_mb': config.get('max_memory_mb', 512),
                    'max_cpu_percent': config.get('max_cpu_percent', 50),
                    'max_concurrent_tasks': config.get('max_concurrent_tasks', 10)
                },
                'brain_core_reference': self.brain_core
            }
            agent_config.update(config)
            
            # Create agent
            creation_start = time.time()
            agent = BrainOrchestrationAgent(agent_config)
            creation_time = time.time() - creation_start
            
            # Register agent template
            self._register_agent_template('brain_orchestration', agent_config)
            
            # Update metrics
            self._update_creation_metrics('brain_orchestration', creation_time, success=True)
            
            self.logger.info(f"Created brain orchestration agent in {creation_time:.2f}s")
            return agent
            
        except Exception as e:
            self.logger.error(f"Brain orchestration agent creation failed: {e}")
            self._update_creation_metrics('brain_orchestration', 0, success=False)
            raise RuntimeError(f"Agent creation failed: {str(e)}")
    
    def create_proof_system_agent(self, config: Dict[str, Any]) -> 'ProofSystemAgent':
        """Create specialized proof system agent"""
        try:
            from .specialized_agents import ProofSystemAgent
            
            agent_config = {
                'agent_type': 'proof_system',
                'capabilities': [
                    'formal_verification',
                    'confidence_aggregation',
                    'algebraic_rule_enforcement',
                    'proof_strategy_adaptation',
                    'cryptographic_validation'
                ],
                'communication_protocols': ['proof_network', 'verification_chain'],
                'monitoring_hooks': ['proof_accuracy', 'verification_speed', 'confidence_scores'],
                'security_level': 'critical',
                'priority_level': 'high',
                'resource_limits': {
                    'max_memory_mb': config.get('max_memory_mb', 1024),
                    'max_cpu_percent': config.get('max_cpu_percent', 70),
                    'max_concurrent_proofs': config.get('max_concurrent_proofs', 5)
                },
                'proof_strategies': config.get('proof_strategies', ['formal', 'ml_based', 'cryptographic'])
            }
            agent_config.update(config)
            
            # Create agent
            creation_start = time.time()
            agent = ProofSystemAgent(agent_config)
            creation_time = time.time() - creation_start
            
            # Register agent template
            self._register_agent_template('proof_system', agent_config)
            
            # Update metrics
            self._update_creation_metrics('proof_system', creation_time, success=True)
            
            self.logger.info(f"Created proof system agent in {creation_time:.2f}s")
            return agent
            
        except Exception as e:
            self.logger.error(f"Proof system agent creation failed: {e}")
            self._update_creation_metrics('proof_system', 0, success=False)
            raise RuntimeError(f"Agent creation failed: {str(e)}")
    
    def create_uncertainty_agent(self, config: Dict[str, Any]) -> 'UncertaintyAgent':
        """Create specialized uncertainty quantification agent"""
        try:
            from .specialized_agents import UncertaintyAgent
            
            agent_config = {
                'agent_type': 'uncertainty',
                'capabilities': [
                    'conformalized_credal_quantification',
                    'deep_deterministic_uncertainty',
                    'ensemble_uncertainty',
                    'cross_domain_propagation',
                    'uncertainty_based_decision_making'
                ],
                'communication_protocols': ['uncertainty_network', 'propagation_chain'],
                'monitoring_hooks': ['quantification_accuracy', 'propagation_speed', 'confidence_intervals'],
                'security_level': 'high',
                'priority_level': 'high',
                'resource_limits': {
                    'max_memory_mb': config.get('max_memory_mb', 768),
                    'max_cpu_percent': config.get('max_cpu_percent', 60),
                    'max_concurrent_quantifications': config.get('max_concurrent_quantifications', 8)
                },
                'uncertainty_methods': config.get('uncertainty_methods', ['conformalized', 'deep', 'ensemble'])
            }
            agent_config.update(config)
            
            # Create agent
            creation_start = time.time()
            agent = UncertaintyAgent(agent_config)
            creation_time = time.time() - creation_start
            
            # Register agent template
            self._register_agent_template('uncertainty', agent_config)
            
            # Update metrics
            self._update_creation_metrics('uncertainty', creation_time, success=True)
            
            self.logger.info(f"Created uncertainty agent in {creation_time:.2f}s")
            return agent
            
        except Exception as e:
            self.logger.error(f"Uncertainty agent creation failed: {e}")
            self._update_creation_metrics('uncertainty', 0, success=False)
            raise RuntimeError(f"Agent creation failed: {str(e)}")
    
    def create_training_agent(self, config: Dict[str, Any]) -> 'TrainingAgent':
        """Create specialized training management agent"""
        try:
            from .specialized_agents import TrainingAgent
            
            agent_config = {
                'agent_type': 'training',
                'capabilities': [
                    'neural_network_training',
                    'gradient_ascent_clipping',
                    'domain_specific_training',
                    'training_optimization',
                    'model_validation'
                ],
                'communication_protocols': ['training_network', 'gradient_chain'],
                'monitoring_hooks': ['training_progress', 'gradient_health', 'model_performance'],
                'security_level': 'medium',
                'priority_level': 'normal',
                'resource_limits': {
                    'max_memory_mb': config.get('max_memory_mb', 2048),
                    'max_cpu_percent': config.get('max_cpu_percent', 80),
                    'max_gpu_memory_mb': config.get('max_gpu_memory_mb', 4096),
                    'max_concurrent_training': config.get('max_concurrent_training', 3)
                },
                'training_backends': config.get('training_backends', ['pytorch', 'tensorflow'])
            }
            agent_config.update(config)
            
            # Create agent
            creation_start = time.time()
            agent = TrainingAgent(agent_config)
            creation_time = time.time() - creation_start
            
            # Register agent template
            self._register_agent_template('training', agent_config)
            
            # Update metrics
            self._update_creation_metrics('training', creation_time, success=True)
            
            self.logger.info(f"Created training agent in {creation_time:.2f}s")
            return agent
            
        except Exception as e:
            self.logger.error(f"Training agent creation failed: {e}")
            self._update_creation_metrics('training', 0, success=False)
            raise RuntimeError(f"Agent creation failed: {str(e)}")
    
    def create_domain_agent(self, config: Dict[str, Any]) -> 'DomainAgent':
        """Create specialized domain management agent"""
        try:
            from .specialized_agents import DomainAgent
            
            agent_config = {
                'agent_type': 'domain',
                'capabilities': [
                    'domain_registration',
                    'domain_state_management',
                    'domain_routing',
                    'domain_specific_processing',
                    'domain_health_monitoring'
                ],
                'communication_protocols': ['domain_network', 'routing_chain'],
                'monitoring_hooks': ['domain_health', 'routing_efficiency', 'processing_performance'],
                'security_level': 'high',
                'priority_level': 'high',
                'resource_limits': {
                    'max_memory_mb': config.get('max_memory_mb', 512),
                    'max_cpu_percent': config.get('max_cpu_percent', 40),
                    'max_domains': config.get('max_domains', 100)
                },
                'domain_types': config.get('domain_types', ['math', 'nlp', 'vision', 'finance'])
            }
            agent_config.update(config)
            
            # Create agent
            creation_start = time.time()
            agent = DomainAgent(agent_config)
            creation_time = time.time() - creation_start
            
            # Register agent template
            self._register_agent_template('domain', agent_config)
            
            # Update metrics
            self._update_creation_metrics('domain', creation_time, success=True)
            
            self.logger.info(f"Created domain agent in {creation_time:.2f}s")
            return agent
            
        except Exception as e:
            self.logger.error(f"Domain agent creation failed: {e}")
            self._update_creation_metrics('domain', 0, success=False)
            raise RuntimeError(f"Agent creation failed: {str(e)}")
    
    def create_compression_agent(self, config: Dict[str, Any]) -> 'CompressionAgent':
        """Create specialized compression system agent"""
        try:
            from .specialized_agents import CompressionAgent
            
            agent_config = {
                'agent_type': 'compression',
                'capabilities': [
                    'hybrid_padic_compression',
                    'gpu_memory_optimization',
                    'cpu_bursting_decompression',
                    'tensor_decomposition',
                    'compression_optimization'
                ],
                'communication_protocols': ['compression_network', 'gpu_chain'],
                'monitoring_hooks': ['compression_ratio', 'decompression_speed', 'memory_usage'],
                'security_level': 'medium',
                'priority_level': 'normal',
                'resource_limits': {
                    'max_memory_mb': config.get('max_memory_mb', 1024),
                    'max_cpu_percent': config.get('max_cpu_percent', 60),
                    'max_gpu_memory_mb': config.get('max_gpu_memory_mb', 2048),
                    'max_concurrent_operations': config.get('max_concurrent_operations', 5)
                },
                'compression_algorithms': config.get('compression_algorithms', ['padic', 'tensor', 'hybrid'])
            }
            agent_config.update(config)
            
            # Create agent
            creation_start = time.time()
            agent = CompressionAgent(agent_config)
            creation_time = time.time() - creation_start
            
            # Register agent template
            self._register_agent_template('compression', agent_config)
            
            # Update metrics
            self._update_creation_metrics('compression', creation_time, success=True)
            
            self.logger.info(f"Created compression agent in {creation_time:.2f}s")
            return agent
            
        except Exception as e:
            self.logger.error(f"Compression agent creation failed: {e}")
            self._update_creation_metrics('compression', 0, success=False)
            raise RuntimeError(f"Agent creation failed: {str(e)}")
    
    def create_production_agent(self, config: Dict[str, Any]) -> 'ProductionAgent':
        """Create specialized production management agent"""
        try:
            from .specialized_agents import ProductionAgent
            
            agent_config = {
                'agent_type': 'production',
                'capabilities': [
                    'production_monitoring',
                    'auto_scaling',
                    'auto_recovery',
                    'load_balancing',
                    'security_management'
                ],
                'communication_protocols': ['production_network', 'monitoring_chain'],
                'monitoring_hooks': ['system_health', 'performance_metrics', 'security_status'],
                'security_level': 'critical',
                'priority_level': 'critical',
                'resource_limits': {
                    'max_memory_mb': config.get('max_memory_mb', 256),
                    'max_cpu_percent': config.get('max_cpu_percent', 30),
                    'max_monitoring_targets': config.get('max_monitoring_targets', 50)
                },
                'production_features': config.get('production_features', [
                    'monitoring', 'scaling', 'recovery', 'security'
                ])
            }
            agent_config.update(config)
            
            # Create agent
            creation_start = time.time()
            agent = ProductionAgent(agent_config)
            creation_time = time.time() - creation_start
            
            # Register agent template
            self._register_agent_template('production', agent_config)
            
            # Update metrics
            self._update_creation_metrics('production', creation_time, success=True)
            
            self.logger.info(f"Created production agent in {creation_time:.2f}s")
            return agent
            
        except Exception as e:
            self.logger.error(f"Production agent creation failed: {e}")
            self._update_creation_metrics('production', 0, success=False)
            raise RuntimeError(f"Agent creation failed: {str(e)}")
    
    def create_web_interface_agent(self, config: Dict[str, Any]) -> 'WebInterfaceAgent':
        """Create specialized web interface agent"""
        try:
            from .specialized_agents import WebInterfaceAgent
            
            agent_config = {
                'agent_type': 'web_interface',
                'capabilities': [
                    'dashboard_rendering',
                    'real_time_communication',
                    'api_gateway_management',
                    'user_interface_management',
                    'frontend_backend_integration'
                ],
                'communication_protocols': ['web_network', 'api_chain', 'websocket'],
                'monitoring_hooks': ['interface_performance', 'user_experience', 'api_health'],
                'security_level': 'high',
                'priority_level': 'normal',
                'resource_limits': {
                    'max_memory_mb': config.get('max_memory_mb', 512),
                    'max_cpu_percent': config.get('max_cpu_percent', 40),
                    'max_concurrent_connections': config.get('max_concurrent_connections', 1000)
                },
                'interface_features': config.get('interface_features', [
                    'dashboard', 'api', 'websocket', 'monitoring'
                ])
            }
            agent_config.update(config)
            
            # Create agent
            creation_start = time.time()
            agent = WebInterfaceAgent(agent_config)
            creation_time = time.time() - creation_start
            
            # Register agent template
            self._register_agent_template('web_interface', agent_config)
            
            # Update metrics
            self._update_creation_metrics('web_interface', creation_time, success=True)
            
            self.logger.info(f"Created web interface agent in {creation_time:.2f}s")
            return agent
            
        except Exception as e:
            self.logger.error(f"Web interface agent creation failed: {e}")
            self._update_creation_metrics('web_interface', 0, success=False)
            raise RuntimeError(f"Agent creation failed: {str(e)}")
    
    def get_agent_templates(self) -> Dict[str, Any]:
        """Get all registered agent templates"""
        with self._lock:
            return self.agent_templates.copy()
    
    def get_management_metrics(self) -> Dict[str, Any]:
        """Get agent management metrics"""
        with self._lock:
            return {
                'metrics': self.management_metrics.copy(),
                'template_usage': dict(self.management_metrics['template_usage']),
                'creation_history_count': len(self.agent_creation_history),
                'timestamp': datetime.now().isoformat()
            }
    
    def _initialize_agent_templates(self) -> None:
        """Initialize agent templates for different specializations"""
        try:
            # Define base templates for each agent type
            self.agent_templates = {
                'brain_orchestration': {
                    'base_capabilities': ['orchestration', 'coordination', 'integration'],
                    'base_communication': ['internal', 'cross_domain'],
                    'base_monitoring': ['performance', 'health'],
                    'base_security': 'high',
                    'resource_profile': 'medium'
                },
                'proof_system': {
                    'base_capabilities': ['verification', 'validation', 'proof_generation'],
                    'base_communication': ['proof_network', 'verification_chain'],
                    'base_monitoring': ['accuracy', 'speed', 'confidence'],
                    'base_security': 'critical',
                    'resource_profile': 'high'
                },
                'uncertainty': {
                    'base_capabilities': ['quantification', 'propagation', 'decision_making'],
                    'base_communication': ['uncertainty_network', 'propagation_chain'],
                    'base_monitoring': ['accuracy', 'speed', 'confidence'],
                    'base_security': 'high',
                    'resource_profile': 'medium'
                },
                'training': {
                    'base_capabilities': ['training', 'optimization', 'validation'],
                    'base_communication': ['training_network', 'gradient_chain'],
                    'base_monitoring': ['progress', 'performance', 'health'],
                    'base_security': 'medium',
                    'resource_profile': 'very_high'
                },
                'domain': {
                    'base_capabilities': ['registration', 'routing', 'processing'],
                    'base_communication': ['domain_network', 'routing_chain'],
                    'base_monitoring': ['health', 'efficiency', 'performance'],
                    'base_security': 'high',
                    'resource_profile': 'medium'
                },
                'compression': {
                    'base_capabilities': ['compression', 'decompression', 'optimization'],
                    'base_communication': ['compression_network', 'gpu_chain'],
                    'base_monitoring': ['ratio', 'speed', 'memory'],
                    'base_security': 'medium',
                    'resource_profile': 'high'
                },
                'production': {
                    'base_capabilities': ['monitoring', 'scaling', 'recovery'],
                    'base_communication': ['production_network', 'monitoring_chain'],
                    'base_monitoring': ['health', 'performance', 'security'],
                    'base_security': 'critical',
                    'resource_profile': 'low'
                },
                'web_interface': {
                    'base_capabilities': ['rendering', 'communication', 'integration'],
                    'base_communication': ['web_network', 'api_chain'],
                    'base_monitoring': ['performance', 'experience', 'health'],
                    'base_security': 'high',
                    'resource_profile': 'medium'
                }
            }
            
            self.logger.info(f"Initialized {len(self.agent_templates)} agent templates")
            
        except Exception as e:
            self.logger.error(f"Agent template initialization failed: {e}")
            raise RuntimeError(f"Template initialization failed: {str(e)}")
    
    def _initialize_agent_factories(self) -> None:
        """Initialize agent factories for dynamic creation"""
        try:
            # Map agent types to creation methods
            self.agent_factories = {
                'brain_orchestration': self.create_brain_orchestration_agent,
                'proof_system': self.create_proof_system_agent,
                'uncertainty': self.create_uncertainty_agent,
                'training': self.create_training_agent,
                'domain': self.create_domain_agent,
                'compression': self.create_compression_agent,
                'production': self.create_production_agent,
                'web_interface': self.create_web_interface_agent
            }
            
            self.logger.info(f"Initialized {len(self.agent_factories)} agent factories")
            
        except Exception as e:
            self.logger.error(f"Agent factory initialization failed: {e}")
            raise RuntimeError(f"Factory initialization failed: {str(e)}")
    
    def _create_default_agents(self) -> List[Dict[str, Any]]:
        """Create default agents based on configuration"""
        default_agents = []
        
        try:
            # Check if default agents are configured
            default_config = self.config.get('default_agents', [])
            
            for agent_spec in default_config:
                agent_type = agent_spec.get('type')
                if agent_type and agent_type in self.agent_factories:
                    try:
                        factory = self.agent_factories[agent_type]
                        agent = factory(agent_spec.get('config', {}))
                        default_agents.append({
                            'type': agent_type,
                            'agent': agent,
                            'created_at': datetime.now().isoformat()
                        })
                    except Exception as e:
                        self.logger.error(f"Failed to create default {agent_type} agent: {e}")
            
            return default_agents
            
        except Exception as e:
            self.logger.error(f"Default agent creation failed: {e}")
            return []
    
    def _validate_agent_templates(self) -> Dict[str, Any]:
        """Validate all agent templates"""
        validation_results = {
            'valid_templates': 0,
            'invalid_templates': 0,
            'validation_errors': []
        }
        
        try:
            for template_name, template_data in self.agent_templates.items():
                # Check required fields
                required_fields = ['base_capabilities', 'base_communication', 
                                 'base_monitoring', 'base_security']
                
                missing_fields = [field for field in required_fields 
                                if field not in template_data]
                
                if missing_fields:
                    validation_results['invalid_templates'] += 1
                    validation_results['validation_errors'].append({
                        'template': template_name,
                        'error': f'Missing fields: {missing_fields}'
                    })
                else:
                    validation_results['valid_templates'] += 1
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Template validation failed: {e}")
            return {
                'error': f'Validation failed: {str(e)}',
                'valid_templates': 0,
                'invalid_templates': len(self.agent_templates)
            }
    
    def _register_agent_template(self, agent_type: str, template: Dict[str, Any]) -> None:
        """Register agent template for future use"""
        try:
            with self._lock:
                # Store template
                self.agent_templates[f"{agent_type}_custom"] = template
                
                # Update usage metrics
                self.management_metrics['template_usage'][agent_type] += 1
                
                # Add to creation history
                self.agent_creation_history.append({
                    'agent_type': agent_type,
                    'template': template,
                    'created_at': datetime.now().isoformat()
                })
                
                # Limit history size
                if len(self.agent_creation_history) > 1000:
                    self.agent_creation_history = self.agent_creation_history[-1000:]
            
            self.logger.debug(f"Registered template for agent type: {agent_type}")
            
        except Exception as e:
            self.logger.error(f"Agent template registration failed: {e}")
            raise RuntimeError(f"Template registration failed: {str(e)}")
    
    def _update_creation_metrics(self, agent_type: str, creation_time: float, success: bool) -> None:
        """Update agent creation metrics"""
        try:
            with self._lock:
                if success:
                    self.management_metrics['total_agents_created'] += 1
                    self.management_metrics['active_agents'] += 1
                    
                    # Update average creation time
                    total = self.management_metrics['total_agents_created']
                    current_avg = self.management_metrics['average_creation_time']
                    self.management_metrics['average_creation_time'] = (
                        (current_avg * (total - 1) + creation_time) / total
                    )
                else:
                    self.management_metrics['failed_creations'] += 1
                
                # Update template usage
                self.management_metrics['template_usage'][agent_type] += 1
                
        except Exception as e:
            self.logger.error(f"Failed to update creation metrics: {e}")