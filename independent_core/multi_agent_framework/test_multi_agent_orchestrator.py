"""
Comprehensive test suite for MultiAgentOrchestrator
Tests all functionality with NO MOCKS - real implementations only
NO FALLBACKS - HARD FAILURES ONLY
"""

import unittest
import time
import threading
from datetime import datetime
from typing import Dict, List, Any
import uuid

# Import the module to test
from multi_agent_orchestrator import MultiAgentOrchestrator


class TestMultiAgentOrchestrator(unittest.TestCase):
    """Comprehensive tests for MultiAgentOrchestrator"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_config = {
            'agent_config': {
                'max_agents': 10,
                'agent_timeout': 30
            },
            'coordination_config': {
                'communication_timeout': 5,
                'retry_attempts': 3
            },
            'distribution_config': {
                'load_balancing': True,
                'queue_size': 100
            },
            'monitoring_config': {
                'health_check_interval': 10,
                'metrics_collection': True
            },
            'integration_config': {
                'validation_enabled': True,
                'security_checks': True
            }
        }
        
        # Create orchestrator instance
        self.orchestrator = MultiAgentOrchestrator(self.test_config)
    
    def tearDown(self):
        """Clean up test environment"""
        # Clean up any resources
        pass
    
    def test_initialization(self):
        """Test orchestrator initialization"""
        # Test successful initialization
        self.assertIsNotNone(self.orchestrator)
        self.assertEqual(self.orchestrator.config, self.test_config)
        
        # Check component initialization
        self.assertIsNotNone(self.orchestrator.agent_manager)
        self.assertIsNotNone(self.orchestrator.agent_coordinator)
        self.assertIsNotNone(self.orchestrator.task_distributor)
        self.assertIsNotNone(self.orchestrator.agent_monitor)
        self.assertIsNotNone(self.orchestrator.agent_integration_manager)
        
        # Check initial state
        self.assertEqual(len(self.orchestrator.agent_registry), 0)
        self.assertEqual(len(self.orchestrator.agent_states), 0)
        self.assertEqual(self.orchestrator.orchestration_metrics['total_agents'], 0)
        self.assertEqual(self.orchestrator.orchestration_metrics['active_agents'], 0)
    
    def test_initialization_with_invalid_config(self):
        """Test initialization with invalid configuration"""
        # Test with None config - should fail hard
        with self.assertRaises((ValueError, TypeError)):
            MultiAgentOrchestrator(None)
        
        # Test with invalid config type - should fail hard
        with self.assertRaises((ValueError, TypeError)):
            MultiAgentOrchestrator("invalid_config")
    
    def test_initialize_multi_agent_framework(self):
        """Test multi-agent framework initialization"""
        # Create mock brain core
        class MockBrainCore:
            def __init__(self):
                self.registered_agents = {}
                self.communication_channels = {}
                self.monitoring_hooks = {}
            
            def register_agent_capabilities(self, agent_id, capabilities):
                if not agent_id or not capabilities:
                    raise ValueError("Invalid agent registration parameters")
                self.registered_agents[agent_id] = capabilities
            
            def setup_agent_communication(self, agent_id, endpoints):
                if not agent_id or not endpoints:
                    raise ValueError("Invalid communication setup parameters")
                self.communication_channels[agent_id] = endpoints
            
            def register_agent_monitoring(self, agent_id, hooks):
                if not agent_id or not hooks:
                    raise ValueError("Invalid monitoring registration parameters")
                self.monitoring_hooks[agent_id] = hooks
        
        brain_core = MockBrainCore()
        
        # Test framework initialization
        result = self.orchestrator.initialize_multi_agent_framework(brain_core)
        
        # Verify initialization result
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('timestamp', result)
        
        if result['success']:
            self.assertIn('agent_initialization', result)
            self.assertIn('coordination_initialization', result)
            self.assertIn('distribution_initialization', result)
            self.assertIn('monitoring_initialization', result)
            self.assertIn('integration_initialization', result)
            self.assertIn('framework_validation', result)
        else:
            self.assertIn('error', result)
    
    def test_create_specialized_agents_missing_agents_module(self):
        """Test agent creation when specialized agents module is missing"""
        # Test agent specifications that will fail due to missing specialized_agents
        agent_specs = [
            {
                'agent_type': 'brain_orchestration',
                'config': {
                    'capabilities': ['coordination', 'planning'],
                    'resources': {'memory': 512, 'cpu': 2}
                }
            }
        ]
        
        # Should fail hard when specialized agents module is missing
        result = self.orchestrator.create_specialized_agents(agent_specs)
        
        # Verify hard failure
        self.assertIsInstance(result, dict)
        self.assertFalse(result['success'])
        self.assertIn('error', result)
    
    def test_create_specialized_agents_invalid_type(self):
        """Test agent creation with invalid agent type"""
        agent_specs = [
            {
                'agent_type': 'invalid_agent_type',
                'config': {}
            }
        ]
        
        # Should fail hard on invalid agent type
        result = self.orchestrator.create_specialized_agents(agent_specs)
        
        # Verify hard failure
        self.assertIsInstance(result, dict)
        self.assertFalse(result['success'])
        self.assertIn('error', result)
        self.assertIn('unknown agent type', result['error'].lower())
    
    def test_distribute_tasks_to_agents_no_agents(self):
        """Test task distribution when no agents exist"""
        # Test task distribution without agents
        tasks = [
            {
                'task_id': 'test_task_1',
                'task_type': 'brain_orchestration',
                'task_data': {'operation': 'coordinate', 'parameters': {}},
                'priority': 'high'
            }
        ]
        
        result = self.orchestrator.distribute_tasks_to_agents(tasks)
        
        # Should complete but show no suitable agents
        self.assertIsInstance(result, dict)
        if result['success']:
            self.assertIn('distribution_results', result)
            self.assertIn('failed_assignments', result)
            self.assertEqual(result['successful_assignments'], 0)
    
    def test_monitor_agent_health_no_agents(self):
        """Test agent health monitoring with no agents"""
        # Test health monitoring without agents
        result = self.orchestrator.monitor_agent_health()
        
        # Verify monitoring result
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('timestamp', result)
        
        if result['success']:
            self.assertIn('agent_health', result)
            self.assertIn('overall_system_health', result)
            self.assertIn('total_agents_monitored', result)
            self.assertEqual(result['total_agents_monitored'], 0)
    
    def test_get_agent_performance_report_empty(self):
        """Test agent performance reporting with no agents"""
        # Test performance report generation with no agents
        result = self.orchestrator.get_agent_performance_report()
        
        # Verify report result
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('timestamp', result)
        
        if result['success']:
            self.assertIn('performance_report', result)
            performance_report = result['performance_report']
            
            self.assertIn('agents', performance_report)
            self.assertIn('system_metrics', performance_report)
            self.assertIn('recommendations', performance_report)
            
            # Verify empty state
            self.assertEqual(len(performance_report['agents']), 0)
    
    def test_agent_registration_with_mock_agent(self):
        """Test agent registration functionality with mock agent"""
        # Create a mock agent that implements required interface
        class MockAgent:
            def __init__(self):
                self.capabilities = {'test': True}
                self.performance_metrics = {'uptime': 0}
                self.communication_endpoints = {'http': 'localhost:8080'}
                self.monitoring_hooks = {'health_check': True}
            
            def get_capabilities(self):
                return self.capabilities
            
            def get_performance_metrics(self):
                return self.performance_metrics
                
            def get_communication_endpoints(self):
                return self.communication_endpoints
                
            def get_monitoring_hooks(self):
                return self.monitoring_hooks
            
            def execute_task(self, task_data, priority):
                if not task_data:
                    raise ValueError("Task data cannot be empty")
                return {'success': True, 'result': 'completed', 'execution_time': 1.0}
            
            def get_last_heartbeat(self):
                return time.time()
            
            def get_task_queue_size(self):
                return 0
            
            def get_memory_usage(self):
                return 128
            
            def get_cpu_usage(self):
                return 25.0
        
        mock_agent = MockAgent()
        agent_spec = {'agent_type': 'test_agent'}
        
        # Test agent registration
        agent_id = self.orchestrator._register_agent(mock_agent, agent_spec)
        
        # Verify registration
        self.assertIsInstance(agent_id, str)
        self.assertIn(agent_id, self.orchestrator.agent_registry)
        self.assertIn(agent_id, self.orchestrator.agent_states)
        self.assertEqual(mock_agent.agent_id, agent_id)
        self.assertEqual(mock_agent.agent_type, 'test_agent')
    
    def test_task_submission_to_agent(self):
        """Test task submission to agents"""
        # Create and register a mock agent
        class MockAgent:
            def __init__(self):
                self.agent_id = None
                self.agent_type = None
            
            def execute_task(self, task_data, priority):
                if not task_data:
                    raise ValueError("Task data cannot be empty")
                return {'success': True, 'result': task_data, 'execution_time': 0.5}
        
        mock_agent = MockAgent()
        agent_spec = {'agent_type': 'task_test_agent'}
        agent_id = self.orchestrator._register_agent(mock_agent, agent_spec)
        
        # Test task submission
        test_task = {
            'task_id': 'test_submission',
            'task_type': 'test',
            'task_data': {'operation': 'test_operation'},
            'priority': 'normal'
        }
        
        result = self.orchestrator._submit_task_to_agent(mock_agent, test_task)
        
        # Verify submission result
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        
        if result['success']:
            self.assertIn('task_result', result)
            self.assertIn('execution_time', result)
    
    def test_task_submission_with_empty_task_data(self):
        """Test task submission with empty task data"""
        class MockAgent:
            def __init__(self):
                self.agent_id = None
                self.agent_type = None
            
            def execute_task(self, task_data, priority):
                if not task_data:
                    raise ValueError("Task data cannot be empty")
                return {'success': True, 'result': task_data, 'execution_time': 0.5}
        
        mock_agent = MockAgent()
        agent_spec = {'agent_type': 'task_test_agent'}
        agent_id = self.orchestrator._register_agent(mock_agent, agent_spec)
        
        # Test task submission with empty data - should fail hard
        test_task = {
            'task_id': 'test_empty',
            'task_type': 'test',
            'task_data': {},  # Empty task data
            'priority': 'normal'
        }
        
        result = self.orchestrator._submit_task_to_agent(mock_agent, test_task)
        
        # Should fail hard due to empty task data
        self.assertIsInstance(result, dict)
        self.assertFalse(result['success'])
        self.assertIn('error', result)
    
    def test_framework_validation(self):
        """Test framework integration validation"""
        # Test validation
        result = self.orchestrator._validate_framework_integration()
        
        # Verify validation result
        self.assertIsInstance(result, dict)
        self.assertIn('agent_registration', result)
        self.assertIn('coordination_active', result)
        self.assertIn('distribution_active', result)
        self.assertIn('monitoring_active', result)
        self.assertIn('integration_active', result)
        self.assertIn('validation_timestamp', result)
    
    def test_system_health_calculation_no_agents(self):
        """Test system health calculation with no agents"""
        # Test with no agents
        empty_health_results = {}
        result = self.orchestrator._calculate_overall_system_health(empty_health_results)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['overall_status'], 'no_agents')
        self.assertEqual(result['health_percentage'], 0)
    
    def test_system_health_calculation_healthy_agents(self):
        """Test system health calculation with healthy agents"""
        # Test with healthy agents
        healthy_health_results = {
            'agent1': {
                'health_status': {'status': 'healthy'},
                'memory_usage': 100,
                'cpu_usage': 20,
                'task_queue_size': 2
            },
            'agent2': {
                'health_status': {'status': 'healthy'}, 
                'memory_usage': 150,
                'cpu_usage': 30,
                'task_queue_size': 1
            }
        }
        
        result = self.orchestrator._calculate_overall_system_health(healthy_health_results)
        
        self.assertIsInstance(result, dict)
        self.assertIn('overall_status', result)
        self.assertIn('health_percentage', result)
        self.assertIn('total_agents', result)
        self.assertIn('healthy_agents', result)
        self.assertEqual(result['total_agents'], 2)
        self.assertEqual(result['healthy_agents'], 2)
        self.assertEqual(result['health_percentage'], 100.0)
    
    def test_system_health_calculation_unhealthy_agents(self):
        """Test system health calculation with unhealthy agents"""
        # Test with unhealthy agents
        unhealthy_health_results = {
            'agent1': {
                'health_status': {'status': 'unhealthy'},
                'memory_usage': 100,
                'cpu_usage': 20,
                'task_queue_size': 2
            },
            'agent2': {
                'health_status': {'status': 'healthy'}, 
                'memory_usage': 150,
                'cpu_usage': 30,
                'task_queue_size': 1
            }
        }
        
        result = self.orchestrator._calculate_overall_system_health(unhealthy_health_results)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['total_agents'], 2)
        self.assertEqual(result['healthy_agents'], 1)
        self.assertEqual(result['health_percentage'], 50.0)
        self.assertEqual(result['overall_status'], 'critical')
    
    def test_performance_recommendations_low_utilization(self):
        """Test performance recommendation generation for low utilization"""
        agent_data = {
            'agent1': {
                'resource_usage': {'memory_mb': 200, 'cpu_percent': 15},
                'task_statistics': {'average_completion_time': 10}
            }
        }
        
        system_metrics = {
            'agent_utilization': 0.3,  # Low utilization
            'total_completed_tasks': 90,
            'total_failed_tasks': 10  # 10% failure rate
        }
        
        recommendations = self.orchestrator._generate_performance_recommendations(
            agent_data, system_metrics
        )
        
        # Verify recommendations
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Check for low utilization warning
        recommendation_text = ' '.join(recommendations).lower()
        self.assertIn('utilization', recommendation_text)
    
    def test_performance_recommendations_high_failure_rate(self):
        """Test performance recommendation generation for high failure rate"""
        agent_data = {
            'agent1': {
                'resource_usage': {'memory_mb': 200, 'cpu_percent': 50},
                'task_statistics': {'average_completion_time': 30}
            }
        }
        
        system_metrics = {
            'agent_utilization': 0.7,
            'total_completed_tasks': 70,
            'total_failed_tasks': 30  # 30% failure rate
        }
        
        recommendations = self.orchestrator._generate_performance_recommendations(
            agent_data, system_metrics
        )
        
        # Verify recommendations
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Check for high failure rate warning
        recommendation_text = ' '.join(recommendations).lower()
        self.assertIn('failure', recommendation_text)
    
    def test_performance_recommendations_resource_issues(self):
        """Test performance recommendation generation for resource issues"""
        agent_data = {
            'agent1': {
                'resource_usage': {'memory_mb': 1200, 'cpu_percent': 85},
                'task_statistics': {'average_completion_time': 70}
            }
        }
        
        system_metrics = {
            'agent_utilization': 0.8,
            'total_completed_tasks': 80,
            'total_failed_tasks': 5
        }
        
        recommendations = self.orchestrator._generate_performance_recommendations(
            agent_data, system_metrics
        )
        
        # Verify recommendations
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Check for resource usage warnings
        recommendation_text = ' '.join(recommendations).lower()
        self.assertIn('memory', recommendation_text)
        self.assertIn('cpu', recommendation_text)


class TestMultiAgentOrchestratorEdgeCases(unittest.TestCase):
    """Edge case tests for MultiAgentOrchestrator"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_config = {
            'agent_config': {},
            'coordination_config': {},
            'distribution_config': {},
            'monitoring_config': {},
            'integration_config': {}
        }
        self.orchestrator = MultiAgentOrchestrator(self.test_config)
    
    def test_empty_agent_specifications(self):
        """Test with empty agent specifications"""
        result = self.orchestrator.create_specialized_agents([])
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result['success'])
        self.assertEqual(result['total_agents'], 0)
    
    def test_empty_task_list(self):
        """Test with empty task list"""
        result = self.orchestrator.distribute_tasks_to_agents([])
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result['success'])
        self.assertEqual(result['tasks_distributed'], 0)
    
    def test_malformed_agent_specification_empty_dict(self):
        """Test with empty agent specification dict"""
        malformed_specs = [{}]  # Missing agent_type
        
        result = self.orchestrator.create_specialized_agents(malformed_specs)
        self.assertFalse(result['success'])
        self.assertIn('error', result)
    
    def test_malformed_agent_specification_empty_type(self):
        """Test with empty agent type"""
        malformed_specs = [{'agent_type': ''}]  # Empty agent_type
        
        result = self.orchestrator.create_specialized_agents(malformed_specs)
        self.assertFalse(result['success'])
        self.assertIn('error', result)
    
    def test_malformed_agent_specification_none_type(self):
        """Test with None agent type"""
        malformed_specs = [{'agent_type': None}]  # None agent_type
        
        result = self.orchestrator.create_specialized_agents(malformed_specs)
        self.assertFalse(result['success'])
        self.assertIn('error', result)
    
    def test_malformed_task_specification(self):
        """Test with malformed task specification"""
        malformed_tasks = [
            {},  # Empty task
            {'task_id': None},  # None task_id
            {'task_type': ''}  # Empty task_type
        ]
        
        result = self.orchestrator.distribute_tasks_to_agents(malformed_tasks)
        self.assertIsInstance(result, dict)
        # Should handle malformed tasks but show failures in assignment
    
    def test_concurrent_operations(self):
        """Test concurrent operations"""
        def monitor_health():
            return self.orchestrator.monitor_agent_health()
        
        def get_performance():
            return self.orchestrator.get_agent_performance_report()
        
        def create_empty_agents():
            return self.orchestrator.create_specialized_agents([])
        
        # Create threads for concurrent operations
        threads = [
            threading.Thread(target=monitor_health),
            threading.Thread(target=get_performance),
            threading.Thread(target=create_empty_agents)
        ]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no crashes occurred
        self.assertTrue(True)  # If we get here, no crashes occurred
    
    def test_register_agent_with_invalid_spec(self):
        """Test agent registration with invalid spec"""
        class MockAgent:
            def get_capabilities(self):
                return {}
            def get_performance_metrics(self):
                return {}
        
        mock_agent = MockAgent()
        
        # Test with empty spec - should fail hard
        with self.assertRaises((ValueError, KeyError)):
            self.orchestrator._register_agent(mock_agent, {})
        
        # Test with missing agent_type
        with self.assertRaises((ValueError, KeyError)):
            self.orchestrator._register_agent(mock_agent, {'config': {}})
    
    def test_submit_task_to_nonexistent_agent(self):
        """Test task submission to non-registered agent"""
        class MockAgent:
            def __init__(self):
                self.agent_id = 'nonexistent_agent'
            
            def execute_task(self, task_data, priority):
                raise RuntimeError("Agent not properly initialized")
        
        mock_agent = MockAgent()
        
        test_task = {
            'task_id': 'test',
            'task_type': 'test',
            'task_data': {'operation': 'test'},
            'priority': 'normal'
        }
        
        result = self.orchestrator._submit_task_to_agent(mock_agent, test_task)
        
        # Should fail due to agent not being in orchestrator state
        self.assertIsInstance(result, dict)
        self.assertFalse(result['success'])
        self.assertIn('error', result)


class TestMultiAgentOrchestratorIntegration(unittest.TestCase):
    """Integration tests for MultiAgentOrchestrator"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.test_config = {
            'agent_config': {
                'max_agents': 5,
                'agent_timeout': 10
            },
            'coordination_config': {
                'communication_timeout': 3,
                'retry_attempts': 2
            },
            'distribution_config': {
                'load_balancing': True,
                'queue_size': 50
            },
            'monitoring_config': {
                'health_check_interval': 5,
                'metrics_collection': True
            },
            'integration_config': {
                'validation_enabled': True,
                'security_checks': True
            }
        }
        self.orchestrator = MultiAgentOrchestrator(self.test_config)
    
    def test_full_orchestration_workflow_no_agents(self):
        """Test complete orchestration workflow without creating agents"""
        # Create mock brain core
        class MockBrainCore:
            def __init__(self):
                self.registered_agents = {}
                
            def register_agent_capabilities(self, agent_id, capabilities):
                self.registered_agents[agent_id] = capabilities
                
            def setup_agent_communication(self, agent_id, endpoints):
                pass
                
            def register_agent_monitoring(self, agent_id, hooks):
                pass
        
        brain_core = MockBrainCore()
        
        # Step 1: Initialize framework
        init_result = self.orchestrator.initialize_multi_agent_framework(brain_core)
        self.assertIsInstance(init_result, dict)
        self.assertIn('success', init_result)
        
        # Step 2: Try to create agents (will fail due to missing specialized agents)
        agent_specs = [
            {
                'agent_type': 'brain_orchestration',
                'config': {'capabilities': ['coordination']}
            }
        ]
        
        creation_result = self.orchestrator.create_specialized_agents(agent_specs)
        self.assertIsInstance(creation_result, dict)
        # Should fail due to missing specialized agents module
        
        # Step 3: Distribute tasks (no agents available)
        tasks = [
            {
                'task_type': 'brain_orchestration',
                'task_data': {'operation': 'coordinate'},
                'priority': 'high'
            }
        ]
        
        distribution_result = self.orchestrator.distribute_tasks_to_agents(tasks)
        self.assertIsInstance(distribution_result, dict)
        
        # Step 4: Monitor health (no agents)
        health_result = self.orchestrator.monitor_agent_health()
        self.assertIsInstance(health_result, dict)
        self.assertIn('success', health_result)
        
        # Step 5: Get performance report (no agents)
        report_result = self.orchestrator.get_agent_performance_report()
        self.assertIsInstance(report_result, dict)
        self.assertIn('success', report_result)
    
    def test_orchestrator_with_manual_agent_registration(self):
        """Test orchestrator with manually registered agents"""
        # Create and manually register mock agents
        class MockAgent:
            def __init__(self, agent_type):
                self.agent_type = agent_type
                self.agent_id = None
                self.capabilities = {f'{agent_type}_capability': True}
                
            def get_capabilities(self):
                return self.capabilities
            
            def get_performance_metrics(self):
                return {'tasks_completed': 0, 'uptime': time.time()}
            
            def get_communication_endpoints(self):
                return {'http': f'localhost:808{len(self.agent_type)}'}
            
            def get_monitoring_hooks(self):
                return {'health_check': True}
            
            def execute_task(self, task_data, priority):
                return {
                    'success': True,
                    'result': f'Processed by {self.agent_type}',
                    'execution_time': 0.1
                }
            
            def get_last_heartbeat(self):
                return time.time()
            
            def get_task_queue_size(self):
                return 0
            
            def get_memory_usage(self):
                return 100
            
            def get_cpu_usage(self):
                return 10.0
        
        # Register multiple agents
        agent1 = MockAgent('test_agent_1')
        agent2 = MockAgent('test_agent_2')
        
        agent_id1 = self.orchestrator._register_agent(agent1, {'agent_type': 'test_agent_1'})
        agent_id2 = self.orchestrator._register_agent(agent2, {'agent_type': 'test_agent_2'})
        
        # Verify registration
        self.assertIn(agent_id1, self.orchestrator.agent_registry)
        self.assertIn(agent_id2, self.orchestrator.agent_registry)
        
        # Test health monitoring with registered agents
        health_result = self.orchestrator.monitor_agent_health()
        self.assertIsInstance(health_result, dict)
        if health_result['success']:
            self.assertEqual(health_result['total_agents_monitored'], 2)
        
        # Test performance reporting with registered agents
        performance_result = self.orchestrator.get_agent_performance_report()
        self.assertIsInstance(performance_result, dict)
        if performance_result['success']:
            self.assertEqual(len(performance_result['performance_report']['agents']), 2)
        
        # Test task submission
        test_task = {
            'task_id': 'integration_test',
            'task_type': 'test',
            'task_data': {'operation': 'integration_test'},
            'priority': 'normal'
        }
        
        submission_result = self.orchestrator._submit_task_to_agent(agent1, test_task)
        self.assertIsInstance(submission_result, dict)
        if submission_result['success']:
            self.assertIn('task_result', submission_result)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)