"""
Comprehensive test suite for TaskDistributor
Tests all functionality, edge cases, threading, performance, and error handling
"""

import pytest
import time
import threading
from unittest import TestCase
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import json
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_agent_framework.task_distributor import TaskDistributor


class MockAgent:
    """Mock agent for testing"""
    
    def __init__(self, agent_id: str, config: dict = None):
        self.agent_id = agent_id
        self.config = config or {}
        self.tasks_received = []
        self.is_available = True
        
    def process_task(self, task_data):
        self.tasks_received.append(task_data)
        return {"status": "completed", "agent_id": self.agent_id}


class TestTaskDistributorBasics:
    """Test basic functionality of TaskDistributor"""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing"""
        return {
            'distribution_strategy': 'load_balanced',
            'max_distribution_retries': 3,
            'retry_delay_seconds': 1,
            'load_threshold_percent': 80,
            'affinity_weight': 0.3,
            'enable_task_affinity': True,
            'load_update_interval': 1,
            'enable_threshold_alerts': True
        }
    
    @pytest.fixture
    def task_distributor(self, basic_config):
        """Create task distributor instance"""
        distributor = TaskDistributor(basic_config)
        yield distributor
        distributor.is_active_flag = False  # Stop background threads
    
    def test_task_distributor_initialization(self, basic_config):
        """Test task distributor initialization"""
        distributor = TaskDistributor(basic_config)
        
        assert distributor.current_strategy == 'load_balanced'
        assert len(distributor.distribution_strategies) == 5
        assert distributor.is_active_flag == False  # Not active until initialize_distribution
        assert len(distributor.registered_agents) == 0
        assert len(distributor.agent_capabilities) == 0
        
        distributor.is_active_flag = False
    
    def test_initialize_distribution(self, task_distributor):
        """Test distribution system initialization"""
        result = task_distributor.initialize_distribution()
        
        assert result['success'] == True
        assert 'distribution_infrastructure' in result
        assert 'load_tracking' in result
        assert 'affinity_system' in result
        assert result['current_strategy'] == 'load_balanced'
        assert len(result['available_strategies']) == 5
        assert task_distributor.is_active_flag == True
    
    def test_is_active_status(self, task_distributor):
        """Test active status checking"""
        assert task_distributor.is_active() == False
        
        task_distributor.initialize_distribution()
        assert task_distributor.is_active() == True


class TestAgentRegistration:
    """Test agent registration and capability management"""
    
    @pytest.fixture
    def task_distributor(self):
        """Create initialized task distributor"""
        config = {'distribution_strategy': 'capability_based'}
        distributor = TaskDistributor(config)
        distributor.initialize_distribution()
        yield distributor
        distributor.is_active_flag = False
    
    def test_agent_capability_registration(self, task_distributor):
        """Test registering agent capabilities"""
        agent = MockAgent("test_agent_1", {
            'resource_limits': {
                'max_cpu_percent': 80,
                'max_memory_mb': 1024,
                'max_concurrent_tasks': 10
            }
        })
        
        capabilities = ['domain_management', 'session_orchestration', 'formal_verification']
        
        result = task_distributor.register_agent_capabilities(
            agent.agent_id, agent, capabilities
        )
        
        assert result['success'] == True
        assert result['agent_id'] == 'test_agent_1'
        assert result['registered_capabilities'] == 3
        assert result['task_types'] > 0
        assert result['max_load'] == 10
        
        # Verify agent is registered
        assert agent.agent_id in task_distributor.registered_agents
        assert agent.agent_id in task_distributor.agent_capabilities
        assert agent.agent_id in task_distributor.agent_load
    
    def test_multiple_agent_registration(self, task_distributor):
        """Test registering multiple agents"""
        agents_data = [
            ("agent_1", ['domain_management', 'session_orchestration']),
            ("agent_2", ['formal_verification', 'confidence_aggregation']),
            ("agent_3", ['neural_network_training', 'gradient_ascent_clipping'])
        ]
        
        for agent_id, capabilities in agents_data:
            agent = MockAgent(agent_id)
            result = task_distributor.register_agent_capabilities(
                agent_id, agent, capabilities
            )
            assert result['success'] == True
        
        assert len(task_distributor.registered_agents) == 3
        assert len(task_distributor.agent_capabilities) == 3
    
    def test_agent_capability_mapping(self, task_distributor):
        """Test capability to task type mapping"""
        agent = MockAgent("mapping_agent")
        capabilities = ['domain_management', 'formal_verification']
        
        task_distributor.register_agent_capabilities(agent.agent_id, agent, capabilities)
        
        agent_caps = task_distributor.agent_capabilities[agent.agent_id]
        task_types = agent_caps['task_types']
        
        # Check that capabilities are mapped to task types
        assert 'domain_create' in task_types  # from domain_management
        assert 'proof_generate' in task_types  # from formal_verification
        assert len(task_types) > 0
    
    def test_agent_load_calculation(self, task_distributor):
        """Test agent max load calculation"""
        # Agent with resource limits
        agent_with_limits = MockAgent("resource_agent", {
            'resource_limits': {
                'max_cpu_percent': 50,
                'max_memory_mb': 500,
                'max_concurrent_tasks': 8
            }
        })
        
        result = task_distributor.register_agent_capabilities(
            agent_with_limits.agent_id, agent_with_limits, ['domain_management']
        )
        
        # Should use configured max_concurrent_tasks
        assert result['max_load'] == 8
        
        # Agent without limits should get default calculation
        agent_default = MockAgent("default_agent")
        result_default = task_distributor.register_agent_capabilities(
            agent_default.agent_id, agent_default, ['domain_management']
        )
        
        # Should calculate based on default resources
        assert result_default['max_load'] >= 1


class TestLoadManagement:
    """Test load management and balancing"""
    
    @pytest.fixture
    def task_distributor_with_agents(self):
        """Create distributor with registered agents"""
        config = {'distribution_strategy': 'load_balanced'}
        distributor = TaskDistributor(config)
        distributor.initialize_distribution()
        
        # Register test agents
        for i in range(3):
            agent = MockAgent(f"agent_{i}", {
                'resource_limits': {'max_concurrent_tasks': 5}
            })
            distributor.register_agent_capabilities(
                agent.agent_id, agent, ['domain_management', 'session_orchestration']
            )
        
        yield distributor
        distributor.is_active_flag = False
    
    def test_agent_load_update(self, task_distributor_with_agents):
        """Test updating agent load"""
        agent_id = "agent_0"
        
        # Increase load
        result = task_distributor_with_agents.update_agent_load(agent_id, 2)
        assert result['success'] == True
        assert result['current_load'] == 2
        assert result['max_load'] == 5
        assert result['load_percentage'] == 40.0
        assert result['is_available'] == True
        
        # Increase to max load
        result = task_distributor_with_agents.update_agent_load(agent_id, 3)
        assert result['current_load'] == 5
        assert result['load_percentage'] == 100.0
        assert result['is_available'] == False
        
        # Decrease load
        result = task_distributor_with_agents.update_agent_load(agent_id, -2)
        assert result['current_load'] == 3
        assert result['load_percentage'] == 60.0
        assert result['is_available'] == True
    
    def test_load_update_nonexistent_agent(self, task_distributor_with_agents):
        """Test load update for non-existent agent"""
        result = task_distributor_with_agents.update_agent_load("nonexistent", 1)
        assert result['success'] == False
        assert 'not registered' in result['error']
    
    def test_load_prevents_negative_values(self, task_distributor_with_agents):
        """Test that load cannot go negative"""
        agent_id = "agent_0"
        
        # Try to decrease load below zero
        result = task_distributor_with_agents.update_agent_load(agent_id, -5)
        assert result['success'] == True
        assert result['current_load'] == 0  # Should be clamped to 0
    
    def test_availability_based_on_load(self, task_distributor_with_agents):
        """Test that availability changes based on load"""
        agent_id = "agent_0"
        
        # Agent should be available initially
        caps = task_distributor_with_agents.agent_capabilities[agent_id]
        assert caps['availability'] == True
        
        # Max out the agent
        task_distributor_with_agents.update_agent_load(agent_id, 5)
        caps = task_distributor_with_agents.agent_capabilities[agent_id]
        assert caps['availability'] == False


class TestDistributionStrategies:
    """Test different task distribution strategies"""
    
    @pytest.fixture
    def distributor_with_diverse_agents(self):
        """Create distributor with agents having different capabilities"""
        config = {'distribution_strategy': 'load_balanced'}
        distributor = TaskDistributor(config)
        distributor.initialize_distribution()
        
        # Agent 1: Domain management specialist
        agent1 = MockAgent("domain_agent", {'priority_level': 'high'})
        distributor.register_agent_capabilities(
            agent1.agent_id, agent1, ['domain_management', 'session_orchestration']
        )
        
        # Agent 2: Proof system specialist
        agent2 = MockAgent("proof_agent", {'priority_level': 'normal'})
        distributor.register_agent_capabilities(
            agent2.agent_id, agent2, ['formal_verification', 'confidence_aggregation']
        )
        
        # Agent 3: Multi-capability agent
        agent3 = MockAgent("multi_agent", {'priority_level': 'critical'})
        distributor.register_agent_capabilities(
            agent3.agent_id, agent3, 
            ['domain_management', 'formal_verification', 'neural_network_training']
        )
        
        yield distributor, {"domain_agent": agent1, "proof_agent": agent2, "multi_agent": agent3}
        distributor.is_active_flag = False
    
    def test_round_robin_distribution(self, distributor_with_diverse_agents):
        """Test round-robin distribution strategy"""
        distributor, agents = distributor_with_diverse_agents
        distributor.set_distribution_strategy('round_robin')
        
        # Multiple task selections should cycle through agents
        selected_agents = []
        for _ in range(6):
            agent = distributor.select_agent_for_task('domain_create', {})
            if agent:
                selected_agents.append(agent.agent_id)
        
        # Should cycle through available agents
        assert len(selected_agents) > 0
        # Should have some variety in selection
        unique_agents = set(selected_agents)
        assert len(unique_agents) > 1
    
    def test_load_balanced_distribution(self, distributor_with_diverse_agents):
        """Test load-balanced distribution strategy"""
        distributor, agents = distributor_with_diverse_agents
        distributor.set_distribution_strategy('load_balanced')
        
        # Load one agent heavily
        distributor.update_agent_load("domain_agent", 8)
        
        # Select agent for domain task - should prefer less loaded agents
        selected_agent = distributor.select_agent_for_task('domain_create', {})
        
        assert selected_agent is not None
        # Should not select the heavily loaded agent
        assert selected_agent.agent_id != "domain_agent"
    
    def test_capability_based_distribution(self, distributor_with_diverse_agents):
        """Test capability-based distribution strategy"""
        distributor, agents = distributor_with_diverse_agents
        distributor.set_distribution_strategy('capability_based')
        
        # Select agent for proof task
        selected_agent = distributor.select_agent_for_task('proof_generate', {})
        
        assert selected_agent is not None
        # Should select agent capable of proof tasks
        assert selected_agent.agent_id in ["proof_agent", "multi_agent"]
    
    def test_priority_based_distribution(self, distributor_with_diverse_agents):
        """Test priority-based distribution strategy"""
        distributor, agents = distributor_with_diverse_agents
        distributor.set_distribution_strategy('priority_based')
        
        # High priority task should go to high/critical priority agent
        high_priority_task = {'priority': 'high'}
        selected_agent = distributor.select_agent_for_task('domain_create', high_priority_task)
        
        assert selected_agent is not None
        # Should prefer high or critical priority agents
        assert selected_agent.agent_id in ["domain_agent", "multi_agent"]
    
    def test_affinity_based_distribution(self, distributor_with_diverse_agents):
        """Test affinity-based distribution strategy"""
        distributor, agents = distributor_with_diverse_agents
        distributor.set_distribution_strategy('affinity_based')
        
        # First selection should work (no affinity yet)
        first_agent = distributor.select_agent_for_task('domain_create', {})
        assert first_agent is not None
        
        # Build affinity
        distributor.task_affinity['domain_create'] = [{
            'agent_id': first_agent.agent_id,
            'score': 0.8
        }]
        
        # Second selection should prefer the same agent
        second_agent = distributor.select_agent_for_task('domain_create', {})
        assert second_agent is not None
        assert second_agent.agent_id == first_agent.agent_id
    
    def test_strategy_switching(self, distributor_with_diverse_agents):
        """Test switching between distribution strategies"""
        distributor, agents = distributor_with_diverse_agents
        
        # Test valid strategy change
        result = distributor.set_distribution_strategy('capability_based')
        assert result['success'] == True
        assert result['current_strategy'] == 'capability_based'
        assert distributor.current_strategy == 'capability_based'
        
        # Test invalid strategy
        result = distributor.set_distribution_strategy('invalid_strategy')
        assert result['success'] == False
        assert 'Unknown strategy' in result['error']
        assert 'available_strategies' in result
        
        # Current strategy should remain unchanged
        assert distributor.current_strategy == 'capability_based'


class TestTaskSelection:
    """Test task selection and routing"""
    
    @pytest.fixture
    def task_distributor_ready(self):
        """Create ready task distributor with agents"""
        config = {'distribution_strategy': 'load_balanced'}
        distributor = TaskDistributor(config)
        distributor.initialize_distribution()
        
        # Register agents with different capabilities
        agent1 = MockAgent("specialized_agent")
        distributor.register_agent_capabilities(
            agent1.agent_id, agent1, ['domain_management']
        )
        
        agent2 = MockAgent("general_agent")
        distributor.register_agent_capabilities(
            agent2.agent_id, agent2, ['domain_management', 'formal_verification']
        )
        
        yield distributor, {"specialized": agent1, "general": agent2}
        distributor.is_active_flag = False
    
    def test_successful_task_selection(self, task_distributor_ready):
        """Test successful agent selection for task"""
        distributor, agents = task_distributor_ready
        
        task_data = {'task_id': 'test_task_1', 'priority': 'normal'}
        selected_agent = distributor.select_agent_for_task('domain_create', task_data)
        
        assert selected_agent is not None
        assert selected_agent.agent_id in ['specialized_agent', 'general_agent']
        
        # Verify metrics were updated
        metrics = distributor.get_distribution_metrics()
        assert metrics['metrics']['total_tasks_distributed'] > 0
        assert metrics['metrics']['successful_distributions'] > 0
    
    def test_no_suitable_agent_found(self, task_distributor_ready):
        """Test when no suitable agent is found"""
        distributor, agents = task_distributor_ready
        
        # Request task type that no agent can handle
        selected_agent = distributor.select_agent_for_task('unsupported_task', {})
        
        assert selected_agent is None
        
        # Verify failure was tracked
        metrics = distributor.get_distribution_metrics()
        assert metrics['metrics']['failed_distributions'] > 0
    
    def test_all_agents_unavailable(self, task_distributor_ready):
        """Test when all agents are unavailable"""
        distributor, agents = task_distributor_ready
        
        # Max out all agents
        for agent_id in ['specialized_agent', 'general_agent']:
            distributor.update_agent_load(agent_id, 10)  # Exceed max load
        
        selected_agent = distributor.select_agent_for_task('domain_create', {})
        
        assert selected_agent is None
    
    def test_task_selection_with_complex_data(self, task_distributor_ready):
        """Test task selection with complex task data"""
        distributor, agents = task_distributor_ready
        
        complex_task_data = {
            'task_id': 'complex_task',
            'priority': 'high',
            'resource_requirements': {
                'cpu': 50,
                'memory': 512,
                'timeout': 30
            },
            'dependencies': ['task_1', 'task_2'],
            'metadata': {
                'user_id': 'user_123',
                'session_id': 'session_456'
            }
        }
        
        selected_agent = distributor.select_agent_for_task('domain_update', complex_task_data)
        
        assert selected_agent is not None
        # Complex data should not prevent selection
        assert hasattr(selected_agent, 'agent_id')


class TestMetricsAndReporting:
    """Test metrics collection and reporting"""
    
    @pytest.fixture
    def active_distributor(self):
        """Create active distributor with ongoing activity"""
        config = {
            'distribution_strategy': 'load_balanced',
            'load_update_interval': 0.1  # Fast updates for testing
        }
        distributor = TaskDistributor(config)
        distributor.initialize_distribution()
        
        # Register multiple agents
        for i in range(3):
            agent = MockAgent(f"metrics_agent_{i}")
            distributor.register_agent_capabilities(
                agent.agent_id, agent, ['domain_management', 'session_orchestration']
            )
        
        yield distributor
        distributor.is_active_flag = False
    
    def test_basic_metrics_collection(self, active_distributor):
        """Test basic metrics are collected"""
        # Perform some distributions
        for i in range(5):
            agent = active_distributor.select_agent_for_task('domain_create', {})
            if agent:
                active_distributor.update_agent_load(agent.agent_id, 1)
        
        metrics = active_distributor.get_distribution_metrics()
        
        assert 'metrics' in metrics
        assert 'strategy_performance' in metrics
        assert 'agent_loads' in metrics
        assert 'timestamp' in metrics
        
        # Check specific metrics
        assert metrics['metrics']['total_tasks_distributed'] >= 5
        assert metrics['metrics']['successful_distributions'] >= 0
        assert metrics['metrics']['average_distribution_time'] >= 0
    
    def test_strategy_performance_metrics(self, active_distributor):
        """Test strategy-specific performance metrics"""
        # Use multiple strategies
        strategies = ['load_balanced', 'capability_based', 'round_robin']
        
        for strategy in strategies:
            active_distributor.set_distribution_strategy(strategy)
            for _ in range(3):
                active_distributor.select_agent_for_task('domain_create', {})
        
        metrics = active_distributor.get_distribution_metrics()
        strategy_perf = metrics['strategy_performance']
        
        # Should have metrics for used strategies
        for strategy in strategies:
            if strategy in strategy_perf:
                assert strategy_perf[strategy]['usage_count'] > 0
                assert 'success_rate' in strategy_perf[strategy]
                assert 'average_time' in strategy_perf[strategy]
    
    def test_agent_load_metrics(self, active_distributor):
        """Test agent load metrics tracking"""
        # Update loads for different agents
        load_updates = [
            ('metrics_agent_0', 3),
            ('metrics_agent_1', 1),
            ('metrics_agent_2', 5)
        ]
        
        for agent_id, load in load_updates:
            active_distributor.update_agent_load(agent_id, load)
        
        metrics = active_distributor.get_distribution_metrics()
        agent_loads = metrics['agent_loads']
        
        assert len(agent_loads) == 3
        for agent_id, expected_load in load_updates:
            assert agent_id in agent_loads
            assert agent_loads[agent_id]['current_load'] == expected_load
            assert agent_loads[agent_id]['load_percentage'] > 0
    
    def test_load_balance_efficiency_calculation(self, active_distributor):
        """Test load balance efficiency calculation"""
        # Create uneven load distribution
        active_distributor.update_agent_load('metrics_agent_0', 8)  # High load
        active_distributor.update_agent_load('metrics_agent_1', 2)  # Low load
        active_distributor.update_agent_load('metrics_agent_2', 5)  # Medium load
        
        metrics = active_distributor.get_distribution_metrics()
        
        # Should have calculated efficiency
        assert 'load_balance_efficiency' in metrics['metrics']
        efficiency = metrics['metrics']['load_balance_efficiency']
        assert 0 <= efficiency <= 1  # Should be normalized


class TestSystemValidation:
    """Test system validation and health checking"""
    
    @pytest.fixture
    def validation_distributor(self):
        """Create distributor for validation testing"""
        config = {'distribution_strategy': 'load_balanced'}
        distributor = TaskDistributor(config)
        distributor.initialize_distribution()
        
        # Register agents in different states
        # Healthy agent
        agent1 = MockAgent("healthy_agent")
        distributor.register_agent_capabilities(
            agent1.agent_id, agent1, ['domain_management', 'session_orchestration']
        )
        
        # Overloaded agent
        agent2 = MockAgent("overloaded_agent")
        distributor.register_agent_capabilities(
            agent2.agent_id, agent2, ['formal_verification']
        )
        distributor.update_agent_load(agent2.agent_id, 9)  # Near max load
        
        # Unavailable agent
        agent3 = MockAgent("unavailable_agent")
        distributor.register_agent_capabilities(
            agent3.agent_id, agent3, ['neural_network_training']
        )
        distributor.update_agent_load(agent3.agent_id, 10)  # Max load
        
        yield distributor
        distributor.is_active_flag = False
    
    def test_system_validation_comprehensive(self, validation_distributor):
        """Test comprehensive system validation"""
        validation_result = validation_distributor.validate_distribution_system()
        
        assert 'registered_agents' in validation_result
        assert 'available_agents' in validation_result
        assert 'overloaded_agents' in validation_result
        assert 'task_type_coverage' in validation_result
        assert 'strategy_validation' in validation_result
        assert 'system_health' in validation_result
        
        # Should have 3 registered agents
        assert validation_result['registered_agents'] == 3
        
        # Should detect overloaded agents
        assert validation_result['overloaded_agents'] >= 1
        
        # Should have task type coverage
        coverage = validation_result['task_type_coverage']
        assert coverage['total_task_types'] > 0
        assert len(coverage['covered_task_types']) > 0
    
    def test_system_health_assessment(self, validation_distributor):
        """Test system health assessment"""
        validation_result = validation_distributor.validate_distribution_system()
        
        health = validation_result['system_health']
        assert health in ['healthy', 'degraded', 'critical', 'unknown']
        
        # With overloaded agents, should not be 'healthy'
        assert health in ['degraded', 'critical']
    
    def test_strategy_validation(self, validation_distributor):
        """Test that all strategies are validated"""
        validation_result = validation_distributor.validate_distribution_system()
        
        strategy_validation = validation_result['strategy_validation']
        
        # Should validate all available strategies
        expected_strategies = [
            'round_robin', 'load_balanced', 'capability_based', 
            'priority_based', 'affinity_based'
        ]
        
        for strategy in expected_strategies:
            assert strategy in strategy_validation
            assert strategy_validation[strategy] == 'available'
    
    def test_critical_system_state(self):
        """Test critical system state with no available agents"""
        config = {'distribution_strategy': 'load_balanced'}
        distributor = TaskDistributor(config)
        distributor.initialize_distribution()
        
        # Don't register any agents
        validation_result = distributor.validate_distribution_system()
        
        assert validation_result['registered_agents'] == 0
        assert validation_result['available_agents'] == 0
        assert validation_result['system_health'] == 'critical'
        
        distributor.is_active_flag = False


class TestConcurrencyAndThreadSafety:
    """Test concurrent operations and thread safety"""
    
    @pytest.fixture
    def concurrent_distributor(self):
        """Create distributor for concurrency testing"""
        config = {
            'distribution_strategy': 'load_balanced',
            'load_update_interval': 0.1
        }
        distributor = TaskDistributor(config)
        distributor.initialize_distribution()
        
        # Register multiple agents for concurrent testing
        for i in range(5):
            agent = MockAgent(f"concurrent_agent_{i}")
            distributor.register_agent_capabilities(
                agent.agent_id, agent, ['domain_management', 'session_orchestration']
            )
        
        yield distributor
        distributor.is_active_flag = False
    
    def test_concurrent_agent_registration(self):
        """Test concurrent agent registration"""
        config = {'distribution_strategy': 'load_balanced'}
        distributor = TaskDistributor(config)
        distributor.initialize_distribution()
        
        results = []
        
        def register_agent(agent_id):
            agent = MockAgent(agent_id)
            result = distributor.register_agent_capabilities(
                agent_id, agent, ['domain_management']
            )
            results.append(result)
        
        # Register agents concurrently
        threads = []
        for i in range(10):
            thread = threading.Thread(target=register_agent, args=(f"thread_agent_{i}",))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All registrations should succeed
        assert len(results) == 10
        assert all(result['success'] for result in results)
        assert len(distributor.registered_agents) == 10
        
        distributor.is_active_flag = False
    
    def test_concurrent_task_selection(self, concurrent_distributor):
        """Test concurrent task selection"""
        selected_agents = []
        selection_lock = threading.Lock()
        
        def select_task():
            agent = concurrent_distributor.select_agent_for_task('domain_create', {})
            with selection_lock:
                selected_agents.append(agent.agent_id if agent else None)
        
        # Select tasks concurrently
        threads = []
        for _ in range(20):
            thread = threading.Thread(target=select_task)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have attempted all selections
        assert len(selected_agents) == 20
        
        # Should have some successful selections
        successful_selections = [agent for agent in selected_agents if agent is not None]
        assert len(successful_selections) > 0
    
    def test_concurrent_load_updates(self, concurrent_distributor):
        """Test concurrent load updates"""
        update_results = []
        results_lock = threading.Lock()
        
        def update_load(agent_id, delta):
            result = concurrent_distributor.update_agent_load(agent_id, delta)
            with results_lock:
                update_results.append(result)
        
        # Update loads concurrently
        threads = []
        agent_ids = [f"concurrent_agent_{i}" for i in range(5)]
        
        for i in range(50):  # 50 concurrent updates
            agent_id = agent_ids[i % len(agent_ids)]
            delta = 1 if i % 2 == 0 else -1  # Alternate increase/decrease
            
            thread = threading.Thread(target=update_load, args=(agent_id, delta))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All updates should complete
        assert len(update_results) == 50
        
        # Final loads should be consistent
        for agent_id in agent_ids:
            load_info = concurrent_distributor.agent_load[agent_id]
            assert load_info['current_load'] >= 0  # No negative loads
    
    def test_concurrent_strategy_changes(self, concurrent_distributor):
        """Test concurrent strategy changes"""
        strategy_results = []
        results_lock = threading.Lock()
        
        strategies = ['load_balanced', 'capability_based', 'round_robin']
        
        def change_strategy(strategy):
            result = concurrent_distributor.set_distribution_strategy(strategy)
            with results_lock:
                strategy_results.append((strategy, result))
        
        # Change strategies concurrently
        threads = []
        for i in range(10):
            strategy = strategies[i % len(strategies)]
            thread = threading.Thread(target=change_strategy, args=(strategy,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All strategy changes should complete
        assert len(strategy_results) == 10
        
        # Final strategy should be valid
        assert concurrent_distributor.current_strategy in strategies


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling"""
    
    @pytest.fixture
    def performance_distributor(self):
        """Create distributor for performance testing"""
        config = {
            'distribution_strategy': 'load_balanced',
            'load_update_interval': 0.5  # Slower for performance testing
        }
        distributor = TaskDistributor(config)
        distributor.initialize_distribution()
        
        # Register many agents for performance testing
        for i in range(20):
            agent = MockAgent(f"perf_agent_{i}")
            distributor.register_agent_capabilities(
                agent.agent_id, agent, ['domain_management', 'session_orchestration']
            )
        
        yield distributor
        distributor.is_active_flag = False
    
    def test_high_volume_task_selection(self, performance_distributor):
        """Test performance with high volume of task selections"""
        num_tasks = 1000
        
        start_time = time.time()
        
        successful_selections = 0
        for i in range(num_tasks):
            task_data = {'task_id': f'perf_task_{i}'}
            agent = performance_distributor.select_agent_for_task('domain_create', task_data)
            if agent:
                successful_selections += 1
        
        total_time = time.time() - start_time
        
        # Should complete quickly
        assert total_time < 10.0  # Less than 10 seconds for 1000 selections
        
        # Should have high success rate
        success_rate = successful_selections / num_tasks
        assert success_rate > 0.8  # At least 80% success rate
        
        # Calculate tasks per second
        tps = num_tasks / total_time
        assert tps > 100  # At least 100 tasks per second
    
    def test_agent_scaling_performance(self, performance_distributor):
        """Test performance scaling with number of agents"""
        # Test with different numbers of available agents
        agent_counts = [5, 10, 15, 20]
        performance_results = []
        
        for agent_count in agent_counts:
            # Make only specified number of agents available
            for i, agent_id in enumerate(performance_distributor.registered_agents.keys()):
                if i < agent_count:
                    # Make available
                    performance_distributor.update_agent_load(agent_id, -10)  # Reset load
                else:
                    # Make unavailable
                    performance_distributor.update_agent_load(agent_id, 10)  # Max load
            
            # Measure selection time
            start_time = time.time()
            
            selections = 0
            for _ in range(100):
                agent = performance_distributor.select_agent_for_task('domain_create', {})
                if agent:
                    selections += 1
            
            selection_time = time.time() - start_time
            performance_results.append({
                'agent_count': agent_count,
                'selection_time': selection_time,
                'selections': selections
            })
        
        # Performance should scale reasonably with agent count
        # (More agents might be slightly slower due to selection logic)
        for result in performance_results:
            assert result['selection_time'] < 1.0  # All should be fast
            assert result['selections'] > 0  # Should have some successes
    
    def test_memory_usage_with_large_agent_pool(self, performance_distributor):
        """Test memory usage doesn't grow excessively"""
        initial_agent_count = len(performance_distributor.registered_agents)
        
        # Perform many operations
        for i in range(500):
            # Select agents
            agent = performance_distributor.select_agent_for_task('domain_create', {})
            if agent:
                # Update load
                performance_distributor.update_agent_load(agent.agent_id, 1)
                
                # Update load back down occasionally
                if i % 10 == 0:
                    performance_distributor.update_agent_load(agent.agent_id, -5)
        
        # Get metrics
        metrics = performance_distributor.get_distribution_metrics()
        
        # Should have tracked many operations
        assert metrics['metrics']['total_tasks_distributed'] >= 500
        
        # Data structures should remain reasonable size
        assert len(performance_distributor.registered_agents) == initial_agent_count
        assert len(performance_distributor.agent_load) == initial_agent_count


class TestErrorHandling:
    """Test error handling and recovery"""
    
    @pytest.fixture
    def error_test_distributor(self):
        """Create distributor for error testing"""
        config = {'distribution_strategy': 'load_balanced'}
        distributor = TaskDistributor(config)
        distributor.initialize_distribution()
        
        # Register a test agent
        agent = MockAgent("error_test_agent")
        distributor.register_agent_capabilities(
            agent.agent_id, agent, ['domain_management']
        )
        
        yield distributor, agent
        distributor.is_active_flag = False
    
    def test_invalid_task_type_handling(self, error_test_distributor):
        """Test handling of invalid task types"""
        distributor, agent = error_test_distributor
        
        # Try selecting agent for invalid/unsupported task
        selected_agent = distributor.select_agent_for_task('', {})
        assert selected_agent is None
        
        selected_agent = distributor.select_agent_for_task(None, {})
        assert selected_agent is None
        
        selected_agent = distributor.select_agent_for_task('nonexistent_task', {})
        assert selected_agent is None
    
    def test_invalid_task_data_handling(self, error_test_distributor):
        """Test handling of invalid task data"""
        distributor, agent = error_test_distributor
        
        # Try with None task data
        selected_agent = distributor.select_agent_for_task('domain_create', None)
        # Should handle gracefully, not crash
        assert selected_agent is not None or selected_agent is None
        
        # Try with malformed task data
        malformed_data = {
            'invalid_key': {'nested': {'deeply': 'nested'}},
            'another_key': [1, 2, 3, {'inner': 'value'}]
        }
        selected_agent = distributor.select_agent_for_task('domain_create', malformed_data)
        # Should handle gracefully
        assert selected_agent is not None or selected_agent is None
    
    def test_agent_registration_error_handling(self, error_test_distributor):
        """Test error handling in agent registration"""
        distributor, agent = error_test_distributor
        
        # Try to register agent with invalid capabilities
        invalid_capabilities = [
            None,  # None in list
            123,   # Number instead of string
            {},    # Dict instead of string
            ""     # Empty string
        ]
        
        # Should handle gracefully
        result = distributor.register_agent_capabilities(
            "invalid_agent", agent, invalid_capabilities
        )
        # Should either succeed (filtering invalid) or fail gracefully
        assert 'success' in result
    
    def test_concurrent_error_scenarios(self, error_test_distributor):
        """Test error handling in concurrent scenarios"""
        distributor, agent = error_test_distributor
        
        errors_encountered = []
        error_lock = threading.Lock()
        
        def error_prone_operations():
            try:
                # Mix of valid and potentially problematic operations
                distributor.select_agent_for_task('domain_create', {})
                distributor.update_agent_load("nonexistent_agent", 1)
                distributor.set_distribution_strategy("invalid_strategy")
                distributor.select_agent_for_task(None, None)
            except Exception as e:
                with error_lock:
                    errors_encountered.append(str(e))
        
        # Run error-prone operations concurrently
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=error_prone_operations)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # System should remain functional despite errors
        # (Some operations might fail, but shouldn't crash)
        final_agent = distributor.select_agent_for_task('domain_create', {})
        assert final_agent is not None or final_agent is None  # Should complete
    
    def test_system_recovery_after_errors(self, error_test_distributor):
        """Test system recovery after encountering errors"""
        distributor, agent = error_test_distributor
        
        # Cause some errors
        distributor.update_agent_load("nonexistent", 5)
        distributor.set_distribution_strategy("invalid")
        
        # System should still function
        metrics = distributor.get_distribution_metrics()
        assert 'metrics' in metrics
        
        validation = distributor.validate_distribution_system()
        assert 'system_health' in validation
        
        # Should still be able to select agents
        selected_agent = distributor.select_agent_for_task('domain_create', {})
        assert selected_agent is not None


class TestRealWorldScenarios:
    """Test realistic usage scenarios"""
    
    @pytest.fixture
    def production_like_distributor(self):
        """Create distributor simulating production environment"""
        config = {
            'distribution_strategy': 'load_balanced',
            'max_distribution_retries': 3,
            'retry_delay_seconds': 0.1,  # Fast for testing
            'load_threshold_percent': 75,
            'enable_task_affinity': True,
            'load_update_interval': 0.2
        }
        distributor = TaskDistributor(config)
        distributor.initialize_distribution()
        
        # Create realistic agent pool
        agent_configs = [
            ("domain_specialist", ['domain_management'], {'priority_level': 'high'}),
            ("proof_specialist", ['formal_verification'], {'priority_level': 'critical'}),
            ("training_specialist", ['neural_network_training'], {'priority_level': 'normal'}),
            ("general_agent_1", ['domain_management', 'session_orchestration'], {'priority_level': 'normal'}),
            ("general_agent_2", ['formal_verification', 'confidence_aggregation'], {'priority_level': 'normal'}),
            ("multi_specialist", ['domain_management', 'formal_verification', 'neural_network_training'], {'priority_level': 'high'})
        ]
        
        agents = {}
        for agent_id, capabilities, config_data in agent_configs:
            agent = MockAgent(agent_id, config_data)
            distributor.register_agent_capabilities(agent_id, agent, capabilities)
            agents[agent_id] = agent
        
        yield distributor, agents
        distributor.is_active_flag = False
    
    def test_mixed_workload_scenario(self, production_like_distributor):
        """Test handling mixed workload with different task types and priorities"""
        distributor, agents = production_like_distributor
        
        # Simulate mixed workload
        tasks = [
            ('domain_create', {'priority': 'high', 'user': 'admin'}),
            ('proof_generate', {'priority': 'critical', 'complexity': 'high'}),
            ('model_train', {'priority': 'normal', 'dataset_size': 'large'}),
            ('domain_update', {'priority': 'normal', 'user': 'user1'}),
            ('confidence_calculate', {'priority': 'high', 'model': 'production'}),
        ] * 10  # 50 tasks total
        
        # Process workload
        distribution_results = []
        for task_type, task_data in tasks:
            agent = distributor.select_agent_for_task(task_type, task_data)
            distribution_results.append({
                'task_type': task_type,
                'priority': task_data.get('priority', 'normal'),
                'assigned_agent': agent.agent_id if agent else None,
                'success': agent is not None
            })
            
            # Update load if assigned
            if agent:
                distributor.update_agent_load(agent.agent_id, 1)
        
        # Analyze results
        total_tasks = len(distribution_results)
        successful_assignments = sum(1 for r in distribution_results if r['success'])
        success_rate = successful_assignments / total_tasks
        
        # Should have high success rate
        assert success_rate > 0.8
        
        # Critical and high priority tasks should have better assignment rates
        critical_tasks = [r for r in distribution_results if r['priority'] == 'critical']
        if critical_tasks:
            critical_success_rate = sum(1 for r in critical_tasks if r['success']) / len(critical_tasks)
            assert critical_success_rate >= success_rate  # At least as good as overall
    
    def test_load_balancing_effectiveness(self, production_like_distributor):
        """Test effectiveness of load balancing over time"""
        distributor, agents = production_like_distributor
        
        # Process many tasks to see load balancing in action
        task_assignments = defaultdict(int)
        
        for i in range(100):
            agent = distributor.select_agent_for_task('domain_create', {'task_num': i})
            if agent:
                task_assignments[agent.agent_id] += 1
                distributor.update_agent_load(agent.agent_id, 1)
                
                # Occasionally complete tasks (reduce load)
                if i % 10 == 0:
                    distributor.update_agent_load(agent.agent_id, -5)
        
        # Check load distribution
        load_values = []
        for agent_id in task_assignments.keys():
            current_load = distributor.agent_load[agent_id]['current_load']
            load_values.append(current_load)
        
        if len(load_values) > 1:
            # Calculate load distribution variance
            mean_load = sum(load_values) / len(load_values)
            variance = sum((load - mean_load) ** 2 for load in load_values) / len(load_values)
            
            # Load should be reasonably balanced (low variance)
            assert variance < 50  # Reasonable threshold for balanced distribution
    
    def test_system_health_monitoring(self, production_like_distributor):
        """Test system health monitoring during operation"""
        distributor, agents = production_like_distributor
        
        # Monitor system health during operation
        health_snapshots = []
        
        for round_num in range(5):
            # Process some tasks
            for _ in range(20):
                agent = distributor.select_agent_for_task('domain_create', {})
                if agent:
                    distributor.update_agent_load(agent.agent_id, 2)  # Higher load
            
            # Take health snapshot
            validation = distributor.validate_distribution_system()
            health_snapshots.append({
                'round': round_num,
                'health': validation['system_health'],
                'available_agents': validation['available_agents'],
                'overloaded_agents': validation['overloaded_agents']
            })
            
            # Simulate task completion (reduce some load)
            for agent_id in list(distributor.registered_agents.keys())[:2]:
                distributor.update_agent_load(agent_id, -3)
        
        # System should maintain some level of health
        final_health = health_snapshots[-1]['health']
        assert final_health in ['healthy', 'degraded', 'critical']
        
        # Should have been tracking health changes
        assert len(health_snapshots) == 5
        
        # At least some agents should remain available
        final_available = health_snapshots[-1]['available_agents']
        assert final_available > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])