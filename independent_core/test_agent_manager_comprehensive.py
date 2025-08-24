#!/usr/bin/env python3
"""
Comprehensive test for AgentManager to identify all issues
NO MOCKS - REAL VALIDATION WITH HARD FAILURES ONLY
"""

import time
import traceback
from typing import Dict, Any, List, Tuple
import threading
import json
import tempfile
import shutil
from pathlib import Path
import sys
import os


def test_component(name: str, test_func, *args, **kwargs) -> Tuple[bool, str]:
    """Test a component and return status - NO FALLBACKS"""
    try:
        test_func(*args, **kwargs)
        return True, ""
    except Exception as e:
        tb = traceback.format_exc()
        return False, f"Error: {str(e)}\nTraceback: {tb}"


class ComprehensiveAgentManagerTester:
    def __init__(self):
        self.results = []
        self.temp_dirs = []
        self.created_agents = []
    
    def cleanup(self):
        """Clean up temporary resources"""
        for temp_dir in self.temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except:
                pass
        # Clean up any created agents
        for agent in self.created_agents:
            try:
                if hasattr(agent, 'cleanup'):
                    agent.cleanup()
            except:
                pass
    
    def test_imports(self):
        """Test that all imports work - HARD FAILURE IF MISSING"""
        from multi_agent_framework.agent_manager import AgentManager
        assert AgentManager is not None
        
        # Test that it can access brain_core
        try:
            from brain_core import BrainCore
            assert BrainCore is not None
        except ImportError as e:
            raise ImportError(f"BrainCore import failed - required dependency: {e}")
    
    def test_basic_initialization(self):
        """Test basic AgentManager initialization"""
        from multi_agent_framework.agent_manager import AgentManager
        
        # Test with empty config
        manager = AgentManager({})
        assert manager is not None
        assert hasattr(manager, 'config')
        assert hasattr(manager, 'agent_templates')
        assert hasattr(manager, 'agent_factories')
        assert hasattr(manager, 'management_metrics')
        assert hasattr(manager, '_lock')
        
        # Test with comprehensive config
        config = {
            'max_agents': 50,
            'default_memory_mb': 256,
            'default_cpu_percent': 30,
            'agent_timeout_seconds': 300,
            'enable_monitoring': True,
            'security_level': 'high'
        }
        manager = AgentManager(config)
        assert manager.config == config
        
        # Verify initialization metrics
        metrics = manager.get_management_metrics()
        assert isinstance(metrics, dict)
        assert 'total_agents_created' in metrics
        assert 'active_agents' in metrics
        assert 'failed_creations' in metrics
    
    def test_agent_initialization_with_brain_core(self):
        """Test agent initialization with brain core integration"""
        from multi_agent_framework.agent_manager import AgentManager
        from brain_core import BrainCore
        
        # Create real brain core
        brain_core = BrainCore()
        assert brain_core is not None
        
        # Create agent manager
        manager = AgentManager({})
        
        # Initialize agents with brain core
        result = manager.initialize_agents(brain_core)
        
        # Verify initialization results
        assert isinstance(result, dict)
        assert 'success' in result
        assert result['success'] == True
        assert 'agent_templates_initialized' in result
        assert 'agent_factories_initialized' in result
        assert 'default_agents_created' in result
        assert 'template_validation' in result
        
        # Verify brain core reference is stored
        assert hasattr(manager, 'brain_core')
        assert manager.brain_core is brain_core
        
        # Verify templates were initialized
        templates = manager.get_agent_templates()
        assert isinstance(templates, dict)
        assert len(templates) > 0
    
    def test_brain_orchestration_agent_creation(self):
        """Test brain orchestration agent creation"""
        from multi_agent_framework.agent_manager import AgentManager
        from brain_core import BrainCore
        
        brain_core = BrainCore()
        manager = AgentManager({})
        manager.initialize_agents(brain_core)
        
        # Test agent creation with various configs
        configs = [
            {},  # Empty config
            {
                'max_memory_mb': 512,
                'max_cpu_percent': 60,
                'max_concurrent_tasks': 20
            },
            {
                'custom_capabilities': ['domain_analysis'],
                'monitoring_enabled': True
            }
        ]
        
        for i, config in enumerate(configs):
            agent = manager.create_brain_orchestration_agent(config)
            assert agent is not None
            self.created_agents.append(agent)
            
            # Verify agent has required attributes
            assert hasattr(agent, 'config')
            assert hasattr(agent, 'agent_type')
            assert agent.config['agent_type'] == 'brain_orchestration'
            
        # Verify metrics updated
        metrics = manager.get_management_metrics()
        assert metrics['total_agents_created'] >= len(configs)
    
    def test_proof_system_agent_creation(self):
        """Test proof system agent creation"""
        from multi_agent_framework.agent_manager import AgentManager
        from brain_core import BrainCore
        
        brain_core = BrainCore()
        manager = AgentManager({})
        manager.initialize_agents(brain_core)
        
        config = {
            'max_memory_mb': 1024,
            'max_cpu_percent': 80,
            'max_concurrent_proofs': 10,
            'proof_strategies': ['formal', 'ml_based']
        }
        
        agent = manager.create_proof_system_agent(config)
        assert agent is not None
        self.created_agents.append(agent)
        
        # Verify proof system specific configuration
        assert agent.config['agent_type'] == 'proof_system'
        assert 'proof_strategies' in agent.config
        assert 'formal_verification' in agent.config['capabilities']
        assert agent.config['security_level'] == 'critical'
    
    def test_uncertainty_agent_creation(self):
        """Test uncertainty quantification agent creation"""
        from multi_agent_framework.agent_manager import AgentManager
        from brain_core import BrainCore
        
        brain_core = BrainCore()
        manager = AgentManager({})
        manager.initialize_agents(brain_core)
        
        config = {
            'max_memory_mb': 768,
            'max_concurrent_quantifications': 12,
            'uncertainty_methods': ['conformalized_credal', 'ensemble']
        }
        
        agent = manager.create_uncertainty_agent(config)
        assert agent is not None
        self.created_agents.append(agent)
        
        # Verify uncertainty specific configuration
        assert agent.config['agent_type'] == 'uncertainty'
        assert 'conformalized_credal_quantification' in agent.config['capabilities']
        assert agent.config['priority_level'] == 'high'
    
    def test_training_agent_creation(self):
        """Test training agent creation"""
        from multi_agent_framework.agent_manager import AgentManager
        from brain_core import BrainCore
        
        brain_core = BrainCore()
        manager = AgentManager({})
        manager.initialize_agents(brain_core)
        
        config = {
            'training_methods': ['supervised', 'reinforcement'],
            'batch_size': 64,
            'learning_rate': 0.001,
            'max_epochs': 100
        }
        
        agent = manager.create_training_agent(config)
        assert agent is not None
        self.created_agents.append(agent)
        
        # Verify training specific configuration
        assert agent.config['agent_type'] == 'training'
        assert 'adaptive_learning_rate' in agent.config['capabilities']
    
    def test_domain_agent_creation(self):
        """Test domain specialist agent creation"""
        from multi_agent_framework.agent_manager import AgentManager
        from brain_core import BrainCore
        
        brain_core = BrainCore()
        manager = AgentManager({})
        manager.initialize_agents(brain_core)
        
        config = {
            'domain_specialization': 'mathematics',
            'expertise_level': 'expert',
            'knowledge_domains': ['algebra', 'calculus', 'topology']
        }
        
        agent = manager.create_domain_agent(config)
        assert agent is not None
        self.created_agents.append(agent)
        
        # Verify domain specific configuration
        assert agent.config['agent_type'] == 'domain'
        assert 'domain_expertise' in agent.config['capabilities']
        assert agent.config['domain_specialization'] == 'mathematics'
    
    def test_compression_agent_creation(self):
        """Test compression agent creation"""
        from multi_agent_framework.agent_manager import AgentManager
        from brain_core import BrainCore
        
        brain_core = BrainCore()
        manager = AgentManager({})
        manager.initialize_agents(brain_core)
        
        config = {
            'compression_methods': ['padic', 'tropical', 'sheaf'],
            'compression_ratio_target': 0.1,
            'quality_threshold': 0.95
        }
        
        agent = manager.create_compression_agent(config)
        assert agent is not None
        self.created_agents.append(agent)
        
        # Verify compression specific configuration
        assert agent.config['agent_type'] == 'compression'
        assert 'padic_compression' in agent.config['capabilities']
    
    def test_production_agent_creation(self):
        """Test production agent creation"""
        from multi_agent_framework.agent_manager import AgentManager
        from brain_core import BrainCore
        
        brain_core = BrainCore()
        manager = AgentManager({})
        manager.initialize_agents(brain_core)
        
        config = {
            'environment': 'production',
            'monitoring_level': 'comprehensive',
            'auto_scaling': True,
            'health_check_interval': 30
        }
        
        agent = manager.create_production_agent(config)
        assert agent is not None
        self.created_agents.append(agent)
        
        # Verify production specific configuration
        assert agent.config['agent_type'] == 'production'
        assert 'production_optimization' in agent.config['capabilities']
        assert agent.config['security_level'] == 'critical'
    
    def test_web_interface_agent_creation(self):
        """Test web interface agent creation"""
        from multi_agent_framework.agent_manager import AgentManager
        from brain_core import BrainCore
        
        brain_core = BrainCore()
        manager = AgentManager({})
        manager.initialize_agents(brain_core)
        
        config = {
            'port': 8080,
            'enable_api': True,
            'cors_enabled': True,
            'rate_limiting': True,
            'max_requests_per_minute': 1000
        }
        
        agent = manager.create_web_interface_agent(config)
        assert agent is not None
        self.created_agents.append(agent)
        
        # Verify web interface specific configuration
        assert agent.config['agent_type'] == 'web_interface'
        assert 'web_server_management' in agent.config['capabilities']
    
    def test_template_management(self):
        """Test agent template management"""
        from multi_agent_framework.agent_manager import AgentManager
        from brain_core import BrainCore
        
        brain_core = BrainCore()
        manager = AgentManager({})
        manager.initialize_agents(brain_core)
        
        # Get initial templates
        initial_templates = manager.get_agent_templates()
        assert isinstance(initial_templates, dict)
        
        # Create an agent to register a template
        config = {'test_param': 'test_value'}
        agent = manager.create_brain_orchestration_agent(config)
        self.created_agents.append(agent)
        
        # Verify template was registered
        updated_templates = manager.get_agent_templates()
        assert len(updated_templates) >= len(initial_templates)
        assert 'brain_orchestration' in updated_templates
        
        # Verify template structure
        template = updated_templates['brain_orchestration']
        assert isinstance(template, dict)
        assert 'agent_type' in template
        assert 'capabilities' in template
        assert 'resource_limits' in template
    
    def test_metrics_tracking(self):
        """Test management metrics tracking"""
        from multi_agent_framework.agent_manager import AgentManager
        from brain_core import BrainCore
        
        brain_core = BrainCore()
        manager = AgentManager({})
        manager.initialize_agents(brain_core)
        
        # Get initial metrics
        initial_metrics = manager.get_management_metrics()
        initial_count = initial_metrics['total_agents_created']
        
        # Create multiple agents
        configs = [{}, {'test': 'config1'}, {'test': 'config2'}]
        for config in configs:
            agent = manager.create_brain_orchestration_agent(config)
            self.created_agents.append(agent)
        
        # Verify metrics updated
        updated_metrics = manager.get_management_metrics()
        assert updated_metrics['total_agents_created'] >= initial_count + len(configs)
        assert 'average_creation_time' in updated_metrics
        assert updated_metrics['average_creation_time'] >= 0
        assert 'template_usage' in updated_metrics
        assert updated_metrics['template_usage']['brain_orchestration'] >= len(configs)
    
    def test_thread_safety(self):
        """Test thread safety of agent creation"""
        from multi_agent_framework.agent_manager import AgentManager
        from brain_core import BrainCore
        
        brain_core = BrainCore()
        manager = AgentManager({})
        manager.initialize_agents(brain_core)
        
        agents_created = []
        exceptions_caught = []
        
        def create_agent_worker(worker_id):
            try:
                config = {'worker_id': worker_id}
                agent = manager.create_brain_orchestration_agent(config)
                agents_created.append(agent)
            except Exception as e:
                exceptions_caught.append(e)
        
        # Create multiple threads
        threads = []
        num_threads = 5
        for i in range(num_threads):
            t = threading.Thread(target=create_agent_worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Verify results
        assert len(exceptions_caught) == 0, f"Thread safety failed: {exceptions_caught}"
        assert len(agents_created) == num_threads
        
        # Add to cleanup
        self.created_agents.extend(agents_created)
        
        # Verify metrics consistency
        metrics = manager.get_management_metrics()
        assert metrics['total_agents_created'] >= num_threads
    
    def test_error_handling(self):
        """Test error handling and failure scenarios"""
        from multi_agent_framework.agent_manager import AgentManager
        from brain_core import BrainCore
        
        brain_core = BrainCore()
        manager = AgentManager({})
        manager.initialize_agents(brain_core)
        
        # Test invalid configurations
        invalid_configs = [
            {'max_memory_mb': -1},  # Negative memory
            {'max_cpu_percent': 150},  # Invalid CPU percentage
            {'max_concurrent_tasks': 0},  # Zero tasks
        ]
        
        for config in invalid_configs:
            try:
                agent = manager.create_brain_orchestration_agent(config)
                # If no exception, verify the agent handles invalid config appropriately
                assert agent is not None
                self.created_agents.append(agent)
            except Exception as e:
                # Verify exception is properly handled and informative
                assert isinstance(e, (ValueError, RuntimeError))
                assert str(e) != ""  # Non-empty error message
        
        # Verify metrics track failures appropriately
        metrics = manager.get_management_metrics()
        assert 'failed_creations' in metrics
        assert isinstance(metrics['failed_creations'], int)
    
    def test_resource_management(self):
        """Test resource management and limits"""
        from multi_agent_framework.agent_manager import AgentManager
        from brain_core import BrainCore
        
        brain_core = BrainCore()
        manager = AgentManager({})
        manager.initialize_agents(brain_core)
        
        # Test resource limit enforcement
        high_resource_config = {
            'max_memory_mb': 2048,
            'max_cpu_percent': 90,
            'max_concurrent_tasks': 50
        }
        
        agent = manager.create_brain_orchestration_agent(high_resource_config)
        assert agent is not None
        self.created_agents.append(agent)
        
        # Verify resource limits are properly set
        assert agent.config['resource_limits']['max_memory_mb'] == 2048
        assert agent.config['resource_limits']['max_cpu_percent'] == 90
        assert agent.config['resource_limits']['max_concurrent_tasks'] == 50
    
    def test_comprehensive_integration(self):
        """Test comprehensive integration scenario"""
        from multi_agent_framework.agent_manager import AgentManager
        from brain_core import BrainCore
        
        brain_core = BrainCore()
        manager = AgentManager({
            'max_agents': 100,
            'monitoring_enabled': True
        })
        
        # Initialize system
        init_result = manager.initialize_agents(brain_core)
        assert init_result['success'] == True
        
        # Create agents of different types
        agent_configs = [
            ('brain_orchestration', {}),
            ('proof_system', {'proof_strategies': ['formal']}),
            ('uncertainty', {'max_concurrent_quantifications': 5}),
            ('training', {'learning_rate': 0.01}),
            ('domain', {'domain_specialization': 'science'}),
            ('compression', {'compression_methods': ['padic']}),
            ('production', {'environment': 'staging'}),
            ('web_interface', {'port': 9090})
        ]
        
        created_agents = []
        for agent_type, config in agent_configs:
            method_name = f'create_{agent_type}_agent'
            create_method = getattr(manager, method_name)
            agent = create_method(config)
            assert agent is not None
            created_agents.append(agent)
            
        self.created_agents.extend(created_agents)
        
        # Verify comprehensive system state
        final_metrics = manager.get_management_metrics()
        assert final_metrics['total_agents_created'] >= len(agent_configs)
        
        final_templates = manager.get_agent_templates()
        assert len(final_templates) >= len(set(t[0] for t in agent_configs))
    
    def run_all_tests(self) -> Tuple[int, int]:
        """Run all tests and return results - NO FALLBACKS"""
        test_methods = [
            ("Imports", self.test_imports),
            ("Basic Initialization", self.test_basic_initialization),
            ("Agent Initialization with Brain Core", self.test_agent_initialization_with_brain_core),
            ("Brain Orchestration Agent Creation", self.test_brain_orchestration_agent_creation),
            ("Proof System Agent Creation", self.test_proof_system_agent_creation),
            ("Uncertainty Agent Creation", self.test_uncertainty_agent_creation),
            ("Training Agent Creation", self.test_training_agent_creation),
            ("Domain Agent Creation", self.test_domain_agent_creation),
            ("Compression Agent Creation", self.test_compression_agent_creation),
            ("Production Agent Creation", self.test_production_agent_creation),
            ("Web Interface Agent Creation", self.test_web_interface_agent_creation),
            ("Template Management", self.test_template_management),
            ("Metrics Tracking", self.test_metrics_tracking),
            ("Thread Safety", self.test_thread_safety),
            ("Error Handling", self.test_error_handling),
            ("Resource Management", self.test_resource_management),
            ("Comprehensive Integration", self.test_comprehensive_integration)
        ]
        
        print("\n" + "="*80)
        print("COMPREHENSIVE AGENT MANAGER TESTING")
        print("="*80)
        
        passed = 0
        failed = 0
        failed_tests = []
        
        for test_name, test_func in test_methods:
            success, error = test_component(test_name, test_func)
            
            if success:
                print(f"✅ {test_name:<35} - OK")
                passed += 1
            else:
                print(f"❌ {test_name:<35} - FAILED")
                failed += 1
                failed_tests.append((test_name, error))
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total Tests: {passed + failed}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        
        if failed_tests:
            print("\nFAILED TESTS:")
            for test_name, error in failed_tests:
                print(f"  - {test_name}: {error[:100]}...")
                if len(error) > 100:
                    print(f"    Full error: {error}")
        
        print("\n" + "="*80)
        print("ACTION ITEMS")
        print("="*80)
        
        if failed == 0:
            print("\n✅ All AgentManager tests passed!")
        else:
            print("\nIssues to fix (in order of priority):")
            for i, (test_name, error) in enumerate(failed_tests, 1):
                print(f"{i}. {test_name}: {error.split('Traceback')[0].strip()}")
        
        self.cleanup()
        return passed, failed


def main():
    tester = ComprehensiveAgentManagerTester()
    passed, failed = tester.run_all_tests()
    
    # Exit with appropriate code
    exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()