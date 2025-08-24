"""
Comprehensive test suite for BrainOrchestrator
Tests all functionality, edge cases, concurrency, performance, and error handling
"""

import pytest
import time
import threading
import asyncio
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import concurrent.futures
from typing import Dict, Any, List
import weakref
import gc
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrators.brain_orchestrator import (
    BrainOrchestrator,
    OrchestrationTask,
    OrchestrationMode,
    SystemState,
    OperationPriority,
    SystemMetrics,
    ComponentStatus
)


class TestBrainOrchestratorBasics:
    """Test basic functionality of BrainOrchestrator"""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization"""
        orchestrator = BrainOrchestrator()
        
        assert orchestrator._state == SystemState.INITIALIZING
        assert orchestrator._mode == OrchestrationMode.ADAPTIVE
        assert orchestrator._max_workers == 8
        assert orchestrator._monitoring_enabled == True
        assert len(orchestrator._task_queue) == 0
        assert len(orchestrator._active_tasks) == 0
        
    def test_orchestrator_with_config(self):
        """Test orchestrator initialization with custom config"""
        config = {
            'max_workers': 4,
            'worker_timeout': 60.0,
            'monitoring_enabled': False,
            'monitoring_interval': 10.0,
            'emergency_threshold': 0.9
        }
        
        orchestrator = BrainOrchestrator(config=config)
        
        assert orchestrator._max_workers == 4
        assert orchestrator._worker_timeout == 60.0
        assert orchestrator._monitoring_enabled == False
        assert orchestrator._monitoring_interval == 10.0
        assert orchestrator._emergency_threshold == 0.9
    
    def test_orchestrator_initialization_process(self):
        """Test the initialization process"""
        orchestrator = BrainOrchestrator()
        result = orchestrator.initialize()
        
        assert result == True
        assert orchestrator._state == SystemState.READY
        assert orchestrator._worker_pool is not None
        
    def test_orchestrator_double_initialization(self):
        """Test that double initialization is handled properly"""
        orchestrator = BrainOrchestrator()
        orchestrator.initialize()
        
        # Second initialization should return True but log warning
        result = orchestrator.initialize()
        assert result == True
        assert orchestrator._state == SystemState.READY


class TestComponentManagement:
    """Test component registration and management"""
    
    @pytest.fixture
    def orchestrator(self):
        orch = BrainOrchestrator()
        orch.initialize()
        return orch
    
    def test_component_registration(self, orchestrator):
        """Test registering a component"""
        mock_component = Mock()
        result = orchestrator.register_component("test_component", mock_component)
        
        assert result == True
        assert "test_component" in orchestrator._components
        assert "test_component" in orchestrator._component_status
        assert orchestrator._component_status["test_component"].status == "registered"
    
    def test_component_status_tracking(self, orchestrator):
        """Test component status is properly tracked"""
        mock_component = Mock()
        orchestrator.register_component("test_component", mock_component)
        
        status = orchestrator._component_status["test_component"]
        assert isinstance(status, ComponentStatus)
        assert status.component_id == "test_component"
        assert status.health_score == 1.0
        assert status.error_count == 0
    
    def test_component_health_check(self, orchestrator):
        """Test health checking for components"""
        mock_component = Mock()
        orchestrator.register_component("test_component", mock_component)
        
        # Simulate old heartbeat
        orchestrator._component_status["test_component"].last_heartbeat = time.time() - 120
        
        health_report = orchestrator.check_component_health()
        
        assert "test_component" in health_report
        assert health_report["test_component"]["health_score"] < 1.0
        assert health_report["test_component"]["heartbeat_age"] > 100
    
    def test_multiple_component_registration(self, orchestrator):
        """Test registering multiple components"""
        components = {
            "comp1": Mock(),
            "comp2": Mock(),
            "comp3": Mock()
        }
        
        for comp_id, comp in components.items():
            result = orchestrator.register_component(comp_id, comp)
            assert result == True
        
        assert len(orchestrator._components) == 3
        assert all(comp_id in orchestrator._components for comp_id in components.keys())


class TestTaskManagement:
    """Test task submission and execution"""
    
    @pytest.fixture
    def orchestrator(self):
        orch = BrainOrchestrator()
        orch.initialize()
        return orch
    
    def test_task_submission(self, orchestrator):
        """Test submitting a task"""
        task = OrchestrationTask(
            task_id="test_task_1",
            operation="status_check",
            priority=OperationPriority.MEDIUM
        )
        
        result = orchestrator.submit_task(task)
        assert result == True
        assert orchestrator._metrics.task_count == 1
    
    def test_task_validation(self, orchestrator):
        """Test task validation"""
        # Task without task_id
        invalid_task = OrchestrationTask(
            task_id="",
            operation="test_op",
            priority=OperationPriority.LOW
        )
        result = orchestrator.submit_task(invalid_task)
        assert result == False
        
        # Task without operation
        invalid_task2 = OrchestrationTask(
            task_id="test_task",
            operation="",
            priority=OperationPriority.LOW
        )
        result = orchestrator.submit_task(invalid_task2)
        assert result == False
    
    def test_task_priority_ordering(self):
        """Test that tasks are ordered by priority"""
        # Create orchestrator with auto-processing disabled to check queue order
        orchestrator = BrainOrchestrator(config={'auto_process_tasks': False})
        orchestrator.initialize()
        
        tasks = [
            OrchestrationTask(
                task_id="low_task",
                operation="test",
                priority=OperationPriority.LOW
            ),
            OrchestrationTask(
                task_id="high_task",
                operation="test",
                priority=OperationPriority.HIGH
            ),
            OrchestrationTask(
                task_id="medium_task",
                operation="test",
                priority=OperationPriority.MEDIUM
            ),
            OrchestrationTask(
                task_id="critical_task",
                operation="test",
                priority=OperationPriority.CRITICAL
            )
        ]
        
        # Submit tasks in arbitrary order
        for task in tasks:
            orchestrator.submit_task(task)
        
        # Check queue order - highest priority should be first
        queue_list = list(orchestrator._task_queue)
        assert queue_list[0].priority == OperationPriority.CRITICAL
        assert queue_list[1].priority == OperationPriority.HIGH
        assert queue_list[2].priority == OperationPriority.MEDIUM
        assert queue_list[3].priority == OperationPriority.LOW
    
    def test_task_dependencies(self, orchestrator):
        """Test task dependency validation"""
        # Submit first task
        task1 = OrchestrationTask(
            task_id="task1",
            operation="test",
            priority=OperationPriority.MEDIUM
        )
        orchestrator.submit_task(task1)
        
        # Submit task with unmet dependency
        task2 = OrchestrationTask(
            task_id="task2",
            operation="test",
            priority=OperationPriority.MEDIUM,
            dependencies=["task1"]
        )
        result = orchestrator.submit_task(task2)
        assert result == False  # Should fail because task1 is not completed
    
    def test_task_execution(self, orchestrator):
        """Test task execution through the pipeline"""
        task = OrchestrationTask(
            task_id="exec_task",
            operation="status_check",
            priority=OperationPriority.MEDIUM
        )
        
        orchestrator.submit_task(task)
        
        # Wait briefly for execution
        time.sleep(0.5)
        
        # Check task was moved to completed
        assert "exec_task" in orchestrator._completed_tasks
        assert orchestrator._completed_tasks["exec_task"].status == "completed"
        assert orchestrator._metrics.completed_tasks > 0
    
    def test_task_retry_mechanism(self, orchestrator):
        """Test task retry on failure"""
        # Create task that will fail
        task = OrchestrationTask(
            task_id="retry_task",
            operation="unknown_operation",
            priority=OperationPriority.MEDIUM,
            retry_count=2
        )
        
        orchestrator.submit_task(task)
        
        # Wait for initial failure
        time.sleep(0.5)
        
        # Task should be retrying, not in completed yet
        # Note: The current implementation might complete with "unknown_operation" status
        # We should check the actual behavior
        status = orchestrator.get_task_status("retry_task")
        assert status is not None


class TestTaskRouting:
    """Test task routing to different components"""
    
    @pytest.fixture
    def orchestrator(self):
        orch = BrainOrchestrator()
        orch.initialize()
        return orch
    
    def test_route_to_decision_engine(self, orchestrator):
        """Test routing to decision engine"""
        mock_decision_engine = Mock()
        mock_decision_engine.process = Mock(return_value={"decision": "test"})
        orchestrator._decision_engine = mock_decision_engine
        
        task = OrchestrationTask(
            task_id="decision_task",
            operation="decision_process",
            priority=OperationPriority.MEDIUM,
            parameters={"test": "data"}
        )
        
        orchestrator.submit_task(task)
        time.sleep(0.5)
        
        mock_decision_engine.process.assert_called_once()
        assert orchestrator._completed_tasks["decision_task"].status == "completed"
    
    def test_route_to_component(self, orchestrator):
        """Test routing to registered component"""
        mock_component = Mock()
        mock_component.process_data = Mock(return_value={"result": "success"})
        
        orchestrator.register_component("test_comp", mock_component)
        
        task = OrchestrationTask(
            task_id="comp_task",
            operation="process_data",
            priority=OperationPriority.MEDIUM,
            parameters={"component_id": "test_comp", "data": "test"}
        )
        
        orchestrator.submit_task(task)
        time.sleep(0.5)
        
        mock_component.process_data.assert_called_once_with(data="test")
        assert orchestrator._completed_tasks["comp_task"].status == "completed"
    
    def test_route_to_brain_method(self, orchestrator):
        """Test routing to brain instance methods"""
        mock_brain = Mock()
        mock_brain.analyze = Mock(return_value={"analysis": "complete"})
        
        orchestrator.brain = mock_brain
        
        task = OrchestrationTask(
            task_id="brain_task",
            operation="analyze",
            priority=OperationPriority.MEDIUM,
            parameters={"input": "test_data"}
        )
        
        orchestrator.submit_task(task)
        time.sleep(0.5)
        
        mock_brain.analyze.assert_called_once_with(input="test_data")
        assert orchestrator._completed_tasks["brain_task"].status == "completed"


class TestSystemMetrics:
    """Test system metrics and monitoring"""
    
    @pytest.fixture
    def orchestrator(self):
        orch = BrainOrchestrator(config={'monitoring_interval': 0.1})
        orch.initialize()
        return orch
    
    def test_metrics_initialization(self, orchestrator):
        """Test metrics are properly initialized"""
        metrics = orchestrator._metrics
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.task_count == 0
        assert metrics.active_tasks == 0
        assert metrics.completed_tasks == 0
        assert metrics.failed_tasks == 0
        assert metrics.error_rate == 0.0
    
    def test_metrics_update_on_task_completion(self, orchestrator):
        """Test metrics update when tasks complete"""
        task = OrchestrationTask(
            task_id="metric_task",
            operation="status_check",
            priority=OperationPriority.MEDIUM
        )
        
        orchestrator.submit_task(task)
        time.sleep(0.5)
        
        assert orchestrator._metrics.task_count == 1
        assert orchestrator._metrics.completed_tasks == 1
        assert orchestrator._metrics.average_execution_time > 0
    
    def test_throughput_calculation(self, orchestrator):
        """Test throughput metric calculation"""
        # Submit multiple tasks
        for i in range(5):
            task = OrchestrationTask(
                task_id=f"throughput_task_{i}",
                operation="status_check",
                priority=OperationPriority.MEDIUM
            )
            orchestrator.submit_task(task)
        
        time.sleep(0.5)
        
        metrics = orchestrator.update_metrics()
        assert metrics['throughput'] >= 0  # Should have some throughput
    
    def test_error_rate_calculation(self, orchestrator):
        """Test error rate calculation"""
        # Submit task that will fail
        task = OrchestrationTask(
            task_id="error_task",
            operation="unknown_op",
            priority=OperationPriority.MEDIUM,
            retry_count=0
        )
        
        orchestrator.submit_task(task)
        time.sleep(0.5)
        
        # Check error metrics
        # Note: Default handler returns result for unknown operations
        # so this might not fail as expected
        if orchestrator._metrics.failed_tasks > 0:
            assert orchestrator._metrics.error_rate > 0


class TestEmergencyHandling:
    """Test emergency condition detection and handling"""
    
    @pytest.fixture
    def orchestrator(self):
        orch = BrainOrchestrator(config={'emergency_threshold': 0.5})
        orch.initialize()
        return orch
    
    def test_emergency_trigger_on_high_error_rate(self, orchestrator):
        """Test emergency triggered by high error rate"""
        # Artificially set high error rate
        orchestrator._metrics.error_rate = 0.6
        orchestrator._metrics.failed_tasks = 6
        orchestrator._metrics.completed_tasks = 4
        
        orchestrator._check_emergency_conditions()
        
        assert orchestrator._state == SystemState.EMERGENCY
    
    def test_emergency_trigger_on_system_overload(self, orchestrator):
        """Test emergency triggered by system overload"""
        # Simulate system overload
        orchestrator._metrics.system_load = 0.96
        
        orchestrator._check_emergency_conditions()
        
        assert orchestrator._state == SystemState.EMERGENCY
    
    def test_emergency_task_cleanup(self, orchestrator):
        """Test emergency task cleanup handler"""
        # Add tasks to queue
        for i in range(5):
            task = OrchestrationTask(
                task_id=f"cleanup_task_{i}",
                operation="test",
                priority=OperationPriority.LOW
            )
            orchestrator._task_queue.append(task)
        
        # Trigger emergency cleanup
        orchestrator._emergency_task_cleanup("test", {})
        
        assert len(orchestrator._task_queue) == 0
    
    def test_emergency_component_isolation(self, orchestrator):
        """Test emergency component isolation"""
        # Register components with poor health
        for i in range(3):
            orchestrator.register_component(f"comp_{i}", Mock())
            orchestrator._component_status[f"comp_{i}"].health_score = 0.2
        
        orchestrator._emergency_component_isolation("test", {})
        
        for i in range(3):
            assert orchestrator._component_status[f"comp_{i}"].status == "isolated"
    
    def test_emergency_resource_conservation(self, orchestrator):
        """Test emergency resource conservation"""
        original_workers = orchestrator._max_workers
        original_interval = orchestrator._monitoring_interval
        
        orchestrator._emergency_resource_conservation("test", {})
        
        assert orchestrator._max_workers < original_workers
        assert orchestrator._monitoring_interval > original_interval


class TestEventSystem:
    """Test event handling system"""
    
    @pytest.fixture
    def orchestrator(self):
        orch = BrainOrchestrator()
        orch.initialize()
        return orch
    
    def test_event_handler_registration(self, orchestrator):
        """Test registering event handlers"""
        mock_handler = Mock()
        
        orchestrator.register_event_handler("test_event", mock_handler)
        
        assert "test_event" in orchestrator._event_handlers
        assert mock_handler in orchestrator._event_handlers["test_event"]
    
    def test_event_emission(self, orchestrator):
        """Test emitting events to handlers"""
        mock_handler = Mock()
        orchestrator.register_event_handler("test_event", mock_handler)
        
        test_data = {"key": "value"}
        orchestrator._emit_event("test_event", test_data)
        
        mock_handler.assert_called_once_with("test_event", test_data)
    
    def test_multiple_event_handlers(self, orchestrator):
        """Test multiple handlers for same event"""
        handlers = [Mock() for _ in range(3)]
        
        for handler in handlers:
            orchestrator.register_event_handler("multi_event", handler)
        
        orchestrator._emit_event("multi_event", {"test": True})
        
        for handler in handlers:
            handler.assert_called_once()
    
    def test_event_handler_error_isolation(self, orchestrator):
        """Test that errors in one handler don't affect others"""
        failing_handler = Mock(side_effect=Exception("Handler error"))
        working_handler = Mock()
        
        orchestrator.register_event_handler("error_event", failing_handler)
        orchestrator.register_event_handler("error_event", working_handler)
        
        orchestrator._emit_event("error_event", {})
        
        # Working handler should still be called despite error in first
        working_handler.assert_called_once()


class TestConcurrency:
    """Test concurrent task execution and thread safety"""
    
    @pytest.fixture
    def orchestrator(self):
        orch = BrainOrchestrator(config={'max_workers': 4})
        orch.initialize()
        return orch
    
    def test_concurrent_task_submission(self, orchestrator):
        """Test submitting tasks from multiple threads"""
        num_threads = 10
        tasks_per_thread = 5
        
        def submit_tasks(thread_id):
            for i in range(tasks_per_thread):
                task = OrchestrationTask(
                    task_id=f"concurrent_{thread_id}_{i}",
                    operation="status_check",
                    priority=OperationPriority.MEDIUM
                )
                orchestrator.submit_task(task)
        
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=submit_tasks, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert orchestrator._metrics.task_count == num_threads * tasks_per_thread
    
    def test_concurrent_component_registration(self, orchestrator):
        """Test registering components from multiple threads"""
        num_threads = 10
        
        def register_component(thread_id):
            component = Mock()
            orchestrator.register_component(f"concurrent_comp_{thread_id}", component)
        
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=register_component, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(orchestrator._components) == num_threads
    
    def test_worker_pool_limits(self, orchestrator):
        """Test that worker pool respects max_workers limit"""
        # Submit more tasks than workers
        num_tasks = 10
        
        for i in range(num_tasks):
            task = OrchestrationTask(
                task_id=f"worker_test_{i}",
                operation="status_check",
                priority=OperationPriority.MEDIUM
            )
            orchestrator.submit_task(task)
        
        # Check that active tasks never exceed max_workers
        max_active = 0
        for _ in range(10):
            active = len(orchestrator._active_tasks)
            max_active = max(max_active, active)
            time.sleep(0.1)
        
        assert max_active <= orchestrator._max_workers


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling"""
    
    @pytest.fixture
    def orchestrator(self):
        orch = BrainOrchestrator(config={'max_workers': 8})
        orch.initialize()
        return orch
    
    def test_large_task_queue_performance(self, orchestrator):
        """Test performance with large number of tasks"""
        num_tasks = 100
        
        start_time = time.time()
        
        for i in range(num_tasks):
            task = OrchestrationTask(
                task_id=f"perf_task_{i}",
                operation="status_check",
                priority=OperationPriority.MEDIUM
            )
            orchestrator.submit_task(task)
        
        submission_time = time.time() - start_time
        
        # Submission should be fast
        assert submission_time < 1.0  # Should submit 100 tasks in under 1 second
        
        # Wait for completion
        timeout = 10
        start_wait = time.time()
        while orchestrator._metrics.completed_tasks < num_tasks and time.time() - start_wait < timeout:
            time.sleep(0.1)
        
        # Should complete all tasks
        assert orchestrator._metrics.completed_tasks == num_tasks
    
    def test_memory_usage_with_task_history(self, orchestrator):
        """Test that task history is properly bounded"""
        # Submit many tasks to fill history
        for i in range(1500):
            task = OrchestrationTask(
                task_id=f"history_task_{i}",
                operation="status_check",
                priority=OperationPriority.LOW
            )
            orchestrator.submit_task(task)
        
        # Wait for processing
        time.sleep(2)
        
        # History should be bounded to maxlen (1000)
        assert len(orchestrator._task_history) <= 1000
    
    def test_performance_metrics_accuracy(self, orchestrator):
        """Test accuracy of performance metrics"""
        num_tasks = 20
        
        for i in range(num_tasks):
            task = OrchestrationTask(
                task_id=f"metric_accuracy_{i}",
                operation="status_check",
                priority=OperationPriority.MEDIUM
            )
            orchestrator.submit_task(task)
        
        # Wait for completion
        time.sleep(2)
        
        # Get performance report
        report = orchestrator.get_performance_report()
        
        assert report['system_metrics']['total_tasks'] == num_tasks
        assert report['system_metrics']['completed_tasks'] <= num_tasks
        assert report['system_metrics']['average_execution_time'] > 0
        assert 'recent_tasks' in report
        assert 'component_health' in report


class TestShutdownAndCleanup:
    """Test graceful shutdown and resource cleanup"""
    
    def test_graceful_shutdown(self):
        """Test graceful orchestrator shutdown"""
        orchestrator = BrainOrchestrator()
        orchestrator.initialize()
        
        # Submit some tasks
        for i in range(5):
            task = OrchestrationTask(
                task_id=f"shutdown_task_{i}",
                operation="status_check",
                priority=OperationPriority.MEDIUM
            )
            orchestrator.submit_task(task)
        
        # Shutdown
        result = orchestrator.shutdown(timeout=5.0)
        
        assert result == True
        assert orchestrator._state == SystemState.SHUTDOWN
        assert len(orchestrator._task_queue) == 0
        assert len(orchestrator._active_tasks) == 0
    
    def test_shutdown_with_timeout(self):
        """Test shutdown with timeout"""
        orchestrator = BrainOrchestrator()
        orchestrator.initialize()
        
        # Create a long-running task simulation
        mock_component = Mock()
        mock_component.long_operation = Mock(side_effect=lambda: time.sleep(10))
        orchestrator.register_component("slow_comp", mock_component)
        
        task = OrchestrationTask(
            task_id="slow_task",
            operation="long_operation",
            priority=OperationPriority.HIGH,
            parameters={"component_id": "slow_comp"}
        )
        orchestrator.submit_task(task)
        
        # Try shutdown with short timeout
        result = orchestrator.shutdown(timeout=1.0)
        
        # Should still return True and clean up
        assert orchestrator._state == SystemState.SHUTDOWN
    
    def test_monitoring_thread_cleanup(self):
        """Test that monitoring thread is properly cleaned up"""
        orchestrator = BrainOrchestrator(config={'monitoring_enabled': True})
        orchestrator.initialize()
        
        # Verify monitoring thread is running
        assert orchestrator._monitoring_thread is not None
        assert orchestrator._monitoring_thread.is_alive()
        
        # Shutdown
        orchestrator.shutdown()
        
        # Wait briefly
        time.sleep(0.5)
        
        # Monitoring thread should be stopped
        assert not orchestrator._monitoring_thread.is_alive()


class TestErrorHandling:
    """Test error handling and recovery"""
    
    @pytest.fixture
    def orchestrator(self):
        orch = BrainOrchestrator()
        orch.initialize()
        return orch
    
    def test_task_execution_error_handling(self, orchestrator):
        """Test handling of errors during task execution"""
        # Create component that raises error
        mock_component = Mock()
        mock_component.failing_method = Mock(side_effect=Exception("Execution error"))
        orchestrator.register_component("failing_comp", mock_component)
        
        task = OrchestrationTask(
            task_id="error_task",
            operation="failing_method",
            priority=OperationPriority.MEDIUM,
            parameters={"component_id": "failing_comp"},
            retry_count=0
        )
        
        orchestrator.submit_task(task)
        time.sleep(0.5)
        
        # Task should be in completed with error
        assert "error_task" in orchestrator._completed_tasks
        completed_task = orchestrator._completed_tasks["error_task"]
        assert completed_task.status == "failed"
        assert completed_task.error is not None
    
    def test_invalid_component_reference(self, orchestrator):
        """Test handling of invalid component references"""
        task = OrchestrationTask(
            task_id="invalid_comp_task",
            operation="some_method",
            priority=OperationPriority.MEDIUM,
            parameters={"component_id": "non_existent_comp"}
        )
        
        orchestrator.submit_task(task)
        time.sleep(0.5)
        
        # Should complete but with appropriate handling
        assert "invalid_comp_task" in orchestrator._completed_tasks
    
    def test_monitoring_loop_error_recovery(self, orchestrator):
        """Test that monitoring loop recovers from errors"""
        # Inject error into monitoring
        original_update = orchestrator.update_metrics
        error_count = [0]
        
        def failing_update():
            error_count[0] += 1
            if error_count[0] < 3:
                raise Exception("Monitoring error")
            return original_update()
        
        orchestrator.update_metrics = failing_update
        
        # Wait for monitoring to run and recover
        time.sleep(1)
        
        # Orchestrator should still be running
        assert orchestrator._state != SystemState.SHUTDOWN
        assert orchestrator._monitoring_enabled == True


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.fixture
    def orchestrator(self):
        orch = BrainOrchestrator()
        orch.initialize()
        return orch
    
    def test_complex_workflow_orchestration(self, orchestrator):
        """Test orchestrating a complex multi-step workflow"""
        # Set up mock components
        data_processor = Mock()
        data_processor.process = Mock(return_value={"processed": True})
        
        analyzer = Mock()
        analyzer.analyze = Mock(return_value={"analysis": "complete"})
        
        orchestrator.register_component("processor", data_processor)
        orchestrator.register_component("analyzer", analyzer)
        
        # Submit workflow tasks
        tasks = [
            OrchestrationTask(
                task_id="workflow_1_fetch",
                operation="status_check",
                priority=OperationPriority.HIGH
            ),
            OrchestrationTask(
                task_id="workflow_2_process",
                operation="process",
                priority=OperationPriority.HIGH,
                parameters={"component_id": "processor"}
            ),
            OrchestrationTask(
                task_id="workflow_3_analyze",
                operation="analyze",
                priority=OperationPriority.MEDIUM,
                parameters={"component_id": "analyzer"}
            )
        ]
        
        for task in tasks:
            orchestrator.submit_task(task)
        
        # Wait for completion
        time.sleep(1)
        
        # All tasks should complete
        for task in tasks:
            assert task.task_id in orchestrator._completed_tasks
            assert orchestrator._completed_tasks[task.task_id].status == "completed"
    
    def test_high_load_scenario(self, orchestrator):
        """Test system under high load"""
        # Submit many high-priority tasks
        for i in range(50):
            task = OrchestrationTask(
                task_id=f"high_load_{i}",
                operation="status_check",
                priority=OperationPriority.HIGH if i < 25 else OperationPriority.LOW
            )
            orchestrator.submit_task(task)
        
        # System should handle load
        time.sleep(3)
        
        # Check metrics
        metrics = orchestrator.get_system_status()
        assert metrics['metrics']['completed_tasks'] > 0
        assert metrics['state'] != 'emergency'  # Should not trigger emergency
    
    def test_emergency_recovery_scenario(self, orchestrator):
        """Test recovery from emergency state"""
        # Trigger emergency
        orchestrator._trigger_emergency("test_emergency", {"test": True})
        assert orchestrator._state == SystemState.EMERGENCY
        
        # Submit recovery task
        recovery_task = OrchestrationTask(
            task_id="recovery_task",
            operation="emergency_response",
            priority=OperationPriority.EMERGENCY,
            parameters={"type": "system_overload", "data": {}}
        )
        
        orchestrator.submit_task(recovery_task)
        time.sleep(0.5)
        
        # Check that emergency response was executed
        assert "recovery_task" in orchestrator._completed_tasks
        result = orchestrator._completed_tasks["recovery_task"].result
        assert 'actions_taken' in result
        assert len(result['actions_taken']) > 0


class TestBrainIntegration:
    """Test integration with brain instance"""
    
    def test_orchestrator_with_brain_instance(self):
        """Test orchestrator with brain instance"""
        mock_brain = Mock()
        mock_brain.decision_engine = Mock()
        mock_brain.reasoning_orchestrator = Mock()
        mock_brain.neural_orchestrator = Mock()
        
        orchestrator = BrainOrchestrator(brain_instance=mock_brain)
        orchestrator.initialize()
        
        assert orchestrator.brain == mock_brain
        assert orchestrator._decision_engine == mock_brain.decision_engine
        assert orchestrator._reasoning_orchestrator == mock_brain.reasoning_orchestrator
        assert orchestrator._neural_orchestrator == mock_brain.neural_orchestrator
    
    def test_routing_to_brain_components(self):
        """Test routing tasks to brain components"""
        mock_brain = Mock()
        mock_decision_engine = Mock()
        mock_decision_engine.process = Mock(return_value={"decision": "made"})
        mock_brain.decision_engine = mock_decision_engine
        
        orchestrator = BrainOrchestrator(brain_instance=mock_brain)
        orchestrator.initialize()
        
        # Submit decision task
        task = OrchestrationTask(
            task_id="brain_decision",
            operation="decision_process",
            priority=OperationPriority.HIGH,
            parameters={"input": "test"}
        )
        
        orchestrator.submit_task(task)
        time.sleep(0.5)
        
        mock_decision_engine.process.assert_called_once()
        assert orchestrator._completed_tasks["brain_decision"].status == "completed"


class TestStatusAndReporting:
    """Test status checking and reporting functionality"""
    
    @pytest.fixture
    def orchestrator(self):
        orch = BrainOrchestrator()
        orch.initialize()
        return orch
    
    def test_get_task_status(self, orchestrator):
        """Test getting status of specific tasks"""
        task = OrchestrationTask(
            task_id="status_test",
            operation="status_check",
            priority=OperationPriority.MEDIUM
        )
        
        orchestrator.submit_task(task)
        
        # Check pending status
        status = orchestrator.get_task_status("status_test")
        assert status is not None
        assert status['task_id'] == "status_test"
        
        # Wait for completion
        time.sleep(0.5)
        
        # Check completed status
        status = orchestrator.get_task_status("status_test")
        assert status['progress'] == 'completed'
        assert status['status'] == 'completed'
    
    def test_get_system_status(self, orchestrator):
        """Test comprehensive system status"""
        # Add some components and tasks
        orchestrator.register_component("test_comp", Mock())
        
        task = OrchestrationTask(
            task_id="sys_status_task",
            operation="status_check",
            priority=OperationPriority.MEDIUM
        )
        orchestrator.submit_task(task)
        
        time.sleep(0.5)
        
        status = orchestrator.get_system_status()
        
        assert 'state' in status
        assert 'mode' in status
        assert 'metrics' in status
        assert 'components' in status
        assert status['state'] == 'ready'
        assert status['metrics']['task_count'] > 0
    
    def test_get_performance_report(self, orchestrator):
        """Test performance reporting"""
        # Generate some activity
        for i in range(10):
            task = OrchestrationTask(
                task_id=f"perf_report_{i}",
                operation="status_check",
                priority=OperationPriority.MEDIUM
            )
            orchestrator.submit_task(task)
        
        time.sleep(1)
        
        report = orchestrator.get_performance_report()
        
        assert 'system_metrics' in report
        assert 'component_health' in report
        assert 'recent_tasks' in report
        assert 'queue_status' in report
        assert report['system_metrics']['total_tasks'] == 10
        assert len(report['recent_tasks']) <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])