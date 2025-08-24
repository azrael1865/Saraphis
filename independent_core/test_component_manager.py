"""
Comprehensive test suite for ComponentManager
Tests all functionality with real instances and hard failure validation
NO MOCKS - Real component testing only
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List
import tempfile
import json

from production_web.component_manager import ComponentManager


class TestComponentManagerInitialization:
    """Test ComponentManager initialization and configuration"""
    
    def test_initialization_with_config(self):
        """Test proper initialization with configuration"""
        config = {
            'max_update_batch_size': 25,
            'update_debounce_ms': 50,
            'component_timeout_seconds': 15
        }
        
        manager = ComponentManager(config)
        
        assert manager.config == config
        assert manager.max_update_batch_size == 25
        assert manager.update_debounce_ms == 50
        assert manager.component_timeout_seconds == 15
        
        # Verify default registry is loaded
        assert 'chart' in manager.component_registry
        assert 'metric' in manager.component_registry
        assert 'alert' in manager.component_registry
        assert 'table' in manager.component_registry
        
        # Verify initialization structures
        assert isinstance(manager.component_states, dict)
        assert isinstance(manager.component_metrics, dict)
        assert len(manager.component_states) == 0
    
    def test_initialization_with_defaults(self):
        """Test initialization with default configuration"""
        manager = ComponentManager({})
        
        assert manager.max_update_batch_size == 50
        assert manager.update_debounce_ms == 100
        assert manager.component_timeout_seconds == 30
    
    def test_component_registry_structure(self):
        """Test that component registry has proper structure"""
        manager = ComponentManager({})
        
        for component_type, config in manager.component_registry.items():
            assert 'type' in config
            assert 'requires_data' in config
            assert 'update_frequency' in config
            assert 'supported_data_types' in config
            assert isinstance(config['supported_data_types'], list)


class TestComponentRegistration:
    """Test component registration functionality"""
    
    def test_register_valid_component(self):
        """Test registering a valid component"""
        manager = ComponentManager({})
        
        result = manager.register_component(
            component_id="test_chart_1",
            component_type="chart",
            configuration={
                'title': 'Test Chart',
                'chart_type': 'line',
                'height': '400px'
            }
        )
        
        assert result['success'] == True
        assert result['component_id'] == "test_chart_1"
        assert 'component_state' in result
        
        # Verify component was stored
        assert "test_chart_1" in manager.component_states
        component = manager.component_states["test_chart_1"]
        assert component['type'] == 'chart'
        assert component['state'] == 'ready'
        assert component['configuration']['title'] == 'Test Chart'
    
    def test_register_component_with_auto_id(self):
        """Test registering component with auto-generated ID"""
        manager = ComponentManager({})
        
        result = manager.register_component(
            component_id="",  # Empty ID should trigger auto-generation
            component_type="metric",
            configuration={'title': 'Auto ID Test'}
        )
        
        assert result['success'] == True
        assert result['component_id'].startswith('metric_')
        assert len(result['component_id']) > 10  # Should have timestamp and hash
    
    def test_register_invalid_component_type(self):
        """Test registering with invalid component type fails hard"""
        manager = ComponentManager({})
        
        with pytest.raises(ValueError) as exc_info:
            manager.register_component(
                component_id="test_invalid",
                component_type="nonexistent_type",
                configuration={}
            )
        
        assert "HARD FAILURE: Unknown component type" in str(exc_info.value)
        assert "nonexistent_type" in str(exc_info.value)
        assert "Valid types:" in str(exc_info.value)
    
    def test_register_component_with_dependencies(self):
        """Test registering component with dependencies"""
        manager = ComponentManager({})
        
        # Register base component first
        manager.register_component("base_component", "metric", {'title': 'Base'})
        
        # Register dependent component
        result = manager.register_component(
            component_id="dependent_component",
            component_type="chart",
            configuration={
                'title': 'Dependent Chart',
                'dependencies': ['base_component']
            }
        )
        
        assert result['success'] == True
        assert 'base_component' in manager.component_dependencies['dependent_component']
    
    def test_register_component_lifecycle_tracking(self):
        """Test that registration properly tracks lifecycle"""
        manager = ComponentManager({})
        
        result = manager.register_component("lifecycle_test", "alert", {'title': 'Lifecycle'})
        
        assert result['success'] == True
        assert "lifecycle_test" in manager.component_lifecycle
        
        lifecycle = manager.component_lifecycle["lifecycle_test"]
        assert 'created' in lifecycle
        assert 'state_changes' in lifecycle
        assert len(lifecycle['state_changes']) == 2  # 'initializing' and 'ready'
        
        # Check state changes
        states = [change[0] for change in lifecycle['state_changes']]
        assert 'initializing' in states
        assert 'ready' in states


class TestComponentUpdates:
    """Test component update functionality"""
    
    def test_update_single_component(self):
        """Test updating a single component with data"""
        manager = ComponentManager({})
        
        # Register component first
        manager.register_component("update_test", "metric", {
            'title': 'Update Test',
            'format': 'number'
        })
        
        # Update with data
        result = manager._update_single_component("update_test", {
            'value': 42,
            'previous_value': 35
        })
        
        assert result['success'] == True
        
        # Verify data was stored
        component = manager.component_states["update_test"]
        assert component['data']['value'] == 42
        assert component['last_updated'] is not None
        assert component['rendered_content'] is not None
        assert component['state'] == 'active'
    
    def test_update_multiple_components(self):
        """Test updating multiple components at once"""
        manager = ComponentManager({})
        
        components = {
            'metric_1': {
                'type': 'metric',
                'title': 'Metric 1'
            },
            'chart_1': {
                'type': 'chart',
                'title': 'Chart 1',
                'chart_type': 'bar'
            }
        }
        
        data = {
            'metric_1': {'value': 100},
            'chart_1': {
                'datasets': [{'data': [1, 2, 3]}],
                'labels': ['A', 'B', 'C']
            }
        }
        
        result = manager.update_components(components, data)
        
        assert 'metric_1' in result
        assert 'chart_1' in result
        
        # Verify both components were registered and updated
        assert 'metric_1' in manager.component_states
        assert 'chart_1' in manager.component_states
        assert manager.component_states['metric_1']['data']['value'] == 100
        assert manager.component_states['chart_1']['data']['datasets'][0]['data'] == [1, 2, 3]
    
    def test_update_nonexistent_component_fails(self):
        """Test updating non-existent component fails hard"""
        manager = ComponentManager({})
        
        with pytest.raises(ValueError) as exc_info:
            manager._update_single_component("nonexistent", {'value': 42})
        
        assert "HARD FAILURE: Component not found" in str(exc_info.value)
    
    def test_apply_component_update(self):
        """Test applying specific updates to a component"""
        manager = ComponentManager({})
        
        # Register component
        manager.register_component("apply_test", "gauge", {
            'title': 'Gauge Test',
            'min': 0,
            'max': 100
        })
        
        # Apply update
        manager.apply_component_update("apply_test", {
            'data': {'value': 75},
            'configuration': {'color': 'blue'},
            'state': 'active'
        })
        
        # Verify updates were applied
        component = manager.component_states["apply_test"]
        assert component['data']['value'] == 75
        assert component['configuration']['color'] == 'blue'
        assert component['state'] == 'active'
        assert component['last_updated'] is not None
    
    def test_apply_update_to_nonexistent_component_fails(self):
        """Test applying update to non-existent component fails hard"""
        manager = ComponentManager({})
        
        with pytest.raises(ValueError) as exc_info:
            manager.apply_component_update("nonexistent", {'data': {'value': 1}})
        
        assert "HARD FAILURE: Component not found" in str(exc_info.value)


class TestComponentStateManagement:
    """Test component state management"""
    
    def test_get_component_state(self):
        """Test retrieving component state"""
        manager = ComponentManager({})
        
        # Register and update component
        manager.register_component("state_test", "status_indicator", {'title': 'Status'})
        manager.apply_component_update("state_test", {'data': {'status': 'healthy'}})
        
        result = manager.get_component_state("state_test")
        
        assert result['success'] == True
        assert 'component_state' in result
        
        state = result['component_state']
        assert state['id'] == 'state_test'
        assert state['type'] == 'status_indicator'
        assert state['data']['status'] == 'healthy'
    
    def test_get_nonexistent_component_state_fails(self):
        """Test getting state of non-existent component fails hard"""
        manager = ComponentManager({})
        
        with pytest.raises(ValueError) as exc_info:
            manager.get_component_state("nonexistent")
        
        assert "HARD FAILURE: Component not found for state retrieval" in str(exc_info.value)
    
    def test_component_state_transitions(self):
        """Test component state transitions are tracked"""
        manager = ComponentManager({})
        
        # Register component (initializing -> ready)
        manager.register_component("transition_test", "table", {'title': 'Transitions'})
        
        # Update component (ready -> active)  
        manager._update_single_component("transition_test", {
            'columns': ['A', 'B'],
            'rows': [['1', '2']]
        })
        
        lifecycle = manager.component_lifecycle["transition_test"]
        state_changes = [change[0] for change in lifecycle['state_changes']]
        
        assert 'initializing' in state_changes
        assert 'ready' in state_changes
        assert 'active' in state_changes


class TestComponentMetrics:
    """Test component metrics collection and retrieval"""
    
    def test_get_single_component_metrics(self):
        """Test getting metrics for a single component"""
        manager = ComponentManager({})
        
        # Register and update component to generate metrics
        manager.register_component("metrics_test", "chart", {'title': 'Metrics'})
        manager._update_single_component("metrics_test", {
            'datasets': [{'data': [1, 2]}],
            'labels': ['X', 'Y']
        })
        
        result = manager.get_component_metrics("metrics_test")
        
        assert result['component_id'] == 'metrics_test'
        assert 'metrics' in result
        
        metrics = result['metrics']
        assert metrics['render_count'] == 1
        assert metrics['update_count'] == 1
        assert metrics['error_count'] == 0
        assert metrics['last_render'] is not None
        assert metrics['last_update'] is not None
    
    def test_get_all_component_metrics(self):
        """Test getting metrics for all components"""
        manager = ComponentManager({})
        
        # Register multiple components
        manager.register_component("metrics_1", "metric", {'title': 'M1'})
        manager.register_component("metrics_2", "alert", {'title': 'M2'})
        
        # Update them to generate metrics
        manager._update_single_component("metrics_1", {'value': 10})
        manager._update_single_component("metrics_2", {'alerts': []})
        
        result = manager.get_component_metrics()
        
        assert result['total_components'] == 2
        assert 'component_metrics' in result
        assert 'summary' in result
        
        assert 'metrics_1' in result['component_metrics']
        assert 'metrics_2' in result['component_metrics']
        
        summary = result['summary']
        assert summary['total_renders'] == 2
        assert summary['total_updates'] == 2
        assert summary['total_errors'] == 0
    
    def test_get_metrics_for_nonexistent_component_fails(self):
        """Test getting metrics for non-existent component fails hard"""
        manager = ComponentManager({})
        
        with pytest.raises(ValueError) as exc_info:
            manager.get_component_metrics("nonexistent")
        
        assert "HARD FAILURE: No metrics found for component" in str(exc_info.value)


class TestComponentRemoval:
    """Test component removal and cleanup"""
    
    def test_remove_component(self):
        """Test removing a component and cleanup"""
        manager = ComponentManager({})
        
        # Register component
        manager.register_component("remove_test", "heatmap", {'title': 'Remove Test'})
        
        # Verify it exists
        assert "remove_test" in manager.component_states
        assert "remove_test" in manager.component_lifecycle
        assert "remove_test" in manager.component_metrics
        
        # Remove it
        result = manager.remove_component("remove_test")
        
        assert result['success'] == True
        assert "remove_test" in result['message']
        
        # Verify cleanup
        assert "remove_test" not in manager.component_states
        assert "remove_test" not in manager.component_lifecycle  
        assert "remove_test" not in manager.component_metrics
    
    def test_remove_component_with_dependencies(self):
        """Test removing component cleans up dependencies"""
        manager = ComponentManager({})
        
        # Register components with dependencies
        manager.register_component("base", "metric", {'title': 'Base'})
        manager.register_component("dependent", "chart", {
            'title': 'Dependent',
            'dependencies': ['base']
        })
        
        # Verify dependency exists
        assert 'base' in manager.component_dependencies['dependent']
        
        # Remove base component
        result = manager.remove_component("base")
        assert result['success'] == True
        
        # Verify dependency was cleaned up
        assert 'base' not in manager.component_dependencies['dependent']
    
    def test_remove_nonexistent_component_fails(self):
        """Test removing non-existent component fails hard"""
        manager = ComponentManager({})
        
        with pytest.raises(ValueError) as exc_info:
            manager.remove_component("nonexistent")
        
        assert "HARD FAILURE: Component not found for removal" in str(exc_info.value)


class TestComponentRendering:
    """Test component rendering functionality"""
    
    def test_render_chart_component(self):
        """Test rendering chart component"""
        manager = ComponentManager({})
        
        manager.register_component("chart_render", "chart", {
            'title': 'Test Chart',
            'chart_type': 'bar',
            'height': '300px',
            'show_legend': True
        })
        
        manager._update_single_component("chart_render", {
            'datasets': [{'label': 'Data', 'data': [10, 20, 30]}],
            'labels': ['A', 'B', 'C']
        })
        
        component = manager.component_states["chart_render"]
        rendered = component['rendered_content']
        
        assert rendered['type'] == 'chart'
        assert rendered['chart_type'] == 'bar'
        assert rendered['data']['datasets'][0]['data'] == [10, 20, 30]
        assert rendered['data']['labels'] == ['A', 'B', 'C']
        assert rendered['options']['title'] == 'Test Chart'
        assert rendered['style']['height'] == '300px'
    
    def test_render_metric_component(self):
        """Test rendering metric component"""
        manager = ComponentManager({})
        
        manager.register_component("metric_render", "metric", {
            'title': 'CPU Usage',
            'unit': '%',
            'format': 'percentage'
        })
        
        manager._update_single_component("metric_render", {
            'value': 75.5,
            'previous_value': 65.0
        })
        
        component = manager.component_states["metric_render"]
        rendered = component['rendered_content']
        
        assert rendered['type'] == 'metric'
        assert rendered['title'] == 'CPU Usage'
        assert rendered['value'] == 75.5
        assert rendered['unit'] == '%'
        assert rendered['format'] == 'percentage'
        assert rendered['change'] is not None  # Should calculate change
        assert rendered['change_type'] == 'increase'
    
    def test_render_alert_component(self):
        """Test rendering alert component"""
        manager = ComponentManager({})
        
        manager.register_component("alert_render", "alert", {
            'max_visible': 3,
            'auto_dismiss': True,
            'position': 'top-left'
        })
        
        manager._update_single_component("alert_render", {
            'alerts': [
                {
                    'id': 'alert_1',
                    'severity': 'error',
                    'title': 'Critical Error',
                    'message': 'System failure detected',
                    'actions': ['acknowledge', 'dismiss']
                }
            ]
        })
        
        component = manager.component_states["alert_render"]
        rendered = component['rendered_content']
        
        assert rendered['type'] == 'alert'
        assert len(rendered['alerts']) == 1
        assert rendered['alerts'][0]['severity'] == 'error'
        assert rendered['alerts'][0]['title'] == 'Critical Error'
        assert rendered['max_visible'] == 3
        assert rendered['auto_dismiss'] == True
        assert rendered['style']['position'] == 'top-left'
    
    def test_render_gauge_component(self):
        """Test rendering gauge component"""
        manager = ComponentManager({})
        
        manager.register_component("gauge_render", "gauge", {
            'title': 'Memory Usage',
            'min': 0,
            'max': 100,
            'unit': 'GB',
            'thresholds': [
                {'value': 0, 'color': 'green'},
                {'value': 80, 'color': 'orange'},
                {'value': 95, 'color': 'red'}
            ]
        })
        
        manager._update_single_component("gauge_render", {'value': 85})
        
        component = manager.component_states["gauge_render"]
        rendered = component['rendered_content']
        
        assert rendered['type'] == 'gauge'
        assert rendered['title'] == 'Memory Usage'
        assert rendered['value'] == 85
        assert rendered['percentage'] == 85.0  # Should be 85% of 0-100 range
        assert rendered['min'] == 0
        assert rendered['max'] == 100
        assert rendered['unit'] == 'GB'
        assert len(rendered['thresholds']) == 3
    
    def test_render_status_indicator_component(self):
        """Test rendering status indicator component"""
        manager = ComponentManager({})
        
        manager.register_component("status_render", "status_indicator", {
            'title': 'System Status',
            'size': 'large',
            'show_label': True
        })
        
        manager._update_single_component("status_render", {
            'status': 'healthy',
            'message': 'All systems operational'
        })
        
        component = manager.component_states["status_render"]
        rendered = component['rendered_content']
        
        assert rendered['type'] == 'status_indicator'
        assert rendered['title'] == 'System Status'
        assert rendered['status'] == 'healthy'
        assert rendered['color'] == 'green'  # healthy maps to green
        assert rendered['message'] == 'All systems operational'
        assert rendered['style']['size'] == 'large'


class TestComponentDependencies:
    """Test component dependency management"""
    
    def test_cascading_updates(self):
        """Test that updates cascade to dependent components"""
        manager = ComponentManager({})
        
        # Register base component
        manager.register_component("base_data", "metric", {'title': 'Base Data'})
        
        # Register dependent components
        manager.register_component("chart_dependent", "chart", {
            'title': 'Dependent Chart',
            'dependencies': ['base_data']
        })
        
        manager.register_component("gauge_dependent", "gauge", {
            'title': 'Dependent Gauge',
            'dependencies': ['base_data'],
            'min': 0,
            'max': 100
        })
        
        # Update base component through interface update simulation
        interaction_result = {
            'affected_components': ['base_data'],
            'updates': {
                'base_data': {
                    'data': {'value': 42},
                    'requires_render': True
                }
            }
        }
        
        result = manager.update_interface_components("test_user", interaction_result)
        
        # Verify cascading updates occurred
        assert 'base_data' in result
        assert 'chart_dependent' in result
        assert 'gauge_dependent' in result
        
        # Verify dependent components were re-rendered
        chart_component = manager.component_states['chart_dependent']
        gauge_component = manager.component_states['gauge_dependent']
        
        assert chart_component['rendered_content'] is not None
        assert gauge_component['rendered_content'] is not None


class TestInterfaceComponentUpdates:
    """Test interface component update functionality"""
    
    def test_update_interface_components(self):
        """Test updating components through interface interactions"""
        manager = ComponentManager({})
        
        # Register components
        manager.register_component("ui_test_1", "control_panel", {'title': 'Controls'})
        manager.register_component("ui_test_2", "table", {'title': 'Data Table'})
        
        interaction_result = {
            'affected_components': ['ui_test_1', 'ui_test_2'],
            'updates': {
                'ui_test_1': {
                    'data': {'controls': [{'type': 'button', 'label': 'Submit'}]},
                    'configuration': {'background_color': 'blue'},
                    'requires_render': True
                },
                'ui_test_2': {
                    'data': {
                        'columns': ['Name', 'Value'],
                        'rows': [['Test', '123']]
                    },
                    'state': 'active',
                    'requires_render': True
                }
            }
        }
        
        result = manager.update_interface_components("user123", interaction_result)
        
        assert 'ui_test_1' in result
        assert 'ui_test_2' in result
        
        # Verify updates were applied
        component1 = manager.component_states['ui_test_1']
        component2 = manager.component_states['ui_test_2']
        
        assert component1['configuration']['background_color'] == 'blue'
        assert component1['data']['controls'][0]['label'] == 'Submit'
        
        assert component2['state'] == 'active'
        assert component2['data']['rows'][0] == ['Test', '123']
        
        # Verify rendered content exists
        assert component1['rendered_content'] is not None
        assert component2['rendered_content'] is not None


class TestComponentHealthAndMaintenance:
    """Test component health checking and maintenance"""
    
    def test_component_timeout_detection(self):
        """Test that stale components are detected"""
        manager = ComponentManager({
            'component_timeout_seconds': 0.1  # Very short for testing
        })
        
        # Register component and update it
        manager.register_component("timeout_test", "metric", {'title': 'Timeout'})
        manager._update_single_component("timeout_test", {'value': 100})
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Manually trigger health check (normally runs in background thread)
        current_time = time.time()
        component = manager.component_states["timeout_test"]
        
        if component['last_updated']:
            time_since_update = current_time - component['last_updated']
            if time_since_update > manager.component_timeout_seconds:
                manager._update_component_state("timeout_test", 'stale')
        
        # Verify component was marked as stale
        assert manager.component_states["timeout_test"]['state'] == 'stale'
    
    def test_component_lifecycle_cleanup(self):
        """Test that old lifecycle data gets cleaned up"""
        manager = ComponentManager({})
        
        manager.register_component("cleanup_test", "chart", {'title': 'Cleanup'})
        
        # Generate many state changes
        for i in range(150):  # More than the 100 limit
            manager._update_component_state("cleanup_test", f'state_{i}')
        
        lifecycle = manager.component_lifecycle["cleanup_test"]
        
        # Should keep only the most recent 100 state changes
        assert len(lifecycle['state_changes']) <= 100


class TestComponentManagerPerformance:
    """Test ComponentManager performance characteristics"""
    
    def test_concurrent_component_operations(self):
        """Test thread safety of component operations"""
        manager = ComponentManager({})
        
        def register_components(thread_id):
            for i in range(10):
                component_id = f"thread_{thread_id}_comp_{i}"
                manager.register_component(component_id, "metric", {
                    'title': f'Thread {thread_id} Component {i}'
                })
                manager._update_single_component(component_id, {'value': i * 10})
        
        threads = []
        for tid in range(5):
            thread = threading.Thread(target=register_components, args=(tid,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify all components were created
        assert len(manager.component_states) == 50
        
        # Verify no race conditions occurred
        for component_id, component in manager.component_states.items():
            assert component['state'] in ['ready', 'active']
            assert component['type'] == 'metric'
            assert 'rendered_content' in component
    
    def test_large_component_batch_update(self):
        """Test updating large batches of components"""
        manager = ComponentManager({})
        
        # Create large batch of components
        components = {}
        data = {}
        
        for i in range(100):
            comp_id = f"batch_{i}"
            components[comp_id] = {
                'type': 'metric',
                'title': f'Metric {i}'
            }
            data[comp_id] = {'value': i}
        
        # Update all at once
        start_time = time.time()
        result = manager.update_components(components, data)
        end_time = time.time()
        
        # Verify all were updated
        assert len(result) == 100
        assert len(manager.component_states) == 100
        
        # Should complete reasonably quickly
        update_time = end_time - start_time
        assert update_time < 5.0  # Should complete in under 5 seconds


class TestComponentManagerEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_configuration(self):
        """Test component registration with empty configuration"""
        manager = ComponentManager({})
        
        result = manager.register_component("empty_config", "metric", {})
        
        assert result['success'] == True
        assert manager.component_states["empty_config"]['configuration'] == {}
    
    def test_component_update_with_no_data(self):
        """Test updating component with no data"""
        manager = ComponentManager({})
        
        manager.register_component("no_data", "chart", {'title': 'No Data'})
        
        result = manager._update_single_component("no_data", {})
        
        assert result['success'] == True
        component = manager.component_states["no_data"]
        assert component['data'] == {}
        assert component['rendered_content'] is not None
    
    def test_invalid_component_data_fails_gracefully(self):
        """Test that invalid component data is handled properly"""
        manager = ComponentManager({})
        
        manager.register_component("invalid_data", "gauge", {
            'min': 0,
            'max': 100
        })
        
        # Update with string value instead of number - should still work
        result = manager._update_single_component("invalid_data", {'value': 'invalid'})
        
        assert result['success'] == True
        component = manager.component_states["invalid_data"]
        assert component['data']['value'] == 'invalid'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])