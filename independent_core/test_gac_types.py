"""
Comprehensive Unit Tests for GAC Types Module
Tests all data structures, enums, and type definitions in the GAC system
"""

import pytest
import time
import uuid
from enum import Enum
from dataclasses import dataclass, fields, asdict, is_dataclass
from typing import Dict, Any, Optional
import json
import pickle
from datetime import datetime

# Try to import torch, but handle if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create a mock torch for testing purposes
    class MockTensor:
        def __init__(self, *args, **kwargs):
            pass
        def shape(self):
            return [3, 3]
        def mean(self):
            class Item:
                def item(self):
                    return 0.5
            return Item()
    
    class MockTorch:
        Tensor = MockTensor
        def randn(self, *args):
            return MockTensor()
        def allclose(self, a, b):
            return True
    
    torch = MockTorch()

# Import the types we're testing
from gac_system.gac_types import (
    ComponentState,
    EventType,
    GACEvent,
    ComponentMetrics,
    PIDState,
    DirectionType,
    DirectionState
)


class TestComponentStateEnum:
    """Test ComponentState enum functionality"""
    
    def test_all_states_defined(self):
        """Test that all expected component states are defined"""
        expected_states = {
            'INACTIVE': 'inactive',
            'ACTIVE': 'active',
            'PAUSED': 'paused',
            'ERROR': 'error',
            'MAINTENANCE': 'maintenance'
        }
        
        for state_name, state_value in expected_states.items():
            assert hasattr(ComponentState, state_name)
            assert ComponentState[state_name].value == state_value
    
    def test_enum_values_unique(self):
        """Test that all enum values are unique"""
        values = [state.value for state in ComponentState]
        assert len(values) == len(set(values))
    
    def test_enum_comparison(self):
        """Test enum comparison operations"""
        assert ComponentState.ACTIVE == ComponentState.ACTIVE
        assert ComponentState.ACTIVE != ComponentState.INACTIVE
        assert ComponentState.ACTIVE is ComponentState.ACTIVE
    
    def test_enum_iteration(self):
        """Test that we can iterate over enum values"""
        states = list(ComponentState)
        assert len(states) == 5
        assert ComponentState.INACTIVE in states
    
    def test_enum_string_conversion(self):
        """Test string representation of enum values"""
        assert str(ComponentState.ACTIVE) == 'ComponentState.ACTIVE'
        assert ComponentState.ACTIVE.value == 'active'
        assert ComponentState.ACTIVE.name == 'ACTIVE'
    
    def test_enum_from_value(self):
        """Test creating enum from value"""
        state = ComponentState('active')
        assert state == ComponentState.ACTIVE
        
        with pytest.raises(ValueError):
            ComponentState('invalid_state')


class TestEventTypeEnum:
    """Test EventType enum functionality"""
    
    def test_all_event_types_defined(self):
        """Test that all expected event types are defined"""
        expected_events = {
            'GRADIENT_UPDATE': 'gradient_update',
            'THRESHOLD_EXCEEDED': 'threshold_exceeded',
            'COMPONENT_STATE_CHANGE': 'component_state_change',
            'SYSTEM_ALERT': 'system_alert',
            'LEARNING_UPDATE': 'learning_update',
            'PERFORMANCE_METRIC': 'performance_metric'
        }
        
        for event_name, event_value in expected_events.items():
            assert hasattr(EventType, event_name)
            assert EventType[event_name].value == event_value
    
    def test_event_type_uniqueness(self):
        """Test that all event type values are unique"""
        values = [event.value for event in EventType]
        assert len(values) == len(set(values))
    
    def test_event_type_membership(self):
        """Test membership checking for event types"""
        # Test that enum members are in the enum
        assert EventType.GRADIENT_UPDATE in EventType
        
        # Test that we can't accidentally use strings where enums are expected
        # This is what we really want to verify - type safety
        assert EventType.GRADIENT_UPDATE != 'gradient_update'
        assert type(EventType.GRADIENT_UPDATE) != str
    
    def test_event_type_hash(self):
        """Test that event types are hashable"""
        event_set = {EventType.GRADIENT_UPDATE, EventType.SYSTEM_ALERT}
        assert len(event_set) == 2
        assert EventType.GRADIENT_UPDATE in event_set


class TestDirectionTypeEnum:
    """Test DirectionType enum functionality"""
    
    def test_all_directions_defined(self):
        """Test that all expected direction types are defined"""
        expected_directions = {
            'ASCENT': 'ascent',
            'DESCENT': 'descent',
            'STABLE': 'stable',
            'OSCILLATING': 'oscillating'
        }
        
        for dir_name, dir_value in expected_directions.items():
            assert hasattr(DirectionType, dir_name)
            assert DirectionType[dir_name].value == dir_value
    
    def test_direction_type_ordering(self):
        """Test that direction types maintain order"""
        directions = list(DirectionType)
        assert directions[0] == DirectionType.ASCENT
        assert directions[1] == DirectionType.DESCENT
        assert directions[2] == DirectionType.STABLE
        assert directions[3] == DirectionType.OSCILLATING
    
    def test_direction_type_conversion(self):
        """Test conversion between name and value"""
        assert DirectionType['ASCENT'] == DirectionType.ASCENT
        assert DirectionType('ascent') == DirectionType.ASCENT
        
        with pytest.raises(KeyError):
            DirectionType['INVALID']
        
        with pytest.raises(ValueError):
            DirectionType('invalid')


class TestGACEvent:
    """Test GACEvent dataclass functionality"""
    
    def test_event_creation_minimal(self):
        """Test creating event with minimal required fields"""
        event = GACEvent(
            event_type=EventType.GRADIENT_UPDATE,
            source_component='test_component',
            data={'value': 42}
        )
        
        assert event.event_type == EventType.GRADIENT_UPDATE
        assert event.source_component == 'test_component'
        assert event.data == {'value': 42}
        assert isinstance(event.timestamp, float)
        assert isinstance(event.event_id, str)
    
    def test_event_creation_full(self):
        """Test creating event with all fields specified"""
        custom_time = time.time()
        custom_id = 'custom-event-id'
        
        event = GACEvent(
            event_type=EventType.SYSTEM_ALERT,
            source_component='alert_system',
            data={'alert': 'test'},
            timestamp=custom_time,
            event_id=custom_id
        )
        
        assert event.timestamp == custom_time
        assert event.event_id == custom_id
    
    def test_event_default_values(self):
        """Test that default values are properly generated"""
        event1 = GACEvent(
            event_type=EventType.GRADIENT_UPDATE,
            source_component='comp1',
            data={}
        )
        time.sleep(0.01)  # Ensure different timestamps
        event2 = GACEvent(
            event_type=EventType.GRADIENT_UPDATE,
            source_component='comp2',
            data={}
        )
        
        # Timestamps should be different
        assert event1.timestamp != event2.timestamp
        
        # Event IDs should be unique
        assert event1.event_id != event2.event_id
        
        # Both should be valid UUIDs (when default factory is used)
        try:
            uuid.UUID(event1.event_id)
            uuid.UUID(event2.event_id)
        except ValueError:
            pytest.fail("Default event_id should be valid UUID")
    
    def test_event_dataclass_features(self):
        """Test dataclass features like field access and representation"""
        event = GACEvent(
            event_type=EventType.LEARNING_UPDATE,
            source_component='learner',
            data={'learning_rate': 0.01}
        )
        
        # Test field access
        assert hasattr(event, 'event_type')
        assert hasattr(event, 'source_component')
        assert hasattr(event, 'data')
        assert hasattr(event, 'timestamp')
        assert hasattr(event, 'event_id')
        
        # Test that it's a dataclass
        assert is_dataclass(event)
        
        # Test fields() function
        field_names = [f.name for f in fields(event)]
        assert 'event_type' in field_names
        assert 'source_component' in field_names
        assert 'data' in field_names
        assert 'timestamp' in field_names
        assert 'event_id' in field_names
    
    def test_event_equality(self):
        """Test event equality comparison"""
        data = {'test': 'data'}
        event1 = GACEvent(
            event_type=EventType.GRADIENT_UPDATE,
            source_component='comp',
            data=data,
            timestamp=100.0,
            event_id='same-id'
        )
        event2 = GACEvent(
            event_type=EventType.GRADIENT_UPDATE,
            source_component='comp',
            data=data,
            timestamp=100.0,
            event_id='same-id'
        )
        event3 = GACEvent(
            event_type=EventType.GRADIENT_UPDATE,
            source_component='comp',
            data=data,
            timestamp=100.0,
            event_id='different-id'
        )
        
        assert event1 == event2
        assert event1 != event3
    
    def test_event_dict_conversion(self):
        """Test converting event to dictionary"""
        event = GACEvent(
            event_type=EventType.THRESHOLD_EXCEEDED,
            source_component='threshold_monitor',
            data={'threshold': 0.9, 'actual': 0.95}
        )
        
        event_dict = asdict(event)
        
        assert isinstance(event_dict, dict)
        assert event_dict['source_component'] == 'threshold_monitor'
        assert event_dict['data']['threshold'] == 0.9
        assert 'timestamp' in event_dict
        assert 'event_id' in event_dict


class TestComponentMetrics:
    """Test ComponentMetrics dataclass functionality"""
    
    def test_metrics_default_values(self):
        """Test that metrics initialize with correct default values"""
        metrics = ComponentMetrics()
        
        assert metrics.activation_count == 0
        assert metrics.total_processing_time == 0.0
        assert metrics.error_count == 0
        assert metrics.last_activation is None
        assert metrics.success_rate == 1.0
        assert metrics.performance_score == 0.5
    
    def test_metrics_custom_values(self):
        """Test creating metrics with custom values"""
        metrics = ComponentMetrics(
            activation_count=10,
            total_processing_time=5.5,
            error_count=2,
            last_activation=1000.0,
            success_rate=0.8,
            performance_score=0.75
        )
        
        assert metrics.activation_count == 10
        assert metrics.total_processing_time == 5.5
        assert metrics.error_count == 2
        assert metrics.last_activation == 1000.0
        assert metrics.success_rate == 0.8
        assert metrics.performance_score == 0.75
    
    def test_metrics_mutation(self):
        """Test that metrics fields can be modified"""
        metrics = ComponentMetrics()
        
        metrics.activation_count += 1
        assert metrics.activation_count == 1
        
        metrics.total_processing_time += 0.5
        assert metrics.total_processing_time == 0.5
        
        metrics.error_count = 5
        assert metrics.error_count == 5
        
        metrics.last_activation = time.time()
        assert metrics.last_activation is not None
    
    def test_metrics_field_types(self):
        """Test that fields have correct types"""
        metrics = ComponentMetrics()
        
        assert isinstance(metrics.activation_count, int)
        assert isinstance(metrics.total_processing_time, float)
        assert isinstance(metrics.error_count, int)
        assert metrics.last_activation is None or isinstance(metrics.last_activation, float)
        assert isinstance(metrics.success_rate, float)
        assert isinstance(metrics.performance_score, float)
    
    def test_metrics_calculated_properties(self):
        """Test calculated properties based on metrics"""
        metrics = ComponentMetrics(
            activation_count=100,
            error_count=10,
            total_processing_time=50.0
        )
        
        # Calculate average processing time
        if metrics.activation_count > 0:
            avg_time = metrics.total_processing_time / metrics.activation_count
            assert avg_time == 0.5
        
        # Calculate error rate
        if metrics.activation_count > 0:
            error_rate = metrics.error_count / metrics.activation_count
            assert error_rate == 0.1


class TestPIDState:
    """Test PIDState dataclass functionality"""
    
    def test_pid_default_values(self):
        """Test PID state initializes with correct default values"""
        pid = PIDState()
        
        assert pid.kp == 1.0
        assert pid.ki == 0.1
        assert pid.kd == 0.05
        assert pid.integral == 0.0
        assert pid.previous_error == 0.0
        assert pid.setpoint == 0.0
    
    def test_pid_custom_values(self):
        """Test creating PID state with custom values"""
        pid = PIDState(
            kp=2.0,
            ki=0.5,
            kd=0.1,
            integral=10.0,
            previous_error=0.5,
            setpoint=1.0
        )
        
        assert pid.kp == 2.0
        assert pid.ki == 0.5
        assert pid.kd == 0.1
        assert pid.integral == 10.0
        assert pid.previous_error == 0.5
        assert pid.setpoint == 1.0
    
    def test_pid_gains_validation(self):
        """Test that PID gains can be any float value"""
        # Negative gains (sometimes used for reverse-acting controllers)
        pid = PIDState(kp=-1.0, ki=-0.1, kd=-0.05)
        assert pid.kp == -1.0
        assert pid.ki == -0.1
        assert pid.kd == -0.05
        
        # Zero gains (used to disable terms)
        pid = PIDState(kp=0.0, ki=0.0, kd=0.0)
        assert pid.kp == 0.0
        assert pid.ki == 0.0
        assert pid.kd == 0.0
        
        # Very large gains
        pid = PIDState(kp=1000.0, ki=100.0, kd=50.0)
        assert pid.kp == 1000.0
        assert pid.ki == 100.0
        assert pid.kd == 50.0
    
    def test_pid_state_update(self):
        """Test updating PID state during control loop"""
        pid = PIDState()
        
        # Simulate PID update
        error = 1.0 - 0.0  # setpoint - current
        pid.integral += error * 0.01  # dt = 0.01
        derivative = (error - pid.previous_error) / 0.01
        pid.previous_error = error
        
        assert pid.integral > 0
        assert pid.previous_error == error
    
    def test_pid_reset(self):
        """Test resetting PID state"""
        pid = PIDState(integral=10.0, previous_error=5.0)
        
        # Reset integral windup
        pid.integral = 0.0
        pid.previous_error = 0.0
        
        assert pid.integral == 0.0
        assert pid.previous_error == 0.0


class TestDirectionState:
    """Test DirectionState dataclass functionality"""
    
    def test_direction_state_creation(self):
        """Test creating direction state with all fields"""
        metadata = {'gradient_norm': 0.5, 'step': 100}
        state = DirectionState(
            direction=DirectionType.ASCENT,
            confidence=0.95,
            magnitude=1.5,
            timestamp=1000.0,
            metadata=metadata
        )
        
        assert state.direction == DirectionType.ASCENT
        assert state.confidence == 0.95
        assert state.magnitude == 1.5
        assert state.timestamp == 1000.0
        assert state.metadata == metadata
    
    def test_direction_state_confidence_bounds(self):
        """Test confidence value boundaries"""
        # Confidence should typically be between 0 and 1
        state1 = DirectionState(
            direction=DirectionType.STABLE,
            confidence=0.0,
            magnitude=0.0,
            timestamp=0.0,
            metadata={}
        )
        assert state1.confidence == 0.0
        
        state2 = DirectionState(
            direction=DirectionType.STABLE,
            confidence=1.0,
            magnitude=0.0,
            timestamp=0.0,
            metadata={}
        )
        assert state2.confidence == 1.0
        
        # But it can technically be outside these bounds
        state3 = DirectionState(
            direction=DirectionType.STABLE,
            confidence=1.5,  # Over 1
            magnitude=0.0,
            timestamp=0.0,
            metadata={}
        )
        assert state3.confidence == 1.5
    
    def test_direction_state_metadata_operations(self):
        """Test metadata dictionary operations"""
        state = DirectionState(
            direction=DirectionType.OSCILLATING,
            confidence=0.7,
            magnitude=0.8,
            timestamp=time.time(),
            metadata={}
        )
        
        # Add metadata
        state.metadata['oscillation_frequency'] = 2.5
        state.metadata['oscillation_amplitude'] = 0.3
        
        assert 'oscillation_frequency' in state.metadata
        assert state.metadata['oscillation_frequency'] == 2.5
        
        # Update metadata
        state.metadata.update({'damping': 0.1})
        assert 'damping' in state.metadata
    
    def test_direction_state_equality(self):
        """Test equality comparison of direction states"""
        metadata = {'test': 'data'}
        state1 = DirectionState(
            direction=DirectionType.DESCENT,
            confidence=0.8,
            magnitude=1.0,
            timestamp=500.0,
            metadata=metadata
        )
        state2 = DirectionState(
            direction=DirectionType.DESCENT,
            confidence=0.8,
            magnitude=1.0,
            timestamp=500.0,
            metadata=metadata
        )
        state3 = DirectionState(
            direction=DirectionType.ASCENT,  # Different direction
            confidence=0.8,
            magnitude=1.0,
            timestamp=500.0,
            metadata=metadata
        )
        
        assert state1 == state2
        assert state1 != state3
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_direction_state_with_tensor_metadata(self):
        """Test storing tensor data in metadata"""
        tensor_data = torch.randn(3, 3)
        state = DirectionState(
            direction=DirectionType.ASCENT,
            confidence=0.9,
            magnitude=2.0,
            timestamp=time.time(),
            metadata={
                'gradient_tensor': tensor_data,
                'tensor_shape': list(tensor_data.shape) if hasattr(tensor_data, 'shape') else [3, 3],
                'tensor_mean': tensor_data.mean().item() if hasattr(tensor_data.mean(), 'item') else 0.5
            }
        )
        
        if TORCH_AVAILABLE:
            assert isinstance(state.metadata['gradient_tensor'], torch.Tensor)
        assert state.metadata['tensor_shape'] == [3, 3]
        assert isinstance(state.metadata['tensor_mean'], float)


class TestTypeSerialization:
    """Test serialization/deserialization of GAC types"""
    
    def test_event_json_serialization(self):
        """Test JSON serialization of GACEvent"""
        event = GACEvent(
            event_type=EventType.GRADIENT_UPDATE,
            source_component='test',
            data={'value': 123, 'flag': True}
        )
        
        # Convert to dict (handling enum)
        event_dict = asdict(event)
        event_dict['event_type'] = event_dict['event_type'].value
        
        # Should be JSON serializable
        json_str = json.dumps(event_dict)
        assert isinstance(json_str, str)
        
        # Should be able to reconstruct
        loaded_dict = json.loads(json_str)
        assert loaded_dict['source_component'] == 'test'
        assert loaded_dict['data']['value'] == 123
    
    def test_metrics_pickle_serialization(self):
        """Test pickle serialization of ComponentMetrics"""
        metrics = ComponentMetrics(
            activation_count=50,
            total_processing_time=25.5,
            error_count=3,
            last_activation=1000.0,
            success_rate=0.94,
            performance_score=0.85
        )
        
        # Serialize
        pickled = pickle.dumps(metrics)
        
        # Deserialize
        unpickled = pickle.loads(pickled)
        
        assert unpickled.activation_count == metrics.activation_count
        assert unpickled.total_processing_time == metrics.total_processing_time
        assert unpickled.error_count == metrics.error_count
        assert unpickled.last_activation == metrics.last_activation
        assert unpickled.success_rate == metrics.success_rate
        assert unpickled.performance_score == metrics.performance_score
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_direction_state_pickle_with_tensor(self):
        """Test pickle serialization of DirectionState with tensor metadata"""
        tensor = torch.randn(5, 5)
        state = DirectionState(
            direction=DirectionType.ASCENT,
            confidence=0.85,
            magnitude=1.2,
            timestamp=2000.0,
            metadata={'tensor': tensor, 'scalar': 42}
        )
        
        # Serialize
        pickled = pickle.dumps(state)
        
        # Deserialize
        unpickled = pickle.loads(pickled)
        
        assert unpickled.direction == state.direction
        assert unpickled.confidence == state.confidence
        if TORCH_AVAILABLE:
            assert torch.allclose(unpickled.metadata['tensor'], tensor)
        assert unpickled.metadata['scalar'] == 42


class TestTypeValidation:
    """Test type validation and error handling"""
    
    def test_invalid_enum_values(self):
        """Test handling of invalid enum values"""
        with pytest.raises(ValueError):
            ComponentState('invalid_state')
        
        with pytest.raises(ValueError):
            EventType('invalid_event')
        
        with pytest.raises(ValueError):
            DirectionType('invalid_direction')
    
    def test_type_hints_validation(self):
        """Test that type hints are properly defined"""
        # Check GACEvent type hints
        event_annotations = GACEvent.__annotations__
        assert event_annotations['event_type'] == EventType
        assert event_annotations['source_component'] == str
        assert event_annotations['data'] == Dict[str, Any]
        
        # Check DirectionState type hints  
        direction_annotations = DirectionState.__annotations__
        assert direction_annotations['direction'] == DirectionType
        assert direction_annotations['confidence'] == float
        assert direction_annotations['magnitude'] == float
        assert direction_annotations['timestamp'] == float
        assert direction_annotations['metadata'] == Dict[str, Any]
    
    def test_dataclass_immutability(self):
        """Test that dataclasses are mutable by default (not frozen)"""
        event = GACEvent(
            event_type=EventType.GRADIENT_UPDATE,
            source_component='test',
            data={}
        )
        
        # Should be able to modify
        event.source_component = 'modified'
        assert event.source_component == 'modified'
        
        event.data['new_key'] = 'new_value'
        assert 'new_key' in event.data


class TestIntegrationScenarios:
    """Test realistic usage scenarios of GAC types"""
    
    def test_event_flow_simulation(self):
        """Simulate a sequence of events in the GAC system"""
        events = []
        
        # Component activation
        events.append(GACEvent(
            event_type=EventType.COMPONENT_STATE_CHANGE,
            source_component='gac_controller',
            data={'old_state': ComponentState.INACTIVE.value, 
                  'new_state': ComponentState.ACTIVE.value}
        ))
        
        # Gradient updates
        for i in range(5):
            events.append(GACEvent(
                event_type=EventType.GRADIENT_UPDATE,
                source_component='gradient_processor',
                data={'step': i, 'gradient_norm': 0.1 * i}
            ))
        
        # Threshold exceeded
        events.append(GACEvent(
            event_type=EventType.THRESHOLD_EXCEEDED,
            source_component='threshold_monitor',
            data={'threshold': 0.3, 'actual': 0.4}
        ))
        
        # System alert
        events.append(GACEvent(
            event_type=EventType.SYSTEM_ALERT,
            source_component='alert_system',
            data={'alert': 'High gradient detected', 'severity': 'warning'}
        ))
        
        assert len(events) == 8
        assert all(isinstance(e, GACEvent) for e in events)
        assert events[0].event_type == EventType.COMPONENT_STATE_CHANGE
        assert events[-1].event_type == EventType.SYSTEM_ALERT
    
    def test_metrics_tracking_scenario(self):
        """Simulate metrics tracking over time"""
        metrics = ComponentMetrics()
        
        # Simulate component activations
        for i in range(10):
            start_time = time.time()
            
            # Simulate processing
            metrics.activation_count += 1
            time.sleep(0.001)  # Simulate work
            
            # Update metrics
            processing_time = time.time() - start_time
            metrics.total_processing_time += processing_time
            metrics.last_activation = time.time()
            
            # Simulate occasional errors
            if i % 4 == 0:
                metrics.error_count += 1
        
        # Calculate final metrics
        metrics.success_rate = (metrics.activation_count - metrics.error_count) / metrics.activation_count
        metrics.performance_score = metrics.success_rate * (1.0 / (metrics.total_processing_time / metrics.activation_count))
        
        assert metrics.activation_count == 10
        assert metrics.error_count == 3
        assert metrics.success_rate == 0.7
        assert metrics.last_activation is not None
        assert metrics.total_processing_time > 0
    
    def test_pid_control_loop_scenario(self):
        """Simulate a PID control loop scenario"""
        pid = PIDState(setpoint=1.0)
        current_value = 0.0
        dt = 0.01
        history = []
        
        for _ in range(10):
            # Calculate error
            error = pid.setpoint - current_value
            
            # Update integral
            pid.integral += error * dt
            
            # Calculate derivative
            derivative = (error - pid.previous_error) / dt if pid.previous_error != 0 else 0
            
            # Calculate control output
            output = pid.kp * error + pid.ki * pid.integral + pid.kd * derivative
            
            # Update state
            pid.previous_error = error
            
            # Simulate system response
            current_value += output * 0.1
            
            history.append({
                'error': error,
                'output': output,
                'current': current_value
            })
        
        # System should converge toward setpoint
        assert abs(current_value - pid.setpoint) < abs(history[0]['current'] - pid.setpoint)
        assert len(history) == 10
    
    def test_direction_state_transitions(self):
        """Test direction state transitions over time"""
        states = []
        
        # Start stable
        states.append(DirectionState(
            direction=DirectionType.STABLE,
            confidence=0.9,
            magnitude=0.01,
            timestamp=0.0,
            metadata={'gradient_norm': 0.01}
        ))
        
        # Transition to ascent
        states.append(DirectionState(
            direction=DirectionType.ASCENT,
            confidence=0.8,
            magnitude=0.5,
            timestamp=1.0,
            metadata={'gradient_norm': 0.5, 'direction_change': True}
        ))
        
        # Continue ascent with higher confidence
        states.append(DirectionState(
            direction=DirectionType.ASCENT,
            confidence=0.95,
            magnitude=0.8,
            timestamp=2.0,
            metadata={'gradient_norm': 0.8, 'direction_change': False}
        ))
        
        # Oscillation detected
        states.append(DirectionState(
            direction=DirectionType.OSCILLATING,
            confidence=0.7,
            magnitude=0.6,
            timestamp=3.0,
            metadata={'gradient_norm': 0.6, 'oscillation_count': 3}
        ))
        
        # Return to descent
        states.append(DirectionState(
            direction=DirectionType.DESCENT,
            confidence=0.85,
            magnitude=0.4,
            timestamp=4.0,
            metadata={'gradient_norm': 0.4}
        ))
        
        # Verify state progression
        assert states[0].direction == DirectionType.STABLE
        assert states[1].direction == DirectionType.ASCENT
        assert states[3].direction == DirectionType.OSCILLATING
        assert states[4].direction == DirectionType.DESCENT
        
        # Check confidence changes
        confidence_values = [s.confidence for s in states]
        assert min(confidence_values) >= 0.7
        assert max(confidence_values) <= 0.95


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_metadata(self):
        """Test handling of empty metadata"""
        state = DirectionState(
            direction=DirectionType.STABLE,
            confidence=0.5,
            magnitude=0.0,
            timestamp=0.0,
            metadata={}
        )
        
        assert state.metadata == {}
        assert len(state.metadata) == 0
    
    def test_large_metadata(self):
        """Test handling of large metadata dictionaries"""
        large_metadata = {f'key_{i}': f'value_{i}' for i in range(1000)}
        
        event = GACEvent(
            event_type=EventType.PERFORMANCE_METRIC,
            source_component='performance_monitor',
            data=large_metadata
        )
        
        assert len(event.data) == 1000
        assert 'key_500' in event.data
    
    def test_extreme_numeric_values(self):
        """Test handling of extreme numeric values"""
        # Very small values
        pid = PIDState(
            kp=1e-10,
            ki=1e-12,
            kd=1e-11
        )
        assert pid.kp == 1e-10
        
        # Very large values
        metrics = ComponentMetrics(
            activation_count=1_000_000,
            total_processing_time=1e6,
            error_count=100_000
        )
        assert metrics.activation_count == 1_000_000
        
        # Infinity
        state = DirectionState(
            direction=DirectionType.ASCENT,
            confidence=1.0,
            magnitude=float('inf'),
            timestamp=0.0,
            metadata={}
        )
        assert state.magnitude == float('inf')
    
    def test_none_values_in_metadata(self):
        """Test handling of None values in metadata"""
        event = GACEvent(
            event_type=EventType.LEARNING_UPDATE,
            source_component='learner',
            data={'result': None, 'status': 'pending'}
        )
        
        assert event.data['result'] is None
        assert event.data['status'] == 'pending'
    
    def test_circular_reference_in_metadata(self):
        """Test handling of circular references in metadata"""
        metadata = {'key': 'value'}
        metadata['self'] = metadata  # Circular reference
        
        # This should not cause issues during creation
        state = DirectionState(
            direction=DirectionType.STABLE,
            confidence=0.5,
            magnitude=0.0,
            timestamp=0.0,
            metadata=metadata
        )
        
        assert state.metadata is metadata
        assert state.metadata['self'] is metadata


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])