"""
Gradient Ascent Clipping (GAC) System Implementation
Zero-oversight autonomous gradient management with PID controllers, meta-learning, and RL components
"""

import asyncio
import logging
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import pickle

# Import Direction Components
try:
    from .direction_state import DirectionStateManager, DirectionHistory
    from .direction_validator import DirectionValidator, ValidationResult, AnomalyDetection
    from .basic_bounder import BasicGradientBounder, BoundingResult
    from .enhanced_bounder import EnhancedGradientBounder, EnhancedBoundingResult
    DIRECTION_COMPONENTS_AVAILABLE = True
except ImportError as e:
    DIRECTION_COMPONENTS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Log the import status after logger is defined
if not DIRECTION_COMPONENTS_AVAILABLE:
    logger.warning("Direction components not available")

class ComponentState(Enum):
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class EventType(Enum):
    GRADIENT_UPDATE = "gradient_update"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    COMPONENT_STATE_CHANGE = "component_state_change"
    SYSTEM_ALERT = "system_alert"
    LEARNING_UPDATE = "learning_update"
    PERFORMANCE_METRIC = "performance_metric"

@dataclass
class GACEvent:
    event_type: EventType
    source_component: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass
class ComponentMetrics:
    activation_count: int = 0
    total_processing_time: float = 0.0
    error_count: int = 0
    last_activation: Optional[float] = None
    success_rate: float = 1.0
    performance_score: float = 0.5

@dataclass
class PIDState:
    kp: float = 1.0
    ki: float = 0.1
    kd: float = 0.05
    integral: float = 0.0
    previous_error: float = 0.0
    setpoint: float = 0.0

class GACComponent(ABC):
    def __init__(self, component_id: str, config: Dict[str, Any] = None):
        self.component_id = component_id
        self.config = config or {}
        self.state = ComponentState.INACTIVE
        self.metrics = ComponentMetrics()
        self.event_handlers = {}
        self.last_gradient = None
        self.pid_state = PIDState()
        
    @abstractmethod
    async def process_gradient(self, gradient: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_component_info(self) -> Dict[str, Any]:
        pass
    
    def register_event_handler(self, event_type: EventType, handler: Callable):
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def emit_event(self, event_type: EventType, data: Dict[str, Any]):
        event = GACEvent(event_type, self.component_id, data)
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error in {self.component_id}: {e}")
    
    def update_state(self, new_state: ComponentState):
        old_state = self.state
        self.state = new_state
        self.emit_event(EventType.COMPONENT_STATE_CHANGE, {
            "old_state": old_state.value,
            "new_state": new_state.value,
            "timestamp": time.time()
        })
    
    def calculate_pid_output(self, current_value: float, dt: float = 1.0) -> float:
        error = self.pid_state.setpoint - current_value
        
        self.pid_state.integral += error * dt
        derivative = (error - self.pid_state.previous_error) / dt
        
        output = (self.pid_state.kp * error + 
                 self.pid_state.ki * self.pid_state.integral + 
                 self.pid_state.kd * derivative)
        
        self.pid_state.previous_error = error
        return output

class EventBus:
    def __init__(self):
        self.subscribers = defaultdict(list)
        self.event_history = deque(maxlen=10000)
        self.processing_stats = {}
        self._lock = threading.Lock()
    
    def subscribe(self, event_type: EventType, callback: Callable[[GACEvent], None]):
        with self._lock:
            self.subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: EventType, callback: Callable[[GACEvent], None]):
        with self._lock:
            if callback in self.subscribers[event_type]:
                self.subscribers[event_type].remove(callback)
    
    def publish(self, event: GACEvent):
        with self._lock:
            self.event_history.append(event)
            
        for callback in self.subscribers[event.event_type]:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    def get_event_history(self, event_type: Optional[EventType] = None, 
                         source_component: Optional[str] = None) -> List[GACEvent]:
        filtered_events = []
        for event in self.event_history:
            if event_type and event.event_type != event_type:
                continue
            if source_component and event.source_component != source_component:
                continue
            filtered_events.append(event)
        return filtered_events

class ComponentRegistry:
    def __init__(self):
        self.components = {}
        self.component_groups = defaultdict(list)
        self.dependency_graph = {}
        self._lock = threading.Lock()
    
    def register_component(self, component: GACComponent, group: Optional[str] = None):
        with self._lock:
            self.components[component.component_id] = component
            if group:
                self.component_groups[group].append(component.component_id)
    
    def unregister_component(self, component_id: str):
        with self._lock:
            if component_id in self.components:
                component = self.components[component_id]
                del self.components[component_id]
                
                for group_components in self.component_groups.values():
                    if component_id in group_components:
                        group_components.remove(component_id)
    
    def get_component(self, component_id: str) -> Optional[GACComponent]:
        return self.components.get(component_id)
    
    def get_components_by_group(self, group: str) -> List[GACComponent]:
        component_ids = self.component_groups.get(group, [])
        return [self.components[cid] for cid in component_ids if cid in self.components]
    
    def get_all_components(self) -> List[GACComponent]:
        return list(self.components.values())
    
    def add_dependency(self, component_id: str, depends_on: str):
        if component_id not in self.dependency_graph:
            self.dependency_graph[component_id] = []
        self.dependency_graph[component_id].append(depends_on)
    
    def get_execution_order(self) -> List[str]:
        visited = set()
        temp_visited = set()
        order = []
        
        def dfs(component_id):
            if component_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving {component_id}")
            if component_id in visited:
                return
            
            temp_visited.add(component_id)
            for dependency in self.dependency_graph.get(component_id, []):
                dfs(dependency)
            temp_visited.remove(component_id)
            visited.add(component_id)
            order.append(component_id)
        
        for component_id in self.components:
            if component_id not in visited:
                dfs(component_id)
        
        return order

class MetaLearningEngine:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.component_performance_history = defaultdict(list)
        self.adaptation_rules = {}
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.adaptation_threshold = self.config.get('adaptation_threshold', 0.1)
        
    def record_performance(self, component_id: str, performance_metrics: Dict[str, float]):
        self.component_performance_history[component_id].append({
            'timestamp': time.time(),
            'metrics': performance_metrics
        })
        
        history_limit = self.config.get('history_limit', 1000)
        if len(self.component_performance_history[component_id]) > history_limit:
            self.component_performance_history[component_id] = \
                self.component_performance_history[component_id][-history_limit:]
    
    def suggest_adaptations(self, component_id: str) -> Dict[str, Any]:
        history = self.component_performance_history.get(component_id, [])
        if len(history) < 10:
            return {}
        
        recent_performance = np.mean([h['metrics'].get('success_rate', 0.5) 
                                    for h in history[-10:]])
        historical_performance = np.mean([h['metrics'].get('success_rate', 0.5) 
                                        for h in history[:-10]])
        
        adaptations = {}
        
        if recent_performance < historical_performance - self.adaptation_threshold:
            adaptations['increase_sensitivity'] = True
            adaptations['suggested_threshold_adjustment'] = -0.05
        elif recent_performance > historical_performance + self.adaptation_threshold:
            adaptations['decrease_sensitivity'] = True
            adaptations['suggested_threshold_adjustment'] = 0.05
        
        return adaptations

class ReinforcementLearningAgent:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.discount_factor = self.config.get('discount_factor', 0.95)
        self.epsilon = self.config.get('epsilon', 0.1)
        self.action_space = ['increase_threshold', 'decrease_threshold', 'maintain', 'boost_component']
        
    def get_state_representation(self, system_metrics: Dict[str, Any]) -> str:
        performance = system_metrics.get('overall_performance', 0.5)
        load = system_metrics.get('system_load', 0.5)
        error_rate = system_metrics.get('error_rate', 0.0)
        
        perf_bucket = 'high' if performance > 0.8 else ('medium' if performance > 0.5 else 'low')
        load_bucket = 'high' if load > 0.8 else ('medium' if load > 0.5 else 'low')
        error_bucket = 'high' if error_rate > 0.1 else ('medium' if error_rate > 0.05 else 'low')
        
        return f"{perf_bucket}_{load_bucket}_{error_bucket}"
    
    def select_action(self, state: str) -> str:
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        
        q_values = self.q_table[state]
        if not q_values:
            return np.random.choice(self.action_space)
        
        return max(q_values, key=q_values.get)
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q


class DirectionAwareGradientComponent(GACComponent):
    """Wrapper component that integrates direction-aware gradient bounding into GAC system"""
    
    def __init__(self, component_id: str, enhanced_bounder, config: Dict[str, Any] = None):
        super().__init__(component_id, config)
        self.enhanced_bounder = enhanced_bounder
        self.processing_stats = {
            'total_processed': 0,
            'total_processing_time': 0.0,
            'direction_adjustments': 0,
            'validation_failures': 0
        }
    
    async def process_gradient(self, gradient: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """Process gradient through enhanced direction-aware bounding"""
        if gradient is None or not isinstance(gradient, torch.Tensor):
            return gradient
        
        start_time = time.time()
        
        try:
            # Process through enhanced direction-aware bounder
            result = self.enhanced_bounder.bound_gradients(gradient, context)
            
            # Update statistics
            self.processing_stats['total_processed'] += 1
            processing_time = time.time() - start_time
            self.processing_stats['total_processing_time'] += processing_time
            
            # Check if direction adjustments were applied
            if result.direction_based_adjustments:
                self.processing_stats['direction_adjustments'] += 1
            
            # Check validation confidence
            if result.direction_confidence < 0.5:
                self.processing_stats['validation_failures'] += 1
            
            # Emit gradient update event
            self.emit_event(EventType.GRADIENT_UPDATE, {
                'component_id': self.component_id,
                'original_norm': torch.norm(gradient).item(),
                'final_norm': torch.norm(result.bounded_gradients).item(),
                'applied_factor': result.applied_factor,
                'direction_type': result.direction_state.direction.value if result.direction_state else None,
                'direction_confidence': result.direction_confidence,
                'processing_time': processing_time
            })
            
            return result.bounded_gradients
            
        except Exception as e:
            logger.error(f"Direction-aware gradient processing error: {e}")
            self.metrics.error_count += 1
            return gradient  # Return original gradient on error
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information and statistics"""
        base_info = {
            'component_id': self.component_id,
            'component_type': 'direction_aware_gradient_bounder',
            'state': self.state.value,
            'metrics': {
                'activation_count': self.metrics.activation_count,
                'error_count': self.metrics.error_count,
                'success_rate': self.metrics.success_rate,
                'last_activation': self.metrics.last_activation
            },
            'processing_stats': self.processing_stats.copy()
        }
        
        # Add enhanced bounder statistics if available
        if hasattr(self.enhanced_bounder, 'get_enhanced_statistics'):
            try:
                enhanced_stats = self.enhanced_bounder.get_enhanced_statistics()
                base_info['enhanced_statistics'] = enhanced_stats
            except Exception as e:
                logger.warning(f"Failed to get enhanced statistics: {e}")
        
        return base_info


class GACSystem:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.event_bus = EventBus()
        self.component_registry = ComponentRegistry()
        self.meta_learning_engine = MetaLearningEngine(self.config.get('meta_learning', {}))
        self.rl_agent = ReinforcementLearningAgent(self.config.get('reinforcement_learning', {}))
        
        self.system_state = ComponentState.INACTIVE
        self.global_thresholds = self.config.get('thresholds', {
            'gradient_magnitude': 1.0,
            'processing_time': 5.0,
            'error_rate': 0.05
        })
        
        self.performance_metrics = {
            'total_gradients_processed': 0,
            'average_processing_time': 0.0,
            'system_uptime': 0.0,
            'error_count': 0,
            'adaptation_count': 0
        }
        
        self.start_time = time.time()
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 8))
        self._shutdown = False
        
        self.brain_integration_hooks = {
            'pre_training': [],
            'post_training': [],
            'gradient_update': [],
            'error_callback': []
        }
        
        # Initialize Direction Components
        self._initialize_direction_components()
        
        self._setup_event_handlers()
    
    def _initialize_direction_components(self):
        """Initialize GAC Direction Components and register them as components"""
        if not DIRECTION_COMPONENTS_AVAILABLE:
            logger.warning("Direction components not available, skipping initialization")
            return
        
        try:
            # Get direction component configuration
            direction_config = self.config.get('direction_components', {})
            
            # 1. Initialize Direction State Manager
            state_manager_config = direction_config.get('state_manager', {})
            self.direction_state_manager = DirectionStateManager(state_manager_config)
            logger.info("Direction State Manager initialized")
            
            # 2. Initialize Direction Validator
            validator_config = direction_config.get('validator', {})
            self.direction_validator = DirectionValidator(validator_config)
            self.direction_validator.set_direction_state_manager(self.direction_state_manager)
            logger.info("Direction Validator initialized")
            
            # 3. Initialize Basic Gradient Bounder
            basic_config = direction_config.get('basic_bounder', {})
            self.basic_bounder = BasicGradientBounder(basic_config)
            logger.info("Basic Gradient Bounder initialized")
            
            # 4. Initialize Enhanced Gradient Bounder
            enhanced_config = direction_config.get('enhanced_bounder', {})
            enhanced_config['basic_config'] = basic_config  # Pass basic config to enhanced
            self.enhanced_bounder = EnhancedGradientBounder(enhanced_config)
            self.enhanced_bounder.set_direction_components(
                self.direction_state_manager, 
                self.direction_validator
            )
            logger.info("Enhanced Gradient Bounder initialized")
            
            # 5. Create GAC component wrappers and register them
            self._register_direction_component_wrappers()
            
            logger.info("All Direction Components initialized and registered successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize direction components: {e}")
            # Set fallbacks to None if initialization fails
            self.direction_state_manager = None
            self.direction_validator = None
            self.basic_bounder = None
            self.enhanced_bounder = None
    
    def _register_direction_component_wrappers(self):
        """Register direction components as GAC components"""
        try:
            # Create wrapper components that integrate with the GAC system
            if self.enhanced_bounder:
                enhanced_wrapper = DirectionAwareGradientComponent(
                    "enhanced_direction_bounder",
                    self.enhanced_bounder,
                    self.config.get('direction_components', {})
                )
                self.component_registry.register_component(enhanced_wrapper)
                # Priority handling would need to be added to ComponentRegistry if needed
                logger.info("Enhanced Direction Bounder registered as GAC component")
            
        except Exception as e:
            logger.error(f"Failed to register direction component wrappers: {e}")
    
    def _setup_event_handlers(self):
        self.event_bus.subscribe(EventType.GRADIENT_UPDATE, self._handle_gradient_update)
        self.event_bus.subscribe(EventType.THRESHOLD_EXCEEDED, self._handle_threshold_exceeded)
        self.event_bus.subscribe(EventType.COMPONENT_STATE_CHANGE, self._handle_component_state_change)
        self.event_bus.subscribe(EventType.SYSTEM_ALERT, self._handle_system_alert)
    
    def register_brain_hook(self, hook_type: str, callback: Callable):
        if hook_type in self.brain_integration_hooks:
            self.brain_integration_hooks[hook_type].append(callback)
    
    def execute_brain_hooks(self, hook_type: str, *args, **kwargs):
        for callback in self.brain_integration_hooks.get(hook_type, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Brain hook execution error ({hook_type}): {e}")
    
    def start_system(self):
        self.system_state = ComponentState.ACTIVE
        self.start_time = time.time()
        logger.info("GAC System started")
        
        for component in self.component_registry.get_all_components():
            if component.state == ComponentState.INACTIVE:
                component.update_state(ComponentState.ACTIVE)
    
    def stop_system(self):
        self._shutdown = True
        self.system_state = ComponentState.INACTIVE
        
        for component in self.component_registry.get_all_components():
            component.update_state(ComponentState.INACTIVE)
        
        self.executor.shutdown(wait=True)
        logger.info("GAC System stopped")
    
    async def process_gradient(self, gradient: torch.Tensor, 
                             context: Dict[str, Any] = None) -> torch.Tensor:
        if self.system_state != ComponentState.ACTIVE:
            return gradient
        
        context = context or {}
        start_time = time.time()
        
        try:
            processed_gradient = gradient.clone()
            execution_order = self.component_registry.get_execution_order()
            
            for component_id in execution_order:
                component = self.component_registry.get_component(component_id)
                if component and component.state == ComponentState.ACTIVE:
                    try:
                        processed_gradient = await component.process_gradient(
                            processed_gradient, context
                        )
                        component.metrics.activation_count += 1
                        component.metrics.last_activation = time.time()
                    except Exception as e:
                        component.metrics.error_count += 1
                        logger.error(f"Component {component_id} processing error: {e}")
                        
                        self.event_bus.publish(GACEvent(
                            EventType.SYSTEM_ALERT,
                            "gac_system",
                            {"error": str(e), "component": component_id, "severity": "high"}
                        ))
            
            processing_time = time.time() - start_time
            self.performance_metrics['total_gradients_processed'] += 1
            self.performance_metrics['average_processing_time'] = (
                (self.performance_metrics['average_processing_time'] * 
                 (self.performance_metrics['total_gradients_processed'] - 1) + processing_time) /
                self.performance_metrics['total_gradients_processed']
            )
            
            self.event_bus.publish(GACEvent(
                EventType.GRADIENT_UPDATE,
                "gac_system",
                {
                    "gradient_magnitude": torch.norm(processed_gradient).item(),
                    "processing_time": processing_time,
                    "components_processed": len(execution_order)
                }
            ))
            
            self.execute_brain_hooks('gradient_update', processed_gradient, context)
            return processed_gradient
            
        except Exception as e:
            self.performance_metrics['error_count'] += 1
            self.execute_brain_hooks('error_callback', e, gradient, context)
            logger.error(f"GAC System processing error: {e}")
            return gradient
    
    def _handle_gradient_update(self, event: GACEvent):
        magnitude = event.data.get('gradient_magnitude', 0)
        if magnitude > self.global_thresholds['gradient_magnitude']:
            self.event_bus.publish(GACEvent(
                EventType.THRESHOLD_EXCEEDED,
                "gac_system",
                {"threshold_type": "gradient_magnitude", "value": magnitude}
            ))
    
    def _handle_threshold_exceeded(self, event: GACEvent):
        threshold_type = event.data.get('threshold_type')
        value = event.data.get('value')
        logger.warning(f"Threshold exceeded: {threshold_type} = {value}")
        
        system_metrics = self.get_system_metrics()
        state = self.rl_agent.get_state_representation(system_metrics)
        action = self.rl_agent.select_action(state)
        
        self._execute_rl_action(action, threshold_type, value)
    
    def _handle_component_state_change(self, event: GACEvent):
        logger.info(f"Component {event.source_component} state changed: "
                   f"{event.data['old_state']} -> {event.data['new_state']}")
    
    def _handle_system_alert(self, event: GACEvent):
        severity = event.data.get('severity', 'medium')
        error = event.data.get('error', 'Unknown error')
        component = event.data.get('component', 'Unknown component')
        
        logger.error(f"System alert ({severity}): {error} in {component}")
        
        if severity == 'critical':
            self._initiate_emergency_protocols()
    
    def _execute_rl_action(self, action: str, threshold_type: str, value: float):
        if action == 'increase_threshold':
            self.global_thresholds[threshold_type] *= 1.1
        elif action == 'decrease_threshold':
            self.global_thresholds[threshold_type] *= 0.9
        elif action == 'boost_component':
            # Find underperforming components and boost them
            for component in self.component_registry.get_all_components():
                if component.metrics.success_rate < 0.8:
                    component.pid_state.kp *= 1.2
        
        self.performance_metrics['adaptation_count'] += 1
        logger.info(f"RL action executed: {action} for {threshold_type}")
    
    def _initiate_emergency_protocols(self):
        logger.critical("Initiating emergency protocols")
        
        for component in self.component_registry.get_all_components():
            if component.metrics.error_count > 5:
                component.update_state(ComponentState.MAINTENANCE)
        
        self.global_thresholds = {k: v * 2.0 for k, v in self.global_thresholds.items()}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        active_components = sum(1 for c in self.component_registry.get_all_components() 
                              if c.state == ComponentState.ACTIVE)
        total_components = len(self.component_registry.get_all_components())
        
        overall_performance = (
            self.performance_metrics['total_gradients_processed'] / 
            (self.performance_metrics['total_gradients_processed'] + 
             self.performance_metrics['error_count'] + 1)
        )
        
        system_load = min(1.0, self.performance_metrics['average_processing_time'] / 
                         self.global_thresholds['processing_time'])
        
        error_rate = (
            self.performance_metrics['error_count'] / 
            max(1, self.performance_metrics['total_gradients_processed'])
        )
        
        return {
            'system_state': self.system_state.value,
            'active_components': active_components,
            'total_components': total_components,
            'overall_performance': overall_performance,
            'system_load': system_load,
            'error_rate': error_rate,
            'uptime': time.time() - self.start_time,
            'thresholds': self.global_thresholds.copy(),
            'performance_metrics': self.performance_metrics.copy()
        }
    
    def save_system_state(self, filepath: str):
        state_data = {
            'config': self.config,
            'global_thresholds': self.global_thresholds,
            'performance_metrics': self.performance_metrics,
            'rl_q_table': dict(self.rl_agent.q_table),
            'component_performance_history': dict(self.meta_learning_engine.component_performance_history),
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        logger.info(f"System state saved to {filepath}")
    
    def load_system_state(self, filepath: str):
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
            
            self.global_thresholds = state_data.get('global_thresholds', self.global_thresholds)
            self.performance_metrics = state_data.get('performance_metrics', self.performance_metrics)
            
            if 'rl_q_table' in state_data:
                for state, actions in state_data['rl_q_table'].items():
                    self.rl_agent.q_table[state].update(actions)
            
            if 'component_performance_history' in state_data:
                self.meta_learning_engine.component_performance_history.update(
                    state_data['component_performance_history']
                )
            
            logger.info(f"System state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load system state: {e}")
            return False
    
    def get_component_status(self) -> Dict[str, Dict[str, Any]]:
        status = {}
        for component in self.component_registry.get_all_components():
            status[component.component_id] = {
                'state': component.state.value,
                'metrics': {
                    'activation_count': component.metrics.activation_count,
                    'error_count': component.metrics.error_count,
                    'success_rate': component.metrics.success_rate,
                    'last_activation': component.metrics.last_activation
                },
                'pid_state': {
                    'kp': component.pid_state.kp,
                    'ki': component.pid_state.ki,
                    'kd': component.pid_state.kd,
                    'setpoint': component.pid_state.setpoint
                },
                'info': component.get_component_info()
            }
        return status
    
    def register_component(self, component: GACComponent, group: Optional[str] = None):
        self.component_registry.register_component(component, group)
        
        component.register_event_handler(
            EventType.GRADIENT_UPDATE, 
            lambda event: self.event_bus.publish(event)
        )
        component.register_event_handler(
            EventType.COMPONENT_STATE_CHANGE,
            lambda event: self.event_bus.publish(event)
        )
        
        logger.info(f"Component {component.component_id} registered" + 
                   (f" in group {group}" if group else ""))
    
    def unregister_component(self, component_id: str):
        self.component_registry.unregister_component(component_id)
        logger.info(f"Component {component_id} unregistered")
    
    def add_component_dependency(self, component_id: str, depends_on: str):
        self.component_registry.add_dependency(component_id, depends_on)
    
    def create_checkpoint(self, checkpoint_path: str):
        checkpoint_dir = Path(checkpoint_path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        
        self.save_system_state(checkpoint_dir / f"system_state_{timestamp}.json")
        
        component_states = {}
        for component in self.component_registry.get_all_components():
            component_states[component.component_id] = {
                'state': component.state.value,
                'metrics': component.metrics.__dict__,
                'pid_state': component.pid_state.__dict__,
                'config': component.config
            }
        
        with open(checkpoint_dir / f"components_{timestamp}.json", 'w') as f:
            json.dump(component_states, f, indent=2)
        
        logger.info(f"Checkpoint created at {checkpoint_path}")
        return timestamp
    
    def restore_checkpoint(self, checkpoint_path: str, timestamp: int):
        checkpoint_dir = Path(checkpoint_path)
        
        system_state_file = checkpoint_dir / f"system_state_{timestamp}.json"
        components_file = checkpoint_dir / f"components_{timestamp}.json"
        
        if not system_state_file.exists() or not components_file.exists():
            logger.error(f"Checkpoint files not found for timestamp {timestamp}")
            return False
        
        if not self.load_system_state(str(system_state_file)):
            return False
        
        try:
            with open(components_file, 'r') as f:
                component_states = json.load(f)
            
            for component_id, state_data in component_states.items():
                component = self.component_registry.get_component(component_id)
                if component:
                    component.state = ComponentState(state_data['state'])
                    
                    for key, value in state_data['metrics'].items():
                        setattr(component.metrics, key, value)
                    
                    for key, value in state_data['pid_state'].items():
                        setattr(component.pid_state, key, value)
                    
                    component.config.update(state_data['config'])
            
            logger.info(f"Checkpoint restored from timestamp {timestamp}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return False
    
    def initialize_thread_manager(self) -> bool:
        """Initialize the GAC thread manager for synchronous gradient processing"""
        try:
            if hasattr(self, 'thread_manager') and self.thread_manager:
                return True  # Already initialized
            
            self.thread_manager = GACThreadManager(self)
            success = self.thread_manager.start()
            
            if success:
                logger.info("GAC thread manager initialized successfully")
            else:
                logger.error("Failed to start GAC thread manager")
                self.thread_manager = None
            
            return success
            
        except Exception as e:
            logger.error(f"Error initializing GAC thread manager: {e}")
            self.thread_manager = None
            return False
    
    def clip_gradients(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """Synchronous wrapper for gradient processing using thread manager"""
        if not gradients:
            return []
        
        # Ensure thread manager is initialized
        if not hasattr(self, 'thread_manager') or not self.thread_manager:
            if not self.initialize_thread_manager():
                logger.warning("GAC thread manager not available, returning original gradients")
                return gradients
        
        # Use thread manager for processing
        return self.thread_manager.process_gradients_sync(gradients)
    
    def get_gac_stats(self) -> Dict[str, Any]:
        """Get GAC system statistics including thread manager stats"""
        stats = {
            'system_state': self.system_state.value,
            'active_components': len([c for c in self.component_registry.get_all_components() 
                                    if c.state == ComponentState.ACTIVE]),
            'total_components': len(self.component_registry.get_all_components())
        }
        
        if hasattr(self, 'thread_manager') and self.thread_manager:
            stats['thread_manager'] = self.thread_manager.get_stats()
        
        return stats
    
    # ======================== DIRECTION COMPONENT ACCESS METHODS ========================
    
    def get_direction_state_manager(self) -> Optional:
        """Get the direction state manager instance"""
        return getattr(self, 'direction_state_manager', None)
    
    def get_direction_validator(self) -> Optional:
        """Get the direction validator instance"""
        return getattr(self, 'direction_validator', None)
    
    def get_basic_bounder(self) -> Optional:
        """Get the basic gradient bounder instance"""
        return getattr(self, 'basic_bounder', None)
    
    def get_enhanced_bounder(self) -> Optional:
        """Get the enhanced gradient bounder instance"""
        return getattr(self, 'enhanced_bounder', None)
    
    def get_direction_components_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all direction components"""
        return {
            'components_available': DIRECTION_COMPONENTS_AVAILABLE,
            'direction_state_manager': {
                'initialized': hasattr(self, 'direction_state_manager') and self.direction_state_manager is not None,
                'summary': self.direction_state_manager.get_direction_summary() if hasattr(self, 'direction_state_manager') and self.direction_state_manager else None
            },
            'direction_validator': {
                'initialized': hasattr(self, 'direction_validator') and self.direction_validator is not None,
                'summary': self.direction_validator.get_validation_summary() if hasattr(self, 'direction_validator') and self.direction_validator else None
            },
            'basic_bounder': {
                'initialized': hasattr(self, 'basic_bounder') and self.basic_bounder is not None,
                'statistics': self.basic_bounder.get_bounding_statistics() if hasattr(self, 'basic_bounder') and self.basic_bounder else None
            },
            'enhanced_bounder': {
                'initialized': hasattr(self, 'enhanced_bounder') and self.enhanced_bounder is not None,
                'statistics': self.enhanced_bounder.get_enhanced_statistics() if hasattr(self, 'enhanced_bounder') and self.enhanced_bounder else None,
                'health_status': self.enhanced_bounder.is_enhancement_healthy() if hasattr(self, 'enhanced_bounder') and self.enhanced_bounder else None
            }
        }
    
    def reset_direction_components(self):
        """Reset all direction component statistics and state"""
        if hasattr(self, 'direction_state_manager') and self.direction_state_manager:
            self.direction_state_manager.reset()
        
        if hasattr(self, 'direction_validator') and self.direction_validator:
            self.direction_validator.reset()
        
        if hasattr(self, 'basic_bounder') and self.basic_bounder:
            self.basic_bounder.reset_statistics()
        
        if hasattr(self, 'enhanced_bounder') and self.enhanced_bounder:
            self.enhanced_bounder.reset_statistics()
        
        logger.info("All direction components reset successfully")

# ======================== GAC THREAD MANAGER ========================

@dataclass
class GradientRequest:
    """Request for gradient processing"""
    request_id: str
    gradients: List[torch.Tensor]
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass  
class GradientResponse:
    """Response from gradient processing"""
    request_id: str
    processed_gradients: List[torch.Tensor]
    success: bool = True
    error_message: Optional[str] = None
    processing_time: float = 0.0
    timestamp: float = field(default_factory=time.time)

class GACThreadManager:
    """Manages GAC system in dedicated thread with asyncio event loop"""
    
    def __init__(self, gac_system: GACSystem, max_queue_size: int = 100):
        self.gac_system = gac_system
        self.max_queue_size = max_queue_size
        
        # Thread-safe queues for communication
        self.request_queue = queue.Queue(maxsize=max_queue_size)
        self.response_queue = queue.Queue(maxsize=max_queue_size)
        
        # Thread management
        self.gac_thread = None
        self.event_loop = None
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Performance tracking
        self.processed_requests = 0
        self.failed_requests = 0
        self.total_processing_time = 0.0
        
        self.logger = logging.getLogger(f"{__name__}.GACThreadManager")
    
    def start(self) -> bool:
        """Start the GAC thread with asyncio event loop"""
        if self.running:
            self.logger.warning("GAC thread already running")
            return True
        
        try:
            self.running = True
            self.shutdown_event.clear()
            
            # Start GAC processing thread
            self.gac_thread = threading.Thread(
                target=self._run_gac_thread,
                name="GAC-Processing-Thread",
                daemon=True
            )
            self.gac_thread.start()
            
            # Wait a moment for thread to start
            time.sleep(0.1)
            
            self.logger.info("GAC thread manager started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start GAC thread: {e}")
            self.running = False
            return False
    
    def stop(self) -> bool:
        """Stop the GAC thread and cleanup"""
        if not self.running:
            return True
        
        try:
            self.logger.info("Stopping GAC thread manager...")
            
            # Signal shutdown
            self.running = False
            self.shutdown_event.set()
            
            # Wait for thread to finish
            if self.gac_thread and self.gac_thread.is_alive():
                self.gac_thread.join(timeout=5.0)
                
                if self.gac_thread.is_alive():
                    self.logger.warning("GAC thread did not stop gracefully")
                    return False
            
            self.logger.info("GAC thread manager stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping GAC thread: {e}")
            return False
    
    def _run_gac_thread(self):
        """Main GAC thread function with asyncio event loop"""
        try:
            # Create new event loop for this thread
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            
            # Start GAC system in this thread's event loop
            self.event_loop.run_until_complete(self._start_gac_system())
            
            # Main processing loop
            self.event_loop.run_until_complete(self._process_gradient_requests())
            
        except Exception as e:
            self.logger.error(f"GAC thread error: {e}")
        finally:
            # Cleanup
            try:
                if self.event_loop:
                    self.event_loop.run_until_complete(self._stop_gac_system())
                    self.event_loop.close()
            except Exception as e:
                self.logger.error(f"GAC thread cleanup error: {e}")
    
    async def _start_gac_system(self):
        """Start GAC system in async context"""
        try:
            self.gac_system.start_system()
            self.logger.info("GAC system started in dedicated thread")
        except Exception as e:
            self.logger.error(f"Failed to start GAC system in thread: {e}")
            raise
    
    async def _stop_gac_system(self):
        """Stop GAC system in async context"""
        try:
            self.gac_system.stop_system()
            self.logger.info("GAC system stopped in dedicated thread")
        except Exception as e:
            self.logger.error(f"Error stopping GAC system in thread: {e}")
    
    async def _process_gradient_requests(self):
        """Main processing loop for gradient requests"""
        self.logger.info("GAC thread processing loop started")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Check for new requests (non-blocking)
                try:
                    request = self.request_queue.get_nowait()
                    await self._handle_gradient_request(request)
                except queue.Empty:
                    # No requests, sleep briefly
                    await asyncio.sleep(0.001)  # 1ms sleep
                    continue
                    
            except Exception as e:
                self.logger.error(f"Error in gradient processing loop: {e}")
                await asyncio.sleep(0.01)  # Brief pause on error
        
        self.logger.info("GAC thread processing loop ended")
    
    async def _handle_gradient_request(self, request: GradientRequest):
        """Process a single gradient request"""
        start_time = time.time()
        
        try:
            # Process each gradient through GAC system
            processed_gradients = []
            
            for gradient in request.gradients:
                if gradient is not None:
                    processed_grad = await self.gac_system.process_gradient(gradient, request.context)
                    processed_gradients.append(processed_grad)
                else:
                    processed_gradients.append(None)
            
            # Create successful response
            processing_time = time.time() - start_time
            response = GradientResponse(
                request_id=request.request_id,
                processed_gradients=processed_gradients,
                success=True,
                processing_time=processing_time
            )
            
            # Update stats
            self.processed_requests += 1
            self.total_processing_time += processing_time
            
        except Exception as e:
            # Create error response
            processing_time = time.time() - start_time
            response = GradientResponse(
                request_id=request.request_id,
                processed_gradients=request.gradients,  # Return originals on error
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
            
            self.failed_requests += 1
            self.logger.error(f"GAC gradient processing failed: {e}")
        
        # Send response back
        try:
            self.response_queue.put_nowait(response)
        except queue.Full:
            self.logger.error("Response queue full, dropping response")
    
    def process_gradients_sync(self, gradients: List[torch.Tensor], 
                              timeout: float = 1.0,
                              context: Dict[str, Any] = None) -> List[torch.Tensor]:
        """Synchronous interface for gradient processing"""
        if not self.running:
            self.logger.warning("GAC thread not running, returning original gradients")
            return gradients
        
        # Create request
        request_id = str(uuid.uuid4())
        request = GradientRequest(
            request_id=request_id,
            gradients=gradients,
            context=context or {}
        )
        
        try:
            # Send request
            self.request_queue.put(request, timeout=0.1)
            
            # Wait for response
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = self.response_queue.get_nowait()
                    if response.request_id == request_id:
                        if response.success:
                            return response.processed_gradients
                        else:
                            self.logger.warning(f"GAC processing failed: {response.error_message}")
                            return gradients  # Return originals on error
                    else:
                        # Wrong response, put it back
                        self.response_queue.put(response)
                except queue.Empty:
                    time.sleep(0.001)  # Brief sleep
                    continue
            
            # Timeout
            self.logger.warning(f"GAC processing timeout after {timeout}s")
            return gradients
            
        except queue.Full:
            self.logger.warning("GAC request queue full, returning original gradients")
            return gradients
        except Exception as e:
            self.logger.error(f"GAC sync processing error: {e}")
            return gradients
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'running': self.running,
            'processed_requests': self.processed_requests,
            'failed_requests': self.failed_requests,
            'success_rate': (self.processed_requests / 
                           (self.processed_requests + self.failed_requests) 
                           if (self.processed_requests + self.failed_requests) > 0 else 0),
            'avg_processing_time': (self.total_processing_time / self.processed_requests 
                                  if self.processed_requests > 0 else 0),
            'queue_sizes': {
                'request_queue': self.request_queue.qsize(),
                'response_queue': self.response_queue.qsize()
            }
        }

def create_gac_system(config_input: Optional[Union[str, 'GACConfig']] = None) -> GACSystem:
    config = {}
    if config_input:
        if isinstance(config_input, str):
            # Handle string path
            if Path(config_input).exists():
                with open(config_input, 'r') as f:
                    config = json.load(f)
        else:
            # Handle GACConfig object - convert to dict
            from dataclasses import asdict
            config = asdict(config_input)
    
    return GACSystem(config)

async def main():
    # Example usage
    gac_system = create_gac_system()
    
    try:
        gac_system.start_system()
        
        # Example gradient processing
        test_gradient = torch.randn(100, 100)
        processed_gradient = await gac_system.process_gradient(test_gradient)
        
        print(f"Original gradient norm: {torch.norm(test_gradient).item():.4f}")
        print(f"Processed gradient norm: {torch.norm(processed_gradient).item():.4f}")
        
        # Display system metrics
        metrics = gac_system.get_system_metrics()
        print(f"System metrics: {json.dumps(metrics, indent=2)}")
        
    finally:
        gac_system.stop_system()

if __name__ == "__main__":
    asyncio.run(main()) 