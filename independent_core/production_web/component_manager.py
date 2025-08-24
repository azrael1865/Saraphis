"""
Saraphis Component Manager
Production-ready dashboard component management system
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict, deque
import json
import hashlib

logger = logging.getLogger(__name__)


class ComponentManager:
    """Production-ready component management for dynamic dashboards"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Component registry
        self.component_registry = {
            'chart': {
                'type': 'visualization',
                'requires_data': True,
                'update_frequency': 'realtime',
                'supported_data_types': ['timeseries', 'scatter', 'bar', 'pie']
            },
            'metric': {
                'type': 'display',
                'requires_data': True,
                'update_frequency': 'realtime',
                'supported_data_types': ['numeric', 'percentage', 'currency']
            },
            'alert': {
                'type': 'notification',
                'requires_data': True,
                'update_frequency': 'event',
                'supported_data_types': ['alert', 'warning', 'info']
            },
            'table': {
                'type': 'data',
                'requires_data': True,
                'update_frequency': 'periodic',
                'supported_data_types': ['tabular', 'list']
            },
            'gauge': {
                'type': 'visualization',
                'requires_data': True,
                'update_frequency': 'realtime',
                'supported_data_types': ['numeric', 'percentage']
            },
            'heatmap': {
                'type': 'visualization',
                'requires_data': True,
                'update_frequency': 'periodic',
                'supported_data_types': ['matrix', 'grid']
            },
            'timeline': {
                'type': 'visualization',
                'requires_data': True,
                'update_frequency': 'periodic',
                'supported_data_types': ['events', 'timeseries']
            },
            'log_viewer': {
                'type': 'data',
                'requires_data': True,
                'update_frequency': 'stream',
                'supported_data_types': ['logs', 'text']
            },
            'control_panel': {
                'type': 'interactive',
                'requires_data': False,
                'update_frequency': 'manual',
                'supported_data_types': ['controls', 'settings']
            },
            'status_indicator': {
                'type': 'display',
                'requires_data': True,
                'update_frequency': 'realtime',
                'supported_data_types': ['boolean', 'status']
            }
        }
        
        # Component states
        self.component_states = {}  # component_id -> state
        self.component_data_bindings = defaultdict(list)  # data_source -> [component_ids]
        self.component_update_queue = deque()
        
        # Component lifecycle tracking
        self.component_lifecycle = defaultdict(dict)
        self.component_dependencies = defaultdict(set)
        
        # Update configuration
        self.max_update_batch_size = config.get('max_update_batch_size', 50)
        self.update_debounce_ms = config.get('update_debounce_ms', 100)
        self.component_timeout_seconds = config.get('component_timeout_seconds', 30)
        
        # Performance tracking
        self.component_metrics = defaultdict(lambda: {
            'render_count': 0,
            'update_count': 0,
            'error_count': 0,
            'total_render_time': 0,
            'total_update_time': 0,
            'last_render': None,
            'last_update': None
        })
        
        # Thread safety
        self._lock = threading.Lock()
        self._update_timers = {}
        
        # Start background threads
        self._start_background_threads()
        
        self.logger.info("Component Manager initialized")
    
    def register_component(self, component_id: str, component_type: str, 
                         configuration: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new component instance"""
        try:
            # Validate component type
            if component_type not in self.component_registry:
                raise ValueError(f"HARD FAILURE: Unknown component type '{component_type}'. Valid types: {list(self.component_registry.keys())}")
            
            # Generate unique component ID if not provided
            if not component_id:
                component_id = self._generate_component_id(component_type)
            
            with self._lock:
                # Initialize component state
                self.component_states[component_id] = {
                    'id': component_id,
                    'type': component_type,
                    'configuration': configuration,
                    'state': 'initializing',
                    'data': None,
                    'rendered_content': None,
                    'last_updated': None,
                    'error': None,
                    'metadata': {
                        'created_at': time.time(),
                        'version': 1,
                        'registry_info': self.component_registry[component_type]
                    }
                }
                
                # Initialize lifecycle tracking
                self.component_lifecycle[component_id] = {
                    'created': time.time(),
                    'state_changes': [('initializing', time.time())],
                    'render_history': deque(maxlen=100),
                    'error_history': deque(maxlen=50)
                }
                
                # Initialize metrics tracking
                self.component_metrics[component_id] = {
                    'render_count': 0,
                    'update_count': 0,
                    'error_count': 0,
                    'total_render_time': 0.0,
                    'total_update_time': 0.0,
                    'last_render': None,
                    'last_update': None
                }
                
                # Process dependencies
                if 'dependencies' in configuration:
                    self.component_dependencies[component_id] = set(configuration['dependencies'])
                
                # Set initial state to ready
                self._update_component_state(component_id, 'ready')
                
                self.logger.info(f"Component registered: {component_id} ({component_type})")
                
                return {
                    'success': True,
                    'component_id': component_id,
                    'component_state': self.component_states[component_id]
                }
                
        except (ValueError, RuntimeError):
            raise  # Re-raise our own hard failures without double-wrapping
        except Exception as e:
            self.logger.error(f"Component registration failed: {e}")
            raise RuntimeError(f"HARD FAILURE: Component registration failed: {e}")
    
    def update_components(self, components: Dict[str, Any], 
                         data: Dict[str, Any]) -> Dict[str, Any]:
        """Update multiple components with new data"""
        try:
            updated_components = {}
            
            # Process each component
            for component_id, component_config in components.items():
                if component_id not in self.component_states:
                    # Register new component
                    reg_result = self.register_component(
                        component_id,
                        component_config.get('type', 'metric'),
                        component_config
                    )
                    
                    if not reg_result['success']:
                        continue
                
                # Update component with data
                update_result = self._update_single_component(
                    component_id, data.get(component_id, {})
                )
                
                if update_result['success']:
                    updated_components[component_id] = update_result['component_state']
            
            return updated_components
            
        except (ValueError, RuntimeError):
            raise  # Re-raise our own hard failures without double-wrapping
        except Exception as e:
            self.logger.error(f"Component update failed: {e}")
            raise RuntimeError(f"HARD FAILURE: Component update failed: {e}")
    
    def update_interface_components(self, user_id: str, 
                                  interaction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Update interface components based on user interaction"""
        try:
            interface_updates = {}
            affected_components = interaction_result.get('affected_components', [])
            
            with self._lock:
                for component_id in affected_components:
                    if component_id not in self.component_states:
                        continue
                    
                    component = self.component_states[component_id]
                    
                    # Apply interaction updates
                    if 'updates' in interaction_result:
                        updates = interaction_result['updates'].get(component_id, {})
                        
                        # Update configuration
                        if 'configuration' in updates:
                            component['configuration'].update(updates['configuration'])
                        
                        # Update data
                        if 'data' in updates:
                            component['data'] = updates['data']
                            component['last_updated'] = time.time()
                        
                        # Re-render if needed
                        if updates.get('requires_render', True):
                            render_result = self._render_component(component_id)
                            if render_result['success']:
                                component['rendered_content'] = render_result['content']
                        
                        # Update state
                        if 'state' in updates:
                            self._update_component_state(component_id, updates['state'])
                        
                        interface_updates[component_id] = {
                            'state': component['state'],
                            'rendered_content': component['rendered_content'],
                            'last_updated': component['last_updated']
                        }
                
                # Handle cascading updates
                cascaded_updates = self._process_cascading_updates(affected_components)
                interface_updates.update(cascaded_updates)
            
            return interface_updates
            
        except Exception as e:
            self.logger.error(f"Interface component update failed: {e}")
            raise RuntimeError(f"HARD FAILURE: Interface component update failed: {e}")
    
    def apply_component_update(self, component_id: str, update_data: Dict[str, Any]):
        """Apply specific update to component"""
        try:
            with self._lock:
                if component_id not in self.component_states:
                    raise ValueError(f"HARD FAILURE: Component not found: {component_id}")
                
                component = self.component_states[component_id]
                
                # Apply updates
                if 'data' in update_data:
                    component['data'] = update_data['data']
                
                if 'configuration' in update_data:
                    component['configuration'].update(update_data['configuration'])
                
                if 'state' in update_data:
                    self._update_component_state(component_id, update_data['state'])
                
                component['last_updated'] = time.time()
                
                # Queue for rendering
                self._queue_component_update(component_id)
                
        except (ValueError, RuntimeError):
            raise  # Re-raise our own hard failures without double-wrapping
        except Exception as e:
            self.logger.error(f"Component update application failed: {e}")
            raise RuntimeError(f"HARD FAILURE: Component update application failed: {e}")
    
    def get_component_state(self, component_id: str) -> Dict[str, Any]:
        """Get current state of component"""
        try:
            with self._lock:
                if component_id not in self.component_states:
                    raise ValueError(f"HARD FAILURE: Component not found for state retrieval: {component_id}")
                
                return {
                    'success': True,
                    'component_state': self.component_states[component_id].copy()
                }
                
        except (ValueError, RuntimeError):
            raise  # Re-raise our own hard failures without double-wrapping
        except Exception as e:
            self.logger.error(f"Component state retrieval failed: {e}")
            raise RuntimeError(f"HARD FAILURE: Component state retrieval failed: {e}")
    
    def get_component_metrics(self, component_id: Optional[str] = None) -> Dict[str, Any]:
        """Get component performance metrics"""
        try:
            with self._lock:
                if component_id:
                    if component_id not in self.component_metrics:
                        raise ValueError(f'HARD FAILURE: No metrics found for component: {component_id}')
                    
                    return {
                        'component_id': component_id,
                        'metrics': self.component_metrics[component_id].copy()
                    }
                else:
                    # Return all component metrics
                    all_metrics = {}
                    for cid, metrics in self.component_metrics.items():
                        all_metrics[cid] = metrics.copy()
                    
                    return {
                        'total_components': len(self.component_states),
                        'component_metrics': all_metrics,
                        'summary': self._generate_metrics_summary()
                    }
                    
        except (ValueError, RuntimeError):
            raise  # Re-raise our own hard failures without double-wrapping
        except Exception as e:
            self.logger.error(f"Component metrics retrieval failed: {e}")
            raise RuntimeError(f"HARD FAILURE: Component metrics retrieval failed: {e}")
    
    def remove_component(self, component_id: str) -> Dict[str, Any]:
        """Remove component and clean up resources"""
        try:
            with self._lock:
                if component_id not in self.component_states:
                    raise ValueError(f"HARD FAILURE: Component not found for removal: {component_id}")
                
                # Remove from all tracking structures
                del self.component_states[component_id]
                self.component_lifecycle.pop(component_id, None)
                self.component_metrics.pop(component_id, None)
                
                # Remove from dependencies
                self.component_dependencies.pop(component_id, None)
                for deps in self.component_dependencies.values():
                    deps.discard(component_id)
                
                # Remove from data bindings
                for bindings in self.component_data_bindings.values():
                    if component_id in bindings:
                        bindings.remove(component_id)
                
                # Cancel any pending updates
                if component_id in self._update_timers:
                    self._update_timers[component_id].cancel()
                    del self._update_timers[component_id]
                
                self.logger.info(f"Component removed: {component_id}")
                
                return {
                    'success': True,
                    'message': f'Component {component_id} removed successfully'
                }
                
        except (ValueError, RuntimeError):
            raise  # Re-raise our own hard failures without double-wrapping
        except Exception as e:
            self.logger.error(f"Component removal failed: {e}")
            raise RuntimeError(f"HARD FAILURE: Component removal failed: {e}")
    
    def _update_single_component(self, component_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update single component with data"""
        try:
            start_time = time.time()
            
            with self._lock:
                if component_id not in self.component_states:
                    raise ValueError(f"HARD FAILURE: Component not found for single update: {component_id}")
                
                component = self.component_states[component_id]
                
                # Update component data
                component['data'] = data
                component['last_updated'] = time.time()
                
                # Render component
                render_result = self._render_component(component_id)
                
                if render_result['success']:
                    component['rendered_content'] = render_result['content']
                    self._update_component_state(component_id, 'active')
                    
                    # Update metrics
                    metrics = self.component_metrics[component_id]
                    metrics['update_count'] += 1
                    metrics['total_update_time'] += time.time() - start_time
                    metrics['last_update'] = time.time()
                    
                    return {
                        'success': True,
                        'component_state': component.copy()
                    }
                else:
                    component['error'] = render_result.get('error')
                    self._update_component_state(component_id, 'error')
                    
                    # Update error metrics
                    metrics = self.component_metrics[component_id]
                    metrics['error_count'] += 1
                    
                    return {
                        'success': False,
                        'error': render_result.get('error')
                    }
                    
        except (ValueError, RuntimeError):
            raise  # Re-raise our own hard failures without double-wrapping
        except Exception as e:
            self.logger.error(f"Single component update failed: {e}")
            raise RuntimeError(f"HARD FAILURE: Single component update failed: {e}")
    
    def _render_component(self, component_id: str) -> Dict[str, Any]:
        """Render component based on type and data"""
        try:
            start_time = time.time()
            component = self.component_states[component_id]
            component_type = component['type']
            
            # Get renderer for component type
            if component_type == 'chart':
                content = self._render_chart_component(component)
            elif component_type == 'metric':
                content = self._render_metric_component(component)
            elif component_type == 'alert':
                content = self._render_alert_component(component)
            elif component_type == 'table':
                content = self._render_table_component(component)
            elif component_type == 'gauge':
                content = self._render_gauge_component(component)
            elif component_type == 'heatmap':
                content = self._render_heatmap_component(component)
            elif component_type == 'timeline':
                content = self._render_timeline_component(component)
            elif component_type == 'log_viewer':
                content = self._render_log_viewer_component(component)
            elif component_type == 'control_panel':
                content = self._render_control_panel_component(component)
            elif component_type == 'status_indicator':
                content = self._render_status_indicator_component(component)
            else:
                content = self._render_generic_component(component)
            
            # Update metrics
            metrics = self.component_metrics[component_id]
            metrics['render_count'] += 1
            metrics['total_render_time'] += time.time() - start_time
            metrics['last_render'] = time.time()
            
            # Record render in lifecycle
            self.component_lifecycle[component_id]['render_history'].append({
                'timestamp': time.time(),
                'duration': time.time() - start_time,
                'success': True
            })
            
            return {
                'success': True,
                'content': content
            }
            
        except (ValueError, RuntimeError):
            raise  # Re-raise our own hard failures without double-wrapping
        except Exception as e:
            self.logger.error(f"Component rendering failed: {e}")
            
            # Record error in lifecycle
            if component_id in self.component_lifecycle:
                self.component_lifecycle[component_id]['error_history'].append({
                    'timestamp': time.time(),
                    'error': str(e),
                    'type': 'render_error'
                })
            
            raise RuntimeError(f"HARD FAILURE: Component rendering failed: {e}")
    
    def _render_chart_component(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Render chart component"""
        config = component['configuration']
        data = component['data'] or {}
        
        return {
            'type': 'chart',
            'chart_type': config.get('chart_type', 'line'),
            'data': {
                'datasets': data.get('datasets', []),
                'labels': data.get('labels', [])
            },
            'options': {
                'title': config.get('title', 'Chart'),
                'responsive': True,
                'maintainAspectRatio': False,
                'legend': {'display': config.get('show_legend', True)},
                'scales': config.get('scales', {})
            },
            'style': {
                'height': config.get('height', '300px'),
                'width': config.get('width', '100%')
            }
        }
    
    def _render_metric_component(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Render metric component"""
        config = component['configuration']
        data = component['data'] or {}
        
        value = data.get('value', 0)
        previous_value = data.get('previous_value')
        
        # Calculate change
        change = None
        if previous_value is not None and previous_value != 0:
            change = ((value - previous_value) / previous_value) * 100
        
        return {
            'type': 'metric',
            'title': config.get('title', 'Metric'),
            'value': value,
            'unit': config.get('unit', ''),
            'format': config.get('format', 'number'),
            'change': change,
            'change_type': 'increase' if change and change > 0 else 'decrease' if change and change < 0 else 'stable',
            'icon': config.get('icon'),
            'color': config.get('color', 'primary'),
            'style': {
                'text_size': config.get('text_size', 'large'),
                'align': config.get('align', 'center')
            }
        }
    
    def _render_alert_component(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Render alert component"""
        config = component['configuration']
        data = component['data'] or {}
        
        alerts = data.get('alerts', [])
        
        return {
            'type': 'alert',
            'alerts': [
                {
                    'id': alert.get('id'),
                    'severity': alert.get('severity', 'info'),
                    'title': alert.get('title', 'Alert'),
                    'message': alert.get('message', ''),
                    'timestamp': alert.get('timestamp'),
                    'actions': alert.get('actions', [])
                }
                for alert in alerts
            ],
            'max_visible': config.get('max_visible', 5),
            'auto_dismiss': config.get('auto_dismiss', False),
            'dismiss_after': config.get('dismiss_after', 5000),
            'style': {
                'position': config.get('position', 'top-right'),
                'animation': config.get('animation', 'slide')
            }
        }
    
    def _render_table_component(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Render table component"""
        config = component['configuration']
        data = component['data'] or {}
        
        return {
            'type': 'table',
            'columns': data.get('columns', []),
            'rows': data.get('rows', []),
            'pagination': {
                'enabled': config.get('pagination', True),
                'page_size': config.get('page_size', 10),
                'current_page': data.get('current_page', 1)
            },
            'sorting': {
                'enabled': config.get('sortable', True),
                'column': data.get('sort_column'),
                'direction': data.get('sort_direction', 'asc')
            },
            'filtering': {
                'enabled': config.get('filterable', True),
                'filters': data.get('filters', [])
            },
            'style': {
                'striped': config.get('striped', True),
                'hover': config.get('hover', True),
                'bordered': config.get('bordered', True),
                'size': config.get('size', 'normal')
            }
        }
    
    def _render_gauge_component(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Render gauge component"""
        config = component['configuration']
        data = component['data'] or {}
        
        value = data.get('value', 0)
        min_value = config.get('min', 0)
        max_value = config.get('max', 100)
        
        # Validate all values are numeric
        try:
            value = float(value)
            min_value = float(min_value)
            max_value = float(max_value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"HARD FAILURE: Gauge component requires numeric values. Got value={type(value).__name__}, min={type(min_value).__name__}, max={type(max_value).__name__}: {e}")
        
        if max_value <= min_value:
            raise ValueError(f"HARD FAILURE: Gauge max_value ({max_value}) must be greater than min_value ({min_value})")
        
        # Calculate percentage
        percentage = ((value - min_value) / (max_value - min_value)) * 100
        percentage = max(0, min(100, percentage))  # Clamp to 0-100
        
        return {
            'type': 'gauge',
            'title': config.get('title', 'Gauge'),
            'value': value,
            'percentage': percentage,
            'min': min_value,
            'max': max_value,
            'thresholds': config.get('thresholds', [
                {'value': 0, 'color': 'green'},
                {'value': 70, 'color': 'yellow'},
                {'value': 90, 'color': 'red'}
            ]),
            'unit': config.get('unit', ''),
            'style': {
                'type': config.get('gauge_type', 'arc'),
                'size': config.get('size', 'medium'),
                'show_value': config.get('show_value', True),
                'animate': config.get('animate', True)
            }
        }
    
    def _render_heatmap_component(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Render heatmap component"""
        config = component['configuration']
        data = component['data'] or {}
        
        return {
            'type': 'heatmap',
            'title': config.get('title', 'Heatmap'),
            'data': data.get('matrix', []),
            'x_labels': data.get('x_labels', []),
            'y_labels': data.get('y_labels', []),
            'color_scale': config.get('color_scale', {
                'min': 'blue',
                'mid': 'yellow',
                'max': 'red'
            }),
            'show_values': config.get('show_values', True),
            'style': {
                'cell_size': config.get('cell_size', 'auto'),
                'borders': config.get('show_borders', True),
                'tooltip': config.get('show_tooltip', True)
            }
        }
    
    def _render_timeline_component(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Render timeline component"""
        config = component['configuration']
        data = component['data'] or {}
        
        return {
            'type': 'timeline',
            'title': config.get('title', 'Timeline'),
            'events': data.get('events', []),
            'time_range': {
                'start': data.get('start_time'),
                'end': data.get('end_time'),
                'zoom_level': config.get('zoom_level', 'auto')
            },
            'grouping': {
                'enabled': config.get('group_events', False),
                'by': config.get('group_by', 'category')
            },
            'style': {
                'orientation': config.get('orientation', 'horizontal'),
                'event_height': config.get('event_height', 'auto'),
                'show_labels': config.get('show_labels', True)
            }
        }
    
    def _render_log_viewer_component(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Render log viewer component"""
        config = component['configuration']
        data = component['data'] or {}
        
        return {
            'type': 'log_viewer',
            'title': config.get('title', 'Logs'),
            'logs': data.get('logs', []),
            'filters': {
                'level': config.get('level_filter', 'all'),
                'search': data.get('search_term', ''),
                'time_range': data.get('time_range')
            },
            'display': {
                'show_timestamp': config.get('show_timestamp', True),
                'show_level': config.get('show_level', True),
                'wrap_text': config.get('wrap_text', True),
                'max_lines': config.get('max_lines', 1000)
            },
            'auto_scroll': config.get('auto_scroll', True),
            'style': {
                'font': config.get('font', 'monospace'),
                'line_height': config.get('line_height', 'normal'),
                'color_code': config.get('color_code_levels', True)
            }
        }
    
    def _render_control_panel_component(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Render control panel component"""
        config = component['configuration']
        data = component['data'] or {}
        
        return {
            'type': 'control_panel',
            'title': config.get('title', 'Controls'),
            'controls': data.get('controls', []),
            'layout': {
                'type': config.get('layout_type', 'grid'),
                'columns': config.get('columns', 2),
                'spacing': config.get('spacing', 'normal')
            },
            'validation': config.get('validation_rules', {}),
            'submit_action': config.get('submit_action'),
            'style': {
                'border': config.get('show_border', True),
                'padding': config.get('padding', 'normal'),
                'background': config.get('background_color')
            }
        }
    
    def _render_status_indicator_component(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Render status indicator component"""
        config = component['configuration']
        data = component['data'] or {}
        
        status = data.get('status', 'unknown')
        
        # Map status to color
        status_colors = {
            'healthy': 'green',
            'warning': 'yellow',
            'error': 'red',
            'unknown': 'gray',
            'active': 'green',
            'inactive': 'gray'
        }
        
        return {
            'type': 'status_indicator',
            'title': config.get('title', 'Status'),
            'status': status,
            'color': status_colors.get(status, config.get('color', 'gray')),
            'message': data.get('message', ''),
            'details': data.get('details', {}),
            'icon': config.get('icon'),
            'pulse': config.get('pulse', status == 'active'),
            'style': {
                'size': config.get('size', 'medium'),
                'shape': config.get('shape', 'circle'),
                'show_label': config.get('show_label', True)
            }
        }
    
    def _render_generic_component(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Render generic component"""
        return {
            'type': component['type'],
            'id': component['id'],
            'configuration': component['configuration'],
            'data': component['data'],
            'rendered_at': time.time()
        }
    
    def _update_component_state(self, component_id: str, new_state: str):
        """Update component state and track in lifecycle"""
        try:
            component = self.component_states[component_id]
            old_state = component['state']
            component['state'] = new_state
            
            # Track state change
            lifecycle = self.component_lifecycle[component_id]
            lifecycle['state_changes'].append((new_state, time.time()))
            
            # Enforce state_changes limit immediately
            if len(lifecycle['state_changes']) > 100:
                lifecycle['state_changes'] = lifecycle['state_changes'][-100:]
            
            self.logger.debug(f"Component {component_id} state: {old_state} -> {new_state}")
            
        except Exception as e:
            self.logger.error(f"State update failed: {e}")
            raise RuntimeError(f"HARD FAILURE: Component state update failed: {e}")
    
    def _process_cascading_updates(self, initial_components: List[str]) -> Dict[str, Any]:
        """Process cascading updates for dependent components"""
        try:
            cascaded_updates = {}
            processed = set(initial_components)
            to_process = list(initial_components)
            
            while to_process:
                component_id = to_process.pop(0)
                
                # Find components that depend on this one
                for cid, deps in self.component_dependencies.items():
                    if component_id in deps and cid not in processed:
                        # Update dependent component
                        if cid in self.component_states:
                            render_result = self._render_component(cid)
                            if render_result['success']:
                                self.component_states[cid]['rendered_content'] = render_result['content']
                                cascaded_updates[cid] = {
                                    'state': self.component_states[cid]['state'],
                                    'rendered_content': render_result['content'],
                                    'last_updated': time.time()
                                }
                            
                            processed.add(cid)
                            to_process.append(cid)
            
            return cascaded_updates
            
        except Exception as e:
            self.logger.error(f"Cascading update processing failed: {e}")
            raise RuntimeError(f"HARD FAILURE: Cascading update processing failed: {e}")
    
    def _queue_component_update(self, component_id: str):
        """Queue component for batched update"""
        try:
            # Cancel existing timer if present
            if component_id in self._update_timers:
                self._update_timers[component_id].cancel()
            
            # Set new timer for debounced update
            timer = threading.Timer(
                self.update_debounce_ms / 1000.0,
                self._execute_component_update,
                args=[component_id]
            )
            
            self._update_timers[component_id] = timer
            timer.start()
            
        except Exception as e:
            self.logger.error(f"Update queueing failed: {e}")
            raise RuntimeError(f"HARD FAILURE: Update queueing failed: {e}")
    
    def _execute_component_update(self, component_id: str):
        """Execute queued component update"""
        try:
            with self._lock:
                if component_id in self.component_states:
                    render_result = self._render_component(component_id)
                    if render_result['success']:
                        self.component_states[component_id]['rendered_content'] = render_result['content']
                
                # Remove timer
                self._update_timers.pop(component_id, None)
                
        except Exception as e:
            self.logger.error(f"Component update execution failed: {e}")
            raise RuntimeError(f"HARD FAILURE: Component update execution failed: {e}")
    
    def _generate_component_id(self, component_type: str) -> str:
        """Generate unique component ID"""
        timestamp = int(time.time() * 1000)
        random_suffix = hashlib.md5(f"{timestamp}{component_type}".encode()).hexdigest()[:8]
        return f"{component_type}_{timestamp}_{random_suffix}"
    
    def _generate_metrics_summary(self) -> Dict[str, Any]:
        """Generate summary of all component metrics"""
        try:
            total_renders = sum(m['render_count'] for m in self.component_metrics.values())
            total_updates = sum(m['update_count'] for m in self.component_metrics.values())
            total_errors = sum(m['error_count'] for m in self.component_metrics.values())
            
            avg_render_time = 0
            avg_update_time = 0
            
            if total_renders > 0:
                total_render_time = sum(m['total_render_time'] for m in self.component_metrics.values())
                avg_render_time = total_render_time / total_renders
            
            if total_updates > 0:
                total_update_time = sum(m['total_update_time'] for m in self.component_metrics.values())
                avg_update_time = total_update_time / total_updates
            
            return {
                'total_renders': total_renders,
                'total_updates': total_updates,
                'total_errors': total_errors,
                'average_render_time': avg_render_time,
                'average_update_time': avg_update_time,
                'error_rate': total_errors / (total_renders + total_updates) if (total_renders + total_updates) > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Metrics summary generation failed: {e}")
            raise RuntimeError(f"HARD FAILURE: Metrics summary generation failed: {e}")
    
    def _start_background_threads(self):
        """Start background maintenance threads"""
        # Component health check thread
        health_thread = threading.Thread(
            target=self._component_health_check_loop,
            daemon=True
        )
        health_thread.start()
        
        # Metrics cleanup thread
        cleanup_thread = threading.Thread(
            target=self._metrics_cleanup_loop,
            daemon=True
        )
        cleanup_thread.start()
    
    def _component_health_check_loop(self):
        """Monitor component health"""
        while True:
            try:
                current_time = time.time()
                
                with self._lock:
                    for component_id, component in self.component_states.items():
                        # Check for stale components
                        if component['last_updated']:
                            time_since_update = current_time - component['last_updated']
                            
                            if time_since_update > self.component_timeout_seconds:
                                self._update_component_state(component_id, 'stale')
                                self.logger.warning(
                                    f"Component {component_id} marked as stale "
                                    f"(no updates for {time_since_update:.1f}s)"
                                )
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Component health check error: {e}")
                raise RuntimeError(f"HARD FAILURE: Component health check failed: {e}")
    
    def _metrics_cleanup_loop(self):
        """Clean up old metrics data"""
        while True:
            try:
                # Clean up old lifecycle data
                with self._lock:
                    for component_id, lifecycle in self.component_lifecycle.items():
                        # Keep only recent state changes
                        if len(lifecycle['state_changes']) > 100:
                            lifecycle['state_changes'] = lifecycle['state_changes'][-100:]
                
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Metrics cleanup error: {e}")
                raise RuntimeError(f"HARD FAILURE: Metrics cleanup failed: {e}")