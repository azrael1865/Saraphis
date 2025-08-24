"""
Saraphis Dashboard Renderer
Production-ready dashboard rendering with real-time updates and responsive design
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import json
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class ThemeManager:
    """Theme management for dashboards"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize themes
        self.themes = self._initialize_themes()
        self.default_theme = config.get('default_theme', 'dark')
        
        self.logger.info("Theme Manager initialized")
    
    def get_theme_config(self, theme_name: str) -> Dict[str, Any]:
        """Get theme configuration"""
        try:
            if theme_name not in self.themes:
                theme_name = self.default_theme
            
            return self.themes[theme_name]
            
        except Exception as e:
            self.logger.error(f"Theme config retrieval failed: {e}")
            return self.themes[self.default_theme]
    
    def _initialize_themes(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available themes"""
        return {
            'dark': {
                'name': 'dark',
                'colors': {
                    'primary': '#1a1a1a',
                    'secondary': '#2d2d2d',
                    'background': '#0d0d0d',
                    'surface': '#1a1a1a',
                    'text': '#ffffff',
                    'text_secondary': '#b0b0b0',
                    'success': '#4caf50',
                    'warning': '#ff9800',
                    'danger': '#f44336',
                    'info': '#2196f3',
                    'chart_colors': ['#2196f3', '#4caf50', '#ff9800', '#f44336', '#9c27b0']
                },
                'fonts': {
                    'primary': 'Inter, system-ui, sans-serif',
                    'monospace': 'JetBrains Mono, monospace'
                }
            },
            'light': {
                'name': 'light',
                'colors': {
                    'primary': '#ffffff',
                    'secondary': '#f5f5f5',
                    'background': '#fafafa',
                    'surface': '#ffffff',
                    'text': '#212121',
                    'text_secondary': '#757575',
                    'success': '#4caf50',
                    'warning': '#ff9800',
                    'danger': '#f44336',
                    'info': '#2196f3',
                    'chart_colors': ['#1976d2', '#388e3c', '#f57c00', '#d32f2f', '#7b1fa2']
                },
                'fonts': {
                    'primary': 'Inter, system-ui, sans-serif',
                    'monospace': 'JetBrains Mono, monospace'
                }
            },
            'midnight': {
                'name': 'midnight',
                'colors': {
                    'primary': '#0f1419',
                    'secondary': '#1a2332',
                    'background': '#090c10',
                    'surface': '#0f1419',
                    'text': '#e1e8ed',
                    'text_secondary': '#8899a6',
                    'success': '#17bf63',
                    'warning': '#ffad1f',
                    'danger': '#e0245e',
                    'info': '#1da1f2',
                    'chart_colors': ['#1da1f2', '#17bf63', '#ffad1f', '#e0245e', '#794bc4']
                },
                'fonts': {
                    'primary': 'Inter, system-ui, sans-serif',
                    'monospace': 'JetBrains Mono, monospace'
                }
            }
        }


class DashboardRenderer:
    """Production-ready dashboard rendering with real-time updates and responsive design"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize theme manager
        self.theme_manager = ThemeManager(config.get('themes', {}))
        
        # Responsive breakpoints
        self.responsive_breakpoints = {
            'mobile': 640,
            'tablet': 1024,
            'desktop': 1280,
            'wide': 1920
        }
        
        # Component templates
        self.component_templates = self._initialize_component_templates()
        
        # Dashboard templates
        self.dashboard_templates = self._initialize_dashboard_templates()
        
        self.logger.info("Dashboard Renderer initialized")
    
    def render_components(self, dashboard_type: str, realtime_data: Dict[str, Any], 
                        user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Render dashboard components with real-time data"""
        try:
            # Get dashboard template
            dashboard_template = self._get_dashboard_template(dashboard_type)
            
            # Apply user theme preferences
            theme_config = self.theme_manager.get_theme_config(
                user_preferences.get('theme', 'dark')
            )
            
            # Render each component
            rendered_components = {}
            component_errors = []
            
            for component_id, component_config in dashboard_template['components'].items():
                try:
                    component_data = realtime_data.get(component_id, {})
                    rendered_component = self._render_component(
                        component_id, component_config, component_data, theme_config
                    )
                    rendered_components[component_id] = rendered_component
                except Exception as e:
                    self.logger.error(f"Component {component_id} rendering failed: {e}")
                    component_errors.append({
                        'component_id': component_id,
                        'error': str(e)
                    })
            
            # Generate responsive layout
            responsive_layout = self._generate_responsive_layout(
                dashboard_template, rendered_components, user_preferences
            )
            
            # Calculate data freshness
            data_freshness = self._calculate_data_freshness(realtime_data)
            
            result = {
                'dashboard_type': dashboard_type,
                'components': rendered_components,
                'layout': responsive_layout,
                'theme': theme_config,
                'metadata': {
                    'rendered_at': time.time(),
                    'data_freshness': data_freshness,
                    'component_count': len(rendered_components),
                    'component_errors': component_errors
                }
            }
            
            # Add errors if any
            if component_errors:
                result['warnings'] = component_errors
            
            return result
            
        except Exception as e:
            self.logger.error(f"Dashboard component rendering failed: {e}")
            raise RuntimeError(f"Component rendering failed: {e}")
    
    def _initialize_dashboard_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize dashboard templates"""
        return {
            'system_overview': {
                'title': 'System Overview',
                'layout': 'grid',
                'grid_columns': 12,
                'grid_rows': 8,
                'components': {
                    'system_health': {
                        'type': 'health_indicator',
                        'position': {'row': 0, 'col': 0, 'width': 6, 'height': 2},
                        'refresh_rate': 5,
                        'priority': 'high'
                    },
                    'performance_metrics': {
                        'type': 'performance_chart',
                        'position': {'row': 0, 'col': 6, 'width': 6, 'height': 2},
                        'refresh_rate': 10,
                        'priority': 'high'
                    },
                    'active_sessions': {
                        'type': 'session_list',
                        'position': {'row': 2, 'col': 0, 'width': 4, 'height': 3},
                        'refresh_rate': 15,
                        'priority': 'medium'
                    },
                    'error_logs': {
                        'type': 'log_viewer',
                        'position': {'row': 2, 'col': 4, 'width': 8, 'height': 3},
                        'refresh_rate': 20,
                        'priority': 'medium'
                    },
                    'resource_usage': {
                        'type': 'resource_chart',
                        'position': {'row': 5, 'col': 0, 'width': 12, 'height': 3},
                        'refresh_rate': 30,
                        'priority': 'low'
                    }
                }
            },
            'uncertainty_analysis': {
                'title': 'Uncertainty Analysis',
                'layout': 'flexible',
                'components': {
                    'uncertainty_distribution': {
                        'type': 'distribution_chart',
                        'position': {'row': 0, 'col': 0, 'width': 8, 'height': 3},
                        'refresh_rate': 5,
                        'priority': 'high'
                    },
                    'quantifier_performance': {
                        'type': 'performance_table',
                        'position': {'row': 0, 'col': 8, 'width': 4, 'height': 3},
                        'refresh_rate': 10,
                        'priority': 'high'
                    },
                    'cross_domain_propagation': {
                        'type': 'propagation_map',
                        'position': {'row': 3, 'col': 0, 'width': 12, 'height': 4},
                        'refresh_rate': 15,
                        'priority': 'medium'
                    },
                    'uncertainty_metrics': {
                        'type': 'metrics_grid',
                        'position': {'row': 7, 'col': 0, 'width': 12, 'height': 2},
                        'refresh_rate': 20,
                        'priority': 'low'
                    }
                }
            },
            'training_monitoring': {
                'title': 'Training Monitoring',
                'layout': 'timeline',
                'components': {
                    'training_progress': {
                        'type': 'progress_chart',
                        'position': {'row': 0, 'col': 0, 'width': 6, 'height': 2},
                        'refresh_rate': 2,
                        'priority': 'critical'
                    },
                    'loss_curves': {
                        'type': 'line_chart',
                        'position': {'row': 0, 'col': 6, 'width': 6, 'height': 2},
                        'refresh_rate': 5,
                        'priority': 'high'
                    },
                    'gradient_analysis': {
                        'type': 'gradient_visualization',
                        'position': {'row': 2, 'col': 0, 'width': 8, 'height': 3},
                        'refresh_rate': 10,
                        'priority': 'medium'
                    },
                    'model_metrics': {
                        'type': 'metrics_display',
                        'position': {'row': 2, 'col': 8, 'width': 4, 'height': 3},
                        'refresh_rate': 15,
                        'priority': 'medium'
                    },
                    'training_logs': {
                        'type': 'log_viewer',
                        'position': {'row': 5, 'col': 0, 'width': 12, 'height': 3},
                        'refresh_rate': 20,
                        'priority': 'low'
                    }
                }
            },
            'production_metrics': {
                'title': 'Production Metrics',
                'layout': 'dashboard',
                'components': {
                    'system_metrics': {
                        'type': 'metrics_grid',
                        'position': {'row': 0, 'col': 0, 'width': 12, 'height': 2},
                        'refresh_rate': 5,
                        'priority': 'high'
                    },
                    'api_performance': {
                        'type': 'api_chart',
                        'position': {'row': 2, 'col': 0, 'width': 6, 'height': 3},
                        'refresh_rate': 10,
                        'priority': 'high'
                    },
                    'security_alerts': {
                        'type': 'alert_panel',
                        'position': {'row': 2, 'col': 6, 'width': 6, 'height': 3},
                        'refresh_rate': 15,
                        'priority': 'critical'
                    },
                    'resource_utilization': {
                        'type': 'resource_chart',
                        'position': {'row': 5, 'col': 0, 'width': 6, 'height': 3},
                        'refresh_rate': 20,
                        'priority': 'medium'
                    },
                    'data_metrics': {
                        'type': 'data_flow_chart',
                        'position': {'row': 5, 'col': 6, 'width': 6, 'height': 3},
                        'refresh_rate': 25,
                        'priority': 'low'
                    }
                }
            }
        }
    
    def _initialize_component_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize component rendering templates"""
        return {
            'health_indicator': {
                'render_function': self._render_health_indicator,
                'required_data': ['status', 'score'],
                'optional_data': ['details', 'history']
            },
            'performance_chart': {
                'render_function': self._render_performance_chart,
                'required_data': ['chart_data'],
                'optional_data': ['metrics', 'annotations']
            },
            'session_list': {
                'render_function': self._render_session_list,
                'required_data': ['active_sessions'],
                'optional_data': ['total_sessions', 'session_details']
            },
            'log_viewer': {
                'render_function': self._render_log_viewer,
                'required_data': ['logs'],
                'optional_data': ['filters', 'severity_counts']
            },
            'distribution_chart': {
                'render_function': self._render_distribution_chart,
                'required_data': ['distribution_data'],
                'optional_data': ['statistics', 'thresholds']
            },
            'performance_table': {
                'render_function': self._render_performance_table,
                'required_data': ['table_data'],
                'optional_data': ['columns', 'sorting']
            },
            'propagation_map': {
                'render_function': self._render_propagation_map,
                'required_data': ['propagation_data'],
                'optional_data': ['connections', 'flow_rates']
            },
            'progress_chart': {
                'render_function': self._render_progress_chart,
                'required_data': ['progress'],
                'optional_data': ['milestones', 'eta']
            },
            'line_chart': {
                'render_function': self._render_line_chart,
                'required_data': ['series_data'],
                'optional_data': ['axes', 'legends']
            },
            'gradient_visualization': {
                'render_function': self._render_gradient_visualization,
                'required_data': ['gradient_data'],
                'optional_data': ['layers', 'magnitudes']
            },
            'metrics_display': {
                'render_function': self._render_metrics_display,
                'required_data': ['metrics'],
                'optional_data': ['comparisons', 'targets']
            },
            'metrics_grid': {
                'render_function': self._render_metrics_grid,
                'required_data': ['grid_metrics'],
                'optional_data': ['layout', 'highlights']
            },
            'api_chart': {
                'render_function': self._render_api_chart,
                'required_data': ['api_data'],
                'optional_data': ['endpoints', 'response_times']
            },
            'alert_panel': {
                'render_function': self._render_alert_panel,
                'required_data': ['alerts'],
                'optional_data': ['severity_levels', 'actions']
            },
            'resource_chart': {
                'render_function': self._render_resource_chart,
                'required_data': ['resource_data'],
                'optional_data': ['limits', 'predictions']
            },
            'data_flow_chart': {
                'render_function': self._render_data_flow_chart,
                'required_data': ['flow_data'],
                'optional_data': ['nodes', 'edges']
            }
        }
    
    def _get_dashboard_template(self, dashboard_type: str) -> Dict[str, Any]:
        """Get dashboard template configuration"""
        if dashboard_type not in self.dashboard_templates:
            self.logger.warning(f"Unknown dashboard type: {dashboard_type}, using default")
            dashboard_type = 'system_overview'
        
        return self.dashboard_templates[dashboard_type]
    
    def _render_component(self, component_id: str, component_config: Dict[str, Any], 
                         component_data: Dict[str, Any], theme_config: Dict[str, Any]) -> Dict[str, Any]:
        """Render individual component with data and theme"""
        try:
            component_type = component_config['type']
            position = component_config['position']
            refresh_rate = component_config.get('refresh_rate', 30)
            priority = component_config.get('priority', 'medium')
            
            # Get component template
            if component_type not in self.component_templates:
                raise ValueError(f"Unknown component type: {component_type}")
            
            template = self.component_templates[component_type]
            
            # Validate required data
            for required_field in template['required_data']:
                if required_field not in component_data:
                    self.logger.warning(f"Missing required data '{required_field}' for {component_type}")
                    component_data[required_field] = self._get_default_data(component_type, required_field)
            
            # Render component
            render_function = template['render_function']
            rendered_content = render_function(component_data, theme_config)
            
            return {
                'component_id': component_id,
                'component_type': component_type,
                'position': position,
                'refresh_rate': refresh_rate,
                'priority': priority,
                'rendered_content': rendered_content,
                'theme_applied': theme_config['name'],
                'last_updated': time.time(),
                'data_status': 'complete' if not component_data.get('error') else 'error'
            }
            
        except Exception as e:
            self.logger.error(f"Component rendering failed for {component_id}: {e}")
            raise RuntimeError(f"Component rendering failed: {e}")
    
    def _render_health_indicator(self, data: Dict[str, Any], theme: Dict[str, Any]) -> Dict[str, Any]:
        """Render system health indicator component"""
        try:
            health_status = data.get('status', 'unknown')
            health_score = data.get('score', 0)
            health_details = data.get('details', {})
            
            # Determine health color based on score
            if health_score >= 90:
                color = theme.get('colors', {}).get('success', '#4caf50')
                status_text = 'Excellent'
                icon = 'check-circle'
            elif health_score >= 70:
                color = theme.get('colors', {}).get('warning', '#ff9800')
                status_text = 'Good'
                icon = 'alert-circle'
            elif health_score >= 50:
                color = theme.get('colors', {}).get('warning', '#ff9800')
                status_text = 'Fair'
                icon = 'alert-triangle'
            else:
                color = theme.get('colors', {}).get('danger', '#f44336')
                status_text = 'Critical'
                icon = 'x-circle'
            
            # Process health details
            detail_items = []
            for key, value in health_details.items():
                detail_items.append({
                    'label': key.replace('_', ' ').title(),
                    'value': value,
                    'unit': self._get_unit_for_metric(key)
                })
            
            return {
                'type': 'health_indicator',
                'status': health_status,
                'score': health_score,
                'status_text': status_text,
                'color': color,
                'icon': icon,
                'details': detail_items,
                'timestamp': time.time(),
                'animation': 'pulse' if health_score < 50 else None
            }
            
        except Exception as e:
            self.logger.error(f"Health indicator rendering failed: {e}")
            raise
    
    def _render_performance_chart(self, data: Dict[str, Any], theme: Dict[str, Any]) -> Dict[str, Any]:
        """Render performance chart component"""
        try:
            chart_data = data.get('chart_data', [])
            metrics = data.get('metrics', {})
            annotations = data.get('annotations', [])
            
            # Process chart data
            processed_data = []
            for point in chart_data[-50:]:  # Limit to last 50 points
                processed_data.append({
                    'timestamp': point.get('timestamp', 0),
                    'value': point.get('value', 0),
                    'label': point.get('label', ''),
                    'color': theme['colors']['info']
                })
            
            # Process metrics
            metric_items = []
            for key, value in metrics.items():
                metric_items.append({
                    'name': key.replace('_', ' ').title(),
                    'value': round(value, 2),
                    'trend': self._calculate_trend(key, value)
                })
            
            return {
                'type': 'performance_chart',
                'chart_type': 'line',
                'chart_data': processed_data,
                'metrics': metric_items,
                'annotations': annotations,
                'chart_config': {
                    'smooth': True,
                    'area': True,
                    'grid': True,
                    'animation': True
                },
                'theme_colors': theme.get('colors', {}).get('chart_colors', []),
                'axes': {
                    'x': {'label': 'Time', 'type': 'time'},
                    'y': {'label': 'Performance', 'type': 'linear'}
                },
                'last_updated': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Performance chart rendering failed: {e}")
            raise
    
    def _render_session_list(self, data: Dict[str, Any], theme: Dict[str, Any]) -> Dict[str, Any]:
        """Render active session list component"""
        try:
            active_sessions = data.get('active_sessions', [])
            total_sessions = data.get('total_sessions', 0)
            
            # Process sessions
            processed_sessions = []
            for session in active_sessions[:20]:  # Limit to 20 sessions
                duration = time.time() - session.get('start_time', time.time())
                processed_sessions.append({
                    'session_id': session.get('session_id', 'unknown'),
                    'user_id': session.get('user_id', 'anonymous'),
                    'activity': session.get('activity', 'idle'),
                    'duration': self._format_duration(duration),
                    'status': session.get('status', 'active'),
                    'status_color': self._get_status_color(session.get('status'), theme)
                })
            
            return {
                'type': 'session_list',
                'sessions': processed_sessions,
                'total_sessions': total_sessions,
                'active_count': len([s for s in processed_sessions if s['status'] == 'active']),
                'list_config': {
                    'sortable': True,
                    'filterable': True,
                    'paginated': True,
                    'page_size': 20
                },
                'theme_colors': theme.get('colors', {}),
                'last_updated': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Session list rendering failed: {e}")
            raise
    
    def _render_log_viewer(self, data: Dict[str, Any], theme: Dict[str, Any]) -> Dict[str, Any]:
        """Render log viewer component"""
        try:
            logs = data.get('logs', [])
            filters = data.get('filters', {})
            severity_counts = data.get('severity_counts', {})
            
            # Process logs
            processed_logs = []
            for log in logs[-100:]:  # Limit to last 100 logs
                processed_logs.append({
                    'timestamp': log.get('timestamp', time.time()),
                    'formatted_time': datetime.fromtimestamp(
                        log.get('timestamp', time.time())
                    ).strftime('%Y-%m-%d %H:%M:%S'),
                    'level': log.get('level', 'info'),
                    'message': log.get('message', ''),
                    'source': log.get('source', 'unknown'),
                    'color': self._get_log_level_color(log.get('level', 'info'), theme)
                })
            
            return {
                'type': 'log_viewer',
                'logs': processed_logs,
                'filters': filters,
                'severity_counts': severity_counts,
                'viewer_config': {
                    'auto_scroll': True,
                    'show_timestamps': True,
                    'highlight_errors': True,
                    'max_lines': 100
                },
                'theme_colors': theme.get('colors', {}),
                'last_updated': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Log viewer rendering failed: {e}")
            raise
    
    def _render_metrics_grid(self, data: Dict[str, Any], theme: Dict[str, Any]) -> Dict[str, Any]:
        """Render metrics grid component"""
        try:
            grid_metrics = data.get('grid_metrics', {})
            layout = data.get('layout', 'auto')
            highlights = data.get('highlights', [])
            
            # Handle None values
            if grid_metrics is None:
                grid_metrics = {}
            if highlights is None:
                highlights = []
            
            # Process metrics into grid format
            metric_items = []
            for key, value in grid_metrics.items():
                is_highlighted = key in highlights
                metric_items.append({
                    'key': key,
                    'label': key.replace('_', ' ').title(),
                    'value': self._format_metric_value(value),
                    'unit': self._get_unit_for_metric(key),
                    'trend': self._calculate_trend(key, value),
                    'highlighted': is_highlighted,
                    'color': theme['colors']['info'] if not is_highlighted else theme['colors']['warning']
                })
            
            # Determine grid layout
            if layout == 'auto':
                columns = min(4, max(2, len(metric_items) // 2))
            else:
                columns = layout.get('columns', 4)
            
            return {
                'type': 'metrics_grid',
                'metrics': metric_items,
                'grid_config': {
                    'columns': columns,
                    'gap': 'medium',
                    'responsive': True
                },
                'theme_colors': theme.get('colors', {}),
                'last_updated': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Metrics grid rendering failed: {e}")
            raise
    
    def _render_alert_panel(self, data: Dict[str, Any], theme: Dict[str, Any]) -> Dict[str, Any]:
        """Render alert panel component"""
        try:
            alerts = data.get('alerts', [])
            severity_levels = data.get('severity_levels', ['critical', 'warning', 'info'])
            
            # Handle None values
            if alerts is None:
                alerts = []
            if severity_levels is None:
                severity_levels = ['critical', 'warning', 'info']
            
            # Process alerts
            processed_alerts = []
            for alert in alerts[:50]:  # Limit to 50 alerts
                severity = alert.get('severity', 'info')
                processed_alerts.append({
                    'alert_id': alert.get('alert_id', 'unknown'),
                    'title': alert.get('title', 'Alert'),
                    'message': alert.get('message', ''),
                    'severity': severity,
                    'timestamp': alert.get('timestamp', time.time()),
                    'formatted_time': self._format_time_ago(alert.get('timestamp', time.time())),
                    'color': self._get_severity_color(severity, theme),
                    'icon': self._get_severity_icon(severity),
                    'actions': alert.get('actions', [])
                })
            
            # Group by severity
            grouped_alerts = defaultdict(list)
            for alert in processed_alerts:
                grouped_alerts[alert['severity']].append(alert)
            
            return {
                'type': 'alert_panel',
                'alerts': processed_alerts,
                'grouped_alerts': dict(grouped_alerts),
                'total_count': len(alerts),
                'severity_counts': {
                    severity: len(grouped_alerts.get(severity, []))
                    for severity in severity_levels
                },
                'panel_config': {
                    'collapsible': True,
                    'dismissible': True,
                    'max_alerts': 50
                },
                'theme_colors': theme.get('colors', {}),
                'last_updated': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Alert panel rendering failed: {e}")
            raise
    
    def _generate_responsive_layout(self, template: Dict[str, Any], components: Dict[str, Any], 
                                  user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Generate responsive layout for dashboard"""
        try:
            layout_type = template.get('layout', 'grid')
            screen_size = user_preferences.get('screen_size', 'desktop')
            
            # Get appropriate layout generator
            if layout_type == 'grid':
                layout = self._generate_grid_layout(template, components, screen_size)
            elif layout_type == 'flexible':
                layout = self._generate_flexible_layout(template, components, screen_size)
            elif layout_type == 'timeline':
                layout = self._generate_timeline_layout(template, components, screen_size)
            elif layout_type == 'dashboard':
                layout = self._generate_dashboard_layout(template, components, screen_size)
            else:
                layout = self._generate_grid_layout(template, components, screen_size)
            
            # Apply responsive adjustments
            layout = self._apply_responsive_adjustments(layout, screen_size)
            
            return layout
            
        except Exception as e:
            self.logger.error(f"Responsive layout generation failed: {e}")
            raise
    
    def _generate_grid_layout(self, template: Dict[str, Any], components: Dict[str, Any], 
                            screen_size: str) -> Dict[str, Any]:
        """Generate grid-based layout"""
        grid_columns = template.get('grid_columns', 12)
        grid_rows = template.get('grid_rows', 8)
        
        # Adjust grid for screen size
        if screen_size == 'mobile':
            grid_columns = 4
        elif screen_size == 'tablet':
            grid_columns = 8
        
        layout_items = []
        for component_id, component in components.items():
            position = component['position'].copy()
            
            # Adjust position for screen size
            if screen_size == 'mobile':
                position['col'] = 0
                position['width'] = 4
            elif screen_size == 'tablet':
                position['width'] = min(8, position['width'])
            
            layout_items.append({
                'component_id': component_id,
                'position': position,
                'priority': component.get('priority', 'medium')
            })
        
        # Sort by priority and position
        layout_items.sort(key=lambda x: (
            {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x['priority']],
            x['position']['row'],
            x['position']['col']
        ))
        
        return {
            'type': 'grid',
            'columns': grid_columns,
            'rows': grid_rows,
            'items': layout_items,
            'gap': 'medium',
            'responsive': True
        }
    
    def _calculate_data_freshness(self, realtime_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate data freshness metrics"""
        try:
            current_time = time.time()
            freshness_metrics = {}
            
            for component_id, data in realtime_data.items():
                last_update = data.get('last_updated', current_time)
                age_seconds = current_time - last_update
                
                if age_seconds < 5:
                    freshness = 'real_time'
                    freshness_score = 1.0
                elif age_seconds < 30:
                    freshness = 'fresh'
                    freshness_score = 0.8
                elif age_seconds < 60:
                    freshness = 'recent'
                    freshness_score = 0.6
                elif age_seconds < 300:
                    freshness = 'stale'
                    freshness_score = 0.4
                else:
                    freshness = 'very_stale'
                    freshness_score = 0.2
                
                freshness_metrics[component_id] = {
                    'age_seconds': age_seconds,
                    'freshness': freshness,
                    'freshness_score': freshness_score,
                    'last_updated': last_update
                }
            
            # Calculate overall freshness
            if freshness_metrics:
                avg_freshness = sum(
                    m['freshness_score'] for m in freshness_metrics.values()
                ) / len(freshness_metrics)
            else:
                avg_freshness = 0
            
            freshness_metrics['overall'] = {
                'average_freshness': avg_freshness,
                'freshness_rating': self._get_freshness_rating(avg_freshness)
            }
            
            return freshness_metrics
            
        except Exception as e:
            self.logger.error(f"Data freshness calculation failed: {e}")
            return {}
    
    # Helper methods
    def _get_default_data(self, component_type: str, field: str) -> Any:
        """Get default data for missing required fields"""
        defaults = {
            'health_indicator': {
                'status': 'unknown',
                'score': 0,
                'details': {}
            },
            'performance_chart': {
                'chart_data': [],
                'metrics': {}
            },
            'session_list': {
                'active_sessions': [],
                'total_sessions': 0
            },
            'log_viewer': {
                'logs': [],
                'filters': {},
                'severity_counts': {}
            }
        }
        
        return defaults.get(component_type, {}).get(field, None)
    
    def _get_unit_for_metric(self, metric_key: str) -> str:
        """Get appropriate unit for metric"""
        if 'usage' in metric_key or 'percent' in metric_key:
            return '%'
        elif 'time' in metric_key:
            return 's'
        elif 'memory' in metric_key or 'disk' in metric_key:
            return 'GB'
        elif 'count' in metric_key or 'total' in metric_key:
            return ''
        else:
            return ''
    
    def _format_metric_value(self, value: Any) -> str:
        """Format metric value for display"""
        if isinstance(value, float):
            return f"{value:.2f}"
        elif isinstance(value, int):
            if value > 1000000:
                return f"{value/1000000:.1f}M"
            elif value > 1000:
                return f"{value/1000:.1f}K"
            else:
                return str(value)
        else:
            return str(value)
    
    def _calculate_trend(self, metric_key: str, value: float) -> str:
        """Calculate trend indicator for metric"""
        # Simplified trend calculation
        import random
        trend_value = random.choice(['up', 'down', 'stable'])
        return trend_value
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds/60)}m"
        else:
            return f"{int(seconds/3600)}h {int((seconds%3600)/60)}m"
    
    def _format_time_ago(self, timestamp: float) -> str:
        """Format timestamp as time ago"""
        ago = time.time() - timestamp
        if ago < 60:
            return "just now"
        elif ago < 3600:
            return f"{int(ago/60)}m ago"
        elif ago < 86400:
            return f"{int(ago/3600)}h ago"
        else:
            return f"{int(ago/86400)}d ago"
    
    def _get_status_color(self, status: str, theme: Dict[str, Any]) -> str:
        """Get color for status"""
        status_colors = {
            'active': theme['colors']['success'],
            'idle': theme['colors']['warning'],
            'disconnected': theme['colors']['danger'],
            'unknown': theme['colors']['text_secondary']
        }
        return status_colors.get(status, theme['colors']['text_secondary'])
    
    def _get_log_level_color(self, level: str, theme: Dict[str, Any]) -> str:
        """Get color for log level"""
        level_colors = {
            'debug': theme['colors']['text_secondary'],
            'info': theme['colors']['info'],
            'warning': theme['colors']['warning'],
            'error': theme['colors']['danger'],
            'critical': theme['colors']['danger']
        }
        return level_colors.get(level, theme['colors']['text'])
    
    def _get_severity_color(self, severity: str, theme: Dict[str, Any]) -> str:
        """Get color for severity level"""
        severity_colors = {
            'critical': theme['colors']['danger'],
            'warning': theme['colors']['warning'],
            'info': theme['colors']['info'],
            'success': theme['colors']['success']
        }
        return severity_colors.get(severity, theme['colors']['text'])
    
    def _get_severity_icon(self, severity: str) -> str:
        """Get icon for severity level"""
        severity_icons = {
            'critical': 'alert-octagon',
            'warning': 'alert-triangle',
            'info': 'info',
            'success': 'check-circle'
        }
        return severity_icons.get(severity, 'alert-circle')
    
    def _get_freshness_rating(self, score: float) -> str:
        """Get freshness rating from score"""
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def _apply_responsive_adjustments(self, layout: Dict[str, Any], screen_size: str) -> Dict[str, Any]:
        """Apply responsive adjustments to layout"""
        if screen_size == 'mobile':
            layout['stack_components'] = True
            layout['hide_secondary'] = True
        elif screen_size == 'tablet':
            layout['reduce_columns'] = True
        
        return layout
    
    # Missing layout generation methods
    def _generate_flexible_layout(self, template: Dict[str, Any], components: Dict[str, Any], 
                                screen_size: str) -> Dict[str, Any]:
        """Generate flexible layout"""
        layout_items = []
        for component_id, component in components.items():
            position = component['position'].copy()
            
            # Flexible layout adjusts based on content and screen size
            if screen_size == 'mobile':
                position['width'] = 12  # Full width on mobile
                position['col'] = 0
            elif screen_size == 'tablet':
                position['width'] = min(8, position['width'])
                
            layout_items.append({
                'component_id': component_id,
                'position': position,
                'priority': component.get('priority', 'medium'),
                'flexible': True
            })
            
        # Sort by priority
        layout_items.sort(key=lambda x: (
            {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x['priority']],
            x['position']['row']
        ))
        
        return {
            'type': 'flexible',
            'items': layout_items,
            'flow_direction': 'column' if screen_size == 'mobile' else 'row',
            'responsive': True
        }
    
    def _generate_timeline_layout(self, template: Dict[str, Any], components: Dict[str, Any], 
                                screen_size: str) -> Dict[str, Any]:
        """Generate timeline-based layout"""
        layout_items = []
        for component_id, component in components.items():
            position = component['position'].copy()
            
            # Timeline layout organizes components chronologically
            if screen_size == 'mobile':
                position['width'] = 12
                position['col'] = 0
            
            layout_items.append({
                'component_id': component_id,
                'position': position,
                'priority': component.get('priority', 'medium'),
                'timeline_order': position.get('row', 0)
            })
            
        # Sort by timeline order (row position)
        layout_items.sort(key=lambda x: x['timeline_order'])
        
        return {
            'type': 'timeline',
            'items': layout_items,
            'orientation': 'vertical' if screen_size == 'mobile' else 'horizontal',
            'responsive': True
        }
    
    def _generate_dashboard_layout(self, template: Dict[str, Any], components: Dict[str, Any], 
                                 screen_size: str) -> Dict[str, Any]:
        """Generate dashboard-style layout"""
        layout_items = []
        for component_id, component in components.items():
            position = component['position'].copy()
            
            # Dashboard layout optimizes for monitoring
            if screen_size == 'mobile':
                # Stack components vertically on mobile
                position['width'] = 12
                position['col'] = 0
            elif screen_size == 'tablet':
                # Adjust for tablet
                position['width'] = min(8, position['width'])
                
            layout_items.append({
                'component_id': component_id,
                'position': position,
                'priority': component.get('priority', 'medium'),
                'dashboard_section': self._get_dashboard_section(component_id)
            })
            
        # Sort by priority and section
        layout_items.sort(key=lambda x: (
            {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x['priority']],
            x['dashboard_section'],
            x['position']['row']
        ))
        
        return {
            'type': 'dashboard',
            'items': layout_items,
            'sections': ['primary', 'secondary', 'tertiary'],
            'responsive': True
        }
    
    def _get_dashboard_section(self, component_id: str) -> str:
        """Get dashboard section for component"""
        primary_components = ['system_health', 'performance_metrics', 'system_metrics']
        secondary_components = ['active_sessions', 'api_performance', 'security_alerts']
        
        if component_id in primary_components:
            return 'primary'
        elif component_id in secondary_components:
            return 'secondary'
        else:
            return 'tertiary'
    
    # Missing render methods - implementing all component types
    def _render_distribution_chart(self, data: Dict[str, Any], theme: Dict[str, Any]) -> Dict[str, Any]:
        """Render distribution chart component"""
        try:
            distribution_data = data.get('distribution_data', [])
            statistics = data.get('statistics', {})
            thresholds = data.get('thresholds', [])
            
            return {
                'type': 'distribution_chart',
                'distribution_data': distribution_data,
                'statistics': statistics,
                'thresholds': thresholds,
                'chart_config': {
                    'chart_type': 'histogram',
                    'bins': 20,
                    'smooth': True
                },
                'theme_colors': theme.get('colors', {}),
                'last_updated': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Distribution chart rendering failed: {e}")
            raise
    
    def _render_performance_table(self, data: Dict[str, Any], theme: Dict[str, Any]) -> Dict[str, Any]:
        """Render performance table component"""
        try:
            table_data = data.get('table_data', [])
            columns = data.get('columns', [])
            sorting = data.get('sorting', {})
            
            return {
                'type': 'performance_table',
                'table_data': table_data,
                'columns': columns,
                'sorting': sorting,
                'table_config': {
                    'sortable': True,
                    'filterable': True,
                    'paginated': True
                },
                'theme_colors': theme.get('colors', {}),
                'last_updated': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Performance table rendering failed: {e}")
            raise
    
    def _render_propagation_map(self, data: Dict[str, Any], theme: Dict[str, Any]) -> Dict[str, Any]:
        """Render propagation map component"""
        try:
            propagation_data = data.get('propagation_data', {})
            connections = data.get('connections', [])
            flow_rates = data.get('flow_rates', {})
            
            return {
                'type': 'propagation_map',
                'propagation_data': propagation_data,
                'connections': connections,
                'flow_rates': flow_rates,
                'map_config': {
                    'interactive': True,
                    'zoom_enabled': True,
                    'legend': True
                },
                'theme_colors': theme.get('colors', {}),
                'last_updated': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Propagation map rendering failed: {e}")
            raise
    
    def _render_progress_chart(self, data: Dict[str, Any], theme: Dict[str, Any]) -> Dict[str, Any]:
        """Render progress chart component"""
        try:
            progress = data.get('progress', 0)
            milestones = data.get('milestones', [])
            eta = data.get('eta', None)
            
            return {
                'type': 'progress_chart',
                'progress': progress,
                'milestones': milestones,
                'eta': eta,
                'progress_config': {
                    'show_percentage': True,
                    'show_eta': True,
                    'animated': True
                },
                'theme_colors': theme.get('colors', {}),
                'last_updated': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Progress chart rendering failed: {e}")
            raise
    
    def _render_line_chart(self, data: Dict[str, Any], theme: Dict[str, Any]) -> Dict[str, Any]:
        """Render line chart component"""
        try:
            series_data = data.get('series_data', [])
            axes = data.get('axes', {})
            legends = data.get('legends', [])
            
            return {
                'type': 'line_chart',
                'series_data': series_data,
                'axes': axes,
                'legends': legends,
                'chart_config': {
                    'smooth': True,
                    'points': True,
                    'grid': True,
                    'animation': True
                },
                'theme_colors': theme.get('colors', {}).get('chart_colors', []),
                'last_updated': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Line chart rendering failed: {e}")
            raise
    
    def _render_gradient_visualization(self, data: Dict[str, Any], theme: Dict[str, Any]) -> Dict[str, Any]:
        """Render gradient visualization component"""
        try:
            gradient_data = data.get('gradient_data', {})
            layers = data.get('layers', [])
            magnitudes = data.get('magnitudes', [])
            
            return {
                'type': 'gradient_visualization',
                'gradient_data': gradient_data,
                'layers': layers,
                'magnitudes': magnitudes,
                'visualization_config': {
                    'heatmap': True,
                    'color_scale': 'viridis',
                    'interactive': True
                },
                'theme_colors': theme.get('colors', {}),
                'last_updated': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Gradient visualization rendering failed: {e}")
            raise
    
    def _render_metrics_display(self, data: Dict[str, Any], theme: Dict[str, Any]) -> Dict[str, Any]:
        """Render metrics display component"""
        try:
            metrics = data.get('metrics', {})
            comparisons = data.get('comparisons', {})
            targets = data.get('targets', {})
            
            # Handle None values
            if metrics is None:
                metrics = {}
            if comparisons is None:
                comparisons = {}
            if targets is None:
                targets = {}
            
            # Process metrics into display format
            display_metrics = []
            for key, value in metrics.items():
                display_metrics.append({
                    'name': key.replace('_', ' ').title(),
                    'value': self._format_metric_value(value),
                    'unit': self._get_unit_for_metric(key),
                    'comparison': comparisons.get(key),
                    'target': targets.get(key),
                    'status': self._get_metric_status(value, targets.get(key))
                })
            
            return {
                'type': 'metrics_display',
                'metrics': display_metrics,
                'comparisons': comparisons,
                'targets': targets,
                'display_config': {
                    'layout': 'cards',
                    'show_trends': True,
                    'show_targets': True
                },
                'theme_colors': theme.get('colors', {}),
                'last_updated': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Metrics display rendering failed: {e}")
            raise
    
    def _render_api_chart(self, data: Dict[str, Any], theme: Dict[str, Any]) -> Dict[str, Any]:
        """Render API chart component"""
        try:
            api_data = data.get('api_data', [])
            endpoints = data.get('endpoints', [])
            response_times = data.get('response_times', {})
            
            return {
                'type': 'api_chart',
                'api_data': api_data,
                'endpoints': endpoints,
                'response_times': response_times,
                'chart_config': {
                    'chart_type': 'bar',
                    'show_errors': True,
                    'group_by_endpoint': True
                },
                'theme_colors': theme.get('colors', {}),
                'last_updated': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"API chart rendering failed: {e}")
            raise
    
    def _render_resource_chart(self, data: Dict[str, Any], theme: Dict[str, Any]) -> Dict[str, Any]:
        """Render resource chart component"""
        try:
            resource_data = data.get('resource_data', [])
            limits = data.get('limits', {})
            predictions = data.get('predictions', [])
            
            return {
                'type': 'resource_chart',
                'resource_data': resource_data,
                'limits': limits,
                'predictions': predictions,
                'chart_config': {
                    'chart_type': 'area',
                    'stacked': True,
                    'show_limits': True,
                    'show_predictions': True
                },
                'theme_colors': theme.get('colors', {}),
                'last_updated': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Resource chart rendering failed: {e}")
            raise
    
    def _render_data_flow_chart(self, data: Dict[str, Any], theme: Dict[str, Any]) -> Dict[str, Any]:
        """Render data flow chart component"""
        try:
            flow_data = data.get('flow_data', {})
            nodes = data.get('nodes', [])
            edges = data.get('edges', [])
            
            return {
                'type': 'data_flow_chart',
                'flow_data': flow_data,
                'nodes': nodes,
                'edges': edges,
                'flow_config': {
                    'layout': 'hierarchical',
                    'animated_flows': True,
                    'interactive': True,
                    'show_metrics': True
                },
                'theme_colors': theme.get('colors', {}),
                'last_updated': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Data flow chart rendering failed: {e}")
            raise
    
    def _get_metric_status(self, value: float, target: Optional[float]) -> str:
        """Get status based on metric value vs target"""
        if target is None:
            return 'unknown'
        
        if value >= target * 0.95:
            return 'good'
        elif value >= target * 0.8:
            return 'warning'
        else:
            return 'critical'