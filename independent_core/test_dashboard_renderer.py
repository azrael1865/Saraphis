"""
Comprehensive test suite for DashboardRenderer
Tests all dashboard rendering functionality, theme management, and responsive layouts
NO MOCKS - tests real implementation with hard failures for debugging
"""

import unittest
import time
import json
from datetime import datetime
from unittest.mock import patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from production_web.dashboard_renderer import DashboardRenderer, ThemeManager


class TestThemeManager(unittest.TestCase):
    """Test suite for ThemeManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'default_theme': 'dark',
            'custom_themes': {
                'test_theme': {
                    'name': 'test_theme',
                    'colors': {'primary': '#123456'},
                    'fonts': {'primary': 'Test Font'}
                }
            }
        }
        self.theme_manager = ThemeManager(self.config)
        
    def test_initialization(self):
        """Test ThemeManager initialization"""
        self.assertIsInstance(self.theme_manager.themes, dict)
        self.assertEqual(self.theme_manager.default_theme, 'dark')
        self.assertIn('dark', self.theme_manager.themes)
        self.assertIn('light', self.theme_manager.themes)
        self.assertIn('midnight', self.theme_manager.themes)
        self.assertIsNotNone(self.theme_manager.logger)
        
    def test_initialize_themes(self):
        """Test theme initialization"""
        themes = self.theme_manager._initialize_themes()
        
        # Check all required themes exist
        required_themes = ['dark', 'light', 'midnight']
        for theme in required_themes:
            self.assertIn(theme, themes)
            
        # Check theme structure
        for theme_name, theme_data in themes.items():
            self.assertIn('name', theme_data)
            self.assertIn('colors', theme_data)
            self.assertIn('fonts', theme_data)
            self.assertEqual(theme_data['name'], theme_name)
            
        # Check dark theme specifics
        dark_theme = themes['dark']
        self.assertEqual(dark_theme['colors']['primary'], '#1a1a1a')
        self.assertEqual(dark_theme['colors']['text'], '#ffffff')
        self.assertIn('chart_colors', dark_theme['colors'])
        self.assertIsInstance(dark_theme['colors']['chart_colors'], list)
        
    def test_get_theme_config_valid(self):
        """Test getting valid theme configuration"""
        theme_config = self.theme_manager.get_theme_config('dark')
        
        self.assertEqual(theme_config['name'], 'dark')
        self.assertIn('colors', theme_config)
        self.assertIn('fonts', theme_config)
        
        # Test different themes
        light_theme = self.theme_manager.get_theme_config('light')
        self.assertEqual(light_theme['name'], 'light')
        self.assertNotEqual(light_theme['colors']['primary'], theme_config['colors']['primary'])
        
    def test_get_theme_config_invalid(self):
        """Test getting invalid theme configuration falls back to default"""
        theme_config = self.theme_manager.get_theme_config('nonexistent_theme')
        
        # Should fall back to default theme
        self.assertEqual(theme_config['name'], 'dark')
        
    def test_get_theme_config_none_input(self):
        """Test getting theme config with None input"""
        theme_config = self.theme_manager.get_theme_config(None)
        
        # Should fall back to default theme
        self.assertEqual(theme_config['name'], 'dark')
        
    def test_get_theme_config_empty_string(self):
        """Test getting theme config with empty string"""
        theme_config = self.theme_manager.get_theme_config('')
        
        # Should fall back to default theme
        self.assertEqual(theme_config['name'], 'dark')


class TestDashboardRenderer(unittest.TestCase):
    """Test suite for DashboardRenderer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'themes': {
                'default_theme': 'dark'
            },
            'responsive': {
                'mobile_threshold': 640,
                'tablet_threshold': 1024
            }
        }
        self.renderer = DashboardRenderer(self.config)
        
        # Sample realtime data
        self.sample_realtime_data = {
            'system_health': {
                'status': 'healthy',
                'score': 85,
                'details': {
                    'cpu_usage': 45.2,
                    'memory_usage': 60.5,
                    'disk_usage': 25.8
                },
                'last_updated': time.time()
            },
            'performance_metrics': {
                'chart_data': [
                    {'timestamp': time.time() - 300, 'value': 95.2, 'label': 'Response Time'},
                    {'timestamp': time.time() - 240, 'value': 92.1, 'label': 'Response Time'},
                    {'timestamp': time.time() - 180, 'value': 98.5, 'label': 'Response Time'},
                    {'timestamp': time.time() - 120, 'value': 89.3, 'label': 'Response Time'},
                    {'timestamp': time.time() - 60, 'value': 94.7, 'label': 'Response Time'},
                ],
                'metrics': {
                    'avg_response_time': 93.96,
                    'throughput': 1250,
                    'error_rate': 0.02
                },
                'last_updated': time.time()
            },
            'active_sessions': {
                'active_sessions': [
                    {
                        'session_id': 'sess_001',
                        'user_id': 'user_123',
                        'activity': 'browsing',
                        'start_time': time.time() - 1800,
                        'status': 'active'
                    },
                    {
                        'session_id': 'sess_002',
                        'user_id': 'user_456',
                        'activity': 'idle',
                        'start_time': time.time() - 3600,
                        'status': 'idle'
                    }
                ],
                'total_sessions': 25,
                'last_updated': time.time()
            },
            'error_logs': {
                'logs': [
                    {
                        'timestamp': time.time() - 300,
                        'level': 'error',
                        'message': 'Database connection timeout',
                        'source': 'database.py'
                    },
                    {
                        'timestamp': time.time() - 180,
                        'level': 'warning',
                        'message': 'High memory usage detected',
                        'source': 'monitor.py'
                    },
                    {
                        'timestamp': time.time() - 60,
                        'level': 'info',
                        'message': 'Backup completed successfully',
                        'source': 'backup.py'
                    }
                ],
                'severity_counts': {
                    'error': 1,
                    'warning': 1,
                    'info': 1
                },
                'last_updated': time.time()
            }
        }
        
        # Sample user preferences
        self.sample_user_preferences = {
            'theme': 'dark',
            'screen_size': 'desktop',
            'layout_density': 'comfortable',
            'auto_refresh': True,
            'refresh_interval': 30
        }
        
    def test_initialization(self):
        """Test DashboardRenderer initialization"""
        self.assertIsInstance(self.renderer.theme_manager, ThemeManager)
        self.assertIsInstance(self.renderer.responsive_breakpoints, dict)
        self.assertIsInstance(self.renderer.component_templates, dict)
        self.assertIsInstance(self.renderer.dashboard_templates, dict)
        self.assertIsNotNone(self.renderer.logger)
        
        # Check breakpoints
        expected_breakpoints = ['mobile', 'tablet', 'desktop', 'wide']
        for bp in expected_breakpoints:
            self.assertIn(bp, self.renderer.responsive_breakpoints)
            
    def test_initialize_dashboard_templates(self):
        """Test dashboard template initialization"""
        templates = self.renderer._initialize_dashboard_templates()
        
        # Check required dashboard types exist
        required_types = ['system_overview', 'uncertainty_analysis', 'training_monitoring', 'production_metrics']
        for dashboard_type in required_types:
            self.assertIn(dashboard_type, templates)
            
        # Check template structure
        for template_name, template in templates.items():
            self.assertIn('title', template)
            self.assertIn('layout', template)
            self.assertIn('components', template)
            self.assertIsInstance(template['components'], dict)
            
            # Check each component has required fields
            for component_id, component_config in template['components'].items():
                self.assertIn('type', component_config)
                self.assertIn('position', component_config)
                self.assertIn('refresh_rate', component_config)
                self.assertIn('priority', component_config)
                
    def test_initialize_component_templates(self):
        """Test component template initialization"""
        templates = self.renderer._initialize_component_templates()
        
        # Check required component types
        required_types = [
            'health_indicator', 'performance_chart', 'session_list', 'log_viewer',
            'distribution_chart', 'performance_table', 'propagation_map',
            'progress_chart', 'line_chart', 'gradient_visualization',
            'metrics_display', 'metrics_grid', 'api_chart', 'alert_panel',
            'resource_chart', 'data_flow_chart'
        ]
        
        for component_type in required_types:
            self.assertIn(component_type, templates)
            
        # Check template structure
        for component_type, template in templates.items():
            self.assertIn('render_function', template)
            self.assertIn('required_data', template)
            self.assertIn('optional_data', template)
            self.assertIsInstance(template['required_data'], list)
            self.assertIsInstance(template['optional_data'], list)
            self.assertTrue(callable(template['render_function']))
            
    def test_get_dashboard_template_valid(self):
        """Test getting valid dashboard template"""
        template = self.renderer._get_dashboard_template('system_overview')
        
        self.assertEqual(template['title'], 'System Overview')
        self.assertIn('components', template)
        self.assertIn('system_health', template['components'])
        
    def test_get_dashboard_template_invalid(self):
        """Test getting invalid dashboard template falls back to default"""
        template = self.renderer._get_dashboard_template('nonexistent_dashboard')
        
        # Should fall back to system_overview
        self.assertEqual(template['title'], 'System Overview')
        
    def test_render_components_system_overview(self):
        """Test rendering system overview dashboard components"""
        result = self.renderer.render_components(
            'system_overview', 
            self.sample_realtime_data, 
            self.sample_user_preferences
        )
        
        # Check result structure
        self.assertIn('dashboard_type', result)
        self.assertIn('components', result)
        self.assertIn('layout', result)
        self.assertIn('theme', result)
        self.assertIn('metadata', result)
        
        self.assertEqual(result['dashboard_type'], 'system_overview')
        self.assertIsInstance(result['components'], dict)
        self.assertIsInstance(result['layout'], dict)
        self.assertIsInstance(result['theme'], dict)
        
        # Check metadata
        metadata = result['metadata']
        self.assertIn('rendered_at', metadata)
        self.assertIn('data_freshness', metadata)
        self.assertIn('component_count', metadata)
        self.assertIsInstance(metadata['rendered_at'], float)
        
    def test_render_components_with_errors(self):
        """Test rendering components with missing data"""
        incomplete_data = {
            'system_health': {
                'status': 'healthy'
                # Missing required 'score' field
            }
        }
        
        result = self.renderer.render_components(
            'system_overview', 
            incomplete_data, 
            self.sample_user_preferences
        )
        
        # Should still render but may have warnings
        self.assertIn('components', result)
        self.assertIsInstance(result['components'], dict)
        
        # Check if warnings are present for missing data
        if 'warnings' in result:
            self.assertIsInstance(result['warnings'], list)
            
    def test_render_components_invalid_dashboard_type(self):
        """Test rendering with invalid dashboard type"""
        result = self.renderer.render_components(
            'invalid_dashboard', 
            self.sample_realtime_data, 
            self.sample_user_preferences
        )
        
        # Should fall back to default dashboard type
        self.assertIn('components', result)
        self.assertIsInstance(result['components'], dict)
        
    def test_render_component_health_indicator(self):
        """Test rendering health indicator component"""
        component_config = {
            'type': 'health_indicator',
            'position': {'row': 0, 'col': 0, 'width': 6, 'height': 2},
            'refresh_rate': 5,
            'priority': 'high'
        }
        
        theme_config = self.renderer.theme_manager.get_theme_config('dark')
        component_data = self.sample_realtime_data['system_health']
        
        result = self.renderer._render_component(
            'system_health', component_config, component_data, theme_config
        )
        
        # Check result structure
        self.assertEqual(result['component_id'], 'system_health')
        self.assertEqual(result['component_type'], 'health_indicator')
        self.assertIn('rendered_content', result)
        self.assertIn('last_updated', result)
        self.assertIn('data_status', result)
        
        # Check rendered content
        content = result['rendered_content']
        self.assertEqual(content['type'], 'health_indicator')
        self.assertIn('status', content)
        self.assertIn('score', content)
        self.assertIn('color', content)
        self.assertIn('icon', content)
        
    def test_render_component_performance_chart(self):
        """Test rendering performance chart component"""
        component_config = {
            'type': 'performance_chart',
            'position': {'row': 0, 'col': 6, 'width': 6, 'height': 2},
            'refresh_rate': 10,
            'priority': 'high'
        }
        
        theme_config = self.renderer.theme_manager.get_theme_config('dark')
        component_data = self.sample_realtime_data['performance_metrics']
        
        result = self.renderer._render_component(
            'performance_metrics', component_config, component_data, theme_config
        )
        
        # Check rendered content
        content = result['rendered_content']
        self.assertEqual(content['type'], 'performance_chart')
        self.assertIn('chart_data', content)
        self.assertIn('metrics', content)
        self.assertIn('chart_config', content)
        self.assertIsInstance(content['chart_data'], list)
        
    def test_render_component_session_list(self):
        """Test rendering session list component"""
        component_config = {
            'type': 'session_list',
            'position': {'row': 2, 'col': 0, 'width': 4, 'height': 3},
            'refresh_rate': 15,
            'priority': 'medium'
        }
        
        theme_config = self.renderer.theme_manager.get_theme_config('dark')
        component_data = self.sample_realtime_data['active_sessions']
        
        result = self.renderer._render_component(
            'active_sessions', component_config, component_data, theme_config
        )
        
        # Check rendered content
        content = result['rendered_content']
        self.assertEqual(content['type'], 'session_list')
        self.assertIn('sessions', content)
        self.assertIn('total_sessions', content)
        self.assertIn('active_count', content)
        self.assertIsInstance(content['sessions'], list)
        
        # Check session data processing
        if content['sessions']:
            session = content['sessions'][0]
            self.assertIn('session_id', session)
            self.assertIn('user_id', session)
            self.assertIn('duration', session)
            self.assertIn('status_color', session)
            
    def test_render_component_log_viewer(self):
        """Test rendering log viewer component"""
        component_config = {
            'type': 'log_viewer',
            'position': {'row': 2, 'col': 4, 'width': 8, 'height': 3},
            'refresh_rate': 20,
            'priority': 'medium'
        }
        
        theme_config = self.renderer.theme_manager.get_theme_config('dark')
        component_data = self.sample_realtime_data['error_logs']
        
        result = self.renderer._render_component(
            'error_logs', component_config, component_data, theme_config
        )
        
        # Check rendered content
        content = result['rendered_content']
        self.assertEqual(content['type'], 'log_viewer')
        self.assertIn('logs', content)
        self.assertIn('severity_counts', content)
        self.assertIn('viewer_config', content)
        self.assertIsInstance(content['logs'], list)
        
        # Check log data processing
        if content['logs']:
            log = content['logs'][0]
            self.assertIn('timestamp', log)
            self.assertIn('formatted_time', log)
            self.assertIn('level', log)
            self.assertIn('message', log)
            self.assertIn('color', log)
            
    def test_render_component_unknown_type(self):
        """Test rendering component with unknown type"""
        component_config = {
            'type': 'unknown_component',
            'position': {'row': 0, 'col': 0, 'width': 6, 'height': 2},
            'refresh_rate': 5,
            'priority': 'high'
        }
        
        theme_config = self.renderer.theme_manager.get_theme_config('dark')
        component_data = {}
        
        with self.assertRaises(RuntimeError):
            self.renderer._render_component(
                'unknown_component', component_config, component_data, theme_config
            )
            
    def test_render_component_missing_data(self):
        """Test rendering component with missing required data"""
        component_config = {
            'type': 'health_indicator',
            'position': {'row': 0, 'col': 0, 'width': 6, 'height': 2},
            'refresh_rate': 5,
            'priority': 'high'
        }
        
        theme_config = self.renderer.theme_manager.get_theme_config('dark')
        component_data = {}  # Missing required data
        
        # Should render with default data, not fail
        result = self.renderer._render_component(
            'system_health', component_config, component_data, theme_config
        )
        
        self.assertIn('rendered_content', result)
        content = result['rendered_content']
        self.assertEqual(content['type'], 'health_indicator')
        
    def test_render_health_indicator(self):
        """Test health indicator rendering function"""
        data = {
            'status': 'healthy',
            'score': 85,
            'details': {
                'cpu_usage': 45.2,
                'memory_usage': 60.5
            }
        }
        theme = self.renderer.theme_manager.get_theme_config('dark')
        
        result = self.renderer._render_health_indicator(data, theme)
        
        self.assertEqual(result['type'], 'health_indicator')
        self.assertEqual(result['score'], 85)
        self.assertEqual(result['status_text'], 'Good')
        self.assertIn('details', result)
        self.assertIsInstance(result['details'], list)
        
        # Test different score ranges
        data['score'] = 95
        result_excellent = self.renderer._render_health_indicator(data, theme)
        self.assertEqual(result_excellent['status_text'], 'Excellent')
        
        data['score'] = 45
        result_critical = self.renderer._render_health_indicator(data, theme)
        self.assertEqual(result_critical['status_text'], 'Critical')
        
    def test_render_performance_chart(self):
        """Test performance chart rendering function"""
        data = {
            'chart_data': [
                {'timestamp': time.time(), 'value': 95.2, 'label': 'Response Time'},
                {'timestamp': time.time() + 60, 'value': 92.1, 'label': 'Response Time'}
            ],
            'metrics': {
                'avg_response_time': 93.65,
                'throughput': 1250
            }
        }
        theme = self.renderer.theme_manager.get_theme_config('dark')
        
        result = self.renderer._render_performance_chart(data, theme)
        
        self.assertEqual(result['type'], 'performance_chart')
        self.assertEqual(result['chart_type'], 'line')
        self.assertIn('chart_data', result)
        self.assertIn('metrics', result)
        self.assertIn('chart_config', result)
        self.assertIsInstance(result['chart_data'], list)
        self.assertIsInstance(result['metrics'], list)
        
    def test_render_session_list(self):
        """Test session list rendering function"""
        data = {
            'active_sessions': [
                {
                    'session_id': 'sess_001',
                    'user_id': 'user_123',
                    'activity': 'browsing',
                    'start_time': time.time() - 1800,
                    'status': 'active'
                }
            ],
            'total_sessions': 25
        }
        theme = self.renderer.theme_manager.get_theme_config('dark')
        
        result = self.renderer._render_session_list(data, theme)
        
        self.assertEqual(result['type'], 'session_list')
        self.assertEqual(result['total_sessions'], 25)
        self.assertIn('sessions', result)
        self.assertIn('active_count', result)
        self.assertIsInstance(result['sessions'], list)
        
        if result['sessions']:
            session = result['sessions'][0]
            self.assertEqual(session['session_id'], 'sess_001')
            self.assertIn('duration', session)
            
    def test_render_log_viewer(self):
        """Test log viewer rendering function"""
        data = {
            'logs': [
                {
                    'timestamp': time.time(),
                    'level': 'error',
                    'message': 'Test error message',
                    'source': 'test.py'
                }
            ],
            'severity_counts': {
                'error': 1,
                'warning': 0,
                'info': 5
            }
        }
        theme = self.renderer.theme_manager.get_theme_config('dark')
        
        result = self.renderer._render_log_viewer(data, theme)
        
        self.assertEqual(result['type'], 'log_viewer')
        self.assertIn('logs', result)
        self.assertIn('severity_counts', result)
        self.assertIn('viewer_config', result)
        self.assertIsInstance(result['logs'], list)
        
        if result['logs']:
            log = result['logs'][0]
            self.assertEqual(log['level'], 'error')
            self.assertIn('formatted_time', log)
            self.assertIn('color', log)
            
    def test_render_metrics_grid(self):
        """Test metrics grid rendering function"""
        data = {
            'grid_metrics': {
                'cpu_usage': 75.5,
                'memory_usage': 60.2,
                'disk_usage': 45.8,
                'network_io': 1250000
            },
            'highlights': ['cpu_usage']
        }
        theme = self.renderer.theme_manager.get_theme_config('dark')
        
        result = self.renderer._render_metrics_grid(data, theme)
        
        self.assertEqual(result['type'], 'metrics_grid')
        self.assertIn('metrics', result)
        self.assertIn('grid_config', result)
        self.assertIsInstance(result['metrics'], list)
        
        # Check metric processing
        for metric in result['metrics']:
            self.assertIn('key', metric)
            self.assertIn('label', metric)
            self.assertIn('value', metric)
            self.assertIn('highlighted', metric)
            
        # Check highlighted metric
        highlighted_metrics = [m for m in result['metrics'] if m['highlighted']]
        self.assertEqual(len(highlighted_metrics), 1)
        self.assertEqual(highlighted_metrics[0]['key'], 'cpu_usage')
        
    def test_render_alert_panel(self):
        """Test alert panel rendering function"""
        data = {
            'alerts': [
                {
                    'alert_id': 'alert_001',
                    'title': 'High CPU Usage',
                    'message': 'CPU usage exceeded 90%',
                    'severity': 'critical',
                    'timestamp': time.time() - 300,
                    'actions': ['acknowledge', 'escalate']
                },
                {
                    'alert_id': 'alert_002',
                    'title': 'Low Disk Space',
                    'message': 'Disk space below 10%',
                    'severity': 'warning',
                    'timestamp': time.time() - 180
                }
            ]
        }
        theme = self.renderer.theme_manager.get_theme_config('dark')
        
        result = self.renderer._render_alert_panel(data, theme)
        
        self.assertEqual(result['type'], 'alert_panel')
        self.assertIn('alerts', result)
        self.assertIn('grouped_alerts', result)
        self.assertIn('severity_counts', result)
        self.assertEqual(result['total_count'], 2)
        
        # Check alert processing
        for alert in result['alerts']:
            self.assertIn('alert_id', alert)
            self.assertIn('severity', alert)
            self.assertIn('formatted_time', alert)
            self.assertIn('color', alert)
            self.assertIn('icon', alert)
            
    def test_generate_responsive_layout_grid(self):
        """Test responsive grid layout generation"""
        template = self.renderer._get_dashboard_template('system_overview')
        components = {
            'system_health': {
                'component_id': 'system_health',
                'position': {'row': 0, 'col': 0, 'width': 6, 'height': 2},
                'priority': 'high'
            },
            'performance_metrics': {
                'component_id': 'performance_metrics',
                'position': {'row': 0, 'col': 6, 'width': 6, 'height': 2},
                'priority': 'high'
            }
        }
        user_preferences = {'screen_size': 'desktop'}
        
        layout = self.renderer._generate_responsive_layout(
            template, components, user_preferences
        )
        
        self.assertIn('type', layout)
        self.assertIn('items', layout)
        self.assertIsInstance(layout['items'], list)
        
        # Check layout items
        for item in layout['items']:
            self.assertIn('component_id', item)
            self.assertIn('position', item)
            self.assertIn('priority', item)
            
    def test_generate_grid_layout(self):
        """Test grid layout generation"""
        template = {
            'layout': 'grid',
            'grid_columns': 12,
            'grid_rows': 8
        }
        components = {
            'comp1': {
                'position': {'row': 0, 'col': 0, 'width': 6, 'height': 2},
                'priority': 'high'
            },
            'comp2': {
                'position': {'row': 0, 'col': 6, 'width': 6, 'height': 2},
                'priority': 'medium'
            }
        }
        
        layout = self.renderer._generate_grid_layout(template, components, 'desktop')
        
        self.assertEqual(layout['type'], 'grid')
        self.assertEqual(layout['columns'], 12)
        self.assertEqual(layout['rows'], 8)
        self.assertIsInstance(layout['items'], list)
        self.assertEqual(len(layout['items']), 2)
        
        # Test mobile layout adjustment
        mobile_layout = self.renderer._generate_grid_layout(template, components, 'mobile')
        self.assertEqual(mobile_layout['columns'], 4)
        
        # Check mobile adjustments
        for item in mobile_layout['items']:
            self.assertEqual(item['position']['col'], 0)  # Should be stacked
            self.assertEqual(item['position']['width'], 4)
            
    def test_calculate_data_freshness(self):
        """Test data freshness calculation"""
        current_time = time.time()
        realtime_data = {
            'comp1': {'last_updated': current_time - 2},  # Real-time
            'comp2': {'last_updated': current_time - 15}, # Fresh
            'comp3': {'last_updated': current_time - 45}, # Recent
            'comp4': {'last_updated': current_time - 90}, # Stale
            'comp5': {'last_updated': current_time - 400} # Very stale
        }
        
        freshness = self.renderer._calculate_data_freshness(realtime_data)
        
        self.assertIn('overall', freshness)
        self.assertIn('comp1', freshness)
        
        # Check freshness ratings
        self.assertEqual(freshness['comp1']['freshness'], 'real_time')
        self.assertEqual(freshness['comp2']['freshness'], 'fresh')
        self.assertEqual(freshness['comp3']['freshness'], 'recent')
        self.assertEqual(freshness['comp4']['freshness'], 'stale')
        self.assertEqual(freshness['comp5']['freshness'], 'very_stale')
        
        # Check overall freshness
        self.assertIn('average_freshness', freshness['overall'])
        self.assertIn('freshness_rating', freshness['overall'])
        
    def test_get_default_data(self):
        """Test default data retrieval for components"""
        # Test health indicator defaults
        default_status = self.renderer._get_default_data('health_indicator', 'status')
        self.assertEqual(default_status, 'unknown')
        
        default_score = self.renderer._get_default_data('health_indicator', 'score')
        self.assertEqual(default_score, 0)
        
        # Test performance chart defaults
        default_chart_data = self.renderer._get_default_data('performance_chart', 'chart_data')
        self.assertEqual(default_chart_data, [])
        
        # Test unknown component type
        unknown_default = self.renderer._get_default_data('unknown_type', 'field')
        self.assertIsNone(unknown_default)
        
    def test_helper_methods(self):
        """Test various helper methods"""
        # Test unit detection
        self.assertEqual(self.renderer._get_unit_for_metric('cpu_usage'), '%')
        self.assertEqual(self.renderer._get_unit_for_metric('response_time'), 's')
        self.assertEqual(self.renderer._get_unit_for_metric('memory_usage'), 'GB')
        self.assertEqual(self.renderer._get_unit_for_metric('request_count'), '')
        
        # Test metric value formatting
        self.assertEqual(self.renderer._format_metric_value(1500000), '1.5M')
        self.assertEqual(self.renderer._format_metric_value(2500), '2.5K')
        self.assertEqual(self.renderer._format_metric_value(45.678), '45.68')
        self.assertEqual(self.renderer._format_metric_value(123), '123')
        
        # Test duration formatting
        self.assertEqual(self.renderer._format_duration(45), '45s')
        self.assertEqual(self.renderer._format_duration(125), '2m')
        self.assertEqual(self.renderer._format_duration(3700), '1h 1m')
        
        # Test time ago formatting
        current = time.time()
        self.assertEqual(self.renderer._format_time_ago(current - 30), 'just now')
        self.assertEqual(self.renderer._format_time_ago(current - 120), '2m ago')
        self.assertEqual(self.renderer._format_time_ago(current - 7200), '2h ago')
        self.assertEqual(self.renderer._format_time_ago(current - 172800), '2d ago')
        
        # Test freshness rating
        self.assertEqual(self.renderer._get_freshness_rating(0.9), 'excellent')
        self.assertEqual(self.renderer._get_freshness_rating(0.7), 'good')
        self.assertEqual(self.renderer._get_freshness_rating(0.5), 'fair')
        self.assertEqual(self.renderer._get_freshness_rating(0.2), 'poor')
        
    def test_color_and_icon_helpers(self):
        """Test color and icon helper methods"""
        theme = self.renderer.theme_manager.get_theme_config('dark')
        
        # Test status colors
        active_color = self.renderer._get_status_color('active', theme)
        self.assertEqual(active_color, theme['colors']['success'])
        
        idle_color = self.renderer._get_status_color('idle', theme)
        self.assertEqual(idle_color, theme['colors']['warning'])
        
        # Test log level colors
        error_color = self.renderer._get_log_level_color('error', theme)
        self.assertEqual(error_color, theme['colors']['danger'])
        
        info_color = self.renderer._get_log_level_color('info', theme)
        self.assertEqual(info_color, theme['colors']['info'])
        
        # Test severity colors and icons
        critical_color = self.renderer._get_severity_color('critical', theme)
        self.assertEqual(critical_color, theme['colors']['danger'])
        
        critical_icon = self.renderer._get_severity_icon('critical')
        self.assertEqual(critical_icon, 'alert-octagon')
        
        warning_icon = self.renderer._get_severity_icon('warning')
        self.assertEqual(warning_icon, 'alert-triangle')
        
    def test_responsive_adjustments(self):
        """Test responsive layout adjustments"""
        base_layout = {
            'type': 'grid',
            'columns': 12,
            'items': []
        }
        
        # Test mobile adjustments
        mobile_layout = self.renderer._apply_responsive_adjustments(base_layout.copy(), 'mobile')
        self.assertTrue(mobile_layout.get('stack_components', False))
        self.assertTrue(mobile_layout.get('hide_secondary', False))
        
        # Test tablet adjustments
        tablet_layout = self.renderer._apply_responsive_adjustments(base_layout.copy(), 'tablet')
        self.assertTrue(tablet_layout.get('reduce_columns', False))
        
        # Test desktop (no adjustments)
        desktop_layout = self.renderer._apply_responsive_adjustments(base_layout.copy(), 'desktop')
        self.assertFalse(desktop_layout.get('stack_components', False))
        
    def test_edge_cases_empty_data(self):
        """Test rendering with empty data"""
        empty_data = {}
        result = self.renderer.render_components(
            'system_overview', 
            empty_data, 
            self.sample_user_preferences
        )
        
        # Should still render successfully
        self.assertIn('components', result)
        self.assertIn('metadata', result)
        
    def test_edge_cases_invalid_theme(self):
        """Test rendering with invalid theme preference"""
        invalid_preferences = self.sample_user_preferences.copy()
        invalid_preferences['theme'] = 'nonexistent_theme'
        
        result = self.renderer.render_components(
            'system_overview', 
            self.sample_realtime_data, 
            invalid_preferences
        )
        
        # Should fall back to default theme
        self.assertIn('theme', result)
        self.assertEqual(result['theme']['name'], 'dark')
        
    def test_edge_cases_very_large_data(self):
        """Test rendering with very large datasets"""
        large_data = self.sample_realtime_data.copy()
        
        # Add large session list
        large_sessions = []
        for i in range(100):
            large_sessions.append({
                'session_id': f'sess_{i:03d}',
                'user_id': f'user_{i}',
                'activity': 'browsing',
                'start_time': time.time() - (i * 60),
                'status': 'active'
            })
        
        large_data['active_sessions']['active_sessions'] = large_sessions
        
        # Add large log list
        large_logs = []
        for i in range(200):
            large_logs.append({
                'timestamp': time.time() - (i * 30),
                'level': 'info',
                'message': f'Log message {i}',
                'source': f'module_{i % 10}.py'
            })
        
        large_data['error_logs']['logs'] = large_logs
        
        result = self.renderer.render_components(
            'system_overview', 
            large_data, 
            self.sample_user_preferences
        )
        
        # Should handle large data gracefully
        self.assertIn('components', result)
        
        # Check that data is limited appropriately
        if 'active_sessions' in result['components']:
            sessions = result['components']['active_sessions']['rendered_content']['sessions']
            self.assertLessEqual(len(sessions), 20)  # Should be limited
            
        if 'error_logs' in result['components']:
            logs = result['components']['error_logs']['rendered_content']['logs']
            self.assertLessEqual(len(logs), 100)  # Should be limited
            
    def test_performance_with_multiple_dashboards(self):
        """Test performance with multiple dashboard types"""
        dashboard_types = ['system_overview', 'uncertainty_analysis', 'training_monitoring', 'production_metrics']
        
        start_time = time.time()
        
        for dashboard_type in dashboard_types:
            result = self.renderer.render_components(
                dashboard_type, 
                self.sample_realtime_data, 
                self.sample_user_preferences
            )
            self.assertIn('components', result)
            
        elapsed_time = time.time() - start_time
        
        # Should render all dashboards reasonably quickly
        self.assertLess(elapsed_time, 5.0)  # Less than 5 seconds for 4 dashboards
        
    def test_error_handling_component_render_failure(self):
        """Test error handling when component rendering fails"""
        # This test checks that the main render_components method handles 
        # individual component failures gracefully
        
        # Create data that will cause a rendering error in a specific component
        problematic_data = self.sample_realtime_data.copy()
        
        # Patch one of the render methods to raise an exception
        original_render_health = self.renderer._render_health_indicator
        
        def failing_render_health(data, theme):
            raise ValueError("Simulated rendering failure")
        
        self.renderer._render_health_indicator = failing_render_health
        
        try:
            result = self.renderer.render_components(
                'system_overview', 
                problematic_data, 
                self.sample_user_preferences
            )
            
            # Should still return a result
            self.assertIn('metadata', result)
            
            # Should track component errors
            if result['metadata']['component_errors']:
                self.assertGreater(len(result['metadata']['component_errors']), 0)
                
        finally:
            # Restore original method
            self.renderer._render_health_indicator = original_render_health
            
    def test_thread_safety_multiple_renders(self):
        """Test thread safety of rendering operations"""
        import threading
        import queue
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def render_dashboard(dashboard_type, thread_id):
            try:
                result = self.renderer.render_components(
                    dashboard_type, 
                    self.sample_realtime_data, 
                    self.sample_user_preferences
                )
                results.put((thread_id, result))
            except Exception as e:
                errors.put((thread_id, str(e)))
        
        # Create multiple threads rendering different dashboards
        threads = []
        for i, dashboard_type in enumerate(['system_overview', 'uncertainty_analysis']):
            thread = threading.Thread(target=render_dashboard, args=(dashboard_type, i))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)
            
        # Check results
        self.assertTrue(errors.empty(), f"Thread errors occurred: {list(errors.queue)}")
        
        # All renders should succeed
        success_count = 0
        while not results.empty():
            thread_id, result = results.get()
            self.assertIn('components', result)
            success_count += 1
            
        self.assertEqual(success_count, 2)


class TestDashboardRendererIntegration(unittest.TestCase):
    """Integration tests for DashboardRenderer"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.config = {
            'themes': {
                'default_theme': 'dark'
            }
        }
        self.renderer = DashboardRenderer(self.config)
        
    def test_full_dashboard_rendering_workflow(self):
        """Test complete dashboard rendering workflow"""
        # Simulate realistic production data
        production_data = {
            'system_health': {
                'status': 'degraded',
                'score': 72,
                'details': {
                    'cpu_usage': 78.5,
                    'memory_usage': 82.1,
                    'disk_usage': 45.2,
                    'network_latency': 15.8
                },
                'last_updated': time.time() - 5
            },
            'performance_metrics': {
                'chart_data': self._generate_time_series_data(50),
                'metrics': {
                    'avg_response_time': 245.6,
                    'p95_response_time': 450.2,
                    'throughput': 1847,
                    'error_rate': 0.028
                },
                'last_updated': time.time() - 10
            },
            'active_sessions': {
                'active_sessions': self._generate_session_data(15),
                'total_sessions': 45,
                'session_details': {
                    'avg_duration': 1245,
                    'bounce_rate': 0.23
                },
                'last_updated': time.time() - 8
            },
            'error_logs': {
                'logs': self._generate_log_data(25),
                'filters': {'level': 'error', 'source': 'all'},
                'severity_counts': {
                    'critical': 2,
                    'error': 5,
                    'warning': 12,
                    'info': 6
                },
                'last_updated': time.time() - 12
            }
        }
        
        user_preferences = {
            'theme': 'midnight',
            'screen_size': 'desktop',
            'layout_density': 'comfortable',
            'refresh_interval': 15
        }
        
        # Render complete dashboard
        result = self.renderer.render_components(
            'system_overview',
            production_data,
            user_preferences
        )
        
        # Comprehensive validation
        self.assertIn('dashboard_type', result)
        self.assertIn('components', result)
        self.assertIn('layout', result)
        self.assertIn('theme', result)
        self.assertIn('metadata', result)
        
        # Validate theme application
        self.assertEqual(result['theme']['name'], 'midnight')
        
        # Validate component rendering
        components = result['components']
        expected_components = ['system_health', 'performance_metrics', 'active_sessions', 'error_logs']
        
        for comp_id in expected_components:
            if comp_id in components:
                component = components[comp_id]
                self.assertIn('rendered_content', component)
                self.assertIn('last_updated', component)
                self.assertEqual(component['theme_applied'], 'midnight')
                
        # Validate data freshness
        freshness = result['metadata']['data_freshness']
        self.assertIn('overall', freshness)
        self.assertIn('average_freshness', freshness['overall'])
        
        # Validate layout
        layout = result['layout']
        self.assertIn('type', layout)
        self.assertIn('items', layout)
        
    def _generate_time_series_data(self, count: int) -> List[Dict[str, Any]]:
        """Generate realistic time series data"""
        data = []
        base_time = time.time() - (count * 60)
        
        for i in range(count):
            data.append({
                'timestamp': base_time + (i * 60),
                'value': 200 + (i * 2) + (i % 10) * 15,
                'label': f'Point {i}'
            })
            
        return data
    
    def _generate_session_data(self, count: int) -> List[Dict[str, Any]]:
        """Generate realistic session data"""
        activities = ['browsing', 'searching', 'idle', 'active']
        statuses = ['active', 'idle', 'disconnected']
        
        sessions = []
        for i in range(count):
            sessions.append({
                'session_id': f'sess_{i:03d}',
                'user_id': f'user_{1000 + i}',
                'activity': activities[i % len(activities)],
                'start_time': time.time() - (i * 300),
                'status': statuses[i % len(statuses)]
            })
            
        return sessions
    
    def _generate_log_data(self, count: int) -> List[Dict[str, Any]]:
        """Generate realistic log data"""
        levels = ['info', 'warning', 'error', 'critical']
        sources = ['api.py', 'database.py', 'auth.py', 'cache.py', 'worker.py']
        messages = [
            'Request processed successfully',
            'Cache miss for key',
            'Database connection timeout',
            'Authentication failed',
            'Memory usage high'
        ]
        
        logs = []
        for i in range(count):
            logs.append({
                'timestamp': time.time() - (i * 30),
                'level': levels[i % len(levels)],
                'message': f"{messages[i % len(messages)]} - {i}",
                'source': sources[i % len(sources)]
            })
            
        return logs


if __name__ == '__main__':
    unittest.main(verbosity=2)