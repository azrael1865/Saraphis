#!/usr/bin/env python3
"""
Test script for Saraphis Production Web Interface & Dashboard System
"""

import logging
import time
import json
import random
import asyncio
from datetime import datetime
import concurrent.futures
from typing import Dict, List, Any

# Import all production web components
from production_web import (
    WebInterfaceManager,
    DashboardRenderer,
    ThemeManager,
    ComponentManager,
    RealTimeDataManager,
    CacheManager,
    WebSocketManager,
    UserInterfaceManager,
    APIEndpointManager,
    DashboardMetricsCollector
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_web_interface_manager():
    """Test Web Interface Manager functionality"""
    print("\n" + "="*50)
    print("Testing Web Interface Manager")
    print("="*50)
    
    config = {
        'max_concurrent_dashboards': 10,
        'session_timeout': 300,
        'refresh_interval': 5,
        'dashboard': {},
        'components': {},
        'realtime': {},
        'websocket': {},
        'ui': {},
        'api': {},
        'metrics': {}
    }
    
    web_manager = WebInterfaceManager(config)
    
    # Test dashboard rendering
    print("\n1. Testing dashboard rendering...")
    user_id = 'test_user_123'
    dashboard_type = 'system_overview'
    
    result = web_manager.render_dashboard(user_id, dashboard_type)
    print(f"Render result: Success={result.get('success', False)}")
    if result.get('success'):
        print(f"Session ID: {result.get('session_id')}")
        print(f"Components: {len(result.get('components', {}))}")
        print(f"WebSocket channel: {result.get('websocket_channel')}")
    
    # Test user interaction
    print("\n2. Testing user interaction handling...")
    interaction_data = {
        'session_id': result.get('session_id'),
        'interaction_type': 'click',
        'target': 'button_refresh',
        'coordinates': {'x': 100, 'y': 200},
        'dashboard_type': dashboard_type
    }
    
    interaction_result = web_manager.handle_user_interaction(user_id, interaction_data)
    print(f"Interaction result: {interaction_result.get('success', False)}")
    
    # Test API endpoints
    print("\n3. Testing API endpoints...")
    endpoints = web_manager.get_api_endpoints()
    print(f"API endpoints available: {endpoints.get('success', False)}")
    if endpoints.get('success'):
        print(f"Total endpoints: {len(endpoints.get('endpoints', {}))}")
        print(f"Base URL: {endpoints.get('base_url')}")
        print(f"WebSocket URL: {endpoints.get('websocket_url')}")
    
    # Test WebSocket status
    print("\n4. Testing WebSocket status...")
    ws_status = web_manager.get_websocket_status()
    print(f"WebSocket status: {json.dumps(ws_status, indent=2)}")
    
    # Test dashboard metrics
    print("\n5. Testing dashboard metrics...")
    metrics = web_manager.get_dashboard_metrics()
    print(f"Metrics retrieval: Success={metrics.get('success', False)}")
    if metrics.get('success'):
        print(f"Current state: {json.dumps(metrics.get('current_state', {}), indent=2)}")
    
    # Test dashboard close
    print("\n6. Testing dashboard close...")
    close_result = web_manager.close_dashboard(user_id, dashboard_type)
    print(f"Close result: {close_result}")


def test_dashboard_renderer():
    """Test Dashboard Renderer functionality"""
    print("\n" + "="*50)
    print("Testing Dashboard Renderer")
    print("="*50)
    
    config = {
        'default_theme': 'dark',
        'responsive': True,
        'animation_enabled': True
    }
    
    renderer = DashboardRenderer(config)
    
    # Test theme manager
    print("\n1. Testing theme manager...")
    themes = renderer.theme_manager.get_available_themes()
    print(f"Available themes: {themes}")
    
    # Test dashboard rendering
    print("\n2. Testing dashboard component rendering...")
    
    # Mock real-time data
    realtime_data = {
        'brain_metrics': {
            'quantum_state': {
                'coherence': 0.95,
                'entanglement': 0.87,
                'fidelity': 0.92
            },
            'performance': {
                'accuracy': 0.96,
                'latency': 0.023,
                'throughput': 1250
            }
        },
        'system_health': {
            'components': {
                'brain': 'healthy',
                'api_gateway': 'healthy',
                'database': 'healthy'
            },
            'resources': {
                'cpu_usage': 45.2,
                'memory_usage': 67.8,
                'disk_usage': 23.5
            }
        }
    }
    
    user_preferences = {
        'theme': 'dark',
        'refresh_rate': 5,
        'show_animations': True
    }
    
    dashboard_types = ['system_overview', 'uncertainty_analysis', 'training_monitoring', 'production_metrics']
    
    for dashboard_type in dashboard_types:
        print(f"\nRendering {dashboard_type} dashboard...")
        result = renderer.render_dashboard(dashboard_type, realtime_data, user_preferences)
        
        if result['success']:
            print(f"  Layout: {result['layout']['type']}")
            print(f"  Components: {len(result['components'])}")
            print(f"  Theme: {result['theme']['name']}")
            print(f"  Data freshness: {result['metadata']['data_freshness'].get('overall_freshness', 'N/A')}")


def test_component_manager():
    """Test Component Manager functionality"""
    print("\n" + "="*50)
    print("Testing Component Manager")
    print("="*50)
    
    config = {
        'max_update_batch_size': 50,
        'update_debounce_ms': 100,
        'component_timeout_seconds': 30
    }
    
    component_manager = ComponentManager(config)
    
    # Test component registration
    print("\n1. Testing component registration...")
    
    components_to_register = [
        ('chart', {'title': 'Performance Chart', 'chart_type': 'line'}),
        ('metric', {'title': 'Accuracy', 'unit': '%', 'format': 'percentage'}),
        ('gauge', {'title': 'CPU Usage', 'min': 0, 'max': 100, 'unit': '%'}),
        ('table', {'title': 'Recent Events', 'pagination': True}),
        ('alert', {'title': 'System Alerts', 'max_visible': 5})
    ]
    
    registered_components = {}
    
    for comp_type, config in components_to_register:
        result = component_manager.register_component(
            component_id='',  # Auto-generate
            component_type=comp_type,
            configuration=config
        )
        
        if result['success']:
            comp_id = result['component_id']
            registered_components[comp_id] = comp_type
            print(f"  Registered {comp_type}: {comp_id}")
    
    # Test component updates
    print("\n2. Testing component updates...")
    
    update_data = {
        comp_id: {
            'type': comp_type,
            'data': _generate_component_data(comp_type)
        }
        for comp_id, comp_type in registered_components.items()
    }
    
    updated_components = component_manager.update_components(update_data, update_data)
    print(f"Updated components: {len(updated_components)}")
    
    # Test component metrics
    print("\n3. Testing component metrics...")
    metrics = component_manager.get_component_metrics()
    print(f"Total components: {metrics.get('total_components', 0)}")
    if 'summary' in metrics:
        summary = metrics['summary']
        print(f"  Total renders: {summary.get('total_renders', 0)}")
        print(f"  Total updates: {summary.get('total_updates', 0)}")
        print(f"  Error rate: {summary.get('error_rate', 0):.2%}")


def test_realtime_data_manager():
    """Test Real-Time Data Manager functionality"""
    print("\n" + "="*50)
    print("Testing Real-Time Data Manager")
    print("="*50)
    
    config = {
        'max_stream_size': 1000,
        'batch_size': 10,
        'processing_threads': 2,
        'cache': {
            'ttl_seconds': 60,
            'max_cache_size': 100
        }
    }
    
    data_manager = RealTimeDataManager(config)
    
    # Test cache manager
    print("\n1. Testing cache manager...")
    cache_key = 'test_data_1'
    cache_value = {'value': 42, 'timestamp': time.time()}
    
    data_manager.cache_manager.set(cache_key, cache_value)
    retrieved = data_manager.cache_manager.get(cache_key)
    print(f"Cache test: Set and retrieved successfully = {retrieved == cache_value}")
    
    # Test data pushing
    print("\n2. Testing real-time data push...")
    
    data_sources = ['brain_metrics', 'system_health', 'performance_metrics']
    
    for i in range(10):
        for source in data_sources:
            test_data = {
                'source': source,
                'value': random.random(),
                'timestamp': time.time()
            }
            data_manager.push_data(source, test_data)
        time.sleep(0.1)
    
    # Give time for processing
    time.sleep(1)
    
    # Test dashboard data retrieval
    print("\n3. Testing dashboard data retrieval...")
    dashboard_data = data_manager.get_dashboard_data(
        'system_overview',
        {'time_range': {'start': time.time() - 300, 'end': time.time()}}
    )
    
    if not dashboard_data.get('error'):
        print(f"Retrieved data sources: {list(dashboard_data.keys())}")
        if '_metadata' in dashboard_data:
            print(f"Data freshness: {json.dumps(dashboard_data['_metadata'].get('freshness', {}), indent=2)}")
    
    # Test stream statistics
    print("\n4. Testing stream statistics...")
    stats = data_manager.get_stream_statistics()
    print(f"Active streams: {stats.get('active_streams', 0)}")
    print(f"Queue size: {stats.get('queue_size', 0)}")
    print(f"Performance: {json.dumps(stats.get('performance', {}), indent=2)}")


def test_user_interface_manager():
    """Test User Interface Manager functionality"""
    print("\n" + "="*50)
    print("Testing User Interface Manager")
    print("="*50)
    
    config = {}
    ui_manager = UserInterfaceManager(config)
    
    user_id = 'test_user_456'
    
    # Test user preferences
    print("\n1. Testing user preferences...")
    preferences = ui_manager.get_user_preferences(user_id)
    print(f"Default preferences: {json.dumps(preferences, indent=2)}")
    
    # Update preferences
    new_preferences = {
        'theme': 'midnight',
        'refresh_rate': 10,
        'notifications_enabled': False
    }
    
    update_result = ui_manager.update_user_preferences(user_id, new_preferences)
    print(f"\nPreference update: Success={update_result.get('success', False)}")
    
    # Test user permissions
    print("\n2. Testing user permissions...")
    permissions = ui_manager.get_user_permissions(user_id)
    print(f"User role: {permissions.get('role')}")
    print(f"Allowed dashboards: {permissions.get('dashboards')}")
    print(f"Features: {permissions.get('features')}")
    
    # Test interaction validation
    print("\n3. Testing interaction validation...")
    
    interactions = [
        {
            'interaction_type': 'click',
            'target': 'button_save',
            'coordinates': {'x': 100, 'y': 50}
        },
        {
            'interaction_type': 'input',
            'field': 'search_box',
            'value': 'test query'
        },
        {
            'interaction_type': 'export',
            'format': 'csv',
            'data_type': 'metrics'
        }
    ]
    
    for interaction in interactions:
        validation = ui_manager.validate_interaction(user_id, interaction)
        print(f"\n{interaction['interaction_type']}: Valid={validation.get('valid', False)}")
        if not validation.get('valid'):
            print(f"  Details: {validation.get('details')}")


def test_api_endpoint_manager():
    """Test API Endpoint Manager functionality"""
    print("\n" + "="*50)
    print("Testing API Endpoint Manager")
    print("="*50)
    
    config = {}
    endpoint_manager = APIEndpointManager(config)
    
    # Test endpoint listing
    print("\n1. Testing endpoint listing...")
    all_endpoints = endpoint_manager.get_all_endpoints()
    print(f"Total endpoints: {len(all_endpoints)}")
    
    # Show sample endpoints
    sample_endpoints = list(all_endpoints.items())[:5]
    for path, info in sample_endpoints:
        print(f"\n{path}:")
        print(f"  Methods: {info['methods']}")
        print(f"  Description: {info['description']}")
        print(f"  Auth required: {info['auth_required']}")
    
    # Test endpoint matching
    print("\n2. Testing endpoint matching...")
    test_paths = [
        ('/api/v1/dashboards', 'GET'),
        ('/api/v1/dashboards/dash_123', 'GET'),
        ('/api/v1/components/comp_456/update', 'POST'),
        ('/api/v1/invalid/endpoint', 'GET')
    ]
    
    for path, method in test_paths:
        match = endpoint_manager.match_endpoint(path, method)
        if match:
            print(f"\n{method} {path}: Matched to {match['endpoint']}")
            if match['params']:
                print(f"  Parameters: {match['params']}")
        else:
            print(f"\n{method} {path}: No match found")
    
    # Test request validation
    print("\n3. Testing request validation...")
    
    test_requests = [
        {
            'endpoint': '/api/v1/dashboards',
            'method': 'POST',
            'data': {
                'name': 'Test Dashboard',
                'type': 'system_overview'
            }
        },
        {
            'endpoint': '/api/v1/dashboards',
            'method': 'POST',
            'data': {
                'name': 'Invalid Dashboard'
                # Missing 'type' field
            }
        }
    ]
    
    for req in test_requests:
        validation = endpoint_manager.validate_request(
            req['endpoint'], req['method'], req['data']
        )
        print(f"\n{req['method']} {req['endpoint']}: Valid={validation.get('valid', False)}")
        if not validation.get('valid'):
            print(f"  Details: {validation.get('details')}")


def test_dashboard_metrics():
    """Test Dashboard Metrics Collector functionality"""
    print("\n" + "="*50)
    print("Testing Dashboard Metrics Collector")
    print("="*50)
    
    config = {
        'retention_period_hours': 24,
        'aggregation_intervals': [1, 5, 15],
        'percentiles': [50, 90, 95, 99]
    }
    
    metrics_collector = DashboardMetricsCollector(config)
    
    # Simulate dashboard activity
    print("\n1. Simulating dashboard activity...")
    
    users = ['user1', 'user2', 'user3']
    dashboards = ['system_overview', 'production_metrics']
    
    for i in range(20):
        user = random.choice(users)
        dashboard = random.choice(dashboards)
        render_time = random.uniform(0.1, 2.0)
        success = random.random() > 0.1  # 90% success rate
        
        metrics_collector.record_dashboard_render(
            user, dashboard, render_time, 
            component_count=random.randint(5, 20),
            success=success
        )
        
        # Record some interactions
        if random.random() > 0.5:
            metrics_collector.record_user_interaction(
                user, 
                random.choice(['click', 'input', 'scroll']),
                random.uniform(0.01, 0.1),
                dashboard
            )
    
    # Record resource usage
    print("\n2. Recording resource usage...")
    metrics_collector.record_resource_usage(
        memory_mb=512.5,
        cpu_percent=35.2,
        active_dashboards=3,
        active_components=45
    )
    
    # Get comprehensive metrics
    print("\n3. Retrieving dashboard metrics...")
    metrics = metrics_collector.get_dashboard_metrics()
    
    if 'summary' in metrics:
        summary = metrics['summary']
        print(f"\nSummary:")
        print(f"  Total renders: {summary.get('total_renders', 0)}")
        print(f"  Success rate: {summary.get('success_rate', 0):.2%}")
        print(f"  Average render time: {summary.get('average_render_time', 0):.3f}s")
        print(f"  Active users: {summary.get('active_users', 0)}")
    
    if 'resources' in metrics:
        resources = metrics['resources']
        print(f"\nResource Usage:")
        print(f"  Memory: {resources.get('memory', {}).get('current_mb', 0):.1f} MB")
        print(f"  CPU: {resources.get('cpu', {}).get('current_percent', 0):.1f}%")
    
    if 'alerts' in metrics:
        alerts = metrics['alerts']
        if alerts:
            print(f"\nActive Alerts: {len(alerts)}")
            for alert in alerts:
                print(f"  - {alert['type']}: {alert.get('details', {})}")


def test_websocket_functionality():
    """Test WebSocket functionality (simulated)"""
    print("\n" + "="*50)
    print("Testing WebSocket Functionality (Simulated)")
    print("="*50)
    
    # Note: Full WebSocket testing would require async implementation
    # This is a simplified synchronous test
    
    config = {
        'max_connections_per_user': 5,
        'message_rate_limit': 100,
        'heartbeat_interval': 30
    }
    
    ws_manager = WebSocketManager(config)
    
    print("\n1. WebSocket Manager initialized")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Test connection status
    print("\n2. Testing connection status...")
    status = ws_manager.get_connection_status()
    print(f"Status: {json.dumps(status, indent=2)}")
    
    # Test channel operations
    print("\n3. Testing channel operations...")
    user_id = 'test_user'
    channel = 'dashboard_test_user_system_overview'
    
    ws_manager.subscribe_to_channel(user_id, channel)
    print(f"Subscribed user {user_id} to channel {channel}")
    
    # Note: In production, this would trigger actual WebSocket broadcasts
    ws_manager.broadcast_dashboard_update(
        user_id, 'system_overview',
        {'type': 'data_update', 'timestamp': time.time()}
    )
    print("Dashboard update broadcast sent")


def test_load_performance():
    """Test system under load"""
    print("\n" + "="*50)
    print("Testing Load Performance")
    print("="*50)
    
    # Initialize system
    config = {
        'max_concurrent_dashboards': 50,
        'cache': {'ttl_seconds': 30, 'max_cache_size': 1000}
    }
    
    web_manager = WebInterfaceManager(config)
    
    print("\n1. Performing concurrent dashboard renders...")
    
    def render_dashboard(i):
        """Render a dashboard"""
        user_id = f'load_test_user_{i % 10}'
        dashboard_type = random.choice(['system_overview', 'production_metrics'])
        
        start_time = time.time()
        result = web_manager.render_dashboard(user_id, dashboard_type)
        duration = time.time() - start_time
        
        return {
            'success': result.get('success', False),
            'duration': duration,
            'user_id': user_id,
            'dashboard_type': dashboard_type
        }
    
    # Run concurrent renders
    num_requests = 50
    num_workers = 10
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(render_dashboard, i) for i in range(num_requests)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful = sum(1 for r in results if r['success'])
    durations = [r['duration'] for r in results if r['success']]
    
    print(f"\nLoad test results:")
    print(f"Total requests: {num_requests}")
    print(f"Successful: {successful} ({successful/num_requests*100:.1f}%)")
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests per second: {num_requests/total_time:.1f}")
    
    if durations:
        print(f"Average render time: {sum(durations)/len(durations):.3f}s")
        print(f"Min render time: {min(durations):.3f}s")
        print(f"Max render time: {max(durations):.3f}s")


def _generate_component_data(component_type: str) -> Dict[str, Any]:
    """Generate mock data for component type"""
    if component_type == 'chart':
        return {
            'datasets': [{
                'label': 'Performance',
                'data': [random.random() * 100 for _ in range(10)]
            }],
            'labels': [f'T{i}' for i in range(10)]
        }
    elif component_type == 'metric':
        return {
            'value': random.random() * 100,
            'previous_value': random.random() * 100
        }
    elif component_type == 'gauge':
        return {
            'value': random.random() * 100
        }
    elif component_type == 'table':
        return {
            'columns': ['ID', 'Name', 'Value'],
            'rows': [
                [f'ID{i}', f'Item {i}', random.randint(0, 100)]
                for i in range(5)
            ]
        }
    elif component_type == 'alert':
        return {
            'alerts': [{
                'id': f'alert_{i}',
                'severity': random.choice(['info', 'warning', 'error']),
                'title': f'Alert {i}',
                'message': 'Test alert message',
                'timestamp': time.time()
            } for i in range(3)]
        }
    else:
        return {'data': 'test'}


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("SARAPHIS PRODUCTION WEB INTERFACE & DASHBOARD SYSTEM TEST")
    print("="*70)
    
    try:
        test_web_interface_manager()
        test_dashboard_renderer()
        test_component_manager()
        test_realtime_data_manager()
        test_user_interface_manager()
        test_api_endpoint_manager()
        test_dashboard_metrics()
        test_websocket_functionality()
        test_load_performance()
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\nERROR: Test failed - {e}")


if __name__ == "__main__":
    main()