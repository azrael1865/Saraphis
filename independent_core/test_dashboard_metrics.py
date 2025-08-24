#!/usr/bin/env python3
"""
Comprehensive test suite for DashboardMetricsCollector - NO MOCKS, HARD FAILURES ONLY
Tests all metrics collection, aggregation, and analysis functionality
Uses real components and fails hard for proper debugging
"""

import unittest
import time
import threading
import tempfile
import json
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any

from production_web.dashboard_metrics import DashboardMetricsCollector


class TestDashboardMetricsCollector(unittest.TestCase):
    """Comprehensive test for DashboardMetricsCollector with NO mocks - hard failures only"""
    
    def setUp(self):
        """Set up real test fixtures - no mocks"""
        self.config = {
            'retention_period_hours': 1,  # Short retention for testing
            'aggregation_intervals': [1, 5, 15],  # minutes
            'percentiles': [50, 90, 95, 99],
            'render_time_threshold': 1000,  # ms
            'error_rate_threshold': 5,  # percent
            'memory_threshold': 512,  # MB
            'cpu_threshold': 80,  # percent
            'disconnect_threshold': 10
        }
        
        self.collector = DashboardMetricsCollector(self.config)
        
        # Allow initialization to complete
        time.sleep(0.1)
    
    def tearDown(self):
        """Clean up after tests"""
        # Stop background threads by clearing the collector
        if hasattr(self.collector, '_lock'):
            with self.collector._lock:
                pass  # Ensure threads can finish current operations
        time.sleep(0.1)  # Allow threads to finish
    
    def test_initialization_hard_requirements(self):
        """Test initialization with hard requirements - must not fail"""
        self.assertIsNotNone(self.collector.config)
        self.assertIsNotNone(self.collector._lock)
        self.assertEqual(self.collector.retention_period_hours, 1)
        self.assertEqual(self.collector.aggregation_intervals, [1, 5, 15])
        self.assertEqual(self.collector.percentiles, [50, 90, 95, 99])
        
        # Verify data structures are initialized
        self.assertIsInstance(self.collector.render_metrics, dict)
        self.assertIsInstance(self.collector.component_metrics, dict)
        self.assertIsInstance(self.collector.user_activity_metrics, dict)
        self.assertIsInstance(self.collector.realtime_metrics, dict)
        self.assertIsInstance(self.collector.resource_metrics, dict)
        self.assertIsInstance(self.collector.error_metrics, dict)
        self.assertIsInstance(self.collector.active_alerts, dict)
        self.assertIsInstance(self.collector.historical_data, dict)
    
    def test_record_dashboard_render_success(self):
        """Test recording successful dashboard render"""
        user_id = "test_user_001"
        dashboard_type = "analytics_dashboard"
        render_time = 0.250  # 250ms
        component_count = 15
        
        # Record the render
        self.collector.record_dashboard_render(
            user_id, dashboard_type, render_time, component_count, success=True
        )
        
        # Verify metrics were recorded
        metrics = self.collector.render_metrics[dashboard_type]
        self.assertEqual(metrics['total_renders'], 1)
        self.assertEqual(metrics['successful_renders'], 1)
        self.assertEqual(metrics['failed_renders'], 0)
        self.assertEqual(len(metrics['render_times']), 1)
        self.assertEqual(metrics['render_times'][0]['value'], render_time)
        self.assertEqual(len(metrics['component_counts']), 1)
        self.assertEqual(metrics['component_counts'][0], component_count)
        self.assertIsNotNone(metrics['last_render'])
        
        # Verify user activity was recorded
        user_metrics = self.collector.user_activity_metrics[user_id]
        self.assertEqual(user_metrics['dashboard_views'][dashboard_type], 1)
        self.assertIsNotNone(user_metrics['last_activity'])
        
        # Verify historical data
        self.assertGreater(len(self.collector.historical_data['dashboard_renders']), 0)
        render_record = self.collector.historical_data['dashboard_renders'][0]
        self.assertEqual(render_record['user_id'], user_id)
        self.assertEqual(render_record['dashboard_type'], dashboard_type)
        self.assertEqual(render_record['render_time'], render_time)
        self.assertEqual(render_record['component_count'], component_count)
        self.assertTrue(render_record['success'])
    
    def test_record_dashboard_render_failure(self):
        """Test recording failed dashboard render"""
        user_id = "test_user_002"
        dashboard_type = "performance_dashboard"
        render_time = 2.5  # 2.5 seconds (slow)
        
        # Record the failed render
        self.collector.record_dashboard_render(
            user_id, dashboard_type, render_time, success=False
        )
        
        # Verify failure was recorded
        metrics = self.collector.render_metrics[dashboard_type]
        self.assertEqual(metrics['total_renders'], 1)
        self.assertEqual(metrics['successful_renders'], 0)
        self.assertEqual(metrics['failed_renders'], 1)
        
        # Should create slow render alert
        time.sleep(0.1)  # Allow alert processing
        self.assertIn('slow_render', self.collector.active_alerts)
    
    def test_record_component_performance_render(self):
        """Test recording component render performance"""
        component_id = "chart_001"
        component_type = "LineChart"
        duration = 0.120  # 120ms
        
        # Record component render
        self.collector.record_component_performance(
            component_id, component_type, 'render', duration, success=True
        )
        
        # Verify metrics were recorded
        key = f"{component_type}:{component_id}"
        metrics = self.collector.component_metrics[key]
        self.assertEqual(metrics['render_count'], 1)
        self.assertEqual(metrics['error_count'], 0)
        self.assertEqual(len(metrics['render_times']), 1)
        self.assertEqual(metrics['render_times'][0]['value'], duration)
    
    def test_record_component_performance_update(self):
        """Test recording component update performance"""
        component_id = "table_001"
        component_type = "DataTable"
        duration = 0.085  # 85ms
        
        # Record component update
        self.collector.record_component_performance(
            component_id, component_type, 'update', duration, success=True
        )
        
        # Verify metrics were recorded
        key = f"{component_type}:{component_id}"
        metrics = self.collector.component_metrics[key]
        self.assertEqual(metrics['update_count'], 1)
        self.assertEqual(metrics['error_count'], 0)
        self.assertEqual(len(metrics['update_times']), 1)
        self.assertEqual(metrics['update_times'][0]['value'], duration)
    
    def test_record_component_performance_error(self):
        """Test recording component error"""
        component_id = "graph_001"
        component_type = "NetworkGraph"
        duration = 0.050
        
        # Record component error
        self.collector.record_component_performance(
            component_id, component_type, 'render', duration, success=False
        )
        
        # Verify error was recorded
        key = f"{component_type}:{component_id}"
        metrics = self.collector.component_metrics[key]
        self.assertEqual(metrics['render_count'], 1)
        self.assertEqual(metrics['error_count'], 1)
    
    def test_record_user_interaction(self):
        """Test recording user interactions"""
        user_id = "test_user_003"
        interaction_type = "filter_apply"
        processing_time = 0.045
        dashboard_type = "metrics_dashboard"
        
        # Record interaction
        self.collector.record_user_interaction(
            user_id, interaction_type, processing_time, dashboard_type
        )
        
        # Verify user metrics updated
        user_metrics = self.collector.user_activity_metrics[user_id]
        self.assertEqual(user_metrics['interaction_count'], 1)
        self.assertEqual(user_metrics['feature_usage'][interaction_type], 1)
        self.assertIsNotNone(user_metrics['last_activity'])
        
        # Verify historical data
        interactions = self.collector.historical_data['interactions']
        self.assertGreater(len(interactions), 0)
        interaction_record = interactions[0]
        self.assertEqual(interaction_record['user_id'], user_id)
        self.assertEqual(interaction_record['interaction_type'], interaction_type)
        self.assertEqual(interaction_record['processing_time'], processing_time)
        self.assertEqual(interaction_record['dashboard_type'], dashboard_type)
    
    def test_record_websocket_connection_metrics(self):
        """Test recording WebSocket connection metrics"""
        # Record connection count
        self.collector.record_websocket_metrics('connection_count', 25)
        
        # Verify recording
        connections = self.collector.realtime_metrics['websocket_connections']
        self.assertEqual(len(connections), 1)
        self.assertEqual(connections[0]['value'], 25)
        self.assertIsNotNone(connections[0]['timestamp'])
    
    def test_record_websocket_latency_metrics(self):
        """Test recording WebSocket latency metrics"""
        latency = 45.2  # ms
        
        # Record update latency
        self.collector.record_websocket_metrics('update_latency', latency)
        
        # Verify recording
        latencies = self.collector.realtime_metrics['update_latencies']
        self.assertEqual(len(latencies), 1)
        self.assertEqual(latencies[0]['value'], latency)
    
    def test_record_websocket_broadcast_success(self):
        """Test recording WebSocket broadcast success rate"""
        success_rate = 0.987  # 98.7%
        
        # Record broadcast success
        self.collector.record_websocket_metrics('broadcast_success', success_rate)
        
        # Verify recording
        broadcasts = self.collector.realtime_metrics['broadcast_success_rate']
        self.assertEqual(len(broadcasts), 1)
        self.assertEqual(broadcasts[0]['success_rate'], success_rate)
    
    def test_record_websocket_stream_throughput(self):
        """Test recording stream throughput metrics"""
        stream_name = "price_updates"
        throughput = 1250.5  # messages/sec
        
        # Record stream throughput
        self.collector.record_websocket_metrics(f'stream_throughput:{stream_name}', throughput)
        
        # Verify recording
        stream_data = self.collector.realtime_metrics['data_stream_throughput'][stream_name]
        self.assertEqual(len(stream_data), 1)
        self.assertEqual(stream_data[0]['value'], throughput)
    
    def test_record_resource_usage_normal(self):
        """Test recording normal resource usage"""
        memory_mb = 256.5
        cpu_percent = 45.2
        active_dashboards = 8
        active_components = 120
        
        # Record resource usage
        self.collector.record_resource_usage(
            memory_mb, cpu_percent, active_dashboards, active_components
        )
        
        # Verify all metrics were recorded
        self.assertEqual(len(self.collector.resource_metrics['memory_usage']), 1)
        self.assertEqual(len(self.collector.resource_metrics['cpu_usage']), 1)
        self.assertEqual(len(self.collector.resource_metrics['active_dashboards']), 1)
        self.assertEqual(len(self.collector.resource_metrics['active_components']), 1)
        
        self.assertEqual(self.collector.resource_metrics['memory_usage'][0]['value'], memory_mb)
        self.assertEqual(self.collector.resource_metrics['cpu_usage'][0]['value'], cpu_percent)
        self.assertEqual(self.collector.resource_metrics['active_dashboards'][0]['value'], active_dashboards)
        self.assertEqual(self.collector.resource_metrics['active_components'][0]['value'], active_components)
    
    def test_record_resource_usage_high_memory_alert(self):
        """Test resource usage that triggers high memory alert"""
        memory_mb = 600  # Above 512 threshold
        cpu_percent = 30
        active_dashboards = 5
        active_components = 50
        
        # Record high memory usage
        self.collector.record_resource_usage(
            memory_mb, cpu_percent, active_dashboards, active_components
        )
        
        # Should trigger high memory alert
        time.sleep(0.1)  # Allow alert processing
        self.assertIn('high_memory_usage', self.collector.active_alerts)
        alert = self.collector.active_alerts['high_memory_usage']
        self.assertEqual(alert['details']['memory_mb'], memory_mb)
    
    def test_record_resource_usage_high_cpu_alert(self):
        """Test resource usage that triggers high CPU alert"""
        memory_mb = 200
        cpu_percent = 90  # Above 80 threshold
        active_dashboards = 3
        active_components = 25
        
        # Record high CPU usage
        self.collector.record_resource_usage(
            memory_mb, cpu_percent, active_dashboards, active_components
        )
        
        # Should trigger high CPU alert
        time.sleep(0.1)  # Allow alert processing
        self.assertIn('high_cpu_usage', self.collector.active_alerts)
        alert = self.collector.active_alerts['high_cpu_usage']
        self.assertEqual(alert['details']['cpu_percent'], cpu_percent)
    
    def test_record_dashboard_error(self):
        """Test recording dashboard errors"""
        user_id = "test_user_004"
        dashboard_type = "system_dashboard"
        error_message = "Failed to load chart data: Connection timeout"
        error_type = "data_load_error"
        
        # Record error
        self.collector.record_dashboard_error(
            user_id, dashboard_type, error_message, error_type
        )
        
        # Verify error was recorded
        error_key = f"{dashboard_type}:{error_type}"
        error_metric = self.collector.error_metrics[error_key]
        self.assertEqual(error_metric['count'], 1)
        self.assertEqual(error_metric['types'][error_type], 1)
        self.assertIn(user_id, error_metric['affected_users'])
        self.assertIn(dashboard_type, error_metric['affected_dashboards'])
        self.assertIsNotNone(error_metric['last_occurrence'])
        self.assertEqual(len(error_metric['error_messages']), 1)
        
        error_msg = error_metric['error_messages'][0]
        self.assertEqual(error_msg['message'], error_message)
        self.assertEqual(error_msg['user_id'], user_id)
    
    def test_record_dashboard_close(self):
        """Test recording dashboard close events"""
        user_id = "test_user_005"
        dashboard_type = "admin_dashboard"
        session_duration = 1847.5  # seconds
        
        # Record dashboard close
        self.collector.record_dashboard_close(
            user_id, dashboard_type, session_duration
        )
        
        # Verify session was recorded
        user_metrics = self.collector.user_activity_metrics[user_id]
        self.assertEqual(user_metrics['session_count'], 1)
        self.assertEqual(user_metrics['total_session_duration'], session_duration)
        
        # Verify historical data
        sessions = self.collector.historical_data['sessions']
        self.assertGreater(len(sessions), 0)
        session_record = sessions[0]
        self.assertEqual(session_record['user_id'], user_id)
        self.assertEqual(session_record['dashboard_type'], dashboard_type)
        self.assertEqual(session_record['duration'], session_duration)
    
    def test_record_access_denied(self):
        """Test recording access denied events"""
        user_id = "unauthorized_user"
        dashboard_type = "admin_dashboard"
        
        # Record access denied
        self.collector.record_access_denied(user_id, dashboard_type)
        
        # Verify historical data
        access_denied = self.collector.historical_data['access_denied']
        self.assertGreater(len(access_denied), 0)
        record = access_denied[0]
        self.assertEqual(record['user_id'], user_id)
        self.assertEqual(record['dashboard_type'], dashboard_type)
    
    def test_record_interaction_error(self):
        """Test recording interaction errors"""
        user_id = "test_user_006"
        error_message = "Invalid filter criteria provided"
        
        # Record interaction error
        self.collector.record_interaction_error(user_id, error_message)
        
        # Verify error was recorded
        error_metric = self.collector.error_metrics['interaction_errors']
        self.assertEqual(error_metric['count'], 1)
        self.assertIn(user_id, error_metric['affected_users'])
        self.assertEqual(len(error_metric['error_messages']), 1)
        
        error_msg = error_metric['error_messages'][0]
        self.assertEqual(error_msg['message'], error_message)
        self.assertEqual(error_msg['user_id'], user_id)
    
    def test_get_dashboard_metrics_comprehensive(self):
        """Test comprehensive metrics retrieval"""
        # Generate test data
        self._generate_test_data()
        
        # Get comprehensive metrics
        metrics = self.collector.get_dashboard_metrics()
        
        # Verify structure
        self.assertIn('summary', metrics)
        self.assertIn('dashboards', metrics)
        self.assertIn('components', metrics)
        self.assertIn('users', metrics)
        self.assertIn('realtime', metrics)
        self.assertIn('resources', metrics)
        self.assertIn('errors', metrics)
        self.assertIn('alerts', metrics)
        self.assertIn('timestamp', metrics)
        
        # Verify summary metrics
        summary = metrics['summary']
        self.assertGreaterEqual(summary['total_renders'], 0)
        self.assertGreaterEqual(summary['successful_renders'], 0)
        self.assertIn('success_rate', summary)
        self.assertIn('average_render_time', summary)
        self.assertIn('active_users', summary)
        
        # Verify dashboard metrics have proper structure
        if metrics['dashboards']:
            dashboard_key = next(iter(metrics['dashboards']))
            dashboard_metric = metrics['dashboards'][dashboard_key]
            self.assertIn('total_renders', dashboard_metric)
            self.assertIn('success_rate', dashboard_metric)
            
        # Verify component summary
        components = metrics['components']
        self.assertIn('total_components', components)
        self.assertIn('error_rate', components)
        
        # Verify user summary
        users = metrics['users']
        self.assertIn('active_users', users)
        self.assertIn('total_interactions', users)
        
        # Verify real-time metrics
        realtime = metrics['realtime']
        self.assertIn('websocket', realtime)
        self.assertIn('latency', realtime)
        self.assertIn('broadcast', realtime)
    
    def test_metrics_aggregation_intervals(self):
        """Test metrics aggregation at different intervals"""
        # Generate timestamped data
        dashboard_type = "test_dashboard"
        current_time = time.time()
        
        # Create renders at different times
        for i in range(10):
            timestamp_offset = i * 30  # 30 seconds apart
            self.collector.render_metrics[dashboard_type]['render_times'].append({
                'value': 0.1 + (i * 0.01),  # 100ms to 190ms
                'timestamp': current_time - timestamp_offset
            })
        
        # Run aggregation
        self.collector.aggregate_metrics()
        
        # Verify aggregated data exists
        aggregated = self.collector.aggregated_metrics[dashboard_type]
        self.assertIn('1m', aggregated)  # 1 minute interval
        
        # Verify aggregation structure
        one_min = aggregated['1m']
        self.assertIn('count', one_min)
        self.assertIn('average_render_time', one_min)
        self.assertIn('min_render_time', one_min)
        self.assertIn('max_render_time', one_min)
        self.assertIn('percentiles', one_min)
    
    def test_percentile_calculations(self):
        """Test percentile calculations with real data"""
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        percentiles = self.collector._calculate_percentiles(values)
        
        self.assertIn('p50', percentiles)
        self.assertIn('p90', percentiles)
        self.assertIn('p95', percentiles)
        self.assertIn('p99', percentiles)
        
        # Verify reasonable percentile values
        self.assertGreaterEqual(percentiles['p50'], 50)
        self.assertGreaterEqual(percentiles['p90'], 90)
        self.assertGreaterEqual(percentiles['p99'], 90)
    
    def test_cleanup_old_metrics(self):
        """Test cleanup of old metrics data"""
        # Add old data (older than retention period)
        old_timestamp = time.time() - (2 * 3600)  # 2 hours ago (beyond 1 hour retention)
        
        # Add old render data
        self.collector.render_metrics['test_dashboard']['render_times'].append({
            'value': 0.1,
            'timestamp': old_timestamp
        })
        
        # Add old historical data
        self.collector.historical_data['dashboard_renders'].append({
            'timestamp': old_timestamp,
            'user_id': 'old_user',
            'dashboard_type': 'test_dashboard'
        })
        
        # Add recent data that should be kept
        recent_timestamp = time.time() - 300  # 5 minutes ago
        self.collector.render_metrics['test_dashboard']['render_times'].append({
            'value': 0.2,
            'timestamp': recent_timestamp
        })
        
        # Run cleanup
        self.collector.cleanup_old_metrics()
        
        # Verify old data was removed but recent data remains
        render_times = self.collector.render_metrics['test_dashboard']['render_times']
        remaining_times = [rt for rt in render_times if isinstance(rt, dict)]
        
        # Should only have recent data
        for rt in remaining_times:
            self.assertGreaterEqual(rt['timestamp'], time.time() - 3600)  # Within retention period
    
    def test_alert_creation_and_management(self):
        """Test alert creation and management"""
        # Create multiple alerts of same type
        details1 = {'render_time': 2.5, 'dashboard_type': 'slow_dashboard'}
        details2 = {'render_time': 3.0, 'dashboard_type': 'slow_dashboard'}
        
        self.collector._create_alert('slow_render', details1)
        self.collector._create_alert('slow_render', details2)
        
        # Verify alert was created and updated
        self.assertIn('slow_render', self.collector.active_alerts)
        alert = self.collector.active_alerts['slow_render']
        self.assertEqual(alert['type'], 'slow_render')
        self.assertEqual(alert['count'], 2)  # Should have been updated
        self.assertEqual(alert['details'], details2)  # Should have latest details
        
        # Test getting active alerts
        active_alerts = self.collector._get_active_alerts()
        self.assertIsInstance(active_alerts, list)
        self.assertGreater(len(active_alerts), 0)
    
    def test_error_rate_alert_threshold(self):
        """Test error rate alert triggering"""
        dashboard_type = "error_prone_dashboard"
        
        # Generate enough successful renders
        for _ in range(20):
            self.collector.record_dashboard_render(
                "user1", dashboard_type, 0.1, success=True
            )
        
        # Generate failed renders to trigger error rate
        for _ in range(5):  # This should give us > 5% error rate
            self.collector.record_dashboard_render(
                "user1", dashboard_type, 0.1, success=False
            )
        
        # Force error rate check
        self.collector._check_error_rate_alert(dashboard_type)
        
        # Should have triggered high error rate alert
        time.sleep(0.1)  # Allow alert processing
        self.assertIn('high_error_rate', self.collector.active_alerts)
    
    def test_concurrent_operations_thread_safety(self):
        """Test thread safety with concurrent operations"""
        def record_renders():
            for i in range(50):
                self.collector.record_dashboard_render(
                    f"user_{i}", "concurrent_dashboard", 0.1 + (i * 0.001)
                )
                time.sleep(0.001)
        
        def record_components():
            for i in range(30):
                self.collector.record_component_performance(
                    f"comp_{i}", "TestComponent", "render", 0.05
                )
                time.sleep(0.001)
        
        def record_resources():
            for i in range(20):
                self.collector.record_resource_usage(100 + i, 20 + i, 5, 25)
                time.sleep(0.002)
        
        # Run operations concurrently
        threads = [
            threading.Thread(target=record_renders),
            threading.Thread(target=record_components),
            threading.Thread(target=record_resources)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Verify all data was recorded correctly
        self.assertGreater(
            self.collector.render_metrics['concurrent_dashboard']['total_renders'], 0
        )
        self.assertGreater(len(self.collector.component_metrics), 0)
        self.assertGreater(len(self.collector.resource_metrics['memory_usage']), 0)
        
        # Get metrics without crashing
        metrics = self.collector.get_dashboard_metrics()
        self.assertIsInstance(metrics, dict)
        self.assertIn('summary', metrics)
    
    def test_empty_data_handling(self):
        """Test handling of empty data scenarios"""
        # Test percentiles with empty data
        empty_percentiles = self.collector._calculate_percentiles([])
        self.assertEqual(empty_percentiles, {})
        
        # Test summary with no data
        summary = self.collector._calculate_summary_metrics()
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary['total_renders'], 0)
        
        # Test dashboard metrics with no data
        formatted = self.collector._format_dashboard_metrics({
            'total_renders': 0,
            'successful_renders': 0,
            'failed_renders': 0,
            'render_times': [],
            'component_counts': [],
            'last_render': None
        })
        self.assertIsInstance(formatted, dict)
        self.assertEqual(formatted['success_rate'], 1.0)  # Default when no renders
    
    def _generate_test_data(self):
        """Generate comprehensive test data for testing"""
        # Dashboard renders
        dashboard_types = ["analytics", "performance", "admin"]
        users = ["user1", "user2", "user3"]
        
        for i in range(30):
            user = users[i % 3]
            dashboard = dashboard_types[i % 3]
            render_time = 0.1 + (i * 0.01)
            success = i % 10 != 0  # 10% failure rate
            
            self.collector.record_dashboard_render(
                user, dashboard, render_time, 10 + (i % 5), success
            )
        
        # Component performance
        components = [("Chart", "chart1"), ("Table", "table1"), ("Graph", "graph1")]
        for i, (comp_type, comp_id) in enumerate(components * 10):
            self.collector.record_component_performance(
                comp_id, comp_type, "render", 0.05 + (i * 0.005)
            )
        
        # User interactions
        for i in range(50):
            user = users[i % 3]
            interaction = ["filter", "sort", "export"][i % 3]
            self.collector.record_user_interaction(
                user, interaction, 0.02 + (i * 0.001)
            )
        
        # WebSocket metrics
        for i in range(20):
            self.collector.record_websocket_metrics('connection_count', 15 + i)
            self.collector.record_websocket_metrics('update_latency', 20 + i)
        
        # Resource usage
        for i in range(10):
            self.collector.record_resource_usage(
                200 + (i * 10), 30 + (i * 2), 5 + i, 50 + (i * 5)
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)