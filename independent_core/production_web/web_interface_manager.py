"""
Saraphis Web Interface Manager
Production-ready web interface management with real-time dashboards
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque

from .dashboard_renderer import DashboardRenderer
from .component_manager import ComponentManager
from .realtime_data_manager import RealTimeDataManager
from .websocket_manager import WebSocketManager
from .user_interface_manager import UserInterfaceManager
from .api_endpoint_manager import APIEndpointManager
from .dashboard_metrics import DashboardMetricsCollector

logger = logging.getLogger(__name__)


class WebInterfaceManager:
    """Production-ready web interface management with real-time dashboards and user interactions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.interface_history = deque(maxlen=10000)
        
        # Initialize components
        self.dashboard_renderer = DashboardRenderer(config.get('dashboard', {}))
        self.component_manager = ComponentManager(config.get('components', {}))
        self.realtime_data_manager = RealTimeDataManager(config.get('realtime', {}))
        self.websocket_manager = WebSocketManager(config.get('websocket', {}))
        self.user_interface_manager = UserInterfaceManager(config.get('ui', {}))
        self.api_endpoint_manager = APIEndpointManager(config.get('api', {}))
        self.metrics_collector = DashboardMetricsCollector(config.get('metrics', {}))
        
        # Interface configuration
        self.max_concurrent_dashboards = config.get('max_concurrent_dashboards', 100)
        self.session_timeout = config.get('session_timeout', 3600)
        self.refresh_interval = config.get('refresh_interval', 5)
        
        # Active dashboards tracking
        self.active_dashboards = {}
        self.dashboard_sessions = defaultdict(dict)
        
        # Thread safety
        self._lock = threading.Lock()
        self.is_running = True
        
        # Start background threads
        self._start_background_threads()
        
        self.logger.info("Web Interface Manager initialized")
    
    def render_dashboard(self, user_id: str, dashboard_type: str) -> Dict[str, Any]:
        """Render comprehensive dashboard with real-time data"""
        try:
            start_time = time.time()
            
            # Get user preferences and permissions
            user_preferences = self.user_interface_manager.get_user_preferences(user_id)
            user_permissions = self.user_interface_manager.get_user_permissions(user_id)
            
            # Validate dashboard access
            access_validation = self.user_interface_manager.validate_dashboard_access(
                user_id, dashboard_type, user_permissions
            )
            
            if not access_validation['authorized']:
                self.metrics_collector.record_access_denied(user_id, dashboard_type)
                return {
                    'success': False,
                    'error': 'Access denied',
                    'details': access_validation['details']
                }
            
            # Check dashboard capacity
            if not self._check_dashboard_capacity():
                return {
                    'success': False,
                    'error': 'Dashboard capacity exceeded',
                    'details': f'Maximum {self.max_concurrent_dashboards} dashboards allowed'
                }
            
            # Get real-time data for dashboard
            realtime_data = self.realtime_data_manager.get_dashboard_data(
                dashboard_type, user_preferences
            )
            
            # Render dashboard components
            dashboard_components = self.dashboard_renderer.render_components(
                dashboard_type, realtime_data, user_preferences
            )
            
            # Update component states
            component_states = self.component_manager.update_components(
                dashboard_components, realtime_data
            )
            
            # Create dashboard session
            session_id = self._create_dashboard_session(user_id, dashboard_type)
            
            # Generate dashboard response
            dashboard_response = {
                'success': True,
                'session_id': session_id,
                'dashboard_type': dashboard_type,
                'components': component_states,
                'realtime_data': realtime_data,
                'user_preferences': user_preferences,
                'layout': dashboard_components.get('layout', {}),
                'theme': dashboard_components.get('theme', {}),
                'last_updated': time.time(),
                'websocket_channel': f'dashboard_{user_id}_{dashboard_type}',
                'refresh_interval': self.refresh_interval,
                'metadata': {
                    'render_time': time.time() - start_time,
                    'component_count': len(component_states),
                    'data_freshness': dashboard_components.get('metadata', {}).get('data_freshness', {})
                }
            }
            
            # Track active dashboard
            self._track_active_dashboard(user_id, dashboard_type, session_id)
            
            # Subscribe to WebSocket channel
            self.websocket_manager.subscribe_to_channel(
                user_id, f'dashboard_{user_id}_{dashboard_type}'
            )
            
            # Broadcast initial dashboard state
            self.websocket_manager.broadcast_dashboard_update(
                user_id, dashboard_type, dashboard_response
            )
            
            # Record metrics
            self.metrics_collector.record_dashboard_render(
                user_id, dashboard_type, time.time() - start_time
            )
            
            # Store in history
            self.interface_history.append({
                'timestamp': time.time(),
                'user_id': user_id,
                'dashboard_type': dashboard_type,
                'action': 'render',
                'success': True
            })
            
            return dashboard_response
            
        except Exception as e:
            self.logger.error(f"Dashboard rendering failed: {e}")
            self.metrics_collector.record_dashboard_error(user_id, dashboard_type, str(e))
            
            return {
                'success': False,
                'error': f'Dashboard rendering failed: {str(e)}'
            }
    
    def handle_user_interaction(self, user_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user interactions and update interface accordingly"""
        try:
            start_time = time.time()
            
            # Validate user session
            session_validation = self._validate_user_session(user_id, interaction_data.get('session_id'))
            if not session_validation['valid']:
                return {
                    'success': False,
                    'error': 'Invalid session',
                    'details': session_validation['details']
                }
            
            # Validate user interaction
            interaction_validation = self.user_interface_manager.validate_interaction(
                user_id, interaction_data
            )
            
            if not interaction_validation['valid']:
                return {
                    'success': False,
                    'error': 'Invalid interaction',
                    'details': interaction_validation['details']
                }
            
            # Process user interaction
            interaction_result = self.user_interface_manager.process_interaction(
                user_id, interaction_data
            )
            
            # Update interface based on interaction
            interface_updates = self.component_manager.update_interface_components(
                user_id, interaction_result
            )
            
            # Apply updates to dashboard
            dashboard_type = interaction_data.get('dashboard_type')
            if dashboard_type:
                self._apply_dashboard_updates(user_id, dashboard_type, interface_updates)
            
            # Broadcast interface updates
            self.websocket_manager.broadcast_interface_update(
                user_id, interface_updates
            )
            
            # Record metrics
            self.metrics_collector.record_user_interaction(
                user_id, interaction_data.get('interaction_type'), time.time() - start_time
            )
            
            # Store in history
            self.interface_history.append({
                'timestamp': time.time(),
                'user_id': user_id,
                'interaction_type': interaction_data.get('interaction_type'),
                'action': 'interaction',
                'success': True
            })
            
            return {
                'success': True,
                'interaction_result': interaction_result,
                'interface_updates': interface_updates,
                'processing_time': time.time() - start_time,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"User interaction handling failed: {e}")
            self.metrics_collector.record_interaction_error(user_id, str(e))
            
            return {
                'success': False,
                'error': f'Interaction handling failed: {str(e)}'
            }
    
    def get_api_endpoints(self) -> Dict[str, Any]:
        """Get available API endpoints for web interface"""
        try:
            endpoints = self.api_endpoint_manager.get_all_endpoints()
            
            return {
                'success': True,
                'endpoints': endpoints,
                'base_url': self.config.get('base_url', '/api/v1'),
                'websocket_url': self.config.get('websocket_url', '/ws'),
                'documentation_url': self.config.get('documentation_url', '/docs'),
                'api_version': self.config.get('api_version', '1.0')
            }
            
        except Exception as e:
            self.logger.error(f"API endpoints retrieval failed: {e}")
            return {
                'success': False,
                'error': f'Endpoints retrieval failed: {str(e)}'
            }
    
    def get_websocket_status(self) -> Dict[str, Any]:
        """Get WebSocket connection status and statistics"""
        try:
            websocket_status = self.websocket_manager.get_connection_status()
            
            return {
                'success': True,
                'websocket_status': websocket_status,
                'active_connections': self.websocket_manager.get_active_connections(),
                'connection_statistics': self.websocket_manager.get_connection_statistics(),
                'channel_info': self.websocket_manager.get_channel_info()
            }
            
        except Exception as e:
            self.logger.error(f"WebSocket status retrieval failed: {e}")
            return {
                'success': False,
                'error': f'WebSocket status failed: {str(e)}'
            }
    
    def close_dashboard(self, user_id: str, dashboard_type: str) -> Dict[str, Any]:
        """Close active dashboard and clean up resources"""
        try:
            dashboard_key = f"{user_id}_{dashboard_type}"
            
            with self._lock:
                if dashboard_key in self.active_dashboards:
                    # Remove from active dashboards
                    dashboard_info = self.active_dashboards.pop(dashboard_key)
                    session_id = dashboard_info['session_id']
                    
                    # Clean up session
                    if user_id in self.dashboard_sessions:
                        self.dashboard_sessions[user_id].pop(session_id, None)
                    
                    # Unsubscribe from WebSocket channel
                    self.websocket_manager.unsubscribe_from_channel(
                        user_id, f'dashboard_{user_id}_{dashboard_type}'
                    )
                    
                    # Record metrics
                    duration = time.time() - dashboard_info['start_time']
                    self.metrics_collector.record_dashboard_close(
                        user_id, dashboard_type, duration
                    )
                    
                    return {
                        'success': True,
                        'message': 'Dashboard closed successfully',
                        'session_duration': duration
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Dashboard not found'
                    }
                    
        except Exception as e:
            self.logger.error(f"Dashboard closing failed: {e}")
            return {
                'success': False,
                'error': f'Dashboard closing failed: {str(e)}'
            }
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get comprehensive dashboard metrics"""
        try:
            metrics = self.metrics_collector.get_dashboard_metrics()
            
            # Add current state metrics
            with self._lock:
                metrics['current_state'] = {
                    'active_dashboards': len(self.active_dashboards),
                    'active_sessions': sum(len(sessions) for sessions in self.dashboard_sessions.values()),
                    'websocket_connections': len(self.websocket_manager.get_active_connections())
                }
            
            return {
                'success': True,
                'metrics': metrics,
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Dashboard metrics retrieval failed: {e}")
            return {
                'success': False,
                'error': f'Metrics retrieval failed: {str(e)}'
            }
    
    def _check_dashboard_capacity(self) -> bool:
        """Check if dashboard capacity is available"""
        with self._lock:
            return len(self.active_dashboards) < self.max_concurrent_dashboards
    
    def _create_dashboard_session(self, user_id: str, dashboard_type: str) -> str:
        """Create dashboard session"""
        import secrets
        session_id = f"sess_{int(time.time())}_{secrets.token_urlsafe(8)}"
        
        with self._lock:
            self.dashboard_sessions[user_id][session_id] = {
                'dashboard_type': dashboard_type,
                'created_at': time.time(),
                'last_activity': time.time()
            }
        
        return session_id
    
    def _track_active_dashboard(self, user_id: str, dashboard_type: str, session_id: str):
        """Track active dashboard"""
        dashboard_key = f"{user_id}_{dashboard_type}"
        
        with self._lock:
            self.active_dashboards[dashboard_key] = {
                'user_id': user_id,
                'dashboard_type': dashboard_type,
                'session_id': session_id,
                'start_time': time.time(),
                'last_activity': time.time()
            }
    
    def _validate_user_session(self, user_id: str, session_id: Optional[str]) -> Dict[str, Any]:
        """Validate user session"""
        try:
            if not session_id:
                return {
                    'valid': False,
                    'details': 'No session ID provided'
                }
            
            with self._lock:
                if user_id not in self.dashboard_sessions:
                    return {
                        'valid': False,
                        'details': 'User has no active sessions'
                    }
                
                if session_id not in self.dashboard_sessions[user_id]:
                    return {
                        'valid': False,
                        'details': 'Invalid session ID'
                    }
                
                # Check session timeout
                session = self.dashboard_sessions[user_id][session_id]
                if time.time() - session['last_activity'] > self.session_timeout:
                    # Remove expired session
                    del self.dashboard_sessions[user_id][session_id]
                    return {
                        'valid': False,
                        'details': 'Session expired'
                    }
                
                # Update last activity
                session['last_activity'] = time.time()
                
                return {
                    'valid': True,
                    'session': session
                }
                
        except Exception as e:
            self.logger.error(f"Session validation failed: {e}")
            return {
                'valid': False,
                'details': str(e)
            }
    
    def _apply_dashboard_updates(self, user_id: str, dashboard_type: str, updates: Dict[str, Any]):
        """Apply updates to dashboard"""
        try:
            dashboard_key = f"{user_id}_{dashboard_type}"
            
            with self._lock:
                if dashboard_key in self.active_dashboards:
                    # Update last activity
                    self.active_dashboards[dashboard_key]['last_activity'] = time.time()
                    
                    # Apply component updates
                    for component_id, update_data in updates.items():
                        self.component_manager.apply_component_update(
                            component_id, update_data
                        )
                        
        except Exception as e:
            self.logger.error(f"Dashboard update application failed: {e}")
    
    def _start_background_threads(self):
        """Start background maintenance threads"""
        # Session cleanup thread
        session_thread = threading.Thread(
            target=self._session_cleanup_loop,
            daemon=True
        )
        session_thread.start()
        
        # Dashboard refresh thread
        refresh_thread = threading.Thread(
            target=self._dashboard_refresh_loop,
            daemon=True
        )
        refresh_thread.start()
        
        # Metrics aggregation thread
        metrics_thread = threading.Thread(
            target=self._metrics_aggregation_loop,
            daemon=True
        )
        metrics_thread.start()
    
    def _session_cleanup_loop(self):
        """Clean up expired sessions"""
        while self.is_running:
            try:
                current_time = time.time()
                
                with self._lock:
                    # Clean up expired sessions
                    expired_users = []
                    for user_id, sessions in self.dashboard_sessions.items():
                        expired_sessions = []
                        for session_id, session in sessions.items():
                            if current_time - session['last_activity'] > self.session_timeout:
                                expired_sessions.append(session_id)
                        
                        # Remove expired sessions
                        for session_id in expired_sessions:
                            del sessions[session_id]
                        
                        # Mark users with no sessions
                        if not sessions:
                            expired_users.append(user_id)
                    
                    # Remove users with no sessions
                    for user_id in expired_users:
                        del self.dashboard_sessions[user_id]
                    
                    # Clean up inactive dashboards
                    inactive_dashboards = []
                    for dashboard_key, dashboard_info in self.active_dashboards.items():
                        if current_time - dashboard_info['last_activity'] > self.session_timeout:
                            inactive_dashboards.append(dashboard_key)
                    
                    # Remove inactive dashboards
                    for dashboard_key in inactive_dashboards:
                        dashboard_info = self.active_dashboards.pop(dashboard_key)
                        self.logger.info(f"Removed inactive dashboard: {dashboard_key}")
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                self.logger.error(f"Session cleanup error: {e}")
                time.sleep(300)  # 5 minutes on error
    
    def _dashboard_refresh_loop(self):
        """Refresh active dashboards with new data"""
        while self.is_running:
            try:
                with self._lock:
                    active_dashboards = list(self.active_dashboards.items())
                
                for dashboard_key, dashboard_info in active_dashboards:
                    try:
                        user_id = dashboard_info['user_id']
                        dashboard_type = dashboard_info['dashboard_type']
                        
                        # Get user preferences
                        user_preferences = self.user_interface_manager.get_user_preferences(user_id)
                        
                        # Get fresh data
                        realtime_data = self.realtime_data_manager.get_dashboard_data(
                            dashboard_type, user_preferences
                        )
                        
                        # Prepare update
                        update_data = {
                            'type': 'data_refresh',
                            'realtime_data': realtime_data,
                            'timestamp': time.time()
                        }
                        
                        # Broadcast update
                        self.websocket_manager.broadcast_dashboard_update(
                            user_id, dashboard_type, update_data
                        )
                        
                    except Exception as e:
                        self.logger.error(f"Dashboard refresh failed for {dashboard_key}: {e}")
                
                time.sleep(self.refresh_interval)
                
            except Exception as e:
                self.logger.error(f"Dashboard refresh loop error: {e}")
                time.sleep(30)  # 30 seconds on error
    
    def _metrics_aggregation_loop(self):
        """Aggregate dashboard metrics"""
        while self.is_running:
            try:
                # Aggregate metrics
                self.metrics_collector.aggregate_metrics()
                
                # Clean up old metrics
                self.metrics_collector.cleanup_old_metrics()
                
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Metrics aggregation error: {e}")
                time.sleep(600)  # 10 minutes on error
    
    def shutdown(self):
        """Shutdown web interface manager"""
        self.logger.info("Shutting down Web Interface Manager")
        self.is_running = False
        
        # Close all active dashboards
        with self._lock:
            for dashboard_key in list(self.active_dashboards.keys()):
                dashboard_info = self.active_dashboards[dashboard_key]
                self.close_dashboard(dashboard_info['user_id'], dashboard_info['dashboard_type'])
        
        # Shutdown components
        self.websocket_manager.shutdown()
        self.realtime_data_manager.shutdown()