"""
Saraphis Production Web Interface & Dashboard System
Production-ready web interface with real-time dashboards
"""

# Import core components first (no external dependencies)
from .dashboard_renderer import DashboardRenderer, ThemeManager

# Import components with potential dependencies
try:
    from .web_interface_manager import WebInterfaceManager
    from .component_manager import ComponentManager
    from .realtime_data_manager import RealTimeDataManager, CacheManager
    from .websocket_manager import WebSocketManager
    from .user_interface_manager import UserInterfaceManager
    from .api_endpoint_manager import APIEndpointManager
    from .dashboard_metrics import DashboardMetricsCollector
    
    _ALL_IMPORTS_SUCCESSFUL = True
except ImportError as e:
    # Log the import error but don't fail the entire module
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Some production_web components unavailable due to missing dependencies: {e}")
    
    # Define minimal exports
    WebInterfaceManager = None
    ComponentManager = None
    RealTimeDataManager = None
    CacheManager = None
    WebSocketManager = None
    UserInterfaceManager = None
    APIEndpointManager = None
    DashboardMetricsCollector = None
    
    _ALL_IMPORTS_SUCCESSFUL = False

__all__ = [
    'WebInterfaceManager',
    'DashboardRenderer',
    'ThemeManager',
    'ComponentManager',
    'RealTimeDataManager',
    'CacheManager',
    'WebSocketManager',
    'UserInterfaceManager',
    'APIEndpointManager',
    'DashboardMetricsCollector'
]