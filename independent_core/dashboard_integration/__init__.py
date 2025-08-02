"""
Dashboard Integration Hub for Independent Core
Provides clean integration between core AI system and domain-specific dashboards
"""

from .dashboard_bridge import DashboardBridge
from .accuracy_integration import AccuracyDashboardIntegration

__all__ = ['DashboardBridge', 'AccuracyDashboardIntegration']