"""
Accuracy Dashboard Integration for Core AI System
Provides clean integration between training manager and accuracy visualization
"""

import logging
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
import sys
from pathlib import Path

# Add path for core imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from .dashboard_bridge import dashboard_bridge

logger = logging.getLogger(__name__)


class AccuracyDashboardIntegration:
    """
    Integration layer between core training system and accuracy dashboards
    Handles real-time metrics flow and dashboard coordination
    """
    
    def __init__(self, training_manager=None):
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        self.training_manager = training_manager
        self._active_accuracy_dashboards: Dict[str, Any] = {}
        self._metrics_cache = {}
        
    def register_accuracy_dashboard(self, domain_name: str, dashboard_factory) -> bool:
        """Register an accuracy dashboard for a domain"""
        try:
            return dashboard_bridge.register_domain_dashboard(
                domain_name=domain_name,
                dashboard_factory=dashboard_factory,
                config={'dashboard_type': 'accuracy'}
            )
        except Exception as e:
            self.logger.error(f"Failed to register accuracy dashboard for {domain_name}: {e}")
            return False
    
    def create_accuracy_dashboard(self, domain_name: str, **kwargs) -> Optional[Any]:
        """Create accuracy dashboard for domain"""
        try:
            dashboard = dashboard_bridge.create_dashboard(domain_name, **kwargs)
            if dashboard:
                with self._lock:
                    self._active_accuracy_dashboards[domain_name] = dashboard
                    
                # Connect to training manager if available
                if self.training_manager:
                    self._connect_training_metrics(domain_name, dashboard)
                    
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Failed to create accuracy dashboard for {domain_name}: {e}")
            return None
    
    def update_training_metrics(self, domain_name: str, model_id: str, 
                              epoch: int, metrics: Dict[str, Any]) -> None:
        """Update training metrics for dashboard"""
        try:
            with self._lock:
                dashboard = self._active_accuracy_dashboards.get(domain_name)
                if dashboard and hasattr(dashboard, 'handle_training_accuracy_update'):
                    dashboard.handle_training_accuracy_update(
                        model_id=model_id,
                        epoch=epoch,
                        train_metrics=metrics.get('train_metrics', {}),
                        val_metrics=metrics.get('val_metrics', {})
                    )
                    
                # Cache metrics
                cache_key = f"{domain_name}_{model_id}"
                self._metrics_cache[cache_key] = {
                    'epoch': epoch,
                    'metrics': metrics,
                    'timestamp': datetime.now()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to update training metrics for {domain_name}: {e}")
    
    def get_cached_metrics(self, domain_name: str, model_id: str) -> Optional[Dict[str, Any]]:
        """Get cached metrics for a domain/model"""
        with self._lock:
            cache_key = f"{domain_name}_{model_id}"
            return self._metrics_cache.get(cache_key)
    
    def _connect_training_metrics(self, domain_name: str, dashboard: Any) -> None:
        """Connect dashboard to training manager metrics"""
        try:
            # This would integrate with the actual training manager
            # For now, we set up the connection structure
            self.logger.info(f"Connected training metrics for {domain_name} dashboard")
            
        except Exception as e:
            self.logger.error(f"Failed to connect training metrics for {domain_name}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of accuracy dashboard integration"""
        with self._lock:
            return {
                'active_accuracy_dashboards': list(self._active_accuracy_dashboards.keys()),
                'cached_metrics_count': len(self._metrics_cache),
                'training_manager_connected': self.training_manager is not None,
                'last_updated': datetime.now().isoformat()
            }


# Global accuracy dashboard integration instance
accuracy_integration = AccuracyDashboardIntegration()