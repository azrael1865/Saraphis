"""
Fraud Domain Dashboard Integration Entry Point
Provides clean interface for integrating fraud detection dashboards with core system
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add paths for core imports
core_dir = Path(__file__).parent.parent / 'independent_core'
sys.path.insert(0, str(core_dir))

from dashboard_integration.dashboard_bridge import dashboard_bridge
from dashboard_integration.accuracy_integration import accuracy_integration
from .visualization.fraud_dashboard_factory import register_fraud_dashboards, FraudDashboardFactory

logger = logging.getLogger(__name__)


class FraudDashboardIntegration:
    """
    Main integration class for fraud detection dashboards
    Provides simple interface for setting up and managing fraud dashboards
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.is_initialized = False
        self.active_dashboards = {}
    
    def initialize(self) -> bool:
        """
        Initialize fraud dashboard integration
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Register fraud dashboards with core bridge
            success = register_fraud_dashboards()
            
            if success:
                self.is_initialized = True
                self.logger.info("Fraud dashboard integration initialized successfully")
            else:
                self.logger.error("Failed to register fraud dashboards")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize fraud dashboard integration: {e}")
            return False
    
    def create_accuracy_dashboard(self, model_id: str = None, 
                                config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Create fraud accuracy dashboard
        
        Args:
            model_id: Optional model ID to track
            config: Optional dashboard configuration
            
        Returns:
            Dashboard instance or None
        """
        try:
            if not self.is_initialized:
                self.logger.warning("Integration not initialized, attempting to initialize now")
                if not self.initialize():
                    return None
            
            # Create dashboard through core bridge
            dashboard = accuracy_integration.create_accuracy_dashboard(
                domain_name='fraud_detection',
                config=config or {},
                model_id=model_id
            )
            
            if dashboard:
                dashboard_id = f"fraud_accuracy_{model_id or 'default'}"
                self.active_dashboards[dashboard_id] = dashboard
                self.logger.info(f"Created fraud accuracy dashboard: {dashboard_id}")
            
            return dashboard
            
        except Exception as e:
            self.logger.error(f"Failed to create fraud accuracy dashboard: {e}")
            return None
    
    def update_training_metrics(self, model_id: str, epoch: int, 
                              train_metrics: Dict[str, Any], 
                              val_metrics: Dict[str, Any]) -> None:
        """
        Update training metrics for fraud dashboards
        
        Args:
            model_id: Model identifier
            epoch: Training epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        try:
            accuracy_integration.update_training_metrics(
                domain_name='fraud_detection',
                model_id=model_id,
                epoch=epoch,
                metrics={'train_metrics': train_metrics, 'val_metrics': val_metrics}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update training metrics: {e}")
    
    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get status of fraud dashboards"""
        try:
            return {
                'initialized': self.is_initialized,
                'active_dashboards': list(self.active_dashboards.keys()),
                'bridge_status': dashboard_bridge.get_status(),
                'accuracy_integration_status': accuracy_integration.get_status()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard status: {e}")
            return {'error': str(e)}
    
    def shutdown(self) -> None:
        """Shutdown fraud dashboard integration"""
        try:
            # Deactivate dashboards
            dashboard_bridge.deactivate_dashboard('fraud_detection')
            
            # Clear active dashboards
            self.active_dashboards.clear()
            self.is_initialized = False
            
            self.logger.info("Fraud dashboard integration shutdown")
            
        except Exception as e:
            self.logger.error(f"Failed to shutdown fraud dashboard integration: {e}")


# Global integration instance
fraud_dashboard_integration = FraudDashboardIntegration()


def setup_fraud_dashboards() -> bool:
    """
    Convenience function to set up fraud dashboards
    
    Returns:
        bool: True if setup successful
    """
    return fraud_dashboard_integration.initialize()


def create_fraud_accuracy_dashboard(model_id: str = None) -> Optional[Any]:
    """
    Convenience function to create fraud accuracy dashboard
    
    Args:
        model_id: Optional model ID to track
        
    Returns:
        Dashboard instance or None
    """
    return fraud_dashboard_integration.create_accuracy_dashboard(model_id=model_id)