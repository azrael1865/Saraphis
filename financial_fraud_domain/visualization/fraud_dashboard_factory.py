"""
Fraud Detection Dashboard Factory
Creates domain-specific dashboards for fraud detection accuracy monitoring
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add paths for imports
current_dir = Path(__file__).parent.parent
core_dir = current_dir.parent / 'independent_core'
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(core_dir))

from accuracy_visualizer import FraudAccuracyVisualizer

logger = logging.getLogger(__name__)


class FraudDashboardFactory:
    """
    Factory for creating fraud detection dashboards
    Integrates with core dashboard bridge through clean interface
    """
    
    @staticmethod
    def create_accuracy_dashboard(**kwargs) -> Optional[FraudAccuracyVisualizer]:
        """
        Create fraud-specific accuracy dashboard
        
        Args:
            **kwargs: Dashboard configuration options
            
        Returns:
            FraudAccuracyVisualizer instance or None
        """
        try:
            # Extract configuration
            config = kwargs.get('config', {})
            dashboard_type = kwargs.get('dashboard_type', 'accuracy')
            
            # Create fraud accuracy visualizer
            visualizer = FraudAccuracyVisualizer(config=config)
            
            logger.info("Created fraud accuracy dashboard")
            return visualizer
            
        except Exception as e:
            logger.error(f"Failed to create fraud accuracy dashboard: {e}")
            return None
    
    @staticmethod
    def create_performance_dashboard(**kwargs) -> Optional[Any]:
        """Create fraud-specific performance dashboard"""
        try:
            # Future: Implement performance dashboard
            logger.info("Performance dashboard creation requested (not yet implemented)")
            return None
            
        except Exception as e:
            logger.error(f"Failed to create fraud performance dashboard: {e}")
            return None
    
    @staticmethod
    def create_real_time_dashboard(**kwargs) -> Optional[Any]:
        """Create fraud-specific real-time monitoring dashboard"""
        try:
            # Future: Implement real-time dashboard
            logger.info("Real-time dashboard creation requested (not yet implemented)")
            return None
            
        except Exception as e:
            logger.error(f"Failed to create fraud real-time dashboard: {e}")
            return None


def register_fraud_dashboards():
    """Register fraud dashboards with core bridge"""
    try:
        from independent_core.dashboard_integration import dashboard_bridge
        
        # Register accuracy dashboard
        success = dashboard_bridge.register_domain_dashboard(
            domain_name='fraud_detection',
            dashboard_factory=FraudDashboardFactory.create_accuracy_dashboard,
            config={'domain_type': 'fraud_detection', 'visualization_type': 'accuracy'}
        )
        
        if success:
            logger.info("Successfully registered fraud dashboards with core bridge")
        else:
            logger.error("Failed to register fraud dashboards")
            
        return success
        
    except Exception as e:
        logger.error(f"Failed to register fraud dashboards: {e}")
        return False