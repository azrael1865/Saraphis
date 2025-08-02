"""
Fraud Accuracy Visualizer
Clean domain-specific implementation of accuracy visualization for fraud detection
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import threading

# Add paths for imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

logger = logging.getLogger(__name__)


class FraudAccuracyVisualizer:
    """
    Fraud-specific accuracy visualization component
    Handles accuracy metrics visualization for fraud detection models
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        self.config = config or {}
        
        # Fraud-specific accuracy tracking
        self.fraud_metrics_cache = {}
        self.confusion_matrix_history = []
        self.model_performance_trends = {}
        
        # Integration with existing fraud domain components
        self._initialize_fraud_components()
        
    def _initialize_fraud_components(self):
        """Initialize integration with existing fraud domain components"""
        try:
            # Import existing fraud domain accuracy components
            from ..accuracy_tracking_db import AccuracyTrackingDatabase
            from ..real_time_accuracy_monitor import RealTimeAccuracyMonitor
            
            self.accuracy_db = AccuracyTrackingDatabase()
            self.real_time_monitor = RealTimeAccuracyMonitor()
            
            self.logger.info("Initialized fraud accuracy visualizer with domain components")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize all fraud components: {e}")
            # Graceful fallback - visualizer can still work without full integration
            self.accuracy_db = None
            self.real_time_monitor = None
    
    def handle_training_accuracy_update(self, model_id: str, epoch: int, 
                                      train_metrics: Dict[str, Any], 
                                      val_metrics: Dict[str, Any]) -> None:
        """
        Handle training accuracy updates from core training system
        
        Args:
            model_id: ID of the fraud detection model
            epoch: Current training epoch
            train_metrics: Training metrics including accuracy, precision, recall, F1
            val_metrics: Validation metrics
        """
        try:
            with self._lock:
                # Cache metrics for visualization
                cache_key = f"{model_id}_epoch_{epoch}"
                self.fraud_metrics_cache[cache_key] = {
                    'model_id': model_id,
                    'epoch': epoch,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'timestamp': datetime.now(),
                    'fraud_specific': {
                        'fraud_detection_rate': train_metrics.get('recall', 0),  # Key fraud metric
                        'false_positive_rate': self._calculate_false_positive_rate(train_metrics),
                        'precision_at_threshold': train_metrics.get('precision', 0)
                    }
                }
                
                # Update performance trends
                if model_id not in self.model_performance_trends:
                    self.model_performance_trends[model_id] = []
                
                self.model_performance_trends[model_id].append({
                    'epoch': epoch,
                    'timestamp': datetime.now(),
                    'fraud_detection_accuracy': val_metrics.get('accuracy', 0),
                    'precision': val_metrics.get('precision', 0),
                    'recall': val_metrics.get('recall', 0),
                    'f1_score': val_metrics.get('f1_score', 0)
                })
                
                # Store in database if available
                if self.accuracy_db:
                    try:
                        self.accuracy_db.record_accuracy_metrics(
                            model_id=model_id,
                            model_version=f"epoch_{epoch}",
                            y_true=train_metrics.get('y_true', []),
                            y_pred=train_metrics.get('y_pred', []),
                            y_proba=train_metrics.get('y_proba'),
                            data_type='training'
                        )
                    except Exception as e:
                        self.logger.warning(f"Could not store metrics in database: {e}")
                
                self.logger.debug(f"Updated fraud accuracy metrics for model {model_id}, epoch {epoch}")
                
        except Exception as e:
            self.logger.error(f"Failed to handle training accuracy update: {e}")
    
    def _calculate_false_positive_rate(self, metrics: Dict[str, Any]) -> float:
        """Calculate false positive rate from metrics"""
        try:
            # Try to calculate from confusion matrix if available
            if 'confusion_matrix' in metrics:
                cm = metrics['confusion_matrix']
                if len(cm) >= 2 and len(cm[0]) >= 2:
                    fp = cm[0][1]  # False positives
                    tn = cm[0][0]  # True negatives
                    return fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            # Fallback calculation
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            if precision > 0 and recall > 0:
                # Approximate FPR calculation
                return max(0, 1 - precision)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def get_fraud_detection_summary(self, model_id: str) -> Dict[str, Any]:
        """Get fraud detection performance summary for a model"""
        try:
            with self._lock:
                trends = self.model_performance_trends.get(model_id, [])
                if not trends:
                    return {'error': 'No performance data available'}
                
                latest = trends[-1]
                
                # Calculate improvement over time
                if len(trends) > 1:
                    first = trends[0]
                    accuracy_improvement = latest['fraud_detection_accuracy'] - first['fraud_detection_accuracy']
                    precision_improvement = latest['precision'] - first['precision']
                else:
                    accuracy_improvement = 0
                    precision_improvement = 0
                
                return {
                    'model_id': model_id,
                    'latest_epoch': latest['epoch'],
                    'current_metrics': {
                        'fraud_detection_accuracy': latest['fraud_detection_accuracy'],
                        'precision': latest['precision'],
                        'recall': latest['recall'],
                        'f1_score': latest['f1_score']
                    },
                    'improvements': {
                        'accuracy_improvement': accuracy_improvement,
                        'precision_improvement': precision_improvement
                    },
                    'total_epochs_trained': len(trends),
                    'last_updated': latest['timestamp'].isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get fraud detection summary: {e}")
            return {'error': str(e)}
    
    def get_cached_metrics(self, model_id: str, epoch: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get cached metrics for model/epoch"""
        try:
            with self._lock:
                if epoch is not None:
                    cache_key = f"{model_id}_epoch_{epoch}"
                    return self.fraud_metrics_cache.get(cache_key)
                else:
                    # Return latest metrics for model
                    model_metrics = [
                        metrics for key, metrics in self.fraud_metrics_cache.items()
                        if metrics.get('model_id') == model_id
                    ]
                    if model_metrics:
                        return max(model_metrics, key=lambda x: x.get('epoch', 0))
                    return None
                    
        except Exception as e:
            self.logger.error(f"Failed to get cached metrics: {e}")
            return None
    
    def create_dashboard_layout(self) -> Dict[str, Any]:
        """Create dashboard layout configuration for fraud detection"""
        return {
            'dashboard_type': 'fraud_accuracy',
            'title': 'Fraud Detection Accuracy Dashboard',
            'widgets': [
                {
                    'type': 'accuracy_trend',
                    'title': 'Fraud Detection Accuracy Over Time',
                    'metrics': ['accuracy', 'precision', 'recall', 'f1_score']
                },
                {
                    'type': 'confusion_matrix',
                    'title': 'Fraud vs Legitimate Classification',
                    'labels': ['Legitimate', 'Fraud']
                },
                {
                    'type': 'performance_cards',
                    'title': 'Key Fraud Detection Metrics',
                    'metrics': ['fraud_detection_rate', 'false_positive_rate', 'precision_at_threshold']
                }
            ],
            'real_time_enabled': True,
            'export_options': ['pdf', 'png', 'csv']
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of fraud accuracy visualizer"""
        with self._lock:
            return {
                'cached_metrics_count': len(self.fraud_metrics_cache),
                'models_tracked': len(self.model_performance_trends),
                'database_connected': self.accuracy_db is not None,
                'real_time_monitor_connected': self.real_time_monitor is not None,
                'last_update': datetime.now().isoformat()
            }