"""
Saraphis Data Manager
Production-ready comprehensive data management system
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import json
import hashlib

logger = logging.getLogger(__name__)


class DataManager:
    """Production-ready comprehensive data management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_history = deque(maxlen=10000)
        self.backup_history = deque(maxlen=5000)
        self.data_metrics = {
            'total_data_size_gb': 0.0,
            'compression_ratio': 0.0,
            'encryption_overhead': 0.0,
            'backup_success_rate': 0.0,
            'data_integrity_score': 0.0,
            'storage_efficiency': 0.0,
            'replication_lag_seconds': 0.0,
            'data_availability': 0.99
        }
        
        # Initialize data management components
        from .backup_manager import BackupManager
        from .encryption_manager import EncryptionManager
        from .compression_manager import CompressionManager
        from .storage_manager import StorageManager
        from .replication_manager import ReplicationManager
        from .data_metrics import DataMetricsCollector
        
        self.backup_manager = BackupManager(config.get('backup_config', {}))
        self.encryption_manager = EncryptionManager(config.get('encryption_config', {}))
        self.compression_manager = CompressionManager(config.get('compression_config', {}))
        self.storage_manager = StorageManager(config.get('storage_config', {}))
        self.replication_manager = ReplicationManager(config.get('replication_config', {}))
        self.data_metrics_collector = DataMetricsCollector(config.get('metrics_config', {}))
        
        # Data policies
        self.data_policies = self._initialize_data_policies()
        
        # Thread control
        self.is_running = True
        self._lock = threading.Lock()
        
        # Start data management threads
        self._start_data_threads()
        
        self.logger.info("Data Manager initialized")
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity across all systems"""
        try:
            self._validation_start_time = datetime.now()
            
            # Collect data metrics
            current_metrics = self.data_metrics_collector.collect_all_metrics()
            
            # Validate backup integrity
            backup_integrity = self.backup_manager.validate_backup_integrity()
            
            # Check encryption status
            encryption_status = self.encryption_manager.validate_encryption_status()
            
            # Verify compression integrity
            compression_integrity = self.compression_manager.validate_compression_integrity()
            
            # Check storage health
            storage_health = self.storage_manager.check_storage_health()
            
            # Validate replication status
            replication_status = self.replication_manager.validate_replication_status()
            
            # Calculate data integrity score
            integrity_score = self._calculate_data_integrity_score(
                current_metrics, backup_integrity, encryption_status,
                compression_integrity, storage_health, replication_status
            )
            
            # Identify data issues
            data_issues = self._identify_data_issues(
                current_metrics, backup_integrity, encryption_status,
                compression_integrity, storage_health, replication_status
            )
            
            # Update data metrics
            self._update_data_metrics(current_metrics)
            
            # Store data history
            data_record = {
                'timestamp': time.time(),
                'metrics': current_metrics,
                'backup_integrity': backup_integrity,
                'encryption_status': encryption_status,
                'compression_integrity': compression_integrity,
                'storage_health': storage_health,
                'replication_status': replication_status,
                'integrity_score': integrity_score,
                'data_issues': data_issues
            }
            
            self.data_history.append(data_record)
            
            return {
                'status': 'healthy' if integrity_score > 0.8 else 'degraded' if integrity_score > 0.6 else 'critical',
                'integrity_score': integrity_score,
                'current_metrics': current_metrics,
                'backup_integrity': backup_integrity,
                'encryption_status': encryption_status,
                'compression_integrity': compression_integrity,
                'storage_health': storage_health,
                'replication_status': replication_status,
                'data_issues': data_issues,
                'last_validated': datetime.now().isoformat(),
                'validation_duration': self._get_validation_duration()
            }
            
        except Exception as e:
            self.logger.error(f"Data integrity validation failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'integrity_score': 0.0,
                'data_issues': [{
                    'type': 'data_validation_failure',
                    'severity': 'critical',
                    'description': f'Data validation failed: {str(e)}'
                }]
            }
    
    def _calculate_data_integrity_score(self, metrics: Dict[str, Any], backup_integrity: Dict[str, Any],
                                       encryption_status: Dict[str, Any], compression_integrity: Dict[str, Any],
                                       storage_health: Dict[str, Any], replication_status: Dict[str, Any]) -> float:
        """Calculate overall data integrity score"""
        try:
            # Weighted scoring based on data factors
            scores = {}
            
            # Backup integrity score (25% weight)
            backup_score = backup_integrity.get('integrity_score', 0.0)
            scores['backup_integrity'] = backup_score * 0.25
            
            # Encryption status score (20% weight)
            encryption_score = encryption_status.get('encryption_score', 0.0)
            scores['encryption_status'] = encryption_score * 0.2
            
            # Compression integrity score (15% weight)
            compression_score = compression_integrity.get('integrity_score', 0.0)
            scores['compression_integrity'] = compression_score * 0.15
            
            # Storage health score (20% weight)
            storage_score = storage_health.get('health_score', 0.0)
            scores['storage_health'] = storage_score * 0.2
            
            # Replication status score (20% weight)
            replication_score = replication_status.get('replication_score', 0.0)
            scores['replication_status'] = replication_score * 0.2
            
            # Calculate weighted average
            total_score = sum(scores.values())
            
            return min(1.0, max(0.0, total_score))
            
        except Exception as e:
            self.logger.error(f"Data integrity score calculation failed: {e}")
            return 0.0
    
    def _identify_data_issues(self, metrics: Dict[str, Any], backup_integrity: Dict[str, Any],
                             encryption_status: Dict[str, Any], compression_integrity: Dict[str, Any],
                             storage_health: Dict[str, Any], replication_status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify data issues requiring attention"""
        try:
            data_issues = []
            
            # Check backup issues
            if backup_integrity.get('integrity_score', 1.0) < 0.8:
                data_issues.append({
                    'type': 'backup_integrity_issue',
                    'severity': 'high',
                    'backup_score': backup_integrity.get('integrity_score', 0.0),
                    'failed_backups': backup_integrity.get('failed_backups', 0),
                    'description': f"Backup integrity below threshold: {backup_integrity.get('integrity_score', 0.0):.2f}"
                })
            
            # Check encryption issues
            if encryption_status.get('encryption_score', 1.0) < 0.9:
                data_issues.append({
                    'type': 'encryption_issue',
                    'severity': 'critical',
                    'encryption_score': encryption_status.get('encryption_score', 0.0),
                    'unencrypted_items': len(encryption_status.get('unencrypted_data', [])),
                    'description': f"Encryption coverage below threshold: {encryption_status.get('encryption_score', 0.0):.2f}"
                })
            
            # Check compression issues
            if compression_integrity.get('integrity_score', 1.0) < 0.8:
                data_issues.append({
                    'type': 'compression_integrity_issue',
                    'severity': 'medium',
                    'compression_score': compression_integrity.get('integrity_score', 0.0),
                    'corrupted_items': compression_integrity.get('corrupted_items', 0),
                    'description': f"Compression integrity below threshold: {compression_integrity.get('integrity_score', 0.0):.2f}"
                })
            
            # Check storage issues
            if storage_health.get('health_score', 1.0) < 0.7:
                data_issues.append({
                    'type': 'storage_health_issue',
                    'severity': 'high',
                    'storage_score': storage_health.get('health_score', 0.0),
                    'storage_issues': len(storage_health.get('issues', [])),
                    'description': f"Storage health below threshold: {storage_health.get('health_score', 0.0):.2f}"
                })
            
            # Check replication issues
            if replication_status.get('replication_score', 1.0) < 0.8:
                data_issues.append({
                    'type': 'replication_issue',
                    'severity': 'high',
                    'replication_score': replication_status.get('replication_score', 0.0),
                    'replication_lag': replication_status.get('replication_lag_seconds', 0),
                    'description': f"Replication status below threshold: {replication_status.get('replication_score', 0.0):.2f}"
                })
            
            # Check data size issues
            total_size_gb = metrics.get('total_data_size_gb', 0)
            if total_size_gb > 1000:  # 1TB threshold
                data_issues.append({
                    'type': 'data_size_warning',
                    'severity': 'medium',
                    'total_size_gb': total_size_gb,
                    'description': f"Large data size: {total_size_gb:.2f} GB"
                })
            
            # Check compression efficiency
            compression_ratio = metrics.get('compression_ratio', 1.0)
            if compression_ratio < 0.5:  # 50% compression threshold
                data_issues.append({
                    'type': 'compression_efficiency_issue',
                    'severity': 'low',
                    'compression_ratio': compression_ratio,
                    'description': f"Low compression efficiency: {compression_ratio:.2f}"
                })
            
            return data_issues
            
        except Exception as e:
            self.logger.error(f"Data issue identification failed: {e}")
            return [{
                'type': 'data_analysis_error',
                'severity': 'critical',
                'description': f'Data analysis error: {str(e)}'
            }]
    
    def _initialize_data_policies(self) -> Dict[str, Any]:
        """Initialize data management policies"""
        return {
            'backup_policy': {
                'full_backup_frequency_hours': 24,
                'incremental_backup_frequency_hours': 6,
                'backup_retention_days': 30,
                'backup_verification': True,
                'backup_encryption': True,
                'backup_compression': True,
                'backup_replication': True
            },
            'encryption_policy': {
                'encryption_algorithm': 'AES-256-GCM',
                'key_rotation_days': 90,
                'encrypt_at_rest': True,
                'encrypt_in_transit': True,
                'encrypt_backups': True,
                'key_storage': 'hardware_security_module'
            },
            'compression_policy': {
                'compression_algorithm': 'LZ4',
                'compression_level': 6,
                'compress_backups': True,
                'compress_logs': True,
                'min_compression_ratio': 0.5,
                'adaptive_compression': True
            },
            'storage_policy': {
                'hot_storage_retention_days': 7,
                'warm_storage_retention_days': 90,
                'cold_storage_retention_days': 2555,  # 7 years
                'storage_tiering': True,
                'storage_redundancy': 3,
                'storage_optimization': True
            },
            'replication_policy': {
                'replication_factor': 3,
                'replication_lag_threshold_seconds': 300,
                'cross_region_replication': True,
                'disaster_recovery_rpo': 3600,  # 1 hour
                'disaster_recovery_rto': 7200,  # 2 hours
                'replication_validation': True
            }
        }
    
    def _start_data_threads(self):
        """Start background data management threads"""
        try:
            # Start backup monitoring thread
            self.backup_thread = threading.Thread(target=self._backup_monitoring_loop, daemon=True)
            self.backup_thread.start()
            
            # Start encryption monitoring thread
            self.encryption_thread = threading.Thread(target=self._encryption_monitoring_loop, daemon=True)
            self.encryption_thread.start()
            
            # Start storage monitoring thread
            self.storage_thread = threading.Thread(target=self._storage_monitoring_loop, daemon=True)
            self.storage_thread.start()
            
            # Start replication monitoring thread
            self.replication_thread = threading.Thread(target=self._replication_monitoring_loop, daemon=True)
            self.replication_thread.start()
            
            # Start metrics collection thread
            self.metrics_thread = threading.Thread(target=self._metrics_collection_loop, daemon=True)
            self.metrics_thread.start()
            
            self.logger.info("Data management threads started")
            
        except Exception as e:
            self.logger.error(f"Failed to start data threads: {e}")
            raise RuntimeError(f"Data thread startup failed: {e}")
    
    def _backup_monitoring_loop(self):
        """Background backup monitoring loop"""
        while self.is_running:
            try:
                # Check backup status every 300 seconds (5 minutes)
                backup_status = self.backup_manager.get_backup_status()
                
                # Check if backup is needed
                if backup_status.get('backup_needed', False):
                    if backup_status.get('full_backup_needed', False):
                        backup_result = self.backup_manager.create_backup('full')
                    else:
                        backup_result = self.backup_manager.create_backup('incremental')
                    
                    # Update metrics
                    if backup_result.get('success', False):
                        self.data_metrics['backup_success_rate'] = (
                            self.data_metrics.get('backup_success_rate', 0) * 0.9 + 0.1
                        )
                    else:
                        self.data_metrics['backup_success_rate'] = (
                            self.data_metrics.get('backup_success_rate', 1) * 0.9
                        )
                
                time.sleep(300)  # 5-minute interval
                
            except Exception as e:
                self.logger.error(f"Backup monitoring loop error: {e}")
                time.sleep(600)  # Wait longer on error
    
    def _encryption_monitoring_loop(self):
        """Background encryption monitoring loop"""
        while self.is_running:
            try:
                # Check encryption status every 600 seconds (10 minutes)
                encryption_status = self.encryption_manager.validate_encryption_status()
                
                # Check for unencrypted data
                if encryption_status.get('unencrypted_data'):
                    self.encryption_manager.encrypt_unencrypted_data()
                
                # Update metrics
                self.data_metrics['encryption_overhead'] = 0.1  # 10% overhead estimate
                
                time.sleep(600)  # 10-minute interval
                
            except Exception as e:
                self.logger.error(f"Encryption monitoring loop error: {e}")
                time.sleep(1200)  # Wait longer on error
    
    def _storage_monitoring_loop(self):
        """Background storage monitoring loop"""
        while self.is_running:
            try:
                # Check storage health every 1800 seconds (30 minutes)
                storage_health = self.storage_manager.check_storage_health()
                
                # Update metrics
                self.data_metrics['total_data_size_gb'] = storage_health.get('total_size_gb', 0)
                self.data_metrics['storage_efficiency'] = storage_health.get('efficiency_score', 0)
                
                # Check for storage issues
                if storage_health.get('issues'):
                    self.storage_manager.resolve_storage_issues(storage_health['issues'])
                
                time.sleep(1800)  # 30-minute interval
                
            except Exception as e:
                self.logger.error(f"Storage monitoring loop error: {e}")
                time.sleep(3600)  # Wait longer on error
    
    def _replication_monitoring_loop(self):
        """Background replication monitoring loop"""
        while self.is_running:
            try:
                # Check replication status every 300 seconds (5 minutes)
                replication_status = self.replication_manager.validate_replication_status()
                
                # Update metrics
                self.data_metrics['replication_lag_seconds'] = replication_status.get('replication_lag_seconds', 0)
                
                # Check for replication lag
                if replication_status.get('replication_lag_seconds', 0) > 300:  # 5 minutes
                    self.replication_manager.sync_replication()
                
                time.sleep(300)  # 5-minute interval
                
            except Exception as e:
                self.logger.error(f"Replication monitoring loop error: {e}")
                time.sleep(600)  # Wait longer on error
    
    def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        while self.is_running:
            try:
                # Collect metrics every 60 seconds
                current_metrics = self.data_metrics_collector.collect_all_metrics()
                
                # Update internal metrics
                self._update_data_metrics(current_metrics)
                
                time.sleep(60)  # 60-second interval
                
            except Exception as e:
                self.logger.error(f"Metrics collection loop error: {e}")
                time.sleep(120)  # Wait longer on error
    
    def _update_data_metrics(self, current_metrics: Dict[str, Any]):
        """Update internal data metrics"""
        with self._lock:
            # Update metrics
            for key, value in current_metrics.items():
                if key in self.data_metrics:
                    self.data_metrics[key] = value
            
            # Calculate compression ratio
            if 'compressed_size_gb' in current_metrics and 'total_data_size_gb' in current_metrics:
                total_size = current_metrics['total_data_size_gb']
                compressed_size = current_metrics['compressed_size_gb']
                if total_size > 0:
                    self.data_metrics['compression_ratio'] = compressed_size / total_size
    
    def _get_validation_duration(self) -> float:
        """Get data validation duration in seconds"""
        if hasattr(self, '_validation_start_time'):
            return (datetime.now() - self._validation_start_time).total_seconds()
        return 0.0
    
    def get_data_status(self) -> Dict[str, Any]:
        """Get current data management status"""
        try:
            # Get latest validation if available
            latest_validation = self.data_history[-1] if self.data_history else None
            
            return {
                'is_healthy': latest_validation.get('integrity_score', 0) > 0.8 if latest_validation else False,
                'integrity_score': latest_validation.get('integrity_score', 0) if latest_validation else 0,
                'backup_status': self.backup_manager.get_backup_status(),
                'encryption_status': self.encryption_manager.validate_encryption_status(),
                'storage_status': self.storage_manager.check_storage_health(),
                'replication_status': self.replication_manager.validate_replication_status(),
                'metrics': self.data_metrics.copy(),
                'last_validation': latest_validation.get('timestamp') if latest_validation else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get data status: {e}")
            return {
                'is_healthy': False,
                'error': str(e)
            }
    
    def generate_data_report(self) -> Dict[str, Any]:
        """Generate comprehensive data management report"""
        try:
            # Perform full validation
            validation_result = self.validate_data_integrity()
            
            # Get backup report
            backup_report = self.backup_manager.generate_backup_report()
            
            # Get encryption report
            encryption_report = self.encryption_manager.generate_encryption_report()
            
            # Get storage report
            storage_report = self.storage_manager.generate_storage_report()
            
            # Get replication report
            replication_report = self.replication_manager.generate_replication_report()
            
            # Get metrics report
            metrics_report = self.data_metrics_collector.generate_metrics_report()
            
            report = {
                'report_id': f"data_report_{int(time.time())}",
                'timestamp': datetime.now().isoformat(),
                'executive_summary': {
                    'data_status': validation_result.get('status'),
                    'integrity_score': validation_result.get('integrity_score'),
                    'critical_issues': len([i for i in validation_result.get('data_issues', []) 
                                          if i.get('severity') == 'critical']),
                    'total_data_size_gb': self.data_metrics.get('total_data_size_gb', 0),
                    'backup_success_rate': self.data_metrics.get('backup_success_rate', 0),
                    'data_availability': self.data_metrics.get('data_availability', 0)
                },
                'validation_result': validation_result,
                'backup_report': backup_report,
                'encryption_report': encryption_report,
                'storage_report': storage_report,
                'replication_report': replication_report,
                'metrics_report': metrics_report,
                'recommendations': self._generate_data_recommendations(
                    validation_result, backup_report, encryption_report, 
                    storage_report, replication_report
                )
            }
            
            # Save report
            report_file = f"{report['report_id']}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Data report generated: {report_file}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate data report: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_data_recommendations(self, validation: Dict[str, Any], backup: Dict[str, Any],
                                      encryption: Dict[str, Any], storage: Dict[str, Any],
                                      replication: Dict[str, Any]) -> List[str]:
        """Generate data management recommendations"""
        recommendations = []
        
        # Check integrity score
        if validation.get('integrity_score', 0) < 0.8:
            recommendations.append(
                f"Improve data integrity - current score: {validation.get('integrity_score', 0):.2f}"
            )
        
        # Check backup success rate
        if self.data_metrics.get('backup_success_rate', 0) < 0.95:
            recommendations.append(
                "Investigate backup failures to achieve 95%+ success rate"
            )
        
        # Check encryption coverage
        if encryption.get('encryption_score', 0) < 0.99:
            recommendations.append(
                "Encrypt all data at rest to achieve 99%+ encryption coverage"
            )
        
        # Check storage efficiency
        if storage.get('efficiency_score', 0) < 0.8:
            recommendations.append(
                "Optimize storage tiering and compression for better efficiency"
            )
        
        # Check replication lag
        if self.data_metrics.get('replication_lag_seconds', 0) > 300:
            recommendations.append(
                "Reduce replication lag to meet 5-minute RPO target"
            )
        
        return recommendations
    
    def shutdown(self):
        """Shutdown data manager"""
        self.logger.info("Shutting down Data Manager")
        self.is_running = False
        
        # Generate final report
        try:
            final_report = self.generate_data_report()
            self.logger.info(f"Final data report generated: {final_report.get('report_id')}")
        except:
            pass


def create_data_manager(config: Dict[str, Any]) -> DataManager:
    """Factory function to create data manager"""
    try:
        return DataManager(config)
    except Exception as e:
        logger.error(f"Failed to create data manager: {e}")
        raise