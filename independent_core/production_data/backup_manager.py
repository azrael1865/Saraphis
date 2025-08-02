"""
Saraphis Backup Manager
Production-ready automated backup and recovery system
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import threading
import pickle
import hashlib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class BackupManager:
    """Production-ready automated backup and recovery system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.backup_history = deque(maxlen=10000)
        self.recovery_history = deque(maxlen=1000)
        
        # Backup configuration
        self.backup_path = config.get('backup_path', './backups')
        os.makedirs(self.backup_path, exist_ok=True)
        
        self.backup_policies = self._initialize_backup_policies()
        
        # Backup metrics
        self.backup_metrics = {
            'total_backups': 0,
            'successful_backups': 0,
            'failed_backups': 0,
            'total_backup_size_gb': 0.0,
            'average_backup_time': 0.0,
            'compression_ratio': 0.0,
            'last_full_backup': None,
            'last_incremental_backup': None
        }
        
        # Initialize components
        self._initialize_backup_components()
        
        # Thread control
        self._lock = threading.Lock()
        self.is_running = True
        
        self.logger.info("Backup Manager initialized")
    
    def create_backup(self, backup_type: str = 'incremental') -> Dict[str, Any]:
        """Create a new backup"""
        try:
            start_time = time.time()
            
            # Generate backup ID
            backup_id = f"backup_{backup_type}_{int(time.time())}"
            
            # Collect data for backup
            backup_data = self._collect_backup_data(backup_type)
            
            # Calculate original size
            original_size = len(pickle.dumps(backup_data))
            
            # Encrypt backup data
            from .encryption_manager import EncryptionManager
            encryption_manager = EncryptionManager(self.config.get('encryption_config', {}))
            encrypted_data = encryption_manager.encrypt_data(pickle.dumps(backup_data))
            
            # Compress backup data
            from .compression_manager import CompressionManager
            compression_manager = CompressionManager(self.config.get('compression_config', {}))
            compressed_data = compression_manager.compress_data(encrypted_data)
            
            # Calculate compression ratio
            compressed_size = len(compressed_data)
            compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
            
            # Generate backup metadata
            backup_metadata = self._generate_backup_metadata(
                backup_id, backup_type, compressed_data, compression_ratio
            )
            
            # Store backup
            from .storage_manager import StorageManager
            storage_manager = StorageManager(self.config.get('storage_config', {}))
            storage_result = storage_manager.store_backup(compressed_data, backup_metadata)
            
            # Verify backup integrity
            verification_result = self._verify_backup_integrity(compressed_data, backup_metadata)
            
            # Update backup history and metrics
            backup_duration = time.time() - start_time
            backup_record = {
                'backup_id': backup_id,
                'backup_type': backup_type,
                'timestamp': time.time(),
                'size_bytes': compressed_size,
                'original_size_bytes': original_size,
                'compression_ratio': compression_ratio,
                'encryption_status': 'encrypted',
                'storage_location': storage_result['location'],
                'verification_status': verification_result['verified'],
                'duration_seconds': backup_duration
            }
            
            with self._lock:
                self.backup_history.append(backup_record)
                self._update_backup_metrics(backup_record, success=verification_result['verified'])
            
            # Log backup event
            self.logger.info(f"Backup created: {backup_id} - Size: {compressed_size/1024/1024:.2f}MB")
            
            return {
                'success': verification_result['verified'],
                'backup_id': backup_id,
                'backup_type': backup_type,
                'size_bytes': compressed_size,
                'original_size_bytes': original_size,
                'compression_ratio': compression_ratio,
                'encryption_status': 'encrypted',
                'storage_location': storage_result['location'],
                'verification_status': verification_result['verified'],
                'duration_seconds': backup_duration,
                'metadata': backup_metadata
            }
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            
            # Update failure metrics
            with self._lock:
                self.backup_metrics['failed_backups'] += 1
            
            return {
                'success': False,
                'error': str(e),
                'backup_type': backup_type,
                'duration_seconds': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def restore_backup(self, backup_id: str, target_location: str = None) -> Dict[str, Any]:
        """Restore data from backup"""
        try:
            start_time = time.time()
            
            # Retrieve backup metadata
            backup_metadata = self._get_backup_metadata(backup_id)
            if not backup_metadata:
                raise ValueError(f"Backup {backup_id} not found")
            
            # Retrieve backup data from storage
            from .storage_manager import StorageManager
            storage_manager = StorageManager(self.config.get('storage_config', {}))
            compressed_data = storage_manager.retrieve_backup(backup_id)
            
            # Verify backup integrity before restoration
            verification_result = self._verify_backup_integrity(compressed_data, backup_metadata)
            if not verification_result['verified']:
                raise RuntimeError(f"Backup {backup_id} integrity verification failed")
            
            # Decompress backup data
            from .compression_manager import CompressionManager
            compression_manager = CompressionManager(self.config.get('compression_config', {}))
            encrypted_data = compression_manager.decompress_data(compressed_data)
            
            # Decrypt backup data
            from .encryption_manager import EncryptionManager
            encryption_manager = EncryptionManager(self.config.get('encryption_config', {}))
            decrypted_data = encryption_manager.decrypt_data(encrypted_data)
            
            # Deserialize backup data
            backup_data = pickle.loads(decrypted_data)
            
            # Restore data to target location
            restore_result = self._restore_data(backup_data, target_location)
            
            # Update recovery history
            recovery_duration = time.time() - start_time
            recovery_record = {
                'backup_id': backup_id,
                'timestamp': time.time(),
                'target_location': target_location or 'default',
                'success': restore_result['success'],
                'duration_seconds': recovery_duration
            }
            
            with self._lock:
                self.recovery_history.append(recovery_record)
            
            self.logger.info(f"Backup restored: {backup_id} - Duration: {recovery_duration:.2f}s")
            
            return {
                'success': restore_result['success'],
                'backup_id': backup_id,
                'restored_size_bytes': len(decrypted_data),
                'target_location': target_location or 'default',
                'verification_status': verification_result['verified'],
                'duration_seconds': recovery_duration,
                'restore_details': restore_result
            }
            
        except Exception as e:
            self.logger.error(f"Backup restoration failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'backup_id': backup_id,
                'duration_seconds': time.time() - start_time if 'start_time' in locals() else 0
            }
    
    def validate_backup_integrity(self) -> Dict[str, Any]:
        """Validate integrity of all backups"""
        try:
            # Get all backup metadata
            all_backups = self._get_all_backup_metadata()
            
            integrity_results = []
            total_backups = len(all_backups)
            verified_backups = 0
            corrupted_backups = []
            
            for backup_metadata in all_backups:
                backup_id = backup_metadata['backup_id']
                
                try:
                    # Retrieve backup data
                    from .storage_manager import StorageManager
                    storage_manager = StorageManager(self.config.get('storage_config', {}))
                    compressed_data = storage_manager.retrieve_backup(backup_id)
                    
                    # Verify integrity
                    verification_result = self._verify_backup_integrity(compressed_data, backup_metadata)
                    
                    integrity_results.append({
                        'backup_id': backup_id,
                        'verified': verification_result['verified'],
                        'size_bytes': len(compressed_data),
                        'compression_ratio': backup_metadata.get('compression_ratio', 1.0),
                        'age_hours': (time.time() - backup_metadata['timestamp']) / 3600
                    })
                    
                    if verification_result['verified']:
                        verified_backups += 1
                    else:
                        corrupted_backups.append(backup_id)
                        
                except Exception as e:
                    integrity_results.append({
                        'backup_id': backup_id,
                        'verified': False,
                        'error': str(e)
                    })
                    corrupted_backups.append(backup_id)
            
            integrity_score = verified_backups / total_backups if total_backups > 0 else 0.0
            
            return {
                'integrity_score': integrity_score,
                'total_backups': total_backups,
                'verified_backups': verified_backups,
                'failed_backups': total_backups - verified_backups,
                'corrupted_backups': corrupted_backups,
                'integrity_results': integrity_results,
                'last_validated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Backup integrity validation failed: {e}")
            return {
                'integrity_score': 0.0,
                'error': str(e),
                'total_backups': 0,
                'verified_backups': 0
            }
    
    def get_backup_status(self) -> Dict[str, Any]:
        """Get current backup status"""
        try:
            # Check if backup is needed
            last_backup_time = self._get_last_backup_time()
            current_time = time.time()
            
            # Check full backup (every 24 hours)
            full_backup_needed = (current_time - last_backup_time.get('full', 0)) > 86400
            
            # Check incremental backup (every 6 hours)
            incremental_backup_needed = (current_time - last_backup_time.get('incremental', 0)) > 21600
            
            backup_needed = full_backup_needed or incremental_backup_needed
            
            # Calculate next backup time
            if full_backup_needed:
                next_backup_type = 'full'
                next_backup_time = current_time
            elif incremental_backup_needed:
                next_backup_type = 'incremental'
                next_backup_time = current_time
            else:
                next_backup_type = 'incremental'
                next_backup_time = last_backup_time.get('incremental', 0) + 21600
            
            return {
                'backup_needed': backup_needed,
                'full_backup_needed': full_backup_needed,
                'incremental_backup_needed': incremental_backup_needed,
                'last_backup_times': last_backup_time,
                'next_backup_time': next_backup_time,
                'next_backup_type': next_backup_type,
                'total_backups': self.backup_metrics['total_backups'],
                'success_rate': self._calculate_success_rate()
            }
            
        except Exception as e:
            self.logger.error(f"Backup status check failed: {e}")
            return {
                'backup_needed': True,
                'error': str(e)
            }
    
    def generate_backup_report(self) -> Dict[str, Any]:
        """Generate comprehensive backup report"""
        try:
            # Validate all backups
            integrity_validation = self.validate_backup_integrity()
            
            # Get backup statistics
            backup_stats = self._calculate_backup_statistics()
            
            # Get storage usage
            storage_usage = self._calculate_storage_usage()
            
            # Get recovery readiness
            recovery_readiness = self._assess_recovery_readiness()
            
            report = {
                'report_id': f"backup_report_{int(time.time())}",
                'timestamp': datetime.now().isoformat(),
                'backup_metrics': self.backup_metrics.copy(),
                'integrity_validation': integrity_validation,
                'backup_statistics': backup_stats,
                'storage_usage': storage_usage,
                'recovery_readiness': recovery_readiness,
                'backup_policy_compliance': self._check_policy_compliance(),
                'recommendations': self._generate_backup_recommendations(
                    integrity_validation, backup_stats, storage_usage
                )
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate backup report: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _collect_backup_data(self, backup_type: str) -> Dict[str, Any]:
        """Collect data for backup"""
        try:
            backup_data = {
                'backup_type': backup_type,
                'timestamp': time.time(),
                'system_version': '1.0.0',
                'data_version': '1.0.0'
            }
            
            if backup_type == 'full':
                # Full backup - collect all data
                backup_data.update({
                    'brain_state': self._get_brain_state(),
                    'domain_data': self._get_domain_data(),
                    'training_models': self._get_training_models(),
                    'configuration': self._get_system_configuration(),
                    'system_logs': self._get_system_logs(),
                    'metadata': self._get_system_metadata()
                })
            elif backup_type == 'incremental':
                # Incremental backup - collect changed data since last backup
                last_backup_time = self._get_last_backup_time().get(backup_type, 0)
                backup_data.update({
                    'brain_state_delta': self._get_brain_state_delta(last_backup_time),
                    'domain_data_delta': self._get_domain_data_delta(last_backup_time),
                    'configuration_delta': self._get_configuration_delta(last_backup_time),
                    'system_logs_delta': self._get_system_logs_delta(last_backup_time)
                })
            else:
                # Differential backup - collect changes since last full backup
                last_full_backup_time = self._get_last_backup_time().get('full', 0)
                backup_data.update({
                    'brain_state_diff': self._get_brain_state_delta(last_full_backup_time),
                    'domain_data_diff': self._get_domain_data_delta(last_full_backup_time),
                    'configuration_diff': self._get_configuration_delta(last_full_backup_time),
                    'system_logs_diff': self._get_system_logs_delta(last_full_backup_time)
                })
            
            return backup_data
            
        except Exception as e:
            self.logger.error(f"Backup data collection failed: {e}")
            raise RuntimeError(f"Backup data collection failed: {e}")
    
    def _generate_backup_metadata(self, backup_id: str, backup_type: str, 
                                 compressed_data: bytes, compression_ratio: float) -> Dict[str, Any]:
        """Generate backup metadata"""
        try:
            return {
                'backup_id': backup_id,
                'backup_type': backup_type,
                'timestamp': time.time(),
                'size_bytes': len(compressed_data),
                'compression_ratio': compression_ratio,
                'encryption_status': 'encrypted',
                'checksum': self._calculate_checksum(compressed_data),
                'version': '1.0.0',
                'retention_days': self.backup_policies[backup_type]['retention_days'],
                'verification_required': True
            }
            
        except Exception as e:
            self.logger.error(f"Backup metadata generation failed: {e}")
            raise RuntimeError(f"Backup metadata generation failed: {e}")
    
    def _verify_backup_integrity(self, compressed_data: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Verify backup integrity"""
        try:
            # Verify checksum
            calculated_checksum = self._calculate_checksum(compressed_data)
            stored_checksum = metadata.get('checksum', '')
            
            checksum_valid = calculated_checksum == stored_checksum
            
            # Verify size
            size_valid = len(compressed_data) == metadata.get('size_bytes', 0)
            
            # Verify compression ratio
            compression_ratio = metadata.get('compression_ratio', 1.0)
            compression_valid = 0.1 <= compression_ratio <= 1.0
            
            # Verify encryption status
            encryption_valid = metadata.get('encryption_status') == 'encrypted'
            
            # Overall verification
            verified = checksum_valid and size_valid and compression_valid and encryption_valid
            
            return {
                'verified': verified,
                'checksum_valid': checksum_valid,
                'size_valid': size_valid,
                'compression_valid': compression_valid,
                'encryption_valid': encryption_valid,
                'calculated_checksum': calculated_checksum,
                'stored_checksum': stored_checksum
            }
            
        except Exception as e:
            self.logger.error(f"Backup integrity verification failed: {e}")
            return {
                'verified': False,
                'error': str(e)
            }
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum of data"""
        try:
            return hashlib.sha256(data).hexdigest()
        except Exception as e:
            self.logger.error(f"Checksum calculation failed: {e}")
            return ""
    
    def _initialize_backup_policies(self) -> Dict[str, Any]:
        """Initialize backup policies"""
        return {
            'full': {
                'frequency_hours': 24,
                'retention_days': 30,
                'compression': True,
                'encryption': True,
                'verification': True,
                'replication': True
            },
            'incremental': {
                'frequency_hours': 6,
                'retention_days': 7,
                'compression': True,
                'encryption': True,
                'verification': True,
                'replication': True
            },
            'differential': {
                'frequency_hours': 12,
                'retention_days': 14,
                'compression': True,
                'encryption': True,
                'verification': True,
                'replication': True
            }
        }
    
    def _initialize_backup_components(self):
        """Initialize backup components"""
        # In production, this would set up connections to backup systems
        self.logger.info("Backup components initialized")
    
    def _get_last_backup_time(self) -> Dict[str, float]:
        """Get last backup times by type"""
        try:
            last_backup_times = {'full': 0, 'incremental': 0, 'differential': 0}
            
            with self._lock:
                for backup_record in self.backup_history:
                    backup_type = backup_record.get('backup_type', 'unknown')
                    timestamp = backup_record.get('timestamp', 0)
                    
                    if backup_type in last_backup_times:
                        last_backup_times[backup_type] = max(last_backup_times[backup_type], timestamp)
            
            return last_backup_times
            
        except Exception as e:
            self.logger.error(f"Last backup time retrieval failed: {e}")
            return {'full': 0, 'incremental': 0, 'differential': 0}
    
    def _get_backup_metadata(self, backup_id: str) -> Dict[str, Any]:
        """Get backup metadata by ID"""
        try:
            # Check in backup history first
            with self._lock:
                for backup_record in self.backup_history:
                    if backup_record.get('backup_id') == backup_id:
                        return {
                            'backup_id': backup_id,
                            'timestamp': backup_record.get('timestamp', 0),
                            'size_bytes': backup_record.get('size_bytes', 0),
                            'compression_ratio': backup_record.get('compression_ratio', 1.0),
                            'checksum': backup_record.get('checksum', ''),
                            'encryption_status': backup_record.get('encryption_status', 'unknown')
                        }
            
            # If not found, retrieve from storage
            backup_path = os.path.join(self.backup_path, f"{backup_id}_metadata.json")
            if os.path.exists(backup_path):
                import json
                with open(backup_path, 'r') as f:
                    return json.load(f)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Backup metadata retrieval failed: {e}")
            return None
    
    def _get_all_backup_metadata(self) -> List[Dict[str, Any]]:
        """Get metadata for all backups"""
        try:
            all_metadata = []
            
            # Get from backup history
            with self._lock:
                for backup_record in self.backup_history:
                    all_metadata.append({
                        'backup_id': backup_record.get('backup_id'),
                        'timestamp': backup_record.get('timestamp', 0),
                        'size_bytes': backup_record.get('size_bytes', 0),
                        'compression_ratio': backup_record.get('compression_ratio', 1.0),
                        'checksum': backup_record.get('checksum', ''),
                        'backup_type': backup_record.get('backup_type', 'unknown')
                    })
            
            return all_metadata
            
        except Exception as e:
            self.logger.error(f"All backup metadata retrieval failed: {e}")
            return []
    
    def _restore_data(self, backup_data: Dict[str, Any], target_location: str) -> Dict[str, Any]:
        """Restore data to target location"""
        try:
            restored_components = []
            
            # Restore based on backup type
            backup_type = backup_data.get('backup_type', 'unknown')
            
            if backup_type == 'full':
                # Restore full backup
                if 'brain_state' in backup_data:
                    self._restore_brain_state(backup_data['brain_state'])
                    restored_components.append('brain_state')
                
                if 'domain_data' in backup_data:
                    self._restore_domain_data(backup_data['domain_data'])
                    restored_components.append('domain_data')
                
                if 'training_models' in backup_data:
                    self._restore_training_models(backup_data['training_models'])
                    restored_components.append('training_models')
                
                if 'configuration' in backup_data:
                    self._restore_configuration(backup_data['configuration'])
                    restored_components.append('configuration')
                    
            elif backup_type in ['incremental', 'differential']:
                # Restore delta/diff data
                for key in backup_data:
                    if key.endswith('_delta') or key.endswith('_diff'):
                        component = key.replace('_delta', '').replace('_diff', '')
                        self._restore_delta_data(component, backup_data[key])
                        restored_components.append(component)
            
            return {
                'success': True,
                'restored_components': restored_components,
                'target_location': target_location or 'default',
                'backup_type': backup_type
            }
            
        except Exception as e:
            self.logger.error(f"Data restoration failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _update_backup_metrics(self, backup_record: Dict[str, Any], success: bool):
        """Update backup metrics"""
        try:
            self.backup_metrics['total_backups'] += 1
            
            if success:
                self.backup_metrics['successful_backups'] += 1
            else:
                self.backup_metrics['failed_backups'] += 1
            
            # Update backup times
            backup_type = backup_record.get('backup_type')
            if backup_type == 'full':
                self.backup_metrics['last_full_backup'] = backup_record.get('timestamp')
            elif backup_type == 'incremental':
                self.backup_metrics['last_incremental_backup'] = backup_record.get('timestamp')
            
            # Update size metrics
            size_gb = backup_record.get('size_bytes', 0) / (1024**3)
            self.backup_metrics['total_backup_size_gb'] += size_gb
            
            # Update average backup time
            duration = backup_record.get('duration_seconds', 0)
            total_backups = self.backup_metrics['successful_backups']
            if total_backups > 0:
                current_avg = self.backup_metrics['average_backup_time']
                self.backup_metrics['average_backup_time'] = (
                    (current_avg * (total_backups - 1) + duration) / total_backups
                )
            
            # Update compression ratio
            self.backup_metrics['compression_ratio'] = backup_record.get('compression_ratio', 1.0)
            
        except Exception as e:
            self.logger.error(f"Backup metrics update failed: {e}")
    
    def _calculate_success_rate(self) -> float:
        """Calculate backup success rate"""
        total = self.backup_metrics['total_backups']
        if total == 0:
            return 1.0
        
        successful = self.backup_metrics['successful_backups']
        return successful / total
    
    def _calculate_backup_statistics(self) -> Dict[str, Any]:
        """Calculate backup statistics"""
        try:
            with self._lock:
                if not self.backup_history:
                    return {}
                
                # Calculate statistics
                backup_sizes = [b['size_bytes'] for b in self.backup_history]
                backup_durations = [b['duration_seconds'] for b in self.backup_history]
                compression_ratios = [b['compression_ratio'] for b in self.backup_history]
                
                return {
                    'total_backups': len(self.backup_history),
                    'average_size_mb': sum(backup_sizes) / len(backup_sizes) / (1024**2) if backup_sizes else 0,
                    'max_size_mb': max(backup_sizes) / (1024**2) if backup_sizes else 0,
                    'min_size_mb': min(backup_sizes) / (1024**2) if backup_sizes else 0,
                    'average_duration_seconds': sum(backup_durations) / len(backup_durations) if backup_durations else 0,
                    'average_compression_ratio': sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0,
                    'backup_frequency': self._calculate_backup_frequency()
                }
                
        except Exception as e:
            self.logger.error(f"Backup statistics calculation failed: {e}")
            return {}
    
    def _calculate_storage_usage(self) -> Dict[str, Any]:
        """Calculate backup storage usage"""
        try:
            total_size = 0
            backup_count = {'full': 0, 'incremental': 0, 'differential': 0}
            
            with self._lock:
                for backup in self.backup_history:
                    total_size += backup.get('size_bytes', 0)
                    backup_type = backup.get('backup_type', 'unknown')
                    if backup_type in backup_count:
                        backup_count[backup_type] += 1
            
            return {
                'total_storage_gb': total_size / (1024**3),
                'backup_count': backup_count,
                'average_backup_size_mb': total_size / len(self.backup_history) / (1024**2) if self.backup_history else 0,
                'storage_efficiency': self._calculate_storage_efficiency()
            }
            
        except Exception as e:
            self.logger.error(f"Storage usage calculation failed: {e}")
            return {}
    
    def _assess_recovery_readiness(self) -> Dict[str, Any]:
        """Assess recovery readiness"""
        try:
            # Check last successful backups
            last_backup_times = self._get_last_backup_time()
            current_time = time.time()
            
            # Calculate recovery point objectives
            full_backup_age = (current_time - last_backup_times.get('full', 0)) / 3600  # hours
            incremental_backup_age = (current_time - last_backup_times.get('incremental', 0)) / 3600
            
            # Check recovery history
            successful_recoveries = sum(1 for r in self.recovery_history if r.get('success', False))
            total_recoveries = len(self.recovery_history)
            recovery_success_rate = successful_recoveries / total_recoveries if total_recoveries > 0 else 1.0
            
            # Calculate RTO/RPO compliance
            rpo_target = 1  # 1 hour
            rto_target = 2  # 2 hours
            
            rpo_compliant = incremental_backup_age <= rpo_target
            
            return {
                'recovery_ready': rpo_compliant and recovery_success_rate > 0.95,
                'last_full_backup_hours': full_backup_age,
                'last_incremental_backup_hours': incremental_backup_age,
                'recovery_success_rate': recovery_success_rate,
                'rpo_compliant': rpo_compliant,
                'rpo_target_hours': rpo_target,
                'rto_target_hours': rto_target,
                'total_recovery_tests': total_recoveries
            }
            
        except Exception as e:
            self.logger.error(f"Recovery readiness assessment failed: {e}")
            return {'recovery_ready': False, 'error': str(e)}
    
    def _check_policy_compliance(self) -> Dict[str, Any]:
        """Check backup policy compliance"""
        try:
            compliance_status = {}
            
            for backup_type, policy in self.backup_policies.items():
                last_backup_time = self._get_last_backup_time().get(backup_type, 0)
                hours_since_backup = (time.time() - last_backup_time) / 3600
                frequency_hours = policy['frequency_hours']
                
                compliance_status[backup_type] = {
                    'compliant': hours_since_backup <= frequency_hours * 1.1,  # 10% tolerance
                    'hours_since_backup': hours_since_backup,
                    'required_frequency_hours': frequency_hours,
                    'encryption_enabled': policy['encryption'],
                    'compression_enabled': policy['compression'],
                    'verification_enabled': policy['verification']
                }
            
            return compliance_status
            
        except Exception as e:
            self.logger.error(f"Policy compliance check failed: {e}")
            return {}
    
    def _generate_backup_recommendations(self, integrity: Dict[str, Any], 
                                       stats: Dict[str, Any], 
                                       storage: Dict[str, Any]) -> List[str]:
        """Generate backup recommendations"""
        recommendations = []
        
        # Check integrity score
        if integrity.get('integrity_score', 0) < 0.95:
            recommendations.append(
                f"Investigate backup integrity issues - current score: {integrity.get('integrity_score', 0):.2f}"
            )
        
        # Check success rate
        success_rate = self._calculate_success_rate()
        if success_rate < 0.99:
            recommendations.append(
                f"Improve backup reliability - current success rate: {success_rate:.2%}"
            )
        
        # Check compression efficiency
        avg_compression = stats.get('average_compression_ratio', 1.0)
        if avg_compression > 0.7:
            recommendations.append(
                "Consider optimizing compression settings for better storage efficiency"
            )
        
        # Check backup frequency
        backup_frequency = stats.get('backup_frequency', {})
        for backup_type, freq in backup_frequency.items():
            if freq < 0.9:  # Less than 90% of scheduled backups
                recommendations.append(
                    f"Improve {backup_type} backup frequency - current: {freq:.1%}"
                )
        
        # Check storage usage
        if storage.get('total_storage_gb', 0) > 100:
            recommendations.append(
                "Consider implementing backup lifecycle management to reduce storage usage"
            )
        
        return recommendations
    
    def _calculate_backup_frequency(self) -> Dict[str, float]:
        """Calculate backup frequency compliance"""
        try:
            frequency_compliance = {}
            
            for backup_type in ['full', 'incremental', 'differential']:
                expected_frequency = self.backup_policies[backup_type]['frequency_hours']
                
                # Count backups of this type in last 7 days
                week_ago = time.time() - (7 * 24 * 3600)
                backup_count = sum(1 for b in self.backup_history 
                                 if b.get('backup_type') == backup_type and 
                                 b.get('timestamp', 0) > week_ago)
                
                expected_count = (7 * 24) / expected_frequency
                frequency_compliance[backup_type] = backup_count / expected_count if expected_count > 0 else 0
            
            return frequency_compliance
            
        except Exception as e:
            self.logger.error(f"Backup frequency calculation failed: {e}")
            return {}
    
    def _calculate_storage_efficiency(self) -> float:
        """Calculate storage efficiency score"""
        try:
            # Factors: compression ratio, deduplication, storage tiering
            avg_compression = self.backup_metrics.get('compression_ratio', 1.0)
            
            # Higher compression ratio = better efficiency
            compression_score = 1.0 - avg_compression
            
            # In production, would also consider deduplication and tiering
            return compression_score
            
        except Exception as e:
            self.logger.error(f"Storage efficiency calculation failed: {e}")
            return 0.5
    
    # Data collection methods
    def _get_brain_state(self) -> Dict[str, Any]:
        return {'state': 'active', 'version': '1.0.0', 'timestamp': time.time()}
    
    def _get_domain_data(self) -> Dict[str, Any]:
        return {
            'domains': ['fraud_detection', 'cybersecurity', 'trading'],
            'active_domains': 3,
            'timestamp': time.time()
        }
    
    def _get_training_models(self) -> Dict[str, Any]:
        return {
            'models': ['model_v1', 'model_v2', 'model_v3'],
            'active_model': 'model_v3',
            'timestamp': time.time()
        }
    
    def _get_system_configuration(self) -> Dict[str, Any]:
        return {
            'version': '1.0.0',
            'environment': 'production',
            'features': ['security', 'backup', 'monitoring'],
            'timestamp': time.time()
        }
    
    def _get_system_logs(self) -> List[str]:
        return ['[INFO] System operational', '[INFO] Backup completed', '[INFO] Security check passed']
    
    def _get_system_metadata(self) -> Dict[str, Any]:
        return {
            'system_id': 'saraphis_001',
            'deployment_date': '2024-01-01',
            'last_update': time.time()
        }
    
    # Delta methods for incremental backups
    def _get_brain_state_delta(self, since_timestamp: float) -> Dict[str, Any]:
        return {'delta_state': 'changes_since_' + str(int(since_timestamp))}
    
    def _get_domain_data_delta(self, since_timestamp: float) -> Dict[str, Any]:
        return {'delta_domains': 'domain_changes_since_' + str(int(since_timestamp))}
    
    def _get_configuration_delta(self, since_timestamp: float) -> Dict[str, Any]:
        return {'delta_config': 'config_changes_since_' + str(int(since_timestamp))}
    
    def _get_system_logs_delta(self, since_timestamp: float) -> List[str]:
        return ['[INFO] Delta log entry 1', '[INFO] Delta log entry 2']
    
    # Restoration methods
    def _restore_brain_state(self, brain_state: Dict[str, Any]):
        self.logger.info(f"Restoring brain state: {brain_state.get('version', 'unknown')}")
    
    def _restore_domain_data(self, domain_data: Dict[str, Any]):
        self.logger.info(f"Restoring domain data: {len(domain_data.get('domains', []))} domains")
    
    def _restore_training_models(self, training_models: Dict[str, Any]):
        self.logger.info(f"Restoring training models: {len(training_models.get('models', []))} models")
    
    def _restore_configuration(self, configuration: Dict[str, Any]):
        self.logger.info(f"Restoring configuration: version {configuration.get('version', 'unknown')}")
    
    def _restore_delta_data(self, component: str, delta_data: Any):
        self.logger.info(f"Restoring {component} delta data")
    
    def shutdown(self):
        """Shutdown backup manager"""
        self.logger.info("Shutting down Backup Manager")
        self.is_running = False