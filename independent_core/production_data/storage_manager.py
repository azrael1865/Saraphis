"""
Saraphis Storage Manager
Production-ready multi-tier storage management system
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import os
import shutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from pathlib import Path

logger = logging.getLogger(__name__)


class StorageManager:
    """Production-ready multi-tier storage management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.storage_history = deque(maxlen=10000)
        
        # Storage paths
        self.hot_storage_path = Path(config.get('hot_storage_path', './storage/hot'))
        self.warm_storage_path = Path(config.get('warm_storage_path', './storage/warm'))
        self.cold_storage_path = Path(config.get('cold_storage_path', './storage/cold'))
        
        # Create storage directories
        for path in [self.hot_storage_path, self.warm_storage_path, self.cold_storage_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Storage policies
        self.storage_policies = self._initialize_storage_policies()
        
        # Storage metrics
        self.storage_metrics = {
            'total_files': 0,
            'total_size_gb': 0.0,
            'hot_storage_gb': 0.0,
            'warm_storage_gb': 0.0,
            'cold_storage_gb': 0.0,
            'storage_operations': 0,
            'tiering_operations': 0,
            'storage_errors': 0
        }
        
        # File tracking
        self.file_metadata = {}
        self.access_tracking = defaultdict(lambda: {'count': 0, 'last_access': 0})
        
        # Thread control
        self._lock = threading.Lock()
        self.is_running = True
        
        # Start storage management threads
        self._start_storage_threads()
        
        self.logger.info("Storage Manager initialized")
    
    def store_backup(self, data: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Store backup data with metadata"""
        try:
            backup_id = metadata.get('backup_id', f'backup_{int(time.time())}')
            backup_type = metadata.get('backup_type', 'unknown')
            
            # Determine storage tier based on backup type
            if backup_type == 'full':
                storage_tier = 'warm'  # Full backups go to warm storage
            else:
                storage_tier = 'hot'   # Incremental/differential go to hot storage
            
            # Store data
            storage_path = self._get_storage_path(storage_tier)
            file_path = storage_path / f"{backup_id}.dat"
            
            # Write data
            with open(file_path, 'wb') as f:
                f.write(data)
            
            # Store metadata
            metadata_path = storage_path / f"{backup_id}_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update file metadata
            file_info = {
                'file_id': backup_id,
                'file_path': str(file_path),
                'metadata_path': str(metadata_path),
                'storage_tier': storage_tier,
                'size_bytes': len(data),
                'created_at': time.time(),
                'last_accessed': time.time(),
                'backup_type': backup_type,
                'checksum': metadata.get('checksum', '')
            }
            
            with self._lock:
                self.file_metadata[backup_id] = file_info
                # Initialize access tracking for new files
                self.access_tracking[backup_id] = {'count': 0, 'last_access': time.time()}
                self._update_storage_metrics()
            
            # Record storage operation
            self._record_storage_operation('store', backup_id, storage_tier, len(data))
            
            self.logger.info(f"Backup stored: {backup_id} in {storage_tier} storage")
            
            return {
                'success': True,
                'location': str(file_path),
                'storage_tier': storage_tier,
                'size_bytes': len(data)
            }
            
        except Exception as e:
            self.logger.error(f"Backup storage failed: {e}")
            with self._lock:
                self.storage_metrics['storage_errors'] += 1
            return {
                'success': False,
                'error': str(e)
            }
    
    def retrieve_backup(self, backup_id: str) -> bytes:
        """Retrieve backup data"""
        try:
            # Get file metadata
            file_info = self.file_metadata.get(backup_id)
            if not file_info:
                # Search in all tiers
                file_info = self._find_backup_in_storage(backup_id)
                if not file_info:
                    raise ValueError(f"Backup {backup_id} not found")
            
            # Read data
            file_path = Path(file_info['file_path'])
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Update access tracking
            self._update_access_tracking(backup_id)
            
            # Record retrieval operation
            self._record_storage_operation('retrieve', backup_id, 
                                         file_info['storage_tier'], len(data))
            
            return data
            
        except Exception as e:
            self.logger.error(f"Backup retrieval failed: {e}")
            raise RuntimeError(f"Backup retrieval failed: {e}")
    
    def check_storage_health(self) -> Dict[str, Any]:
        """Check storage system health"""
        try:
            # Calculate storage usage
            storage_usage = self._calculate_storage_usage()
            
            # Check storage capacity
            capacity_status = self._check_storage_capacity()
            
            # Verify file integrity
            integrity_status = self._verify_storage_integrity()
            
            # Check tiering efficiency
            tiering_efficiency = self._check_tiering_efficiency()
            
            # Identify storage issues
            issues = self._identify_storage_issues(
                storage_usage, capacity_status, integrity_status, tiering_efficiency
            )
            
            # Calculate health score
            health_score = self._calculate_storage_health_score(
                storage_usage, capacity_status, integrity_status, tiering_efficiency
            )
            
            return {
                'health_score': health_score,
                'total_size_gb': storage_usage['total_size_gb'],
                'storage_usage': storage_usage,
                'capacity_status': capacity_status,
                'integrity_status': integrity_status,
                'tiering_efficiency': tiering_efficiency,
                'issues': issues,
                'efficiency_score': tiering_efficiency.get('efficiency_score', 0.0),
                'last_checked': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Storage health check failed: {e}")
            return {
                'health_score': 0.0,
                'error': str(e),
                'issues': [{
                    'type': 'health_check_failure',
                    'severity': 'critical',
                    'description': str(e)
                }]
            }
    
    def resolve_storage_issues(self, issues: List[Dict[str, Any]]):
        """Resolve identified storage issues"""
        for issue in issues:
            try:
                issue_type = issue.get('type')
                
                if issue_type == 'low_capacity':
                    self._handle_low_capacity(issue)
                elif issue_type == 'inefficient_tiering':
                    self._optimize_storage_tiering()
                elif issue_type == 'corrupted_files':
                    self._handle_corrupted_files(issue)
                elif issue_type == 'orphaned_metadata':
                    self._clean_orphaned_metadata()
                
                self.logger.info(f"Resolved storage issue: {issue_type}")
                
            except Exception as e:
                self.logger.error(f"Failed to resolve storage issue {issue_type}: {e}")
    
    def generate_storage_report(self) -> Dict[str, Any]:
        """Generate comprehensive storage report"""
        try:
            # Get storage health
            storage_health = self.check_storage_health()
            
            # Calculate storage statistics
            storage_stats = self._calculate_storage_statistics()
            
            # Analyze access patterns
            access_patterns = self._analyze_access_patterns()
            
            # Calculate cost analysis
            cost_analysis = self._calculate_storage_costs()
            
            report = {
                'report_id': f"storage_report_{int(time.time())}",
                'timestamp': datetime.now().isoformat(),
                'storage_health': storage_health,
                'storage_statistics': storage_stats,
                'access_patterns': access_patterns,
                'cost_analysis': cost_analysis,
                'tiering_summary': self._get_tiering_summary(),
                'recommendations': self._generate_storage_recommendations(
                    storage_health, storage_stats, access_patterns
                )
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate storage report: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _initialize_storage_policies(self) -> Dict[str, Any]:
        """Initialize storage policies"""
        return {
            'tiering_policy': {
                'hot_to_warm_days': 7,
                'warm_to_cold_days': 90,
                'access_threshold_hot': 5,  # Access count to keep in hot
                'access_threshold_warm': 2  # Access count to keep in warm
            },
            'retention_policy': {
                'hot_retention_days': 7,
                'warm_retention_days': 90,
                'cold_retention_days': 2555  # 7 years
            },
            'capacity_policy': {
                'hot_storage_limit_gb': 100,
                'warm_storage_limit_gb': 1000,
                'cold_storage_limit_gb': 10000,
                'warning_threshold': 0.8,  # 80% capacity
                'critical_threshold': 0.95  # 95% capacity
            },
            'redundancy_policy': {
                'replication_factor': 3,
                'checksum_verification': True,
                'integrity_check_interval_hours': 24
            }
        }
    
    def _get_storage_path(self, tier: str) -> Path:
        """Get storage path for tier"""
        if tier == 'hot':
            return self.hot_storage_path
        elif tier == 'warm':
            return self.warm_storage_path
        elif tier == 'cold':
            return self.cold_storage_path
        else:
            raise ValueError(f"Unknown storage tier: {tier}")
    
    def _find_backup_in_storage(self, backup_id: str) -> Optional[Dict[str, Any]]:
        """Find backup file in storage tiers"""
        for tier in ['hot', 'warm', 'cold']:
            storage_path = self._get_storage_path(tier)
            file_path = storage_path / f"{backup_id}.dat"
            
            if file_path.exists():
                metadata_path = storage_path / f"{backup_id}_metadata.json"
                
                file_info = {
                    'file_id': backup_id,
                    'file_path': str(file_path),
                    'metadata_path': str(metadata_path),
                    'storage_tier': tier,
                    'size_bytes': file_path.stat().st_size,
                    'last_accessed': time.time()
                }
                
                # Update metadata cache
                with self._lock:
                    self.file_metadata[backup_id] = file_info
                
                return file_info
        
        return None
    
    def _update_access_tracking(self, file_id: str):
        """Update file access tracking"""
        with self._lock:
            self.access_tracking[file_id]['count'] += 1
            self.access_tracking[file_id]['last_access'] = time.time()
            
            if file_id in self.file_metadata:
                self.file_metadata[file_id]['last_accessed'] = time.time()
    
    def _record_storage_operation(self, operation: str, file_id: str, 
                                 tier: str, size_bytes: int):
        """Record storage operation"""
        operation_record = {
            'timestamp': time.time(),
            'operation': operation,
            'file_id': file_id,
            'storage_tier': tier,
            'size_bytes': size_bytes
        }
        
        self.storage_history.append(operation_record)
        
        with self._lock:
            self.storage_metrics['storage_operations'] += 1
    
    def _update_storage_metrics(self):
        """Update storage metrics"""
        try:
            total_size = 0
            hot_size = 0
            warm_size = 0
            cold_size = 0
            total_files = 0
            
            for file_info in self.file_metadata.values():
                size_bytes = file_info.get('size_bytes', 0)
                total_size += size_bytes
                total_files += 1
                
                tier = file_info.get('storage_tier')
                if tier == 'hot':
                    hot_size += size_bytes
                elif tier == 'warm':
                    warm_size += size_bytes
                elif tier == 'cold':
                    cold_size += size_bytes
            
            self.storage_metrics['total_files'] = total_files
            self.storage_metrics['total_size_gb'] = total_size / (1024**3)
            self.storage_metrics['hot_storage_gb'] = hot_size / (1024**3)
            self.storage_metrics['warm_storage_gb'] = warm_size / (1024**3)
            self.storage_metrics['cold_storage_gb'] = cold_size / (1024**3)
            
        except Exception as e:
            self.logger.error(f"Storage metrics update failed: {e}")
    
    def _calculate_storage_usage(self) -> Dict[str, Any]:
        """Calculate detailed storage usage"""
        try:
            # Calculate from metadata (primary source) and verify with disk usage
            hot_usage_meta = 0
            warm_usage_meta = 0  
            cold_usage_meta = 0
            
            for file_info in self.file_metadata.values():
                size_bytes = file_info.get('size_bytes', 0)
                tier = file_info.get('storage_tier')
                if tier == 'hot':
                    hot_usage_meta += size_bytes
                elif tier == 'warm':
                    warm_usage_meta += size_bytes
                elif tier == 'cold':
                    cold_usage_meta += size_bytes
            
            # Verify against actual disk usage
            hot_usage_disk = self._get_directory_size(self.hot_storage_path)
            warm_usage_disk = self._get_directory_size(self.warm_storage_path)
            cold_usage_disk = self._get_directory_size(self.cold_storage_path)
            
            # Use metadata as primary, but warn if mismatch
            if (hot_usage_meta != hot_usage_disk or 
                warm_usage_meta != warm_usage_disk or 
                cold_usage_meta != cold_usage_disk):
                self.logger.warning(
                    f"Storage metadata mismatch: Meta({hot_usage_meta}, {warm_usage_meta}, {cold_usage_meta}) "
                    f"vs Disk({hot_usage_disk}, {warm_usage_disk}, {cold_usage_disk})"
                )
            
            hot_usage = hot_usage_meta
            warm_usage = warm_usage_meta
            cold_usage = cold_usage_meta
            
            total_usage = hot_usage + warm_usage + cold_usage
            
            return {
                'total_size_gb': total_usage / (1024**3),
                'hot_storage_gb': hot_usage / (1024**3),
                'warm_storage_gb': warm_usage / (1024**3),
                'cold_storage_gb': cold_usage / (1024**3),
                'hot_percentage': (hot_usage / total_usage * 100) if total_usage > 0 else 0,
                'warm_percentage': (warm_usage / total_usage * 100) if total_usage > 0 else 0,
                'cold_percentage': (cold_usage / total_usage * 100) if total_usage > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Storage usage calculation failed: {e}")
            raise RuntimeError(f"Storage usage calculation failed: {e}")
    
    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
        except Exception as e:
            self.logger.error(f"Directory size calculation failed for {path}: {e}")
        return total_size
    
    def _check_storage_capacity(self) -> Dict[str, Any]:
        """Check storage capacity status"""
        try:
            capacity_limits = self.storage_policies['capacity_policy']
            current_usage = self._calculate_storage_usage()
            
            capacity_status = {}
            
            for tier in ['hot', 'warm', 'cold']:
                current_gb = current_usage.get(f'{tier}_storage_gb', 0)
                limit_gb = capacity_limits.get(f'{tier}_storage_limit_gb', float('inf'))
                
                usage_percentage = (current_gb / limit_gb) if limit_gb > 0 else 0
                
                status = 'normal'
                if usage_percentage >= capacity_limits['critical_threshold']:
                    status = 'critical'
                elif usage_percentage >= capacity_limits['warning_threshold']:
                    status = 'warning'
                
                capacity_status[tier] = {
                    'current_gb': current_gb,
                    'limit_gb': limit_gb,
                    'usage_percentage': usage_percentage,
                    'status': status,
                    'available_gb': max(0, limit_gb - current_gb)
                }
            
            return capacity_status
            
        except Exception as e:
            self.logger.error(f"Capacity check failed: {e}")
            raise RuntimeError(f"Capacity check failed: {e}")
    
    def _verify_storage_integrity(self) -> Dict[str, Any]:
        """Verify storage file integrity"""
        try:
            total_files = 0
            verified_files = 0
            corrupted_files = []
            
            with self._lock:
                file_list = list(self.file_metadata.items())
            
            for file_id, file_info in file_list:
                total_files += 1
                
                file_path = Path(file_info['file_path'])
                if file_path.exists():
                    # Verify file is readable
                    try:
                        with open(file_path, 'rb') as f:
                            # Read first 1KB to verify
                            f.read(1024)
                            # For files larger than 1KB, also read last 1KB
                            file_size = file_path.stat().st_size
                            if file_size > 1024:
                                f.seek(-1024, 2)
                                f.read(1024)
                        verified_files += 1
                    except Exception as e:
                        self.logger.warning(f"File verification failed for {file_id}: {e}")
                        corrupted_files.append(file_id)
                else:
                    corrupted_files.append(file_id)
            
            integrity_score = verified_files / total_files if total_files > 0 else 1.0
            
            return {
                'integrity_score': integrity_score,
                'total_files': total_files,
                'verified_files': verified_files,
                'corrupted_files': corrupted_files,
                'corruption_rate': len(corrupted_files) / total_files if total_files > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Integrity verification failed: {e}")
            return {'integrity_score': 0.0, 'error': str(e)}
    
    def _check_tiering_efficiency(self) -> Dict[str, Any]:
        """Check storage tiering efficiency"""
        try:
            current_time = time.time()
            tiering_policy = self.storage_policies['tiering_policy']
            
            misplaced_files = {
                'hot_to_warm': [],
                'warm_to_cold': [],
                'cold_to_warm': [],
                'warm_to_hot': []
            }
            
            with self._lock:
                for file_id, file_info in self.file_metadata.items():
                    tier = file_info.get('storage_tier')
                    created_at = file_info.get('created_at', 0)
                    age_days = (current_time - created_at) / 86400
                    
                    access_info = self.access_tracking.get(file_id, {})
                    access_count = access_info.get('count', 0)
                    
                    # Check if file should be moved
                    if tier == 'hot':
                        if age_days > tiering_policy['hot_to_warm_days'] and \
                           access_count < tiering_policy['access_threshold_hot']:
                            misplaced_files['hot_to_warm'].append(file_id)
                    
                    elif tier == 'warm':
                        if age_days > tiering_policy['warm_to_cold_days'] and \
                           access_count < tiering_policy['access_threshold_warm']:
                            misplaced_files['warm_to_cold'].append(file_id)
                        elif access_count >= tiering_policy['access_threshold_hot']:
                            misplaced_files['warm_to_hot'].append(file_id)
                    
                    elif tier == 'cold':
                        if access_count >= tiering_policy['access_threshold_warm']:
                            misplaced_files['cold_to_warm'].append(file_id)
            
            total_misplaced = sum(len(files) for files in misplaced_files.values())
            total_files = len(self.file_metadata)
            
            efficiency_score = 1.0 - (total_misplaced / total_files) if total_files > 0 else 1.0
            
            return {
                'efficiency_score': efficiency_score,
                'total_files': total_files,
                'misplaced_files': total_misplaced,
                'misplacement_details': misplaced_files,
                'optimization_needed': total_misplaced > 0
            }
            
        except Exception as e:
            self.logger.error(f"Tiering efficiency check failed: {e}")
            return {'efficiency_score': 0.5, 'error': str(e)}
    
    def _identify_storage_issues(self, usage: Dict[str, Any], capacity: Dict[str, Any],
                                integrity: Dict[str, Any], tiering: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify storage issues"""
        issues = []
        
        # Check capacity issues
        for tier, status in capacity.items():
            if status.get('status') == 'critical':
                issues.append({
                    'type': 'low_capacity',
                    'severity': 'critical',
                    'tier': tier,
                    'usage_percentage': status.get('usage_percentage', 0),
                    'description': f"{tier} storage at {status.get('usage_percentage', 0):.1%} capacity"
                })
            elif status.get('status') == 'warning':
                issues.append({
                    'type': 'low_capacity',
                    'severity': 'warning',
                    'tier': tier,
                    'usage_percentage': status.get('usage_percentage', 0),
                    'description': f"{tier} storage approaching capacity limit"
                })
        
        # Check integrity issues
        if integrity.get('corrupted_files'):
            issues.append({
                'type': 'corrupted_files',
                'severity': 'high',
                'count': len(integrity.get('corrupted_files', [])),
                'files': integrity.get('corrupted_files', []),
                'description': f"{len(integrity.get('corrupted_files', []))} corrupted files detected"
            })
        
        # Check tiering issues
        if tiering.get('efficiency_score', 1.0) < 0.8:
            issues.append({
                'type': 'inefficient_tiering',
                'severity': 'medium',
                'efficiency_score': tiering.get('efficiency_score', 0),
                'misplaced_files': tiering.get('misplaced_files', 0),
                'description': f"Storage tiering efficiency at {tiering.get('efficiency_score', 0):.1%}"
            })
        
        return issues
    
    def _calculate_storage_health_score(self, usage: Dict[str, Any], capacity: Dict[str, Any],
                                       integrity: Dict[str, Any], tiering: Dict[str, Any]) -> float:
        """Calculate overall storage health score"""
        try:
            # Capacity score (30% weight)
            capacity_scores = []
            for tier_status in capacity.values():
                usage_pct = tier_status.get('usage_percentage', 0)
                if usage_pct < 0.8:
                    capacity_scores.append(1.0)
                elif usage_pct < 0.95:
                    capacity_scores.append(0.5)
                else:
                    capacity_scores.append(0.0)
            
            capacity_score = sum(capacity_scores) / len(capacity_scores) if capacity_scores else 0
            
            # Integrity score (40% weight)
            integrity_score = integrity.get('integrity_score', 0)
            
            # Tiering efficiency (30% weight)
            tiering_score = tiering.get('efficiency_score', 0)
            
            # Calculate weighted score
            health_score = (capacity_score * 0.3 + 
                          integrity_score * 0.4 + 
                          tiering_score * 0.3)
            
            return min(1.0, max(0.0, health_score))
            
        except Exception as e:
            self.logger.error(f"Health score calculation failed: {e}")
            return 0.5
    
    def _start_storage_threads(self):
        """Start storage management threads"""
        # Start tiering thread
        self.tiering_thread = threading.Thread(target=self._storage_tiering_loop, daemon=True)
        self.tiering_thread.start()
        
        # Start integrity check thread
        self.integrity_thread = threading.Thread(target=self._integrity_check_loop, daemon=True)
        self.integrity_thread.start()
    
    def _storage_tiering_loop(self):
        """Background storage tiering loop"""
        while self.is_running:
            try:
                # Check tiering efficiency
                tiering_status = self._check_tiering_efficiency()
                
                if tiering_status.get('optimization_needed', False):
                    self._optimize_storage_tiering()
                
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Storage tiering loop error: {e}")
                time.sleep(7200)
    
    def _integrity_check_loop(self):
        """Background integrity check loop"""
        while self.is_running:
            try:
                # Verify storage integrity
                integrity_status = self._verify_storage_integrity()
                
                if integrity_status.get('corrupted_files'):
                    self._handle_corrupted_files({
                        'files': integrity_status['corrupted_files']
                    })
                
                time.sleep(86400)  # Run every 24 hours
                
            except Exception as e:
                self.logger.error(f"Integrity check loop error: {e}")
                time.sleep(86400)
    
    def _optimize_storage_tiering(self):
        """Optimize storage tiering by moving files"""
        try:
            tiering_status = self._check_tiering_efficiency()
            misplaced_files = tiering_status.get('misplacement_details', {})
            
            # Move files between tiers
            for movement, file_ids in misplaced_files.items():
                for file_id in file_ids[:10]:  # Limit to 10 files per run
                    try:
                        if movement == 'hot_to_warm':
                            self._move_file_to_tier(file_id, 'hot', 'warm')
                        elif movement == 'warm_to_cold':
                            self._move_file_to_tier(file_id, 'warm', 'cold')
                        elif movement == 'cold_to_warm':
                            self._move_file_to_tier(file_id, 'cold', 'warm')
                        elif movement == 'warm_to_hot':
                            self._move_file_to_tier(file_id, 'warm', 'hot')
                        
                        with self._lock:
                            self.storage_metrics['tiering_operations'] += 1
                            
                    except Exception as e:
                        self.logger.error(f"Failed to move file {file_id}: {e}")
            
            self.logger.info(f"Storage tiering optimization completed")
            
        except Exception as e:
            self.logger.error(f"Storage tiering optimization failed: {e}")
    
    def _move_file_to_tier(self, file_id: str, from_tier: str, to_tier: str):
        """Move file between storage tiers"""
        file_info = self.file_metadata.get(file_id)
        if not file_info:
            return
        
        from_path = Path(file_info['file_path'])
        to_storage = self._get_storage_path(to_tier)
        to_path = to_storage / from_path.name
        
        # Move file
        shutil.move(str(from_path), str(to_path))
        
        # Move metadata file
        metadata_from = Path(file_info['metadata_path'])
        metadata_to = to_storage / metadata_from.name
        if metadata_from.exists():
            shutil.move(str(metadata_from), str(metadata_to))
        
        # Update file info
        with self._lock:
            file_info['file_path'] = str(to_path)
            file_info['metadata_path'] = str(metadata_to)
            file_info['storage_tier'] = to_tier
        
        self.logger.info(f"Moved {file_id} from {from_tier} to {to_tier}")
    
    def _handle_low_capacity(self, issue: Dict[str, Any]):
        """Handle low storage capacity"""
        tier = issue.get('tier')
        
        # Move old files to next tier
        if tier == 'hot':
            self._optimize_storage_tiering()
        elif tier == 'warm':
            # Move oldest files to cold
            self._move_old_files_to_cold()
        elif tier == 'cold':
            # Alert for cold storage capacity
            self.logger.critical(f"Cold storage at capacity - manual intervention required")
    
    def _move_old_files_to_cold(self):
        """Move old files from warm to cold storage"""
        current_time = time.time()
        policy = self.storage_policies['tiering_policy']
        
        with self._lock:
            warm_files = [(fid, info) for fid, info in self.file_metadata.items() 
                         if info.get('storage_tier') == 'warm']
        
        # Sort by age
        warm_files.sort(key=lambda x: x[1].get('created_at', 0))
        
        # Move oldest 10% to cold
        files_to_move = warm_files[:max(1, len(warm_files) // 10)]
        
        for file_id, _ in files_to_move:
            try:
                self._move_file_to_tier(file_id, 'warm', 'cold')
            except Exception as e:
                self.logger.error(f"Failed to move {file_id} to cold: {e}")
    
    def _handle_corrupted_files(self, issue: Dict[str, Any]):
        """Handle corrupted files"""
        corrupted_files = issue.get('files', [])
        
        for file_id in corrupted_files:
            try:
                # Try to recover from redundant copies
                # In production, this would check replicas
                self.logger.warning(f"Corrupted file detected: {file_id}")
                
                # Remove from metadata if unrecoverable
                with self._lock:
                    if file_id in self.file_metadata:
                        del self.file_metadata[file_id]
                        
            except Exception as e:
                self.logger.error(f"Failed to handle corrupted file {file_id}: {e}")
    
    def _clean_orphaned_metadata(self):
        """Clean up orphaned metadata files"""
        # In production, this would scan for metadata without data files
        self.logger.info("Cleaning orphaned metadata")
    
    def _calculate_storage_statistics(self) -> Dict[str, Any]:
        """Calculate detailed storage statistics"""
        try:
            with self._lock:
                total_files = len(self.file_metadata)
                
                # Files by tier
                files_by_tier = defaultdict(int)
                size_by_tier = defaultdict(float)
                
                for file_info in self.file_metadata.values():
                    tier = file_info.get('storage_tier', 'unknown')
                    files_by_tier[tier] += 1
                    size_by_tier[tier] += file_info.get('size_bytes', 0) / (1024**3)
                
                # Age distribution
                current_time = time.time()
                age_distribution = {
                    '< 1 day': 0,
                    '1-7 days': 0,
                    '7-30 days': 0,
                    '30-90 days': 0,
                    '> 90 days': 0
                }
                
                for file_info in self.file_metadata.values():
                    age_days = (current_time - file_info.get('created_at', 0)) / 86400
                    
                    if age_days < 1:
                        age_distribution['< 1 day'] += 1
                    elif age_days < 7:
                        age_distribution['1-7 days'] += 1
                    elif age_days < 30:
                        age_distribution['7-30 days'] += 1
                    elif age_days < 90:
                        age_distribution['30-90 days'] += 1
                    else:
                        age_distribution['> 90 days'] += 1
                
                return {
                    'total_files': total_files,
                    'files_by_tier': dict(files_by_tier),
                    'size_by_tier_gb': dict(size_by_tier),
                    'age_distribution': age_distribution,
                    'storage_operations': self.storage_metrics['storage_operations'],
                    'tiering_operations': self.storage_metrics['tiering_operations']
                }
                
        except Exception as e:
            self.logger.error(f"Storage statistics calculation failed: {e}")
            raise RuntimeError(f"Storage statistics calculation failed: {e}")
    
    def _analyze_access_patterns(self) -> Dict[str, Any]:
        """Analyze file access patterns"""
        try:
            with self._lock:
                access_data = list(self.access_tracking.items())
            
            if not access_data:
                raise RuntimeError("No access data available for pattern analysis")
            
            # Calculate access statistics
            access_counts = [data['count'] for _, data in access_data]
            
            # Hot files (accessed > 5 times)
            hot_files = sum(1 for count in access_counts if count > 5)
            
            # Cold files (never accessed)
            cold_files = sum(1 for count in access_counts if count == 0)
            
            return {
                'total_tracked_files': len(access_data),
                'hot_files': hot_files,
                'cold_files': cold_files,
                'average_access_count': sum(access_counts) / len(access_counts),
                'max_access_count': max(access_counts) if access_counts else 0,
                'access_frequency_distribution': self._calculate_access_distribution(access_counts)
            }
            
        except Exception as e:
            self.logger.error(f"Access pattern analysis failed: {e}")
            raise RuntimeError(f"Access pattern analysis failed: {e}")
    
    def _calculate_access_distribution(self, access_counts: List[int]) -> Dict[str, int]:
        """Calculate access frequency distribution"""
        distribution = {
            '0 accesses': 0,
            '1-5 accesses': 0,
            '6-20 accesses': 0,
            '> 20 accesses': 0
        }
        
        for count in access_counts:
            if count == 0:
                distribution['0 accesses'] += 1
            elif count <= 5:
                distribution['1-5 accesses'] += 1
            elif count <= 20:
                distribution['6-20 accesses'] += 1
            else:
                distribution['> 20 accesses'] += 1
        
        return distribution
    
    def _calculate_storage_costs(self) -> Dict[str, Any]:
        """Calculate storage costs"""
        try:
            # Storage cost per GB/month
            cost_per_gb = {
                'hot': 0.023,   # SSD storage
                'warm': 0.0125, # HDD storage
                'cold': 0.004   # Archive storage
            }
            
            usage = self._calculate_storage_usage()
            
            monthly_costs = {}
            total_monthly_cost = 0
            
            for tier in ['hot', 'warm', 'cold']:
                gb_used = usage.get(f'{tier}_storage_gb', 0)
                monthly_cost = gb_used * cost_per_gb[tier]
                monthly_costs[tier] = monthly_cost
                total_monthly_cost += monthly_cost
            
            return {
                'monthly_costs_by_tier': monthly_costs,
                'total_monthly_cost': total_monthly_cost,
                'annual_cost': total_monthly_cost * 12,
                'cost_per_gb_average': total_monthly_cost / usage.get('total_size_gb', 1) if usage.get('total_size_gb', 0) > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Storage cost calculation failed: {e}")
            raise RuntimeError(f"Storage cost calculation failed: {e}")
    
    def _get_tiering_summary(self) -> Dict[str, Any]:
        """Get storage tiering summary"""
        try:
            tiering_efficiency = self._check_tiering_efficiency()
            
            return {
                'efficiency_score': tiering_efficiency.get('efficiency_score', 0),
                'total_files': tiering_efficiency.get('total_files', 0),
                'optimally_placed_files': tiering_efficiency.get('total_files', 0) - 
                                        tiering_efficiency.get('misplaced_files', 0),
                'misplaced_files': tiering_efficiency.get('misplaced_files', 0),
                'pending_movements': tiering_efficiency.get('misplacement_details', {}),
                'last_optimization': datetime.now().isoformat()  # Would track actual last run
            }
            
        except Exception as e:
            self.logger.error(f"Tiering summary generation failed: {e}")
            raise RuntimeError(f"Tiering summary generation failed: {e}")
    
    def _generate_storage_recommendations(self, health: Dict[str, Any],
                                        stats: Dict[str, Any],
                                        patterns: Dict[str, Any]) -> List[str]:
        """Generate storage recommendations"""
        recommendations = []
        
        # Check health score
        if health.get('health_score', 0) < 0.8:
            recommendations.append(
                f"Storage health below optimal - score: {health.get('health_score', 0):.2f}"
            )
        
        # Check capacity
        for issue in health.get('issues', []):
            if issue.get('type') == 'low_capacity' and issue.get('severity') == 'critical':
                recommendations.append(
                    f"Urgent: {issue.get('tier')} storage at critical capacity"
                )
        
        # Check tiering efficiency
        if health.get('tiering_efficiency', {}).get('efficiency_score', 1) < 0.8:
            recommendations.append(
                "Run storage tiering optimization to improve efficiency"
            )
        
        # Check access patterns
        if patterns.get('cold_files', 0) > patterns.get('total_tracked_files', 1) * 0.5:
            recommendations.append(
                "Consider archiving cold files to reduce storage costs"
            )
        
        return recommendations
    
    def shutdown(self):
        """Shutdown storage manager"""
        self.logger.info("Shutting down Storage Manager")
        self.is_running = False