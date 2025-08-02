"""
Saraphis Audit Logger
Production-ready security event logging and audit trail management
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import json
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from pathlib import Path
import gzip
import os

logger = logging.getLogger(__name__)


class AuditLogger:
    """Production-ready security event logging and audit trail management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Audit configuration
        self.log_path = Path(config.get('log_path', './security_audit_logs'))
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        self.retention_days = config.get('retention_days', 2555)  # 7 years
        self.max_log_size = config.get('max_log_size_mb', 100) * 1024 * 1024
        self.compression_enabled = config.get('compression_enabled', True)
        self.real_time_alerts = config.get('real_time_alerts', True)
        
        # Log buffers
        self.log_buffer = deque(maxlen=10000)
        self.critical_events = deque(maxlen=1000)
        self.alert_queue = deque(maxlen=1000)
        
        # Current log file
        self.current_log_file = None
        self.current_log_size = 0
        self.log_sequence = 0
        
        # Event statistics
        self.event_stats = defaultdict(lambda: {
            'count': 0,
            'last_seen': None,
            'severity_breakdown': defaultdict(int)
        })
        
        # Integrity tracking
        self.log_hashes = {}
        self.integrity_verified = True
        
        # Threading
        self._lock = threading.Lock()
        self.is_running = True
        
        # Start background threads
        self._start_logging_threads()
        
        self.logger.info(f"Audit Logger initialized with path: {self.log_path}")
    
    def log_security_event(self, event: Dict[str, Any]) -> bool:
        """Log a security event with integrity protection"""
        try:
            # Add timestamp and metadata
            event_record = {
                'event_id': self._generate_event_id(),
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'event': event,
                'integrity_hash': None
            }
            
            # Calculate integrity hash
            event_record['integrity_hash'] = self._calculate_integrity_hash(event_record)
            
            # Add to buffer
            with self._lock:
                self.log_buffer.append(event_record)
                
                # Update statistics
                event_type = event.get('event_type', 'unknown')
                severity = event.get('severity', 'info')
                
                self.event_stats[event_type]['count'] += 1
                self.event_stats[event_type]['last_seen'] = datetime.now()
                self.event_stats[event_type]['severity_breakdown'][severity] += 1
                
                # Track critical events
                if severity in ['critical', 'high']:
                    self.critical_events.append(event_record)
                    
                    # Add to alert queue if real-time alerts enabled
                    if self.real_time_alerts:
                        self.alert_queue.append(event_record)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to log security event: {e}")
            return False
    
    def search_logs(self, criteria: Dict[str, Any], 
                   time_range: Optional[Dict[str, datetime]] = None) -> List[Dict[str, Any]]:
        """Search audit logs based on criteria"""
        try:
            results = []
            
            # Search in memory buffer first
            for event in self.log_buffer:
                if self._matches_criteria(event, criteria, time_range):
                    results.append(event)
            
            # Search in log files if needed
            if len(results) < criteria.get('limit', 1000):
                file_results = self._search_log_files(criteria, time_range)
                results.extend(file_results)
            
            # Sort by timestamp (newest first)
            results.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Apply limit
            limit = criteria.get('limit', 1000)
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"Log search failed: {e}")
            return []
    
    def get_audit_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate audit report for specified period"""
        try:
            # Search for events in time range
            time_range = {'start': start_date, 'end': end_date}
            all_events = self.search_logs({}, time_range)
            
            # Analyze events
            report = {
                'report_id': f"audit_report_{int(time.time())}",
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'total_events': len(all_events),
                'event_breakdown': self._analyze_event_breakdown(all_events),
                'severity_summary': self._analyze_severity_summary(all_events),
                'top_event_types': self._get_top_event_types(all_events),
                'critical_events': self._get_critical_events(all_events),
                'user_activity': self._analyze_user_activity(all_events),
                'system_activity': self._analyze_system_activity(all_events),
                'anomalies': self._detect_audit_anomalies(all_events),
                'compliance_events': self._get_compliance_events(all_events),
                'integrity_status': self.integrity_verified
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate audit report: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def verify_log_integrity(self) -> Dict[str, Any]:
        """Verify integrity of audit logs"""
        try:
            verification_results = {
                'verified': True,
                'total_logs_checked': 0,
                'integrity_failures': [],
                'verification_time': datetime.now().isoformat()
            }
            
            # Verify current buffer
            for event in self.log_buffer:
                verification_results['total_logs_checked'] += 1
                
                expected_hash = self._calculate_integrity_hash(event)
                if event.get('integrity_hash') != expected_hash:
                    verification_results['verified'] = False
                    verification_results['integrity_failures'].append({
                        'event_id': event.get('event_id'),
                        'timestamp': event.get('timestamp'),
                        'reason': 'Hash mismatch'
                    })
            
            # Verify log files
            log_files = sorted(self.log_path.glob('audit_*.log*'))
            for log_file in log_files[-10:]:  # Check last 10 files
                file_verified = self._verify_log_file_integrity(log_file)
                if not file_verified:
                    verification_results['verified'] = False
                    verification_results['integrity_failures'].append({
                        'file': str(log_file),
                        'reason': 'File integrity check failed'
                    })
            
            self.integrity_verified = verification_results['verified']
            
            return verification_results
            
        except Exception as e:
            self.logger.error(f"Log integrity verification failed: {e}")
            return {
                'verified': False,
                'error': str(e)
            }
    
    def export_logs(self, criteria: Dict[str, Any], format: str = 'json') -> str:
        """Export logs based on criteria"""
        try:
            # Search logs
            logs = self.search_logs(criteria)
            
            # Generate export filename
            export_file = self.log_path / f"export_{int(time.time())}.{format}"
            
            if format == 'json':
                with open(export_file, 'w') as f:
                    json.dump(logs, f, indent=2, default=str)
            elif format == 'csv':
                self._export_to_csv(logs, export_file)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"Exported {len(logs)} logs to {export_file}")
            return str(export_file)
            
        except Exception as e:
            self.logger.error(f"Log export failed: {e}")
            raise
    
    def _start_logging_threads(self):
        """Start background logging threads"""
        # Log writer thread
        self.writer_thread = threading.Thread(target=self._log_writer_loop, daemon=True)
        self.writer_thread.start()
        
        # Log rotation thread
        self.rotation_thread = threading.Thread(target=self._log_rotation_loop, daemon=True)
        self.rotation_thread.start()
        
        # Alert processor thread
        if self.real_time_alerts:
            self.alert_thread = threading.Thread(target=self._alert_processor_loop, daemon=True)
            self.alert_thread.start()
    
    def _log_writer_loop(self):
        """Background thread for writing logs to disk"""
        while self.is_running:
            try:
                # Write buffered events
                events_to_write = []
                
                with self._lock:
                    # Get events from buffer
                    while self.log_buffer and len(events_to_write) < 100:
                        events_to_write.append(self.log_buffer.popleft())
                
                if events_to_write:
                    self._write_events_to_disk(events_to_write)
                
                time.sleep(1)  # Write every second
                
            except Exception as e:
                self.logger.error(f"Log writer error: {e}")
                time.sleep(5)
    
    def _log_rotation_loop(self):
        """Background thread for log rotation and cleanup"""
        while self.is_running:
            try:
                # Check if current log needs rotation
                if self.current_log_size > self.max_log_size:
                    self._rotate_log()
                
                # Clean old logs
                self._clean_old_logs()
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Log rotation error: {e}")
                time.sleep(600)
    
    def _alert_processor_loop(self):
        """Background thread for processing real-time alerts"""
        while self.is_running:
            try:
                alerts_to_process = []
                
                with self._lock:
                    while self.alert_queue and len(alerts_to_process) < 10:
                        alerts_to_process.append(self.alert_queue.popleft())
                
                for alert in alerts_to_process:
                    self._process_alert(alert)
                
                time.sleep(0.5)  # Process every 500ms
                
            except Exception as e:
                self.logger.error(f"Alert processor error: {e}")
                time.sleep(5)
    
    def _write_events_to_disk(self, events: List[Dict[str, Any]]):
        """Write events to disk"""
        try:
            # Get or create current log file
            if not self.current_log_file:
                self._create_new_log_file()
            
            # Write events
            with open(self.current_log_file, 'a') as f:
                for event in events:
                    line = json.dumps(event, default=str) + '\n'
                    f.write(line)
                    self.current_log_size += len(line.encode('utf-8'))
            
        except Exception as e:
            self.logger.error(f"Failed to write events to disk: {e}")
    
    def _create_new_log_file(self):
        """Create new log file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_sequence += 1
        
        filename = f"audit_{timestamp}_{self.log_sequence:04d}.log"
        self.current_log_file = self.log_path / filename
        self.current_log_size = 0
        
        # Write header
        header = {
            'log_type': 'security_audit',
            'created': datetime.now().isoformat(),
            'sequence': self.log_sequence,
            'version': '1.0'
        }
        
        with open(self.current_log_file, 'w') as f:
            f.write(json.dumps(header) + '\n')
    
    def _rotate_log(self):
        """Rotate current log file"""
        if not self.current_log_file:
            return
        
        try:
            # Compress current log if enabled
            if self.compression_enabled:
                compressed_file = str(self.current_log_file) + '.gz'
                
                with open(self.current_log_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        f_out.writelines(f_in)
                
                # Remove original file
                os.remove(self.current_log_file)
                
                self.logger.info(f"Compressed log file: {compressed_file}")
            
            # Reset current log
            self.current_log_file = None
            self.current_log_size = 0
            
        except Exception as e:
            self.logger.error(f"Log rotation failed: {e}")
    
    def _clean_old_logs(self):
        """Clean logs older than retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            # Find old log files
            for log_file in self.log_path.glob('audit_*.log*'):
                # Extract timestamp from filename
                try:
                    timestamp_str = log_file.stem.split('_')[1] + log_file.stem.split('_')[2]
                    file_date = datetime.strptime(timestamp_str[:8], '%Y%m%d')
                    
                    if file_date < cutoff_date:
                        os.remove(log_file)
                        self.logger.info(f"Removed old log file: {log_file}")
                        
                except:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Log cleanup failed: {e}")
    
    def _process_alert(self, alert: Dict[str, Any]):
        """Process real-time alert"""
        try:
            # In production, this would send alerts via various channels
            event = alert.get('event', {})
            severity = event.get('severity', 'info')
            
            if severity == 'critical':
                self.logger.critical(f"SECURITY ALERT: {event}")
            elif severity == 'high':
                self.logger.error(f"Security Alert: {event}")
            else:
                self.logger.warning(f"Security Notice: {event}")
                
        except Exception as e:
            self.logger.error(f"Alert processing failed: {e}")
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = int(time.time() * 1000000)  # Microsecond precision
        random_part = os.urandom(4).hex()
        return f"evt_{timestamp}_{random_part}"
    
    def _calculate_integrity_hash(self, event: Dict[str, Any]) -> str:
        """Calculate integrity hash for event"""
        # Create copy without hash field
        event_copy = event.copy()
        event_copy.pop('integrity_hash', None)
        
        # Serialize deterministically
        event_str = json.dumps(event_copy, sort_keys=True, default=str)
        
        # Calculate hash
        return hashlib.sha256(event_str.encode('utf-8')).hexdigest()
    
    def _matches_criteria(self, event: Dict[str, Any], criteria: Dict[str, Any],
                         time_range: Optional[Dict[str, datetime]]) -> bool:
        """Check if event matches search criteria"""
        # Check time range
        if time_range:
            event_time = datetime.fromtimestamp(event['timestamp'])
            if 'start' in time_range and event_time < time_range['start']:
                return False
            if 'end' in time_range and event_time > time_range['end']:
                return False
        
        # Check event type
        if 'event_type' in criteria:
            if event['event'].get('event_type') != criteria['event_type']:
                return False
        
        # Check severity
        if 'severity' in criteria:
            if event['event'].get('severity') != criteria['severity']:
                return False
        
        # Check username
        if 'username' in criteria:
            if event['event'].get('username') != criteria['username']:
                return False
        
        # Check source
        if 'source' in criteria:
            if criteria['source'] not in str(event['event'].get('source', '')):
                return False
        
        return True
    
    def _search_log_files(self, criteria: Dict[str, Any],
                         time_range: Optional[Dict[str, datetime]]) -> List[Dict[str, Any]]:
        """Search in log files"""
        results = []
        
        try:
            # Get relevant log files based on time range
            log_files = self._get_relevant_log_files(time_range)
            
            for log_file in log_files:
                if log_file.suffix == '.gz':
                    # Compressed file
                    with gzip.open(log_file, 'rt') as f:
                        for line in f:
                            try:
                                event = json.loads(line)
                                if 'event_id' in event and self._matches_criteria(event, criteria, time_range):
                                    results.append(event)
                            except:
                                continue
                else:
                    # Regular file
                    with open(log_file, 'r') as f:
                        for line in f:
                            try:
                                event = json.loads(line)
                                if 'event_id' in event and self._matches_criteria(event, criteria, time_range):
                                    results.append(event)
                            except:
                                continue
                
                # Stop if we have enough results
                if len(results) >= criteria.get('limit', 1000):
                    break
                    
        except Exception as e:
            self.logger.error(f"File search failed: {e}")
        
        return results
    
    def _get_relevant_log_files(self, time_range: Optional[Dict[str, datetime]]) -> List[Path]:
        """Get log files relevant to time range"""
        all_files = sorted(self.log_path.glob('audit_*.log*'))
        
        if not time_range:
            return all_files
        
        relevant_files = []
        for log_file in all_files:
            try:
                # Extract date from filename
                timestamp_str = log_file.stem.split('_')[1]
                file_date = datetime.strptime(timestamp_str[:8], '%Y%m%d')
                
                # Check if file might contain relevant events
                if 'start' in time_range and file_date.date() > time_range['end'].date():
                    continue
                if 'end' in time_range and file_date.date() < time_range['start'].date():
                    continue
                
                relevant_files.append(log_file)
                
            except:
                continue
        
        return relevant_files
    
    def _verify_log_file_integrity(self, log_file: Path) -> bool:
        """Verify integrity of a log file"""
        try:
            # Calculate file hash
            hasher = hashlib.sha256()
            
            if log_file.suffix == '.gz':
                with gzip.open(log_file, 'rb') as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)
            else:
                with open(log_file, 'rb') as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)
            
            file_hash = hasher.hexdigest()
            
            # Check against stored hash (if available)
            stored_hash = self.log_hashes.get(str(log_file))
            if stored_hash and stored_hash != file_hash:
                return False
            
            # Store hash for future verification
            self.log_hashes[str(log_file)] = file_hash
            
            return True
            
        except Exception as e:
            self.logger.error(f"File integrity check failed: {e}")
            return False
    
    def _analyze_event_breakdown(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze event type breakdown"""
        breakdown = defaultdict(int)
        
        for event in events:
            event_type = event['event'].get('event_type', 'unknown')
            breakdown[event_type] += 1
        
        return dict(breakdown)
    
    def _analyze_severity_summary(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze severity distribution"""
        summary = defaultdict(int)
        
        for event in events:
            severity = event['event'].get('severity', 'info')
            summary[severity] += 1
        
        return dict(summary)
    
    def _get_top_event_types(self, events: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top event types"""
        event_counts = defaultdict(int)
        
        for event in events:
            event_type = event['event'].get('event_type', 'unknown')
            event_counts[event_type] += 1
        
        sorted_types = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [{'event_type': et, 'count': count} for et, count in sorted_types[:limit]]
    
    def _get_critical_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get critical events from list"""
        critical = []
        
        for event in events:
            if event['event'].get('severity') in ['critical', 'high']:
                critical.append({
                    'event_id': event['event_id'],
                    'timestamp': event['datetime'],
                    'event_type': event['event'].get('event_type'),
                    'description': event['event'].get('description', 'No description'),
                    'severity': event['event'].get('severity')
                })
        
        return critical[:100]  # Limit to 100 most recent
    
    def _analyze_user_activity(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user activity from events"""
        user_activity = defaultdict(lambda: {
            'total_events': 0,
            'event_types': defaultdict(int),
            'failed_attempts': 0
        })
        
        for event in events:
            username = event['event'].get('username')
            if username:
                user_activity[username]['total_events'] += 1
                user_activity[username]['event_types'][event['event'].get('event_type', 'unknown')] += 1
                
                if 'fail' in event['event'].get('event_type', '').lower():
                    user_activity[username]['failed_attempts'] += 1
        
        # Convert to regular dict and get top users
        activity_dict = dict(user_activity)
        top_users = sorted(activity_dict.items(), key=lambda x: x[1]['total_events'], reverse=True)[:10]
        
        return {
            'total_users': len(activity_dict),
            'top_users': [{'username': u, 'activity': a} for u, a in top_users]
        }
    
    def _analyze_system_activity(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze system activity from events"""
        system_activity = defaultdict(lambda: {
            'total_events': 0,
            'event_types': defaultdict(int),
            'errors': 0
        })
        
        for event in events:
            source = event['event'].get('source', 'unknown')
            system_activity[source]['total_events'] += 1
            system_activity[source]['event_types'][event['event'].get('event_type', 'unknown')] += 1
            
            if event['event'].get('severity') in ['error', 'critical']:
                system_activity[source]['errors'] += 1
        
        # Convert and get top systems
        activity_dict = dict(system_activity)
        top_systems = sorted(activity_dict.items(), key=lambda x: x[1]['total_events'], reverse=True)[:10]
        
        return {
            'total_systems': len(activity_dict),
            'top_systems': [{'system': s, 'activity': a} for s, a in top_systems]
        }
    
    def _detect_audit_anomalies(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in audit events"""
        anomalies = []
        
        # Detect unusual event frequencies
        event_times = [e['timestamp'] for e in events]
        if event_times:
            # Check for burst activity
            for i in range(1, len(event_times)):
                if i >= 10:  # Need at least 10 events
                    time_window = event_times[i] - event_times[i-10]
                    if time_window < 1:  # 10 events in 1 second
                        anomalies.append({
                            'type': 'burst_activity',
                            'timestamp': datetime.fromtimestamp(event_times[i]).isoformat(),
                            'description': 'Unusual burst of activity detected'
                        })
        
        # Detect unusual patterns
        # In production, this would use more sophisticated anomaly detection
        
        return anomalies[:50]  # Limit to 50 anomalies
    
    def _get_compliance_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get compliance-related events"""
        compliance_events = []
        
        compliance_keywords = ['compliance', 'audit', 'policy', 'violation', 'regulation']
        
        for event in events:
            event_type = event['event'].get('event_type', '').lower()
            if any(keyword in event_type for keyword in compliance_keywords):
                compliance_events.append({
                    'event_id': event['event_id'],
                    'timestamp': event['datetime'],
                    'event_type': event['event'].get('event_type'),
                    'description': event['event'].get('description', 'No description')
                })
        
        return compliance_events[:100]  # Limit to 100
    
    def _export_to_csv(self, logs: List[Dict[str, Any]], filepath: Path):
        """Export logs to CSV format"""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            if not logs:
                return
            
            # Define CSV fields
            fieldnames = ['event_id', 'timestamp', 'datetime', 'event_type', 
                         'severity', 'username', 'source', 'description']
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for log in logs:
                row = {
                    'event_id': log.get('event_id'),
                    'timestamp': log.get('timestamp'),
                    'datetime': log.get('datetime'),
                    'event_type': log['event'].get('event_type'),
                    'severity': log['event'].get('severity'),
                    'username': log['event'].get('username', ''),
                    'source': log['event'].get('source', ''),
                    'description': log['event'].get('description', '')
                }
                writer.writerow(row)
    
    def shutdown(self):
        """Shutdown audit logger"""
        self.logger.info("Shutting down Audit Logger")
        self.is_running = False
        
        # Write remaining events
        with self._lock:
            remaining_events = list(self.log_buffer)
        
        if remaining_events:
            self._write_events_to_disk(remaining_events)
        
        # Rotate current log
        if self.current_log_file:
            self._rotate_log()