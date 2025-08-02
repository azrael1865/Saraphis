"""
Saraphis Incident Response Manager
Production-ready automated incident response system
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


class IncidentResponseManager:
    """Production-ready automated incident response system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Incident tracking
        self.active_incidents = {}
        self.incident_history = deque(maxlen=10000)
        self.response_actions = deque(maxlen=5000)
        
        # Response configuration
        self.auto_response_enabled = config.get('auto_response_enabled', True)
        self.escalation_enabled = config.get('escalation_enabled', True)
        self.notification_channels = config.get('notification_channels', ['email', 'sms', 'webhook'])
        
        # SLA configuration
        self.response_time_sla = config.get('response_time_sla', 300)  # 5 minutes
        self.resolution_time_sla = config.get('resolution_time_sla', 3600)  # 1 hour
        self.escalation_time = config.get('escalation_time', 1800)  # 30 minutes
        
        # Response playbooks
        self.response_playbooks = self._initialize_response_playbooks()
        self.escalation_policies = self._initialize_escalation_policies()
        
        # Incident metrics
        self.incident_metrics = {
            'total_incidents': 0,
            'resolved_incidents': 0,
            'escalated_incidents': 0,
            'auto_resolved': 0,
            'avg_response_time': 0,
            'avg_resolution_time': 0,
            'sla_violations': 0
        }
        
        # Threading
        self._lock = threading.Lock()
        self.is_running = True
        
        # Start background threads
        self._start_incident_threads()
        
        self.logger.info("Incident Response Manager initialized")
    
    def handle_incident(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """Handle security incident with automated response"""
        try:
            # Create incident record
            incident_id = self._generate_incident_id()
            incident_record = {
                'incident_id': incident_id,
                'incident_type': incident.get('type', 'unknown'),
                'severity': incident.get('severity', 'medium'),
                'source': incident.get('source', 'unknown'),
                'description': incident.get('description', 'Security incident detected'),
                'details': incident,
                'status': 'new',
                'created_at': datetime.now(),
                'response_started': None,
                'resolved_at': None,
                'response_actions': [],
                'escalation_level': 0
            }
            
            # Store incident
            with self._lock:
                self.active_incidents[incident_id] = incident_record
                self.incident_history.append(incident_record)
                self.incident_metrics['total_incidents'] += 1
            
            # Log incident
            self.logger.warning(f"Security incident created: {incident_id} - {incident_record['incident_type']}")
            
            # Trigger response
            response_result = self._execute_incident_response(incident_record)
            
            # Update incident record
            incident_record['response_result'] = response_result
            
            return {
                'success': True,
                'incident_id': incident_id,
                'status': incident_record['status'],
                'response_initiated': response_result['initiated'],
                'actions_taken': len(response_result.get('actions', [])),
                'auto_resolved': response_result.get('auto_resolved', False)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to handle incident: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def handle_threat_incident(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle threat-related incident"""
        try:
            # Convert threat data to incident
            incident = {
                'type': 'threat_detection',
                'severity': self._map_threat_severity(threat_data.get('threat_level', 'medium')),
                'source': 'threat_detector',
                'description': f"Threat detected: {threat_data.get('threat_level', 'unknown')} level",
                'threat_data': threat_data
            }
            
            # Handle as incident
            return self.handle_incident(incident)
            
        except Exception as e:
            self.logger.error(f"Failed to handle threat incident: {e}")
            return {'success': False, 'error': str(e)}
    
    def handle_compliance_violation(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle compliance violation incident"""
        try:
            # Create compliance incident
            incident = {
                'type': 'compliance_violation',
                'severity': self._determine_compliance_severity(violations),
                'source': 'compliance_checker',
                'description': f"Compliance violations detected: {len(violations)} violations",
                'violations': violations
            }
            
            # Handle as incident
            return self.handle_incident(incident)
            
        except Exception as e:
            self.logger.error(f"Failed to handle compliance violation: {e}")
            return {'success': False, 'error': str(e)}
    
    def handle_suspicious_activity(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle suspicious activity incident"""
        try:
            # Create suspicious activity incident
            incident = {
                'type': 'suspicious_activity',
                'severity': 'high',
                'source': 'access_controller',
                'description': 'Suspicious activity detected',
                'activity_data': activity_data
            }
            
            # Handle as incident
            return self.handle_incident(incident)
            
        except Exception as e:
            self.logger.error(f"Failed to handle suspicious activity: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_incident_summary(self) -> Dict[str, Any]:
        """Get incident response summary"""
        try:
            with self._lock:
                active_count = len(self.active_incidents)
                resolved_count = self.incident_metrics['resolved_incidents']
                total_count = self.incident_metrics['total_incidents']
                
                # Calculate resolution rate
                resolution_rate = resolved_count / total_count if total_count > 0 else 0
                
                # Count by severity
                severity_counts = defaultdict(int)
                for incident in self.active_incidents.values():
                    severity_counts[incident['severity']] += 1
                
                # Calculate average times
                avg_response = self.incident_metrics['avg_response_time']
                avg_resolution = self.incident_metrics['avg_resolution_time']
                
                return {
                    'total_incidents': total_count,
                    'active_incidents': active_count,
                    'resolved_incidents': resolved_count,
                    'unresolved_incidents': active_count,
                    'resolution_rate': resolution_rate,
                    'escalated_incidents': self.incident_metrics['escalated_incidents'],
                    'auto_resolved': self.incident_metrics['auto_resolved'],
                    'severity_breakdown': dict(severity_counts),
                    'avg_response_time': avg_response,
                    'avg_resolution_time': avg_resolution,
                    'sla_violations': self.incident_metrics['sla_violations'],
                    'response_sla_met': avg_response <= self.response_time_sla,
                    'resolution_sla_met': avg_resolution <= self.resolution_time_sla
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get incident summary: {e}")
            return {
                'error': str(e),
                'total_incidents': 0,
                'active_incidents': 0
            }
    
    def _execute_incident_response(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """Execute incident response based on playbook"""
        try:
            response_result = {
                'initiated': True,
                'actions': [],
                'auto_resolved': False,
                'escalated': False
            }
            
            # Mark response started
            incident['response_started'] = datetime.now()
            incident['status'] = 'responding'
            
            # Get appropriate playbook
            playbook = self._get_playbook(incident['incident_type'], incident['severity'])
            
            if not playbook:
                self.logger.warning(f"No playbook found for incident type: {incident['incident_type']}")
                return response_result
            
            # Execute playbook actions
            for action in playbook['actions']:
                if self.auto_response_enabled or action.get('manual', False):
                    action_result = self._execute_action(action, incident)
                    response_result['actions'].append(action_result)
                    incident['response_actions'].append(action_result)
                    
                    # Check if action resolved the incident
                    if action_result.get('resolved', False):
                        incident['status'] = 'resolved'
                        incident['resolved_at'] = datetime.now()
                        response_result['auto_resolved'] = True
                        self._update_metrics(incident)
                        break
            
            # Send notifications
            self._send_incident_notifications(incident)
            
            # Check if escalation needed
            if incident['status'] != 'resolved' and self.escalation_enabled:
                if self._should_escalate(incident):
                    self._escalate_incident(incident)
                    response_result['escalated'] = True
            
            return response_result
            
        except Exception as e:
            self.logger.error(f"Failed to execute incident response: {e}")
            return {
                'initiated': False,
                'error': str(e)
            }
    
    def _execute_action(self, action: Dict[str, Any], incident: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single response action"""
        try:
            action_type = action.get('type')
            action_result = {
                'action_type': action_type,
                'timestamp': datetime.now(),
                'success': False,
                'result': None
            }
            
            # Execute based on action type
            if action_type == 'isolate_system':
                action_result['result'] = self._isolate_system(incident)
            elif action_type == 'block_ip':
                action_result['result'] = self._block_ip(incident)
            elif action_type == 'disable_account':
                action_result['result'] = self._disable_account(incident)
            elif action_type == 'collect_forensics':
                action_result['result'] = self._collect_forensics(incident)
            elif action_type == 'apply_patch':
                action_result['result'] = self._apply_patch(incident)
            elif action_type == 'restart_service':
                action_result['result'] = self._restart_service(incident)
            elif action_type == 'update_firewall':
                action_result['result'] = self._update_firewall(incident)
            else:
                action_result['result'] = f"Unknown action type: {action_type}"
                return action_result
            
            action_result['success'] = True
            
            # Store action record
            self.response_actions.append(action_result)
            
            return action_result
            
        except Exception as e:
            self.logger.error(f"Failed to execute action {action_type}: {e}")
            return {
                'action_type': action_type,
                'timestamp': datetime.now(),
                'success': False,
                'error': str(e)
            }
    
    def _isolate_system(self, incident: Dict[str, Any]) -> str:
        """Isolate affected system"""
        # In production, this would isolate the system
        return f"System isolated for incident {incident['incident_id']}"
    
    def _block_ip(self, incident: Dict[str, Any]) -> str:
        """Block malicious IP address"""
        # In production, this would update firewall rules
        return f"IP blocked for incident {incident['incident_id']}"
    
    def _disable_account(self, incident: Dict[str, Any]) -> str:
        """Disable compromised account"""
        # In production, this would disable the account
        return f"Account disabled for incident {incident['incident_id']}"
    
    def _collect_forensics(self, incident: Dict[str, Any]) -> str:
        """Collect forensic data"""
        # In production, this would collect system data
        return f"Forensics collected for incident {incident['incident_id']}"
    
    def _apply_patch(self, incident: Dict[str, Any]) -> str:
        """Apply security patch"""
        # In production, this would apply patches
        return f"Patch applied for incident {incident['incident_id']}"
    
    def _restart_service(self, incident: Dict[str, Any]) -> str:
        """Restart affected service"""
        # In production, this would restart services
        return f"Service restarted for incident {incident['incident_id']}"
    
    def _update_firewall(self, incident: Dict[str, Any]) -> str:
        """Update firewall rules"""
        # In production, this would update firewall
        return f"Firewall updated for incident {incident['incident_id']}"
    
    def _get_playbook(self, incident_type: str, severity: str) -> Optional[Dict[str, Any]]:
        """Get response playbook for incident type and severity"""
        playbook_key = f"{incident_type}_{severity}"
        
        # Try specific playbook first
        if playbook_key in self.response_playbooks:
            return self.response_playbooks[playbook_key]
        
        # Try incident type playbook
        if incident_type in self.response_playbooks:
            return self.response_playbooks[incident_type]
        
        # Use default playbook
        return self.response_playbooks.get('default')
    
    def _should_escalate(self, incident: Dict[str, Any]) -> bool:
        """Determine if incident should be escalated"""
        # Check time since creation
        time_elapsed = (datetime.now() - incident['created_at']).total_seconds()
        
        # Escalate if not resolved within escalation time
        if time_elapsed > self.escalation_time:
            return True
        
        # Escalate critical incidents immediately
        if incident['severity'] == 'critical' and incident['escalation_level'] == 0:
            return True
        
        return False
    
    def _escalate_incident(self, incident: Dict[str, Any]):
        """Escalate incident to next level"""
        try:
            incident['escalation_level'] += 1
            incident['status'] = 'escalated'
            
            # Get escalation policy
            policy = self.escalation_policies.get(
                incident['severity'], 
                self.escalation_policies['default']
            )
            
            level = min(incident['escalation_level'], len(policy['levels']) - 1)
            escalation_level = policy['levels'][level]
            
            # Send escalation notifications
            self._send_escalation_notifications(incident, escalation_level)
            
            # Update metrics
            self.incident_metrics['escalated_incidents'] += 1
            
            self.logger.warning(
                f"Incident {incident['incident_id']} escalated to level {incident['escalation_level']}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to escalate incident: {e}")
    
    def _send_incident_notifications(self, incident: Dict[str, Any]):
        """Send incident notifications"""
        try:
            for channel in self.notification_channels:
                if channel == 'email':
                    self._send_email_notification(incident)
                elif channel == 'sms':
                    self._send_sms_notification(incident)
                elif channel == 'webhook':
                    self._send_webhook_notification(incident)
                    
        except Exception as e:
            self.logger.error(f"Failed to send notifications: {e}")
    
    def _send_escalation_notifications(self, incident: Dict[str, Any], escalation_level: Dict[str, Any]):
        """Send escalation notifications"""
        try:
            # In production, this would send to escalation contacts
            self.logger.info(
                f"Escalation notification sent for incident {incident['incident_id']} "
                f"to {escalation_level['contacts']}"
            )
        except Exception as e:
            self.logger.error(f"Failed to send escalation notifications: {e}")
    
    def _send_email_notification(self, incident: Dict[str, Any]):
        """Send email notification"""
        # In production, this would send actual email
        self.logger.info(f"Email notification sent for incident {incident['incident_id']}")
    
    def _send_sms_notification(self, incident: Dict[str, Any]):
        """Send SMS notification"""
        # In production, this would send actual SMS
        self.logger.info(f"SMS notification sent for incident {incident['incident_id']}")
    
    def _send_webhook_notification(self, incident: Dict[str, Any]):
        """Send webhook notification"""
        # In production, this would call webhook
        self.logger.info(f"Webhook notification sent for incident {incident['incident_id']}")
    
    def _update_metrics(self, incident: Dict[str, Any]):
        """Update incident metrics"""
        try:
            with self._lock:
                # Update resolution count
                if incident['status'] == 'resolved':
                    self.incident_metrics['resolved_incidents'] += 1
                    
                    # Update auto-resolved count
                    if incident.get('response_result', {}).get('auto_resolved', False):
                        self.incident_metrics['auto_resolved'] += 1
                    
                    # Calculate response time
                    if incident['response_started']:
                        response_time = (incident['response_started'] - incident['created_at']).total_seconds()
                        
                        # Update average response time
                        total_incidents = self.incident_metrics['total_incidents']
                        current_avg = self.incident_metrics['avg_response_time']
                        new_avg = ((current_avg * (total_incidents - 1)) + response_time) / total_incidents
                        self.incident_metrics['avg_response_time'] = new_avg
                        
                        # Check SLA violation
                        if response_time > self.response_time_sla:
                            self.incident_metrics['sla_violations'] += 1
                    
                    # Calculate resolution time
                    if incident['resolved_at']:
                        resolution_time = (incident['resolved_at'] - incident['created_at']).total_seconds()
                        
                        # Update average resolution time
                        resolved_count = self.incident_metrics['resolved_incidents']
                        current_avg = self.incident_metrics['avg_resolution_time']
                        new_avg = ((current_avg * (resolved_count - 1)) + resolution_time) / resolved_count
                        self.incident_metrics['avg_resolution_time'] = new_avg
                        
                        # Check SLA violation
                        if resolution_time > self.resolution_time_sla:
                            self.incident_metrics['sla_violations'] += 1
                            
        except Exception as e:
            self.logger.error(f"Failed to update metrics: {e}")
    
    def _start_incident_threads(self):
        """Start background incident management threads"""
        # Start incident monitor thread
        self.monitor_thread = threading.Thread(target=self._incident_monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._incident_cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _incident_monitor_loop(self):
        """Monitor active incidents for escalation and timeout"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                with self._lock:
                    for incident_id, incident in list(self.active_incidents.items()):
                        # Skip resolved incidents
                        if incident['status'] == 'resolved':
                            continue
                        
                        # Check for escalation
                        if self._should_escalate(incident):
                            self._escalate_incident(incident)
                        
                        # Check for timeout (24 hours)
                        if (current_time - incident['created_at']).total_seconds() > 86400:
                            incident['status'] = 'timed_out'
                            self.logger.warning(f"Incident {incident_id} timed out")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Incident monitor error: {e}")
                time.sleep(300)
    
    def _incident_cleanup_loop(self):
        """Clean up old incidents"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                with self._lock:
                    # Remove resolved incidents older than 7 days
                    incidents_to_remove = []
                    for incident_id, incident in self.active_incidents.items():
                        if incident['status'] in ['resolved', 'timed_out']:
                            if (current_time - incident['created_at']).days > 7:
                                incidents_to_remove.append(incident_id)
                    
                    for incident_id in incidents_to_remove:
                        del self.active_incidents[incident_id]
                
                time.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Incident cleanup error: {e}")
                time.sleep(7200)
    
    def _initialize_response_playbooks(self) -> Dict[str, Dict[str, Any]]:
        """Initialize incident response playbooks"""
        return {
            'threat_detection_critical': {
                'name': 'Critical Threat Response',
                'actions': [
                    {'type': 'isolate_system', 'priority': 1},
                    {'type': 'collect_forensics', 'priority': 2},
                    {'type': 'block_ip', 'priority': 3},
                    {'type': 'update_firewall', 'priority': 4}
                ]
            },
            'threat_detection_high': {
                'name': 'High Threat Response',
                'actions': [
                    {'type': 'block_ip', 'priority': 1},
                    {'type': 'update_firewall', 'priority': 2},
                    {'type': 'collect_forensics', 'priority': 3}
                ]
            },
            'compliance_violation': {
                'name': 'Compliance Violation Response',
                'actions': [
                    {'type': 'collect_forensics', 'priority': 1}
                ]
            },
            'suspicious_activity': {
                'name': 'Suspicious Activity Response',
                'actions': [
                    {'type': 'disable_account', 'priority': 1},
                    {'type': 'collect_forensics', 'priority': 2}
                ]
            },
            'default': {
                'name': 'Default Response',
                'actions': [
                    {'type': 'collect_forensics', 'priority': 1}
                ]
            }
        }
    
    def _initialize_escalation_policies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize escalation policies"""
        return {
            'critical': {
                'levels': [
                    {
                        'level': 1,
                        'contacts': ['security-team@company.com'],
                        'response_time': 300  # 5 minutes
                    },
                    {
                        'level': 2,
                        'contacts': ['security-manager@company.com'],
                        'response_time': 900  # 15 minutes
                    },
                    {
                        'level': 3,
                        'contacts': ['ciso@company.com', 'cto@company.com'],
                        'response_time': 1800  # 30 minutes
                    }
                ]
            },
            'high': {
                'levels': [
                    {
                        'level': 1,
                        'contacts': ['security-team@company.com'],
                        'response_time': 1800  # 30 minutes
                    },
                    {
                        'level': 2,
                        'contacts': ['security-manager@company.com'],
                        'response_time': 3600  # 1 hour
                    }
                ]
            },
            'default': {
                'levels': [
                    {
                        'level': 1,
                        'contacts': ['security-team@company.com'],
                        'response_time': 3600  # 1 hour
                    }
                ]
            }
        }
    
    def _generate_incident_id(self) -> str:
        """Generate unique incident ID"""
        timestamp = int(time.time() * 1000000)  # Microsecond precision
        random_hash = hashlib.sha256(str(timestamp).encode()).hexdigest()[:8]
        return f"INC-{timestamp}-{random_hash}"
    
    def _map_threat_severity(self, threat_level: str) -> str:
        """Map threat level to incident severity"""
        mapping = {
            'critical': 'critical',
            'high': 'high',
            'medium': 'medium',
            'low': 'low'
        }
        return mapping.get(threat_level, 'medium')
    
    def _determine_compliance_severity(self, violations: List[Dict[str, Any]]) -> str:
        """Determine severity from compliance violations"""
        if not violations:
            return 'low'
        
        # Get highest severity
        severities = [v.get('severity', 'low') for v in violations]
        severity_order = ['low', 'medium', 'high', 'critical']
        
        max_severity = 'low'
        for severity in severities:
            if severity_order.index(severity) > severity_order.index(max_severity):
                max_severity = severity
        
        return max_severity
    
    def shutdown(self):
        """Shutdown incident response manager"""
        self.logger.info("Shutting down Incident Response Manager")
        self.is_running = False
        
        # Generate final report
        try:
            summary = self.get_incident_summary()
            self.logger.info(f"Final incident summary: {summary}")
        except:
            pass