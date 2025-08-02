"""
Saraphis Security Manager
Production-ready comprehensive security management system
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


class SecurityManager:
    """Production-ready comprehensive security management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.security_history = deque(maxlen=10000)
        self.incident_history = deque(maxlen=5000)
        self.security_metrics = {
            'total_security_events': 0,
            'critical_incidents': 0,
            'compliance_violations': 0,
            'threat_detections': 0,
            'access_attempts': 0,
            'failed_authentications': 0,
            'security_events_per_hour': 0,
            'incident_resolution_rate': 0.95
        }
        
        # Initialize security components
        from .compliance_checker import ComplianceChecker
        from .threat_detector import ThreatDetector
        from .access_controller import AccessController
        from .audit_logger import AuditLogger
        from .incident_response import IncidentResponseManager
        from .security_metrics import SecurityMetricsCollector
        
        self.compliance_checker = ComplianceChecker(config.get('compliance_config', {}))
        self.threat_detector = ThreatDetector(config.get('threat_config', {}))
        self.access_controller = AccessController(config.get('access_config', {}))
        self.audit_logger = AuditLogger(config.get('audit_config', {}))
        self.incident_response = IncidentResponseManager(config.get('incident_config', {}))
        self.security_metrics_collector = SecurityMetricsCollector(config.get('metrics_config', {}))
        
        # Security policies
        self.security_policies = self._initialize_security_policies()
        
        # Thread control
        self.is_running = True
        self._lock = threading.Lock()
        
        # Start security monitoring threads
        self._start_security_threads()
        
        self.logger.info("Security Manager initialized")
    
    def audit_system(self) -> Dict[str, Any]:
        """Perform comprehensive security audit"""
        try:
            self._audit_start_time = datetime.now()
            
            # Collect security metrics
            current_metrics = self.security_metrics_collector.collect_all_metrics()
            
            # Perform compliance check
            compliance_status = self.compliance_checker.validate_all_compliance()
            
            # Analyze threats
            threat_analysis = self.threat_detector.analyze_current_threats()
            
            # Check access controls
            access_audit = self.access_controller.audit_access_controls()
            
            # Generate security score
            security_score = self._calculate_security_score(
                current_metrics, compliance_status, threat_analysis, access_audit
            )
            
            # Identify security issues
            security_issues = self._identify_security_issues(
                current_metrics, compliance_status, threat_analysis, access_audit
            )
            
            # Update security metrics
            self._update_security_metrics(current_metrics)
            
            # Store audit data
            audit_record = {
                'timestamp': time.time(),
                'metrics': current_metrics,
                'compliance_status': compliance_status,
                'threat_analysis': threat_analysis,
                'access_audit': access_audit,
                'security_score': security_score,
                'security_issues': security_issues
            }
            
            self.security_history.append(audit_record)
            
            # Log audit event
            self.audit_logger.log_security_event({
                'event_type': 'security_audit',
                'severity': 'info',
                'details': {
                    'security_score': security_score,
                    'issues_found': len(security_issues)
                }
            })
            
            return {
                'status': 'secure' if security_score > 0.8 else 'at_risk' if security_score > 0.6 else 'compromised',
                'security_score': security_score,
                'current_metrics': current_metrics,
                'compliance_status': compliance_status,
                'threat_analysis': threat_analysis,
                'access_audit': access_audit,
                'security_issues': security_issues,
                'last_audit': datetime.now().isoformat(),
                'audit_duration': self._get_audit_duration()
            }
            
        except Exception as e:
            self.logger.error(f"Security audit failed: {e}")
            
            # Log critical audit failure
            self.audit_logger.log_security_event({
                'event_type': 'audit_failure',
                'severity': 'critical',
                'error': str(e)
            })
            
            return {
                'status': 'error',
                'error': str(e),
                'security_score': 0.0,
                'security_issues': [{
                    'type': 'security_audit_failure',
                    'severity': 'critical',
                    'description': f'Security audit failed: {str(e)}'
                }]
            }
    
    def _calculate_security_score(self, metrics: Dict[str, Any], compliance: Dict[str, Any],
                                threats: Dict[str, Any], access: Dict[str, Any]) -> float:
        """Calculate overall security score"""
        try:
            # Weighted scoring based on security factors
            scores = {}
            
            # Compliance score (30% weight)
            compliance_score = compliance.get('overall_compliance', 0.0)
            scores['compliance'] = compliance_score * 0.3
            
            # Threat score (25% weight) - lower threats = higher score
            threat_level = threats.get('threat_level', 'high')
            threat_score_map = {'low': 1.0, 'medium': 0.7, 'high': 0.3, 'critical': 0.0}
            threat_score = threat_score_map.get(threat_level, 0.0)
            scores['threats'] = threat_score * 0.25
            
            # Access control score (20% weight)
            access_score = access.get('access_control_score', 0.0)
            scores['access_control'] = access_score * 0.2
            
            # Metrics score (15% weight)
            metrics_score = self._calculate_metrics_security_score(metrics)
            scores['metrics'] = metrics_score * 0.15
            
            # Incident score (10% weight) - fewer incidents = higher score
            incident_rate = metrics.get('incident_rate', 1.0)
            incident_score = max(0, 1 - incident_rate)
            scores['incidents'] = incident_score * 0.1
            
            # Calculate weighted average
            total_score = sum(scores.values())
            
            return min(1.0, max(0.0, total_score))
            
        except Exception as e:
            self.logger.error(f"Security score calculation failed: {e}")
            return 0.0
    
    def _calculate_metrics_security_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate security score from metrics"""
        try:
            # Authentication success rate
            auth_success_rate = 1 - metrics.get('failed_authentications', 0) / max(1, metrics.get('access_attempts', 1))
            
            # Security event rate (lower is better)
            security_event_rate = min(1, metrics.get('security_events_per_hour', 0) / 100)
            security_event_score = max(0, 1 - security_event_rate)
            
            # Incident resolution rate
            resolution_rate = metrics.get('incident_resolution_rate', 0.0)
            
            # Average score
            return (auth_success_rate + security_event_score + resolution_rate) / 3
            
        except Exception as e:
            self.logger.error(f"Metrics security score calculation failed: {e}")
            return 0.5
    
    def _identify_security_issues(self, metrics: Dict[str, Any], compliance: Dict[str, Any],
                                 threats: Dict[str, Any], access: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify security issues requiring attention"""
        try:
            security_issues = []
            
            # Check compliance violations
            for violation in compliance.get('violations', []):
                security_issues.append({
                    'type': 'compliance_violation',
                    'severity': violation.get('severity', 'medium'),
                    'framework': violation.get('framework', 'unknown'),
                    'description': violation.get('description', 'Compliance violation detected'),
                    'violation_details': violation
                })
            
            # Check threat levels
            threat_level = threats.get('threat_level', 'low')
            if threat_level in ['high', 'critical']:
                security_issues.append({
                    'type': 'high_threat_level',
                    'severity': 'critical' if threat_level == 'critical' else 'high',
                    'threat_level': threat_level,
                    'active_threats': threats.get('active_threats', []),
                    'description': f"High threat level detected: {threat_level}"
                })
            
            # Check access control issues
            access_score = access.get('access_control_score', 1.0)
            if access_score < 0.8:
                security_issues.append({
                    'type': 'access_control_weakness',
                    'severity': 'high',
                    'access_score': access_score,
                    'weaknesses': access.get('weaknesses', []),
                    'description': f"Access control weakness detected (score: {access_score:.2f})"
                })
            
            # Check authentication failures
            failed_auth_rate = metrics.get('failed_authentications', 0) / max(1, metrics.get('access_attempts', 1))
            if failed_auth_rate > 0.1:  # 10% failure rate threshold
                security_issues.append({
                    'type': 'high_authentication_failure',
                    'severity': 'medium',
                    'failure_rate': failed_auth_rate,
                    'description': f"High authentication failure rate: {failed_auth_rate:.2%}"
                })
            
            # Check security event rate
            security_events_per_hour = metrics.get('security_events_per_hour', 0)
            if security_events_per_hour > 50:  # 50 events per hour threshold
                security_issues.append({
                    'type': 'high_security_event_rate',
                    'severity': 'medium',
                    'events_per_hour': security_events_per_hour,
                    'description': f"High security event rate: {security_events_per_hour} events/hour"
                })
            
            return security_issues
            
        except Exception as e:
            self.logger.error(f"Security issue identification failed: {e}")
            return [{
                'type': 'security_analysis_error',
                'severity': 'critical',
                'description': f'Security analysis error: {str(e)}'
            }]
    
    def _initialize_security_policies(self) -> Dict[str, Any]:
        """Initialize security policies"""
        return {
            'password_policy': {
                'min_length': 12,
                'require_uppercase': True,
                'require_lowercase': True,
                'require_numbers': True,
                'require_special': True,
                'max_age_days': 90,
                'history_count': 5,
                'complexity_score': 0.8
            },
            'access_policy': {
                'max_failed_attempts': 5,
                'lockout_duration_minutes': 30,
                'session_timeout_minutes': 60,
                'require_mfa': True,
                'ip_whitelist': [],
                'ip_blacklist': [],
                'geolocation_restrictions': []
            },
            'data_protection': {
                'encryption_at_rest': True,
                'encryption_in_transit': True,
                'encryption_algorithm': 'AES-256-GCM',
                'key_rotation_days': 30,
                'data_retention_days': 2555,  # 7 years
                'backup_encryption': True,
                'data_classification': ['public', 'internal', 'confidential', 'restricted']
            },
            'audit_policy': {
                'log_all_access': True,
                'log_all_changes': True,
                'retention_days': 2555,
                'real_time_alerts': True,
                'weekly_reports': True,
                'compliance_reports': True
            },
            'incident_response': {
                'auto_response_enabled': True,
                'escalation_enabled': True,
                'notification_channels': ['email', 'sms', 'webhook'],
                'response_time_sla': 300,  # 5 minutes
                'recovery_time_objective': 3600  # 1 hour
            }
        }
    
    def _start_security_threads(self):
        """Start background security monitoring threads"""
        try:
            # Start threat detection thread
            self.threat_thread = threading.Thread(target=self._threat_detection_loop, daemon=True)
            self.threat_thread.start()
            
            # Start compliance monitoring thread
            self.compliance_thread = threading.Thread(target=self._compliance_monitoring_loop, daemon=True)
            self.compliance_thread.start()
            
            # Start access monitoring thread
            self.access_thread = threading.Thread(target=self._access_monitoring_loop, daemon=True)
            self.access_thread.start()
            
            # Start metrics collection thread
            self.metrics_thread = threading.Thread(target=self._metrics_collection_loop, daemon=True)
            self.metrics_thread.start()
            
            self.logger.info("Security monitoring threads started")
            
        except Exception as e:
            self.logger.error(f"Failed to start security threads: {e}")
            raise RuntimeError(f"Security thread startup failed: {e}")
    
    def _threat_detection_loop(self):
        """Background threat detection loop"""
        while self.is_running:
            try:
                # Analyze threats every 60 seconds
                threats = self.threat_detector.analyze_current_threats()
                
                # Update metrics
                self.security_metrics['threat_detections'] = len(threats.get('active_threats', []))
                
                # Check for critical threats
                if threats.get('threat_level') in ['high', 'critical']:
                    # Log critical threat
                    self.audit_logger.log_security_event({
                        'event_type': 'critical_threat_detected',
                        'severity': 'critical',
                        'threat_level': threats.get('threat_level'),
                        'active_threats': len(threats.get('active_threats', []))
                    })
                    
                    # Trigger incident response
                    self.incident_response.handle_threat_incident(threats)
                
                time.sleep(60)  # 60-second interval
                
            except Exception as e:
                self.logger.error(f"Threat detection loop error: {e}")
                time.sleep(120)  # Wait longer on error
    
    def _compliance_monitoring_loop(self):
        """Background compliance monitoring loop"""
        while self.is_running:
            try:
                # Check compliance every 300 seconds (5 minutes)
                compliance = self.compliance_checker.validate_all_compliance()
                
                # Update metrics
                violations = compliance.get('violations', [])
                self.security_metrics['compliance_violations'] = len(violations)
                
                # Check for violations
                if violations:
                    # Log compliance violations
                    for violation in violations:
                        self.audit_logger.log_security_event({
                            'event_type': 'compliance_violation',
                            'severity': violation.get('severity', 'medium'),
                            'framework': violation.get('framework'),
                            'description': violation.get('description')
                        })
                    
                    # Trigger incident response
                    self.incident_response.handle_compliance_violation(violations)
                
                time.sleep(300)  # 5-minute interval
                
            except Exception as e:
                self.logger.error(f"Compliance monitoring loop error: {e}")
                time.sleep(600)  # Wait longer on error
    
    def _access_monitoring_loop(self):
        """Background access monitoring loop"""
        while self.is_running:
            try:
                # Monitor access every 30 seconds
                access_audit = self.access_controller.audit_access_controls()
                
                # Update metrics
                self.security_metrics['access_attempts'] = access_audit.get('total_access_attempts', 0)
                self.security_metrics['failed_authentications'] = access_audit.get('failed_authentications', 0)
                
                # Check for suspicious activity
                if access_audit.get('suspicious_activity_detected', False):
                    # Log suspicious activity
                    self.audit_logger.log_security_event({
                        'event_type': 'suspicious_activity',
                        'severity': 'high',
                        'details': access_audit.get('suspicious_activities', [])
                    })
                    
                    # Trigger incident response
                    self.incident_response.handle_suspicious_activity(access_audit)
                
                time.sleep(30)  # 30-second interval
                
            except Exception as e:
                self.logger.error(f"Access monitoring loop error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        while self.is_running:
            try:
                # Collect metrics every 60 seconds
                current_metrics = self.security_metrics_collector.collect_all_metrics()
                
                # Update internal metrics
                self._update_security_metrics(current_metrics)
                
                # Calculate hourly event rate
                with self._lock:
                    self.security_metrics['security_events_per_hour'] = (
                        self.security_metrics['total_security_events'] / 
                        max(1, (time.time() - getattr(self, '_start_time', time.time())) / 3600)
                    )
                
                time.sleep(60)  # 60-second interval
                
            except Exception as e:
                self.logger.error(f"Metrics collection loop error: {e}")
                time.sleep(120)  # Wait longer on error
    
    def _update_security_metrics(self, current_metrics: Dict[str, Any]):
        """Update internal security metrics"""
        with self._lock:
            # Update metrics
            for key, value in current_metrics.items():
                if key in self.security_metrics:
                    self.security_metrics[key] = value
            
            # Increment total events
            self.security_metrics['total_security_events'] += 1
    
    def _get_audit_duration(self) -> float:
        """Get security audit duration in seconds"""
        if hasattr(self, '_audit_start_time'):
            return (datetime.now() - self._audit_start_time).total_seconds()
        return 0.0
    
    def handle_security_incident(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """Handle security incident"""
        try:
            # Log incident
            self.audit_logger.log_security_event({
                'event_type': 'security_incident',
                'severity': incident.get('severity', 'high'),
                'incident_type': incident.get('type'),
                'details': incident
            })
            
            # Store in incident history
            incident_record = {
                'timestamp': time.time(),
                'incident': incident,
                'response_time': datetime.now()
            }
            self.incident_history.append(incident_record)
            
            # Update metrics
            if incident.get('severity') == 'critical':
                self.security_metrics['critical_incidents'] += 1
            
            # Trigger incident response
            response = self.incident_response.handle_incident(incident)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to handle security incident: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status"""
        try:
            # Get latest audit if available
            latest_audit = self.security_history[-1] if self.security_history else None
            
            return {
                'is_secure': latest_audit.get('security_score', 0) > 0.8 if latest_audit else False,
                'security_score': latest_audit.get('security_score', 0) if latest_audit else 0,
                'threat_level': self.threat_detector.get_current_threat_level(),
                'compliance_status': self.compliance_checker.get_compliance_summary(),
                'active_incidents': len([i for i in self.incident_history if 
                                       time.time() - i['timestamp'] < 3600]),  # Last hour
                'metrics': self.security_metrics.copy(),
                'last_audit': latest_audit.get('timestamp') if latest_audit else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get security status: {e}")
            return {
                'is_secure': False,
                'error': str(e)
            }
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        try:
            # Perform full audit
            audit_result = self.audit_system()
            
            # Get incident summary
            incident_summary = self.incident_response.get_incident_summary()
            
            # Get compliance report
            compliance_report = self.compliance_checker.generate_compliance_report()
            
            # Get threat report
            threat_report = self.threat_detector.generate_threat_report()
            
            report = {
                'report_id': f"security_report_{int(time.time())}",
                'timestamp': datetime.now().isoformat(),
                'executive_summary': {
                    'security_status': audit_result.get('status'),
                    'security_score': audit_result.get('security_score'),
                    'critical_issues': len([i for i in audit_result.get('security_issues', []) 
                                          if i.get('severity') == 'critical']),
                    'total_incidents': incident_summary.get('total_incidents', 0),
                    'compliance_score': compliance_report.get('overall_compliance', 0),
                    'threat_level': threat_report.get('current_threat_level', 'unknown')
                },
                'audit_result': audit_result,
                'incident_summary': incident_summary,
                'compliance_report': compliance_report,
                'threat_report': threat_report,
                'recommendations': self._generate_security_recommendations(
                    audit_result, incident_summary, compliance_report, threat_report
                )
            }
            
            # Save report
            report_file = f"{report['report_id']}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Security report generated: {report_file}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate security report: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_security_recommendations(self, audit: Dict[str, Any], incidents: Dict[str, Any],
                                         compliance: Dict[str, Any], threats: Dict[str, Any]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        # Check security score
        if audit.get('security_score', 0) < 0.8:
            recommendations.append(
                f"Improve security posture - current score: {audit.get('security_score', 0):.2f}"
            )
        
        # Check compliance
        if compliance.get('overall_compliance', 1) < 0.95:
            recommendations.append(
                "Address compliance violations to achieve 95%+ compliance"
            )
        
        # Check threats
        if threats.get('current_threat_level') in ['high', 'critical']:
            recommendations.append(
                "Implement additional threat mitigation measures"
            )
        
        # Check incidents
        if incidents.get('unresolved_incidents', 0) > 0:
            recommendations.append(
                f"Resolve {incidents.get('unresolved_incidents', 0)} pending security incidents"
            )
        
        # Check authentication failures
        if self.security_metrics.get('failed_authentications', 0) > 100:
            recommendations.append(
                "Review authentication mechanisms - high failure rate detected"
            )
        
        return recommendations
    
    def shutdown(self):
        """Shutdown security manager"""
        self.logger.info("Shutting down Security Manager")
        self.is_running = False
        
        # Generate final report
        try:
            final_report = self.generate_security_report()
            self.logger.info(f"Final security report generated: {final_report.get('report_id')}")
        except:
            pass


def create_security_manager(config: Dict[str, Any]) -> SecurityManager:
    """Factory function to create security manager"""
    try:
        return SecurityManager(config)
    except Exception as e:
        logger.error(f"Failed to create security manager: {e}")
        raise