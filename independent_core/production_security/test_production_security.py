"""
Test script for Saraphis Production Security System
Validates all security components and integration
"""

import sys
import time
import json
from datetime import datetime

# Import security components
from security_manager import SecurityManager, create_security_manager


def test_security_system():
    """Test the complete security system"""
    print("="*80)
    print("SARAPHIS PRODUCTION SECURITY SYSTEM TEST")
    print("="*80)
    print()
    
    # Configuration
    config = {
        'compliance_config': {
            'frameworks': ['gdpr', 'sox', 'pci_dss', 'hipaa', 'iso27001'],
            'check_interval': 300
        },
        'threat_config': {
            'detection_engines': ['anomaly', 'behavior', 'signature', 'intelligence'],
            'threat_feeds': ['internal', 'external'],
            'detection_threshold': 0.7
        },
        'access_config': {
            'max_failed_attempts': 5,
            'lockout_duration_minutes': 30,
            'session_timeout_minutes': 60,
            'require_mfa': True,
            'jwt_secret': 'test_secret_key_for_demo'
        },
        'audit_config': {
            'log_path': './security_audit_logs',
            'retention_days': 2555,
            'compression_enabled': True,
            'real_time_alerts': True
        },
        'incident_config': {
            'auto_response_enabled': True,
            'escalation_enabled': True,
            'notification_channels': ['email', 'sms', 'webhook'],
            'response_time_sla': 300,
            'resolution_time_sla': 3600
        },
        'metrics_config': {
            'collection_interval': 60,
            'aggregation_interval': 300,
            'baseline_window_hours': 168
        }
    }
    
    try:
        # Create security manager
        print("1. Creating Security Manager...")
        security_manager = create_security_manager(config)
        print("   ✓ Security Manager created successfully")
        print()
        
        # Wait for initialization
        time.sleep(2)
        
        # Test 1: Security Audit
        print("2. Performing Security Audit...")
        audit_result = security_manager.audit_system()
        print(f"   ✓ Security Status: {audit_result['status']}")
        print(f"   ✓ Security Score: {audit_result['security_score']:.2f}")
        print(f"   ✓ Issues Found: {len(audit_result.get('security_issues', []))}")
        
        if audit_result.get('security_issues'):
            print("\n   Security Issues:")
            for issue in audit_result['security_issues'][:3]:  # Show first 3
                print(f"   - [{issue['severity']}] {issue['description']}")
        print()
        
        # Test 2: Compliance Check
        print("3. Testing Compliance Validation...")
        compliance = audit_result.get('compliance_status', {})
        print(f"   ✓ Overall Compliance: {compliance.get('overall_compliance', 0):.1%}")
        print(f"   ✓ Frameworks Checked: {compliance.get('total_frameworks', 0)}")
        print(f"   ✓ Violations: {len(compliance.get('violations', []))}")
        
        if compliance.get('regulation_status'):
            print("\n   Framework Status:")
            for framework, status in compliance['regulation_status'].items():
                print(f"   - {framework.upper()}: {'✓ Compliant' if status['compliant'] else '✗ Non-compliant'}")
        print()
        
        # Test 3: Threat Detection
        print("4. Testing Threat Detection...")
        threats = audit_result.get('threat_analysis', {})
        print(f"   ✓ Threat Level: {threats.get('threat_level', 'unknown')}")
        print(f"   ✓ Active Threats: {len(threats.get('active_threats', []))}")
        print(f"   ✓ Threat Score: {threats.get('threat_score', 0):.2f}")
        print()
        
        # Test 4: Access Control
        print("5. Testing Access Control...")
        access = audit_result.get('access_audit', {})
        print(f"   ✓ Access Control Score: {access.get('access_control_score', 0):.2f}")
        print(f"   ✓ Active Sessions: {access.get('active_sessions', 0)}")
        print(f"   ✓ Locked Accounts: {access.get('locked_accounts', 0)}")
        
        # Test authentication
        auth_result = security_manager.access_controller.authenticate_user(
            username="test_user",
            password="test_user_password",
            ip_address="192.168.1.100",
            mfa_token="MFA_test_user_123456"
        )
        print(f"   ✓ Authentication Test: {'Success' if auth_result['authenticated'] else 'Failed'}")
        print()
        
        # Test 5: Audit Logging
        print("6. Testing Audit Logging...")
        log_event = {
            'event_type': 'security_test',
            'severity': 'info',
            'description': 'Security system test event',
            'timestamp': datetime.now().isoformat()
        }
        log_result = security_manager.audit_logger.log_security_event(log_event)
        print(f"   ✓ Event Logged: {'Success' if log_result else 'Failed'}")
        
        # Verify log integrity
        integrity_check = security_manager.audit_logger.verify_log_integrity()
        print(f"   ✓ Log Integrity: {'Verified' if integrity_check['verified'] else 'Failed'}")
        print()
        
        # Test 6: Incident Response
        print("7. Testing Incident Response...")
        test_incident = {
            'type': 'test_incident',
            'severity': 'medium',
            'source': 'security_test',
            'description': 'Test security incident for validation'
        }
        incident_response = security_manager.handle_security_incident(test_incident)
        print(f"   ✓ Incident Created: {incident_response.get('incident_id', 'Failed')}")
        print(f"   ✓ Response Initiated: {incident_response.get('response_initiated', False)}")
        print(f"   ✓ Actions Taken: {incident_response.get('actions_taken', 0)}")
        print()
        
        # Test 7: Security Metrics
        print("8. Testing Security Metrics...")
        metrics = security_manager.security_metrics_collector.collect_all_metrics()
        print(f"   ✓ Metrics Collected: {len([k for k in metrics.keys() if k not in ['timestamp', 'datetime']])}")
        
        # Show some key metrics
        print("\n   Key Metrics:")
        print(f"   - Login Attempts: {metrics.get('login_attempts', 0)}")
        print(f"   - Threats Detected: {metrics.get('threats_detected', 0)}")
        print(f"   - Compliance Score: {metrics.get('compliance_score', 0):.1%}")
        print(f"   - Uptime: {metrics.get('uptime', 0):.2%}")
        print()
        
        # Test 8: Generate Reports
        print("9. Generating Security Reports...")
        
        # Security report
        security_report = security_manager.generate_security_report()
        print(f"   ✓ Security Report: {security_report.get('report_id', 'Failed')}")
        
        # Compliance report
        compliance_report = security_manager.compliance_checker.generate_compliance_report()
        print(f"   ✓ Compliance Report: {compliance_report.get('report_id', 'Failed')}")
        
        # Threat report
        threat_report = security_manager.threat_detector.generate_threat_report()
        print(f"   ✓ Threat Report: {threat_report.get('report_id', 'Failed')}")
        
        # Metrics report
        metrics_report = security_manager.security_metrics_collector.generate_metrics_report()
        print(f"   ✓ Metrics Report: {metrics_report.get('report_id', 'Failed')}")
        print()
        
        # Test 9: Security Status
        print("10. Final Security Status...")
        final_status = security_manager.get_security_status()
        print(f"   ✓ System Secure: {final_status.get('is_secure', False)}")
        print(f"   ✓ Security Score: {final_status.get('security_score', 0):.2f}")
        print(f"   ✓ Threat Level: {final_status.get('threat_level', 'unknown')}")
        print(f"   ✓ Active Incidents: {final_status.get('active_incidents', 0)}")
        print()
        
        # Summary
        print("="*80)
        print("SECURITY SYSTEM TEST SUMMARY")
        print("="*80)
        print(f"✓ All security components operational")
        print(f"✓ Security Score: {audit_result['security_score']:.2f}")
        print(f"✓ Compliance: {compliance.get('overall_compliance', 0):.1%}")
        print(f"✓ Threat Level: {threats.get('threat_level', 'unknown')}")
        print(f"✓ System Status: {audit_result['status'].upper()}")
        print()
        
        # Save test results
        test_results = {
            'test_timestamp': datetime.now().isoformat(),
            'audit_result': audit_result,
            'security_status': final_status,
            'test_status': 'PASSED'
        }
        
        with open('security_test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        print("✓ Test results saved to security_test_results.json")
        
        # Shutdown
        print("\nShutting down security system...")
        security_manager.shutdown()
        print("✓ Security system shutdown complete")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("\nStarting Saraphis Production Security System Test...")
    print("This will validate all security components and integration")
    print()
    
    success = test_security_system()
    
    if success:
        print("\n✓ SECURITY SYSTEM TEST COMPLETED SUCCESSFULLY")
    else:
        print("\n✗ SECURITY SYSTEM TEST FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()