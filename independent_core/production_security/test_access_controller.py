#!/usr/bin/env python3
"""
Comprehensive Test Suite for AccessController
Tests all aspects of production-ready access control and authentication
"""

import unittest
import time
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, patch

# Import the module to test
from access_controller import AccessController


class TestAccessController(unittest.TestCase):
    """Test AccessController main functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'max_failed_attempts': 5,
            'lockout_duration_minutes': 30,
            'session_timeout_minutes': 60,
            'require_mfa': True,
            'jwt_secret': 'test_secret_key_12345',
            'token_expiry_hours': 24,
            'ip_whitelist': ['192.168.1.100', '10.0.0.50'],
            'ip_blacklist': ['192.168.1.200']
        }
        self.controller = AccessController(self.config)
    
    def test_initialization(self):
        """Test controller initialization"""
        self.assertIsNotNone(self.controller)
        self.assertEqual(self.controller.max_failed_attempts, 5)
        self.assertEqual(self.controller.lockout_duration, 30)
        self.assertEqual(self.controller.session_timeout, 60)
        self.assertTrue(self.controller.require_mfa)
        
        # Check initial state
        self.assertEqual(len(self.controller.user_sessions), 0)
        self.assertEqual(len(self.controller.locked_accounts), 0)
        self.assertEqual(len(self.controller.active_tokens), 0)
        
        # Check metrics initialization
        self.assertEqual(self.controller.access_metrics['total_access_attempts'], 0)
        self.assertEqual(self.controller.access_metrics['successful_authentications'], 0)
        self.assertEqual(self.controller.access_metrics['failed_authentications'], 0)
    
    def test_initialization_with_defaults(self):
        """Test initialization with minimal config"""
        minimal_controller = AccessController({})
        
        # Check defaults
        self.assertEqual(minimal_controller.max_failed_attempts, 5)
        self.assertEqual(minimal_controller.lockout_duration, 30)
        self.assertEqual(minimal_controller.session_timeout, 60)
        self.assertTrue(minimal_controller.require_mfa)
        self.assertEqual(len(minimal_controller.ip_whitelist), 0)
        self.assertEqual(len(minimal_controller.ip_blacklist), 0)
    
    def test_audit_access_controls_basic(self):
        """Test basic access control audit"""
        audit_report = self.controller.audit_access_controls()
        
        # Check audit report structure
        self.assertIn('access_control_score', audit_report)
        self.assertIn('total_access_attempts', audit_report)
        self.assertIn('failed_authentications', audit_report)
        self.assertIn('success_rate', audit_report)
        self.assertIn('active_sessions', audit_report)
        self.assertIn('locked_accounts', audit_report)
        self.assertIn('weaknesses', audit_report)
        self.assertIn('suspicious_activity_detected', audit_report)
        self.assertIn('policy_compliance', audit_report)
        self.assertIn('last_audit', audit_report)
        
        # Check score range
        self.assertGreaterEqual(audit_report['access_control_score'], 0.0)
        self.assertLessEqual(audit_report['access_control_score'], 1.0)
        
        # Check compliance structure
        compliance = audit_report['policy_compliance']
        self.assertIn('mfa_enabled', compliance)
        self.assertIn('session_timeout_compliant', compliance)
        self.assertIn('lockout_policy_compliant', compliance)
        self.assertIn('ip_restrictions_enabled', compliance)
        self.assertIn('rbac_implemented', compliance)
    
    def test_authenticate_user_success_with_mfa(self):
        """Test successful user authentication with MFA"""
        username = 'testuser'
        password = 'testuser_password'
        ip_address = '192.168.1.100'
        mfa_token = 'MFA_testuser_123456'
        
        result = self.controller.authenticate_user(username, password, ip_address, mfa_token)
        
        self.assertTrue(result['authenticated'])
        self.assertIn('session_id', result)
        self.assertIn('token', result)
        self.assertIn('expires', result)
        
        # Check metrics
        self.assertEqual(self.controller.access_metrics['total_access_attempts'], 1)
        self.assertEqual(self.controller.access_metrics['successful_authentications'], 1)
        self.assertEqual(self.controller.access_metrics['failed_authentications'], 0)
        self.assertEqual(self.controller.access_metrics['session_creations'], 1)
        
        # Check session created
        self.assertEqual(len(self.controller.user_sessions), 1)
    
    def test_authenticate_user_missing_mfa(self):
        """Test authentication when MFA is required but not provided"""
        username = 'testuser'
        password = 'testuser_password'
        ip_address = '192.168.1.100'
        
        result = self.controller.authenticate_user(username, password, ip_address)
        
        self.assertFalse(result['authenticated'])
        self.assertTrue(result.get('mfa_required', False))
        self.assertEqual(result['code'], 'MFA_REQUIRED')
        
        # Check metrics
        self.assertEqual(self.controller.access_metrics['mfa_challenges'], 1)
        self.assertEqual(len(self.controller.user_sessions), 0)
    
    def test_authenticate_user_invalid_credentials(self):
        """Test authentication with invalid credentials"""
        username = 'testuser'
        password = 'wrong_password'
        ip_address = '192.168.1.100'
        mfa_token = 'MFA_testuser_123456'
        
        result = self.controller.authenticate_user(username, password, ip_address, mfa_token)
        
        self.assertFalse(result['authenticated'])
        self.assertEqual(result['code'], 'INVALID_CREDENTIALS')
        
        # Check failed attempt recorded
        self.assertEqual(self.controller.access_metrics['failed_authentications'], 1)
        self.assertIn(username, self.controller.failed_attempts)
    
    def test_authenticate_user_invalid_mfa(self):
        """Test authentication with invalid MFA token"""
        username = 'testuser'
        password = 'testuser_password'
        ip_address = '192.168.1.100'
        mfa_token = 'WRONG_MFA_TOKEN'
        
        result = self.controller.authenticate_user(username, password, ip_address, mfa_token)
        
        self.assertFalse(result['authenticated'])
        self.assertEqual(result['code'], 'INVALID_MFA')
        
        # Check failed attempt recorded
        self.assertEqual(self.controller.access_metrics['failed_authentications'], 1)
    
    def test_authenticate_user_ip_blacklisted(self):
        """Test authentication from blacklisted IP"""
        username = 'testuser'
        password = 'testuser_password'
        ip_address = '192.168.1.200'  # Blacklisted in config
        mfa_token = 'MFA_testuser_123456'
        
        result = self.controller.authenticate_user(username, password, ip_address, mfa_token)
        
        self.assertFalse(result['authenticated'])
        self.assertEqual(result['code'], 'IP_DENIED')
        self.assertEqual(self.controller.access_metrics['access_denials'], 1)
    
    def test_authenticate_user_ip_not_whitelisted(self):
        """Test authentication from non-whitelisted IP"""
        username = 'testuser'
        password = 'testuser_password'
        ip_address = '10.0.0.99'  # Not in whitelist
        mfa_token = 'MFA_testuser_123456'
        
        result = self.controller.authenticate_user(username, password, ip_address, mfa_token)
        
        self.assertFalse(result['authenticated'])
        self.assertEqual(result['code'], 'IP_DENIED')
    
    def test_authenticate_user_account_locked(self):
        """Test authentication when account is locked"""
        username = 'testuser'
        password = 'wrong_password'
        ip_address = '192.168.1.100'
        
        # Generate enough failed attempts to lock account
        for _ in range(self.controller.max_failed_attempts):
            self.controller.authenticate_user(username, password, ip_address)
        
        # Account should be locked now
        result = self.controller.authenticate_user(username, 'testuser_password', ip_address, 'MFA_testuser_123456')
        
        self.assertFalse(result['authenticated'])
        self.assertEqual(result['code'], 'ACCOUNT_LOCKED')
        self.assertIn('lockout_expires', result)
        
        # Check account is in locked list
        self.assertIn(username, self.controller.locked_accounts)
    
    def test_validate_session_success(self):
        """Test successful session validation"""
        # First authenticate to get session
        username = 'testuser'
        auth_result = self.controller.authenticate_user(
            username, 'testuser_password', '192.168.1.100', 'MFA_testuser_123456'
        )
        
        session_id = auth_result['session_id']
        token = auth_result['token']
        
        # Validate session
        result = self.controller.validate_session(session_id, token)
        
        self.assertTrue(result['valid'])
        self.assertEqual(result['username'], username)
        self.assertIn('permissions', result)
    
    def test_validate_session_invalid_token(self):
        """Test session validation with invalid token"""
        session_id = 'fake_session_id'
        token = 'invalid_token'
        
        result = self.controller.validate_session(session_id, token)
        
        self.assertFalse(result['valid'])
        self.assertEqual(result['code'], 'INVALID_TOKEN')
    
    def test_validate_session_not_found(self):
        """Test session validation when session not found"""
        # Generate valid JWT but with non-existent session
        import jwt
        payload = {
            'username': 'testuser',
            'session_id': 'nonexistent_session',
            'exp': datetime.now(timezone.utc) + timedelta(hours=1),
            'iat': datetime.now(timezone.utc)
        }
        token = jwt.encode(payload, self.controller.jwt_secret, algorithm='HS256')
        
        result = self.controller.validate_session('nonexistent_session', token)
        
        self.assertFalse(result['valid'])
        self.assertEqual(result['code'], 'SESSION_NOT_FOUND')
    
    def test_validate_session_mismatch(self):
        """Test session validation with session ID mismatch"""
        # First authenticate to get session
        auth_result = self.controller.authenticate_user(
            'testuser', 'testuser_password', '192.168.1.100', 'MFA_testuser_123456'
        )
        
        session_id = auth_result['session_id']
        token = auth_result['token']
        
        # Try to validate with wrong session ID
        result = self.controller.validate_session('wrong_session_id', token)
        
        self.assertFalse(result['valid'])
        self.assertEqual(result['code'], 'SESSION_MISMATCH')
    
    def test_validate_session_expired(self):
        """Test session validation with expired session"""
        # Create session with short timeout for testing
        short_timeout_controller = AccessController({
            'session_timeout_minutes': 0,  # Immediate expiry
            'require_mfa': False
        })
        
        auth_result = short_timeout_controller.authenticate_user(
            'testuser', 'testuser_password', '192.168.1.100'
        )
        
        session_id = auth_result['session_id']
        token = auth_result['token']
        
        # Wait a moment to ensure expiry
        time.sleep(0.1)
        
        result = short_timeout_controller.validate_session(session_id, token)
        
        self.assertFalse(result['valid'])
        self.assertEqual(result['code'], 'SESSION_EXPIRED')
        
        # Check session was removed
        self.assertNotIn(session_id, short_timeout_controller.user_sessions)
    
    def test_check_permission_admin(self):
        """Test permission check for admin user"""
        username = 'admin_user'
        
        # Admin should have all permissions
        self.assertTrue(self.controller.check_permission(username, 'systems', 'delete'))
        self.assertTrue(self.controller.check_permission(username, 'users', 'create'))
        self.assertTrue(self.controller.check_permission(username, 'security', 'audit'))
        self.assertTrue(self.controller.check_permission(username, 'monitoring', 'read'))
    
    def test_check_permission_operator(self):
        """Test permission check for operator user"""
        username = 'operator_user'
        
        # Operator has limited permissions
        self.assertTrue(self.controller.check_permission(username, 'systems', 'read'))
        self.assertTrue(self.controller.check_permission(username, 'systems', 'update'))
        self.assertFalse(self.controller.check_permission(username, 'systems', 'delete'))
        self.assertTrue(self.controller.check_permission(username, 'monitoring', 'read'))
        self.assertFalse(self.controller.check_permission(username, 'security', 'audit'))
    
    def test_check_permission_regular_user(self):
        """Test permission check for regular user"""
        username = 'regular_user'
        
        # Regular user has minimal permissions
        self.assertTrue(self.controller.check_permission(username, 'systems', 'read'))
        self.assertFalse(self.controller.check_permission(username, 'systems', 'update'))
        self.assertFalse(self.controller.check_permission(username, 'systems', 'delete'))
        self.assertTrue(self.controller.check_permission(username, 'users', 'read_self'))
        self.assertFalse(self.controller.check_permission(username, 'users', 'read'))
        self.assertFalse(self.controller.check_permission(username, 'security', 'audit'))
    
    def test_check_permission_auditor(self):
        """Test permission check for auditor user"""
        username = 'auditor_user'
        
        # Auditor has read-only access
        self.assertTrue(self.controller.check_permission(username, 'systems', 'read'))
        self.assertFalse(self.controller.check_permission(username, 'systems', 'update'))
        self.assertTrue(self.controller.check_permission(username, 'security', 'read'))
        self.assertTrue(self.controller.check_permission(username, 'security', 'audit'))
        self.assertTrue(self.controller.check_permission(username, 'monitoring', 'read'))
    
    def test_revoke_session_success(self):
        """Test successful session revocation"""
        # First authenticate to get session
        auth_result = self.controller.authenticate_user(
            'testuser', 'testuser_password', '192.168.1.100', 'MFA_testuser_123456'
        )
        
        session_id = auth_result['session_id']
        
        # Check session exists
        self.assertIn(session_id, self.controller.user_sessions)
        
        # Revoke session
        result = self.controller.revoke_session(session_id)
        
        self.assertTrue(result)
        self.assertNotIn(session_id, self.controller.user_sessions)
    
    def test_revoke_session_not_found(self):
        """Test revoking non-existent session"""
        result = self.controller.revoke_session('nonexistent_session')
        self.assertFalse(result)
    
    def test_ip_restrictions_whitelist_only(self):
        """Test IP restrictions with whitelist only"""
        controller = AccessController({
            'ip_whitelist': ['192.168.1.100'],
            'require_mfa': False
        })
        
        # Should allow whitelisted IP
        self.assertTrue(controller._check_ip_restrictions('192.168.1.100'))
        
        # Should deny non-whitelisted IP
        self.assertFalse(controller._check_ip_restrictions('192.168.1.101'))
    
    def test_ip_restrictions_blacklist_only(self):
        """Test IP restrictions with blacklist only"""
        controller = AccessController({
            'ip_blacklist': ['192.168.1.200'],
            'require_mfa': False
        })
        
        # Should deny blacklisted IP
        self.assertFalse(controller._check_ip_restrictions('192.168.1.200'))
        
        # Should allow non-blacklisted IP
        self.assertTrue(controller._check_ip_restrictions('192.168.1.100'))
    
    def test_ip_restrictions_none(self):
        """Test IP restrictions with no whitelist or blacklist"""
        controller = AccessController({'require_mfa': False})
        
        # Should allow any IP
        self.assertTrue(controller._check_ip_restrictions('192.168.1.100'))
        self.assertTrue(controller._check_ip_restrictions('10.0.0.1'))
        self.assertTrue(controller._check_ip_restrictions('172.16.0.1'))
    
    def test_account_lockout_mechanism(self):
        """Test account lockout mechanism"""
        username = 'testuser'
        password = 'wrong_password'
        ip_address = '192.168.1.100'
        
        # Account should not be locked initially
        self.assertFalse(self.controller._is_account_locked(username))
        
        # Generate failed attempts
        for i in range(self.controller.max_failed_attempts - 1):
            self.controller.authenticate_user(username, password, ip_address)
            self.assertFalse(self.controller._is_account_locked(username))
        
        # One more should lock the account
        self.controller.authenticate_user(username, password, ip_address)
        self.assertTrue(self.controller._is_account_locked(username))
        
        # Check lockout details
        self.assertIn(username, self.controller.locked_accounts)
        lockout = self.controller.locked_accounts[username]
        self.assertIn('locked_at', lockout)
        self.assertIn('expires', lockout)
        self.assertIn('reason', lockout)
    
    def test_account_lockout_expiry(self):
        """Test account lockout expiry"""
        username = 'testuser'
        
        # Manually lock account with immediate expiry
        self.controller.locked_accounts[username] = {
            'locked_at': datetime.now(),
            'expires': datetime.now() - timedelta(minutes=1),  # Already expired
            'reason': 'Test lockout'
        }
        
        # Should be unlocked when checked
        self.assertFalse(self.controller._is_account_locked(username))
        
        # Should be removed from locked accounts
        self.assertNotIn(username, self.controller.locked_accounts)
    
    def test_password_verification_bcrypt(self):
        """Test password verification with bcrypt"""
        username = 'testuser'
        correct_password = f'{username}_password'
        wrong_password = 'wrong_password'
        
        # Should verify correct password
        self.assertTrue(self.controller._verify_credentials(username, correct_password))
        
        # Should reject wrong password
        self.assertFalse(self.controller._verify_credentials(username, wrong_password))
    
    def test_mfa_token_verification(self):
        """Test MFA token verification"""
        username = 'testuser'
        correct_token = f'MFA_{username}_123456'
        wrong_token = 'WRONG_TOKEN'
        
        # Should verify correct MFA token
        self.assertTrue(self.controller._verify_mfa_token(username, correct_token))
        
        # Should reject wrong MFA token
        self.assertFalse(self.controller._verify_mfa_token(username, wrong_token))
    
    def test_jwt_token_generation_and_validation(self):
        """Test JWT token generation and validation"""
        username = 'testuser'
        session_id = 'test_session_id'
        
        # Generate token
        token = self.controller._generate_jwt_token(username, session_id)
        self.assertIsNotNone(token)
        self.assertIn(token, self.controller.active_tokens)
        
        # Validate token manually
        import jwt
        try:
            payload = jwt.decode(token, self.controller.jwt_secret, algorithms=['HS256'])
            self.assertEqual(payload['username'], username)
            self.assertEqual(payload['session_id'], session_id)
            self.assertIn('exp', payload)
            self.assertIn('iat', payload)
        except jwt.InvalidTokenError:
            self.fail("Generated token should be valid")
    
    def test_user_role_assignment(self):
        """Test user role assignment logic"""
        # Admin user
        self.assertEqual(self.controller._get_user_role('admin_user'), 'admin')
        self.assertEqual(self.controller._get_user_role('admin123'), 'admin')
        
        # Operator user
        self.assertEqual(self.controller._get_user_role('operator_user'), 'operator')
        self.assertEqual(self.controller._get_user_role('operator123'), 'operator')
        
        # Regular user
        self.assertEqual(self.controller._get_user_role('regular_user'), 'user')
        self.assertEqual(self.controller._get_user_role('someuser'), 'user')
    
    def test_access_pattern_analysis(self):
        """Test access pattern analysis"""
        # Generate some test access attempts
        current_time = time.time()
        for i in range(10):
            self.controller.access_attempts.append({
                'username': f'user{i % 3}',
                'ip_address': f'192.168.1.{100 + i % 5}',
                'timestamp': current_time - (i * 60),  # Spread over time
                'success': i % 2 == 0  # Alternate success/failure
            })
        
        analysis = self.controller._analyze_access_patterns()
        
        self.assertIn('total_recent_attempts', analysis)
        self.assertIn('failed_recent_attempts', analysis)
        self.assertIn('failure_rate', analysis)
        self.assertIn('top_ips', analysis)
        self.assertIn('top_users', analysis)
        
        # Check failure rate calculation
        if analysis['total_recent_attempts'] > 0:
            expected_rate = analysis['failed_recent_attempts'] / analysis['total_recent_attempts']
            self.assertAlmostEqual(analysis['failure_rate'], expected_rate, places=2)
    
    def test_weakness_identification(self):
        """Test access control weakness identification"""
        # Test with permissive settings
        permissive_controller = AccessController({
            'require_mfa': False,
            'session_timeout_minutes': 240,  # 4 hours
            'max_failed_attempts': 15,
            'ip_whitelist': [],
            'ip_blacklist': []
        })
        
        weaknesses = permissive_controller._identify_access_weaknesses()
        
        # Should identify multiple weaknesses
        weakness_types = [w['type'] for w in weaknesses]
        self.assertIn('mfa_disabled', weakness_types)
        self.assertIn('long_session_timeout', weakness_types)
        self.assertIn('weak_lockout_policy', weakness_types)
        self.assertIn('no_ip_restrictions', weakness_types)
        
        # Check severity levels
        severities = [w['severity'] for w in weaknesses]
        self.assertIn('high', severities)  # MFA disabled should be high severity
    
    def test_suspicious_activity_detection(self):
        """Test suspicious activity detection"""
        # Generate suspicious access patterns
        current_time = time.time()
        
        # High failure rate from single IP
        suspicious_ip = '192.168.1.222'
        for i in range(10):
            self.controller.access_attempts.append({
                'username': f'user{i}',
                'ip_address': suspicious_ip,
                'timestamp': current_time - (i * 10),
                'success': False  # All failures
            })
        
        # High failure rate for single user
        suspicious_user = 'suspicious_user'
        for i in range(8):
            self.controller.access_attempts.append({
                'username': suspicious_user,
                'ip_address': f'192.168.1.{100 + i}',
                'timestamp': current_time - (i * 10),
                'success': False  # All failures
            })
        
        # Create long-lived session
        session_id = 'long_session'
        self.controller.user_sessions[session_id] = {
            'username': 'testuser',
            'created': datetime.now() - timedelta(hours=25),  # 25 hours ago
            'expires': datetime.now() + timedelta(hours=1)
        }
        
        suspicious_activities = self.controller._detect_suspicious_access()
        
        # Should detect suspicious activities
        activity_types = [a['type'] for a in suspicious_activities]
        self.assertIn('brute_force_attempt', activity_types)
        self.assertIn('credential_stuffing', activity_types)
        self.assertIn('long_lived_session', activity_types)
    
    def test_access_control_score_calculation(self):
        """Test access control score calculation"""
        # Test with no issues
        score = self.controller._calculate_access_control_score({}, [], [])
        self.assertEqual(score, 1.0)
        
        # Test with various issues
        weaknesses = [
            {'severity': 'critical'},
            {'severity': 'high'},
            {'severity': 'medium'},
            {'severity': 'low'}
        ]
        
        suspicious = [
            {'severity': 'critical'},
            {'severity': 'high'}
        ]
        
        analysis = {'failure_rate': 0.6}  # High failure rate
        
        score = self.controller._calculate_access_control_score(analysis, weaknesses, suspicious)
        
        # Score should be significantly reduced
        self.assertLess(score, 0.5)
        self.assertGreaterEqual(score, 0.0)
    
    def test_authentication_success_rate(self):
        """Test authentication success rate calculation"""
        # No attempts yet
        self.assertEqual(self.controller._calculate_authentication_success_rate(), 1.0)
        
        # Add some attempts
        self.controller.access_metrics['total_access_attempts'] = 10
        self.controller.access_metrics['successful_authentications'] = 7
        
        rate = self.controller._calculate_authentication_success_rate()
        self.assertEqual(rate, 0.7)
    
    def test_policy_compliance_check(self):
        """Test policy compliance checking"""
        compliance = self.controller._check_policy_compliance()
        
        self.assertIn('mfa_enabled', compliance)
        self.assertIn('session_timeout_compliant', compliance)
        self.assertIn('lockout_policy_compliant', compliance)
        self.assertIn('ip_restrictions_enabled', compliance)
        self.assertIn('rbac_implemented', compliance)
        
        # Check values based on config
        self.assertTrue(compliance['mfa_enabled'])
        self.assertFalse(compliance['session_timeout_compliant'])  # 60 min > 60 min limit
        self.assertTrue(compliance['lockout_policy_compliant'])  # 5 <= 5
        self.assertTrue(compliance['ip_restrictions_enabled'])  # Has whitelist
        self.assertTrue(compliance['rbac_implemented'])  # Has RBAC policies
    
    def test_rbac_policy_structure(self):
        """Test RBAC policy structure and completeness"""
        rbac_policies = self.controller.rbac_policies
        
        # Check all expected roles exist
        expected_roles = ['admin', 'operator', 'user', 'auditor']
        for role in expected_roles:
            self.assertIn(role, rbac_policies)
        
        # Check admin has full permissions
        admin_perms = rbac_policies['admin']
        self.assertEqual(admin_perms['systems'], ['*'])
        self.assertEqual(admin_perms['users'], ['*'])
        self.assertEqual(admin_perms['security'], ['*'])
        
        # Check user has limited permissions
        user_perms = rbac_policies['user']
        self.assertEqual(user_perms['systems'], ['read'])
        self.assertEqual(user_perms['security'], [])
        self.assertIn('read_self', user_perms['users'])
    
    def test_session_activity_tracking(self):
        """Test session activity tracking"""
        # Create session
        auth_result = self.controller.authenticate_user(
            'testuser', 'testuser_password', '192.168.1.100', 'MFA_testuser_123456'
        )
        
        session_id = auth_result['session_id']
        token = auth_result['token']
        
        # Get initial last activity time
        initial_activity = self.controller.user_sessions[session_id]['last_activity']
        
        # Wait a moment
        time.sleep(0.1)
        
        # Validate session (should update last activity)
        self.controller.validate_session(session_id, token)
        
        # Check last activity was updated
        updated_activity = self.controller.user_sessions[session_id]['last_activity']
        self.assertGreater(updated_activity, initial_activity)
    
    def test_concurrent_authentication_attempts(self):
        """Test handling of concurrent authentication attempts"""
        username = 'testuser'
        password = 'testuser_password'
        ip_address = '192.168.1.100'
        mfa_token = 'MFA_testuser_123456'
        
        # Multiple rapid authentication attempts
        results = []
        for _ in range(5):
            result = self.controller.authenticate_user(username, password, ip_address, mfa_token)
            results.append(result)
        
        # All should succeed
        for result in results:
            self.assertTrue(result['authenticated'])
        
        # Should have 5 sessions
        self.assertEqual(len(self.controller.user_sessions), 5)
        self.assertEqual(self.controller.access_metrics['session_creations'], 5)


class TestAccessControllerEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.controller = AccessController({
            'require_mfa': False,  # Disable MFA for simpler testing
            'max_failed_attempts': 3
        })
    
    def test_authenticate_with_none_values(self):
        """Test authentication with None values"""
        # Test with None username
        result = self.controller.authenticate_user(None, 'password', '192.168.1.100')
        self.assertFalse(result['authenticated'])
        
        # Test with None password
        result = self.controller.authenticate_user('testuser', None, '192.168.1.100')
        self.assertFalse(result['authenticated'])
        
        # Test with None IP
        result = self.controller.authenticate_user('testuser', 'password', None)
        self.assertFalse(result['authenticated'])
    
    def test_authenticate_with_empty_strings(self):
        """Test authentication with empty strings"""
        result = self.controller.authenticate_user('', '', '')
        self.assertFalse(result['authenticated'])
    
    def test_validate_session_with_none_values(self):
        """Test session validation with None values"""
        result = self.controller.validate_session(None, None)
        self.assertFalse(result['valid'])
        
        result = self.controller.validate_session('session', None)
        self.assertFalse(result['valid'])
        
        result = self.controller.validate_session(None, 'token')
        self.assertFalse(result['valid'])
    
    def test_check_permission_with_none_values(self):
        """Test permission check with None values"""
        result = self.controller.check_permission(None, 'resource', 'action')
        self.assertFalse(result)
        
        result = self.controller.check_permission('user', None, 'action')
        self.assertFalse(result)
        
        result = self.controller.check_permission('user', 'resource', None)
        self.assertFalse(result)
    
    def test_revoke_session_with_none(self):
        """Test session revocation with None"""
        result = self.controller.revoke_session(None)
        self.assertFalse(result)
    
    def test_massive_failed_attempts(self):
        """Test handling of massive number of failed attempts"""
        username = 'testuser'
        
        # Generate 1000 failed attempts
        for i in range(1000):
            self.controller._record_failed_attempt(username, f'192.168.1.{i % 256}')
        
        # Should still be manageable
        self.assertTrue(self.controller._is_account_locked(username))
        
        # Failed attempts should be cleaned up (only recent ones kept)
        recent_attempts = [
            a for a in self.controller.failed_attempts[username]
            if time.time() - a['timestamp'] < 3600
        ]
        self.assertLessEqual(len(recent_attempts), 1000)
    
    def test_session_cleanup_on_expiry_check(self):
        """Test that expired sessions are cleaned up during validation"""
        # Create session with immediate expiry
        session_id = 'test_session'
        self.controller.user_sessions[session_id] = {
            'username': 'testuser',
            'created': datetime.now() - timedelta(hours=2),
            'expires': datetime.now() - timedelta(hours=1),  # Expired
            'last_activity': datetime.now() - timedelta(hours=1)
        }
        
        # Create fake JWT token for the expired session
        import jwt
        payload = {
            'username': 'testuser',
            'session_id': session_id,
            'exp': datetime.now(timezone.utc) + timedelta(hours=1),
            'iat': datetime.now(timezone.utc)
        }
        token = jwt.encode(payload, self.controller.jwt_secret, algorithm='HS256')
        
        # Validate expired session
        result = self.controller.validate_session(session_id, token)
        
        # Should fail validation
        self.assertFalse(result['valid'])
        self.assertEqual(result['code'], 'SESSION_EXPIRED')
        
        # Session should be cleaned up
        self.assertNotIn(session_id, self.controller.user_sessions)
    
    def test_malformed_jwt_token(self):
        """Test handling of malformed JWT tokens"""
        session_id = 'test_session'
        malformed_tokens = [
            'not.a.jwt',
            'malformed_token',
            '',
            'a.b.c.d.e',  # Too many parts
            'header.payload.signature.extra'
        ]
        
        for token in malformed_tokens:
            result = self.controller.validate_session(session_id, token)
            self.assertFalse(result['valid'])
            self.assertEqual(result['code'], 'INVALID_TOKEN')
    
    def test_audit_with_exception_handling(self):
        """Test audit function handles exceptions gracefully"""
        # Force an exception by corrupting internal state
        original_analyze = self.controller._analyze_access_patterns
        self.controller._analyze_access_patterns = lambda: 1/0  # Causes ZeroDivisionError
        
        audit_result = self.controller.audit_access_controls()
        
        # Should return error state
        self.assertEqual(audit_result['access_control_score'], 0.0)
        self.assertIn('error', audit_result)
        self.assertIn('weaknesses', audit_result)
        
        # Restore original method
        self.controller._analyze_access_patterns = original_analyze


class TestAccessControllerIntegration(unittest.TestCase):
    """Integration tests for AccessController"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.controller = AccessController({
            'max_failed_attempts': 3,
            'lockout_duration_minutes': 1,  # Short lockout for testing
            'session_timeout_minutes': 30,
            'require_mfa': True,
            'ip_whitelist': ['192.168.1.0/24', '10.0.0.0/8'],
            'ip_blacklist': ['192.168.1.99']
        })
    
    def test_full_authentication_flow(self):
        """Test complete authentication flow"""
        username = 'testuser'
        password = 'testuser_password'
        ip_address = '192.168.1.100'
        mfa_token = 'MFA_testuser_123456'
        
        # Step 1: First attempt without MFA should request MFA
        result1 = self.controller.authenticate_user(username, password, ip_address)
        self.assertFalse(result1['authenticated'])
        self.assertTrue(result1.get('mfa_required', False))
        
        # Step 2: Authenticate with MFA should succeed
        result2 = self.controller.authenticate_user(username, password, ip_address, mfa_token)
        self.assertTrue(result2['authenticated'])
        
        session_id = result2['session_id']
        token = result2['token']
        
        # Step 3: Validate session should work
        validation = self.controller.validate_session(session_id, token)
        self.assertTrue(validation['valid'])
        
        # Step 4: Check permissions
        can_read = self.controller.check_permission(username, 'systems', 'read')
        self.assertTrue(can_read)
        
        can_delete = self.controller.check_permission(username, 'systems', 'delete')
        self.assertFalse(can_delete)  # Regular user can't delete
        
        # Step 5: Revoke session
        revoked = self.controller.revoke_session(session_id)
        self.assertTrue(revoked)
        
        # Step 6: Validation should now fail
        validation2 = self.controller.validate_session(session_id, token)
        self.assertFalse(validation2['valid'])
    
    def test_brute_force_protection_flow(self):
        """Test brute force protection flow"""
        username = 'attackuser'
        wrong_password = 'wrong_password'
        ip_address = '192.168.1.100'
        
        # Generate failed attempts
        for i in range(self.controller.max_failed_attempts):
            result = self.controller.authenticate_user(username, wrong_password, ip_address)
            self.assertFalse(result['authenticated'])
            self.assertEqual(result['code'], 'INVALID_CREDENTIALS')
        
        # Account should be locked now
        result = self.controller.authenticate_user(username, 'testuser_password', ip_address, 'MFA_testuser_123456')
        self.assertFalse(result['authenticated'])
        self.assertEqual(result['code'], 'ACCOUNT_LOCKED')
        
        # Run audit - should detect suspicious activity
        audit = self.controller.audit_access_controls()
        self.assertTrue(len(audit.get('suspicious_activities', [])) > 0)
        
        # Score should be reduced
        self.assertLess(audit['access_control_score'], 1.0)
    
    def test_multi_user_session_management(self):
        """Test session management with multiple users"""
        users = ['user1', 'user2', 'admin_user', 'operator_user']
        sessions = {}
        
        # Authenticate multiple users
        for username in users:
            result = self.controller.authenticate_user(
                username, f'{username}_password', '192.168.1.100', f'MFA_{username}_123456'
            )
            self.assertTrue(result['authenticated'])
            sessions[username] = {
                'session_id': result['session_id'],
                'token': result['token']
            }
        
        # All sessions should be active
        self.assertEqual(len(self.controller.user_sessions), 4)
        
        # Validate all sessions
        for username, session_data in sessions.items():
            validation = self.controller.validate_session(
                session_data['session_id'], session_data['token']
            )
            self.assertTrue(validation['valid'])
            self.assertEqual(validation['username'], username)
        
        # Check different permissions for different users
        self.assertTrue(self.controller.check_permission('admin_user', 'systems', 'delete'))
        self.assertFalse(self.controller.check_permission('user1', 'systems', 'delete'))
        self.assertTrue(self.controller.check_permission('operator_user', 'systems', 'update'))
        
        # Revoke one session
        revoked = self.controller.revoke_session(sessions['user1']['session_id'])
        self.assertTrue(revoked)
        
        # Other sessions should still work
        self.assertEqual(len(self.controller.user_sessions), 3)
        
        validation = self.controller.validate_session(
            sessions['user2']['session_id'], sessions['user2']['token']
        )
        self.assertTrue(validation['valid'])


if __name__ == '__main__':
    unittest.main(verbosity=2)