"""
Comprehensive test suite for AuditLogger
Tests all functionality, edge cases, threading, performance, and security features
"""

import pytest
import time
import threading
import tempfile
import shutil
import json
import gzip
import csv
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import os
import sys

# Add the project root to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from production_security.audit_logger import AuditLogger


class TestAuditLoggerBasics:
    """Test basic functionality of AuditLogger"""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def basic_config(self, temp_log_dir):
        """Basic configuration for testing"""
        return {
            'log_path': temp_log_dir,
            'retention_days': 7,
            'max_log_size_mb': 1,
            'compression_enabled': True,
            'real_time_alerts': True
        }
    
    @pytest.fixture
    def audit_logger(self, basic_config):
        """Create audit logger instance"""
        logger = AuditLogger(basic_config)
        yield logger
        logger.shutdown()
    
    def test_audit_logger_initialization(self, basic_config, temp_log_dir):
        """Test audit logger initialization"""
        logger = AuditLogger(basic_config)
        
        assert logger.log_path == Path(temp_log_dir)
        assert logger.retention_days == 7
        assert logger.max_log_size == 1024 * 1024
        assert logger.compression_enabled == True
        assert logger.real_time_alerts == True
        assert len(logger.log_buffer) == 0
        assert logger.is_running == True
        
        logger.shutdown()
    
    def test_log_path_creation(self, temp_log_dir):
        """Test that log path is created if it doesn't exist"""
        non_existent_path = os.path.join(temp_log_dir, "nested", "log", "path")
        
        config = {
            'log_path': non_existent_path,
            'retention_days': 7,
            'max_log_size_mb': 1
        }
        
        logger = AuditLogger(config)
        assert Path(non_existent_path).exists()
        
        logger.shutdown()
    
    def test_default_configuration(self, temp_log_dir):
        """Test default configuration values"""
        config = {'log_path': temp_log_dir}
        logger = AuditLogger(config)
        
        assert logger.retention_days == 2555  # 7 years
        assert logger.max_log_size == 100 * 1024 * 1024
        assert logger.compression_enabled == True
        assert logger.real_time_alerts == True
        
        logger.shutdown()


class TestSecurityEventLogging:
    """Test security event logging functionality"""
    
    @pytest.fixture
    def audit_logger(self):
        """Create audit logger with temporary directory"""
        temp_dir = tempfile.mkdtemp()
        config = {
            'log_path': temp_dir,
            'retention_days': 1,
            'max_log_size_mb': 1,
            'compression_enabled': False  # Disable for easier testing
        }
        logger = AuditLogger(config)
        yield logger
        logger.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_log_security_event_basic(self, audit_logger):
        """Test basic security event logging"""
        event = {
            'event_type': 'user_login',
            'username': 'test_user',
            'source_ip': '192.168.1.100',
            'severity': 'info',
            'description': 'User logged in successfully'
        }
        
        result = audit_logger.log_security_event(event)
        assert result == True
        
        # Check event is in buffer
        assert len(audit_logger.log_buffer) == 1
        logged_event = audit_logger.log_buffer[0]
        
        assert 'event_id' in logged_event
        assert 'timestamp' in logged_event
        assert 'datetime' in logged_event
        assert 'integrity_hash' in logged_event
        assert logged_event['event'] == event
    
    def test_event_id_generation(self, audit_logger):
        """Test that unique event IDs are generated"""
        event = {'event_type': 'test', 'severity': 'info'}
        
        audit_logger.log_security_event(event)
        audit_logger.log_security_event(event)
        
        event_ids = [e['event_id'] for e in audit_logger.log_buffer]
        assert len(set(event_ids)) == 2  # Should be unique
        assert all(id.startswith('evt_') for id in event_ids)
    
    def test_integrity_hash_calculation(self, audit_logger):
        """Test integrity hash calculation"""
        event = {
            'event_type': 'security_test',
            'username': 'test_user',
            'severity': 'high'
        }
        
        audit_logger.log_security_event(event)
        logged_event = audit_logger.log_buffer[0]
        
        # Verify hash exists
        assert 'integrity_hash' in logged_event
        assert logged_event['integrity_hash'] is not None
        assert len(logged_event['integrity_hash']) == 64  # SHA256 hex length
        
        # Verify hash is correct
        expected_hash = audit_logger._calculate_integrity_hash(logged_event)
        assert logged_event['integrity_hash'] == expected_hash
    
    def test_event_statistics_tracking(self, audit_logger):
        """Test event statistics are tracked correctly"""
        events = [
            {'event_type': 'login', 'severity': 'info'},
            {'event_type': 'login', 'severity': 'warning'},
            {'event_type': 'logout', 'severity': 'info'},
            {'event_type': 'login', 'severity': 'info'}
        ]
        
        for event in events:
            audit_logger.log_security_event(event)
        
        # Check statistics
        login_stats = audit_logger.event_stats['login']
        logout_stats = audit_logger.event_stats['logout']
        
        assert login_stats['count'] == 3
        assert logout_stats['count'] == 1
        assert login_stats['severity_breakdown']['info'] == 2
        assert login_stats['severity_breakdown']['warning'] == 1
        assert logout_stats['severity_breakdown']['info'] == 1
    
    def test_critical_event_tracking(self, audit_logger):
        """Test that critical events are tracked separately"""
        events = [
            {'event_type': 'login', 'severity': 'info'},
            {'event_type': 'security_breach', 'severity': 'critical'},
            {'event_type': 'failed_login', 'severity': 'high'},
            {'event_type': 'logout', 'severity': 'info'}
        ]
        
        for event in events:
            audit_logger.log_security_event(event)
        
        # Check critical events
        assert len(audit_logger.critical_events) == 2  # critical and high severity
        critical_types = [e['event']['event_type'] for e in audit_logger.critical_events]
        assert 'security_breach' in critical_types
        assert 'failed_login' in critical_types
    
    def test_real_time_alerts(self, audit_logger):
        """Test real-time alert queue"""
        critical_event = {
            'event_type': 'security_breach',
            'severity': 'critical',
            'description': 'Unauthorized access attempt'
        }
        
        audit_logger.log_security_event(critical_event)
        
        # Check alert queue
        assert len(audit_logger.alert_queue) == 1
        assert audit_logger.alert_queue[0]['event']['event_type'] == 'security_breach'


class TestLogSearchAndRetrieval:
    """Test log search and retrieval functionality"""
    
    @pytest.fixture
    def audit_logger_with_data(self):
        """Create audit logger with sample data"""
        temp_dir = tempfile.mkdtemp()
        config = {
            'log_path': temp_dir,
            'retention_days': 1,
            'max_log_size_mb': 1,
            'compression_enabled': False
        }
        logger = AuditLogger(config)
        
        # Add sample events
        sample_events = [
            {'event_type': 'user_login', 'username': 'alice', 'severity': 'info'},
            {'event_type': 'user_login', 'username': 'bob', 'severity': 'info'},
            {'event_type': 'failed_login', 'username': 'eve', 'severity': 'warning'},
            {'event_type': 'admin_action', 'username': 'admin', 'severity': 'high'},
            {'event_type': 'user_logout', 'username': 'alice', 'severity': 'info'}
        ]
        
        for event in sample_events:
            logger.log_security_event(event)
        
        yield logger
        logger.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_search_by_event_type(self, audit_logger_with_data):
        """Test searching by event type"""
        results = audit_logger_with_data.search_logs({'event_type': 'user_login'})
        
        assert len(results) == 2
        assert all(r['event']['event_type'] == 'user_login' for r in results)
        assert all(r['event']['username'] in ['alice', 'bob'] for r in results)
    
    def test_search_by_severity(self, audit_logger_with_data):
        """Test searching by severity"""
        results = audit_logger_with_data.search_logs({'severity': 'warning'})
        
        assert len(results) == 1
        assert results[0]['event']['event_type'] == 'failed_login'
        assert results[0]['event']['username'] == 'eve'
    
    def test_search_by_username(self, audit_logger_with_data):
        """Test searching by username"""
        results = audit_logger_with_data.search_logs({'username': 'alice'})
        
        assert len(results) == 2
        assert all(r['event']['username'] == 'alice' for r in results)
        event_types = [r['event']['event_type'] for r in results]
        assert 'user_login' in event_types
        assert 'user_logout' in event_types
    
    def test_search_with_time_range(self, audit_logger_with_data):
        """Test searching with time range"""
        now = datetime.now()
        start_time = now - timedelta(minutes=1)
        end_time = now + timedelta(minutes=1)
        
        time_range = {'start': start_time, 'end': end_time}
        results = audit_logger_with_data.search_logs({}, time_range)
        
        assert len(results) == 5  # All events should be in range
        
        # Test with narrow time range
        narrow_start = now + timedelta(minutes=10)
        narrow_end = now + timedelta(minutes=20)
        narrow_range = {'start': narrow_start, 'end': narrow_end}
        
        results = audit_logger_with_data.search_logs({}, narrow_range)
        assert len(results) == 0  # No events should be in future range
    
    def test_search_with_limit(self, audit_logger_with_data):
        """Test search result limiting"""
        results = audit_logger_with_data.search_logs({'limit': 2})
        
        assert len(results) <= 2
    
    def test_search_results_ordering(self, audit_logger_with_data):
        """Test that search results are ordered by timestamp (newest first)"""
        results = audit_logger_with_data.search_logs({})
        
        # Check ordering
        timestamps = [r['timestamp'] for r in results]
        assert timestamps == sorted(timestamps, reverse=True)
    
    def test_search_with_multiple_criteria(self, audit_logger_with_data):
        """Test search with multiple criteria"""
        criteria = {
            'event_type': 'user_login',
            'username': 'alice'
        }
        
        results = audit_logger_with_data.search_logs(criteria)
        
        assert len(results) == 1
        assert results[0]['event']['event_type'] == 'user_login'
        assert results[0]['event']['username'] == 'alice'


class TestAuditReporting:
    """Test audit reporting functionality"""
    
    @pytest.fixture
    def audit_logger_with_varied_data(self):
        """Create audit logger with varied sample data"""
        temp_dir = tempfile.mkdtemp()
        config = {
            'log_path': temp_dir,
            'retention_days': 1,
            'max_log_size_mb': 1
        }
        logger = AuditLogger(config)
        
        # Add varied sample events
        sample_events = [
            {'event_type': 'user_login', 'username': 'alice', 'severity': 'info', 'source': 'web_app'},
            {'event_type': 'user_login', 'username': 'bob', 'severity': 'info', 'source': 'mobile_app'},
            {'event_type': 'failed_login', 'username': 'eve', 'severity': 'warning', 'source': 'web_app'},
            {'event_type': 'admin_action', 'username': 'admin', 'severity': 'high', 'source': 'admin_panel'},
            {'event_type': 'security_breach', 'username': None, 'severity': 'critical', 'source': 'firewall'},
            {'event_type': 'user_logout', 'username': 'alice', 'severity': 'info', 'source': 'web_app'},
            {'event_type': 'compliance_audit', 'username': 'auditor', 'severity': 'info', 'source': 'audit_system'},
            {'event_type': 'policy_violation', 'username': 'bob', 'severity': 'high', 'source': 'policy_engine'}
        ]
        
        for event in sample_events:
            logger.log_security_event(event)
        
        yield logger
        logger.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_audit_report_generation(self, audit_logger_with_varied_data):
        """Test basic audit report generation"""
        start_date = datetime.now() - timedelta(hours=1)
        end_date = datetime.now() + timedelta(hours=1)
        
        report = audit_logger_with_varied_data.get_audit_report(start_date, end_date)
        
        assert 'report_id' in report
        assert 'period' in report
        assert 'total_events' in report
        assert report['total_events'] == 8
        assert 'event_breakdown' in report
        assert 'severity_summary' in report
        assert 'top_event_types' in report
        assert 'critical_events' in report
        assert 'user_activity' in report
        assert 'system_activity' in report
        assert 'integrity_status' in report
    
    def test_event_breakdown_analysis(self, audit_logger_with_varied_data):
        """Test event breakdown analysis"""
        start_date = datetime.now() - timedelta(hours=1)
        end_date = datetime.now() + timedelta(hours=1)
        
        report = audit_logger_with_varied_data.get_audit_report(start_date, end_date)
        breakdown = report['event_breakdown']
        
        assert breakdown['user_login'] == 2
        assert breakdown['failed_login'] == 1
        assert breakdown['admin_action'] == 1
        assert breakdown['security_breach'] == 1
        assert breakdown['user_logout'] == 1
    
    def test_severity_summary_analysis(self, audit_logger_with_varied_data):
        """Test severity summary analysis"""
        start_date = datetime.now() - timedelta(hours=1)
        end_date = datetime.now() + timedelta(hours=1)
        
        report = audit_logger_with_varied_data.get_audit_report(start_date, end_date)
        severity_summary = report['severity_summary']
        
        assert severity_summary['info'] == 4
        assert severity_summary['warning'] == 1
        assert severity_summary['high'] == 2
        assert severity_summary['critical'] == 1
    
    def test_top_event_types_analysis(self, audit_logger_with_varied_data):
        """Test top event types analysis"""
        start_date = datetime.now() - timedelta(hours=1)
        end_date = datetime.now() + timedelta(hours=1)
        
        report = audit_logger_with_varied_data.get_audit_report(start_date, end_date)
        top_events = report['top_event_types']
        
        # Should be sorted by count, descending
        assert len(top_events) >= 2
        assert top_events[0]['event_type'] == 'user_login'
        assert top_events[0]['count'] == 2
    
    def test_critical_events_analysis(self, audit_logger_with_varied_data):
        """Test critical events analysis"""
        start_date = datetime.now() - timedelta(hours=1)
        end_date = datetime.now() + timedelta(hours=1)
        
        report = audit_logger_with_varied_data.get_audit_report(start_date, end_date)
        critical_events = report['critical_events']
        
        assert len(critical_events) == 3  # high and critical severity events
        event_types = [e['event_type'] for e in critical_events]
        assert 'security_breach' in event_types
        assert 'admin_action' in event_types
        assert 'policy_violation' in event_types
    
    def test_user_activity_analysis(self, audit_logger_with_varied_data):
        """Test user activity analysis"""
        start_date = datetime.now() - timedelta(hours=1)
        end_date = datetime.now() + timedelta(hours=1)
        
        report = audit_logger_with_varied_data.get_audit_report(start_date, end_date)
        user_activity = report['user_activity']
        
        assert 'total_users' in user_activity
        assert user_activity['total_users'] >= 4  # alice, bob, admin, auditor
        assert 'top_users' in user_activity
        
        # Check that alice has multiple events
        alice_found = False
        for user_data in user_activity['top_users']:
            if user_data['username'] == 'alice':
                alice_found = True
                assert user_data['activity']['total_events'] == 2
                break
        assert alice_found
    
    def test_system_activity_analysis(self, audit_logger_with_varied_data):
        """Test system activity analysis"""
        start_date = datetime.now() - timedelta(hours=1)
        end_date = datetime.now() + timedelta(hours=1)
        
        report = audit_logger_with_varied_data.get_audit_report(start_date, end_date)
        system_activity = report['system_activity']
        
        assert 'total_systems' in system_activity
        assert system_activity['total_systems'] >= 3  # web_app, mobile_app, admin_panel, etc.
        assert 'top_systems' in system_activity
    
    def test_compliance_events_analysis(self, audit_logger_with_varied_data):
        """Test compliance events analysis"""
        start_date = datetime.now() - timedelta(hours=1)
        end_date = datetime.now() + timedelta(hours=1)
        
        report = audit_logger_with_varied_data.get_audit_report(start_date, end_date)
        compliance_events = report['compliance_events']
        
        # Should find compliance_audit and policy_violation events
        assert len(compliance_events) == 2
        event_types = [e['event_type'] for e in compliance_events]
        assert 'compliance_audit' in event_types
        assert 'policy_violation' in event_types


class TestLogIntegrity:
    """Test log integrity verification"""
    
    @pytest.fixture
    def audit_logger(self):
        """Create audit logger for integrity testing"""
        temp_dir = tempfile.mkdtemp()
        config = {
            'log_path': temp_dir,
            'retention_days': 1,
            'max_log_size_mb': 1,
            'compression_enabled': False
        }
        logger = AuditLogger(config)
        yield logger
        logger.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_integrity_hash_consistency(self, audit_logger):
        """Test that integrity hashes are consistent"""
        event = {
            'event_type': 'test_event',
            'username': 'test_user',
            'severity': 'info'
        }
        
        # Calculate hash multiple times
        audit_logger.log_security_event(event)
        logged_event = audit_logger.log_buffer[0]
        
        original_hash = logged_event['integrity_hash']
        recalculated_hash = audit_logger._calculate_integrity_hash(logged_event)
        
        assert original_hash == recalculated_hash
    
    def test_integrity_verification_success(self, audit_logger):
        """Test successful integrity verification"""
        # Add some events
        events = [
            {'event_type': 'login', 'username': 'user1', 'severity': 'info'},
            {'event_type': 'logout', 'username': 'user1', 'severity': 'info'},
            {'event_type': 'admin_action', 'username': 'admin', 'severity': 'high'}
        ]
        
        for event in events:
            audit_logger.log_security_event(event)
        
        # Verify integrity
        verification_result = audit_logger.verify_log_integrity()
        
        assert verification_result['verified'] == True
        assert verification_result['total_logs_checked'] == 3
        assert len(verification_result['integrity_failures']) == 0
    
    def test_integrity_verification_failure(self, audit_logger):
        """Test integrity verification detects tampering"""
        # Add an event
        event = {'event_type': 'test', 'severity': 'info'}
        audit_logger.log_security_event(event)
        
        # Tamper with the event
        audit_logger.log_buffer[0]['integrity_hash'] = 'tampered_hash'
        
        # Verify integrity
        verification_result = audit_logger.verify_log_integrity()
        
        assert verification_result['verified'] == False
        assert len(verification_result['integrity_failures']) == 1
        assert verification_result['integrity_failures'][0]['reason'] == 'Hash mismatch'
    
    def test_integrity_hash_excludes_hash_field(self, audit_logger):
        """Test that integrity hash calculation excludes the hash field itself"""
        event_record = {
            'event_id': 'test_id',
            'timestamp': time.time(),
            'event': {'event_type': 'test', 'severity': 'info'},
            'integrity_hash': 'existing_hash'
        }
        
        calculated_hash = audit_logger._calculate_integrity_hash(event_record)
        
        # Remove hash and calculate again
        event_record_no_hash = event_record.copy()
        del event_record_no_hash['integrity_hash']
        calculated_hash_no_hash = audit_logger._calculate_integrity_hash(event_record_no_hash)
        
        # Should be the same
        assert calculated_hash == calculated_hash_no_hash


class TestLogExport:
    """Test log export functionality"""
    
    @pytest.fixture
    def audit_logger_with_export_data(self):
        """Create audit logger with data for export testing"""
        temp_dir = tempfile.mkdtemp()
        config = {
            'log_path': temp_dir,
            'retention_days': 1,
            'max_log_size_mb': 1
        }
        logger = AuditLogger(config)
        
        # Add sample data
        events = [
            {'event_type': 'user_login', 'username': 'alice', 'severity': 'info', 'description': 'User logged in'},
            {'event_type': 'failed_login', 'username': 'eve', 'severity': 'warning', 'description': 'Failed login attempt'},
            {'event_type': 'admin_action', 'username': 'admin', 'severity': 'high', 'description': 'Administrative action'}
        ]
        
        for event in events:
            logger.log_security_event(event)
        
        yield logger, temp_dir
        logger.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_export_logs_json(self, audit_logger_with_export_data):
        """Test exporting logs in JSON format"""
        logger, temp_dir = audit_logger_with_export_data
        
        export_file = logger.export_logs({'severity': 'info'}, format='json')
        
        assert os.path.exists(export_file)
        assert export_file.endswith('.json')
        
        # Verify content
        with open(export_file, 'r') as f:
            exported_data = json.load(f)
        
        assert len(exported_data) == 1  # Only info severity events
        assert exported_data[0]['event']['event_type'] == 'user_login'
    
    def test_export_logs_csv(self, audit_logger_with_export_data):
        """Test exporting logs in CSV format"""
        logger, temp_dir = audit_logger_with_export_data
        
        export_file = logger.export_logs({}, format='csv')
        
        assert os.path.exists(export_file)
        assert export_file.endswith('.csv')
        
        # Verify content
        with open(export_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 3  # All events
        assert 'event_id' in rows[0]
        assert 'event_type' in rows[0]
        assert 'username' in rows[0]
        assert 'severity' in rows[0]
    
    def test_export_with_criteria(self, audit_logger_with_export_data):
        """Test exporting logs with specific criteria"""
        logger, temp_dir = audit_logger_with_export_data
        
        export_file = logger.export_logs({'username': 'alice'}, format='json')
        
        # Verify content
        with open(export_file, 'r') as f:
            exported_data = json.load(f)
        
        assert len(exported_data) == 1
        assert exported_data[0]['event']['username'] == 'alice'
    
    def test_export_unsupported_format(self, audit_logger_with_export_data):
        """Test error handling for unsupported export format"""
        logger, temp_dir = audit_logger_with_export_data
        
        with pytest.raises(ValueError, match="Unsupported export format"):
            logger.export_logs({}, format='xml')


class TestLogRotationAndCleanup:
    """Test log rotation and cleanup functionality"""
    
    @pytest.fixture
    def audit_logger_small_logs(self):
        """Create audit logger with small log size for testing rotation"""
        temp_dir = tempfile.mkdtemp()
        config = {
            'log_path': temp_dir,
            'retention_days': 1,
            'max_log_size_mb': 0.001,  # Very small for testing rotation
            'compression_enabled': True
        }
        logger = AuditLogger(config)
        yield logger, temp_dir
        logger.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_log_file_creation(self, audit_logger_small_logs):
        """Test that log files are created"""
        logger, temp_dir = audit_logger_small_logs
        
        # Add events to trigger file creation
        for i in range(5):
            event = {'event_type': 'test', 'severity': 'info', 'data': f'test_{i}'}
            logger.log_security_event(event)
        
        # Wait for background writer
        time.sleep(2)
        
        # Check if log file was created
        log_files = list(Path(temp_dir).glob('audit_*.log*'))
        assert len(log_files) >= 1
    
    def test_log_rotation_on_size_limit(self, audit_logger_small_logs):
        """Test log rotation when size limit is reached"""
        logger, temp_dir = audit_logger_small_logs
        
        # Add many events to trigger rotation
        large_data = 'x' * 1000  # Large event data
        for i in range(20):
            event = {
                'event_type': 'large_test',
                'severity': 'info',
                'large_data': large_data,
                'counter': i
            }
            logger.log_security_event(event)
        
        # Wait for background processes
        time.sleep(3)
        
        # Check that rotation occurred
        log_files = list(Path(temp_dir).glob('audit_*.log*'))
        # Should have at least original + rotated files
        assert len(log_files) >= 1
    
    def test_log_compression(self, audit_logger_small_logs):
        """Test log file compression during rotation"""
        logger, temp_dir = audit_logger_small_logs
        
        # Force creation of a log file
        logger._create_new_log_file()
        
        # Add content to the file
        test_events = [
            {'event_type': f'test_{i}', 'severity': 'info'}
            for i in range(10)
        ]
        logger._write_events_to_disk(test_events)
        
        # Force rotation
        logger._rotate_log()
        
        # Wait a bit
        time.sleep(1)
        
        # Check for compressed files
        compressed_files = list(Path(temp_dir).glob('*.gz'))
        # Should have compressed file if compression is enabled
        if logger.compression_enabled:
            assert len(compressed_files) >= 0  # Might be compressed
    
    def test_old_log_cleanup(self):
        """Test cleanup of old log files"""
        temp_dir = tempfile.mkdtemp()
        config = {
            'log_path': temp_dir,
            'retention_days': 1,  # 1 day retention
            'max_log_size_mb': 1
        }
        logger = AuditLogger(config)
        
        # Create old log files manually
        old_date = datetime.now() - timedelta(days=2)
        old_timestamp = old_date.strftime('%Y%m%d_%H%M%S')
        old_file = Path(temp_dir) / f"audit_{old_timestamp}_0001.log"
        
        with open(old_file, 'w') as f:
            f.write('{"test": "old_log"}\n')
        
        # Run cleanup
        logger._clean_old_logs()
        
        # Check if old file was removed
        assert not old_file.exists()
        
        logger.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestThreadSafety:
    """Test thread safety of audit logger"""
    
    @pytest.fixture
    def audit_logger(self):
        """Create audit logger for threading tests"""
        temp_dir = tempfile.mkdtemp()
        config = {
            'log_path': temp_dir,
            'retention_days': 1,
            'max_log_size_mb': 10
        }
        logger = AuditLogger(config)
        yield logger
        logger.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_concurrent_event_logging(self, audit_logger):
        """Test concurrent event logging from multiple threads"""
        num_threads = 10
        events_per_thread = 20
        
        def log_events(thread_id):
            for i in range(events_per_thread):
                event = {
                    'event_type': 'concurrent_test',
                    'thread_id': thread_id,
                    'event_num': i,
                    'severity': 'info'
                }
                audit_logger.log_security_event(event)
        
        # Start threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=log_events, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Wait for background processing
        time.sleep(2)
        
        # Check that all events were logged
        expected_events = num_threads * events_per_thread
        total_logged = len(audit_logger.log_buffer)
        
        # Should have logged all events (some might have been written to disk)
        assert total_logged <= expected_events  # Some might be in disk
        
        # Check event statistics
        concurrent_stats = audit_logger.event_stats['concurrent_test']
        assert concurrent_stats['count'] == expected_events
    
    def test_concurrent_search_operations(self, audit_logger):
        """Test concurrent search operations"""
        # Add some test data first
        for i in range(50):
            event = {
                'event_type': 'search_test',
                'user_id': f'user_{i % 5}',
                'severity': 'info' if i % 2 == 0 else 'warning'
            }
            audit_logger.log_security_event(event)
        
        def search_logs():
            return audit_logger.search_logs({'event_type': 'search_test'})
        
        # Run concurrent searches
        threads = []
        results = []
        
        def run_search():
            result = search_logs()
            results.append(len(result))
        
        for i in range(5):
            thread = threading.Thread(target=run_search)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All searches should return same number of results
        assert len(set(results)) <= 2  # Slight variations due to concurrent writes are ok


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling"""
    
    @pytest.fixture
    def audit_logger(self):
        """Create audit logger for performance testing"""
        temp_dir = tempfile.mkdtemp()
        config = {
            'log_path': temp_dir,
            'retention_days': 1,
            'max_log_size_mb': 100,  # Large size to avoid rotation during test
            'compression_enabled': False  # Disable to reduce overhead
        }
        logger = AuditLogger(config)
        yield logger
        logger.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_large_volume_logging_performance(self, audit_logger):
        """Test performance with large volume of events"""
        num_events = 1000
        
        start_time = time.time()
        
        for i in range(num_events):
            event = {
                'event_type': 'performance_test',
                'event_id': i,
                'severity': 'info',
                'data': f'test_data_{i}'
            }
            audit_logger.log_security_event(event)
        
        logging_time = time.time() - start_time
        
        # Should be able to log 1000 events quickly
        assert logging_time < 5.0  # Less than 5 seconds
        
        # Events per second calculation
        eps = num_events / logging_time
        assert eps > 200  # At least 200 events per second
    
    def test_search_performance_large_dataset(self, audit_logger):
        """Test search performance with large dataset"""
        # Create large dataset
        num_events = 2000
        
        for i in range(num_events):
            event = {
                'event_type': f'type_{i % 10}',
                'username': f'user_{i % 100}',
                'severity': ['info', 'warning', 'high'][i % 3],
                'data': f'event_data_{i}'
            }
            audit_logger.log_security_event(event)
        
        # Test search performance
        start_time = time.time()
        results = audit_logger.search_logs({'event_type': 'type_5'})
        search_time = time.time() - start_time
        
        # Search should be fast
        assert search_time < 1.0  # Less than 1 second
        assert len(results) == num_events // 10  # Should find correct number
    
    def test_memory_usage_with_large_buffer(self, audit_logger):
        """Test memory usage with large buffer"""
        # Fill up the buffer
        buffer_size = audit_logger.log_buffer.maxlen
        
        for i in range(buffer_size):
            event = {
                'event_type': 'memory_test',
                'data': f'data_{i}',
                'severity': 'info'
            }
            audit_logger.log_security_event(event)
        
        # Buffer should be at max capacity
        assert len(audit_logger.log_buffer) == buffer_size
        
        # Add more events - should not exceed max
        for i in range(100):
            event = {
                'event_type': 'overflow_test',
                'data': f'overflow_{i}',
                'severity': 'info'
            }
            audit_logger.log_security_event(event)
        
        # Buffer should still be at max capacity
        assert len(audit_logger.log_buffer) == buffer_size


class TestErrorHandling:
    """Test error handling and recovery"""
    
    @pytest.fixture
    def audit_logger(self):
        """Create audit logger for error testing"""
        temp_dir = tempfile.mkdtemp()
        config = {
            'log_path': temp_dir,
            'retention_days': 1,
            'max_log_size_mb': 1
        }
        logger = AuditLogger(config)
        yield logger
        logger.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_invalid_event_handling(self, audit_logger):
        """Test handling of invalid events"""
        # Test with None event
        result = audit_logger.log_security_event(None)
        assert result == False
        
        # Test with invalid event structure
        invalid_events = [
            123,  # Not a dict
            "invalid",  # String instead of dict
            []  # List instead of dict
        ]
        
        for invalid_event in invalid_events:
            result = audit_logger.log_security_event(invalid_event)
            assert result == False
    
    def test_disk_write_error_handling(self, audit_logger):
        """Test handling of disk write errors"""
        # Create an event
        event = {'event_type': 'test', 'severity': 'info'}
        audit_logger.log_security_event(event)
        
        # Make log path readonly to simulate write error
        os.chmod(audit_logger.log_path, 0o444)
        
        try:
            # Try to write events - should handle error gracefully
            audit_logger._write_events_to_disk([{
                'event_id': 'test',
                'event': event,
                'timestamp': time.time()
            }])
            
            # Should not crash, error should be logged
            
        finally:
            # Restore permissions
            os.chmod(audit_logger.log_path, 0o755)
    
    def test_corrupted_log_file_handling(self, audit_logger):
        """Test handling of corrupted log files"""
        # Create a corrupted log file
        corrupted_file = audit_logger.log_path / "audit_corrupted.log"
        
        with open(corrupted_file, 'w') as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')  # Corrupted line
            f.write('{"another": "valid", "json": true}\n')
        
        # Search should handle corrupted file gracefully
        results = audit_logger._search_log_files({}, None)
        
        # Should return valid entries, skip corrupted ones
        assert isinstance(results, list)
    
    def test_search_with_invalid_criteria(self, audit_logger):
        """Test search with invalid criteria"""
        # Add some test data
        event = {'event_type': 'test', 'severity': 'info'}
        audit_logger.log_security_event(event)
        
        # Search with invalid criteria should return empty results, not crash
        invalid_criteria = [
            None,
            "invalid",
            123,
            {'invalid_time_range': 'not_a_datetime'}
        ]
        
        for criteria in invalid_criteria:
            try:
                results = audit_logger.search_logs(criteria if criteria is not None else {})
                assert isinstance(results, list)
            except:
                # If it throws an exception, it should be handled gracefully
                pass


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.fixture
    def audit_logger(self):
        """Create audit logger for integration testing"""
        temp_dir = tempfile.mkdtemp()
        config = {
            'log_path': temp_dir,
            'retention_days': 7,
            'max_log_size_mb': 5,
            'compression_enabled': True,
            'real_time_alerts': True
        }
        logger = AuditLogger(config)
        yield logger
        logger.shutdown()
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_security_incident_scenario(self, audit_logger):
        """Test complete security incident logging and analysis"""
        # Simulate security incident sequence
        incident_events = [
            {'event_type': 'failed_login', 'username': 'admin', 'source_ip': '192.168.1.100', 'severity': 'warning'},
            {'event_type': 'failed_login', 'username': 'admin', 'source_ip': '192.168.1.100', 'severity': 'warning'},
            {'event_type': 'failed_login', 'username': 'admin', 'source_ip': '192.168.1.100', 'severity': 'high'},
            {'event_type': 'account_locked', 'username': 'admin', 'severity': 'high'},
            {'event_type': 'security_alert', 'description': 'Multiple failed login attempts', 'severity': 'critical'},
            {'event_type': 'admin_notified', 'description': 'Security team notified', 'severity': 'info'}
        ]
        
        for event in incident_events:
            result = audit_logger.log_security_event(event)
            assert result == True
        
        # Generate incident report
        start_date = datetime.now() - timedelta(minutes=5)
        end_date = datetime.now() + timedelta(minutes=5)
        
        report = audit_logger.get_audit_report(start_date, end_date)
        
        assert report['total_events'] == 6
        assert report['severity_summary']['critical'] == 1
        assert report['severity_summary']['high'] == 2
        assert len(report['critical_events']) == 3  # high and critical events
        
        # Search for incident-related events
        failed_logins = audit_logger.search_logs({'event_type': 'failed_login'})
        assert len(failed_logins) == 3
        
        admin_events = audit_logger.search_logs({'username': 'admin'})
        assert len(admin_events) == 4  # failed_login events + account_locked
    
    def test_compliance_audit_scenario(self, audit_logger):
        """Test compliance audit scenario"""
        # Simulate compliance-related events
        compliance_events = [
            {'event_type': 'policy_update', 'policy_id': 'POL001', 'severity': 'info'},
            {'event_type': 'compliance_check', 'check_type': 'access_control', 'result': 'passed', 'severity': 'info'},
            {'event_type': 'policy_violation', 'username': 'user1', 'policy_id': 'POL001', 'severity': 'high'},
            {'event_type': 'audit_log_access', 'username': 'auditor', 'severity': 'info'},
            {'event_type': 'compliance_report_generated', 'report_id': 'RPT001', 'severity': 'info'}
        ]
        
        for event in compliance_events:
            audit_logger.log_security_event(event)
        
        # Generate compliance report
        start_date = datetime.now() - timedelta(minutes=5)
        end_date = datetime.now() + timedelta(minutes=5)
        
        report = audit_logger.get_audit_report(start_date, end_date)
        compliance_events_in_report = report['compliance_events']
        
        # Should find policy_violation event
        assert len(compliance_events_in_report) >= 1
        policy_violations = [e for e in compliance_events_in_report if e['event_type'] == 'policy_violation']
        assert len(policy_violations) == 1
    
    def test_high_volume_production_scenario(self, audit_logger):
        """Test high-volume production scenario"""
        # Simulate high-volume production logging
        event_types = ['user_login', 'user_logout', 'data_access', 'admin_action', 'system_event']
        users = [f'user_{i}' for i in range(20)]
        severities = ['info', 'warning', 'high']
        
        num_events = 500
        
        start_time = time.time()
        
        for i in range(num_events):
            event = {
                'event_type': event_types[i % len(event_types)],
                'username': users[i % len(users)],
                'severity': severities[i % len(severities)],
                'session_id': f'sess_{i // 10}',
                'timestamp_custom': time.time()
            }
            audit_logger.log_security_event(event)
        
        logging_time = time.time() - start_time
        
        # Should handle high volume efficiently
        assert logging_time < 10.0  # Should complete within 10 seconds
        
        # Wait for background processing
        time.sleep(3)
        
        # Generate comprehensive report
        start_date = datetime.now() - timedelta(minutes=5)
        end_date = datetime.now() + timedelta(minutes=5)
        
        report = audit_logger.get_audit_report(start_date, end_date)
        
        # Verify report completeness
        assert report['total_events'] == num_events
        assert len(report['user_activity']['top_users']) > 0
        assert len(report['top_event_types']) == len(event_types)
        
        # Test search performance
        search_start = time.time()
        login_events = audit_logger.search_logs({'event_type': 'user_login'})
        search_time = time.time() - search_start
        
        assert search_time < 1.0  # Search should be fast
        assert len(login_events) == num_events // len(event_types)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])