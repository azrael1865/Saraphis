#!/usr/bin/env python3
"""
Direct test of DataManager methods without complex dependencies
Tests that the methods exist and have correct signatures
"""

import sys
import os
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock all problematic modules completely
problematic_modules = [
    'lz4',
    'lz4.frame',
    'cryptography',
    'cryptography.fernet',
    'cryptography.hazmat',
    'cryptography.hazmat.primitives',
    'cryptography.hazmat.primitives.ciphers',
    'cryptography.hazmat.primitives.kdf',
    'cryptography.hazmat.primitives.hashes',
    'cryptography.hazmat.backends'
]

for module_name in problematic_modules:
    sys.modules[module_name] = Mock()

# Create fully mocked versions of the manager classes
class FullMockManager:
    def __init__(self, config=None):
        self.config = config or {}
    
    def __getattr__(self, name):
        """Return a mock for any method call"""
        return Mock(return_value={'status': 'mocked', 'method': name})


class TestDataManagerDirectly(unittest.TestCase):
    """Test DataManager using direct mocking approach"""
    
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = {
            'backup_config': {'backup_path': str(self.test_dir / 'backups')},
            'storage_config': {'storage_path': str(self.test_dir / 'storage')}
        }
    
    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @patch('production_data.data_manager.BackupManager', FullMockManager)
    @patch('production_data.data_manager.EncryptionManager', FullMockManager)
    @patch('production_data.data_manager.CompressionManager', FullMockManager) 
    @patch('production_data.data_manager.StorageManager', FullMockManager)
    @patch('production_data.data_manager.ReplicationManager', FullMockManager)
    @patch('production_data.data_manager.DataMetricsCollector', FullMockManager)
    def test_data_manager_methods_exist(self):
        """Test that all expected methods exist in DataManager"""
        from production_data.data_manager import DataManager
        
        manager = DataManager(self.config)
        
        # Test that all expected methods exist
        expected_methods = [
            'create_backup', 'restore_backup', 'list_backups',
            'encrypt_data', 'decrypt_data', 'rotate_encryption_keys', 
            'store_data', 'retrieve_data', 'delete_data',
            'compress_data', 'decompress_data',
            'get_data_metrics', 'get_historical_metrics',
            'sync_replicas', 'check_replica_health',
            'cleanup_resources', 'validate_data_integrity', 'shutdown'
        ]
        
        missing_methods = []
        for method_name in expected_methods:
            if not hasattr(manager, method_name):
                missing_methods.append(method_name)
            else:
                # Test that the method is callable
                method = getattr(manager, method_name)
                self.assertTrue(callable(method), f"Method {method_name} is not callable")
        
        self.assertEqual(len(missing_methods), 0, f"Missing methods: {missing_methods}")
        
        # Test basic initialization
        self.assertIsNotNone(manager.config)
        self.assertIsNotNone(manager.data_history)
        self.assertIsNotNone(manager.backup_history)
        self.assertIsNotNone(manager.data_metrics)
        
        manager.shutdown()
    
    @patch('production_data.data_manager.BackupManager', FullMockManager)
    @patch('production_data.data_manager.EncryptionManager', FullMockManager)
    @patch('production_data.data_manager.CompressionManager', FullMockManager)
    @patch('production_data.data_manager.StorageManager', FullMockManager)
    @patch('production_data.data_manager.ReplicationManager', FullMockManager)
    @patch('production_data.data_manager.DataMetricsCollector', FullMockManager)
    def test_backup_methods(self):
        """Test backup-related methods"""
        from production_data.data_manager import DataManager
        
        manager = DataManager(self.config)
        
        # Mock specific backup manager behavior
        manager.backup_manager.create_backup = Mock(return_value={'backup_id': 'test_001', 'status': 'success'})
        manager.backup_manager.restore_backup = Mock(return_value={'data': 'restored'})
        manager.backup_manager.list_backups = Mock(return_value=[{'backup_id': 'test_001'}])
        
        # Test create_backup
        result = manager.create_backup({'test': 'data'}, 'test_001')
        self.assertIn('backup_id', result)
        
        # Test restore_backup  
        result = manager.restore_backup('test_001')
        self.assertIsNotNone(result)
        
        # Test list_backups
        result = manager.list_backups()
        self.assertIsInstance(result, list)
        
        manager.shutdown()
    
    @patch('production_data.data_manager.BackupManager', FullMockManager)
    @patch('production_data.data_manager.EncryptionManager', FullMockManager)
    @patch('production_data.data_manager.CompressionManager', FullMockManager)
    @patch('production_data.data_manager.StorageManager', FullMockManager)
    @patch('production_data.data_manager.ReplicationManager', FullMockManager)
    @patch('production_data.data_manager.DataMetricsCollector', FullMockManager)
    def test_data_storage_methods(self):
        """Test data storage methods"""
        from production_data.data_manager import DataManager
        
        manager = DataManager(self.config)
        
        # Mock storage manager behavior
        manager.storage_manager.store_data = Mock(return_value={'status': 'stored', 'data_id': 'test_data'})
        manager.storage_manager.retrieve_data = Mock(return_value={'test': 'data'})
        manager.storage_manager.delete_data = Mock(return_value={'status': 'deleted'})
        
        # Test store_data
        result = manager.store_data({'test': 'data'}, 'test_data')
        self.assertEqual(result['status'], 'stored')
        
        # Test retrieve_data
        result = manager.retrieve_data('test_data')
        self.assertIsNotNone(result)
        
        # Test delete_data
        result = manager.delete_data('test_data')
        self.assertEqual(result['status'], 'deleted')
        
        manager.shutdown()
    
    @patch('production_data.data_manager.BackupManager', FullMockManager)
    @patch('production_data.data_manager.EncryptionManager', FullMockManager)
    @patch('production_data.data_manager.CompressionManager', FullMockManager)
    @patch('production_data.data_manager.StorageManager', FullMockManager)
    @patch('production_data.data_manager.ReplicationManager', FullMockManager)
    @patch('production_data.data_manager.DataMetricsCollector', FullMockManager)
    def test_encryption_methods(self):
        """Test encryption methods"""
        from production_data.data_manager import DataManager
        
        manager = DataManager(self.config)
        
        # Mock encryption manager behavior
        manager.encryption_manager.encrypt_data = Mock(return_value=b'encrypted_data')
        manager.encryption_manager.decrypt_data = Mock(return_value=b'decrypted_data')
        manager.encryption_manager.rotate_keys = Mock(return_value={'status': 'success'})
        
        # Test encrypt_data
        result = manager.encrypt_data(b'test_data')
        self.assertEqual(result, b'encrypted_data')
        
        # Test decrypt_data
        result = manager.decrypt_data(b'encrypted_data')
        self.assertEqual(result, b'decrypted_data')
        
        # Test rotate_encryption_keys
        result = manager.rotate_encryption_keys()
        self.assertEqual(result['status'], 'success')
        
        manager.shutdown()
    
    @patch('production_data.data_manager.BackupManager', FullMockManager)
    @patch('production_data.data_manager.EncryptionManager', FullMockManager)
    @patch('production_data.data_manager.CompressionManager', FullMockManager)
    @patch('production_data.data_manager.StorageManager', FullMockManager)
    @patch('production_data.data_manager.ReplicationManager', FullMockManager)
    @patch('production_data.data_manager.DataMetricsCollector', FullMockManager)
    def test_validate_data_integrity(self):
        """Test data integrity validation"""
        from production_data.data_manager import DataManager
        
        manager = DataManager(self.config)
        
        # Mock all manager methods needed for validation
        manager.data_metrics_collector.collect_all_metrics = Mock(return_value={'total_size': 1000})
        manager.backup_manager.validate_backup_integrity = Mock(return_value={'integrity_score': 0.9})
        manager.encryption_manager.validate_encryption_status = Mock(return_value={'encryption_score': 0.95})
        manager.compression_manager.validate_compression_integrity = Mock(return_value={'integrity_score': 0.85})
        manager.storage_manager.check_storage_health = Mock(return_value={'health_score': 0.88})
        manager.replication_manager.validate_replication_status = Mock(return_value={'replication_score': 0.92})
        
        result = manager.validate_data_integrity()
        
        self.assertIn('status', result)
        self.assertIn('integrity_score', result)
        self.assertIsInstance(result['integrity_score'], float)
        
        manager.shutdown()
    
    @patch('production_data.data_manager.BackupManager', FullMockManager)
    @patch('production_data.data_manager.EncryptionManager', FullMockManager)
    @patch('production_data.data_manager.CompressionManager', FullMockManager)
    @patch('production_data.data_manager.StorageManager', FullMockManager)
    @patch('production_data.data_manager.ReplicationManager', FullMockManager)
    @patch('production_data.data_manager.DataMetricsCollector', FullMockManager)
    def test_cleanup_resources(self):
        """Test cleanup_resources method"""
        from production_data.data_manager import DataManager
        
        manager = DataManager(self.config)
        
        # Add some test data to history
        for i in range(100):
            manager.data_history.append({'test': f'data_{i}'})
            manager.backup_history.append({'test': f'backup_{i}'})
        
        result = manager.cleanup_resources()
        
        self.assertIn('status', result)
        self.assertIn('resources_freed', result)
        
        manager.shutdown()


class TestDataManagerErrorHandling(unittest.TestCase):
    """Test error handling in DataManager"""
    
    @patch('production_data.data_manager.BackupManager', FullMockManager)
    @patch('production_data.data_manager.EncryptionManager', FullMockManager)
    @patch('production_data.data_manager.CompressionManager', FullMockManager)
    @patch('production_data.data_manager.StorageManager', FullMockManager)
    @patch('production_data.data_manager.ReplicationManager', FullMockManager)
    @patch('production_data.data_manager.DataMetricsCollector', FullMockManager)
    def test_backup_creation_error(self):
        """Test error handling in backup creation"""
        from production_data.data_manager import DataManager
        
        manager = DataManager({})
        
        # Make backup_manager raise an exception
        manager.backup_manager.create_backup = Mock(side_effect=Exception("Backup failed"))
        
        with self.assertRaises(Exception):
            manager.create_backup({'test': 'data'}, 'failing_backup')
        
        manager.shutdown()
    
    @patch('production_data.data_manager.BackupManager', FullMockManager)
    @patch('production_data.data_manager.EncryptionManager', FullMockManager)
    @patch('production_data.data_manager.CompressionManager', FullMockManager)
    @patch('production_data.data_manager.StorageManager', FullMockManager)
    @patch('production_data.data_manager.ReplicationManager', FullMockManager)
    @patch('production_data.data_manager.DataMetricsCollector', FullMockManager)
    def test_data_integrity_validation_error(self):
        """Test error handling in data integrity validation"""
        from production_data.data_manager import DataManager
        
        manager = DataManager({})
        
        # Make metrics collector raise an exception
        manager.data_metrics_collector.collect_all_metrics = Mock(side_effect=Exception("Metrics failed"))
        
        result = manager.validate_data_integrity()
        
        # Should return error result instead of raising
        self.assertEqual(result['status'], 'error')
        self.assertIn('error', result)
        self.assertEqual(result['integrity_score'], 0.0)
        
        manager.shutdown()


if __name__ == '__main__':
    unittest.main(verbosity=2)