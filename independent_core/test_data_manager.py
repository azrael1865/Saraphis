#!/usr/bin/env python3
"""
Comprehensive test suite for DataManager
Tests all functionality to identify root issues that need fixing in source code.
"""

import unittest
import tempfile
import shutil
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import os

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from production_data.data_manager import DataManager
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class TestDataManagerInitialization(unittest.TestCase):
    """Test DataManager initialization and configuration"""
    
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = {
            'backup_config': {'backup_path': str(self.test_dir / 'backups')},
            'encryption_config': {'key_path': str(self.test_dir / 'keys')},
            'compression_config': {'algorithm': 'gzip'},
            'storage_config': {'storage_path': str(self.test_dir / 'storage')},
            'replication_config': {'replicas': 2},
            'metrics_config': {'collection_interval': 60}
        }
    
    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_data_manager_initialization(self):
        """Test DataManager initialization with valid config"""
        with patch('production_data.backup_manager.BackupManager'), \
             patch('production_data.encryption_manager.EncryptionManager'), \
             patch('production_data.compression_manager.CompressionManager'), \
             patch('production_data.storage_manager.StorageManager'), \
             patch('production_data.replication_manager.ReplicationManager'), \
             patch('production_data.data_metrics.DataMetricsCollector'):
            
            manager = DataManager(self.config)
            
            self.assertEqual(manager.config, self.config)
            self.assertIsNotNone(manager.data_history)
            self.assertIsNotNone(manager.backup_history)
            self.assertIsInstance(manager.data_metrics, dict)
            self.assertTrue(manager.is_running)
            self.assertIsNotNone(manager._lock)
    
    def test_data_manager_empty_config(self):
        """Test DataManager initialization with empty config"""
        with patch('production_data.backup_manager.BackupManager'), \
             patch('production_data.encryption_manager.EncryptionManager'), \
             patch('production_data.compression_manager.CompressionManager'), \
             patch('production_data.storage_manager.StorageManager'), \
             patch('production_data.replication_manager.ReplicationManager'), \
             patch('production_data.data_metrics.DataMetricsCollector'):
            
            manager = DataManager({})
            
            self.assertEqual(manager.config, {})
            self.assertIsNotNone(manager.data_metrics)


class TestDataIntegrityValidation(unittest.TestCase):
    """Test data integrity validation functionality"""
    
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = {
            'backup_config': {'backup_path': str(self.test_dir / 'backups')},
            'storage_config': {'storage_path': str(self.test_dir / 'storage')}
        }
        
        # Mock all managers
        self.backup_manager_mock = Mock()
        self.encryption_manager_mock = Mock()
        self.compression_manager_mock = Mock()
        self.storage_manager_mock = Mock()
        self.replication_manager_mock = Mock()
        self.metrics_collector_mock = Mock()
        
        with patch('production_data.data_manager.BackupManager', return_value=self.backup_manager_mock), \
             patch('production_data.data_manager.EncryptionManager', return_value=self.encryption_manager_mock), \
             patch('production_data.data_manager.CompressionManager', return_value=self.compression_manager_mock), \
             patch('production_data.data_manager.StorageManager', return_value=self.storage_manager_mock), \
             patch('production_data.data_manager.ReplicationManager', return_value=self.replication_manager_mock), \
             patch('production_data.data_manager.DataMetricsCollector', return_value=self.metrics_collector_mock):
            
            self.manager = DataManager(self.config)
    
    def tearDown(self):
        if hasattr(self, 'manager'):
            self.manager.shutdown()
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_validate_data_integrity_success(self):
        """Test successful data integrity validation"""
        # Mock successful responses
        self.metrics_collector_mock.collect_all_metrics.return_value = {
            'total_size': 1000,
            'compression_ratio': 0.7
        }
        self.backup_manager_mock.validate_backup_integrity.return_value = {
            'integrity_score': 0.9,
            'failed_backups': 0
        }
        self.encryption_manager_mock.validate_encryption_status.return_value = {
            'encryption_score': 0.95
        }
        self.compression_manager_mock.validate_compression_integrity.return_value = {
            'integrity_score': 0.85
        }
        self.storage_manager_mock.check_storage_health.return_value = {
            'health_score': 0.88
        }
        self.replication_manager_mock.validate_replication_status.return_value = {
            'replication_score': 0.92
        }
        
        result = self.manager.validate_data_integrity()
        
        self.assertIn('status', result)
        self.assertIn('integrity_score', result)
        self.assertIsInstance(result['integrity_score'], float)
        self.assertGreaterEqual(result['integrity_score'], 0.0)
        self.assertLessEqual(result['integrity_score'], 1.0)
    
    def test_validate_data_integrity_failure(self):
        """Test data integrity validation with failure"""
        # Mock failure in metrics collection
        self.metrics_collector_mock.collect_all_metrics.side_effect = Exception("Metrics collection failed")
        
        result = self.manager.validate_data_integrity()
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('error', result)
        self.assertEqual(result['integrity_score'], 0.0)
        self.assertIn('data_issues', result)
    
    def test_calculate_data_integrity_score(self):
        """Test data integrity score calculation"""
        metrics = {'total_size': 1000}
        backup_integrity = {'integrity_score': 0.9}
        encryption_status = {'encryption_score': 0.95}
        compression_integrity = {'integrity_score': 0.85}
        storage_health = {'health_score': 0.88}
        replication_status = {'replication_score': 0.92}
        
        score = self.manager._calculate_data_integrity_score(
            metrics, backup_integrity, encryption_status,
            compression_integrity, storage_health, replication_status
        )
        
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_identify_data_issues(self):
        """Test data issue identification"""
        # Mock data with issues
        metrics = {'total_size': 1000}
        backup_integrity = {'integrity_score': 0.7, 'failed_backups': 2}  # Below threshold
        encryption_status = {'encryption_score': 0.95}
        compression_integrity = {'integrity_score': 0.85}
        storage_health = {'health_score': 0.88}
        replication_status = {'replication_score': 0.92}
        
        issues = self.manager._identify_data_issues(
            metrics, backup_integrity, encryption_status,
            compression_integrity, storage_health, replication_status
        )
        
        self.assertIsInstance(issues, list)
        self.assertGreater(len(issues), 0)
        
        # Check that backup issue is identified
        backup_issue = next((issue for issue in issues if issue['type'] == 'backup_integrity_issue'), None)
        self.assertIsNotNone(backup_issue)
        self.assertEqual(backup_issue['severity'], 'high')


class TestBackupManagement(unittest.TestCase):
    """Test backup management functionality"""
    
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = {
            'backup_config': {'backup_path': str(self.test_dir / 'backups')}
        }
        
        self.backup_manager_mock = Mock()
        
        with patch('production_data.data_manager.BackupManager', return_value=self.backup_manager_mock), \
             patch('production_data.data_manager.EncryptionManager'), \
             patch('production_data.data_manager.CompressionManager'), \
             patch('production_data.data_manager.StorageManager'), \
             patch('production_data.data_manager.ReplicationManager'), \
             patch('production_data.data_manager.DataMetricsCollector'):
            
            self.manager = DataManager(self.config)
    
    def tearDown(self):
        if hasattr(self, 'manager'):
            self.manager.shutdown()
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_create_backup(self):
        """Test backup creation"""
        test_data = {'key': 'value', 'number': 42}
        backup_id = 'test_backup_001'
        
        self.backup_manager_mock.create_backup.return_value = {
            'backup_id': backup_id,
            'status': 'success',
            'size': 1024
        }
        
        result = self.manager.create_backup(test_data, backup_id)
        
        self.backup_manager_mock.create_backup.assert_called_once_with(test_data, backup_id)
        self.assertIn('backup_id', result)
        self.assertEqual(result['status'], 'success')
    
    def test_restore_backup(self):
        """Test backup restoration"""
        backup_id = 'test_backup_001'
        expected_data = {'key': 'value', 'number': 42}
        
        self.backup_manager_mock.restore_backup.return_value = expected_data
        
        result = self.manager.restore_backup(backup_id)
        
        self.backup_manager_mock.restore_backup.assert_called_once_with(backup_id)
        self.assertEqual(result, expected_data)
    
    def test_list_backups(self):
        """Test backup listing"""
        expected_backups = [
            {'backup_id': 'backup_001', 'timestamp': time.time(), 'size': 1024},
            {'backup_id': 'backup_002', 'timestamp': time.time(), 'size': 2048}
        ]
        
        self.backup_manager_mock.list_backups.return_value = expected_backups
        
        result = self.manager.list_backups()
        
        self.backup_manager_mock.list_backups.assert_called_once()
        self.assertEqual(result, expected_backups)


class TestEncryptionManagement(unittest.TestCase):
    """Test encryption management functionality"""
    
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = {
            'encryption_config': {'key_path': str(self.test_dir / 'keys')}
        }
        
        self.encryption_manager_mock = Mock()
        
        with patch('production_data.data_manager.BackupManager'), \
             patch('production_data.data_manager.EncryptionManager', return_value=self.encryption_manager_mock), \
             patch('production_data.data_manager.CompressionManager'), \
             patch('production_data.data_manager.StorageManager'), \
             patch('production_data.data_manager.ReplicationManager'), \
             patch('production_data.data_manager.DataMetricsCollector'):
            
            self.manager = DataManager(self.config)
    
    def tearDown(self):
        if hasattr(self, 'manager'):
            self.manager.shutdown()
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_encrypt_data(self):
        """Test data encryption"""
        test_data = b"sensitive data"
        expected_encrypted = b"encrypted_data"
        
        self.encryption_manager_mock.encrypt_data.return_value = expected_encrypted
        
        result = self.manager.encrypt_data(test_data)
        
        self.encryption_manager_mock.encrypt_data.assert_called_once_with(test_data)
        self.assertEqual(result, expected_encrypted)
    
    def test_decrypt_data(self):
        """Test data decryption"""
        encrypted_data = b"encrypted_data"
        expected_decrypted = b"sensitive data"
        
        self.encryption_manager_mock.decrypt_data.return_value = expected_decrypted
        
        result = self.manager.decrypt_data(encrypted_data)
        
        self.encryption_manager_mock.decrypt_data.assert_called_once_with(encrypted_data)
        self.assertEqual(result, expected_decrypted)
    
    def test_rotate_encryption_keys(self):
        """Test encryption key rotation"""
        self.encryption_manager_mock.rotate_keys.return_value = {'status': 'success', 'new_key_id': 'key_002'}
        
        result = self.manager.rotate_encryption_keys()
        
        self.encryption_manager_mock.rotate_keys.assert_called_once()
        self.assertEqual(result['status'], 'success')


class TestDataManipulation(unittest.TestCase):
    """Test data manipulation and processing"""
    
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = {
            'storage_config': {'storage_path': str(self.test_dir / 'storage')}
        }
        
        with patch('production_data.backup_manager.BackupManager'), \
             patch('production_data.encryption_manager.EncryptionManager'), \
             patch('production_data.compression_manager.CompressionManager'), \
             patch('production_data.storage_manager.StorageManager'), \
             patch('production_data.replication_manager.ReplicationManager'), \
             patch('production_data.data_metrics.DataMetricsCollector'):
            
            self.manager = DataManager(self.config)
    
    def tearDown(self):
        if hasattr(self, 'manager'):
            self.manager.shutdown()
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_store_data(self):
        """Test data storage"""
        test_data = {'key': 'value', 'timestamp': time.time()}
        data_id = 'test_data_001'
        
        with patch.object(self.manager.storage_manager, 'store_data') as mock_store:
            mock_store.return_value = {'data_id': data_id, 'status': 'stored'}
            
            result = self.manager.store_data(test_data, data_id)
            
            mock_store.assert_called_once_with(test_data, data_id)
            self.assertEqual(result['status'], 'stored')
    
    def test_retrieve_data(self):
        """Test data retrieval"""
        data_id = 'test_data_001'
        expected_data = {'key': 'value', 'timestamp': time.time()}
        
        with patch.object(self.manager.storage_manager, 'retrieve_data') as mock_retrieve:
            mock_retrieve.return_value = expected_data
            
            result = self.manager.retrieve_data(data_id)
            
            mock_retrieve.assert_called_once_with(data_id)
            self.assertEqual(result, expected_data)
    
    def test_delete_data(self):
        """Test data deletion"""
        data_id = 'test_data_001'
        
        with patch.object(self.manager.storage_manager, 'delete_data') as mock_delete:
            mock_delete.return_value = {'status': 'deleted', 'data_id': data_id}
            
            result = self.manager.delete_data(data_id)
            
            mock_delete.assert_called_once_with(data_id)
            self.assertEqual(result['status'], 'deleted')


class TestCompressionManagement(unittest.TestCase):
    """Test compression management functionality"""
    
    def setUp(self):
        self.config = {'compression_config': {'algorithm': 'gzip', 'level': 6}}
        
        self.compression_manager_mock = Mock()
        
        with patch('production_data.data_manager.BackupManager'), \
             patch('production_data.data_manager.EncryptionManager'), \
             patch('production_data.data_manager.CompressionManager', return_value=self.compression_manager_mock), \
             patch('production_data.data_manager.StorageManager'), \
             patch('production_data.data_manager.ReplicationManager'), \
             patch('production_data.data_manager.DataMetricsCollector'):
            
            self.manager = DataManager(self.config)
    
    def tearDown(self):
        if hasattr(self, 'manager'):
            self.manager.shutdown()
    
    def test_compress_data(self):
        """Test data compression"""
        test_data = b"test data for compression"
        compressed_data = b"compressed_data"
        
        self.compression_manager_mock.compress_data.return_value = compressed_data
        
        result = self.manager.compress_data(test_data)
        
        self.compression_manager_mock.compress_data.assert_called_once_with(test_data)
        self.assertEqual(result, compressed_data)
    
    def test_decompress_data(self):
        """Test data decompression"""
        compressed_data = b"compressed_data"
        decompressed_data = b"test data for compression"
        
        self.compression_manager_mock.decompress_data.return_value = decompressed_data
        
        result = self.manager.decompress_data(compressed_data)
        
        self.compression_manager_mock.decompress_data.assert_called_once_with(compressed_data)
        self.assertEqual(result, decompressed_data)


class TestMetricsCollection(unittest.TestCase):
    """Test metrics collection and monitoring"""
    
    def setUp(self):
        self.config = {'metrics_config': {'collection_interval': 30}}
        
        self.metrics_collector_mock = Mock()
        
        with patch('production_data.data_manager.BackupManager'), \
             patch('production_data.data_manager.EncryptionManager'), \
             patch('production_data.data_manager.CompressionManager'), \
             patch('production_data.data_manager.StorageManager'), \
             patch('production_data.data_manager.ReplicationManager'), \
             patch('production_data.data_manager.DataMetricsCollector', return_value=self.metrics_collector_mock):
            
            self.manager = DataManager(self.config)
    
    def tearDown(self):
        if hasattr(self, 'manager'):
            self.manager.shutdown()
    
    def test_get_data_metrics(self):
        """Test getting data metrics"""
        expected_metrics = {
            'total_data_size_gb': 1.5,
            'compression_ratio': 0.7,
            'backup_success_rate': 0.95
        }
        
        self.metrics_collector_mock.get_current_metrics.return_value = expected_metrics
        
        result = self.manager.get_data_metrics()
        
        self.metrics_collector_mock.get_current_metrics.assert_called_once()
        self.assertEqual(result, expected_metrics)
    
    def test_get_historical_metrics(self):
        """Test getting historical metrics"""
        start_time = time.time() - 3600  # 1 hour ago
        end_time = time.time()
        
        expected_history = [
            {'timestamp': start_time + 300, 'metrics': {'size': 1000}},
            {'timestamp': start_time + 600, 'metrics': {'size': 1100}}
        ]
        
        self.metrics_collector_mock.get_historical_metrics.return_value = expected_history
        
        result = self.manager.get_historical_metrics(start_time, end_time)
        
        self.metrics_collector_mock.get_historical_metrics.assert_called_once_with(start_time, end_time)
        self.assertEqual(result, expected_history)


class TestReplicationManagement(unittest.TestCase):
    """Test replication management functionality"""
    
    def setUp(self):
        self.config = {'replication_config': {'replicas': 3, 'sync_interval': 60}}
        
        self.replication_manager_mock = Mock()
        
        with patch('production_data.data_manager.BackupManager'), \
             patch('production_data.data_manager.EncryptionManager'), \
             patch('production_data.data_manager.CompressionManager'), \
             patch('production_data.data_manager.StorageManager'), \
             patch('production_data.data_manager.ReplicationManager', return_value=self.replication_manager_mock), \
             patch('production_data.data_manager.DataMetricsCollector'):
            
            self.manager = DataManager(self.config)
    
    def tearDown(self):
        if hasattr(self, 'manager'):
            self.manager.shutdown()
    
    def test_sync_replicas(self):
        """Test replica synchronization"""
        self.replication_manager_mock.sync_replicas.return_value = {
            'status': 'success',
            'synced_replicas': 3,
            'failed_replicas': 0
        }
        
        result = self.manager.sync_replicas()
        
        self.replication_manager_mock.sync_replicas.assert_called_once()
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['synced_replicas'], 3)
    
    def test_check_replica_health(self):
        """Test replica health checking"""
        expected_health = {
            'healthy_replicas': 2,
            'unhealthy_replicas': 1,
            'total_replicas': 3,
            'health_score': 0.67
        }
        
        self.replication_manager_mock.check_replica_health.return_value = expected_health
        
        result = self.manager.check_replica_health()
        
        self.replication_manager_mock.check_replica_health.assert_called_once()
        self.assertEqual(result, expected_health)


class TestThreadSafety(unittest.TestCase):
    """Test thread safety and concurrent operations"""
    
    def setUp(self):
        self.config = {}
        
        with patch('production_data.backup_manager.BackupManager'), \
             patch('production_data.encryption_manager.EncryptionManager'), \
             patch('production_data.compression_manager.CompressionManager'), \
             patch('production_data.storage_manager.StorageManager'), \
             patch('production_data.replication_manager.ReplicationManager'), \
             patch('production_data.data_metrics.DataMetricsCollector'):
            
            self.manager = DataManager(self.config)
    
    def tearDown(self):
        if hasattr(self, 'manager'):
            self.manager.shutdown()
    
    def test_concurrent_data_operations(self):
        """Test concurrent data operations"""
        results = []
        errors = []
        
        def store_operation(index):
            try:
                with patch.object(self.manager.storage_manager, 'store_data') as mock_store:
                    mock_store.return_value = {'status': 'stored', 'data_id': f'data_{index}'}
                    result = self.manager.store_data({'index': index}, f'data_{index}')
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=store_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 10)
    
    def test_concurrent_backup_operations(self):
        """Test concurrent backup operations"""
        results = []
        errors = []
        
        def backup_operation(index):
            try:
                with patch.object(self.manager.backup_manager, 'create_backup') as mock_backup:
                    mock_backup.return_value = {'backup_id': f'backup_{index}', 'status': 'success'}
                    result = self.manager.create_backup({'index': index}, f'backup_{index}')
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=backup_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        self.assertEqual(len(results), 5)


class TestDataManagerCleanup(unittest.TestCase):
    """Test DataManager cleanup and shutdown functionality"""
    
    def setUp(self):
        self.config = {}
        
        with patch('production_data.backup_manager.BackupManager'), \
             patch('production_data.encryption_manager.EncryptionManager'), \
             patch('production_data.compression_manager.CompressionManager'), \
             patch('production_data.storage_manager.StorageManager'), \
             patch('production_data.replication_manager.ReplicationManager'), \
             patch('production_data.data_metrics.DataMetricsCollector'):
            
            self.manager = DataManager(self.config)
    
    def test_shutdown(self):
        """Test proper shutdown"""
        self.assertTrue(self.manager.is_running)
        
        self.manager.shutdown()
        
        self.assertFalse(self.manager.is_running)
    
    def test_cleanup_resources(self):
        """Test resource cleanup"""
        with patch.object(self.manager, 'cleanup_resources') as mock_cleanup:
            mock_cleanup.return_value = {'status': 'cleaned', 'resources_freed': 5}
            
            result = self.manager.cleanup_resources()
            
            mock_cleanup.assert_called_once()
            self.assertEqual(result['status'], 'cleaned')


class TestDataManagerEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        self.config = {}
        
        with patch('production_data.backup_manager.BackupManager'), \
             patch('production_data.encryption_manager.EncryptionManager'), \
             patch('production_data.compression_manager.CompressionManager'), \
             patch('production_data.storage_manager.StorageManager'), \
             patch('production_data.replication_manager.ReplicationManager'), \
             patch('production_data.data_metrics.DataMetricsCollector'):
            
            self.manager = DataManager(self.config)
    
    def tearDown(self):
        if hasattr(self, 'manager'):
            self.manager.shutdown()
    
    def test_large_data_handling(self):
        """Test handling of large data sets"""
        large_data = {'data': 'x' * 1000000}  # 1MB of data
        
        with patch.object(self.manager.storage_manager, 'store_data') as mock_store:
            mock_store.return_value = {'status': 'stored', 'size': len(str(large_data))}
            
            result = self.manager.store_data(large_data, 'large_data_test')
            
            mock_store.assert_called_once_with(large_data, 'large_data_test')
            self.assertEqual(result['status'], 'stored')
    
    def test_invalid_data_types(self):
        """Test handling of invalid data types"""
        invalid_data = object()  # Non-serializable object
        
        with patch.object(self.manager.storage_manager, 'store_data') as mock_store:
            mock_store.side_effect = TypeError("Object not serializable")
            
            with self.assertRaises(TypeError):
                self.manager.store_data(invalid_data, 'invalid_data_test')
    
    def test_network_failure_simulation(self):
        """Test behavior during simulated network failures"""
        with patch.object(self.manager.replication_manager, 'sync_replicas') as mock_sync:
            mock_sync.side_effect = ConnectionError("Network unreachable")
            
            with self.assertRaises(ConnectionError):
                self.manager.sync_replicas()
    
    def test_storage_full_simulation(self):
        """Test behavior when storage is full"""
        test_data = {'key': 'value'}
        
        with patch.object(self.manager.storage_manager, 'store_data') as mock_store:
            mock_store.side_effect = OSError("No space left on device")
            
            with self.assertRaises(OSError):
                self.manager.store_data(test_data, 'storage_full_test')


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)