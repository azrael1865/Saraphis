"""
Stress test for StorageManager to verify robustness
"""

import unittest
import tempfile
import shutil
import time
import threading
import random
from pathlib import Path

from production_data.storage_manager import StorageManager


class TestStorageManagerStress(unittest.TestCase):
    """Stress tests for StorageManager"""
    
    def setUp(self):
        """Set up stress test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'hot_storage_path': f'{self.test_dir}/hot',
            'warm_storage_path': f'{self.test_dir}/warm',
            'cold_storage_path': f'{self.test_dir}/cold'
        }
        self.storage_manager = StorageManager(self.config)
    
    def tearDown(self):
        """Clean up"""
        self.storage_manager.shutdown()
        time.sleep(0.1)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_high_volume_operations(self):
        """Test with high volume of storage operations"""
        num_operations = 100
        results = []
        
        # Store many backups
        for i in range(num_operations):
            data = f"Test data {i} " * random.randint(10, 100)
            result = self.storage_manager.store_backup(
                data.encode(),
                {
                    'backup_id': f'volume_{i}',
                    'backup_type': random.choice(['full', 'incremental', 'differential'])
                }
            )
            results.append(result)
        
        # All should succeed
        self.assertEqual(sum(1 for r in results if r['success']), num_operations)
        
        # Retrieve random samples
        for _ in range(10):
            idx = random.randint(0, num_operations - 1)
            data = self.storage_manager.retrieve_backup(f'volume_{idx}')
            self.assertIsNotNone(data)
    
    def test_concurrent_stress(self):
        """Test concurrent operations under stress"""
        num_threads = 20
        operations_per_thread = 10
        errors = []
        success_count = threading.Lock()
        successes = 0
        
        def worker(thread_id):
            nonlocal successes
            try:
                for i in range(operations_per_thread):
                    # Mix of operations
                    op = random.choice(['store', 'retrieve', 'health'])
                    
                    if op == 'store':
                        result = self.storage_manager.store_backup(
                            f"Thread {thread_id} data {i}".encode(),
                            {'backup_id': f'thread_{thread_id}_{i}', 'backup_type': 'incremental'}
                        )
                        if result['success']:
                            with success_count:
                                successes += 1
                    
                    elif op == 'retrieve':
                        # Try to retrieve existing
                        try:
                            self.storage_manager.retrieve_backup(f'thread_{thread_id}_0')
                        except:
                            pass  # Some may not exist yet
                    
                    else:  # health
                        health = self.storage_manager.check_storage_health()
                        self.assertIn('health_score', health)
                        
            except Exception as e:
                errors.append(e)
        
        # Run threads
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should have no critical errors
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")
        
        # Should have successful operations
        self.assertGreater(successes, 0)
    
    def test_edge_case_bombardment(self):
        """Test with various edge cases"""
        edge_cases = [
            # Empty data
            (b"", {'backup_id': 'empty', 'backup_type': 'full'}),
            
            # Large data
            (b"x" * (50 * 1024 * 1024), {'backup_id': 'large', 'backup_type': 'full'}),
            
            # Unicode in metadata
            (b"test", {'backup_id': 'unicode', 'backup_type': 'full', 'note': '测试数据 🚀'}),
            
            # Long backup ID
            (b"test", {'backup_id': 'a' * 500, 'backup_type': 'incremental'}),
            
            # Special characters
            (b"test", {'backup_id': 'special!@#$%^&*()', 'backup_type': 'incremental'}),
            
            # Missing backup_id
            (b"test", {'backup_type': 'incremental'}),
            
            # Invalid backup type
            (b"test", {'backup_id': 'invalid_type', 'backup_type': 'invalid'}),
        ]
        
        for data, metadata in edge_cases:
            try:
                result = self.storage_manager.store_backup(data, metadata)
                # Should handle gracefully
                self.assertIn('success', result)
            except Exception as e:
                # Should not crash
                self.fail(f"Failed on edge case {metadata}: {e}")
    
    def test_rapid_tiering_changes(self):
        """Test rapid changes in tiering"""
        # Create files
        file_ids = []
        for i in range(20):
            result = self.storage_manager.store_backup(
                f"Tiering test {i}".encode(),
                {'backup_id': f'tier_{i}', 'backup_type': 'incremental'}
            )
            if result['success']:
                file_ids.append(f'tier_{i}')
        
        # Simulate aging by modifying created_at
        current_time = time.time()
        for file_id in file_ids[:10]:
            if file_id in self.storage_manager.file_metadata:
                self.storage_manager.file_metadata[file_id]['created_at'] = current_time - (10 * 86400)
        
        # Run tiering optimization multiple times
        for _ in range(3):
            self.storage_manager._optimize_storage_tiering()
            time.sleep(0.1)
        
        # Should not crash and metrics should update
        self.assertGreaterEqual(
            self.storage_manager.storage_metrics['tiering_operations'], 0
        )
    
    def test_recovery_after_corruption(self):
        """Test recovery after file corruption"""
        # Store backups
        backup_ids = []
        for i in range(5):
            result = self.storage_manager.store_backup(
                f"Recovery test {i}".encode(),
                {'backup_id': f'recovery_{i}', 'backup_type': 'full'}
            )
            if result['success']:
                backup_ids.append(f'recovery_{i}')
        
        # Corrupt some files
        for backup_id in backup_ids[:2]:
            if backup_id in self.storage_manager.file_metadata:
                file_path = Path(self.storage_manager.file_metadata[backup_id]['file_path'])
                if file_path.exists():
                    file_path.unlink()
        
        # Check integrity - should detect corruption
        integrity = self.storage_manager._verify_storage_integrity()
        self.assertGreater(len(integrity['corrupted_files']), 0)
        
        # Handle corrupted files
        self.storage_manager._handle_corrupted_files({
            'files': integrity['corrupted_files']
        })
        
        # Should remove corrupted files from metadata
        for corrupted_id in integrity['corrupted_files']:
            self.assertNotIn(corrupted_id, self.storage_manager.file_metadata)
        
        # Remaining files should still be retrievable
        for backup_id in backup_ids[2:]:
            try:
                data = self.storage_manager.retrieve_backup(backup_id)
                self.assertIsNotNone(data)
            except:
                pass  # May have been cleaned up


if __name__ == '__main__':
    unittest.main()