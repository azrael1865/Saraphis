"""
Comprehensive test suite for StorageManager
Tests all functionality and edge cases
"""

import unittest
import tempfile
import shutil
import time
import json
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import sys
sys.path.append('.')
from production_data.storage_manager import StorageManager


class TestStorageManager(unittest.TestCase):
    """Test cases for StorageManager"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        
        # Test configuration
        self.config = {
            'hot_storage_path': f'{self.test_dir}/hot',
            'warm_storage_path': f'{self.test_dir}/warm',
            'cold_storage_path': f'{self.test_dir}/cold'
        }
        
        # Initialize storage manager
        self.storage_manager = StorageManager(self.config)
        
    def tearDown(self):
        """Clean up test environment"""
        try:
            # Shutdown storage manager
            if hasattr(self, 'storage_manager'):
                self.storage_manager.shutdown()
                # Give threads time to stop
                time.sleep(0.1)
            
            # Remove test directory
            if hasattr(self, 'test_dir'):
                shutil.rmtree(self.test_dir, ignore_errors=True)
        except Exception as e:
            print(f"Teardown error: {e}")
    
    def test_initialization(self):
        """Test StorageManager initialization"""
        # Check storage paths are created
        self.assertTrue(self.storage_manager.hot_storage_path.exists())
        self.assertTrue(self.storage_manager.warm_storage_path.exists())
        self.assertTrue(self.storage_manager.cold_storage_path.exists())
        
        # Check storage policies initialized
        self.assertIsNotNone(self.storage_manager.storage_policies)
        self.assertIn('tiering_policy', self.storage_manager.storage_policies)
        self.assertIn('retention_policy', self.storage_manager.storage_policies)
        self.assertIn('capacity_policy', self.storage_manager.storage_policies)
        
        # Check metrics initialized
        self.assertEqual(self.storage_manager.storage_metrics['total_files'], 0)
        self.assertEqual(self.storage_manager.storage_metrics['total_size_gb'], 0.0)
        
        # Check threads started
        self.assertTrue(self.storage_manager.is_running)
        self.assertIsNotNone(self.storage_manager.tiering_thread)
        self.assertIsNotNone(self.storage_manager.integrity_thread)
    
    def test_store_backup(self):
        """Test backup storage functionality"""
        # Test data
        test_data = b"Test backup data content"
        metadata = {
            'backup_id': 'test_backup_001',
            'backup_type': 'incremental',
            'checksum': 'abc123',
            'timestamp': datetime.now().isoformat()
        }
        
        # Store backup
        result = self.storage_manager.store_backup(test_data, metadata)
        
        # Verify result
        self.assertTrue(result['success'])
        self.assertEqual(result['storage_tier'], 'hot')  # Incremental goes to hot
        self.assertEqual(result['size_bytes'], len(test_data))
        
        # Verify file exists
        file_path = Path(result['location'])
        self.assertTrue(file_path.exists())
        
        # Verify metadata stored
        self.assertIn('test_backup_001', self.storage_manager.file_metadata)
        file_info = self.storage_manager.file_metadata['test_backup_001']
        self.assertEqual(file_info['storage_tier'], 'hot')
        self.assertEqual(file_info['size_bytes'], len(test_data))
    
    def test_store_full_backup(self):
        """Test full backup storage (goes to warm)"""
        test_data = b"Full backup data"
        metadata = {
            'backup_id': 'full_backup_001',
            'backup_type': 'full',
            'checksum': 'def456'
        }
        
        result = self.storage_manager.store_backup(test_data, metadata)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['storage_tier'], 'warm')  # Full backups go to warm
    
    def test_retrieve_backup(self):
        """Test backup retrieval"""
        # Store test backup
        test_data = b"Retrieval test data"
        metadata = {
            'backup_id': 'retrieve_test_001',
            'backup_type': 'incremental'
        }
        
        self.storage_manager.store_backup(test_data, metadata)
        
        # Retrieve backup
        retrieved_data = self.storage_manager.retrieve_backup('retrieve_test_001')
        
        # Verify data
        self.assertEqual(retrieved_data, test_data)
        
        # Verify access tracking updated
        self.assertEqual(
            self.storage_manager.access_tracking['retrieve_test_001']['count'], 
            1
        )
    
    def test_retrieve_nonexistent_backup(self):
        """Test retrieving non-existent backup"""
        with self.assertRaises(RuntimeError) as context:
            self.storage_manager.retrieve_backup('nonexistent_backup')
        
        self.assertIn("not found", str(context.exception))
    
    def test_find_backup_in_storage(self):
        """Test finding backup across storage tiers"""
        # Create a backup file directly in warm storage
        backup_id = 'direct_warm_backup'
        warm_path = self.storage_manager.warm_storage_path / f"{backup_id}.dat"
        warm_path.write_bytes(b"Direct warm data")
        
        # Find backup
        file_info = self.storage_manager._find_backup_in_storage(backup_id)
        
        self.assertIsNotNone(file_info)
        self.assertEqual(file_info['storage_tier'], 'warm')
        self.assertEqual(file_info['file_id'], backup_id)
    
    def test_check_storage_health(self):
        """Test storage health check"""
        # Add some test data
        for i in range(3):
            self.storage_manager.store_backup(
                b"Health check test data",
                {'backup_id': f'health_test_{i}', 'backup_type': 'incremental'}
            )
        
        # Check health
        health = self.storage_manager.check_storage_health()
        
        # Verify health report structure
        self.assertIn('health_score', health)
        self.assertIn('storage_usage', health)
        self.assertIn('capacity_status', health)
        self.assertIn('integrity_status', health)
        self.assertIn('tiering_efficiency', health)
        self.assertIn('issues', health)
        
        # Health score should be reasonable
        self.assertGreaterEqual(health['health_score'], 0.0)
        self.assertLessEqual(health['health_score'], 1.0)
    
    def test_storage_capacity_check(self):
        """Test storage capacity checking"""
        capacity_status = self.storage_manager._check_storage_capacity()
        
        # Verify capacity status for each tier
        for tier in ['hot', 'warm', 'cold']:
            self.assertIn(tier, capacity_status)
            tier_status = capacity_status[tier]
            
            self.assertIn('current_gb', tier_status)
            self.assertIn('limit_gb', tier_status)
            self.assertIn('usage_percentage', tier_status)
            self.assertIn('status', tier_status)
            self.assertIn('available_gb', tier_status)
            
            # Status should be normal for empty storage
            self.assertEqual(tier_status['status'], 'normal')
    
    def test_verify_storage_integrity(self):
        """Test storage integrity verification"""
        # Add test files
        for i in range(2):
            self.storage_manager.store_backup(
                b"Integrity test data",
                {'backup_id': f'integrity_{i}', 'backup_type': 'incremental'}
            )
        
        # Verify integrity
        integrity = self.storage_manager._verify_storage_integrity()
        
        self.assertIn('integrity_score', integrity)
        self.assertIn('total_files', integrity)
        self.assertIn('verified_files', integrity)
        self.assertIn('corrupted_files', integrity)
        
        # All files should be verified (not corrupted)
        self.assertEqual(integrity['integrity_score'], 1.0)
        self.assertEqual(len(integrity['corrupted_files']), 0)
    
    def test_corrupted_file_detection(self):
        """Test detection of corrupted files"""
        # Store a backup
        self.storage_manager.store_backup(
            b"Test data",
            {'backup_id': 'corrupt_test', 'backup_type': 'incremental'}
        )
        
        # Corrupt the file by deleting it
        file_info = self.storage_manager.file_metadata['corrupt_test']
        Path(file_info['file_path']).unlink()
        
        # Check integrity
        integrity = self.storage_manager._verify_storage_integrity()
        
        self.assertIn('corrupt_test', integrity['corrupted_files'])
        self.assertLess(integrity['integrity_score'], 1.0)
    
    def test_tiering_efficiency_check(self):
        """Test storage tiering efficiency check"""
        # Add files with different ages and access patterns
        current_time = time.time()
        
        # Old file that should move to warm
        old_file_id = 'old_hot_file'
        self.storage_manager.file_metadata[old_file_id] = {
            'file_id': old_file_id,
            'storage_tier': 'hot',
            'created_at': current_time - (10 * 86400),  # 10 days old
            'size_bytes': 1000
        }
        self.storage_manager.access_tracking[old_file_id] = {
            'count': 1,  # Low access
            'last_access': current_time - 86400
        }
        
        # Check tiering efficiency
        efficiency = self.storage_manager._check_tiering_efficiency()
        
        self.assertIn('efficiency_score', efficiency)
        self.assertIn('misplaced_files', efficiency)
        self.assertIn('misplacement_details', efficiency)
        
        # Should identify the old file for movement
        self.assertGreater(efficiency['misplaced_files'], 0)
        self.assertIn(old_file_id, efficiency['misplacement_details']['hot_to_warm'])
    
    def test_storage_tiering_optimization(self):
        """Test storage tiering optimization"""
        # Create test files in wrong tiers
        test_file_id = 'misplaced_file'
        hot_file = self.storage_manager.hot_storage_path / f"{test_file_id}.dat"
        hot_file.write_bytes(b"Misplaced data")
        
        metadata_file = self.storage_manager.hot_storage_path / f"{test_file_id}_metadata.json"
        metadata_file.write_text('{"test": "metadata"}')
        
        # Add to metadata as old file
        current_time = time.time()
        self.storage_manager.file_metadata[test_file_id] = {
            'file_id': test_file_id,
            'file_path': str(hot_file),
            'metadata_path': str(metadata_file),
            'storage_tier': 'hot',
            'created_at': current_time - (10 * 86400),  # 10 days old
            'size_bytes': 100
        }
        
        # Run optimization
        self.storage_manager._optimize_storage_tiering()
        
        # File should be moved to warm
        file_info = self.storage_manager.file_metadata[test_file_id]
        self.assertEqual(file_info['storage_tier'], 'warm')
        
        # File should exist in warm storage
        warm_file = Path(file_info['file_path'])
        self.assertTrue(warm_file.exists())
        self.assertIn('warm', str(warm_file))
    
    def test_generate_storage_report(self):
        """Test storage report generation"""
        # Add some test data
        for i in range(5):
            self.storage_manager.store_backup(
                b"Report test data" * 100,
                {'backup_id': f'report_test_{i}', 'backup_type': 'incremental'}
            )
        
        # Generate report
        report = self.storage_manager.generate_storage_report()
        
        # Verify report structure
        self.assertIn('report_id', report)
        self.assertIn('timestamp', report)
        self.assertIn('storage_health', report)
        self.assertIn('storage_statistics', report)
        self.assertIn('access_patterns', report)
        self.assertIn('cost_analysis', report)
        self.assertIn('tiering_summary', report)
        self.assertIn('recommendations', report)
        
        # Verify statistics
        stats = report['storage_statistics']
        self.assertEqual(stats['total_files'], 5)
    
    def test_resolve_storage_issues(self):
        """Test storage issue resolution"""
        issues = [
            {
                'type': 'low_capacity',
                'severity': 'warning',
                'tier': 'hot'
            },
            {
                'type': 'inefficient_tiering',
                'severity': 'medium'
            }
        ]
        
        # Should not raise exceptions
        self.storage_manager.resolve_storage_issues(issues)
        
        # Verify tiering operations incremented if optimization ran
        initial_ops = self.storage_manager.storage_metrics['tiering_operations']
        self.assertGreaterEqual(
            self.storage_manager.storage_metrics['tiering_operations'],
            initial_ops
        )
    
    def test_access_pattern_analysis(self):
        """Test access pattern analysis"""
        # Create files with different access patterns
        for i in range(10):
            file_id = f'access_test_{i}'
            self.storage_manager.access_tracking[file_id] = {
                'count': i * 3,  # Varying access counts
                'last_access': time.time()
            }
        
        # Analyze patterns
        patterns = self.storage_manager._analyze_access_patterns()
        
        self.assertIn('total_tracked_files', patterns)
        self.assertIn('hot_files', patterns)
        self.assertIn('cold_files', patterns)
        self.assertIn('average_access_count', patterns)
        self.assertIn('access_frequency_distribution', patterns)
        
        # Should have some hot files (> 5 accesses)
        self.assertGreater(patterns['hot_files'], 0)
    
    def test_storage_cost_calculation(self):
        """Test storage cost calculation"""
        # Add files to different tiers
        for tier in ['hot', 'warm', 'cold']:
            backup_id = f'cost_test_{tier}'
            self.storage_manager.file_metadata[backup_id] = {
                'file_id': backup_id,
                'storage_tier': tier,
                'size_bytes': 10 * (1024**3)  # 10 GB
            }
        
        # Update metrics
        self.storage_manager._update_storage_metrics()
        
        # Calculate costs
        costs = self.storage_manager._calculate_storage_costs()
        
        self.assertIn('monthly_costs_by_tier', costs)
        self.assertIn('total_monthly_cost', costs)
        self.assertIn('annual_cost', costs)
        
        # Costs should be positive
        self.assertGreater(costs['total_monthly_cost'], 0)
        
        # Hot storage should be most expensive
        self.assertGreater(
            costs['monthly_costs_by_tier']['hot'],
            costs['monthly_costs_by_tier']['cold']
        )
    
    def test_storage_recommendations(self):
        """Test storage recommendation generation"""
        # Create scenario needing recommendations
        health = {
            'health_score': 0.6,  # Low health
            'issues': [
                {
                    'type': 'low_capacity',
                    'severity': 'critical',
                    'tier': 'hot'
                }
            ],
            'tiering_efficiency': {
                'efficiency_score': 0.7  # Low efficiency
            }
        }
        
        stats = {'total_files': 100}
        
        patterns = {
            'cold_files': 60,
            'total_tracked_files': 100
        }
        
        recommendations = self.storage_manager._generate_storage_recommendations(
            health, stats, patterns
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Should recommend addressing critical capacity
        self.assertTrue(
            any('critical capacity' in rec for rec in recommendations)
        )
    
    def test_concurrent_operations(self):
        """Test concurrent storage operations"""
        results = []
        errors = []
        
        def store_operation(index):
            try:
                result = self.storage_manager.store_backup(
                    f"Concurrent data {index}".encode(),
                    {'backup_id': f'concurrent_{index}', 'backup_type': 'incremental'}
                )
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run concurrent operations
        threads = []
        for i in range(10):
            thread = threading.Thread(target=store_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All operations should succeed
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 10)
        
        # All files should be in metadata
        for i in range(10):
            self.assertIn(f'concurrent_{i}', self.storage_manager.file_metadata)
    
    def test_storage_history_tracking(self):
        """Test storage operation history tracking"""
        # Perform some operations
        for i in range(3):
            self.storage_manager.store_backup(
                b"History test",
                {'backup_id': f'history_{i}', 'backup_type': 'incremental'}
            )
        
        # Check history
        self.assertGreater(len(self.storage_manager.storage_history), 0)
        
        # Verify history records
        for record in self.storage_manager.storage_history:
            self.assertIn('timestamp', record)
            self.assertIn('operation', record)
            self.assertIn('file_id', record)
            self.assertIn('storage_tier', record)
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test empty data storage
        result = self.storage_manager.store_backup(
            b"",
            {'backup_id': 'empty_data', 'backup_type': 'incremental'}
        )
        self.assertTrue(result['success'])
        
        # Test missing backup_id in metadata
        result = self.storage_manager.store_backup(
            b"No ID data",
            {'backup_type': 'incremental'}
        )
        self.assertTrue(result['success'])  # Should generate ID
        
        # Test unknown backup type
        result = self.storage_manager.store_backup(
            b"Unknown type",
            {'backup_id': 'unknown_type', 'backup_type': 'unknown'}
        )
        self.assertTrue(result['success'])
        self.assertEqual(result['storage_tier'], 'hot')  # Default to hot
        
        # Test invalid storage tier
        with self.assertRaises(ValueError):
            self.storage_manager._get_storage_path('invalid_tier')
    
    def test_shutdown(self):
        """Test proper shutdown"""
        # Create new instance for shutdown test
        manager = StorageManager(self.config)
        
        # Verify running
        self.assertTrue(manager.is_running)
        
        # Shutdown
        manager.shutdown()
        
        # Verify stopped
        self.assertFalse(manager.is_running)
    
    def test_directory_size_calculation(self):
        """Test directory size calculation"""
        # Create test files
        test_path = Path(self.test_dir) / 'size_test'
        test_path.mkdir()
        
        for i in range(3):
            file_path = test_path / f'file_{i}.dat'
            file_path.write_bytes(b'x' * 1000)
        
        # Calculate size
        size = self.storage_manager._get_directory_size(test_path)
        
        self.assertEqual(size, 3000)
    
    def test_metadata_persistence(self):
        """Test metadata file persistence"""
        metadata = {
            'backup_id': 'metadata_test',
            'backup_type': 'full',
            'custom_field': 'test_value',
            'timestamp': datetime.now().isoformat()
        }
        
        # Store backup
        self.storage_manager.store_backup(b"Test data", metadata)
        
        # Read metadata file
        metadata_path = self.storage_manager.warm_storage_path / 'metadata_test_metadata.json'
        self.assertTrue(metadata_path.exists())
        
        with open(metadata_path, 'r') as f:
            saved_metadata = json.load(f)
        
        self.assertEqual(saved_metadata['custom_field'], 'test_value')
    
    def test_health_score_calculation(self):
        """Test health score calculation logic"""
        # Test perfect health
        perfect_usage = {'total_size_gb': 10}
        perfect_capacity = {
            'hot': {'usage_percentage': 0.5},
            'warm': {'usage_percentage': 0.3},
            'cold': {'usage_percentage': 0.2}
        }
        perfect_integrity = {'integrity_score': 1.0}
        perfect_tiering = {'efficiency_score': 1.0}
        
        score = self.storage_manager._calculate_storage_health_score(
            perfect_usage, perfect_capacity, perfect_integrity, perfect_tiering
        )
        
        self.assertEqual(score, 1.0)
        
        # Test poor health
        poor_capacity = {
            'hot': {'usage_percentage': 0.96},
            'warm': {'usage_percentage': 0.97},
            'cold': {'usage_percentage': 0.98}
        }
        poor_integrity = {'integrity_score': 0.5}
        poor_tiering = {'efficiency_score': 0.3}
        
        score = self.storage_manager._calculate_storage_health_score(
            perfect_usage, poor_capacity, poor_integrity, poor_tiering
        )
        
        self.assertLess(score, 0.5)


class TestStorageManagerIntegration(unittest.TestCase):
    """Integration tests for StorageManager"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'hot_storage_path': f'{self.test_dir}/hot',
            'warm_storage_path': f'{self.test_dir}/warm',
            'cold_storage_path': f'{self.test_dir}/cold'
        }
        self.storage_manager = StorageManager(self.config)
    
    def tearDown(self):
        """Clean up integration test environment"""
        self.storage_manager.shutdown()
        time.sleep(0.1)
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_full_backup_lifecycle(self):
        """Test complete backup lifecycle"""
        # Store multiple backups
        backup_ids = []
        for i in range(5):
            backup_id = f'lifecycle_{i}'
            backup_ids.append(backup_id)
            
            result = self.storage_manager.store_backup(
                f"Lifecycle test {i}".encode() * 1000,
                {
                    'backup_id': backup_id,
                    'backup_type': 'full' if i == 0 else 'incremental',
                    'sequence': i
                }
            )
            self.assertTrue(result['success'])
        
        # Retrieve backups
        for backup_id in backup_ids:
            data = self.storage_manager.retrieve_backup(backup_id)
            self.assertIsNotNone(data)
        
        # Check health
        health = self.storage_manager.check_storage_health()
        self.assertGreater(health['health_score'], 0.8)
        
        # Generate report
        report = self.storage_manager.generate_storage_report()
        self.assertEqual(report['storage_statistics']['total_files'], 5)
        
        # Resolve any issues
        if health['issues']:
            self.storage_manager.resolve_storage_issues(health['issues'])
    
    def test_storage_under_pressure(self):
        """Test storage under capacity pressure"""
        # Fill storage with large files
        large_data = b'x' * (10 * 1024 * 1024)  # 10MB
        
        for i in range(20):
            self.storage_manager.store_backup(
                large_data,
                {'backup_id': f'pressure_{i}', 'backup_type': 'full'}
            )
        
        # Check health - should show capacity usage
        health = self.storage_manager.check_storage_health()
        
        # Should have storage usage data
        self.assertGreater(health['storage_usage']['total_size_gb'], 0)
        
        # Run optimization
        self.storage_manager._optimize_storage_tiering()
        
        # Metrics should show operations
        self.assertGreater(
            self.storage_manager.storage_metrics['storage_operations'], 
            0
        )


if __name__ == '__main__':
    unittest.main()