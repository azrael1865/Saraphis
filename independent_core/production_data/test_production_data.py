#!/usr/bin/env python3
"""
Test script for Saraphis Production Data Management System
"""

import logging
import time
import json
from datetime import datetime
import os

# Import all production data components
from production_data import (
    DataManager,
    BackupManager,
    EncryptionManager,
    CompressionManager,
    StorageManager,
    ReplicationManager,
    DataMetricsCollector
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_backup_system():
    """Test backup and restore functionality"""
    print("\n" + "="*50)
    print("Testing Backup System")
    print("="*50)
    
    config = {
        'backup_directory': '/tmp/saraphis_backups',
        'retention_days': 30,
        'compression_enabled': True,
        'encryption_enabled': True
    }
    
    backup_manager = BackupManager(config)
    
    # Test data
    test_data = {
        'source_path': '/tmp/test_source',
        'data': b'This is test data for backup system' * 1000,
        'metadata': {
            'type': 'test_data',
            'created': datetime.now().isoformat()
        }
    }
    
    # Create backup
    print("\n1. Creating backup...")
    backup_result = backup_manager.create_backup(
        source_path=test_data['source_path'],
        backup_type='full',
        metadata=test_data['metadata']
    )
    
    print(f"Backup Result: {json.dumps(backup_result, indent=2)}")
    
    if backup_result['success']:
        backup_id = backup_result['backup_id']
        
        # Test restore
        print("\n2. Testing restore...")
        restore_result = backup_manager.restore_backup(
            backup_id=backup_id,
            target_path='/tmp/test_restore'
        )
        
        print(f"Restore Result: {json.dumps(restore_result, indent=2)}")
        
        # Validate backup
        print("\n3. Validating backup integrity...")
        validation_result = backup_manager.validate_backup_integrity(backup_id)
        print(f"Validation Result: {json.dumps(validation_result, indent=2)}")
    
    # Generate report
    print("\n4. Generating backup report...")
    report = backup_manager.generate_backup_report()
    print(f"Backup Report Summary:")
    print(f"  - Total Backups: {report['backup_statistics']['total_backups']}")
    print(f"  - Success Rate: {report['backup_statistics']['success_rate']:.2%}")
    print(f"  - Storage Used: {report['storage_analysis']['total_size_gb']:.2f} GB")


def test_encryption_system():
    """Test encryption and key management"""
    print("\n" + "="*50)
    print("Testing Encryption System")
    print("="*50)
    
    config = {
        'key_config': {
            'key_rotation_days': 90,
            'key_algorithm': 'AES-256'
        },
        'engine_config': {
            'algorithm': 'AES-256-GCM'
        }
    }
    
    encryption_manager = EncryptionManager(config)
    
    # Test data
    test_data = b'This is sensitive data that needs encryption!' * 100
    
    # Encrypt data
    print("\n1. Encrypting data...")
    encrypted_data = encryption_manager.encrypt_data(test_data)
    print(f"Original size: {len(test_data)} bytes")
    print(f"Encrypted size: {len(encrypted_data)} bytes")
    
    # Decrypt data
    print("\n2. Decrypting data...")
    decrypted_data = encryption_manager.decrypt_data(encrypted_data)
    print(f"Decryption successful: {decrypted_data == test_data}")
    
    # Validate encryption status
    print("\n3. Validating encryption status...")
    status = encryption_manager.validate_encryption_status()
    print(f"Encryption Status:")
    print(f"  - Encryption Score: {status['encryption_score']:.2f}")
    print(f"  - Coverage: {status['encryption_coverage']['coverage_percentage']:.2%}")
    print(f"  - Key Integrity: {status['key_integrity']['integrity_score']:.2f}")
    
    # Generate report
    print("\n4. Generating encryption report...")
    report = encryption_manager.generate_encryption_report()
    print(f"Encryption Report Summary:")
    print(f"  - Total Operations: {report['encryption_statistics']['total_operations']}")
    print(f"  - Success Rate: {report['encryption_statistics']['success_rate']:.2%}")
    print(f"  - Policy Compliance: {report['policy_compliance']['overall_compliance']}")


def test_compression_system():
    """Test compression optimization"""
    print("\n" + "="*50)
    print("Testing Compression System")
    print("="*50)
    
    config = {
        'compression_algorithm': 'LZ4',
        'compression_level': 6,
        'adaptive_compression': True,
        'min_compression_ratio': 0.5
    }
    
    compression_manager = CompressionManager(config)
    
    # Test different data types
    test_cases = [
        {
            'name': 'Text data',
            'data': b'Lorem ipsum dolor sit amet, consectetur adipiscing elit. ' * 1000
        },
        {
            'name': 'Binary data',
            'data': os.urandom(10000)
        },
        {
            'name': 'Repetitive data',
            'data': b'AAAAAAAAAA' * 1000
        }
    ]
    
    for test_case in test_cases:
        print(f"\n1. Compressing {test_case['name']}...")
        
        # Compress
        compressed_data = compression_manager.compress_data(test_case['data'])
        
        # Calculate ratio
        compression_ratio = len(compressed_data) / len(test_case['data'])
        
        print(f"  - Original size: {len(test_case['data'])} bytes")
        print(f"  - Compressed size: {len(compressed_data)} bytes")
        print(f"  - Compression ratio: {compression_ratio:.2f}")
        
        # Decompress
        decompressed_data = compression_manager.decompress_data(compressed_data)
        print(f"  - Decompression successful: {decompressed_data == test_case['data']}")
    
    # Validate integrity
    print("\n2. Validating compression integrity...")
    integrity = compression_manager.validate_compression_integrity()
    print(f"Compression Integrity:")
    print(f"  - Integrity Score: {integrity['integrity_score']:.2f}")
    print(f"  - All Tests Passed: {integrity['test_results']['all_tests_passed']}")
    
    # Generate report
    print("\n3. Generating compression report...")
    report = compression_manager.generate_compression_report()
    print(f"Compression Report Summary:")
    print(f"  - Average Ratio: {report['compression_statistics']['average_ratio']:.2f}")
    print(f"  - Space Saved: {report['savings_analysis']['total_space_saved_gb']:.2f} GB")


def test_storage_system():
    """Test multi-tier storage management"""
    print("\n" + "="*50)
    print("Testing Storage System")
    print("="*50)
    
    config = {
        'storage_tiers': {
            'hot': {'path': '/tmp/hot_storage', 'max_size_gb': 100},
            'warm': {'path': '/tmp/warm_storage', 'max_size_gb': 500},
            'cold': {'path': '/tmp/cold_storage', 'max_size_gb': 1000}
        },
        'tiering_policy': {
            'hot_to_warm_days': 7,
            'warm_to_cold_days': 30
        }
    }
    
    storage_manager = StorageManager(config)
    
    # Store backup
    print("\n1. Storing backup...")
    backup_id = f"backup_{int(time.time())}"
    test_data = b'This is backup data' * 1000
    
    store_result = storage_manager.store_backup(
        backup_id=backup_id,
        backup_data=test_data,
        metadata={'type': 'test_backup'}
    )
    
    print(f"Store Result: {json.dumps(store_result, indent=2)}")
    
    # Retrieve backup
    print("\n2. Retrieving backup...")
    retrieve_result = storage_manager.retrieve_backup(backup_id)
    print(f"Retrieve Result:")
    print(f"  - Success: {retrieve_result['success']}")
    print(f"  - Storage Tier: {retrieve_result.get('storage_tier', 'N/A')}")
    
    # Check storage health
    print("\n3. Checking storage health...")
    health = storage_manager.check_storage_health()
    print(f"Storage Health:")
    print(f"  - Health Score: {health['health_score']:.2f}")
    print(f"  - Total Capacity: {health['total_capacity_gb']:.2f} GB")
    print(f"  - Used Capacity: {health['used_capacity_gb']:.2f} GB")
    
    # Generate report
    print("\n4. Generating storage report...")
    report = storage_manager.generate_storage_report()
    print(f"Storage Report Summary:")
    print(f"  - Total Items: {report['storage_statistics']['total_items']}")
    print(f"  - Storage Efficiency: {report['storage_efficiency']['space_efficiency']:.2%}")


def test_replication_system():
    """Test data replication and disaster recovery"""
    print("\n" + "="*50)
    print("Testing Replication System")
    print("="*50)
    
    config = {
        'replication_factor': 3,
        'replication_lag_threshold_seconds': 300,
        'cross_region_replication': True,
        'disaster_recovery_rpo': 3600,
        'disaster_recovery_rto': 7200
    }
    
    replication_manager = ReplicationManager(config)
    
    # Replicate data
    print("\n1. Replicating data...")
    data_id = f"data_{int(time.time())}"
    test_data = b'Critical data for replication' * 1000
    
    replication_result = replication_manager.replicate_data(
        data_id=data_id,
        data=test_data,
        metadata={'priority': 'high'}
    )
    
    print(f"Replication Result: {json.dumps(replication_result, indent=2)}")
    
    # Validate replication
    print("\n2. Validating replication status...")
    status = replication_manager.validate_replication_status()
    print(f"Replication Status:")
    print(f"  - Replication Score: {status['replication_score']:.2f}")
    print(f"  - Active Nodes: {status['active_nodes']}/{status['total_nodes']}")
    print(f"  - Average Lag: {status['replication_lag_seconds']:.2f} seconds")
    print(f"  - RPO Compliance: {status['rpo_compliance']['rpo_met']}")
    
    # Sync replication
    print("\n3. Syncing replication...")
    sync_result = replication_manager.sync_replication()
    print(f"Sync Result: {json.dumps(sync_result, indent=2)}")
    
    # Generate report
    print("\n4. Generating replication report...")
    report = replication_manager.generate_replication_report()
    print(f"Replication Report Summary:")
    print(f"  - DR Ready: {report['disaster_recovery_readiness']['dr_ready']}")
    print(f"  - DR Score: {report['disaster_recovery_readiness']['dr_score']:.2f}")
    print(f"  - Cross-Region: {report['disaster_recovery_readiness']['cross_region_replication']}")


def test_data_metrics():
    """Test data metrics collection"""
    print("\n" + "="*50)
    print("Testing Data Metrics System")
    print("="*50)
    
    config = {
        'backup_time_threshold': 300,
        'restore_time_threshold': 600,
        'compression_ratio_threshold': 0.5,
        'encryption_time_threshold': 100,
        'replication_lag_threshold': 300,
        'storage_usage_threshold': 80
    }
    
    metrics_collector = DataMetricsCollector(config)
    
    # Record various metrics
    print("\n1. Recording metrics...")
    
    # Backup metrics
    metrics_collector.record_metric('backup', 'backup_time_seconds', 250)
    metrics_collector.record_metric('backup', 'success_rate', 0.98)
    
    # Compression metrics
    metrics_collector.record_metric('compression', 'average_compression_ratio', 0.45)
    
    # Storage metrics
    metrics_collector.record_metric('storage', 'usage_percentage', 75)
    
    # Replication metrics
    metrics_collector.record_metric('replication', 'health_score', 0.95)
    
    time.sleep(1)  # Allow metrics to be processed
    
    # Get current metrics
    print("\n2. Current metrics...")
    current = metrics_collector.get_current_metrics()
    print(f"System Health Score: {current['system_health']['health_score']:.2f}")
    print(f"Health Status: {current['system_health']['status']}")
    
    # Analyze metrics
    print("\n3. Analyzing metrics...")
    analysis = metrics_collector.analyze_metrics()
    print(f"Active Alerts: {len(analysis['active_alerts'])}")
    if analysis.get('recommendations'):
        print("Recommendations:")
        for rec in analysis['recommendations'][:3]:
            print(f"  - {rec}")
    
    # Generate report
    print("\n4. Generating performance report...")
    report = metrics_collector.generate_performance_report()
    print(f"Performance Report Summary:")
    print(f"  - Overall Performance: {report['performance_scores'].get('overall_performance', 0):.2f}")
    print(f"  - SLA Compliance: {report['sla_compliance']['compliance_percentage']:.1f}%")


def test_integrated_system():
    """Test integrated data management system"""
    print("\n" + "="*50)
    print("Testing Integrated Data Management System")
    print("="*50)
    
    # Initialize data manager with all components
    config = {
        'backup_config': {
            'backup_directory': '/tmp/saraphis_backups',
            'retention_days': 30
        },
        'encryption_config': {
            'key_config': {'key_rotation_days': 90}
        },
        'compression_config': {
            'adaptive_compression': True
        },
        'storage_config': {
            'storage_tiers': {
                'hot': {'path': '/tmp/hot_storage', 'max_size_gb': 100}
            }
        },
        'replication_config': {
            'replication_factor': 3
        },
        'metrics_config': {}
    }
    
    data_manager = DataManager(config)
    
    # Process data through full pipeline
    print("\n1. Processing data through full pipeline...")
    test_data = b'Important data for full pipeline test' * 1000
    
    process_result = data_manager.process_data(
        data=test_data,
        data_id=f"test_{int(time.time())}",
        metadata={'importance': 'high'}
    )
    
    print(f"Process Result: {json.dumps(process_result, indent=2)}")
    
    # Validate data integrity
    print("\n2. Validating data integrity...")
    integrity = data_manager.validate_data_integrity()
    print(f"Data Integrity:")
    print(f"  - Integrity Score: {integrity['integrity_score']:.2f}")
    print(f"  - All Systems Healthy: {integrity['all_systems_healthy']}")
    
    # Generate comprehensive report
    print("\n3. Generating comprehensive report...")
    report = data_manager.generate_data_report()
    print(f"\nComprehensive Data Management Report:")
    print(f"  - Overall Health Score: {report['overall_health_score']:.2f}")
    print(f"  - System Status: {report['system_status']}")
    print(f"  - Active Issues: {report['active_issues']}")
    
    if report.get('recommendations'):
        print("\nTop Recommendations:")
        for i, rec in enumerate(report['recommendations'][:5], 1):
            print(f"  {i}. {rec}")
    
    # Shutdown
    print("\n4. Shutting down data manager...")
    data_manager.shutdown()
    print("Data manager shutdown complete")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("SARAPHIS PRODUCTION DATA MANAGEMENT SYSTEM TEST")
    print("="*70)
    
    try:
        # Test individual components
        test_backup_system()
        test_encryption_system()
        test_compression_system()
        test_storage_system()
        test_replication_system()
        test_data_metrics()
        
        # Test integrated system
        test_integrated_system()
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\nERROR: Test failed - {e}")


if __name__ == "__main__":
    main()