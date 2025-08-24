"""
Saraphis Production Data Management & Backup System
Production-ready data lifecycle management and protection
NO FALLBACKS - HARD FAILURES ONLY
"""

# Import all modules - will fail hard if dependencies missing
from .data_manager import DataManager, create_data_manager
from .backup_manager import BackupManager
from .encryption_manager import EncryptionManager, KeyManager, EncryptionEngine
from .compression_manager import CompressionManager  # Requires lz4, brotli - WILL FAIL HARD
from .storage_manager import StorageManager
from .replication_manager import ReplicationManager
from .data_metrics import DataMetricsCollector

__all__ = [
    'DataManager',
    'create_data_manager',
    'BackupManager',
    'EncryptionManager',
    'KeyManager',
    'EncryptionEngine',
    'CompressionManager',
    'StorageManager',
    'ReplicationManager',
    'DataMetricsCollector'
]