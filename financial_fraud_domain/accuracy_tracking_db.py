"""
Accuracy Tracking Database for Financial Fraud Detection
Comprehensive database system for tracking model accuracy metrics with production features.
Part of the Saraphis recursive methodology for accuracy tracking - Phase 2.
"""

import logging
import sqlite3
import json
import pickle
import gzip
import threading
import time
import queue
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import hashlib
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import traceback

# Import from existing modules
# Try absolute imports first (for direct module imports)
try:
    from enhanced_fraud_core_exceptions import (
        EnhancedFraudException, ValidationError, ConfigurationError,
        ResourceError, DataQualityError, ErrorContext, create_error_context
    )
    from enhanced_fraud_core_monitoring import (
        MonitoringManager, PerformanceMetrics, monitor_performance,
        with_caching, CacheManager
    )
    from accuracy_dataset_manager import TrainValidationTestManager, DatasetMetadata
except ImportError as e:
    # Fallback to relative imports (when imported as part of a package)
    try:
        from enhanced_fraud_core_exceptions import (
            EnhancedFraudException, ValidationError, ConfigurationError,
            ResourceError, DataQualityError, ErrorContext, create_error_context
        )
        from enhanced_fraud_core_monitoring import (
            MonitoringManager, PerformanceMetrics, monitor_performance,
            with_caching, CacheManager
        )
        from accuracy_dataset_manager import TrainValidationTestManager, DatasetMetadata
    except ImportError:
        raise ImportError(
            "Failed to import required modules for AccuracyTrackingDatabase. "
            "Please ensure enhanced_fraud_core_exceptions, enhanced_fraud_core_monitoring, "
            "and accuracy_dataset_manager modules are available."
        ) from e

# Configure logging
logger = logging.getLogger(__name__)

# ======================== CUSTOM EXCEPTIONS ========================

class DatabaseError(EnhancedFraudException):
    """Base exception for database-related errors"""
    pass

class ConnectionPoolError(DatabaseError):
    """Exception raised when connection pool operations fail"""
    pass

class DataIntegrityError(DatabaseError):
    """Exception raised when data integrity is compromised"""
    pass

class QueryError(DatabaseError):
    """Exception raised when database queries fail"""
    pass

# ======================== ENUMS AND CONSTANTS ========================

class MetricType(Enum):
    """Types of accuracy metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    CONFUSION_MATRIX = "confusion_matrix"
    CLASSIFICATION_REPORT = "classification_report"

class DataType(Enum):
    """Types of data used for evaluation"""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    PRODUCTION = "production"

class ModelStatus(Enum):
    """Model version status"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    TESTING = "testing"
    DEPRECATED = "deprecated"

# Database schema version
SCHEMA_VERSION = "2.0.0"

# ======================== DATA STRUCTURES ========================

@dataclass
class ModelVersion:
    """Model version information"""
    model_id: str
    version: str
    created_at: datetime
    model_type: str
    parameters: Dict[str, Any]
    training_dataset_id: Optional[str] = None
    status: ModelStatus = ModelStatus.TESTING
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AccuracyMetric:
    """Accuracy metric data"""
    metric_id: str
    model_id: str
    model_version: str
    data_type: DataType
    metric_type: MetricType
    metric_value: Union[float, Dict[str, Any]]
    timestamp: datetime
    dataset_id: Optional[str] = None
    sample_size: Optional[int] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelPrediction:
    """Individual model prediction"""
    prediction_id: str
    model_id: str
    model_version: str
    transaction_id: str
    predicted_label: int
    actual_label: Optional[int]
    prediction_confidence: float
    timestamp: datetime
    features: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pooling"""
    min_connections: int = 2
    max_connections: int = 10
    connection_timeout: float = 30.0
    max_overflow: int = 5
    pool_recycle: int = 3600  # Recycle connections after 1 hour
    retry_attempts: int = 3
    retry_delay: float = 1.0

@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_path: Path = field(default_factory=lambda: Path("accuracy_tracking.db"))
    backup_path: Path = field(default_factory=lambda: Path("accuracy_tracking_backup"))
    enable_wal: bool = True
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    cache_size: int = -2000  # 2MB cache
    busy_timeout: int = 30000  # 30 seconds
    enable_compression: bool = True
    retention_days: int = 90
    batch_size: int = 1000
    vacuum_interval_hours: int = 24

# ======================== CONNECTION POOL ========================

class ConnectionPool:
    """Thread-safe SQLite connection pool"""
    
    def __init__(self, db_path: Path, config: ConnectionPoolConfig):
        self.db_path = db_path
        self.config = config
        self._connections = queue.Queue(maxsize=config.max_connections)
        self._all_connections = []
        self._lock = threading.Lock()
        self._closed = False
        self._created_connections = 0
        
        # Initialize minimum connections
        for _ in range(config.min_connections):
            self._create_connection()
        
        logger.info(f"Connection pool initialized with {config.min_connections} connections")
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection"""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.config.connection_timeout,
            check_same_thread=False,
            isolation_level=None  # Autocommit mode
        )
        
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA cache_size={-2000}")  # 2MB cache
        conn.execute(f"PRAGMA busy_timeout={30000}")  # 30 seconds
        
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys=ON")
        
        with self._lock:
            self._all_connections.append(conn)
            self._created_connections += 1
        
        return conn
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        if self._closed:
            raise ConnectionPoolError("Connection pool is closed")
        
        connection = None
        try:
            # Try to get connection from pool
            try:
                connection = self._connections.get_nowait()
            except queue.Empty:
                # Create new connection if under limit
                with self._lock:
                    if self._created_connections < self.config.max_connections:
                        connection = self._create_connection()
                    else:
                        # Wait for connection with timeout
                        connection = self._connections.get(
                            timeout=self.config.connection_timeout
                        )
            
            # Test connection
            connection.execute("SELECT 1")
            
            yield connection
            
        except Exception as e:
            logger.error(f"Connection pool error: {e}")
            # Return connection to pool even on error
            if connection and not self._closed:
                try:
                    self._connections.put_nowait(connection)
                except queue.Full:
                    pass
            raise
        else:
            # Return connection to pool
            if connection and not self._closed:
                try:
                    self._connections.put_nowait(connection)
                except queue.Full:
                    # Pool is full, close connection
                    connection.close()
                    with self._lock:
                        self._all_connections.remove(connection)
                        self._created_connections -= 1
    
    def close_all(self):
        """Close all connections in the pool"""
        self._closed = True
        
        # Close all connections
        with self._lock:
            for conn in self._all_connections:
                try:
                    conn.close()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            
            self._all_connections.clear()
            self._created_connections = 0
        
        # Clear queue
        while not self._connections.empty():
            try:
                self._connections.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Connection pool closed")

# ======================== ACCURACY TRACKING DATABASE ========================

class AccuracyTrackingDatabase:
    """
    Comprehensive database system for tracking model accuracy metrics.
    Provides thread-safe operations, connection pooling, and production features.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AccuracyTrackingDatabase with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Load configuration
        self.config = self._load_config(config)
        self.db_config = self.config['database']
        self.pool_config = self.config['connection_pool']
        
        # Initialize components
        self._init_database()
        self._init_connection_pool()
        self._init_cache()
        self._init_monitoring()
        
        # Background tasks
        self._init_background_tasks()
        
        # Statistics tracking
        self.operation_stats = defaultdict(int)
        
        self.logger.info(
            f"AccuracyTrackingDatabase initialized with database at {self.db_config.db_path}"
        )
    
    def _load_config(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load and validate configuration"""
        default_config = {
            'database': DatabaseConfig(),
            'connection_pool': ConnectionPoolConfig(),
            'monitoring': {
                'enabled': True,
                'metrics_interval': 60  # seconds
            },
            'cache': {
                'enabled': True,
                'size': 1000,
                'ttl': 3600  # 1 hour
            }
        }
        
        if config:
            # Merge with defaults
            if 'database' in config:
                db_config = DatabaseConfig()
                # Handle both dictionary and DatabaseConfig object
                if isinstance(config['database'], dict):
                for key, value in config['database'].items():
                    if hasattr(db_config, key):
                        setattr(db_config, key, value)
                elif isinstance(config['database'], DatabaseConfig):
                    db_config = config['database']
                default_config['database'] = db_config
            
            if 'connection_pool' in config:
                pool_config = ConnectionPoolConfig()
                # Handle both dictionary and ConnectionPoolConfig object
                if isinstance(config['connection_pool'], dict):
                for key, value in config['connection_pool'].items():
                    if hasattr(pool_config, key):
                        setattr(pool_config, key, value)
                elif isinstance(config['connection_pool'], ConnectionPoolConfig):
                    pool_config = config['connection_pool']
                default_config['connection_pool'] = pool_config
            
            # Update other configs
            for key in ['monitoring', 'cache']:
                if key in config:
                    default_config[key].update(config[key])
        
        return default_config
    
    def _init_database(self) -> None:
        """Initialize database and create schema"""
        # Ensure database directory exists
        self.db_config.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create database with schema
        conn = sqlite3.connect(str(self.db_config.db_path))
        try:
            self._create_schema(conn)
            self._create_indexes(conn)
            conn.commit()
        finally:
            conn.close()
        
        self.logger.info("Database schema initialized")
    
    def _init_connection_pool(self) -> None:
        """Initialize connection pool"""
        self.connection_pool = ConnectionPool(
            self.db_config.db_path,
            self.pool_config
        )
    
    def _init_cache(self) -> None:
        """Initialize caching system"""
        if self.config['cache']['enabled']:
            from enhanced_fraud_core_monitoring import CacheManager, MonitoringConfig
            
            # Create cache config
            cache_config = MonitoringConfig()
            cache_config.cache_size = self.config['cache']['size']
            cache_config.cache_ttl = self.config['cache']['ttl']
            
            self.cache_manager = CacheManager(cache_config)
        else:
            self.cache_manager = None
    
    def _init_monitoring(self) -> None:
        """Initialize monitoring system"""
        if self.config['monitoring']['enabled']:
            # This would integrate with the monitoring system
            self.monitoring_enabled = True
        else:
            self.monitoring_enabled = False
    
    def _init_background_tasks(self) -> None:
        """Initialize background tasks"""
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._stop_background_tasks = threading.Event()
        
        # Start cleanup task
        self.executor.submit(self._cleanup_task)
        
        # Start vacuum task
        self.executor.submit(self._vacuum_task)
    
    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create database schema"""
        # Schema version table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS schema_version (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Model versions table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS model_versions (
                model_id TEXT PRIMARY KEY,
                version TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                model_type TEXT NOT NULL,
                parameters TEXT,
                training_dataset_id TEXT,
                status TEXT NOT NULL,
                metadata TEXT,
                UNIQUE(model_id, version)
            )
        ''')
        
        # Accuracy metrics table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS accuracy_metrics (
                metric_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                data_type TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                metric_value TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                dataset_id TEXT,
                sample_size INTEGER,
                additional_info TEXT,
                FOREIGN KEY (model_id) REFERENCES model_versions(model_id)
            )
        ''')
        
        # Model predictions table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS model_predictions (
                prediction_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                transaction_id TEXT NOT NULL,
                predicted_label INTEGER NOT NULL,
                actual_label INTEGER,
                prediction_confidence REAL NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                features TEXT,
                metadata TEXT,
                FOREIGN KEY (model_id) REFERENCES model_versions(model_id)
            )
        ''')
        
        # Confusion matrices table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS confusion_matrices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                data_type TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                true_positives INTEGER NOT NULL,
                true_negatives INTEGER NOT NULL,
                false_positives INTEGER NOT NULL,
                false_negatives INTEGER NOT NULL,
                dataset_id TEXT,
                FOREIGN KEY (model_id) REFERENCES model_versions(model_id)
            )
        ''')
        
        # Performance metrics table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                data_type TEXT,
                metadata TEXT,
                FOREIGN KEY (model_id) REFERENCES model_versions(model_id)
            )
        ''')
        
        # Data splits table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS data_splits (
                split_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                split_type TEXT NOT NULL,
                train_size INTEGER,
                val_size INTEGER,
                test_size INTEGER,
                created_at TIMESTAMP NOT NULL,
                configuration TEXT,
                metadata TEXT
            )
        ''')
        
        # Update schema version
        conn.execute(
            "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
            (SCHEMA_VERSION,)
        )
    
    def _create_indexes(self, conn: sqlite3.Connection) -> None:
        """Create database indexes for performance"""
        indexes = [
            # Model versions indexes
            "CREATE INDEX IF NOT EXISTS idx_model_versions_status ON model_versions(status)",
            "CREATE INDEX IF NOT EXISTS idx_model_versions_created_at ON model_versions(created_at)",
            
            # Accuracy metrics indexes
            "CREATE INDEX IF NOT EXISTS idx_accuracy_metrics_model ON accuracy_metrics(model_id, model_version)",
            "CREATE INDEX IF NOT EXISTS idx_accuracy_metrics_timestamp ON accuracy_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_accuracy_metrics_data_type ON accuracy_metrics(data_type)",
            "CREATE INDEX IF NOT EXISTS idx_accuracy_metrics_metric_type ON accuracy_metrics(metric_type)",
            
            # Model predictions indexes
            "CREATE INDEX IF NOT EXISTS idx_predictions_model ON model_predictions(model_id, model_version)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON model_predictions(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_transaction ON model_predictions(transaction_id)",
            
            # Confusion matrices indexes
            "CREATE INDEX IF NOT EXISTS idx_confusion_model ON confusion_matrices(model_id, model_version)",
            "CREATE INDEX IF NOT EXISTS idx_confusion_timestamp ON confusion_matrices(timestamp)",
            
            # Performance metrics indexes
            "CREATE INDEX IF NOT EXISTS idx_performance_model ON performance_metrics(model_id, model_version)",
            "CREATE INDEX IF NOT EXISTS idx_performance_metric ON performance_metrics(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)"
        ]
        
        for index in indexes:
            conn.execute(index)
    
    # ======================== MODEL VERSION OPERATIONS ========================
    
    def create_model_version(
        self,
        model_id: str,
        version: str,
        model_type: str,
        parameters: Dict[str, Any],
        training_dataset_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelVersion:
        """
        Create a new model version.
        
        Args:
            model_id: Unique model identifier
            version: Model version string
            model_type: Type of model
            parameters: Model parameters
            training_dataset_id: ID of training dataset
            metadata: Additional metadata
            
        Returns:
            Created ModelVersion object
        """
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            created_at=datetime.now(),
            model_type=model_type,
            parameters=parameters,
            training_dataset_id=training_dataset_id,
            status=ModelStatus.TESTING,
            metadata=metadata or {}
        )
        
        with self.connection_pool.get_connection() as conn:
            try:
                conn.execute('''
                    INSERT INTO model_versions 
                    (model_id, version, created_at, model_type, parameters, 
                     training_dataset_id, status, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_version.model_id,
                    model_version.version,
                    model_version.created_at.isoformat(),
                    model_version.model_type,
                    json.dumps(model_version.parameters),
                    model_version.training_dataset_id,
                    model_version.status.value,
                    json.dumps(model_version.metadata)
                ))
                conn.commit()
                
                self.operation_stats['model_versions_created'] += 1
                self.logger.info(f"Created model version: {model_id} v{version}")
                
                return model_version
                
            except sqlite3.IntegrityError as e:
                raise DataIntegrityError(
                    f"Model version already exists: {model_id} v{version}",
                    context=create_error_context(
                        component="AccuracyTrackingDatabase",
                        operation="create_model_version"
                    )
                )
    
    def get_model_version(self, model_id: str, version: Optional[str] = None) -> Optional[ModelVersion]:
        """
        Get model version information.
        
        Args:
            model_id: Model identifier
            version: Specific version (if None, gets latest)
            
        Returns:
            ModelVersion object or None
        """
        with self.connection_pool.get_connection() as conn:
            if version:
                cursor = conn.execute(
                    "SELECT * FROM model_versions WHERE model_id = ? AND version = ?",
                    (model_id, version)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM model_versions WHERE model_id = ? ORDER BY created_at DESC LIMIT 1",
                    (model_id,)
                )
            
            row = cursor.fetchone()
            if row:
                return self._row_to_model_version(row)
            return None
    
    def update_model_status(self, model_id: str, version: str, status: ModelStatus) -> bool:
        """
        Update model version status.
        
        Args:
            model_id: Model identifier
            version: Model version
            status: New status
            
        Returns:
            Success status
        """
        with self.connection_pool.get_connection() as conn:
            cursor = conn.execute(
                "UPDATE model_versions SET status = ? WHERE model_id = ? AND version = ?",
                (status.value, model_id, version)
            )
            conn.commit()
            
            if cursor.rowcount > 0:
                self.logger.info(f"Updated model {model_id} v{version} status to {status.value}")
                return True
            return False
    
    # ======================== ACCURACY METRICS OPERATIONS ========================
    
    def record_accuracy_metrics(
        self,
        model_id: str,
        model_version: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        data_type: DataType = DataType.TEST,
        dataset_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record comprehensive accuracy metrics for a model.
        
        Args:
            model_id: Model identifier
            model_version: Model version
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            data_type: Type of data being evaluated
            dataset_id: Associated dataset ID
            
        Returns:
            Dictionary of recorded metrics
        """
        # Validate inputs
        if len(y_true) != len(y_pred):
            raise ValidationError("y_true and y_pred must have the same length")
        
        if y_proba is not None and len(y_proba) != len(y_true):
            raise ValidationError("y_proba must have the same length as y_true")
        
        # Calculate metrics
        metrics = {}
        timestamp = datetime.now()
        sample_size = len(y_true)
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC if probabilities available
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:  # Multi-class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            except Exception as e:
                self.logger.warning(f"Could not calculate ROC AUC: {e}")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        # Store metrics in database
        with self.connection_pool.get_connection() as conn:
            # Store individual metrics
            for metric_name, metric_value in metrics.items():
                if metric_name in ['confusion_matrix', 'classification_report']:
                    # Store as JSON
                    metric_type = MetricType.CONFUSION_MATRIX if metric_name == 'confusion_matrix' else MetricType.CLASSIFICATION_REPORT
                    value_to_store = json.dumps(metric_value)
                else:
                    # Store as float
                    metric_type = MetricType[metric_name.upper()]
                    value_to_store = str(metric_value)
                
                metric_id = self._generate_metric_id()
                
                conn.execute('''
                    INSERT INTO accuracy_metrics
                    (metric_id, model_id, model_version, data_type, metric_type,
                     metric_value, timestamp, dataset_id, sample_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric_id,
                    model_id,
                    model_version,
                    data_type.value,
                    metric_type.value,
                    value_to_store,
                    timestamp.isoformat(),
                    dataset_id,
                    sample_size
                ))
            
            # Store confusion matrix separately for easier querying
            if len(np.unique(y_true)) == 2:  # Binary classification
                tn, fp, fn, tp = cm.ravel()
                conn.execute('''
                    INSERT INTO confusion_matrices
                    (model_id, model_version, data_type, timestamp,
                     true_positives, true_negatives, false_positives, false_negatives, dataset_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_id,
                    model_version,
                    data_type.value,
                    timestamp.isoformat(),
                    int(tp), int(tn), int(fp), int(fn),
                    dataset_id
                ))
            
            conn.commit()
        
        self.operation_stats['metrics_recorded'] += 1
        self.logger.info(
            f"Recorded accuracy metrics for model {model_id} v{model_version} "
            f"on {data_type.value} data (n={sample_size})"
        )
        
        return metrics
    
    def record_predictions_batch(
        self,
        model_id: str,
        model_version: str,
        transaction_ids: List[str],
        predictions: np.ndarray,
        confidences: np.ndarray,
        features: List[Dict[str, Any]],
        actual_labels: Optional[np.ndarray] = None
    ) -> int:
        """
        Record a batch of model predictions.
        
        Args:
            model_id: Model identifier
            model_version: Model version
            transaction_ids: List of transaction IDs
            predictions: Predicted labels
            confidences: Prediction confidences
            features: List of feature dictionaries
            actual_labels: Actual labels (if available)
            
        Returns:
            Number of predictions recorded
        """
        # Validate inputs
        n_predictions = len(transaction_ids)
        if not all(len(arr) == n_predictions for arr in [predictions, confidences, features]):
            raise ValidationError("All input arrays must have the same length")
        
        if actual_labels is not None and len(actual_labels) != n_predictions:
            raise ValidationError("actual_labels must have the same length as predictions")
        
        timestamp = datetime.now()
        
        # Prepare batch data
        batch_data = []
        for i in range(n_predictions):
            prediction_id = self._generate_prediction_id()
            
            batch_data.append((
                prediction_id,
                model_id,
                model_version,
                transaction_ids[i],
                int(predictions[i]),
                int(actual_labels[i]) if actual_labels is not None else None,
                float(confidences[i]),
                timestamp.isoformat(),
                json.dumps(features[i]) if self.db_config.enable_compression else json.dumps(features[i]),
                json.dumps({})  # Empty metadata
            ))
        
        # Insert in batches
        with self.connection_pool.get_connection() as conn:
            conn.executemany('''
                INSERT INTO model_predictions
                (prediction_id, model_id, model_version, transaction_id,
                 predicted_label, actual_label, prediction_confidence,
                 timestamp, features, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', batch_data)
            conn.commit()
        
        self.operation_stats['predictions_recorded'] += n_predictions
        self.logger.info(f"Recorded {n_predictions} predictions for model {model_id} v{model_version}")
        
        return n_predictions
    
    # ======================== QUERY OPERATIONS ========================
    
    def get_accuracy_metrics(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        data_type: Optional[DataType] = None,
        metric_type: Optional[MetricType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[AccuracyMetric]:
        """
        Query accuracy metrics with filters.
        
        Args:
            model_id: Model identifier
            model_version: Specific version (optional)
            data_type: Filter by data type
            metric_type: Filter by metric type
            start_date: Start date for time range
            end_date: End date for time range
            
        Returns:
            List of AccuracyMetric objects
        """
        query = "SELECT * FROM accuracy_metrics WHERE model_id = ?"
        params = [model_id]
        
        if model_version:
            query += " AND model_version = ?"
            params.append(model_version)
        
        if data_type:
            query += " AND data_type = ?"
            params.append(data_type.value)
        
        if metric_type:
            query += " AND metric_type = ?"
            params.append(metric_type.value)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        query += " ORDER BY timestamp DESC"
        
        with self.connection_pool.get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        return [self._row_to_accuracy_metric(row) for row in rows]
    
    def get_model_comparison(
        self,
        model_ids: List[str],
        metric_type: MetricType,
        data_type: DataType = DataType.TEST,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models on a specific metric.
        
        Args:
            model_ids: List of model IDs to compare
            metric_type: Metric to compare
            data_type: Data type to use
            start_date: Start date for comparison
            end_date: End date for comparison
            
        Returns:
            DataFrame with comparison results
        """
        placeholders = ','.join(['?' for _ in model_ids])
        query = f'''
            SELECT 
                model_id,
                model_version,
                AVG(CAST(metric_value AS REAL)) as avg_value,
                MIN(CAST(metric_value AS REAL)) as min_value,
                MAX(CAST(metric_value AS REAL)) as max_value,
                COUNT(*) as n_measurements
            FROM accuracy_metrics
            WHERE model_id IN ({placeholders})
                AND metric_type = ?
                AND data_type = ?
        '''
        
        params = model_ids + [metric_type.value, data_type.value]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        query += " GROUP BY model_id, model_version ORDER BY avg_value DESC"
        
        with self.connection_pool.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        return df
    
    def get_accuracy_trends(
        self,
        model_id: str,
        model_version: str,
        metric_type: MetricType,
        data_type: DataType = DataType.TEST,
        window_days: int = 30
    ) -> pd.DataFrame:
        """
        Get accuracy trends over time.
        
        Args:
            model_id: Model identifier
            model_version: Model version
            metric_type: Metric to track
            data_type: Data type
            window_days: Number of days to look back
            
        Returns:
            DataFrame with time series data
        """
        start_date = datetime.now() - timedelta(days=window_days)
        
        query = '''
            SELECT 
                DATE(timestamp) as date,
                AVG(CAST(metric_value AS REAL)) as avg_value,
                MIN(CAST(metric_value AS REAL)) as min_value,
                MAX(CAST(metric_value AS REAL)) as max_value,
                COUNT(*) as n_measurements
            FROM accuracy_metrics
            WHERE model_id = ?
                AND model_version = ?
                AND metric_type = ?
                AND data_type = ?
                AND timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        '''
        
        with self.connection_pool.get_connection() as conn:
            df = pd.read_sql_query(
                query, conn,
                params=[
                    model_id, model_version,
                    metric_type.value, data_type.value,
                    start_date.isoformat()
                ]
            )
        
        # Convert date strings to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def get_confusion_matrix_history(
        self,
        model_id: str,
        model_version: str,
        data_type: DataType = DataType.TEST,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get confusion matrix history for a model.
        
        Args:
            model_id: Model identifier
            model_version: Model version
            data_type: Data type
            limit: Number of records to return
            
        Returns:
            List of confusion matrix records
        """
        query = '''
            SELECT * FROM confusion_matrices
            WHERE model_id = ? AND model_version = ? AND data_type = ?
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        
        with self.connection_pool.get_connection() as conn:
            cursor = conn.execute(
                query,
                (model_id, model_version, data_type.value, limit)
            )
            
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
        
        results = []
        for row in rows:
            record = dict(zip(columns, row))
            
            # Calculate derived metrics
            tp = record['true_positives']
            tn = record['true_negatives']
            fp = record['false_positives']
            fn = record['false_negatives']
            
            total = tp + tn + fp + fn
            if total > 0:
                record['accuracy'] = (tp + tn) / total
                record['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
                record['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                record['f1_score'] = 2 * (record['precision'] * record['recall']) / (record['precision'] + record['recall']) \
                    if (record['precision'] + record['recall']) > 0 else 0
            
            results.append(record)
        
        return results
    
    # ======================== STATISTICAL ANALYSIS ========================
    
    def calculate_model_statistics(
        self,
        model_id: str,
        model_version: str,
        data_type: DataType = DataType.TEST
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for a model.
        
        Args:
            model_id: Model identifier
            model_version: Model version
            data_type: Data type to analyze
            
        Returns:
            Dictionary with statistical analysis
        """
        stats = {
            'model_id': model_id,
            'model_version': model_version,
            'data_type': data_type.value,
            'metrics': {},
            'trends': {},
            'stability': {}
        }
        
        # Get all metrics for the model
        metrics = self.get_accuracy_metrics(
            model_id=model_id,
            model_version=model_version,
            data_type=data_type
        )
        
        if not metrics:
            return stats
        
        # Group by metric type
        metric_groups = defaultdict(list)
        for metric in metrics:
            if metric.metric_type in [MetricType.ACCURACY, MetricType.PRECISION, 
                                     MetricType.RECALL, MetricType.F1_SCORE]:
                try:
                    value = float(metric.metric_value)
                    metric_groups[metric.metric_type.value].append(value)
                except ValueError:
                    continue
        
        # Calculate statistics for each metric
        for metric_name, values in metric_groups.items():
            if values:
                stats['metrics'][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q1': np.percentile(values, 25),
                    'q3': np.percentile(values, 75),
                    'n_samples': len(values)
                }
                
                # Calculate stability (coefficient of variation)
                if stats['metrics'][metric_name]['mean'] > 0:
                    cv = stats['metrics'][metric_name]['std'] / stats['metrics'][metric_name]['mean']
                    stats['stability'][metric_name] = {
                        'coefficient_of_variation': cv,
                        'is_stable': cv < 0.1  # Less than 10% variation
                    }
        
        # Calculate trends
        for metric_type in [MetricType.ACCURACY, MetricType.F1_SCORE]:
            trend_df = self.get_accuracy_trends(
                model_id=model_id,
                model_version=model_version,
                metric_type=metric_type,
                data_type=data_type,
                window_days=30
            )
            
            if not trend_df.empty:
                # Simple linear regression for trend
                x = np.arange(len(trend_df))
                y = trend_df['avg_value'].values
                
                if len(x) > 1:
                    coeffs = np.polyfit(x, y, 1)
                    trend = 'increasing' if coeffs[0] > 0 else 'decreasing'
                    
                    stats['trends'][metric_type.value] = {
                        'direction': trend,
                        'slope': float(coeffs[0]),
                        'current_value': float(y[-1]),
                        'change_rate': float(coeffs[0] / np.mean(y)) if np.mean(y) > 0 else 0
                    }
        
        return stats
    
    def detect_performance_degradation(
        self,
        model_id: str,
        model_version: str,
        metric_type: MetricType = MetricType.F1_SCORE,
        threshold: float = 0.05,
        window_days: int = 7
    ) -> Dict[str, Any]:
        """
        Detect performance degradation in a model.
        
        Args:
            model_id: Model identifier
            model_version: Model version
            metric_type: Metric to monitor
            threshold: Degradation threshold (relative)
            window_days: Window for comparison
            
        Returns:
            Degradation analysis results
        """
        # Get recent metrics
        end_date = datetime.now()
        mid_date = end_date - timedelta(days=window_days)
        start_date = end_date - timedelta(days=window_days * 2)
        
        # Get metrics for two periods
        period1_metrics = self.get_accuracy_metrics(
            model_id=model_id,
            model_version=model_version,
            metric_type=metric_type,
            start_date=start_date,
            end_date=mid_date
        )
        
        period2_metrics = self.get_accuracy_metrics(
            model_id=model_id,
            model_version=model_version,
            metric_type=metric_type,
            start_date=mid_date,
            end_date=end_date
        )
        
        # Calculate averages
        def get_average(metrics):
            values = []
            for m in metrics:
                try:
                    values.append(float(m.metric_value))
                except ValueError:
                    continue
            return np.mean(values) if values else None
        
        avg1 = get_average(period1_metrics)
        avg2 = get_average(period2_metrics)
        
        result = {
            'degradation_detected': False,
            'period1_avg': avg1,
            'period2_avg': avg2,
            'relative_change': None,
            'absolute_change': None,
            'recommendation': None
        }
        
        if avg1 is not None and avg2 is not None:
            absolute_change = avg2 - avg1
            relative_change = absolute_change / avg1 if avg1 > 0 else 0
            
            result['absolute_change'] = absolute_change
            result['relative_change'] = relative_change
            
            if relative_change < -threshold:
                result['degradation_detected'] = True
                result['recommendation'] = (
                    f"Model performance has degraded by {abs(relative_change)*100:.1f}%. "
                    "Consider retraining or investigating data drift."
                )
        
        return result
    
    # ======================== DATA MANAGEMENT ========================
    
    def cleanup_old_data(self, retention_days: Optional[int] = None) -> Dict[str, int]:
        """
        Clean up old data based on retention policy.
        
        Args:
            retention_days: Days to retain data (uses config if not specified)
            
        Returns:
            Dictionary with cleanup statistics
        """
        if retention_days is None:
            retention_days = self.db_config.retention_days
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        cutoff_str = cutoff_date.isoformat()
        
        cleanup_stats = {}
        
        with self.connection_pool.get_connection() as conn:
            # Clean up each table
            tables = [
                'accuracy_metrics',
                'model_predictions',
                'confusion_matrices',
                'performance_metrics'
            ]
            
            for table in tables:
                cursor = conn.execute(
                    f"DELETE FROM {table} WHERE timestamp < ?",
                    (cutoff_str,)
                )
                cleanup_stats[table] = cursor.rowcount
            
            conn.commit()
        
        total_deleted = sum(cleanup_stats.values())
        self.logger.info(
            f"Cleaned up {total_deleted} records older than {retention_days} days"
        )
        
        return cleanup_stats
    
    def vacuum_database(self) -> bool:
        """
        Vacuum database to reclaim space and optimize performance.
        
        Returns:
            Success status
        """
        try:
            with self.connection_pool.get_connection() as conn:
                conn.execute("VACUUM")
                conn.execute("ANALYZE")
            
            self.logger.info("Database vacuum completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Database vacuum failed: {e}")
            return False
    
    def export_metrics(
        self,
        model_id: str,
        model_version: str,
        output_path: Path,
        format: str = 'csv'
    ) -> bool:
        """
        Export metrics to file.
        
        Args:
            model_id: Model identifier
            model_version: Model version
            output_path: Output file path
            format: Export format ('csv', 'json', 'parquet')
            
        Returns:
            Success status
        """
        try:
            # Get all metrics
            metrics = self.get_accuracy_metrics(
                model_id=model_id,
                model_version=model_version
            )
            
            if not metrics:
                self.logger.warning(f"No metrics found for model {model_id} v{model_version}")
                return False
            
            # Convert to DataFrame
            data = []
            for metric in metrics:
                data.append({
                    'timestamp': metric.timestamp,
                    'data_type': metric.data_type.value,
                    'metric_type': metric.metric_type.value,
                    'metric_value': metric.metric_value,
                    'sample_size': metric.sample_size
                })
            
            df = pd.DataFrame(data)
            
            # Export based on format
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'csv':
                df.to_csv(output_path, index=False)
            elif format == 'json':
                df.to_json(output_path, orient='records', date_format='iso')
            elif format == 'parquet':
                df.to_parquet(output_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Exported metrics to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return False
    
    # ======================== INTEGRATION WITH TRAINVALIDATIONTESTMANAGER ========================
    
    def integrate_with_dataset_manager(
        self,
        dataset_manager: TrainValidationTestManager,
        model_id: str,
        model_version: str
    ) -> Dict[str, Any]:
        """
        Integrate with TrainValidationTestManager for comprehensive tracking.
        
        Args:
            dataset_manager: TrainValidationTestManager instance
            model_id: Model identifier
            model_version: Model version
            
        Returns:
            Integration status and metadata
        """
        integration_results = {
            'model_id': model_id,
            'model_version': model_version,
            'datasets_linked': [],
            'splits_tracked': [],
            'status': 'success'
        }
        
        try:
            # Get active datasets from dataset manager
            if hasattr(dataset_manager, 'dataset_registry'):
                for dataset_id, metadata in dataset_manager.dataset_registry.items():
                    # Link dataset to model
                    self._link_dataset_to_model(
                        model_id=model_id,
                        model_version=model_version,
                        dataset_id=dataset_id,
                        metadata=metadata
                    )
                    integration_results['datasets_linked'].append(dataset_id)
            
            # Track active splits
            if hasattr(dataset_manager, 'active_splits'):
                for split_id, split_info in dataset_manager.active_splits.items():
                    self._track_data_split(split_id, split_info)
                    integration_results['splits_tracked'].append(split_id)
            
            self.logger.info(
                f"Integrated model {model_id} v{model_version} with "
                f"{len(integration_results['datasets_linked'])} datasets"
            )
            
        except Exception as e:
            integration_results['status'] = 'failed'
            integration_results['error'] = str(e)
            self.logger.error(f"Integration failed: {e}")
        
        return integration_results
    
    # ======================== BACKGROUND TASKS ========================
    
    def _cleanup_task(self) -> None:
        """Background task for data cleanup"""
        while not self._stop_background_tasks.is_set():
            try:
                # Run cleanup
                self.cleanup_old_data()
                
                # Sleep for 24 hours
                for _ in range(24 * 60):  # Check every minute for stop signal
                    if self._stop_background_tasks.is_set():
                        break
                    time.sleep(60)
                    
            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")
                time.sleep(3600)  # Sleep 1 hour on error
    
    def _vacuum_task(self) -> None:
        """Background task for database vacuum"""
        while not self._stop_background_tasks.is_set():
            try:
                # Run vacuum
                self.vacuum_database()
                
                # Sleep for configured interval
                sleep_hours = self.db_config.vacuum_interval_hours
                for _ in range(sleep_hours * 60):  # Check every minute
                    if self._stop_background_tasks.is_set():
                        break
                    time.sleep(60)
                    
            except Exception as e:
                self.logger.error(f"Vacuum task error: {e}")
                time.sleep(3600)  # Sleep 1 hour on error
    
    # ======================== UTILITY METHODS ========================
    
    def _generate_metric_id(self) -> str:
        """Generate unique metric ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        random_suffix = os.urandom(4).hex()
        return f"metric_{timestamp}_{random_suffix}"
    
    def _generate_prediction_id(self) -> str:
        """Generate unique prediction ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        random_suffix = os.urandom(4).hex()
        return f"pred_{timestamp}_{random_suffix}"
    
    def _row_to_model_version(self, row: tuple) -> ModelVersion:
        """Convert database row to ModelVersion object"""
        return ModelVersion(
            model_id=row[0],
            version=row[1],
            created_at=datetime.fromisoformat(row[2]),
            model_type=row[3],
            parameters=json.loads(row[4]) if row[4] else {},
            training_dataset_id=row[5],
            status=ModelStatus(row[6]),
            metadata=json.loads(row[7]) if row[7] else {}
        )
    
    def _row_to_accuracy_metric(self, row: tuple) -> AccuracyMetric:
        """Convert database row to AccuracyMetric object"""
        # Parse metric value
        metric_value = row[5]
        try:
            # Try to parse as JSON first
            metric_value = json.loads(metric_value)
        except json.JSONDecodeError:
            # Try to parse as float
            try:
                metric_value = float(metric_value)
            except ValueError:
                # Keep as string
                pass
        
        return AccuracyMetric(
            metric_id=row[0],
            model_id=row[1],
            model_version=row[2],
            data_type=DataType(row[3]),
            metric_type=MetricType(row[4]),
            metric_value=metric_value,
            timestamp=datetime.fromisoformat(row[6]),
            dataset_id=row[7],
            sample_size=row[8],
            additional_info=json.loads(row[9]) if row[9] else {}
        )
    
    def _link_dataset_to_model(
        self,
        model_id: str,
        model_version: str,
        dataset_id: str,
        metadata: DatasetMetadata
    ) -> None:
        """Link dataset to model for tracking"""
        # This would store the relationship between datasets and models
        # Implementation depends on specific tracking requirements
        pass
    
    def _track_data_split(self, split_id: str, split_info: Dict[str, Any]) -> None:
        """Track data split information"""
        # This would store split configuration for reproducibility
        # Implementation depends on specific tracking requirements
        pass
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics and health metrics"""
        stats = {
            'database_size_mb': 0,
            'table_sizes': {},
            'index_sizes': {},
            'row_counts': {},
            'connection_pool_stats': {
                'active_connections': self.connection_pool._created_connections,
                'max_connections': self.connection_pool.config.max_connections
            },
            'operation_stats': dict(self.operation_stats),
            'cache_stats': self.cache_manager.get_stats() if self.cache_manager else None
        }
        
        with self.connection_pool.get_connection() as conn:
            # Get database size
            cursor = conn.execute("SELECT page_count * page_size / 1024.0 / 1024.0 FROM pragma_page_count(), pragma_page_size()")
            stats['database_size_mb'] = cursor.fetchone()[0]
            
            # Get table statistics
            tables = ['model_versions', 'accuracy_metrics', 'model_predictions',
                     'confusion_matrices', 'performance_metrics', 'data_splits']
            
            for table in tables:
                # Row count
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats['row_counts'][table] = cursor.fetchone()[0]
                
                # Table size (approximate)
                cursor = conn.execute(f"SELECT COUNT(*) * AVG(LENGTH(CAST({table}.* AS TEXT))) / 1024.0 / 1024.0 FROM {table}")
                result = cursor.fetchone()
                stats['table_sizes'][table] = result[0] if result[0] else 0
        
        return stats
    
    def shutdown(self) -> None:
        """Shutdown database and cleanup resources"""
        self.logger.info("Shutting down AccuracyTrackingDatabase")
        
        # Stop background tasks
        self._stop_background_tasks.set()
        self.executor.shutdown(wait=True)
        
        # Close connection pool
        self.connection_pool.close_all()
        
        # Clear cache
        if self.cache_manager:
            self.cache_manager.clear()
        
        self.logger.info("AccuracyTrackingDatabase shutdown complete")


# ======================== USAGE EXAMPLES ========================

if __name__ == "__main__":
    # Example usage demonstrating all features
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    
    print("=== AccuracyTrackingDatabase Examples ===\n")
    
    # Initialize database
    config = {
        'database': {
            'db_path': Path("example_accuracy_tracking.db"),
            'retention_days': 30
        },
        'connection_pool': {
            'max_connections': 5
        }
    }
    
    db = AccuracyTrackingDatabase(config)
    
    # Example 1: Create model version
    print("1. Creating model version...")
    model_version = db.create_model_version(
        model_id="fraud_detector_v1",
        version="1.0.0",
        model_type="RandomForest",
        parameters={
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2
        },
        training_dataset_id="dataset_20240115_120000",
        metadata={'framework': 'sklearn', 'feature_count': 20}
    )
    print(f"   Created: {model_version.model_id} v{model_version.version}\n")
    
    # Example 2: Generate sample data and predictions
    print("2. Generating sample predictions...")
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Simulate predictions
    y_pred = np.random.choice([0, 1], size=len(y_test), p=[0.7, 0.3])
    y_proba = np.random.rand(len(y_test), 2)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    # Example 3: Record accuracy metrics
    print("3. Recording accuracy metrics...")
    metrics = db.record_accuracy_metrics(
        model_id="fraud_detector_v1",
        model_version="1.0.0",
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        data_type=DataType.TEST,
        dataset_id="dataset_20240115_120000"
    )
    
    print(f"   Accuracy: {metrics['accuracy']:.3f}")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   F1 Score: {metrics['f1_score']:.3f}\n")
    
    # Example 4: Record batch predictions
    print("4. Recording batch predictions...")
    transaction_ids = [f"txn_{i:06d}" for i in range(100)]
    predictions = np.random.choice([0, 1], size=100)
    confidences = np.random.rand(100)
    features = [{'amount': np.random.rand() * 1000, 'merchant': f'merchant_{i%10}'} 
                for i in range(100)]
    
    n_recorded = db.record_predictions_batch(
        model_id="fraud_detector_v1",
        model_version="1.0.0",
        transaction_ids=transaction_ids,
        predictions=predictions,
        confidences=confidences,
        features=features
    )
    print(f"   Recorded {n_recorded} predictions\n")
    
    # Example 5: Query accuracy metrics
    print("5. Querying accuracy metrics...")
    recent_metrics = db.get_accuracy_metrics(
        model_id="fraud_detector_v1",
        metric_type=MetricType.ACCURACY
    )
    print(f"   Found {len(recent_metrics)} accuracy measurements\n")
    
    # Example 6: Calculate model statistics
    print("6. Calculating model statistics...")
    stats = db.calculate_model_statistics(
        model_id="fraud_detector_v1",
        model_version="1.0.0"
    )
    
    if stats['metrics']:
        for metric_name, metric_stats in stats['metrics'].items():
            print(f"   {metric_name}:")
            print(f"     Mean: {metric_stats['mean']:.3f}")
            print(f"     Std: {metric_stats['std']:.3f}")
            print(f"     Range: [{metric_stats['min']:.3f}, {metric_stats['max']:.3f}]")
    
    # Example 7: Detect performance degradation
    print("\n7. Checking for performance degradation...")
    degradation = db.detect_performance_degradation(
        model_id="fraud_detector_v1",
        model_version="1.0.0"
    )
    
    if degradation['degradation_detected']:
        print(f"   WARNING: {degradation['recommendation']}")
    else:
        print("   No degradation detected")
    
    # Example 8: Get database statistics
    print("\n8. Database statistics:")
    db_stats = db.get_database_statistics()
    print(f"   Database size: {db_stats['database_size_mb']:.2f} MB")
    print(f"   Total models: {db_stats['row_counts'].get('model_versions', 0)}")
    print(f"   Total metrics: {db_stats['row_counts'].get('accuracy_metrics', 0)}")
    print(f"   Total predictions: {db_stats['row_counts'].get('model_predictions', 0)}")
    
    # Cleanup
    db.shutdown()
    
    # Remove example database
    if Path("example_accuracy_tracking.db").exists():
        Path("example_accuracy_tracking.db").unlink()
    
    print("\n=== All examples completed successfully! ===")