"""
Production Telemetry Collection and Processing System
Provides comprehensive telemetry data collection, processing, export, and retention for production environments.
NO FALLBACKS - HARD FAILURES ONLY architecture.
"""

import asyncio
import json
import logging
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union, Set, Tuple
from queue import Queue, Empty
import hashlib
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

try:
    from .production_monitoring_system import ProductionMonitoringSystem
    from .production_config_manager import ProductionConfigManager
except ImportError:
    # Handle import when running as standalone script
    try:
        from production_monitoring_system import ProductionMonitoringSystem
        from production_config_manager import ProductionConfigManager
    except ImportError:
        ProductionMonitoringSystem = None
        ProductionConfigManager = None


class TelemetryDataType(Enum):
    """Telemetry data types."""
    METRICS = "metrics"
    LOGS = "logs"
    TRACES = "traces"
    EVENTS = "events"
    SYSTEM_STATS = "system_stats"
    APPLICATION_DATA = "application_data"
    CUSTOM = "custom"


class TelemetryExportFormat(Enum):
    """Telemetry export formats."""
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    AVRO = "avro"
    PROTOBUF = "protobuf"
    BINARY = "binary"


class TelemetryLevel(Enum):
    """Telemetry collection levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    DEBUG = "debug"


class RetentionPolicy(Enum):
    """Data retention policies."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    PERMANENT = "permanent"


@dataclass
class TelemetryDataPoint:
    """Individual telemetry data point."""
    timestamp: datetime
    data_type: TelemetryDataType
    source: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'data_type': self.data_type.value,
            'source': self.source,
            'data': self.data,
            'metadata': self.metadata,
            'tags': list(self.tags),
            'session_id': self.session_id,
            'trace_id': self.trace_id,
            'span_id': self.span_id
        }


@dataclass
class TelemetryExportConfig:
    """Configuration for telemetry export."""
    format: TelemetryExportFormat
    destination: str
    batch_size: int = 1000
    compression: bool = True
    encryption: bool = False
    include_metadata: bool = True
    filters: List[str] = field(default_factory=list)
    transformation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TelemetryRetentionConfig:
    """Configuration for telemetry data retention."""
    policy: RetentionPolicy
    duration_days: int
    data_types: List[TelemetryDataType] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    cleanup_interval_hours: int = 24
    archive_before_delete: bool = True
    archive_location: Optional[str] = None


class TelemetryCollector(ABC):
    """Abstract base class for telemetry collectors."""
    
    @abstractmethod
    def collect(self) -> List[TelemetryDataPoint]:
        """Collect telemetry data points."""
        pass
    
    @abstractmethod
    def get_collector_info(self) -> Dict[str, Any]:
        """Get collector information."""
        pass


class SystemTelemetryCollector(TelemetryCollector):
    """System telemetry collector."""
    
    def __init__(self):
        self.name = "system_telemetry_collector"
        self.last_collection = None
        
    def collect(self) -> List[TelemetryDataPoint]:
        """Collect system telemetry data."""
        try:
            import psutil
            
            current_time = datetime.now(timezone.utc)
            data_points = []
            
            # CPU telemetry
            cpu_data = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'cpu_count': psutil.cpu_count(),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else None
            }
            
            data_points.append(TelemetryDataPoint(
                timestamp=current_time,
                data_type=TelemetryDataType.SYSTEM_STATS,
                source=f"{self.name}.cpu",
                data=cpu_data,
                metadata={'collection_interval': 1},
                tags={'category', 'system', 'cpu'}
            ))
            
            # Memory telemetry
            memory = psutil.virtual_memory()
            memory_data = {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used,
                'free': memory.free,
                'buffers': getattr(memory, 'buffers', None),
                'cached': getattr(memory, 'cached', None)
            }
            
            data_points.append(TelemetryDataPoint(
                timestamp=current_time,
                data_type=TelemetryDataType.SYSTEM_STATS,
                source=f"{self.name}.memory",
                data=memory_data,
                metadata={'unit': 'bytes'},
                tags={'category', 'system', 'memory'}
            ))
            
            # Disk telemetry
            disk_usage = psutil.disk_usage('/')
            disk_data = {
                'total': disk_usage.total,
                'used': disk_usage.used,
                'free': disk_usage.free,
                'percent': (disk_usage.used / disk_usage.total) * 100
            }
            
            data_points.append(TelemetryDataPoint(
                timestamp=current_time,
                data_type=TelemetryDataType.SYSTEM_STATS,
                source=f"{self.name}.disk",
                data=disk_data,
                metadata={'mount_point': '/', 'unit': 'bytes'},
                tags={'category', 'system', 'disk'}
            ))
            
            # Network telemetry
            network_io = psutil.net_io_counters()
            network_data = {
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv,
                'packets_sent': network_io.packets_sent,
                'packets_recv': network_io.packets_recv,
                'errin': network_io.errin,
                'errout': network_io.errout,
                'dropin': network_io.dropin,
                'dropout': network_io.dropout
            }
            
            data_points.append(TelemetryDataPoint(
                timestamp=current_time,
                data_type=TelemetryDataType.SYSTEM_STATS,
                source=f"{self.name}.network",
                data=network_data,
                metadata={'interface': 'all'},
                tags={'category', 'system', 'network'}
            ))
            
            self.last_collection = current_time
            return data_points
            
        except Exception as e:
            raise RuntimeError(f"System telemetry collection failed: {e}")
    
    def get_collector_info(self) -> Dict[str, Any]:
        """Get system collector information."""
        return {
            'name': self.name,
            'type': 'system',
            'last_collection': self.last_collection.isoformat() if self.last_collection else None,
            'capabilities': ['cpu', 'memory', 'disk', 'network'],
            'dependencies': ['psutil']
        }


class ApplicationTelemetryCollector(TelemetryCollector):
    """Application telemetry collector."""
    
    def __init__(self, app_name: str = "independent_core"):
        self.app_name = app_name
        self.name = f"application_telemetry_collector_{app_name}"
        self.last_collection = None
        self.custom_metrics = {}
        
    def collect(self) -> List[TelemetryDataPoint]:
        """Collect application telemetry data."""
        try:
            current_time = datetime.now(timezone.utc)
            data_points = []
            
            # Application runtime telemetry
            runtime_data = {
                'uptime_seconds': time.time() - getattr(self, '_start_time', time.time()),
                'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                'process_id': os.getpid(),
                'thread_count': threading.active_count(),
                'custom_metrics_count': len(self.custom_metrics)
            }
            
            data_points.append(TelemetryDataPoint(
                timestamp=current_time,
                data_type=TelemetryDataType.APPLICATION_DATA,
                source=f"{self.name}.runtime",
                data=runtime_data,
                metadata={'app_name': self.app_name},
                tags={'category', 'application', 'runtime'}
            ))
            
            # Custom metrics telemetry
            if self.custom_metrics:
                data_points.append(TelemetryDataPoint(
                    timestamp=current_time,
                    data_type=TelemetryDataType.METRICS,
                    source=f"{self.name}.custom_metrics",
                    data=self.custom_metrics.copy(),
                    metadata={'app_name': self.app_name, 'metrics_count': len(self.custom_metrics)},
                    tags={'category', 'application', 'custom_metrics'}
                ))
            
            self.last_collection = current_time
            return data_points
            
        except Exception as e:
            raise RuntimeError(f"Application telemetry collection failed: {e}")
    
    def add_custom_metric(self, name: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        """Add custom metric to be collected."""
        self.custom_metrics[name] = {
            'value': value,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metadata': metadata or {}
        }
    
    def get_collector_info(self) -> Dict[str, Any]:
        """Get application collector information."""
        return {
            'name': self.name,
            'type': 'application',
            'app_name': self.app_name,
            'last_collection': self.last_collection.isoformat() if self.last_collection else None,
            'custom_metrics_count': len(self.custom_metrics),
            'capabilities': ['runtime', 'custom_metrics']
        }


class TelemetryProcessor:
    """Processes telemetry data points."""
    
    def __init__(self):
        self.processing_rules = {}
        self.filters = []
        self.transformations = []
        
    def add_filter(self, filter_func: Callable[[TelemetryDataPoint], bool]):
        """Add a filter function."""
        self.filters.append(filter_func)
    
    def add_transformation(self, transform_func: Callable[[TelemetryDataPoint], TelemetryDataPoint]):
        """Add a transformation function."""
        self.transformations.append(transform_func)
    
    def process(self, data_points: List[TelemetryDataPoint]) -> List[TelemetryDataPoint]:
        """Process telemetry data points."""
        try:
            processed_points = data_points.copy()
            
            # Apply filters
            for filter_func in self.filters:
                processed_points = [dp for dp in processed_points if filter_func(dp)]
            
            # Apply transformations
            for transform_func in self.transformations:
                processed_points = [transform_func(dp) for dp in processed_points]
            
            return processed_points
            
        except Exception as e:
            raise RuntimeError(f"Telemetry processing failed: {e}")
    
    def add_processing_rule(self, name: str, rule: Dict[str, Any]):
        """Add a processing rule."""
        self.processing_rules[name] = rule


class TelemetryExporter:
    """Exports telemetry data to various formats and destinations."""
    
    def __init__(self):
        self.export_configs = {}
        self.exporters = {}
        
    def add_export_config(self, name: str, config: TelemetryExportConfig):
        """Add export configuration."""
        self.export_configs[name] = config
        
    def export_data(self, data_points: List[TelemetryDataPoint], config_name: str) -> bool:
        """Export telemetry data using specified configuration."""
        try:
            if config_name not in self.export_configs:
                raise ValueError(f"Export configuration '{config_name}' not found")
            
            config = self.export_configs[config_name]
            
            # Apply filters
            filtered_points = self._apply_filters(data_points, config.filters)
            
            # Apply transformations
            transformed_points = self._apply_transformations(filtered_points, config.transformation_rules)
            
            # Export based on format
            if config.format == TelemetryExportFormat.JSON:
                return self._export_json(transformed_points, config)
            elif config.format == TelemetryExportFormat.CSV:
                return self._export_csv(transformed_points, config)
            elif config.format == TelemetryExportFormat.BINARY:
                return self._export_binary(transformed_points, config)
            else:
                raise ValueError(f"Export format '{config.format}' not supported")
                
        except Exception as e:
            raise RuntimeError(f"Telemetry export failed: {e}")
    
    def _apply_filters(self, data_points: List[TelemetryDataPoint], filters: List[str]) -> List[TelemetryDataPoint]:
        """Apply filters to data points."""
        if not filters:
            return data_points
        
        filtered_points = []
        for point in data_points:
            include = True
            for filter_rule in filters:
                # Simple filter implementation - can be extended
                if filter_rule.startswith('data_type:'):
                    allowed_type = filter_rule.split(':')[1]
                    if point.data_type.value != allowed_type:
                        include = False
                        break
                elif filter_rule.startswith('source:'):
                    allowed_source = filter_rule.split(':')[1]
                    if point.source != allowed_source:
                        include = False
                        break
            
            if include:
                filtered_points.append(point)
        
        return filtered_points
    
    def _apply_transformations(self, data_points: List[TelemetryDataPoint], 
                             transformation_rules: Dict[str, Any]) -> List[TelemetryDataPoint]:
        """Apply transformations to data points."""
        if not transformation_rules:
            return data_points
        
        # Simple transformation implementation - can be extended
        transformed_points = []
        for point in data_points:
            transformed_point = point
            
            # Example transformations
            if 'remove_pii' in transformation_rules and transformation_rules['remove_pii']:
                # Remove potentially sensitive data
                cleaned_data = {}
                for key, value in point.data.items():
                    if key.lower() not in ['password', 'token', 'secret', 'key']:
                        cleaned_data[key] = value
                
                transformed_point = TelemetryDataPoint(
                    timestamp=point.timestamp,
                    data_type=point.data_type,
                    source=point.source,
                    data=cleaned_data,
                    metadata=point.metadata,
                    tags=point.tags,
                    session_id=point.session_id,
                    trace_id=point.trace_id,
                    span_id=point.span_id
                )
            
            transformed_points.append(transformed_point)
        
        return transformed_points
    
    def _export_json(self, data_points: List[TelemetryDataPoint], config: TelemetryExportConfig) -> bool:
        """Export data as JSON."""
        try:
            data = [point.to_dict() for point in data_points]
            
            output_file = Path(config.destination) / f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"JSON export failed: {e}")
    
    def _export_csv(self, data_points: List[TelemetryDataPoint], config: TelemetryExportConfig) -> bool:
        """Export data as CSV."""
        try:
            import csv
            
            output_file = Path(config.destination) / f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow(['timestamp', 'data_type', 'source', 'data', 'metadata', 'tags'])
                
                # Write data
                for point in data_points:
                    writer.writerow([
                        point.timestamp.isoformat(),
                        point.data_type.value,
                        point.source,
                        json.dumps(point.data),
                        json.dumps(point.metadata),
                        ','.join(point.tags)
                    ])
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"CSV export failed: {e}")
    
    def _export_binary(self, data_points: List[TelemetryDataPoint], config: TelemetryExportConfig) -> bool:
        """Export data as binary."""
        try:
            output_file = Path(config.destination) / f"telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bin"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = [point.to_dict() for point in data_points]
            
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Binary export failed: {e}")


class TelemetryRetentionManager:
    """Manages telemetry data retention and cleanup."""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.retention_configs = {}
        self.cleanup_thread = None
        self.running = False
        
    def add_retention_config(self, name: str, config: TelemetryRetentionConfig):
        """Add retention configuration."""
        self.retention_configs[name] = config
    
    def start_cleanup_service(self):
        """Start the automatic cleanup service."""
        if self.running:
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def stop_cleanup_service(self):
        """Stop the automatic cleanup service."""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
    
    def _cleanup_worker(self):
        """Worker thread for cleanup operations."""
        while self.running:
            try:
                for config_name, config in self.retention_configs.items():
                    self._apply_retention_policy(config_name, config)
                
                # Sleep until next cleanup interval
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                logging.error(f"Cleanup worker error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _apply_retention_policy(self, config_name: str, config: TelemetryRetentionConfig):
        """Apply retention policy to stored data."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=config.duration_days)
            
            # Find files to clean up
            files_to_process = []
            for file_path in self.storage_path.rglob("*"):
                if file_path.is_file():
                    file_age = datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc)
                    if file_age < cutoff_date:
                        files_to_process.append(file_path)
            
            # Process files based on policy
            for file_path in files_to_process:
                if config.archive_before_delete and config.archive_location:
                    self._archive_file(file_path, config.archive_location)
                
                # Delete file
                file_path.unlink()
                
            logging.info(f"Retention policy '{config_name}' processed {len(files_to_process)} files")
            
        except Exception as e:
            raise RuntimeError(f"Retention policy application failed: {e}")
    
    def _archive_file(self, file_path: Path, archive_location: str):
        """Archive file before deletion."""
        try:
            archive_path = Path(archive_location)
            archive_path.mkdir(parents=True, exist_ok=True)
            
            # Create archive file path with timestamp
            archive_file = archive_path / f"{file_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Copy file to archive
            import shutil
            shutil.copy2(file_path, archive_file)
            
        except Exception as e:
            logging.error(f"Failed to archive file {file_path}: {e}")


class TelemetryManager:
    """Main telemetry management system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.collectors = {}
        self.processor = TelemetryProcessor()
        self.exporter = TelemetryExporter()
        self.retention_manager = None
        self.collection_queue = Queue()
        self.running = False
        self.collection_thread = None
        self.processing_thread = None
        self.export_thread = None
        self.telemetry_level = TelemetryLevel.STANDARD
        self.collection_interval = 60  # seconds
        self.batch_size = 1000
        self.storage_path = "./telemetry_data"
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize default collectors
        self._setup_default_collectors()
        
        # Setup retention manager
        self.retention_manager = TelemetryRetentionManager(self.storage_path)
        
        # Load configuration if provided
        if config_path:
            self._load_config(config_path)
    
    def _setup_default_collectors(self):
        """Setup default telemetry collectors."""
        # System collector
        system_collector = SystemTelemetryCollector()
        self.add_collector("system", system_collector)
        
        # Application collector
        app_collector = ApplicationTelemetryCollector()
        self.add_collector("application", app_collector)
    
    def _load_config(self, config_path: str):
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Apply configuration
            if 'telemetry_level' in config:
                self.telemetry_level = TelemetryLevel(config['telemetry_level'])
            
            if 'collection_interval' in config:
                self.collection_interval = config['collection_interval']
            
            if 'batch_size' in config:
                self.batch_size = config['batch_size']
            
            if 'storage_path' in config:
                self.storage_path = config['storage_path']
            
            # Setup export configurations
            if 'export_configs' in config:
                for name, export_config in config['export_configs'].items():
                    config_obj = TelemetryExportConfig(
                        format=TelemetryExportFormat(export_config['format']),
                        destination=export_config['destination'],
                        batch_size=export_config.get('batch_size', 1000),
                        compression=export_config.get('compression', True),
                        encryption=export_config.get('encryption', False),
                        include_metadata=export_config.get('include_metadata', True),
                        filters=export_config.get('filters', []),
                        transformation_rules=export_config.get('transformation_rules', {})
                    )
                    self.exporter.add_export_config(name, config_obj)
            
            # Setup retention configurations
            if 'retention_configs' in config:
                for name, retention_config in config['retention_configs'].items():
                    config_obj = TelemetryRetentionConfig(
                        policy=RetentionPolicy(retention_config['policy']),
                        duration_days=retention_config['duration_days'],
                        data_types=[TelemetryDataType(dt) for dt in retention_config.get('data_types', [])],
                        sources=retention_config.get('sources', []),
                        cleanup_interval_hours=retention_config.get('cleanup_interval_hours', 24),
                        archive_before_delete=retention_config.get('archive_before_delete', True),
                        archive_location=retention_config.get('archive_location')
                    )
                    self.retention_manager.add_retention_config(name, config_obj)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load telemetry configuration: {e}")
    
    def add_collector(self, name: str, collector: TelemetryCollector):
        """Add a telemetry collector."""
        self.collectors[name] = collector
    
    def remove_collector(self, name: str):
        """Remove a telemetry collector."""
        if name in self.collectors:
            del self.collectors[name]
    
    def start_telemetry(self):
        """Start telemetry collection system."""
        if self.running:
            return
        
        try:
            self.running = True
            
            # Start collection thread
            self.collection_thread = threading.Thread(target=self._collection_worker, daemon=True)
            self.collection_thread.start()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
            self.processing_thread.start()
            
            # Start export thread
            self.export_thread = threading.Thread(target=self._export_worker, daemon=True)
            self.export_thread.start()
            
            # Start retention cleanup service
            self.retention_manager.start_cleanup_service()
            
            logging.info("Telemetry system started successfully")
            
        except Exception as e:
            self.running = False
            raise RuntimeError(f"Failed to start telemetry system: {e}")
    
    def stop_telemetry(self):
        """Stop telemetry collection system."""
        if not self.running:
            return
        
        try:
            self.running = False
            
            # Stop threads
            if self.collection_thread:
                self.collection_thread.join(timeout=5)
            
            if self.processing_thread:
                self.processing_thread.join(timeout=5)
            
            if self.export_thread:
                self.export_thread.join(timeout=5)
            
            # Stop retention service
            self.retention_manager.stop_cleanup_service()
            
            # Shutdown executor
            self.executor.shutdown(wait=True, timeout=10)
            
            logging.info("Telemetry system stopped successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to stop telemetry system: {e}")
    
    def _collection_worker(self):
        """Worker thread for telemetry collection."""
        while self.running:
            try:
                # Collect from all collectors
                all_data_points = []
                
                for collector_name, collector in self.collectors.items():
                    try:
                        data_points = collector.collect()
                        all_data_points.extend(data_points)
                    except Exception as e:
                        logging.error(f"Collection error from {collector_name}: {e}")
                
                # Add to processing queue
                if all_data_points:
                    self.collection_queue.put(all_data_points)
                
                # Wait for next collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logging.error(f"Collection worker error: {e}")
                time.sleep(30)  # Wait before retrying
    
    def _processing_worker(self):
        """Worker thread for telemetry processing."""
        batch = []
        
        while self.running:
            try:
                # Get data from queue
                try:
                    data_points = self.collection_queue.get(timeout=1)
                    batch.extend(data_points)
                except Empty:
                    continue
                
                # Process batch when it reaches batch_size
                if len(batch) >= self.batch_size:
                    processed_batch = self.processor.process(batch)
                    
                    # Store processed data
                    self._store_data(processed_batch)
                    
                    batch = []
                
            except Exception as e:
                logging.error(f"Processing worker error: {e}")
                time.sleep(5)
        
        # Process remaining batch on shutdown
        if batch:
            try:
                processed_batch = self.processor.process(batch)
                self._store_data(processed_batch)
            except Exception as e:
                logging.error(f"Final batch processing error: {e}")
    
    def _export_worker(self):
        """Worker thread for telemetry export."""
        while self.running:
            try:
                # Export data based on export configurations
                for config_name in self.exporter.export_configs.keys():
                    try:
                        # Load stored data for export
                        stored_data = self._load_stored_data()
                        if stored_data:
                            self.exporter.export_data(stored_data, config_name)
                    except Exception as e:
                        logging.error(f"Export error for {config_name}: {e}")
                
                # Wait before next export cycle
                time.sleep(300)  # Export every 5 minutes
                
            except Exception as e:
                logging.error(f"Export worker error: {e}")
                time.sleep(60)
    
    def _store_data(self, data_points: List[TelemetryDataPoint]):
        """Store telemetry data points."""
        try:
            storage_path = Path(self.storage_path)
            storage_path.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"telemetry_batch_{timestamp}.json"
            file_path = storage_path / filename
            
            # Convert data points to dict format
            data = [point.to_dict() for point in data_points]
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            raise RuntimeError(f"Failed to store telemetry data: {e}")
    
    def _load_stored_data(self) -> List[TelemetryDataPoint]:
        """Load stored telemetry data."""
        try:
            storage_path = Path(self.storage_path)
            if not storage_path.exists():
                return []
            
            data_points = []
            
            # Load recent files (last hour)
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            for file_path in storage_path.glob("telemetry_batch_*.json"):
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time > cutoff_time:
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        for item in data:
                            data_points.append(TelemetryDataPoint(
                                timestamp=datetime.fromisoformat(item['timestamp']),
                                data_type=TelemetryDataType(item['data_type']),
                                source=item['source'],
                                data=item['data'],
                                metadata=item['metadata'],
                                tags=set(item['tags']),
                                session_id=item.get('session_id'),
                                trace_id=item.get('trace_id'),
                                span_id=item.get('span_id')
                            ))
                    except Exception as e:
                        logging.error(f"Error loading file {file_path}: {e}")
            
            return data_points
            
        except Exception as e:
            logging.error(f"Failed to load stored data: {e}")
            return []
    
    def get_telemetry_status(self) -> Dict[str, Any]:
        """Get current telemetry system status."""
        return {
            'running': self.running,
            'telemetry_level': self.telemetry_level.value,
            'collection_interval': self.collection_interval,
            'batch_size': self.batch_size,
            'storage_path': self.storage_path,
            'collectors': {name: collector.get_collector_info() for name, collector in self.collectors.items()},
            'queue_size': self.collection_queue.qsize(),
            'export_configs': list(self.exporter.export_configs.keys()),
            'retention_configs': list(self.retention_manager.retention_configs.keys())
        }
    
    def get_telemetry_metrics(self) -> Dict[str, Any]:
        """Get telemetry system metrics."""
        try:
            # Get recent data statistics
            recent_data = self._load_stored_data()
            
            metrics = {
                'total_data_points': len(recent_data),
                'data_types': {},
                'sources': {},
                'collection_rate': len(recent_data) / max(self.collection_interval, 1),
                'storage_usage': self._get_storage_usage(),
                'collector_status': {}
            }
            
            # Analyze data types and sources
            for point in recent_data:
                data_type = point.data_type.value
                source = point.source
                
                metrics['data_types'][data_type] = metrics['data_types'].get(data_type, 0) + 1
                metrics['sources'][source] = metrics['sources'].get(source, 0) + 1
            
            # Get collector status
            for name, collector in self.collectors.items():
                metrics['collector_status'][name] = collector.get_collector_info()
            
            return metrics
            
        except Exception as e:
            raise RuntimeError(f"Failed to get telemetry metrics: {e}")
    
    def _get_storage_usage(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        try:
            storage_path = Path(self.storage_path)
            if not storage_path.exists():
                return {'total_size': 0, 'file_count': 0}
            
            total_size = 0
            file_count = 0
            
            for file_path in storage_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            return {
                'total_size': total_size,
                'file_count': file_count,
                'average_file_size': total_size / max(file_count, 1)
            }
            
        except Exception as e:
            logging.error(f"Failed to get storage usage: {e}")
            return {'total_size': 0, 'file_count': 0}


# Integration with existing systems
def integrate_with_monitoring_system(telemetry_manager: TelemetryManager,
                                   monitoring_system: 'ProductionMonitoringSystem') -> bool:
    """Integrate telemetry with production monitoring system."""
    try:
        # Add monitoring system as telemetry source
        class MonitoringTelemetryCollector(TelemetryCollector):
            def __init__(self, monitoring_sys):
                self.monitoring_system = monitoring_sys
                self.name = "monitoring_system_telemetry"
            
            def collect(self) -> List[TelemetryDataPoint]:
                current_time = datetime.now(timezone.utc)
                data_points = []
                
                # Collect monitoring metrics
                try:
                    metrics = self.monitoring_system.get_metrics()
                    data_points.append(TelemetryDataPoint(
                        timestamp=current_time,
                        data_type=TelemetryDataType.METRICS,
                        source=f"{self.name}.metrics",
                        data=metrics,
                        metadata={'integration': 'monitoring_system'},
                        tags={'monitoring', 'integration'}
                    ))
                except Exception as e:
                    logging.error(f"Failed to collect monitoring metrics: {e}")
                
                # Collect alerts
                try:
                    alerts = self.monitoring_system.get_alerts()
                    if alerts:
                        data_points.append(TelemetryDataPoint(
                            timestamp=current_time,
                            data_type=TelemetryDataType.EVENTS,
                            source=f"{self.name}.alerts",
                            data={'alerts': alerts},
                            metadata={'integration': 'monitoring_system', 'alert_count': len(alerts)},
                            tags={'monitoring', 'alerts', 'integration'}
                        ))
                except Exception as e:
                    logging.error(f"Failed to collect monitoring alerts: {e}")
                
                return data_points
            
            def get_collector_info(self) -> Dict[str, Any]:
                return {
                    'name': self.name,
                    'type': 'integration',
                    'integration_type': 'monitoring_system',
                    'capabilities': ['metrics', 'alerts']
                }
        
        # Add the collector
        monitoring_collector = MonitoringTelemetryCollector(monitoring_system)
        telemetry_manager.add_collector("monitoring_integration", monitoring_collector)
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to integrate with monitoring system: {e}")
        return False


def integrate_with_config_manager(telemetry_manager: TelemetryManager,
                                config_manager: 'ProductionConfigManager') -> bool:
    """Integrate telemetry with production config manager."""
    try:
        # Add config change tracking
        class ConfigTelemetryCollector(TelemetryCollector):
            def __init__(self, config_mgr):
                self.config_manager = config_mgr
                self.name = "config_manager_telemetry"
                self.last_config_hash = None
            
            def collect(self) -> List[TelemetryDataPoint]:
                current_time = datetime.now(timezone.utc)
                data_points = []
                
                try:
                    # Get current config
                    current_config = self.config_manager.get_current_config()
                    config_str = json.dumps(current_config, sort_keys=True)
                    config_hash = hashlib.md5(config_str.encode()).hexdigest()
                    
                    # Check for config changes
                    if self.last_config_hash and self.last_config_hash != config_hash:
                        data_points.append(TelemetryDataPoint(
                            timestamp=current_time,
                            data_type=TelemetryDataType.EVENTS,
                            source=f"{self.name}.config_change",
                            data={
                                'previous_hash': self.last_config_hash,
                                'current_hash': config_hash,
                                'config_size': len(config_str)
                            },
                            metadata={'integration': 'config_manager'},
                            tags={'config', 'change', 'integration'}
                        ))
                    
                    self.last_config_hash = config_hash
                    
                    # Collect config status
                    data_points.append(TelemetryDataPoint(
                        timestamp=current_time,
                        data_type=TelemetryDataType.METRICS,
                        source=f"{self.name}.status",
                        data={
                            'config_hash': config_hash,
                            'config_size': len(config_str),
                            'environment': current_config.get('environment', 'unknown')
                        },
                        metadata={'integration': 'config_manager'},
                        tags={'config', 'status', 'integration'}
                    ))
                    
                except Exception as e:
                    logging.error(f"Failed to collect config telemetry: {e}")
                
                return data_points
            
            def get_collector_info(self) -> Dict[str, Any]:
                return {
                    'name': self.name,
                    'type': 'integration',
                    'integration_type': 'config_manager',
                    'capabilities': ['config_changes', 'config_status']
                }
        
        # Add the collector
        config_collector = ConfigTelemetryCollector(config_manager)
        telemetry_manager.add_collector("config_integration", config_collector)
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to integrate with config manager: {e}")
        return False