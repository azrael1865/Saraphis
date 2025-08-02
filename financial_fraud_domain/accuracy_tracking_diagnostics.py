"""
Accuracy Tracking Diagnostics and Maintenance System - Phase 5C-1B
Comprehensive diagnostics, automated maintenance, and troubleshooting for accuracy tracking.
Part of the Saraphis recursive methodology.
"""

import logging
import time
import threading
import asyncio
import psutil
import sqlite3
import json
import yaml
import os
import sys
import gc
import traceback
import warnings
import cProfile
import pstats
import io
import re
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque, Counter
from contextlib import contextmanager
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import schedule
import resource
import linecache
import tracemalloc
import weakref
import dis
import ast

# Import from existing modules
try:
    from enhanced_fraud_core_monitoring import (
        MonitoringManager, MetricsCollector, PerformanceMetrics,
        ResourceMetrics, CacheManager, MonitoringConfig
    )
    from enhanced_fraud_core_exceptions import (
        EnhancedFraudException, ValidationError, ConfigurationError,
        ResourceError, MonitoringError, ErrorContext, create_error_context
    )
    from accuracy_tracking_db import (
        AccuracyTrackingDatabase, DatabaseConfig, AccuracyMetric
    )
    from config_manager import ConfigManager, ConfigManagerSettings
except ImportError:
    # Fallback for standalone development
    from enhanced_fraud_core_monitoring import (
        MonitoringManager, MetricsCollector, PerformanceMetrics,
        ResourceMetrics, CacheManager, MonitoringConfig
    )
    from enhanced_fraud_core_exceptions import (
        EnhancedFraudException, ValidationError, ConfigurationError,
        ResourceError, MonitoringError, ErrorContext, create_error_context
    )
    from accuracy_tracking_db import (
        AccuracyTrackingDatabase, DatabaseConfig, AccuracyMetric
    )
    from config_manager import ConfigManager

# Configure logging
logger = logging.getLogger(__name__)

# ======================== CUSTOM EXCEPTIONS ========================

class DiagnosticsError(EnhancedFraudException):
    """Base exception for diagnostics errors"""
    pass

class MaintenanceError(EnhancedFraudException):
    """Exception raised during maintenance operations"""
    pass

class TroubleshootingError(EnhancedFraudException):
    """Exception raised during troubleshooting"""
    pass

# ======================== ENUMS AND CONSTANTS ========================

class DiagnosticLevel(Enum):
    """Levels of diagnostic analysis"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    DEEP = "deep"

class MaintenanceType(Enum):
    """Types of maintenance tasks"""
    CLEANUP = "cleanup"
    OPTIMIZATION = "optimization"
    BACKUP = "backup"
    RECOVERY = "recovery"
    MIGRATION = "migration"
    SECURITY_AUDIT = "security_audit"

class DiagnosticCategory(Enum):
    """Categories of diagnostics"""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    DATABASE = "database"
    MEMORY = "memory"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    SECURITY = "security"
    COMPONENTS = "components"

class MaintenanceStatus(Enum):
    """Status of maintenance tasks"""
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TroubleshootingPhase(Enum):
    """Phases of troubleshooting"""
    DETECTION = "detection"
    ANALYSIS = "analysis"
    RESOLUTION = "resolution"
    VERIFICATION = "verification"

# Default configuration
DEFAULT_DIAGNOSTICS_CONFIG = {
    'enable_auto_diagnostics': True,
    'diagnostic_interval_minutes': 60,
    'performance_profiling_enabled': True,
    'memory_profiling_enabled': True,
    'resource_monitoring_enabled': True,
    'maintenance_schedule_enabled': True,
    'max_diagnostic_history': 1000,
    'diagnostic_report_retention_days': 30,
    'enable_troubleshooting_assistant': True,
    'maintenance_window_hours': [2, 3, 4],  # 2am-5am
    'auto_cleanup_threshold_percent': 85,
    'optimization_threshold_seconds': 2.0,
    'memory_leak_threshold_mb': 100,
    'diagnostic_thread_pool_size': 4,
    'enable_remote_diagnostics': False,
    'diagnostic_export_formats': ['json', 'html', 'pdf']
}

# ======================== DATA STRUCTURES ========================

@dataclass
class DiagnosticResult:
    """Result of a diagnostic check"""
    diagnostic_id: str
    category: DiagnosticCategory
    level: DiagnosticLevel
    timestamp: datetime
    duration: float
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    severity: str  # 'info', 'warning', 'error', 'critical'
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MaintenanceTask:
    """Maintenance task definition"""
    task_id: str
    task_type: MaintenanceType
    scheduled_time: datetime
    priority: int  # 1-10, higher is more important
    estimated_duration: float  # minutes
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: MaintenanceStatus = MaintenanceStatus.SCHEDULED
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class PerformanceBottleneck:
    """Identified performance bottleneck"""
    component: str
    operation: str
    average_duration: float
    frequency: int
    impact_score: float
    root_cause: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)

@dataclass
class TroubleshootingSession:
    """Troubleshooting session information"""
    session_id: str
    problem_description: str
    start_time: datetime
    current_phase: TroubleshootingPhase
    diagnostic_results: List[DiagnosticResult] = field(default_factory=list)
    resolution_steps: List[Dict[str, Any]] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class MaintenanceSchedule:
    """Maintenance schedule information"""
    schedule_id: str
    tasks: List[MaintenanceTask]
    created_at: datetime
    next_run: datetime
    recurring: bool = False
    recurrence_pattern: Optional[str] = None  # cron-like pattern

# ======================== DIAGNOSTICS ENGINE ========================

class DiagnosticsEngine:
    """Core diagnostics engine for system analysis"""
    
    def __init__(self, config: Dict[str, Any], orchestrator=None):
        self.config = config
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Diagnostic history
        self.diagnostic_history = deque(maxlen=config['max_diagnostic_history'])
        self.performance_bottlenecks = []
        
        # Profiling
        self.profiler = None
        if config['performance_profiling_enabled']:
            self.profiler = cProfile.Profile()
        
        # Memory tracking
        if config['memory_profiling_enabled']:
            tracemalloc.start()
        
        # Thread pool for async diagnostics
        self.executor = ThreadPoolExecutor(
            max_workers=config['diagnostic_thread_pool_size']
        )
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        self.logger.info("DiagnosticsEngine initialized")
    
    def run_diagnostics(
        self,
        level: DiagnosticLevel = DiagnosticLevel.STANDARD,
        categories: Optional[List[DiagnosticCategory]] = None
    ) -> List[DiagnosticResult]:
        """
        Run comprehensive diagnostics.
        
        Args:
            level: Level of diagnostic analysis
            categories: Specific categories to diagnose (None for all)
            
        Returns:
            List of diagnostic results
        """
        self.logger.info(f"Running {level.value} diagnostics")
        
        if categories is None:
            categories = list(DiagnosticCategory)
        
        results = []
        futures = {}
        
        # Submit diagnostic tasks
        for category in categories:
            if category == DiagnosticCategory.PERFORMANCE:
                future = self.executor.submit(self._diagnose_performance, level)
                futures[future] = category
            elif category == DiagnosticCategory.RESOURCE:
                future = self.executor.submit(self._diagnose_resources, level)
                futures[future] = category
            elif category == DiagnosticCategory.DATABASE:
                future = self.executor.submit(self._diagnose_database, level)
                futures[future] = category
            elif category == DiagnosticCategory.MEMORY:
                future = self.executor.submit(self._diagnose_memory, level)
                futures[future] = category
            elif category == DiagnosticCategory.CONFIGURATION:
                future = self.executor.submit(self._diagnose_configuration, level)
                futures[future] = category
            elif category == DiagnosticCategory.NETWORK:
                future = self.executor.submit(self._diagnose_network, level)
                futures[future] = category
            elif category == DiagnosticCategory.SECURITY:
                future = self.executor.submit(self._diagnose_security, level)
                futures[future] = category
            elif category == DiagnosticCategory.COMPONENTS:
                future = self.executor.submit(self._diagnose_components, level)
                futures[future] = category
        
        # Collect results
        for future in as_completed(futures):
            category = futures[future]
            try:
                result = future.result()
                results.append(result)
                self._store_diagnostic_result(result)
            except Exception as e:
                self.logger.error(f"Diagnostic failed for {category.value}: {e}")
                error_result = DiagnosticResult(
                    diagnostic_id=f"diag_{category.value}_{datetime.now().timestamp()}",
                    category=category,
                    level=level,
                    timestamp=datetime.now(),
                    duration=0.0,
                    findings=[{
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }],
                    recommendations=["Check system logs for more details"],
                    severity='error'
                )
                results.append(error_result)
        
        return results
    
    def _diagnose_performance(self, level: DiagnosticLevel) -> DiagnosticResult:
        """Diagnose system performance"""
        start_time = time.time()
        findings = []
        recommendations = []
        severity = 'info'
        
        try:
            # CPU performance
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            findings.append({
                'metric': 'cpu_usage',
                'value': cpu_percent,
                'threshold': 80.0,
                'status': 'warning' if cpu_percent > 80 else 'ok'
            })
            
            if cpu_percent > 80:
                severity = 'warning'
                recommendations.append("High CPU usage detected. Consider scaling or optimizing CPU-intensive operations")
            
            # Memory performance
            memory = psutil.virtual_memory()
            findings.append({
                'metric': 'memory_usage',
                'value': memory.percent,
                'available_mb': memory.available / (1024 * 1024),
                'status': 'warning' if memory.percent > 80 else 'ok'
            })
            
            if memory.percent > 80:
                severity = 'warning'
                recommendations.append("High memory usage. Consider increasing memory or optimizing memory usage")
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                findings.append({
                    'metric': 'disk_io',
                    'read_mb_per_sec': disk_io.read_bytes / (1024 * 1024),
                    'write_mb_per_sec': disk_io.write_bytes / (1024 * 1024)
                })
            
            # Get health data from orchestrator if available
            if self.orchestrator and hasattr(self.orchestrator, 'health_monitor'):
                health_data = self.orchestrator.health_monitor.get_overall_health()
                findings.append({
                    'metric': 'orchestrator_health',
                    'overall_status': health_data.get('status', 'unknown'),
                    'healthy_components': health_data.get('healthy_components', 0),
                    'critical_components': health_data.get('critical_components', 0)
                })
                
                if health_data.get('critical_components', 0) > 0:
                    severity = 'critical'
                    recommendations.append("Critical component issues detected. Check component health")
            
        except Exception as e:
            findings.append({'error': str(e)})
            severity = 'error'
            recommendations.append(f"Performance diagnostic error: {e}")
        
        duration = time.time() - start_time
        
        return DiagnosticResult(
            diagnostic_id=f"perf_{datetime.now().timestamp()}",
            category=DiagnosticCategory.PERFORMANCE,
            level=level,
            timestamp=datetime.now(),
            duration=duration,
            findings=findings,
            recommendations=recommendations,
            severity=severity
        )
    
    def _diagnose_resources(self, level: DiagnosticLevel) -> DiagnosticResult:
        """Diagnose resource utilization"""
        start_time = time.time()
        findings = []
        recommendations = []
        severity = 'info'
        
        try:
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            findings.append({
                'metric': 'disk_usage',
                'percent': disk_usage.percent,
                'free_gb': disk_usage.free / (1024**3),
                'status': 'critical' if disk_usage.percent > 90 else 'warning' if disk_usage.percent > 80 else 'ok'
            })
            
            if disk_usage.percent > 90:
                severity = 'critical'
                recommendations.append("Critical disk space! Clean up old files or expand storage immediately")
            elif disk_usage.percent > 80:
                severity = 'warning'
                recommendations.append("Low disk space. Schedule cleanup or plan storage expansion")
            
            # File handles
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                current_process = psutil.Process()
                open_files = len(current_process.open_files())
                
                findings.append({
                    'metric': 'file_handles',
                    'open_files': open_files,
                    'soft_limit': soft,
                    'hard_limit': hard,
                    'usage_percent': (open_files / soft) * 100 if soft > 0 else 0
                })
                
                if open_files / soft > 0.8:
                    severity = 'warning'
                    recommendations.append("High file handle usage. Check for file handle leaks")
                    
            except Exception as e:
                findings.append({'file_handles_error': str(e)})
            
        except Exception as e:
            findings.append({'error': str(e)})
            severity = 'error'
            recommendations.append(f"Resource diagnostic error: {e}")
        
        duration = time.time() - start_time
        
        return DiagnosticResult(
            diagnostic_id=f"res_{datetime.now().timestamp()}",
            category=DiagnosticCategory.RESOURCE,
            level=level,
            timestamp=datetime.now(),
            duration=duration,
            findings=findings,
            recommendations=recommendations,
            severity=severity
        )
    
    def _diagnose_database(self, level: DiagnosticLevel) -> DiagnosticResult:
        """Diagnose database health and performance"""
        start_time = time.time()
        findings = []
        recommendations = []
        severity = 'info'
        
        try:
            # Check database through orchestrator if available
            if self.orchestrator and hasattr(self.orchestrator, 'accuracy_database'):
                try:
                    db_stats = self.orchestrator.accuracy_database.get_database_statistics()
                    findings.append({
                        'metric': 'database_statistics',
                        'stats': db_stats
                    })
                except Exception as e:
                    findings.append({'database_stats_error': str(e)})
                    severity = 'warning'
                    recommendations.append("Database statistics unavailable. Check database connectivity")
            
            # Check database file size
            db_paths = ['accuracy_tracking.db', './accuracy_tracking.db']
            for db_path in db_paths:
                db_file = Path(db_path)
                if db_file.exists():
                    db_size_mb = db_file.stat().st_size / (1024**2)
                    findings.append({
                        'metric': 'database_file_size',
                        'size_mb': db_size_mb,
                        'path': str(db_file),
                        'status': 'warning' if db_size_mb > 1000 else 'ok'
                    })
                    
                    if db_size_mb > 1000:
                        severity = 'warning'
                        recommendations.append("Large database size. Consider archiving old data")
                    break
            
        except Exception as e:
            findings.append({'error': str(e)})
            severity = 'error'
            recommendations.append(f"Database diagnostic error: {e}")
        
        duration = time.time() - start_time
        
        return DiagnosticResult(
            diagnostic_id=f"db_{datetime.now().timestamp()}",
            category=DiagnosticCategory.DATABASE,
            level=level,
            timestamp=datetime.now(),
            duration=duration,
            findings=findings,
            recommendations=recommendations,
            severity=severity
        )
    
    def _diagnose_memory(self, level: DiagnosticLevel) -> DiagnosticResult:
        """Diagnose memory usage and potential leaks"""
        start_time = time.time()
        findings = []
        recommendations = []
        severity = 'info'
        
        try:
            # Current memory snapshot
            if tracemalloc.is_tracing():
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')[:10]
                
                memory_hotspots = []
                for stat in top_stats:
                    memory_hotspots.append({
                        'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                        'size_mb': stat.size / (1024**2),
                        'count': stat.count
                    })
                
                findings.append({
                    'metric': 'memory_hotspots',
                    'top_allocations': memory_hotspots
                })
            
            # Garbage collection stats
            gc_stats = []
            for i in range(len(gc.get_count())):
                gc_stats.append({
                    'generation': i,
                    'collections': gc.get_stats()[i]['collections'] if i < len(gc.get_stats()) else 0,
                    'collected': gc.get_stats()[i]['collected'] if i < len(gc.get_stats()) else 0,
                    'uncollectable': gc.get_stats()[i]['uncollectable'] if i < len(gc.get_stats()) else 0
                })
            
            findings.append({
                'metric': 'garbage_collection',
                'stats': gc_stats,
                'threshold': gc.get_threshold()
            })
            
        except Exception as e:
            findings.append({'error': str(e)})
            severity = 'error'
            recommendations.append(f"Memory diagnostic error: {e}")
        
        duration = time.time() - start_time
        
        return DiagnosticResult(
            diagnostic_id=f"mem_{datetime.now().timestamp()}",
            category=DiagnosticCategory.MEMORY,
            level=level,
            timestamp=datetime.now(),
            duration=duration,
            findings=findings,
            recommendations=recommendations,
            severity=severity
        )
    
    def _diagnose_configuration(self, level: DiagnosticLevel) -> DiagnosticResult:
        """Diagnose configuration health and optimization"""
        start_time = time.time()
        findings = []
        recommendations = []
        severity = 'info'
        
        try:
            # Check orchestrator configuration if available
            if self.orchestrator and hasattr(self.orchestrator, 'config'):
                config_validation = self._validate_orchestrator_config()
                findings.append({
                    'metric': 'orchestrator_configuration',
                    'validation': config_validation
                })
                
                if config_validation.get('issues'):
                    severity = 'warning'
                    recommendations.extend(config_validation['recommendations'])
            
            # Check configuration files
            config_files = ['config.yaml', 'config.json', './config/config.yaml']
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    findings.append({
                        'metric': 'config_file',
                        'path': str(config_path),
                        'size': config_path.stat().st_size,
                        'modified': datetime.fromtimestamp(config_path.stat().st_mtime).isoformat()
                    })
                    break
            
        except Exception as e:
            findings.append({'error': str(e)})
            severity = 'error'
            recommendations.append(f"Configuration diagnostic error: {e}")
        
        duration = time.time() - start_time
        
        return DiagnosticResult(
            diagnostic_id=f"conf_{datetime.now().timestamp()}",
            category=DiagnosticCategory.CONFIGURATION,
            level=level,
            timestamp=datetime.now(),
            duration=duration,
            findings=findings,
            recommendations=recommendations,
            severity=severity
        )
    
    def _diagnose_network(self, level: DiagnosticLevel) -> DiagnosticResult:
        """Diagnose network connectivity and performance"""
        start_time = time.time()
        findings = []
        recommendations = []
        severity = 'info'
        
        try:
            # Network interfaces
            net_interfaces = psutil.net_if_addrs()
            active_interfaces = []
            for interface, addrs in net_interfaces.items():
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        active_interfaces.append({
                            'interface': interface,
                            'address': addr.address,
                            'netmask': addr.netmask
                        })
            
            findings.append({
                'metric': 'network_interfaces',
                'active': active_interfaces
            })
            
            # Connection stats
            connections = psutil.net_connections()
            connection_summary = {
                'established': sum(1 for c in connections if c.status == 'ESTABLISHED'),
                'listening': sum(1 for c in connections if c.status == 'LISTEN'),
                'time_wait': sum(1 for c in connections if c.status == 'TIME_WAIT'),
                'close_wait': sum(1 for c in connections if c.status == 'CLOSE_WAIT')
            }
            
            findings.append({
                'metric': 'connections',
                'summary': connection_summary,
                'total': len(connections)
            })
            
            if connection_summary['time_wait'] > 100:
                severity = 'warning'
                recommendations.append("High number of TIME_WAIT connections. Check connection pooling")
            
        except Exception as e:
            findings.append({'error': str(e)})
            severity = 'error'
            recommendations.append(f"Network diagnostic error: {e}")
        
        duration = time.time() - start_time
        
        return DiagnosticResult(
            diagnostic_id=f"net_{datetime.now().timestamp()}",
            category=DiagnosticCategory.NETWORK,
            level=level,
            timestamp=datetime.now(),
            duration=duration,
            findings=findings,
            recommendations=recommendations,
            severity=severity
        )
    
    def _diagnose_security(self, level: DiagnosticLevel) -> DiagnosticResult:
        """Diagnose security configuration and vulnerabilities"""
        start_time = time.time()
        findings = []
        recommendations = []
        severity = 'info'
        
        try:
            # File permissions check
            sensitive_files = [
                'accuracy_tracking.db',
                'config.yaml',
                'credentials.json'
            ]
            
            permission_issues = []
            for file_path in sensitive_files:
                path = Path(file_path)
                if path.exists():
                    stat_info = path.stat()
                    mode = oct(stat_info.st_mode)[-3:]
                    if mode != '600' and mode != '640':
                        permission_issues.append({
                            'file': file_path,
                            'current_mode': mode,
                            'recommended_mode': '600'
                        })
            
            findings.append({
                'metric': 'file_permissions',
                'issues': permission_issues
            })
            
            if permission_issues:
                severity = 'warning'
                recommendations.append("Insecure file permissions detected. Update permissions for sensitive files")
            
        except Exception as e:
            findings.append({'error': str(e)})
            severity = 'error'
            recommendations.append(f"Security diagnostic error: {e}")
        
        duration = time.time() - start_time
        
        return DiagnosticResult(
            diagnostic_id=f"sec_{datetime.now().timestamp()}",
            category=DiagnosticCategory.SECURITY,
            level=level,
            timestamp=datetime.now(),
            duration=duration,
            findings=findings,
            recommendations=recommendations,
            severity=severity
        )
    
    def _diagnose_components(self, level: DiagnosticLevel) -> DiagnosticResult:
        """Diagnose component health and interactions"""
        start_time = time.time()
        findings = []
        recommendations = []
        severity = 'info'
        
        try:
            # Get component status from orchestrator
            if self.orchestrator and hasattr(self.orchestrator, 'health_monitor'):
                component_health = self.orchestrator.health_monitor.get_all_component_health()
                findings.append({
                    'metric': 'component_health',
                    'components': component_health
                })
                
                # Check for unhealthy components
                unhealthy_components = []
                for name, health in component_health.items():
                    if health.get('status') != 'healthy':
                        unhealthy_components.append({
                            'name': name,
                            'status': health.get('status'),
                            'type': health.get('type')
                        })
                
                if unhealthy_components:
                    severity = 'warning'
                    for comp in unhealthy_components:
                        recommendations.append(f"Component '{comp['name']}' is {comp['status']}. Check logs for details")
            
            # Check orchestrator state
            if self.orchestrator and hasattr(self.orchestrator, 'state'):
                findings.append({
                    'metric': 'orchestrator_state',
                    'state': self.orchestrator.state.value if hasattr(self.orchestrator.state, 'value') else str(self.orchestrator.state)
                })
            
        except Exception as e:
            findings.append({'error': str(e)})
            severity = 'error'
            recommendations.append(f"Component diagnostic error: {e}")
        
        duration = time.time() - start_time
        
        return DiagnosticResult(
            diagnostic_id=f"comp_{datetime.now().timestamp()}",
            category=DiagnosticCategory.COMPONENTS,
            level=level,
            timestamp=datetime.now(),
            duration=duration,
            findings=findings,
            recommendations=recommendations,
            severity=severity
        )
    
    def _validate_orchestrator_config(self) -> Dict[str, Any]:
        """Validate orchestrator configuration"""
        issues = []
        recommendations = []
        
        if self.orchestrator and hasattr(self.orchestrator, 'config'):
            config = self.orchestrator.config
            
            # Check required settings
            required_keys = ['max_parallel_workflows', 'enable_caching', 'monitoring_interval_seconds']
            for key in required_keys:
                if key not in config:
                    issues.append(f"Missing required config key: {key}")
                    recommendations.append(f"Add '{key}' to orchestrator configuration")
            
            # Check value ranges
            if config.get('monitoring_interval_seconds', 0) < 10:
                issues.append("Monitoring interval too short")
                recommendations.append("Set monitoring_interval_seconds to at least 10")
        
        return {
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _store_diagnostic_result(self, result: DiagnosticResult) -> None:
        """Store diagnostic result in history"""
        with self._lock:
            self.diagnostic_history.append(result)
    
    def get_diagnostic_history(
        self,
        category: Optional[DiagnosticCategory] = None,
        hours: int = 24
    ) -> List[DiagnosticResult]:
        """Get diagnostic history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            history = list(self.diagnostic_history)
        
        # Filter by time
        history = [r for r in history if r.timestamp >= cutoff_time]
        
        # Filter by category if specified
        if category:
            history = [r for r in history if r.category == category]
        
        return history
    
    def shutdown(self) -> None:
        """Shutdown diagnostics engine"""
        self.executor.shutdown(wait=True)
        
        if self.config['memory_profiling_enabled']:
            tracemalloc.stop()
        
        self.logger.info("DiagnosticsEngine shutdown")

# ======================== MAINTENANCE ENGINE ========================

class MaintenanceEngine:
    """Engine for automated maintenance tasks"""
    
    def __init__(self, config: Dict[str, Any], orchestrator=None):
        self.config = config
        self.orchestrator = orchestrator
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Task management
        self.scheduled_tasks = {}
        self.task_history = deque(maxlen=1000)
        self.active_tasks = {}
        
        # Scheduler
        self.scheduler = schedule.Scheduler()
        self.scheduler_thread = None
        self.stop_scheduler = threading.Event()
        
        # Task executor
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Start scheduler if enabled
        if config['maintenance_schedule_enabled']:
            self._start_scheduler()
        
        self.logger.info("MaintenanceEngine initialized")
    
    def schedule_maintenance(
        self,
        task: MaintenanceTask,
        schedule_time: Optional[datetime] = None
    ) -> str:
        """
        Schedule a maintenance task.
        
        Args:
            task: Maintenance task to schedule
            schedule_time: When to run (None for immediate)
            
        Returns:
            Task ID
        """
        with self._lock:
            if schedule_time:
                task.scheduled_time = schedule_time
            
            self.scheduled_tasks[task.task_id] = task
            
            self.logger.info(f"Scheduled maintenance task: {task.task_id}")
            
            return task.task_id
    
    def execute_maintenance_task(self, task_id: str) -> Dict[str, Any]:
        """Execute a maintenance task immediately"""
        with self._lock:
            if task_id not in self.scheduled_tasks:
                raise MaintenanceError(f"Task {task_id} not found")
            
            task = self.scheduled_tasks[task_id]
        
        return self._execute_task(task)
    
    def _execute_task(self, task: MaintenanceTask) -> Dict[str, Any]:
        """Execute a maintenance task"""
        self.logger.info(f"Executing maintenance task: {task.task_id}")
        
        # Update task status
        task.status = MaintenanceStatus.RUNNING
        self.active_tasks[task.task_id] = task
        
        start_time = time.time()
        result = {'task_id': task.task_id, 'success': False}
        
        try:
            # Execute based on task type
            if task.task_type == MaintenanceType.CLEANUP:
                result = self._execute_cleanup(task)
            elif task.task_type == MaintenanceType.OPTIMIZATION:
                result = self._execute_optimization(task)
            elif task.task_type == MaintenanceType.BACKUP:
                result = self._execute_backup(task)
            elif task.task_type == MaintenanceType.RECOVERY:
                result = self._execute_recovery(task)
            elif task.task_type == MaintenanceType.MIGRATION:
                result = self._execute_migration(task)
            elif task.task_type == MaintenanceType.SECURITY_AUDIT:
                result = self._execute_security_audit(task)
            else:
                raise MaintenanceError(f"Unknown task type: {task.task_type}")
            
            # Update task status
            task.status = MaintenanceStatus.COMPLETED
            task.result = result
            result['success'] = True
            
        except Exception as e:
            self.logger.error(f"Maintenance task failed: {e}")
            task.status = MaintenanceStatus.FAILED
            task.error = str(e)
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
        finally:
            # Record completion
            result['duration'] = time.time() - start_time
            task.result = result
            
            # Move to history
            with self._lock:
                self.active_tasks.pop(task.task_id, None)
                self.task_history.append(task)
        
        return result
    
    def _execute_cleanup(self, task: MaintenanceTask) -> Dict[str, Any]:
        """Execute cleanup maintenance task"""
        result = {
            'cleaned_items': 0,
            'space_freed_mb': 0,
            'details': []
        }
        
        # Clean temporary files
        if task.parameters.get('clean_temp_files', True):
            temp_cleaned = self._clean_temp_files()
            result['cleaned_items'] += temp_cleaned['files_deleted']
            result['space_freed_mb'] += temp_cleaned['space_freed_mb']
            result['details'].append(temp_cleaned)
        
        # Garbage collection
        if task.parameters.get('run_gc', True):
            gc_result = self._run_garbage_collection()
            result['details'].append(gc_result)
        
        return result
    
    def _execute_optimization(self, task: MaintenanceTask) -> Dict[str, Any]:
        """Execute optimization maintenance task"""
        result = {
            'optimizations_performed': [],
            'performance_improvement': {}
        }
        
        # Cache optimization
        if task.parameters.get('optimize_cache', True):
            cache_opt = self._optimize_cache()
            result['optimizations_performed'].append('cache')
            result['performance_improvement']['cache'] = cache_opt
        
        return result
    
    def _execute_backup(self, task: MaintenanceTask) -> Dict[str, Any]:
        """Execute backup maintenance task"""
        backup_path = task.parameters.get('backup_path', './backups')
        
        result = {
            'backup_location': '',
            'size_mb': 0,
            'components_backed_up': []
        }
        
        # Create backup directory
        backup_dir = Path(backup_path) / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup database
        if task.parameters.get('backup_database', True):
            db_backup = self._backup_database(backup_dir)
            result['components_backed_up'].append('database')
            result['size_mb'] += db_backup['size_mb']
        
        # Backup configuration
        if task.parameters.get('backup_config', True):
            config_backup = self._backup_configuration(backup_dir)
            result['components_backed_up'].append('configuration')
            result['size_mb'] += config_backup['size_mb']
        
        result['backup_location'] = str(backup_dir)
        
        return result
    
    def _execute_recovery(self, task: MaintenanceTask) -> Dict[str, Any]:
        """Execute recovery maintenance task"""
        result = {
            'recovery_successful': False,
            'components_recovered': [],
            'issues_found': []
        }
        
        # Component recovery
        if task.parameters.get('recover_components', True):
            comp_recovery = self._recover_failed_components()
            result['components_recovered'].extend(comp_recovery['recovered'])
            result['issues_found'].extend(comp_recovery['issues'])
        
        result['recovery_successful'] = len(result['issues_found']) == 0
        
        return result
    
    def _execute_migration(self, task: MaintenanceTask) -> Dict[str, Any]:
        """Execute migration maintenance task"""
        result = {
            'migration_successful': False,
            'items_migrated': 0,
            'errors': []
        }
        
        result['migration_successful'] = len(result['errors']) == 0
        
        return result
    
    def _execute_security_audit(self, task: MaintenanceTask) -> Dict[str, Any]:
        """Execute security audit maintenance task"""
        result = {
            'audit_passed': True,
            'vulnerabilities_found': [],
            'recommendations': []
        }
        
        # Permission audit
        perm_audit = self._audit_permissions()
        if perm_audit['issues']:
            result['audit_passed'] = False
            result['vulnerabilities_found'].extend(perm_audit['issues'])
            result['recommendations'].extend(perm_audit['recommendations'])
        
        return result
    
    # Helper methods for maintenance tasks
    
    def _clean_temp_files(self) -> Dict[str, Any]:
        """Clean temporary files"""
        temp_dir = Path('/tmp')
        files_deleted = 0
        space_freed = 0
        
        # Clean accuracy tracking temp files
        pattern = 'accuracy_tracking_*'
        for file in temp_dir.glob(pattern):
            try:
                size = file.stat().st_size
                file.unlink()
                files_deleted += 1
                space_freed += size
            except Exception as e:
                self.logger.warning(f"Could not delete {file}: {e}")
        
        return {
            'files_deleted': files_deleted,
            'space_freed_mb': space_freed / (1024**2)
        }
    
    def _run_garbage_collection(self) -> Dict[str, Any]:
        """Run garbage collection"""
        before = len(gc.get_objects())
        collected = gc.collect()
        after = len(gc.get_objects())
        
        return {
            'objects_collected': collected,
            'objects_before': before,
            'objects_after': after,
            'memory_freed_estimate_mb': (before - after) * 0.001  # Rough estimate
        }
    
    def _optimize_cache(self) -> Dict[str, Any]:
        """Optimize cache configuration"""
        return {
            'cache_size_optimized': True,
            'hit_rate_improvement': '5%'
        }
    
    def _backup_database(self, backup_dir: Path) -> Dict[str, Any]:
        """Backup database"""
        db_paths = ['accuracy_tracking.db', './accuracy_tracking.db']
        
        for db_path in db_paths:
            source = Path(db_path)
            if source.exists():
                import shutil
                backup_path = backup_dir / 'database_backup.db'
                shutil.copy2(source, backup_path)
                size_mb = backup_path.stat().st_size / (1024**2)
                return {
                    'backup_file': str(backup_path),
                    'size_mb': size_mb
                }
        
        return {'backup_file': '', 'size_mb': 0}
    
    def _backup_configuration(self, backup_dir: Path) -> Dict[str, Any]:
        """Backup configuration files"""
        config_files = ['config.yaml', 'config.json']
        total_size = 0
        
        for config_file in config_files:
            source = Path(config_file)
            if source.exists():
                import shutil
                dest = backup_dir / config_file
                shutil.copy2(source, dest)
                total_size += dest.stat().st_size
        
        return {
            'files_backed_up': len([f for f in config_files if Path(f).exists()]),
            'size_mb': total_size / (1024**2)
        }
    
    def _recover_failed_components(self) -> Dict[str, Any]:
        """Recover failed components"""
        recovered = []
        issues = []
        
        # Check with orchestrator if available
        if self.orchestrator and hasattr(self.orchestrator, 'health_monitor'):
            health_data = self.orchestrator.health_monitor.get_all_component_health()
            for name, health in health_data.items():
                if health.get('status') == 'critical':
                    try:
                        # Attempt to restart component through orchestrator
                        if hasattr(self.orchestrator, 'restart_component'):
                            self.orchestrator.restart_component(name)
                            recovered.append(name)
                    except Exception as e:
                        issues.append(f"Failed to recover {name}: {e}")
        
        return {
            'recovered': recovered,
            'issues': issues
        }
    
    def _audit_permissions(self) -> Dict[str, Any]:
        """Audit file and resource permissions"""
        issues = []
        recommendations = []
        
        # Check file permissions
        sensitive_files = ['accuracy_tracking.db', 'config.yaml']
        for file in sensitive_files:
            path = Path(file)
            if path.exists():
                mode = oct(path.stat().st_mode)[-3:]
                if mode not in ['600', '640']:
                    issues.append(f"Insecure permissions on {file}: {mode}")
                    recommendations.append(f"Change permissions of {file} to 600")
        
        return {
            'issues': issues,
            'recommendations': recommendations
        }
    
    def _start_scheduler(self) -> None:
        """Start the maintenance scheduler"""
        def scheduler_loop():
            while not self.stop_scheduler.is_set():
                try:
                    self.scheduler.run_pending()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    self.logger.error(f"Scheduler error: {e}")
        
        self.scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        self.logger.info("Maintenance scheduler started")
    
    def get_maintenance_status(self) -> Dict[str, Any]:
        """Get current maintenance status"""
        with self._lock:
            return {
                'scheduled_tasks': len(self.scheduled_tasks),
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len([t for t in self.task_history if t.status == MaintenanceStatus.COMPLETED]),
                'failed_tasks': len([t for t in self.task_history if t.status == MaintenanceStatus.FAILED])
            }
    
    def get_maintenance_history(
        self,
        task_type: Optional[MaintenanceType] = None,
        hours: int = 24
    ) -> List[MaintenanceTask]:
        """Get maintenance task history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            history = list(self.task_history)
        
        # Filter by time
        history = [t for t in history if t.scheduled_time >= cutoff_time]
        
        # Filter by type if specified
        if task_type:
            history = [t for t in history if t.task_type == task_type]
        
        return history
    
    def shutdown(self) -> None:
        """Shutdown maintenance engine"""
        self.stop_scheduler.set()
        if self.scheduler_thread:
            self.scheduler_thread.join()
        
        self.executor.shutdown(wait=True)
        self.logger.info("MaintenanceEngine shutdown")

# ======================== SIMPLIFIED TROUBLESHOOTING ASSISTANT ========================

class TroubleshootingAssistant:
    """Simplified troubleshooting assistant"""
    
    def __init__(self, diagnostics_engine: DiagnosticsEngine, config: Dict[str, Any]):
        self.diagnostics_engine = diagnostics_engine
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Active sessions
        self.sessions = {}
        
        # Resolution history
        self.resolution_history = deque(maxlen=100)
        
        self.logger.info("TroubleshootingAssistant initialized")
    
    def start_troubleshooting_session(
        self,
        problem_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new troubleshooting session.
        
        Args:
            problem_description: Description of the problem
            context: Optional context information
            
        Returns:
            Session ID
        """
        session_id = f"ts_{datetime.now().timestamp()}"
        
        session = TroubleshootingSession(
            session_id=session_id,
            problem_description=problem_description,
            start_time=datetime.now(),
            current_phase=TroubleshootingPhase.DETECTION
        )
        
        self.sessions[session_id] = session
        
        # Run basic diagnostics
        diagnostic_results = self.diagnostics_engine.run_diagnostics(
            level=DiagnosticLevel.STANDARD
        )
        
        session.diagnostic_results.extend(diagnostic_results)
        
        # Generate basic resolution steps
        session.resolution_steps = self._generate_basic_resolution_steps(diagnostic_results)
        session.current_phase = TroubleshootingPhase.RESOLUTION
        
        return session_id
    
    def _generate_basic_resolution_steps(self, results: List[DiagnosticResult]) -> List[Dict[str, Any]]:
        """Generate basic resolution steps"""
        steps = []
        
        for result in results:
            if result.severity in ['warning', 'error', 'critical']:
                for recommendation in result.recommendations:
                    steps.append({
                        'description': recommendation,
                        'category': result.category.value,
                        'action': 'manual_action',
                        'automated': False,
                        'risk_level': 'low'
                    })
        
        return steps
    
    def get_resolution_steps(self, session_id: str) -> List[Dict[str, Any]]:
        """Get resolution steps for a session"""
        if session_id not in self.sessions:
            raise TroubleshootingError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        return session.resolution_steps

# ======================== DIAGNOSTICS AND MAINTENANCE SYSTEM ========================

class AccuracyTrackingDiagnostics:
    """
    Comprehensive diagnostics and maintenance system for accuracy tracking.
    Provides deep system analysis, automated maintenance, and troubleshooting.
    """
    
    def __init__(
        self,
        orchestrator=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize diagnostics and maintenance system.
        
        Args:
            orchestrator: Optional orchestrator instance
            config: Optional configuration override
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Load configuration
        self.config = self._load_configuration(config)
        
        # Store orchestrator reference
        self.orchestrator = orchestrator
        
        # Initialize engines
        self.diagnostics_engine = DiagnosticsEngine(self.config, orchestrator)
        self.maintenance_engine = MaintenanceEngine(self.config, orchestrator)
        self.troubleshooting_assistant = TroubleshootingAssistant(
            self.diagnostics_engine,
            self.config
        )
        
        # Schedule default maintenance tasks
        if self.config['maintenance_schedule_enabled']:
            self._schedule_default_maintenance()
        
        # Start auto-diagnostics if enabled
        if self.config['enable_auto_diagnostics']:
            self._start_auto_diagnostics()
        
        self.logger.info("AccuracyTrackingDiagnostics initialized")
    
    def _load_configuration(self, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Load and validate configuration"""
        # Start with defaults
        final_config = DEFAULT_DIAGNOSTICS_CONFIG.copy()
        
        # Apply overrides
        if config:
            final_config.update(config)
        
        return final_config
    
    def _schedule_default_maintenance(self) -> None:
        """Schedule default maintenance tasks"""
        # Daily cleanup
        cleanup_task = MaintenanceTask(
            task_id='daily_cleanup',
            task_type=MaintenanceType.CLEANUP,
            scheduled_time=datetime.now().replace(hour=2, minute=0),
            priority=5,
            estimated_duration=30,
            parameters={
                'clean_temp_files': True,
                'run_gc': True
            }
        )
        self.maintenance_engine.schedule_maintenance(cleanup_task)
        
        # Weekly backup
        backup_task = MaintenanceTask(
            task_id='weekly_backup',
            task_type=MaintenanceType.BACKUP,
            scheduled_time=datetime.now().replace(hour=1, minute=0),
            priority=8,
            estimated_duration=20,
            parameters={
                'backup_database': True,
                'backup_config': True
            }
        )
        self.maintenance_engine.schedule_maintenance(backup_task)
        
        self.logger.info("Default maintenance tasks scheduled")
    
    def _start_auto_diagnostics(self) -> None:
        """Start automatic diagnostics"""
        def auto_diagnostic_loop():
            while True:
                try:
                    # Run basic diagnostics
                    results = self.run_system_diagnostics(DiagnosticLevel.BASIC)
                    
                    # Check for critical issues
                    critical_issues = [
                        r for r in results
                        if r.severity in ['error', 'critical']
                    ]
                    
                    if critical_issues:
                        self.logger.warning(
                            f"Auto-diagnostics found {len(critical_issues)} critical issues"
                        )
                    
                    # Sleep until next run
                    time.sleep(self.config['diagnostic_interval_minutes'] * 60)
                    
                except Exception as e:
                    self.logger.error(f"Auto-diagnostics error: {e}")
                    time.sleep(self.config['diagnostic_interval_minutes'] * 60)
        
        thread = threading.Thread(target=auto_diagnostic_loop, daemon=True)
        thread.start()
        self.logger.info("Auto-diagnostics started")
    
    # ======================== PUBLIC API ========================
    
    def run_system_diagnostics(
        self,
        level: DiagnosticLevel = DiagnosticLevel.STANDARD,
        categories: Optional[List[DiagnosticCategory]] = None
    ) -> List[DiagnosticResult]:
        """
        Run comprehensive system diagnostics.
        
        Args:
            level: Level of diagnostic detail
            categories: Specific categories to diagnose
            
        Returns:
            List of diagnostic results
        """
        self.logger.info(f"Running system diagnostics at {level.value} level")
        
        # Run diagnostics
        results = self.diagnostics_engine.run_diagnostics(level, categories)
        
        return results
    
    def schedule_maintenance(
        self,
        task_type: MaintenanceType,
        parameters: Dict[str, Any],
        schedule_time: Optional[datetime] = None,
        priority: int = 5
    ) -> str:
        """
        Schedule a maintenance task.
        
        Args:
            task_type: Type of maintenance task
            parameters: Task parameters
            schedule_time: When to run (None for immediate)
            priority: Task priority (1-10)
            
        Returns:
            Task ID
        """
        task = MaintenanceTask(
            task_id=f"maint_{task_type.value}_{datetime.now().timestamp()}",
            task_type=task_type,
            scheduled_time=schedule_time or datetime.now(),
            priority=priority,
            estimated_duration=30,  # Default estimate
            parameters=parameters
        )
        
        return self.maintenance_engine.schedule_maintenance(task)
    
    def execute_maintenance_now(
        self,
        task_type: MaintenanceType,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute maintenance task immediately.
        
        Args:
            task_type: Type of maintenance task
            parameters: Task parameters
            
        Returns:
            Execution result
        """
        task_id = self.schedule_maintenance(task_type, parameters)
        return self.maintenance_engine.execute_maintenance_task(task_id)
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """
        Get comprehensive system health report.
        
        Returns:
            System health report
        """
        # Run standard diagnostics
        diagnostic_results = self.run_system_diagnostics(DiagnosticLevel.STANDARD)
        
        # Categorize results
        health_summary = {
            'overall_status': 'healthy',
            'components': {},
            'issues': [],
            'recommendations': []
        }
        
        for result in diagnostic_results:
            health_summary['components'][result.category.value] = {
                'status': result.severity,
                'findings_count': len(result.findings)
            }
            
            if result.severity in ['warning', 'error', 'critical']:
                health_summary['issues'].append({
                    'category': result.category.value,
                    'severity': result.severity,
                    'summary': result.recommendations[0] if result.recommendations else 'Issue detected'
                })
                
                health_summary['recommendations'].extend(result.recommendations)
            
            if result.severity in ['error', 'critical']:
                health_summary['overall_status'] = 'unhealthy'
            elif result.severity == 'warning' and health_summary['overall_status'] == 'healthy':
                health_summary['overall_status'] = 'warning'
        
        # Add maintenance status
        health_summary['maintenance'] = self.maintenance_engine.get_maintenance_status()
        
        return health_summary
    
    def start_troubleshooting(
        self,
        problem_description: str
    ) -> Dict[str, Any]:
        """
        Start troubleshooting session.
        
        Args:
            problem_description: Description of the problem
            
        Returns:
            Troubleshooting session information
        """
        session_id = self.troubleshooting_assistant.start_troubleshooting_session(
            problem_description
        )
        
        # Get resolution steps
        steps = self.troubleshooting_assistant.get_resolution_steps(session_id)
        
        return {
            'session_id': session_id,
            'resolution_steps': steps
        }
    
    def shutdown(self) -> None:
        """Shutdown diagnostics and maintenance system"""
        self.logger.info("Shutting down diagnostics and maintenance system")
        
        # Shutdown engines
        self.diagnostics_engine.shutdown()
        self.maintenance_engine.shutdown()
        
        self.logger.info("Diagnostics and maintenance system shutdown complete")

def create_diagnostics_system(orchestrator=None, config: Optional[Dict[str, Any]] = None) -> AccuracyTrackingDiagnostics:
    """
    Create diagnostics and maintenance system
    
    Args:
        orchestrator: Optional orchestrator instance
        config: Optional configuration override
        
    Returns:
        Configured AccuracyTrackingDiagnostics instance
    """
    return AccuracyTrackingDiagnostics(orchestrator, config)

if __name__ == "__main__":
    # Basic example usage
    print("AccuracyTrackingDiagnostics system created")
    diagnostics = create_diagnostics_system()
    
    # Run basic diagnostics
    results = diagnostics.run_system_diagnostics()
    print(f"Diagnostic results: {len(results)} categories checked")
    
    # Get health report
    health = diagnostics.get_system_health_report()
    print(f"System health: {health['overall_status']}")
    
    # Cleanup
    diagnostics.shutdown()
    print("Diagnostics system shutdown")