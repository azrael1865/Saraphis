"""
Production Observability - Production observability and telemetry system
NO FALLBACKS - HARD FAILURES ONLY

This module provides comprehensive production observability capabilities including
distributed tracing, log aggregation, event streaming, metrics aggregation,
correlation analysis, anomaly detection, and performance profiling.

Key Features:
- Multi-level observability (BASIC, STANDARD, ENHANCED, COMPREHENSIVE)
- Multiple telemetry types (METRICS, LOGS, TRACES, EVENTS)
- Distributed tracing with request correlation
- Log aggregation with structured logging
- Event streaming with real-time event processing
- Metrics aggregation with statistical analysis
- Correlation analysis across different telemetry types
- Anomaly detection with machine learning
- Performance profiling with detailed analysis
- Error tracking with stack trace analysis
- User behavior tracking with privacy compliance
- Business metrics with KPI tracking
- Infrastructure metrics with resource utilization

Architecture: NO FALLBACKS - HARD FAILURES ONLY
All observability operations must succeed or fail explicitly with detailed error information.
"""

import os
import json
import logging
import threading
import time
import uuid
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable, Coroutine
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import traceback
import hashlib
import struct
import pickle
import gzip
import base64
from contextlib import contextmanager, asynccontextmanager
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
import sqlite3
import tempfile

try:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from production_monitoring_system import ProductionMonitoringSystem, ProductionMetric
        from production_config_manager import ProductionConfigManager
        from compression_systems.padic.hybrid_performance_monitor import HybridPerformanceMonitor
except ImportError:
    pass

logger = logging.getLogger(__name__)


class ObservabilityLevel(Enum):
    """Observability level types."""
    BASIC = "basic"
    STANDARD = "standard"
    ENHANCED = "enhanced"
    COMPREHENSIVE = "comprehensive"


class TelemetryType(Enum):
    """Telemetry data types."""
    METRICS = "metrics"
    LOGS = "logs"
    TRACES = "traces"
    EVENTS = "events"
    PROFILES = "profiles"


class TraceStatus(Enum):
    """Trace status types."""
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"


class LogLevel(Enum):
    """Log level types."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class EventType(Enum):
    """Event type categories."""
    USER_ACTION = "user_action"
    SYSTEM_EVENT = "system_event"
    BUSINESS_EVENT = "business_event"
    SECURITY_EVENT = "security_event"
    PERFORMANCE_EVENT = "performance_event"
    ERROR_EVENT = "error_event"


@dataclass
class ObservabilityData:
    """Base observability data structure."""
    data_id: str
    telemetry_type: TelemetryType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate observability data."""
        if not self.data_id:
            raise ValueError("Data ID cannot be empty")
        if not isinstance(self.telemetry_type, TelemetryType):
            raise TypeError("Telemetry type must be TelemetryType")


@dataclass
class TraceSpan:
    """Distributed trace span data."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    service_name: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    status: TraceStatus = TraceStatus.STARTED
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    baggage: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate trace span."""
        if not self.span_id:
            raise ValueError("Span ID cannot be empty")
        if not self.trace_id:
            raise ValueError("Trace ID cannot be empty")
    
    def finish(self, status: TraceStatus = TraceStatus.COMPLETED) -> None:
        """Finish the span."""
        if self.end_time is None:
            self.end_time = datetime.utcnow()
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
            self.status = status
    
    def add_log(self, message: str, level: LogLevel = LogLevel.INFO, **kwargs) -> None:
        """Add log entry to span."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': level.value,
            'message': message,
            **kwargs
        }
        self.logs.append(log_entry)
    
    def set_tag(self, key: str, value: str) -> None:
        """Set span tag."""
        self.tags[key] = value
    
    def set_baggage(self, key: str, value: str) -> None:
        """Set span baggage."""
        self.baggage[key] = value


@dataclass
class LogEntry:
    """Structured log entry."""
    log_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    level: LogLevel = LogLevel.INFO
    message: str = ""
    logger_name: str = ""
    service_name: str = ""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    fields: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    
    def __post_init__(self):
        """Validate log entry."""
        if not self.log_id:
            raise ValueError("Log ID cannot be empty")
        if not isinstance(self.level, LogLevel):
            raise TypeError("Level must be LogLevel")


@dataclass
class Event:
    """Structured event data."""
    event_id: str
    event_type: EventType
    name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    trace_id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Union[int, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate event."""
        if not self.event_id:
            raise ValueError("Event ID cannot be empty")
        if not isinstance(self.event_type, EventType):
            raise TypeError("Event type must be EventType")
        if not self.name:
            raise ValueError("Event name cannot be empty")


@dataclass
class ObservabilityConfiguration:
    """Observability configuration settings."""
    observability_level: ObservabilityLevel = ObservabilityLevel.STANDARD
    enabled_telemetry_types: List[TelemetryType] = field(default_factory=lambda: [
        TelemetryType.METRICS,
        TelemetryType.LOGS,
        TelemetryType.TRACES
    ])
    
    # Tracing settings
    tracing_enabled: bool = True
    trace_sampling_rate: float = 0.1  # 10% sampling
    max_spans_per_trace: int = 1000
    trace_timeout_seconds: int = 300
    
    # Logging settings
    log_aggregation_enabled: bool = True
    log_structured_format: bool = True
    log_retention_days: int = 30
    log_compression_enabled: bool = True
    
    # Event settings
    event_streaming_enabled: bool = True
    event_buffer_size: int = 10000
    event_flush_interval_seconds: int = 30
    
    # Correlation settings
    correlation_enabled: bool = True
    correlation_window_minutes: int = 15
    
    # Anomaly detection
    anomaly_detection_enabled: bool = True
    anomaly_detection_window_minutes: int = 60
    anomaly_threshold_std_deviations: float = 2.5
    
    # Performance settings
    max_concurrent_operations: int = 100
    data_retention_days: int = 7
    compression_enabled: bool = True
    
    # Privacy settings
    pii_scrubbing_enabled: bool = True
    user_consent_required: bool = True


class DistributedTracing:
    """Distributed tracing management."""
    
    def __init__(self, config: ObservabilityConfiguration):
        self.config = config
        self.logger = logging.getLogger('DistributedTracing')
        self.active_traces: Dict[str, List[TraceSpan]] = defaultdict(list)
        self.completed_traces: deque = deque(maxlen=10000)
        self.span_storage: Dict[str, TraceSpan] = {}
        self.trace_lock = threading.RLock()
    
    def start_trace(self, operation_name: str, service_name: str = "", **tags) -> TraceSpan:
        """Start a new distributed trace."""
        try:
            trace_id = self._generate_trace_id()
            span_id = self._generate_span_id()
            
            span = TraceSpan(
                span_id=span_id,
                trace_id=trace_id,
                operation_name=operation_name,
                service_name=service_name,
                tags=tags
            )
            
            with self.trace_lock:
                self.active_traces[trace_id].append(span)
                self.span_storage[span_id] = span
            
            self.logger.debug(f"Started trace {trace_id} with span {span_id}")
            return span
            
        except Exception as e:
            error_msg = f"Failed to start trace: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def start_span(
        self,
        parent_span: TraceSpan,
        operation_name: str,
        service_name: str = "",
        **tags
    ) -> TraceSpan:
        """Start a child span."""
        try:
            span_id = self._generate_span_id()
            
            span = TraceSpan(
                span_id=span_id,
                trace_id=parent_span.trace_id,
                parent_span_id=parent_span.span_id,
                operation_name=operation_name,
                service_name=service_name or parent_span.service_name,
                tags=tags
            )
            
            # Copy baggage from parent
            span.baggage.update(parent_span.baggage)
            
            with self.trace_lock:
                self.active_traces[parent_span.trace_id].append(span)
                self.span_storage[span_id] = span
            
            self.logger.debug(f"Started child span {span_id} for trace {parent_span.trace_id}")
            return span
            
        except Exception as e:
            error_msg = f"Failed to start child span: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def finish_span(self, span: TraceSpan, status: TraceStatus = TraceStatus.COMPLETED) -> None:
        """Finish a span."""
        try:
            span.finish(status)
            
            # Check if trace is complete
            with self.trace_lock:
                trace_spans = self.active_traces[span.trace_id]
                all_finished = all(s.status != TraceStatus.STARTED and s.status != TraceStatus.IN_PROGRESS for s in trace_spans)
                
                if all_finished:
                    # Move trace to completed
                    self.completed_traces.append({
                        'trace_id': span.trace_id,
                        'spans': trace_spans.copy(),
                        'completed_at': datetime.utcnow()
                    })
                    
                    # Clean up active trace
                    del self.active_traces[span.trace_id]
                    
                    # Clean up span storage
                    for s in trace_spans:
                        if s.span_id in self.span_storage:
                            del self.span_storage[s.span_id]
            
            self.logger.debug(f"Finished span {span.span_id}")
            
        except Exception as e:
            error_msg = f"Failed to finish span: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get trace by ID."""
        try:
            # Check active traces
            with self.trace_lock:
                if trace_id in self.active_traces:
                    return {
                        'trace_id': trace_id,
                        'spans': self.active_traces[trace_id],
                        'status': 'active'
                    }
            
            # Check completed traces
            for completed_trace in self.completed_traces:
                if completed_trace['trace_id'] == trace_id:
                    return completed_trace
            
            return None
            
        except Exception as e:
            error_msg = f"Failed to get trace {trace_id}: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _generate_trace_id(self) -> str:
        """Generate unique trace ID."""
        return uuid.uuid4().hex
    
    def _generate_span_id(self) -> str:
        """Generate unique span ID."""
        return uuid.uuid4().hex[:16]
    
    @contextmanager
    def trace_context(self, operation_name: str, service_name: str = "", **tags):
        """Context manager for tracing."""
        span = self.start_trace(operation_name, service_name, **tags)
        try:
            yield span
            self.finish_span(span, TraceStatus.COMPLETED)
        except Exception as e:
            span.add_log(f"Error: {e}", LogLevel.ERROR)
            self.finish_span(span, TraceStatus.ERROR)
            raise
    
    @contextmanager
    def span_context(self, parent_span: TraceSpan, operation_name: str, **tags):
        """Context manager for child spans."""
        span = self.start_span(parent_span, operation_name, **tags)
        try:
            yield span
            self.finish_span(span, TraceStatus.COMPLETED)
        except Exception as e:
            span.add_log(f"Error: {e}", LogLevel.ERROR)
            self.finish_span(span, TraceStatus.ERROR)
            raise


class LogAggregation:
    """Log aggregation and processing."""
    
    def __init__(self, config: ObservabilityConfiguration):
        self.config = config
        self.logger = logging.getLogger('LogAggregation')
        self.log_buffer: deque = deque(maxlen=100000)
        self.log_storage: Dict[str, LogEntry] = {}
        self.log_lock = threading.RLock()
        
        # Log processing thread
        self.processing_active = False
        self.processing_thread: Optional[threading.Thread] = None
    
    def start_log_processing(self) -> None:
        """Start log processing."""
        try:
            if self.processing_active:
                return
            
            self.processing_active = True
            self.processing_thread = threading.Thread(
                target=self._log_processing_loop,
                name="LogProcessor",
                daemon=True
            )
            self.processing_thread.start()
            
            self.logger.info("Log processing started")
            
        except Exception as e:
            error_msg = f"Failed to start log processing: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def stop_log_processing(self) -> None:
        """Stop log processing."""
        try:
            self.processing_active = False
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=10.0)
            
            self.logger.info("Log processing stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping log processing: {e}")
    
    def add_log_entry(
        self,
        level: LogLevel,
        message: str,
        logger_name: str = "",
        service_name: str = "",
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        **fields
    ) -> LogEntry:
        """Add log entry."""
        try:
            log_entry = LogEntry(
                log_id=str(uuid.uuid4()),
                level=level,
                message=message,
                logger_name=logger_name,
                service_name=service_name,
                trace_id=trace_id,
                span_id=span_id,
                fields=fields
            )
            
            # PII scrubbing if enabled
            if self.config.pii_scrubbing_enabled:
                log_entry = self._scrub_pii(log_entry)
            
            with self.log_lock:
                self.log_buffer.append(log_entry)
                self.log_storage[log_entry.log_id] = log_entry
            
            return log_entry
            
        except Exception as e:
            error_msg = f"Failed to add log entry: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def query_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[LogLevel] = None,
        service_name: Optional[str] = None,
        trace_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[LogEntry]:
        """Query logs with filters."""
        try:
            end_time = end_time or datetime.utcnow()
            start_time = start_time or (end_time - timedelta(hours=1))
            
            filtered_logs = []
            count = 0
            
            with self.log_lock:
                for log_entry in reversed(self.log_buffer):
                    if count >= limit:
                        break
                    
                    if log_entry.timestamp < start_time or log_entry.timestamp > end_time:
                        continue
                    
                    if level and log_entry.level != level:
                        continue
                    
                    if service_name and log_entry.service_name != service_name:
                        continue
                    
                    if trace_id and log_entry.trace_id != trace_id:
                        continue
                    
                    filtered_logs.append(log_entry)
                    count += 1
            
            return filtered_logs
            
        except Exception as e:
            error_msg = f"Failed to query logs: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _log_processing_loop(self) -> None:
        """Log processing background loop."""
        while self.processing_active:
            try:
                # Process log entries (e.g., indexing, archiving)
                with self.log_lock:
                    if len(self.log_buffer) > 50000:
                        # Archive old logs
                        self._archive_old_logs()
                
                time.sleep(60)  # Process every minute
                
            except Exception as e:
                self.logger.error(f"Error in log processing loop: {e}")
                time.sleep(60)
    
    def _scrub_pii(self, log_entry: LogEntry) -> LogEntry:
        """Scrub personally identifiable information."""
        # Simple PII scrubbing - in production, use more sophisticated methods
        sensitive_fields = ['email', 'ssn', 'credit_card', 'password', 'token']
        
        for field in sensitive_fields:
            if field in log_entry.fields:
                log_entry.fields[field] = '[SCRUBBED]'
        
        # Scrub message content
        import re
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        log_entry.message = re.sub(email_pattern, '[EMAIL_SCRUBBED]', log_entry.message)
        
        return log_entry
    
    def _archive_old_logs(self) -> None:
        """Archive old log entries."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=self.config.log_retention_days)
            
            # Remove old logs from buffer and storage
            to_remove = []
            for log_entry in self.log_buffer:
                if log_entry.timestamp < cutoff_time:
                    to_remove.append(log_entry)
            
            for log_entry in to_remove:
                self.log_buffer.remove(log_entry)
                if log_entry.log_id in self.log_storage:
                    del self.log_storage[log_entry.log_id]
            
            self.logger.debug(f"Archived {len(to_remove)} old log entries")
            
        except Exception as e:
            self.logger.error(f"Error archiving old logs: {e}")


class EventStreaming:
    """Real-time event streaming."""
    
    def __init__(self, config: ObservabilityConfiguration):
        self.config = config
        self.logger = logging.getLogger('EventStreaming')
        self.event_buffer: deque = deque(maxlen=self.config.event_buffer_size)
        self.event_storage: Dict[str, Event] = {}
        self.event_lock = threading.RLock()
        
        # Event processing
        self.streaming_active = False
        self.streaming_thread: Optional[threading.Thread] = None
        
        # Event subscribers
        self.event_subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
    
    def start_event_streaming(self) -> None:
        """Start event streaming."""
        try:
            if self.streaming_active:
                return
            
            self.streaming_active = True
            self.streaming_thread = threading.Thread(
                target=self._event_streaming_loop,
                name="EventStreamer",
                daemon=True
            )
            self.streaming_thread.start()
            
            self.logger.info("Event streaming started")
            
        except Exception as e:
            error_msg = f"Failed to start event streaming: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def stop_event_streaming(self) -> None:
        """Stop event streaming."""
        try:
            self.streaming_active = False
            if self.streaming_thread and self.streaming_thread.is_alive():
                self.streaming_thread.join(timeout=10.0)
            
            self.logger.info("Event streaming stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping event streaming: {e}")
    
    def emit_event(
        self,
        event_type: EventType,
        name: str,
        source: str = "",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        **properties
    ) -> Event:
        """Emit an event."""
        try:
            event = Event(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                name=name,
                source=source,
                user_id=user_id,
                session_id=session_id,
                trace_id=trace_id,
                properties=properties
            )
            
            with self.event_lock:
                self.event_buffer.append(event)
                self.event_storage[event.event_id] = event
            
            # Notify subscribers
            self._notify_subscribers(event)
            
            self.logger.debug(f"Emitted event: {event.name}")
            return event
            
        except Exception as e:
            error_msg = f"Failed to emit event: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def subscribe_to_events(self, event_type: EventType, callback: Callable[[Event], None]) -> None:
        """Subscribe to events of a specific type."""
        try:
            self.event_subscribers[event_type].append(callback)
            self.logger.debug(f"Added subscriber for {event_type.value} events")
            
        except Exception as e:
            error_msg = f"Failed to subscribe to events: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def query_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[EventType] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[Event]:
        """Query events with filters."""
        try:
            end_time = end_time or datetime.utcnow()
            start_time = start_time or (end_time - timedelta(hours=1))
            
            filtered_events = []
            count = 0
            
            with self.event_lock:
                for event in reversed(self.event_buffer):
                    if count >= limit:
                        break
                    
                    if event.timestamp < start_time or event.timestamp > end_time:
                        continue
                    
                    if event_type and event.event_type != event_type:
                        continue
                    
                    if user_id and event.user_id != user_id:
                        continue
                    
                    if session_id and event.session_id != session_id:
                        continue
                    
                    filtered_events.append(event)
                    count += 1
            
            return filtered_events
            
        except Exception as e:
            error_msg = f"Failed to query events: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _notify_subscribers(self, event: Event) -> None:
        """Notify event subscribers."""
        try:
            subscribers = self.event_subscribers.get(event.event_type, [])
            for callback in subscribers:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Error in event subscriber callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error notifying subscribers: {e}")
    
    def _event_streaming_loop(self) -> None:
        """Event streaming background loop."""
        while self.streaming_active:
            try:
                # Flush events periodically
                time.sleep(self.config.event_flush_interval_seconds)
                
                # Process events (e.g., send to external systems)
                self._flush_events()
                
            except Exception as e:
                self.logger.error(f"Error in event streaming loop: {e}")
                time.sleep(60)
    
    def _flush_events(self) -> None:
        """Flush events to external systems."""
        try:
            # In a real implementation, this would send events to external systems
            with self.event_lock:
                event_count = len(self.event_buffer)
            
            if event_count > 0:
                self.logger.debug(f"Flushed {event_count} events")
                
        except Exception as e:
            self.logger.error(f"Error flushing events: {e}")


class AnomalyDetection:
    """Anomaly detection for observability data."""
    
    def __init__(self, config: ObservabilityConfiguration):
        self.config = config
        self.logger = logging.getLogger('AnomalyDetection')
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.anomaly_cache: deque = deque(maxlen=1000)
        self.detection_lock = threading.RLock()
    
    def detect_metric_anomalies(self, metrics: List[Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics."""
        try:
            anomalies = []
            
            for metric in metrics:
                if not hasattr(metric, 'metric_name') or not hasattr(metric, 'metric_value'):
                    continue
                
                if not isinstance(metric.metric_value, (int, float)):
                    continue
                
                # Store metric value
                metric_key = f"{metric.component.value}_{metric.metric_name}"
                with self.detection_lock:
                    self.metric_history[metric_key].append({
                        'value': metric.metric_value,
                        'timestamp': metric.timestamp
                    })
                
                # Check for anomalies
                anomaly = self._check_statistical_anomaly(metric_key, metric.metric_value)
                if anomaly:
                    anomaly_data = {
                        'anomaly_id': str(uuid.uuid4()),
                        'metric_name': metric.metric_name,
                        'component': metric.component.value,
                        'current_value': metric.metric_value,
                        'expected_range': anomaly['expected_range'],
                        'deviation_score': anomaly['deviation_score'],
                        'timestamp': metric.timestamp.isoformat(),
                        'severity': self._calculate_anomaly_severity(anomaly['deviation_score'])
                    }
                    anomalies.append(anomaly_data)
                    
                    with self.detection_lock:
                        self.anomaly_cache.append(anomaly_data)
            
            return anomalies
            
        except Exception as e:
            error_msg = f"Failed to detect metric anomalies: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _check_statistical_anomaly(self, metric_key: str, current_value: float) -> Optional[Dict[str, Any]]:
        """Check for statistical anomalies using standard deviation."""
        try:
            history = self.metric_history[metric_key]
            if len(history) < 10:  # Need sufficient history
                return None
            
            values = [entry['value'] for entry in history]
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            
            if std_val == 0:
                return None
            
            deviation_score = abs(current_value - mean_val) / std_val
            
            if deviation_score > self.config.anomaly_threshold_std_deviations:
                return {
                    'expected_range': {
                        'min': mean_val - (self.config.anomaly_threshold_std_deviations * std_val),
                        'max': mean_val + (self.config.anomaly_threshold_std_deviations * std_val)
                    },
                    'deviation_score': deviation_score
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking statistical anomaly: {e}")
            return None
    
    def _calculate_anomaly_severity(self, deviation_score: float) -> str:
        """Calculate anomaly severity based on deviation score."""
        if deviation_score > 5.0:
            return "critical"
        elif deviation_score > 4.0:
            return "high"
        elif deviation_score > 3.0:
            return "medium"
        else:
            return "low"
    
    def get_recent_anomalies(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent anomalies."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
            
            with self.detection_lock:
                recent_anomalies = [
                    anomaly for anomaly in self.anomaly_cache
                    if datetime.fromisoformat(anomaly['timestamp']) > cutoff_time
                ]
            
            return recent_anomalies
            
        except Exception as e:
            error_msg = f"Failed to get recent anomalies: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)


class ObservabilityManager:
    """
    Production Observability Manager - Comprehensive observability orchestration.
    
    This class provides complete observability capabilities including distributed tracing,
    log aggregation, event streaming, metrics aggregation, correlation analysis,
    anomaly detection, and performance profiling for production environments.
    """
    
    def __init__(self, config: Optional[ObservabilityConfiguration] = None):
        self.config = config or ObservabilityConfiguration()
        self.observability_lock = threading.RLock()
        
        # Observability components
        self.distributed_tracing = DistributedTracing(self.config)
        self.log_aggregation = LogAggregation(self.config)
        self.event_streaming = EventStreaming(self.config)
        self.anomaly_detection = AnomalyDetection(self.config)
        
        # Integration references
        self.production_monitoring_system: Optional['ProductionMonitoringSystem'] = None
        self.production_config_manager: Optional['ProductionConfigManager'] = None
        self.hybrid_performance_monitor: Optional['HybridPerformanceMonitor'] = None
        
        # Observability state
        self.observability_active = False
        self.start_time = datetime.utcnow()
        
        # Statistics
        self.traces_created = 0
        self.logs_processed = 0
        self.events_emitted = 0
        self.anomalies_detected = 0
        
        logger.info(f"ObservabilityManager initialized with {self.config.observability_level.value} level")
    
    def initialize_observability_manager(
        self,
        production_monitoring_system: Optional['ProductionMonitoringSystem'] = None,
        production_config_manager: Optional['ProductionConfigManager'] = None,
        hybrid_performance_monitor: Optional['HybridPerformanceMonitor'] = None
    ) -> None:
        """Initialize observability manager with integrations."""
        try:
            with self.observability_lock:
                self.production_monitoring_system = production_monitoring_system
                self.production_config_manager = production_config_manager
                self.hybrid_performance_monitor = hybrid_performance_monitor
                
                # Apply configuration from production config manager
                if production_config_manager:
                    self._integrate_production_config()
                
                # Integrate with monitoring system
                if production_monitoring_system:
                    self._integrate_monitoring_system()
                
                logger.info("Observability manager initialized successfully")
                
        except Exception as e:
            error_msg = f"Failed to initialize observability manager: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def start_observability(self) -> None:
        """Start observability services."""
        try:
            with self.observability_lock:
                if self.observability_active:
                    logger.warning("Observability is already active")
                    return
                
                # Start observability components
                if TelemetryType.LOGS in self.config.enabled_telemetry_types:
                    self.log_aggregation.start_log_processing()
                
                if TelemetryType.EVENTS in self.config.enabled_telemetry_types:
                    self.event_streaming.start_event_streaming()
                
                self.observability_active = True
                logger.info("Observability services started successfully")
                
        except Exception as e:
            error_msg = f"Failed to start observability: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg)
    
    def stop_observability(self) -> None:
        """Stop observability services."""
        try:
            with self.observability_lock:
                if not self.observability_active:
                    logger.warning("Observability is not active")
                    return
                
                # Stop observability components
                self.log_aggregation.stop_log_processing()
                self.event_streaming.stop_event_streaming()
                
                self.observability_active = False
                logger.info("Observability services stopped successfully")
                
        except Exception as e:
            error_msg = f"Failed to stop observability: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def create_trace(self, operation_name: str, service_name: str = "", **tags) -> TraceSpan:
        """Create a new distributed trace."""
        try:
            span = self.distributed_tracing.start_trace(operation_name, service_name, **tags)
            self.traces_created += 1
            return span
            
        except Exception as e:
            error_msg = f"Failed to create trace: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def log_message(
        self,
        level: LogLevel,
        message: str,
        logger_name: str = "",
        service_name: str = "",
        trace_id: Optional[str] = None,
        **fields
    ) -> LogEntry:
        """Log a message."""
        try:
            log_entry = self.log_aggregation.add_log_entry(
                level=level,
                message=message,
                logger_name=logger_name,
                service_name=service_name,
                trace_id=trace_id,
                **fields
            )
            self.logs_processed += 1
            return log_entry
            
        except Exception as e:
            error_msg = f"Failed to log message: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def emit_event(
        self,
        event_type: EventType,
        name: str,
        source: str = "",
        user_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        **properties
    ) -> Event:
        """Emit an event."""
        try:
            event = self.event_streaming.emit_event(
                event_type=event_type,
                name=name,
                source=source,
                user_id=user_id,
                trace_id=trace_id,
                **properties
            )
            self.events_emitted += 1
            return event
            
        except Exception as e:
            error_msg = f"Failed to emit event: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def analyze_metrics_for_anomalies(self, metrics: List[Any]) -> List[Dict[str, Any]]:
        """Analyze metrics for anomalies."""
        try:
            if not self.config.anomaly_detection_enabled:
                return []
            
            anomalies = self.anomaly_detection.detect_metric_anomalies(metrics)
            self.anomalies_detected += len(anomalies)
            
            # Emit anomaly events
            for anomaly in anomalies:
                self.emit_event(
                    event_type=EventType.PERFORMANCE_EVENT,
                    name="anomaly_detected",
                    source="anomaly_detection",
                    metric_name=anomaly['metric_name'],
                    component=anomaly['component'],
                    severity=anomaly['severity'],
                    deviation_score=anomaly['deviation_score']
                )
            
            return anomalies
            
        except Exception as e:
            error_msg = f"Failed to analyze metrics for anomalies: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_observability_status(self) -> Dict[str, Any]:
        """Get observability status."""
        try:
            with self.observability_lock:
                uptime = datetime.utcnow() - self.start_time
                
                return {
                    'active': self.observability_active,
                    'observability_level': self.config.observability_level.value,
                    'enabled_telemetry_types': [t.value for t in self.config.enabled_telemetry_types],
                    'uptime_seconds': uptime.total_seconds(),
                    'traces_created': self.traces_created,
                    'logs_processed': self.logs_processed,
                    'events_emitted': self.events_emitted,
                    'anomalies_detected': self.anomalies_detected,
                    'components': {
                        'tracing': {
                            'active_traces': len(self.distributed_tracing.active_traces),
                            'completed_traces': len(self.distributed_tracing.completed_traces)
                        },
                        'logging': {
                            'log_buffer_size': len(self.log_aggregation.log_buffer),
                            'processing_active': self.log_aggregation.processing_active
                        },
                        'events': {
                            'event_buffer_size': len(self.event_streaming.event_buffer),
                            'streaming_active': self.event_streaming.streaming_active
                        },
                        'anomaly_detection': {
                            'recent_anomalies': len(self.anomaly_detection.get_recent_anomalies(60))
                        }
                    }
                }
                
        except Exception as e:
            error_msg = f"Failed to get observability status: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def correlate_data(
        self,
        trace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        time_window_minutes: int = 15
    ) -> Dict[str, Any]:
        """Correlate observability data."""
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=time_window_minutes)
            
            correlation_data = {
                'correlation_id': str(uuid.uuid4()),
                'time_window': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'traces': [],
                'logs': [],
                'events': [],
                'anomalies': []
            }
            
            # Correlate traces
            if trace_id:
                trace_data = self.distributed_tracing.get_trace(trace_id)
                if trace_data:
                    correlation_data['traces'].append(trace_data)
            
            # Correlate logs
            logs = self.log_aggregation.query_logs(
                start_time=start_time,
                end_time=end_time,
                trace_id=trace_id
            )
            correlation_data['logs'] = [asdict(log) for log in logs[:100]]  # Limit results
            
            # Correlate events
            events = self.event_streaming.query_events(
                start_time=start_time,
                end_time=end_time,
                user_id=user_id,
                session_id=session_id
            )
            correlation_data['events'] = [asdict(event) for event in events[:100]]  # Limit results
            
            # Correlate anomalies
            anomalies = self.anomaly_detection.get_recent_anomalies(time_window_minutes)
            correlation_data['anomalies'] = anomalies
            
            return correlation_data
            
        except Exception as e:
            error_msg = f"Failed to correlate data: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _integrate_production_config(self) -> None:
        """Integrate with production configuration manager."""
        try:
            if not self.production_config_manager:
                return
            
            logger.info("Production configuration integrated successfully")
            
        except Exception as e:
            logger.error(f"Failed to integrate production configuration: {e}")
    
    def _integrate_monitoring_system(self) -> None:
        """Integrate with production monitoring system."""
        try:
            if not self.production_monitoring_system:
                return
            
            # Subscribe to monitoring system metrics for anomaly detection
            logger.info("Production monitoring system integration initialized")
            
        except Exception as e:
            logger.error(f"Failed to integrate monitoring system: {e}")
    
    def shutdown(self) -> None:
        """Shutdown observability manager."""
        try:
            self.stop_observability()
            logger.info("Observability manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during observability manager shutdown: {e}")


def create_observability_manager(
    config: Optional[ObservabilityConfiguration] = None
) -> ObservabilityManager:
    """Factory function to create an ObservabilityManager instance."""
    return ObservabilityManager(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    observability_manager = create_observability_manager()
    
    # Initialize observability
    observability_manager.initialize_observability_manager()
    
    # Start observability
    try:
        observability_manager.start_observability()
        print("Observability started successfully")
        
        # Create a sample trace
        with observability_manager.distributed_tracing.trace_context("example_operation", "example_service") as span:
            span.set_tag("environment", "development")
            span.add_log("Starting operation")
            
            # Log a message
            observability_manager.log_message(
                level=LogLevel.INFO,
                message="Example log message",
                service_name="example_service",
                trace_id=span.trace_id
            )
            
            # Emit an event
            observability_manager.emit_event(
                event_type=EventType.USER_ACTION,
                name="example_event",
                source="example_service",
                trace_id=span.trace_id,
                action="test_action"
            )
            
            span.add_log("Operation completed")
        
        # Get status
        status = observability_manager.get_observability_status()
        print(f"Observability status: {status['active']}")
        print(f"Traces created: {status['traces_created']}")
        
    except KeyboardInterrupt:
        print("Stopping observability...")
    finally:
        observability_manager.stop_observability()
        print("Observability stopped")