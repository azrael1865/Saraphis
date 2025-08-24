# Gradient Ascent Clipping (GAC) Type Definitions
# Data structures and types for GAC system
# Part of the Saraphis recursive methodology

"""
GAC Types - Data structures for gradient ascent clipping system

This module defines all data structures, enums, and type definitions
used throughout the GAC system for gradient information, events,
and component state management.

Types:
- ComponentState: Enum for component lifecycle states
- EventType: Enum for system event types  
- GACEvent: Event data structure
- ComponentMetrics: Performance metrics structure
- PIDState: PID controller state information
- DirectionType: Enum for gradient direction types
- DirectionState: Direction state information
"""

import torch
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

# Define all types here to avoid circular imports

class ComponentState(Enum):
    """Component lifecycle states"""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class EventType(Enum):
    """System event types"""
    COMPONENT_START = "component_start"
    COMPONENT_STOP = "component_stop"
    COMPONENT_ERROR = "component_error"
    GRADIENT_PROCESSED = "gradient_processed"
    GRADIENT_UPDATE = "gradient_update"  # Added for compatibility
    THRESHOLD_ADJUSTED = "threshold_adjusted"
    LEARNING_UPDATE = "learning_update"
    DIRECTION_CHANGE = "direction_change"
    ANOMALY_DETECTED = "anomaly_detected"

@dataclass
class GACEvent:
    """Event data structure"""
    event_type: EventType
    component_id: str
    data: Dict[str, Any]
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            import time
            self.timestamp = time.time()

@dataclass
class ComponentMetrics:
    """Performance metrics for a component"""
    processing_time: float = 0.0
    total_processed: int = 0
    error_count: int = 0
    success_rate: float = 1.0
    average_magnitude: float = 0.0
    last_update_time: float = 0.0

@dataclass
class PIDState:
    """PID controller state"""
    proportional: float = 0.0
    integral: float = 0.0
    derivative: float = 0.0
    last_error: float = 0.0
    last_time: float = 0.0
    output: float = 0.0

class DirectionType(Enum):
    """Gradient direction types"""
    ASCENT = "ascent"
    DESCENT = "descent" 
    STABLE = "stable"
    OSCILLATING = "oscillating"
    POSITIVE = "positive"  # Added for compatibility
    NEGATIVE = "negative"  # Added for completeness

@dataclass
class DirectionState:
    """Direction state information"""
    direction: DirectionType
    confidence: float
    magnitude: float
    timestamp: float
    metadata: Dict[str, Any]

__all__ = [
    'ComponentState',
    'EventType', 
    'GACEvent',
    'ComponentMetrics', 
    'PIDState',
    'DirectionType',
    'DirectionState'
]