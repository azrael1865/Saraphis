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

# Re-export all types defined in the main GAC module
# This provides a clean interface while avoiding circular imports
from .gradient_ascent_clipping import (
    ComponentState,
    EventType, 
    GACEvent,
    ComponentMetrics,
    PIDState
)

class DirectionType(Enum):
    """Gradient direction types"""
    ASCENT = "ascent"
    DESCENT = "descent" 
    STABLE = "stable"
    OSCILLATING = "oscillating"

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