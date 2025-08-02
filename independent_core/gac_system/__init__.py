# Gradient Ascent Clipping (GAC) System Package
# Zero-oversight autonomous gradient ascent clipping system
# Part of the Saraphis recursive methodology

"""
GAC System Package - Autonomous gradient ascent clipping

This package provides the complete zero-oversight gradient ascent clipping
system with self-tuning thresholds, meta-learning, and reinforcement
learning control.

Main Components:
- gradient_ascent_clipping.py: Main system orchestrator
- gac_config.py: Configuration management
- gac_interface.py: Component interfaces
- gac_types.py: Data structures and types

Direction Components:
- direction_state.py: Direction state management
- direction_validator.py: Direction validation and anomaly detection
- basic_bounder.py: Basic gradient bounding
- enhanced_bounder.py: Enhanced direction-aware bounding
"""

from .gradient_ascent_clipping import GACSystem
from .gac_config import GACConfig
from .gac_interface import GACComponent
from .gac_types import *

# Direction Components
from .direction_state import DirectionStateManager, DirectionHistory
from .direction_validator import DirectionValidator, ValidationResult, AnomalyDetection
from .basic_bounder import BasicGradientBounder, BoundingResult
from .enhanced_bounder import EnhancedGradientBounder, EnhancedBoundingResult

__version__ = "1.0.0"
__author__ = "Saraphis AI Core"
__description__ = "Zero-oversight gradient ascent clipping system" 