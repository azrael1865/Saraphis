# Gradient Ascent Clipping (GAC) Component Interfaces
# Component interface definitions for GAC system
# Part of the Saraphis recursive methodology

"""
GACComponent - Base interface for all GAC system components

This module defines the standardized interfaces for all GAC components
including lifecycle management, event communication, and configuration.

Components:
- PID Controller
- Meta-Learning system
- RL Controller
- Explosion detection
- Adaptive learning rate
"""

# Re-export the existing GACComponent from gradient_ascent_clipping.py
# This fixes the import chain without duplicating implementation
from .gradient_ascent_clipping import GACComponent

__all__ = ['GACComponent']