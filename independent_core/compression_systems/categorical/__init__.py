"""
Categorical Storage and IEEE 754 Channel Processing

This module provides categorical weight storage and IEEE 754 channel decomposition
for improved compression ratios in the Saraphis compression system.
"""

from .ieee754_channel_extractor import IEEE754ChannelExtractor
from .categorical_storage_manager import CategoricalStorageManager
from .weight_categorizer import WeightCategorizer

__all__ = [
    'IEEE754ChannelExtractor',
    'CategoricalStorageManager', 
    'WeightCategorizer'
]