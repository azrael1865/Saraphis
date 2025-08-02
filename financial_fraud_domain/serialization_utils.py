"""
Serialization Utilities for Financial Fraud Detection
Serialization utilities and helpers
"""

import json
import logging
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class SerializationUtils:
    """Serialization utilities"""
    
    def __init__(self):
        """Initialize serialization utilities"""
        logger.info("SerializationUtils initialized")
    
    def serialize_to_json(self, data: Any) -> str:
        """Serialize data to JSON"""
        try:
            return json.dumps(data, default=self._json_serializer, indent=2)
        except Exception as e:
            logger.error(f"JSON serialization failed: {e}")
            return "{}"
    
    def deserialize_from_json(self, json_str: str) -> Any:
        """Deserialize data from JSON"""
        try:
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"JSON deserialization failed: {e}")
            return None
    
    def serialize_to_pickle(self, data: Any) -> bytes:
        """Serialize data to pickle"""
        try:
            return pickle.dumps(data)
        except Exception as e:
            logger.error(f"Pickle serialization failed: {e}")
            return b""
    
    def deserialize_from_pickle(self, pickle_data: bytes) -> Any:
        """Deserialize data from pickle"""
        try:
            return pickle.loads(pickle_data)
        except Exception as e:
            logger.error(f"Pickle deserialization failed: {e}")
            return None
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for datetime and other objects"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

if __name__ == "__main__":
    utils = SerializationUtils()
    print("Serialization utilities initialized")