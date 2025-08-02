"""
Simple Universal AI Core API
=============================

A minimal, clean API that provides a universal core for hooking external modules.
This is what you actually want - simple, focused, extensible.
"""

from typing import Dict, Any, Optional, List
import logging
from .core.simple_core import UniversalCore, Module, Pipeline, create_core

logger = logging.getLogger(__name__)


class SimpleAPI:
    """
    Simple Universal AI Core API.
    
    This provides a clean, minimal interface for:
    1. Registering external modules
    2. Creating processing pipelines  
    3. Processing data through modules/pipelines
    4. Basic configuration and events
    
    NO built-in domain logic, NO complex enterprise features.
    Just a simple core that external modules can hook into.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.core = create_core(config)
        logger.info("Simple Universal AI API initialized")
    
    # Module Management
    
    def register_module(self, module: Module) -> None:
        """Register an external module."""
        self.core.register_module(module)
    
    def unregister_module(self, module_name: str) -> bool:
        """Unregister a module."""
        return self.core.unregister_module(module_name)
    
    def get_module(self, module_name: str) -> Optional[Module]:
        """Get a registered module."""
        return self.core.get_module(module_name)
    
    def list_modules(self) -> Dict[str, Dict[str, Any]]:
        """List all registered modules."""
        return self.core.list_modules()
    
    # Pipeline Management
    
    def create_pipeline(self, pipeline_name: str) -> Pipeline:
        """Create a processing pipeline."""
        return self.core.create_pipeline(pipeline_name)
    
    def get_pipeline(self, pipeline_name: str) -> Optional[Pipeline]:
        """Get a pipeline."""
        return self.core.get_pipeline(pipeline_name)
    
    def delete_pipeline(self, pipeline_name: str) -> bool:
        """Delete a pipeline."""
        return self.core.delete_pipeline(pipeline_name)
    
    def list_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """List all pipelines."""
        return self.core.list_pipelines()
    
    # Processing
    
    def process(self, module_or_pipeline_name: str, data: Any) -> Any:
        """
        Process data with a module or pipeline.
        
        Args:
            module_or_pipeline_name: Name of module or pipeline to use
            data: Data to process
            
        Returns:
            Processed data
        """
        # Try pipeline first, then module
        if module_or_pipeline_name in self.core.pipelines:
            return self.core.process_with_pipeline(module_or_pipeline_name, data)
        elif module_or_pipeline_name in self.core.modules:
            return self.core.process_with_module(module_or_pipeline_name, data)
        else:
            raise ValueError(f"No module or pipeline named '{module_or_pipeline_name}' found")
    
    def process_with_module(self, module_name: str, data: Any) -> Any:
        """Process data with a specific module."""
        return self.core.process_with_module(module_name, data)
    
    def process_with_pipeline(self, pipeline_name: str, data: Any) -> Any:
        """Process data with a pipeline."""
        return self.core.process_with_pipeline(pipeline_name, data)
    
    # Events
    
    def on(self, event_name: str, handler) -> None:
        """Register an event handler."""
        self.core.on(event_name, handler)
    
    def off(self, event_name: str, handler) -> None:
        """Unregister an event handler."""
        self.core.off(event_name, handler)
    
    # Configuration
    
    def get_config(self, key: str = None) -> Any:
        """Get configuration."""
        return self.core.get_config(key)
    
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration."""
        self.core.set_config(key, value)
    
    # System Info
    
    def status(self) -> Dict[str, Any]:
        """Get system status."""
        return self.core.get_status()
    
    def info(self) -> Dict[str, Any]:
        """Get system information."""
        return self.core.get_info()


# Simple factory function

def create_simple_api(config: Optional[Dict[str, Any]] = None) -> SimpleAPI:
    """Create a simple Universal AI Core API."""
    return SimpleAPI(config)


# Export the simple interface
__all__ = [
    'SimpleAPI',
    'Module', 
    'Pipeline',
    'create_simple_api'
]