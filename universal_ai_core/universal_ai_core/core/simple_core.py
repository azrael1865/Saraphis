"""
Simple Universal AI Core
========================

A minimal, universal core that external modules can hook into.
This provides the basic infrastructure without domain-specific logic.
"""

from typing import Dict, Any, Optional, List, Callable
from abc import ABC, abstractmethod
import logging
import json
import time

logger = logging.getLogger(__name__)


class Module(ABC):
    """Base class for external modules that can hook into the core."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process data through this module."""
        pass
    
    def validate_input(self, data: Any) -> bool:
        """Validate input data (override if needed)."""
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get module information."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "config": self.config
        }


class Pipeline:
    """Simple processing pipeline that chains modules together."""
    
    def __init__(self, name: str):
        self.name = name
        self.modules: List[Module] = []
    
    def add_module(self, module: Module) -> None:
        """Add a module to the pipeline."""
        self.modules.append(module)
        logger.info(f"Added module '{module.name}' to pipeline '{self.name}'")
    
    def remove_module(self, module_name: str) -> bool:
        """Remove a module from the pipeline."""
        for i, module in enumerate(self.modules):
            if module.name == module_name:
                self.modules.pop(i)
                logger.info(f"Removed module '{module_name}' from pipeline '{self.name}'")
                return True
        return False
    
    def process(self, data: Any) -> Any:
        """Process data through all modules in the pipeline."""
        result = data
        
        for module in self.modules:
            if not module.enabled:
                continue
                
            if not module.validate_input(result):
                logger.warning(f"Module '{module.name}' rejected input data")
                continue
            
            try:
                result = module.process(result)
            except Exception as e:
                logger.error(f"Module '{module.name}' failed: {e}")
                raise
        
        return result
    
    def get_info(self) -> Dict[str, Any]:
        """Get pipeline information."""
        return {
            "name": self.name,
            "module_count": len(self.modules),
            "modules": [module.get_info() for module in self.modules]
        }


class UniversalCore:
    """
    Simple Universal AI Core.
    
    This is a minimal core that provides:
    1. Module registration and management
    2. Pipeline creation and execution
    3. Simple event system
    4. Basic configuration management
    
    External modules can hook into this core by:
    1. Inheriting from the Module base class
    2. Registering with the core
    3. Being added to pipelines
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.modules: Dict[str, Module] = {}
        self.pipelines: Dict[str, Pipeline] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        logger.info("Universal Core initialized")
    
    # Module Management
    
    def register_module(self, module: Module) -> None:
        """Register a module with the core."""
        self.modules[module.name] = module
        logger.info(f"Registered module: {module.name}")
        self.emit_event("module_registered", {"module_name": module.name})
    
    def unregister_module(self, module_name: str) -> bool:
        """Unregister a module from the core."""
        if module_name in self.modules:
            del self.modules[module_name]
            logger.info(f"Unregistered module: {module_name}")
            self.emit_event("module_unregistered", {"module_name": module_name})
            return True
        return False
    
    def get_module(self, module_name: str) -> Optional[Module]:
        """Get a registered module."""
        return self.modules.get(module_name)
    
    def list_modules(self) -> Dict[str, Dict[str, Any]]:
        """List all registered modules."""
        return {name: module.get_info() for name, module in self.modules.items()}
    
    # Pipeline Management
    
    def create_pipeline(self, pipeline_name: str) -> Pipeline:
        """Create a new pipeline."""
        pipeline = Pipeline(pipeline_name)
        self.pipelines[pipeline_name] = pipeline
        logger.info(f"Created pipeline: {pipeline_name}")
        self.emit_event("pipeline_created", {"pipeline_name": pipeline_name})
        return pipeline
    
    def get_pipeline(self, pipeline_name: str) -> Optional[Pipeline]:
        """Get a pipeline."""
        return self.pipelines.get(pipeline_name)
    
    def delete_pipeline(self, pipeline_name: str) -> bool:
        """Delete a pipeline."""
        if pipeline_name in self.pipelines:
            del self.pipelines[pipeline_name]
            logger.info(f"Deleted pipeline: {pipeline_name}")
            self.emit_event("pipeline_deleted", {"pipeline_name": pipeline_name})
            return True
        return False
    
    def list_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """List all pipelines."""
        return {name: pipeline.get_info() for name, pipeline in self.pipelines.items()}
    
    # Processing
    
    def process_with_module(self, module_name: str, data: Any) -> Any:
        """Process data with a specific module."""
        module = self.get_module(module_name)
        if not module:
            raise ValueError(f"Module '{module_name}' not found")
        
        if not module.enabled:
            raise ValueError(f"Module '{module_name}' is disabled")
        
        start_time = time.time()
        result = module.process(data)
        processing_time = time.time() - start_time
        
        self.emit_event("module_processed", {
            "module_name": module_name,
            "processing_time": processing_time
        })
        
        return result
    
    def process_with_pipeline(self, pipeline_name: str, data: Any) -> Any:
        """Process data with a pipeline."""
        pipeline = self.get_pipeline(pipeline_name)
        if not pipeline:
            raise ValueError(f"Pipeline '{pipeline_name}' not found")
        
        start_time = time.time()
        result = pipeline.process(data)
        processing_time = time.time() - start_time
        
        self.emit_event("pipeline_processed", {
            "pipeline_name": pipeline_name,
            "processing_time": processing_time
        })
        
        return result
    
    # Event System
    
    def on(self, event_name: str, handler: Callable) -> None:
        """Register an event handler."""
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(handler)
    
    def off(self, event_name: str, handler: Callable) -> None:
        """Unregister an event handler."""
        if event_name in self.event_handlers:
            try:
                self.event_handlers[event_name].remove(handler)
            except ValueError:
                pass
    
    def emit_event(self, event_name: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Emit an event to all registered handlers."""
        if event_name in self.event_handlers:
            for handler in self.event_handlers[event_name]:
                try:
                    handler(event_name, data or {})
                except Exception as e:
                    logger.error(f"Event handler failed for '{event_name}': {e}")
    
    # Configuration
    
    def get_config(self, key: str = None) -> Any:
        """Get configuration value."""
        if key is None:
            return self.config
        return self.config.get(key)
    
    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
        self.emit_event("config_changed", {"key": key, "value": value})
    
    # System Info
    
    def get_status(self) -> Dict[str, Any]:
        """Get core system status."""
        return {
            "modules": len(self.modules),
            "pipelines": len(self.pipelines),
            "event_handlers": sum(len(handlers) for handlers in self.event_handlers.values()),
            "config_keys": len(self.config)
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "core_status": self.get_status(),
            "modules": self.list_modules(),
            "pipelines": self.list_pipelines(),
            "config": self.config
        }


# Simple helper functions for common use cases

def create_core(config: Optional[Dict[str, Any]] = None) -> UniversalCore:
    """Create a new Universal Core instance."""
    return UniversalCore(config)


def load_modules_from_config(core: UniversalCore, config: Dict[str, Any]) -> None:
    """Load modules from configuration."""
    modules_config = config.get("modules", {})
    
    for module_name, module_config in modules_config.items():
        module_type = module_config.get("type")
        if not module_type:
            logger.warning(f"No type specified for module '{module_name}'")
            continue
        
        try:
            # This is where you'd load different module types
            # For now, we'll just log that we would load them
            logger.info(f"Would load module '{module_name}' of type '{module_type}'")
        except Exception as e:
            logger.error(f"Failed to load module '{module_name}': {e}")


# Example usage and simple modules for demonstration

class SimpleProcessorModule(Module):
    """Example simple processor module."""
    
    def process(self, data: Any) -> Any:
        """Simple processing - just add metadata."""
        if isinstance(data, dict):
            data["processed_by"] = self.name
            data["processed_at"] = time.time()
        return data


class SimpleValidatorModule(Module):
    """Example simple validator module."""
    
    def process(self, data: Any) -> Any:
        """Simple validation - check if data is not empty."""
        if not data:
            raise ValueError("Data cannot be empty")
        
        if isinstance(data, dict):
            data["validated_by"] = self.name
            data["valid"] = True
        
        return data


if __name__ == "__main__":
    # Example usage
    
    # Create core
    core = create_core({"debug": True})
    
    # Create and register modules
    processor = SimpleProcessorModule("processor", {"enabled": True})
    validator = SimpleValidatorModule("validator", {"enabled": True})
    
    core.register_module(processor)
    core.register_module(validator)
    
    # Create pipeline
    pipeline = core.create_pipeline("simple_pipeline")
    pipeline.add_module(validator)
    pipeline.add_module(processor)
    
    # Process some data
    test_data = {"input": "test data"}
    
    print("Original data:", test_data)
    
    # Process with pipeline
    result = core.process_with_pipeline("simple_pipeline", test_data)
    print("Processed data:", result)
    
    # Get system info
    info = core.get_info()
    print("System info:", json.dumps(info, indent=2, default=str))