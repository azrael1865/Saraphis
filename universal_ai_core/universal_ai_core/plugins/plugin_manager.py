#!/usr/bin/env python3
"""
Universal AI Core Plugin Manager
===============================

This module provides comprehensive plugin management capabilities for the Universal AI Core system.
Extracted and adapted from the Charon Builder plugin management system, made domain-agnostic while preserving
all sophisticated plugin orchestration capabilities.

Features:
- Dynamic plugin discovery and loading
- Plugin lifecycle management (load, unload, reload, hot-swap)
- Plugin dependency resolution with cycle detection
- Plugin validation and error handling
- Plugin versioning and compatibility checking
- Plugin performance monitoring and profiling
- Async plugin coordination and orchestration
- Plugin capability and hook registration
- Thread-safe plugin operations
- Configuration-driven plugin behavior
"""

import asyncio
import json
import logging
import threading
import time
import hashlib
import inspect
import importlib
import importlib.util
import os
import sys
import weakref
import gc
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union, Callable, Type
import traceback
import psutil

logger = logging.getLogger(__name__)


class PluginState(Enum):
    """Plugin lifecycle states"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    ERROR = "error"
    UNLOADING = "unloading"
    DISABLED = "disabled"


class PluginPriority(Enum):
    """Plugin execution priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class LoadingStrategy(Enum):
    """Plugin loading strategies"""
    EAGER = "eager"
    LAZY = "lazy"
    ON_DEMAND = "on_demand"
    BACKGROUND = "background"


@dataclass
class PluginVersion:
    """Plugin version representation"""
    major: int
    minor: int
    patch: int
    pre_release: Optional[str] = None
    build: Optional[str] = None
    
    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre_release:
            version += f"-{self.pre_release}"
        if self.build:
            version += f"+{self.build}"
        return version
    
    def is_compatible(self, other: 'PluginVersion') -> bool:
        """Check if versions are compatible (same major version)"""
        return self.major == other.major
    
    @classmethod
    def from_string(cls, version_str: str) -> 'PluginVersion':
        """Parse version from string"""
        parts = version_str.split('.')
        if len(parts) < 3:
            raise ValueError(f"Invalid version format: {version_str}")
        
        major, minor = int(parts[0]), int(parts[1])
        
        # Handle pre-release and build info
        patch_part = parts[2]
        patch = int(patch_part.split('-')[0].split('+')[0])
        
        pre_release = None
        build = None
        
        if '-' in patch_part:
            pre_release = patch_part.split('-')[1].split('+')[0]
        if '+' in patch_part:
            build = patch_part.split('+')[1]
        
        return cls(major, minor, patch, pre_release, build)


@dataclass
class PluginDependency:
    """Plugin dependency specification"""
    name: str
    version_requirement: str = "*"
    optional: bool = False
    import_name: Optional[str] = None
    
    def is_satisfied_by(self, version: PluginVersion) -> bool:
        """Check if dependency is satisfied by given version"""
        if self.version_requirement == "*":
            return True
        
        # Basic version matching - can be extended for complex requirements
        required_version = PluginVersion.from_string(self.version_requirement)
        return version.is_compatible(required_version)


@dataclass
class PluginMetadata:
    """Comprehensive plugin metadata"""
    name: str
    version: PluginVersion
    author: str
    description: str
    plugin_type: str
    entry_point: str
    dependencies: List[PluginDependency] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    hooks: List[str] = field(default_factory=list)
    configuration_schema: Dict[str, Any] = field(default_factory=dict)
    min_core_version: Optional[str] = None
    max_core_version: Optional[str] = None
    license: str = "Unknown"
    homepage: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    plugin_id: str = ""
    
    def __post_init__(self):
        if not self.plugin_id:
            content = f"{self.name}:{self.version}:{self.entry_point}"
            self.plugin_id = hashlib.md5(content.encode()).hexdigest()


@dataclass
class PluginInfo:
    """Runtime plugin information"""
    metadata: PluginMetadata
    state: PluginState = PluginState.UNLOADED
    module_path: Optional[str] = None
    file_path: Optional[Path] = None
    plugin_class: Optional[Type] = None
    instance: Optional[Any] = None
    load_time: Optional[datetime] = None
    error_message: Optional[str] = None
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginEvent:
    """Plugin lifecycle event"""
    plugin_id: str
    event_type: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = field(default_factory=dict)


class PluginPerformanceMonitor:
    """Performance monitoring for plugins"""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: {
            'call_count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'error_count': 0,
            'memory_usage': 0.0,
            'last_call': None
        })
        self._lock = threading.Lock()
    
    def record_call(self, plugin_id: str, duration: float, memory_usage: float = 0.0, success: bool = True):
        """Record a plugin method call"""
        with self._lock:
            stats = self.metrics[plugin_id]
            stats['call_count'] += 1
            stats['total_time'] += duration
            stats['avg_time'] = stats['total_time'] / stats['call_count']
            stats['min_time'] = min(stats['min_time'], duration)
            stats['max_time'] = max(stats['max_time'], duration)
            stats['memory_usage'] = memory_usage
            stats['last_call'] = datetime.utcnow()
            
            if not success:
                stats['error_count'] += 1
    
    def get_stats(self, plugin_id: str) -> Dict[str, Any]:
        """Get performance statistics for a plugin"""
        with self._lock:
            return dict(self.metrics[plugin_id])
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all plugins"""
        with self._lock:
            return {pid: dict(stats) for pid, stats in self.metrics.items()}
    
    def reset_stats(self, plugin_id: Optional[str] = None):
        """Reset performance statistics"""
        with self._lock:
            if plugin_id:
                if plugin_id in self.metrics:
                    del self.metrics[plugin_id]
            else:
                self.metrics.clear()


class CircuitBreaker:
    """Circuit breaker for plugin fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise RuntimeError("Circuit breaker is OPEN - plugin calls disabled")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset"""
        if self.last_failure_time is None:
            return True
        
        return (datetime.utcnow() - self.last_failure_time).total_seconds() > self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class DependencyResolver:
    """Plugin dependency resolution with cycle detection"""
    
    def __init__(self):
        self.dependencies = {}
        self.cycles = []
        self.missing_dependencies = set()
        self.resolution_order = []
    
    def add_plugin(self, plugin_id: str, dependencies: List[str]):
        """Add plugin and its dependencies"""
        self.dependencies[plugin_id] = dependencies
    
    def resolve(self) -> Tuple[List[str], bool]:
        """Resolve plugin loading order"""
        self.cycles.clear()
        self.missing_dependencies.clear()
        self.resolution_order.clear()
        
        # Check for missing dependencies
        all_plugins = set(self.dependencies.keys())
        for plugin_id, deps in self.dependencies.items():
            for dep in deps:
                if dep not in all_plugins:
                    self.missing_dependencies.add(dep)
        
        if self.missing_dependencies:
            return [], False
        
        # Topological sort with cycle detection
        visited = set()
        rec_stack = set()
        temp_order = []
        
        def dfs(plugin_id: str, path: List[str]):
            if plugin_id in rec_stack:
                # Found cycle
                cycle_start = path.index(plugin_id)
                cycle = path[cycle_start:] + [plugin_id]
                self.cycles.append(cycle)
                return False
            
            if plugin_id in visited:
                return True
            
            visited.add(plugin_id)
            rec_stack.add(plugin_id)
            path.append(plugin_id)
            
            # Visit dependencies first
            for dep in self.dependencies.get(plugin_id, []):
                if not dfs(dep, path.copy()):
                    return False
            
            rec_stack.remove(plugin_id)
            temp_order.append(plugin_id)
            return True
        
        # Process all plugins
        for plugin_id in self.dependencies:
            if plugin_id not in visited:
                if not dfs(plugin_id, []):
                    return [], False
        
        self.resolution_order = list(reversed(temp_order))
        return self.resolution_order, True
    
    def suggest_cycle_resolution(self) -> List[str]:
        """Suggest ways to resolve dependency cycles"""
        suggestions = []
        for cycle in self.cycles:
            suggestions.append(f"Break cycle: {' -> '.join(cycle)} by making one dependency optional")
        return suggestions


class PluginManager:
    """
    Universal plugin manager with sophisticated lifecycle management.
    
    Extracted and adapted from Charon Builder ai_coordinator.py and plugin_manager.py,
    made domain-agnostic while preserving all advanced capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plugin manager.
        
        Args:
            config: Configuration dictionary for the plugin manager
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.PluginManager")
        
        # Core plugin storage
        self.plugins: Dict[str, PluginInfo] = {}
        self.plugin_types: Dict[str, List[str]] = defaultdict(list)
        self.hooks: Dict[str, List[Callable]] = defaultdict(list)
        self.capabilities: Dict[str, Dict[str, Any]] = {}
        
        # Plugin discovery and loading
        self.plugin_directories = self.config.get('plugin_directories', ['plugins'])
        self.auto_load = self.config.get('auto_load', True)
        self.loading_strategy = LoadingStrategy(self.config.get('loading_strategy', 'eager'))
        
        # Performance and monitoring
        self.performance_monitor = PluginPerformanceMonitor()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.dependency_resolver = DependencyResolver()
        
        # Async coordination
        self.task_queue = None
        self.worker_tasks = []
        self.running = False
        self.max_workers = self.config.get('max_workers', 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Thread safety
        self._loading_lock = threading.RLock()
        self._state_lock = threading.RLock()
        
        # Event tracking
        self.events: List[PluginEvent] = []
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Configuration and caching
        self.plugin_config = self._load_plugin_config()
        self.discovery_cache = {}
        self.last_discovery = None
        
        # Statistics
        self.statistics = {
            'plugins_loaded': 0,
            'plugins_failed': 0,
            'total_load_time': 0.0,
            'hot_swaps': 0,
            'dependency_resolutions': 0
        }
        
        self.logger.info("🔌 Universal AI Core Plugin Manager initialized")
    
    async def start(self):
        """Start the plugin manager and async workers"""
        if self.running:
            self.logger.warning("Plugin manager is already running")
            return
        
        self.running = True
        
        # Initialize task queue
        self.task_queue = asyncio.Queue(maxsize=self.config.get('max_queue_size', 1000))
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker_task = asyncio.create_task(self._async_worker(f"plugin-worker-{i}"))
            self.worker_tasks.append(worker_task)
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitoring_loop())
        self.worker_tasks.append(monitor_task)
        
        # Auto-discover and load plugins if configured
        if self.auto_load:
            await self.discover_and_load_plugins()
        
        self.logger.info(f"🚀 Plugin manager started with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the plugin manager and cleanup resources"""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        
        # Unload all plugins
        await self.unload_all_plugins()
        
        # Cleanup resources
        self.executor.shutdown(wait=True)
        
        self.logger.info("🛑 Plugin manager stopped")
    
    async def _async_worker(self, worker_name: str):
        """Async worker for plugin operations"""
        self.logger.info(f"🔧 Starting plugin worker: {worker_name}")
        
        while self.running:
            try:
                # Get task from queue with timeout
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                self.logger.debug(f"🔍 {worker_name} processing task: {task['type']}")
                
                # Process task based on type
                if task['type'] == 'load_plugin':
                    await self._async_load_plugin(task['plugin_id'], task.get('config'))
                elif task['type'] == 'unload_plugin':
                    await self._async_unload_plugin(task['plugin_id'])
                elif task['type'] == 'reload_plugin':
                    await self._async_reload_plugin(task['plugin_id'])
                elif task['type'] == 'call_hook':
                    await self._async_call_hook(task['hook_name'], task['args'], task['kwargs'])
                
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                self.logger.info(f"🛑 Worker {worker_name} cancelled")
                break
            except Exception as e:
                self.logger.error(f"❌ {worker_name} error: {e}")
    
    async def _monitoring_loop(self):
        """Monitoring loop for plugin health and performance"""
        self.logger.info("📊 Starting plugin monitoring loop")
        
        while self.running:
            try:
                await asyncio.sleep(self.config.get('monitoring_interval', 30))
                
                # Check plugin health
                await self._health_check()
                
                # Log performance statistics
                self._log_performance_stats()
                
                # Cleanup expired cache entries
                self._cleanup_caches()
                
                # Memory cleanup
                if self.config.get('auto_gc', True):
                    gc.collect()
                
            except asyncio.CancelledError:
                self.logger.info("🛑 Monitoring loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"❌ Monitoring loop error: {e}")
    
    def discover_plugins(self) -> List[str]:
        """
        Discover plugins in configured directories.
        
        Adapted from Charon Builder plugin_manager.py lines 243-275.
        """
        plugins = []
        
        for directory in self.plugin_directories:
            plugin_dir = Path(directory)
            if not plugin_dir.exists():
                self.logger.warning(f"Plugin directory not found: {plugin_dir}")
                continue
            
            self.logger.info(f"🔍 Discovering plugins in: {plugin_dir}")
            
            # Recursive scanning for Python files
            for file_path in plugin_dir.rglob("*.py"):
                if file_path.name.startswith('__') or file_path.name == 'setup.py':
                    continue
                
                try:
                    # Convert to module path
                    relative_path = file_path.relative_to(plugin_dir)
                    module_path = str(relative_path.with_suffix('')).replace(os.sep, '.')
                    
                    # Look for plugin metadata
                    if self._has_plugin_metadata(file_path):
                        plugins.append(module_path)
                        self.logger.debug(f"📦 Found potential plugin: {module_path}")
                
                except Exception as e:
                    self.logger.warning(f"Error processing {file_path}: {e}")
        
        self.logger.info(f"🔍 Discovered {len(plugins)} potential plugins")
        return plugins
    
    def _has_plugin_metadata(self, file_path: Path) -> bool:
        """Check if file contains plugin metadata"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Look for plugin markers
                return any(marker in content for marker in [
                    'PluginMetadata', '__plugin_metadata__', '@plugin',
                    'class.*Plugin', 'PLUGIN_INFO'
                ])
        except Exception:
            return False
    
    async def discover_and_load_plugins(self):
        """Discover and load all plugins"""
        self.logger.info("🔍 Starting plugin discovery and loading")
        
        discovered = self.discover_plugins()
        loaded_count = 0
        failed_count = 0
        
        for module_path in discovered:
            try:
                plugin_id = await self.load_plugin_from_path(module_path)
                if plugin_id:
                    loaded_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                self.logger.error(f"Failed to load plugin {module_path}: {e}")
                failed_count += 1
        
        self.logger.info(f"📦 Plugin discovery complete: {loaded_count} loaded, {failed_count} failed")
    
    async def load_plugin_from_path(self, module_path: str) -> Optional[str]:
        """
        Load plugin from module path.
        
        Adapted from Charon Builder plugin_manager.py lines 277-324.
        """
        with self._loading_lock:
            try:
                self.logger.info(f"📦 Loading plugin: {module_path}")
                
                # Dynamic module loading
                spec = importlib.util.find_spec(module_path)
                if spec is None or spec.loader is None:
                    self.logger.error(f"Could not find module spec for: {module_path}")
                    return None
                
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find plugin classes
                plugin_classes = self._find_plugin_classes(module)
                if not plugin_classes:
                    self.logger.warning(f"No plugin classes found in: {module_path}")
                    return None
                
                # Load first valid plugin class
                for plugin_class in plugin_classes:
                    plugin_id = await self._instantiate_plugin(plugin_class, module, module_path)
                    if plugin_id:
                        return plugin_id
                
                return None
                
            except Exception as e:
                self.logger.error(f"❌ Error loading plugin {module_path}: {e}")
                self.logger.debug(traceback.format_exc())
                self.statistics['plugins_failed'] += 1
                return None
    
    def _find_plugin_classes(self, module) -> List[Type]:
        """Find plugin classes in module"""
        plugin_classes = []
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check if class is a plugin
            if hasattr(obj, '__plugin_metadata__') or hasattr(obj, 'get_metadata'):
                plugin_classes.append(obj)
            elif name.endswith('Plugin') and hasattr(obj, '__init__'):
                plugin_classes.append(obj)
        
        return plugin_classes
    
    async def _instantiate_plugin(self, plugin_class: Type, module, module_path: str) -> Optional[str]:
        """Instantiate and register plugin"""
        try:
            # Get plugin metadata
            metadata = self._extract_metadata(plugin_class)
            if not metadata:
                return None
            
            plugin_id = metadata.plugin_id
            
            # Check if already loaded
            if plugin_id in self.plugins:
                self.logger.warning(f"Plugin {plugin_id} already loaded")
                return plugin_id
            
            # Create plugin info
            plugin_info = PluginInfo(
                metadata=metadata,
                state=PluginState.LOADING,
                module_path=module_path,
                plugin_class=plugin_class
            )
            
            self.plugins[plugin_id] = plugin_info
            
            # Load plugin configuration
            plugin_config = self.plugin_config.get('plugin_settings', {}).get(plugin_id, {})
            plugin_info.config = plugin_config
            
            # Instantiate plugin
            start_time = time.time()
            
            try:
                instance = plugin_class(config=plugin_config)
                plugin_info.instance = instance
                plugin_info.state = PluginState.LOADED
                plugin_info.load_time = datetime.utcnow()
                
                load_time = time.time() - start_time
                self.statistics['total_load_time'] += load_time
                self.statistics['plugins_loaded'] += 1
                
                # Initialize plugin if it has initialize method
                if hasattr(instance, 'initialize'):
                    plugin_info.state = PluginState.INITIALIZING
                    
                    if asyncio.iscoroutinefunction(instance.initialize):
                        await instance.initialize()
                    else:
                        # Run in executor for blocking initialization
                        await asyncio.get_event_loop().run_in_executor(
                            self.executor, instance.initialize
                        )
                
                plugin_info.state = PluginState.ACTIVE
                
                # Register plugin capabilities and hooks
                self._register_plugin_capabilities(plugin_info)
                self._register_plugin_hooks(plugin_info)
                
                # Add to type index
                self.plugin_types[metadata.plugin_type].append(plugin_id)
                
                # Create circuit breaker
                self.circuit_breakers[plugin_id] = CircuitBreaker(
                    failure_threshold=self.config.get('circuit_breaker_threshold', 5),
                    recovery_timeout=self.config.get('circuit_breaker_timeout', 60)
                )
                
                # Fire plugin loaded event
                self._fire_event('plugin_loaded', plugin_id, {'load_time': load_time})
                
                self.logger.info(f"✅ Plugin loaded: {plugin_id} ({load_time:.3f}s)")
                return plugin_id
                
            except Exception as e:
                plugin_info.state = PluginState.ERROR
                plugin_info.error_message = str(e)
                self.logger.error(f"❌ Failed to initialize plugin {plugin_id}: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error instantiating plugin: {e}")
            return None
    
    def _extract_metadata(self, plugin_class: Type) -> Optional[PluginMetadata]:
        """Extract metadata from plugin class"""
        try:
            # Check for explicit metadata
            if hasattr(plugin_class, '__plugin_metadata__'):
                metadata_dict = plugin_class.__plugin_metadata__
                if isinstance(metadata_dict, dict):
                    return self._dict_to_metadata(metadata_dict)
            
            # Check for get_metadata method
            if hasattr(plugin_class, 'get_metadata'):
                metadata = plugin_class.get_metadata()
                if isinstance(metadata, PluginMetadata):
                    return metadata
                elif isinstance(metadata, dict):
                    return self._dict_to_metadata(metadata)
            
            # Generate basic metadata from class
            return PluginMetadata(
                name=plugin_class.__name__,
                version=PluginVersion(1, 0, 0),
                author="Unknown",
                description=plugin_class.__doc__ or "No description",
                plugin_type="unknown",
                entry_point=f"{plugin_class.__module__}:{plugin_class.__name__}"
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {plugin_class}: {e}")
            return None
    
    def _dict_to_metadata(self, metadata_dict: Dict[str, Any]) -> PluginMetadata:
        """Convert dictionary to PluginMetadata"""
        # Parse version
        version_str = metadata_dict.get('version', '1.0.0')
        version = PluginVersion.from_string(version_str) if isinstance(version_str, str) else version_str
        
        # Parse dependencies
        deps_data = metadata_dict.get('dependencies', [])
        dependencies = []
        for dep in deps_data:
            if isinstance(dep, str):
                dependencies.append(PluginDependency(name=dep))
            elif isinstance(dep, dict):
                dependencies.append(PluginDependency(**dep))
        
        return PluginMetadata(
            name=metadata_dict.get('name', 'Unknown'),
            version=version,
            author=metadata_dict.get('author', 'Unknown'),
            description=metadata_dict.get('description', 'No description'),
            plugin_type=metadata_dict.get('plugin_type', 'unknown'),
            entry_point=metadata_dict.get('entry_point', ''),
            dependencies=dependencies,
            capabilities=metadata_dict.get('capabilities', []),
            hooks=metadata_dict.get('hooks', []),
            configuration_schema=metadata_dict.get('configuration_schema', {}),
            min_core_version=metadata_dict.get('min_core_version'),
            max_core_version=metadata_dict.get('max_core_version'),
            license=metadata_dict.get('license', 'Unknown'),
            homepage=metadata_dict.get('homepage'),
            tags=metadata_dict.get('tags', [])
        )
    
    def _register_plugin_capabilities(self, plugin_info: PluginInfo):
        """
        Register plugin capabilities.
        
        Adapted from Charon Builder plugin_manager.py lines 398-412.
        """
        try:
            instance = plugin_info.instance
            if hasattr(instance, 'get_capabilities'):
                capabilities = instance.get_capabilities()
                if capabilities:
                    self.capabilities[plugin_info.metadata.plugin_id] = capabilities
                    self.logger.debug(f"📋 Registered capabilities for {plugin_info.metadata.name}")
        except Exception as e:
            self.logger.error(f"Error registering capabilities for {plugin_info.metadata.name}: {e}")
    
    def _register_plugin_hooks(self, plugin_info: PluginInfo):
        """Register plugin hooks"""
        try:
            instance = plugin_info.instance
            if hasattr(instance, 'get_hooks'):
                hooks = instance.get_hooks()
                if hooks:
                    for hook_name, hook_callable in hooks.items():
                        self.hooks[hook_name].append(hook_callable)
                    self.logger.debug(f"🪝 Registered hooks for {plugin_info.metadata.name}")
        except Exception as e:
            self.logger.error(f"Error registering hooks for {plugin_info.metadata.name}: {e}")
    
    async def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin"""
        with self._loading_lock:
            if plugin_id not in self.plugins:
                self.logger.warning(f"Plugin not found: {plugin_id}")
                return False
            
            plugin_info = self.plugins[plugin_id]
            
            try:
                plugin_info.state = PluginState.UNLOADING
                
                # Call plugin shutdown if available
                if plugin_info.instance and hasattr(plugin_info.instance, 'shutdown'):
                    if asyncio.iscoroutinefunction(plugin_info.instance.shutdown):
                        await plugin_info.instance.shutdown()
                    else:
                        await asyncio.get_event_loop().run_in_executor(
                            self.executor, plugin_info.instance.shutdown
                        )
                
                # Unregister capabilities and hooks
                self._unregister_plugin_capabilities(plugin_id)
                self._unregister_plugin_hooks(plugin_id)
                
                # Remove from type index
                if plugin_info.metadata.plugin_type in self.plugin_types:
                    self.plugin_types[plugin_info.metadata.plugin_type].remove(plugin_id)
                
                # Cleanup circuit breaker
                if plugin_id in self.circuit_breakers:
                    del self.circuit_breakers[plugin_id]
                
                # Remove plugin
                del self.plugins[plugin_id]
                
                # Fire event
                self._fire_event('plugin_unloaded', plugin_id)
                
                self.logger.info(f"🗑️ Plugin unloaded: {plugin_id}")
                return True
                
            except Exception as e:
                plugin_info.state = PluginState.ERROR
                plugin_info.error_message = str(e)
                self.logger.error(f"❌ Error unloading plugin {plugin_id}: {e}")
                return False
    
    def _unregister_plugin_capabilities(self, plugin_id: str):
        """Unregister plugin capabilities"""
        if plugin_id in self.capabilities:
            del self.capabilities[plugin_id]
    
    def _unregister_plugin_hooks(self, plugin_id: str):
        """Unregister plugin hooks"""
        plugin_info = self.plugins.get(plugin_id)
        if not plugin_info or not plugin_info.instance:
            return
        
        if hasattr(plugin_info.instance, 'get_hooks'):
            hooks = plugin_info.instance.get_hooks()
            if hooks:
                for hook_name, hook_callable in hooks.items():
                    if hook_name in self.hooks:
                        try:
                            self.hooks[hook_name].remove(hook_callable)
                            if not self.hooks[hook_name]:
                                del self.hooks[hook_name]
                        except ValueError:
                            pass
    
    async def reload_plugin(self, plugin_id: str) -> bool:
        """Reload a plugin (hot-swap)"""
        if plugin_id not in self.plugins:
            self.logger.warning(f"Plugin not found for reload: {plugin_id}")
            return False
        
        plugin_info = self.plugins[plugin_id]
        module_path = plugin_info.module_path
        
        self.logger.info(f"🔄 Reloading plugin: {plugin_id}")
        
        # Unload current plugin
        if not await self.unload_plugin(plugin_id):
            return False
        
        # Reload module
        try:
            if module_path in sys.modules:
                importlib.reload(sys.modules[module_path])
        except Exception as e:
            self.logger.error(f"Error reloading module {module_path}: {e}")
        
        # Load plugin again
        new_plugin_id = await self.load_plugin_from_path(module_path)
        
        if new_plugin_id:
            self.statistics['hot_swaps'] += 1
            self._fire_event('plugin_reloaded', new_plugin_id)
            self.logger.info(f"✅ Plugin reloaded: {new_plugin_id}")
            return True
        
        return False
    
    async def unload_all_plugins(self):
        """Unload all plugins"""
        plugin_ids = list(self.plugins.keys())
        
        for plugin_id in plugin_ids:
            try:
                await self.unload_plugin(plugin_id)
            except Exception as e:
                self.logger.error(f"Error unloading plugin {plugin_id}: {e}")
    
    def get_plugin(self, plugin_id: str) -> Optional[Any]:
        """Get plugin instance by ID"""
        plugin_info = self.plugins.get(plugin_id)
        if plugin_info and plugin_info.state == PluginState.ACTIVE:
            plugin_info.last_accessed = datetime.utcnow()
            plugin_info.access_count += 1
            return plugin_info.instance
        return None
    
    def get_plugins_by_type(self, plugin_type: str) -> List[Any]:
        """Get all plugins of a specific type"""
        plugin_ids = self.plugin_types.get(plugin_type, [])
        plugins = []
        
        for plugin_id in plugin_ids:
            plugin = self.get_plugin(plugin_id)
            if plugin:
                plugins.append(plugin)
        
        return plugins
    
    def get_plugins_by_capability(self, capability: str) -> List[Any]:
        """Get plugins that have a specific capability"""
        plugins = []
        
        for plugin_id, capabilities in self.capabilities.items():
            if capability in capabilities:
                plugin = self.get_plugin(plugin_id)
                if plugin:
                    plugins.append(plugin)
        
        return plugins
    
    async def call_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Call all plugins registered for a hook"""
        if hook_name not in self.hooks:
            return []
        
        results = []
        
        for hook_callable in self.hooks[hook_name]:
            try:
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(hook_callable):
                    result = await hook_callable(*args, **kwargs)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        self.executor, hook_callable, *args, **kwargs
                    )
                
                duration = time.time() - start_time
                self.performance_monitor.record_call(
                    f"hook_{hook_name}", duration, success=True
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error calling hook {hook_name}: {e}")
                self.performance_monitor.record_call(
                    f"hook_{hook_name}", 0, success=False
                )
        
        return results
    
    async def _async_load_plugin(self, plugin_id: str, config: Optional[Dict] = None):
        """Async plugin loading task"""
        await self.load_plugin_from_path(plugin_id)
    
    async def _async_unload_plugin(self, plugin_id: str):
        """Async plugin unloading task"""
        await self.unload_plugin(plugin_id)
    
    async def _async_reload_plugin(self, plugin_id: str):
        """Async plugin reloading task"""
        await self.reload_plugin(plugin_id)
    
    async def _async_call_hook(self, hook_name: str, args: tuple, kwargs: dict):
        """Async hook calling task"""
        await self.call_hook(hook_name, *args, **kwargs)
    
    async def _health_check(self):
        """Perform health check on all plugins"""
        for plugin_id, plugin_info in self.plugins.items():
            try:
                if plugin_info.state == PluginState.ACTIVE and plugin_info.instance:
                    if hasattr(plugin_info.instance, 'health_check'):
                        health = await plugin_info.instance.health_check()
                        if not health:
                            self.logger.warning(f"Plugin {plugin_id} failed health check")
            except Exception as e:
                self.logger.error(f"Health check error for {plugin_id}: {e}")
    
    def _log_performance_stats(self):
        """Log performance statistics"""
        stats = self.performance_monitor.get_all_stats()
        if stats:
            total_calls = sum(s['call_count'] for s in stats.values())
            total_errors = sum(s['error_count'] for s in stats.values())
            avg_time = sum(s['avg_time'] for s in stats.values()) / len(stats)
            
            self.logger.info(
                f"📊 Plugin Performance: {total_calls} calls, "
                f"{total_errors} errors, {avg_time:.3f}s avg"
            )
    
    def _cleanup_caches(self):
        """Cleanup expired cache entries"""
        cache_ttl = self.config.get('cache_ttl', 3600)
        current_time = time.time()
        
        # Cleanup discovery cache
        expired_keys = [
            key for key, (_, timestamp) in self.discovery_cache.items()
            if current_time - timestamp > cache_ttl
        ]
        
        for key in expired_keys:
            del self.discovery_cache[key]
    
    def _fire_event(self, event_type: str, plugin_id: str, details: Optional[Dict] = None):
        """Fire plugin event"""
        event = PluginEvent(
            plugin_id=plugin_id,
            event_type=event_type,
            details=details or {}
        )
        
        self.events.append(event)
        
        # Keep only recent events
        max_events = self.config.get('max_events', 1000)
        if len(self.events) > max_events:
            self.events = self.events[-max_events:]
        
        # Call event handlers
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"Error in event handler: {e}")
    
    def _load_plugin_config(self) -> Dict[str, Any]:
        """
        Load plugin configuration from file.
        
        Adapted from Charon Builder plugin_manager.py lines 225-241.
        """
        config_path = Path(self.config.get('config_file', 'config/plugins.json'))
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading plugin config: {e}")
        
        return {
            'auto_load': True,
            'enabled_plugins': [],
            'disabled_plugins': [],
            'plugin_settings': {}
        }
    
    def get_plugin_info(self, plugin_id: str) -> Optional[PluginInfo]:
        """Get plugin information"""
        return self.plugins.get(plugin_id)
    
    def get_all_plugins(self) -> Dict[str, PluginInfo]:
        """Get all plugin information"""
        return self.plugins.copy()
    
    def get_plugin_statistics(self) -> Dict[str, Any]:
        """Get plugin manager statistics"""
        active_plugins = sum(1 for p in self.plugins.values() if p.state == PluginState.ACTIVE)
        error_plugins = sum(1 for p in self.plugins.values() if p.state == PluginState.ERROR)
        
        return {
            'total_plugins': len(self.plugins),
            'active_plugins': active_plugins,
            'error_plugins': error_plugins,
            'plugin_types': len(self.plugin_types),
            'total_capabilities': len(self.capabilities),
            'total_hooks': len(self.hooks),
            'queue_size': self.task_queue.qsize() if self.task_queue else 0,
            'running': self.running,
            'performance_stats': self.performance_monitor.get_all_stats(),
            **self.statistics
        }
    
    def list_plugin_types(self) -> List[str]:
        """List all plugin types"""
        return list(self.plugin_types.keys())
    
    def list_capabilities(self) -> List[str]:
        """List all plugin capabilities"""
        all_capabilities = set()
        for capabilities in self.capabilities.values():
            all_capabilities.update(capabilities.keys())
        return list(all_capabilities)
    
    def list_hooks(self) -> List[str]:
        """List all plugin hooks"""
        return list(self.hooks.keys())


# Main async function for testing
async def main():
    """Main function for testing the plugin manager"""
    print("🔌 UNIVERSAL AI CORE PLUGIN MANAGER")
    print("=" * 60)
    
    # Initialize plugin manager
    config = {
        'plugin_directories': ['plugins'],
        'max_workers': 4,
        'auto_load': True,
        'monitoring_interval': 10
    }
    
    manager = PluginManager(config)
    
    try:
        # Start manager
        await manager.start()
        
        # Show statistics
        stats = manager.get_plugin_statistics()
        print(f"\n📊 Plugin Manager Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Wait a bit for any background operations
        await asyncio.sleep(5)
        
        print("\n✅ Plugin manager test completed!")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())