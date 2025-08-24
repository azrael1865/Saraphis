"""
Domain Registry for Universal AI Core Brain.
Manages domain registration, metadata tracking, and domain lifecycle.
"""

import logging
import threading
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import re


class TensorJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles PyTorch tensors and numpy arrays."""
    
    def default(self, obj):
        try:
            # Handle PyTorch tensors
            if hasattr(obj, 'detach') and hasattr(obj, 'cpu') and hasattr(obj, 'numpy'):
                return obj.detach().cpu().numpy().tolist()
            # Handle numpy arrays
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            # Handle numpy scalars
            elif hasattr(obj, 'item'):
                return obj.item()
            else:
                return super().default(obj)
        except:
            return str(obj)


class DomainStatus(Enum):
    """Status states for registered domains."""
    REGISTERED = "registered"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    DEPRECATED = "deprecated"


class DomainType(Enum):
    """Types of domains that can be registered."""
    STANDARD = "standard"
    SPECIALIZED = "specialized"
    EXPERIMENTAL = "experimental"
    PLUGIN = "plugin"
    CORE = "core"


@dataclass
class DomainConfig:
    """Configuration for a domain."""
    
    # Basic settings
    domain_type: DomainType = DomainType.STANDARD
    description: str = ""
    version: str = "1.0.0"
    
    # Resource allocation
    max_memory_mb: int = 512
    max_cpu_percent: float = 25.0
    priority: int = 5  # 1-10, higher is more priority
    
    # Neural network settings
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    activation_function: str = "relu"
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    
    # Feature settings
    enable_caching: bool = True
    cache_size: int = 100
    enable_logging: bool = True
    log_level: str = "INFO"
    
    # Integration settings
    shared_foundation_layers: int = 3
    allow_cross_domain_access: bool = False
    dependencies: List[str] = field(default_factory=list)
    
    # Metadata
    author: str = ""
    contact: str = ""
    tags: List[str] = field(default_factory=list)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate configuration values.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Memory validation
        if self.max_memory_mb <= 0:
            errors.append("max_memory_mb must be positive")
        elif self.max_memory_mb > 8192:  # 8GB max
            errors.append("max_memory_mb cannot exceed 8192 (8GB)")
        
        # CPU validation
        if not 0 < self.max_cpu_percent <= 100:
            errors.append("max_cpu_percent must be between 0 and 100")
        
        # Priority validation
        if not 1 <= self.priority <= 10:
            errors.append("priority must be between 1 and 10")
        
        # Neural network validation
        if not self.hidden_layers:
            errors.append("hidden_layers cannot be empty")
        else:
            for layer_size in self.hidden_layers:
                if layer_size <= 0:
                    errors.append("hidden layer sizes must be positive")
                    break
        
        if not 0 <= self.dropout_rate < 1:
            errors.append("dropout_rate must be between 0 and 1")
        
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        # Version format validation
        if not re.match(r'^\d+\.\d+\.\d+$', self.version):
            errors.append("version must be in format X.Y.Z")
        
        # Dependency validation
        for dep in self.dependencies:
            if not isinstance(dep, str) or not dep.strip():
                errors.append("dependencies must be non-empty strings")
                break
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        data = asdict(self)
        data['domain_type'] = self.domain_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainConfig':
        """Create config from dictionary."""
        if 'domain_type' in data and isinstance(data['domain_type'], str):
            data['domain_type'] = DomainType(data['domain_type'])
        return cls(**data)


@dataclass
class DomainMetadata:
    """Metadata for a registered domain."""
    name: str
    config: DomainConfig
    status: DomainStatus = DomainStatus.REGISTERED
    registered_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    activation_count: int = 0
    error_count: int = 0
    total_predictions: int = 0
    average_confidence: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    error_messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'name': self.name,
            'config': self.config.to_dict(),
            'status': self.status.value,
            'registered_at': self.registered_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'activation_count': self.activation_count,
            'error_count': self.error_count,
            'total_predictions': self.total_predictions,
            'average_confidence': self.average_confidence,
            'resource_usage': self.resource_usage,
            'error_messages': self.error_messages,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DomainMetadata':
        """Create metadata from dictionary."""
        # Convert timestamps
        if 'registered_at' in data:
            data['registered_at'] = datetime.fromisoformat(data['registered_at'])
        if 'last_updated' in data:
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        if 'last_accessed' in data and data['last_accessed']:
            data['last_accessed'] = datetime.fromisoformat(data['last_accessed'])
        
        # Convert config
        if 'config' in data and isinstance(data['config'], dict):
            data['config'] = DomainConfig.from_dict(data['config'])
        
        # Convert status
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = DomainStatus(data['status'])
        
        return cls(**data)


class DomainRegistry:
    """
    Registry for managing domains in the Universal AI Core Brain.
    Tracks domain registration, metadata, and lifecycle management.
    Includes domain isolation mechanisms to prevent catastrophic forgetting.
    """
    
    def __init__(self, persistence_path: Optional[Path] = None, 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the domain registry.
        
        Args:
            persistence_path: Optional path for persisting registry state
            logger: Optional logger instance
        """
        self._domains: Dict[str, DomainMetadata] = {}
        self._domain_order: List[str] = []  # Maintains registration order
        self._dependencies: Dict[str, Set[str]] = {}  # domain -> dependencies
        self._dependents: Dict[str, Set[str]] = {}  # domain -> dependents
        
        # Domain isolation storage
        self._domain_states: Dict[str, Dict[str, Any]] = {}  # Isolated domain states
        self._domain_knowledge: Dict[str, Dict[str, Any]] = {}  # Isolated domain knowledge
        self._isolation_metadata: Dict[str, Dict[str, Any]] = {}  # Isolation tracking
        self._domain_access_logs: Dict[str, List[Dict[str, Any]]] = {}  # Access audit trail
        
        self._lock = threading.RLock()
        self._persistence_path = persistence_path
        
        # Setup logging
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Statistics
        self._total_registrations = 0
        self._total_removals = 0
        self._registry_created = datetime.now()
        
        # Reserved domain names
        self._reserved_names = {
            'core', 'brain', 'system', 'admin', 'root', 'shared',
            'foundation', 'base', 'common', 'global', 'all'
        }
        
        # Load persisted state if available
        if self._persistence_path and self._persistence_path.exists():
            self._load_registry()
        
        self.logger.info("DomainRegistry initialized with isolation support")
    
    def register_domain(self, domain_name: str, domain_config: Optional[DomainConfig] = None) -> bool:
        """
        Register a new domain.
        
        Args:
            domain_name: Unique name for the domain
            domain_config: Configuration for the domain (uses defaults if None)
            
        Returns:
            True if registration successful, False otherwise
        """
        if not isinstance(domain_name, str) or not domain_name.strip():
            self.logger.error("Domain name must be a non-empty string")
            return False
        
        domain_name = domain_name.strip().lower()
        
        # Validate domain name
        is_valid, error = self._validate_domain_name(domain_name)
        if not is_valid:
            self.logger.error(f"Invalid domain name '{domain_name}': {error}")
            return False
        
        # Use default config if none provided
        if domain_config is None:
            domain_config = DomainConfig()
        elif not isinstance(domain_config, DomainConfig):
            self.logger.error("domain_config must be a DomainConfig instance")
            return False
        
        # Validate configuration
        is_valid, errors = domain_config.validate()
        if not is_valid:
            self.logger.error(f"Invalid configuration for domain '{domain_name}': {', '.join(errors)}")
            return False
        
        with self._lock:
            # Check if already registered
            if domain_name in self._domains:
                self.logger.warning(f"Domain '{domain_name}' is already registered")
                return False
            
            # Check dependencies exist
            for dep in domain_config.dependencies:
                if dep not in self._domains:
                    self.logger.error(f"Dependency '{dep}' not found for domain '{domain_name}'")
                    return False
            
            # Create metadata
            metadata = DomainMetadata(
                name=domain_name,
                config=domain_config,
                status=DomainStatus.REGISTERED
            )
            
            # Register domain
            self._domains[domain_name] = metadata
            self._domain_order.append(domain_name)
            
            # Update dependency graph
            self._dependencies[domain_name] = set(domain_config.dependencies)
            for dep in domain_config.dependencies:
                if dep not in self._dependents:
                    self._dependents[dep] = set()
                self._dependents[dep].add(domain_name)
            
            self._total_registrations += 1
            
            # Persist if configured
            if self._persistence_path:
                self._persist_registry()
            
            self.logger.info(f"Domain '{domain_name}' registered successfully "
                           f"(type: {domain_config.domain_type.value}, "
                           f"version: {domain_config.version})")
            
            return True
    
    def get_domain_info(self, domain_name: str) -> Optional[Dict[str, Any]]:
        """
        Get domain metadata and information.
        
        Args:
            domain_name: Name of the domain
            
        Returns:
            Dictionary with domain information or None if not found
        """
        if not isinstance(domain_name, str):
            return None
        
        domain_name = domain_name.strip().lower()
        
        with self._lock:
            if domain_name not in self._domains:
                return None
            
            metadata = self._domains[domain_name]
            
            # Update last accessed
            metadata.last_accessed = datetime.now()
            
            # Build comprehensive info
            info = {
                'name': metadata.name,
                'status': metadata.status.value,
                'type': metadata.config.domain_type.value,
                'version': metadata.config.version,
                'description': metadata.config.description,
                'registered_at': metadata.registered_at.isoformat(),
                'last_updated': metadata.last_updated.isoformat(),
                'last_accessed': metadata.last_accessed.isoformat(),
                'activation_count': metadata.activation_count,
                'error_count': metadata.error_count,
                'total_predictions': metadata.total_predictions,
                'average_confidence': metadata.average_confidence,
                'resource_usage': metadata.resource_usage,
                'config': metadata.config.to_dict(),
                'dependencies': list(self._dependencies.get(domain_name, [])),
                'dependents': list(self._dependents.get(domain_name, [])),
                'metadata': metadata.metadata
            }
            
            return info
    
    def list_domains(self, status_filter: Optional[DomainStatus] = None,
                    type_filter: Optional[DomainType] = None) -> List[Dict[str, Any]]:
        """
        List all registered domains with optional filtering.
        
        Args:
            status_filter: Optional filter by domain status
            type_filter: Optional filter by domain type
            
        Returns:
            List of domain information dictionaries
        """
        with self._lock:
            domains = []
            
            for domain_name in self._domain_order:
                metadata = self._domains[domain_name]
                
                # Apply filters
                if status_filter and metadata.status != status_filter:
                    continue
                if type_filter and metadata.config.domain_type != type_filter:
                    continue
                
                # Get basic info
                domain_info = {
                    'name': metadata.name,
                    'status': metadata.status.value,
                    'type': metadata.config.domain_type.value,
                    'version': metadata.config.version,
                    'description': metadata.config.description,
                    'registered_at': metadata.registered_at.isoformat(),
                    'activation_count': metadata.activation_count,
                    'error_count': metadata.error_count,
                    'dependencies': list(self._dependencies.get(domain_name, [])),
                    'dependents': list(self._dependents.get(domain_name, []))
                }
                
                domains.append(domain_info)
            
            return domains
    
    def remove_domain(self, domain_name: str, force: bool = False) -> bool:
        """
        Unregister a domain from the registry.
        
        Args:
            domain_name: Name of the domain to remove
            force: If True, remove even if other domains depend on it
            
        Returns:
            True if removal successful, False otherwise
        """
        if not isinstance(domain_name, str):
            return False
        
        domain_name = domain_name.strip().lower()
        
        with self._lock:
            if domain_name not in self._domains:
                self.logger.warning(f"Domain '{domain_name}' not found in registry")
                return False
            
            # Check for dependents
            dependents = self._dependents.get(domain_name, set())
            if dependents and not force:
                self.logger.error(f"Cannot remove domain '{domain_name}': "
                                f"other domains depend on it: {', '.join(dependents)}")
                return False
            
            # Remove from all data structures
            metadata = self._domains.pop(domain_name)
            self._domain_order.remove(domain_name)
            
            # Update dependency graph
            # Remove this domain's dependencies
            if domain_name in self._dependencies:
                for dep in self._dependencies[domain_name]:
                    if dep in self._dependents:
                        self._dependents[dep].discard(domain_name)
                        if not self._dependents[dep]:
                            del self._dependents[dep]
                del self._dependencies[domain_name]
            
            # Remove as dependent (for force removal)
            if domain_name in self._dependents:
                del self._dependents[domain_name]
            
            # Clean up isolation data
            self._domain_states.pop(domain_name, None)
            self._domain_knowledge.pop(domain_name, None)
            self._isolation_metadata.pop(domain_name, None)
            self._domain_access_logs.pop(domain_name, None)
            
            self._total_removals += 1
            
            # Persist if configured
            if self._persistence_path:
                self._persist_registry()
            
            self.logger.info(f"Domain '{domain_name}' removed from registry "
                           f"(was {metadata.status.value}, "
                           f"had {metadata.activation_count} activations)")
            
            return True
    
    def is_domain_registered(self, domain_name: str) -> bool:
        """
        Check if a domain is registered.
        
        Args:
            domain_name: Name of the domain to check
            
        Returns:
            True if domain is registered, False otherwise
        """
        if not isinstance(domain_name, str):
            return False
        
        domain_name = domain_name.strip().lower()
        
        with self._lock:
            return domain_name in self._domains
    
    def update_domain_status(self, domain_name: str, status: DomainStatus,
                           error_message: Optional[str] = None) -> bool:
        """
        Update the status of a registered domain.
        
        Args:
            domain_name: Name of the domain
            status: New status for the domain
            error_message: Optional error message if status is ERROR
            
        Returns:
            True if update successful, False otherwise
        """
        if not isinstance(domain_name, str):
            return False
        
        domain_name = domain_name.strip().lower()
        
        with self._lock:
            if domain_name not in self._domains:
                self.logger.warning(f"Domain '{domain_name}' not found")
                return False
            
            metadata = self._domains[domain_name]
            old_status = metadata.status
            
            # Update status
            metadata.status = status
            metadata.last_updated = datetime.now()
            
            # Handle error status
            if status == DomainStatus.ERROR:
                metadata.error_count += 1
                if error_message:
                    metadata.error_messages.append(f"{datetime.now().isoformat()}: {error_message}")
                    # Keep only last 10 error messages
                    metadata.error_messages = metadata.error_messages[-10:]
            
            # Track activation
            if status == DomainStatus.ACTIVE and old_status != DomainStatus.ACTIVE:
                metadata.activation_count += 1
            
            self.logger.debug(f"Domain '{domain_name}' status updated: "
                            f"{old_status.value} -> {status.value}")
            
            # Persist if configured
            if self._persistence_path:
                self._persist_registry()
            
            return True
    
    def update_domain_metrics(self, domain_name: str, metrics: Dict[str, Any]) -> bool:
        """
        Update performance metrics for a domain.
        
        Args:
            domain_name: Name of the domain
            metrics: Dictionary of metrics to update
            
        Returns:
            True if update successful, False otherwise
        """
        if not isinstance(domain_name, str):
            return False
        
        domain_name = domain_name.strip().lower()
        
        with self._lock:
            if domain_name not in self._domains:
                return False
            
            metadata = self._domains[domain_name]
            
            # Update metrics
            if 'predictions' in metrics:
                metadata.total_predictions += metrics['predictions']
            
            if 'confidence' in metrics:
                # Update running average
                new_conf = metrics['confidence']
                if metadata.average_confidence == 0:
                    metadata.average_confidence = new_conf
                else:
                    # Exponential moving average
                    alpha = 0.1
                    metadata.average_confidence = (
                        alpha * new_conf + (1 - alpha) * metadata.average_confidence
                    )
            
            if 'resource_usage' in metrics:
                metadata.resource_usage.update(metrics['resource_usage'])
            
            metadata.last_updated = datetime.now()
            
            return True
    
    def get_domain_dependencies(self, domain_name: str, 
                              recursive: bool = False) -> List[str]:
        """
        Get dependencies for a domain.
        
        Args:
            domain_name: Name of the domain
            recursive: If True, get all transitive dependencies
            
        Returns:
            List of dependency domain names
        """
        if not isinstance(domain_name, str):
            return []
        
        domain_name = domain_name.strip().lower()
        
        with self._lock:
            if domain_name not in self._domains:
                return []
            
            if not recursive:
                return list(self._dependencies.get(domain_name, []))
            
            # Get transitive dependencies
            visited = set()
            to_visit = [domain_name]
            dependencies = []
            
            while to_visit:
                current = to_visit.pop(0)
                if current in visited:
                    continue
                
                visited.add(current)
                current_deps = self._dependencies.get(current, [])
                
                for dep in current_deps:
                    if dep not in visited:
                        dependencies.append(dep)
                        to_visit.append(dep)
            
            return dependencies
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the registry.
        
        Returns:
            Dictionary with registry statistics
        """
        with self._lock:
            # Count domains by status
            status_counts = {}
            for status in DomainStatus:
                count = sum(1 for d in self._domains.values() if d.status == status)
                status_counts[status.value] = count
            
            # Count domains by type
            type_counts = {}
            for dtype in DomainType:
                count = sum(1 for d in self._domains.values() 
                          if d.config.domain_type == dtype)
                type_counts[dtype.value] = count
            
            # Calculate resource allocation
            total_memory = sum(d.config.max_memory_mb for d in self._domains.values())
            total_cpu = sum(d.config.max_cpu_percent for d in self._domains.values())
            
            # Get error statistics
            total_errors = sum(d.error_count for d in self._domains.values())
            domains_with_errors = sum(1 for d in self._domains.values() if d.error_count > 0)
            
            return {
                'registry_created': self._registry_created.isoformat(),
                'total_domains': len(self._domains),
                'total_registrations': self._total_registrations,
                'total_removals': self._total_removals,
                'status_distribution': status_counts,
                'type_distribution': type_counts,
                'total_memory_allocated_mb': total_memory,
                'total_cpu_allocated_percent': total_cpu,
                'total_errors': total_errors,
                'domains_with_errors': domains_with_errors,
                'domains_with_dependencies': sum(1 for deps in self._dependencies.values() if deps),
                'most_activated': self._get_most_activated_domains(5),
                'highest_confidence': self._get_highest_confidence_domains(5)
            }
    
    def validate_dependency_graph(self) -> Tuple[bool, List[str]]:
        """
        Validate the dependency graph for cycles and missing dependencies.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        with self._lock:
            # Check for missing dependencies
            for domain, deps in self._dependencies.items():
                for dep in deps:
                    if dep not in self._domains:
                        errors.append(f"Domain '{domain}' has missing dependency: '{dep}'")
            
            # Check for circular dependencies
            for domain in self._domains:
                visited = set()
                path = []
                if self._has_circular_dependency(domain, visited, path):
                    errors.append(f"Circular dependency detected: {' -> '.join(path)}")
            
            return len(errors) == 0, errors
    
    def _validate_domain_name(self, domain_name: str) -> Tuple[bool, str]:
        """
        Validate a domain name.
        
        Args:
            domain_name: Domain name to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check length
        if len(domain_name) < 3:
            return False, "Domain name must be at least 3 characters"
        if len(domain_name) > 50:
            return False, "Domain name cannot exceed 50 characters"
        
        # Check format (alphanumeric with underscores and hyphens)
        if not re.match(r'^[a-z0-9_-]+$', domain_name):
            return False, "Domain name must contain only lowercase letters, numbers, underscores, and hyphens"
        
        # Check reserved names
        if domain_name in self._reserved_names:
            return False, f"'{domain_name}' is a reserved domain name"
        
        # Check for valid start/end characters
        if domain_name.startswith(('_', '-')) or domain_name.endswith(('_', '-')):
            return False, "Domain name cannot start or end with underscore or hyphen"
        
        return True, ""
    
    def _has_circular_dependency(self, domain: str, visited: Set[str], 
                               path: List[str]) -> bool:
        """
        Check if a domain has circular dependencies.
        
        Args:
            domain: Domain to check
            visited: Set of visited domains
            path: Current dependency path
            
        Returns:
            True if circular dependency found
        """
        if domain in path:
            path.append(domain)
            return True
        
        if domain in visited:
            return False
        
        visited.add(domain)
        path.append(domain)
        
        for dep in self._dependencies.get(domain, []):
            if self._has_circular_dependency(dep, visited, path):
                return True
        
        path.pop()
        return False
    
    def _get_most_activated_domains(self, limit: int) -> List[Dict[str, Any]]:
        """Get domains with highest activation counts."""
        sorted_domains = sorted(
            self._domains.values(),
            key=lambda x: x.activation_count,
            reverse=True
        )[:limit]
        
        return [
            {
                'name': d.name,
                'activation_count': d.activation_count,
                'status': d.status.value
            }
            for d in sorted_domains
        ]
    
    def _get_highest_confidence_domains(self, limit: int) -> List[Dict[str, Any]]:
        """Get domains with highest average confidence."""
        # Filter domains with predictions
        domains_with_predictions = [
            d for d in self._domains.values() 
            if d.total_predictions > 0
        ]
        
        sorted_domains = sorted(
            domains_with_predictions,
            key=lambda x: x.average_confidence,
            reverse=True
        )[:limit]
        
        return [
            {
                'name': d.name,
                'average_confidence': d.average_confidence,
                'total_predictions': d.total_predictions
            }
            for d in sorted_domains
        ]
    
    def _persist_registry(self) -> None:
        """Persist registry state to disk."""
        if not self._persistence_path:
            return
        
        try:
            # Prepare data for serialization
            data = {
                'version': '1.0.0',
                'created': self._registry_created.isoformat(),
                'updated': datetime.now().isoformat(),
                'statistics': {
                    'total_registrations': self._total_registrations,
                    'total_removals': self._total_removals
                },
                'domains': {
                    name: metadata.to_dict()
                    for name, metadata in self._domains.items()
                },
                'domain_order': self._domain_order,
                'dependencies': {k: list(v) for k, v in self._dependencies.items()},
                'dependents': {k: list(v) for k, v in self._dependents.items()},
                'isolation_data': {
                    'domain_states': self._domain_states,
                    'domain_knowledge': self._domain_knowledge,
                    'isolation_metadata': self._isolation_metadata,
                    'domain_access_logs': self._domain_access_logs
                }
            }
            
            # Write atomically
            temp_path = self._persistence_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2, cls=TensorJSONEncoder)
            
            temp_path.replace(self._persistence_path)
            
            self.logger.debug(f"Registry state persisted to {self._persistence_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to persist registry: {e}")
    
    def _load_registry(self) -> None:
        """Load registry state from disk."""
        if not self._persistence_path or not self._persistence_path.exists():
            return
        
        try:
            with open(self._persistence_path, 'r') as f:
                data = json.load(f)
            
            # Restore statistics
            self._registry_created = datetime.fromisoformat(data['created'])
            self._total_registrations = data['statistics']['total_registrations']
            self._total_removals = data['statistics']['total_removals']
            
            # Restore domains
            self._domains = {}
            for name, metadata_dict in data['domains'].items():
                self._domains[name] = DomainMetadata.from_dict(metadata_dict)
            
            # Restore order
            self._domain_order = data['domain_order']
            
            # Restore dependencies
            self._dependencies = {k: set(v) for k, v in data['dependencies'].items()}
            self._dependents = {k: set(v) for k, v in data['dependents'].items()}
            
            # Restore isolation data if present
            if 'isolation_data' in data:
                isolation_data = data['isolation_data']
                self._domain_states = isolation_data.get('domain_states', {})
                self._domain_knowledge = isolation_data.get('domain_knowledge', {})
                self._isolation_metadata = isolation_data.get('isolation_metadata', {})
                self._domain_access_logs = isolation_data.get('domain_access_logs', {})
            
            isolated_count = len(self._domain_states)
            self.logger.info(f"Registry state loaded from {self._persistence_path} "
                           f"({len(self._domains)} domains, {isolated_count} isolated)")
            
        except Exception as e:
            self.logger.error(f"Failed to load registry: {e}")
    
    def export_registry(self, format: str = 'json') -> Optional[str]:
        """
        Export registry data in specified format.
        
        Args:
            format: Export format ('json', 'yaml')
            
        Returns:
            Exported data as string or None on failure
        """
        with self._lock:
            try:
                # Prepare export data
                export_data = {
                    'registry_info': {
                        'version': '1.0.0',
                        'exported_at': datetime.now().isoformat(),
                        'total_domains': len(self._domains)
                    },
                    'domains': [
                        self.get_domain_info(name)
                        for name in self._domain_order
                    ]
                }
                
                if format == 'json':
                    return json.dumps(export_data, indent=2)
                elif format == 'yaml':
                    try:
                        import yaml
                        return yaml.dump(export_data, default_flow_style=False)
                    except ImportError:
                        self.logger.error("PyYAML not installed")
                        return None
                else:
                    self.logger.error(f"Unsupported export format: {format}")
                    return None
                    
            except Exception as e:
                self.logger.error(f"Failed to export registry: {e}")
                return None
    
    def reset(self) -> None:
        """Reset the registry to initial state."""
        with self._lock:
            self._domains.clear()
            self._domain_order.clear()
            self._dependencies.clear()
            self._dependents.clear()
            
            # Clear isolation data
            self._domain_states.clear()
            self._domain_knowledge.clear()
            self._isolation_metadata.clear()
            self._domain_access_logs.clear()
            
            self._total_registrations = 0
            self._total_removals = 0
            self._registry_created = datetime.now()
            
            if self._persistence_path and self._persistence_path.exists():
                try:
                    self._persistence_path.unlink()
                except Exception as e:
                    self.logger.error(f"Failed to remove persistence file: {e}")
            
            self.logger.info("Registry reset to initial state")
    
    def create_domain_isolation(self, domain_name: str) -> bool:
        """
        Create isolated domain space to prevent catastrophic forgetting.
        
        Args:
            domain_name: Name of the domain to isolate
            
        Returns:
            True if isolation created successfully, False otherwise
        """
        if not isinstance(domain_name, str):
            return False
        
        domain_name = domain_name.strip().lower()
        
        with self._lock:
            if domain_name not in self._domains:
                self.logger.error(f"Domain '{domain_name}' not found - cannot create isolation")
                return False
            
            # Check if isolation already exists
            if domain_name in self._domain_states:
                self.logger.warning(f"Domain '{domain_name}' isolation already exists")
                return True
            
            try:
                # Create isolated containers
                self._domain_states[domain_name] = {
                    'neural_state': {},
                    'learning_state': {},
                    'inference_state': {},
                    'memory_state': {},
                    'performance_metrics': {},
                    'last_checkpoint': None,
                    'created_at': datetime.now().isoformat(),
                    'size_bytes': 0
                }
                
                self._domain_knowledge[domain_name] = {
                    'facts': {},
                    'rules': {},
                    'patterns': {},
                    'embeddings': {},
                    'relationships': {},
                    'metadata': {
                        'total_items': 0,
                        'last_updated': datetime.now().isoformat(),
                        'version': 1
                    }
                }
                
                domain_config = self._domains[domain_name].config
                self._isolation_metadata[domain_name] = {
                    'created_at': datetime.now().isoformat(),
                    'access_count': 0,
                    'last_accessed': None,
                    'violation_count': 0,
                    'cross_domain_access_allowed': domain_config.allow_cross_domain_access,
                    'resource_limits': {
                        'max_state_size_mb': domain_config.max_memory_mb * 0.7,  # 70% for state
                        'max_knowledge_items': 10000,
                        'max_embedding_dim': 1024
                    },
                    'backup_count': 0,
                    'integrity_checks': 0
                }
                
                self._domain_access_logs[domain_name] = []
                
                self.logger.info(f"Domain isolation created for '{domain_name}'")
                
                # Persist if configured
                if self._persistence_path:
                    self._persist_registry()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to create isolation for domain '{domain_name}': {e}")
                # Cleanup partial creation
                self._domain_states.pop(domain_name, None)
                self._domain_knowledge.pop(domain_name, None)
                self._isolation_metadata.pop(domain_name, None)
                self._domain_access_logs.pop(domain_name, None)
                return False
    
    def get_domain_state(self, domain_name: str, requesting_domain: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get isolated domain state with access control.
        
        Args:
            domain_name: Name of the domain to get state from
            requesting_domain: Name of domain making the request (for access control)
            
        Returns:
            Dictionary with domain state or None if access denied
        """
        if not isinstance(domain_name, str):
            return None
        
        domain_name = domain_name.strip().lower()
        
        with self._lock:
            # Ensure isolation exists
            if not self._ensure_isolation(domain_name):
                return None
            
            # Check access permissions
            if requesting_domain and requesting_domain != domain_name:
                isolation_meta = self._isolation_metadata[domain_name]
                if not isolation_meta['cross_domain_access_allowed']:
                    self._log_access_violation(domain_name, requesting_domain, 'get_state')
                    return None
            
            # Log access
            self._log_domain_access(domain_name, requesting_domain or 'system', 'get_state')
            
            # Update access metadata
            isolation_meta = self._isolation_metadata[domain_name]
            isolation_meta['access_count'] += 1
            isolation_meta['last_accessed'] = datetime.now().isoformat()
            
            # Return deep copy to maintain isolation
            import copy
            return copy.deepcopy(self._domain_states[domain_name])
    
    def set_domain_state(self, domain_name: str, state: Dict[str, Any], 
                        requesting_domain: Optional[str] = None) -> bool:
        """
        Set isolated domain state with validation and access control.
        
        Args:
            domain_name: Name of the domain to set state for
            state: New state dictionary
            requesting_domain: Name of domain making the request (for access control)
            
        Returns:
            True if state was set successfully, False otherwise
        """
        if not isinstance(domain_name, str) or not isinstance(state, dict):
            return False
        
        domain_name = domain_name.strip().lower()
        
        with self._lock:
            # Ensure isolation exists
            if not self._ensure_isolation(domain_name):
                return False
            
            # Check access permissions
            if requesting_domain and requesting_domain != domain_name:
                isolation_meta = self._isolation_metadata[domain_name]
                if not isolation_meta['cross_domain_access_allowed']:
                    self._log_access_violation(domain_name, requesting_domain, 'set_state')
                    return False
            
            try:
                # Validate state size
                import sys
                state_size_mb = sys.getsizeof(state) / (1024 * 1024)
                max_size_mb = self._isolation_metadata[domain_name]['resource_limits']['max_state_size_mb']
                
                if state_size_mb > max_size_mb:
                    self.logger.error(f"State size {state_size_mb:.2f}MB exceeds limit {max_size_mb}MB for domain '{domain_name}'")
                    return False
                
                # Create backup of current state
                import copy
                backup_state = copy.deepcopy(self._domain_states[domain_name])
                
                # Update state
                self._domain_states[domain_name] = copy.deepcopy(state)
                self._domain_states[domain_name]['last_updated'] = datetime.now().isoformat()
                self._domain_states[domain_name]['size_bytes'] = sys.getsizeof(state)
                
                # Log access
                self._log_domain_access(domain_name, requesting_domain or 'system', 'set_state')
                
                # Update metadata
                isolation_meta = self._isolation_metadata[domain_name]
                isolation_meta['backup_count'] += 1
                
                self.logger.debug(f"Domain state updated for '{domain_name}' (size: {state_size_mb:.2f}MB)")
                
                # Persist if configured
                if self._persistence_path:
                    self._persist_registry()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to set domain state for '{domain_name}': {e}")
                # Restore backup if we have one
                if 'backup_state' in locals():
                    self._domain_states[domain_name] = backup_state
                return False
    
    def isolate_domain_knowledge(self, domain_name: str, knowledge: Dict[str, Any], 
                                knowledge_type: str = 'facts', 
                                requesting_domain: Optional[str] = None) -> bool:
        """
        Store domain-specific knowledge in isolated container.
        
        Args:
            domain_name: Name of the domain
            knowledge: Knowledge data to store
            knowledge_type: Type of knowledge ('facts', 'rules', 'patterns', 'embeddings')
            requesting_domain: Name of domain making the request
            
        Returns:
            True if knowledge was stored successfully, False otherwise
        """
        if not isinstance(domain_name, str) or not isinstance(knowledge, dict):
            return False
        
        domain_name = domain_name.strip().lower()
        
        valid_types = {'facts', 'rules', 'patterns', 'embeddings', 'relationships'}
        if knowledge_type not in valid_types:
            self.logger.error(f"Invalid knowledge type '{knowledge_type}'. Must be one of: {valid_types}")
            return False
        
        with self._lock:
            # Ensure isolation exists
            if not self._ensure_isolation(domain_name):
                return False
            
            # Check access permissions
            if requesting_domain and requesting_domain != domain_name:
                isolation_meta = self._isolation_metadata[domain_name]
                if not isolation_meta['cross_domain_access_allowed']:
                    self._log_access_violation(domain_name, requesting_domain, f'isolate_knowledge_{knowledge_type}')
                    return False
            
            try:
                # Check knowledge count limits
                current_count = len(self._domain_knowledge[domain_name][knowledge_type])
                max_items = self._isolation_metadata[domain_name]['resource_limits']['max_knowledge_items']
                
                if current_count + len(knowledge) > max_items:
                    self.logger.error(f"Knowledge count would exceed limit {max_items} for domain '{domain_name}'")
                    return False
                
                # Store knowledge with timestamps
                timestamp = datetime.now().isoformat()
                for key, value in knowledge.items():
                    self._domain_knowledge[domain_name][knowledge_type][key] = {
                        'data': value,
                        'stored_at': timestamp,
                        'accessed_count': 0,
                        'last_accessed': None,
                        'source_domain': requesting_domain or domain_name
                    }
                
                # Update metadata
                self._domain_knowledge[domain_name]['metadata']['total_items'] += len(knowledge)
                self._domain_knowledge[domain_name]['metadata']['last_updated'] = timestamp
                self._domain_knowledge[domain_name]['metadata']['version'] += 1
                
                # Log access
                self._log_domain_access(domain_name, requesting_domain or 'system', 
                                      f'isolate_knowledge_{knowledge_type}', 
                                      extra={'items_added': len(knowledge)})
                
                self.logger.debug(f"Stored {len(knowledge)} {knowledge_type} items for domain '{domain_name}'")
                
                # Persist if configured
                if self._persistence_path:
                    self._persist_registry()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to isolate knowledge for domain '{domain_name}': {e}")
                return False
    
    def get_domain_knowledge(self, domain_name: str, knowledge_type: Optional[str] = None,
                           requesting_domain: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve domain-specific knowledge with access control.
        
        Args:
            domain_name: Name of the domain
            knowledge_type: Specific type to retrieve or None for all
            requesting_domain: Name of domain making the request
            
        Returns:
            Dictionary with knowledge data or None if access denied
        """
        if not isinstance(domain_name, str):
            return None
        
        domain_name = domain_name.strip().lower()
        
        with self._lock:
            # Ensure isolation exists
            if not self._ensure_isolation(domain_name):
                return None
            
            # Check access permissions
            if requesting_domain and requesting_domain != domain_name:
                isolation_meta = self._isolation_metadata[domain_name]
                if not isolation_meta['cross_domain_access_allowed']:
                    self._log_access_violation(domain_name, requesting_domain, f'get_knowledge_{knowledge_type or "all"}')
                    return None
            
            # Log access
            self._log_domain_access(domain_name, requesting_domain or 'system', 
                                  f'get_knowledge_{knowledge_type or "all"}')
            
            # Return requested knowledge
            import copy
            if knowledge_type:
                if knowledge_type in self._domain_knowledge[domain_name]:
                    return copy.deepcopy(self._domain_knowledge[domain_name][knowledge_type])
                else:
                    return {}
            else:
                return copy.deepcopy(self._domain_knowledge[domain_name])
    
    def _ensure_isolation(self, domain_name: str) -> bool:
        """
        Ensure domain isolation exists, create if needed.
        
        Args:
            domain_name: Name of the domain
            
        Returns:
            True if isolation exists or was created, False otherwise
        """
        if domain_name not in self._domains:
            self.logger.error(f"Domain '{domain_name}' not registered")
            return False
        
        if domain_name not in self._domain_states:
            self.logger.debug(f"Auto-creating isolation for domain '{domain_name}'")
            return self.create_domain_isolation(domain_name)
        
        return True
    
    def _log_domain_access(self, domain_name: str, accessing_entity: str, 
                          operation: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log domain access for audit trail.
        
        Args:
            domain_name: Name of the domain being accessed
            accessing_entity: Name of entity making the access
            operation: Type of operation performed
            extra: Additional metadata
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'accessing_entity': accessing_entity,
            'operation': operation,
            'success': True
        }
        
        if extra:
            log_entry.update(extra)
        
        if domain_name in self._domain_access_logs:
            self._domain_access_logs[domain_name].append(log_entry)
            # Keep only last 1000 entries
            if len(self._domain_access_logs[domain_name]) > 1000:
                self._domain_access_logs[domain_name] = self._domain_access_logs[domain_name][-1000:]
    
    def _log_access_violation(self, domain_name: str, requesting_domain: str, operation: str) -> None:
        """
        Log access violation for security audit.
        
        Args:
            domain_name: Name of the domain being accessed
            requesting_domain: Name of domain making unauthorized request
            operation: Type of operation attempted
        """
        violation_entry = {
            'timestamp': datetime.now().isoformat(),
            'requesting_domain': requesting_domain,
            'operation': operation,
            'success': False,
            'violation_type': 'unauthorized_cross_domain_access'
        }
        
        if domain_name in self._domain_access_logs:
            self._domain_access_logs[domain_name].append(violation_entry)
        
        # Update violation count
        if domain_name in self._isolation_metadata:
            self._isolation_metadata[domain_name]['violation_count'] += 1
        
        self.logger.warning(f"Access violation: domain '{requesting_domain}' attempted {operation} on '{domain_name}'")
    
    def get_isolation_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about domain isolation.
        
        Returns:
            Dictionary with isolation statistics
        """
        with self._lock:
            stats = {
                'total_isolated_domains': len(self._domain_states),
                'total_knowledge_items': 0,
                'total_access_logs': 0,
                'total_violations': 0,
                'isolation_details': {},
                'knowledge_distribution': {},
                'access_patterns': {}
            }
            
            for domain_name in self._domain_states:
                if domain_name in self._isolation_metadata:
                    meta = self._isolation_metadata[domain_name]
                    knowledge = self._domain_knowledge.get(domain_name, {})
                    
                    # Domain-specific stats
                    domain_stats = {
                        'status': self._domains[domain_name].status.value,
                        'access_count': meta['access_count'],
                        'violation_count': meta['violation_count'],
                        'cross_domain_access_allowed': meta['cross_domain_access_allowed'],
                        'knowledge_items': knowledge.get('metadata', {}).get('total_items', 0),
                        'state_size_bytes': self._domain_states[domain_name].get('size_bytes', 0),
                        'created_at': meta['created_at'],
                        'last_accessed': meta['last_accessed']
                    }
                    
                    stats['isolation_details'][domain_name] = domain_stats
                    stats['total_knowledge_items'] += domain_stats['knowledge_items']
                    stats['total_violations'] += domain_stats['violation_count']
                    
                    # Knowledge type distribution
                    for knowledge_type in ['facts', 'rules', 'patterns', 'embeddings', 'relationships']:
                        if knowledge_type not in stats['knowledge_distribution']:
                            stats['knowledge_distribution'][knowledge_type] = 0
                        stats['knowledge_distribution'][knowledge_type] += len(knowledge.get(knowledge_type, {}))
                    
                    # Access logs count
                    if domain_name in self._domain_access_logs:
                        stats['total_access_logs'] += len(self._domain_access_logs[domain_name])
            
            return stats
    
    def clear_domain_isolation(self, domain_name: str, requesting_domain: Optional[str] = None) -> bool:
        """
        Clear domain isolation data (but keep domain registered).
        
        Args:
            domain_name: Name of the domain
            requesting_domain: Name of domain making the request
            
        Returns:
            True if isolation was cleared, False otherwise
        """
        if not isinstance(domain_name, str):
            return False
        
        domain_name = domain_name.strip().lower()
        
        with self._lock:
            if domain_name not in self._domains:
                self.logger.error(f"Domain '{domain_name}' not registered")
                return False
            
            # Check access permissions (only allow self-clearing or system)
            if requesting_domain and requesting_domain != domain_name:
                self.logger.error(f"Domain '{requesting_domain}' cannot clear isolation for '{domain_name}'")
                return False
            
            try:
                # Remove isolation data
                self._domain_states.pop(domain_name, None)
                self._domain_knowledge.pop(domain_name, None)
                self._isolation_metadata.pop(domain_name, None)
                self._domain_access_logs.pop(domain_name, None)
                
                self.logger.info(f"Isolation cleared for domain '{domain_name}'")
                
                # Persist if configured
                if self._persistence_path:
                    self._persist_registry()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to clear isolation for domain '{domain_name}': {e}")
                return False
    
    def setup_fraud_detection_domain(self, domain_name: str = "fraud_detection", 
                                   domain_path: Optional[Path] = None,
                                   auto_create_structure: bool = True,
                                   ieee_features: bool = True) -> Dict[str, Any]:
        """
        Automatically setup a comprehensive fraud detection domain with all infrastructure.
        Integrates the fraud domain setup automation script functionality.
        
        Args:
            domain_name: Name for the fraud detection domain
            domain_path: Path where domain files should be created
            auto_create_structure: Whether to create directory structure automatically
            ieee_features: Whether to enable IEEE fraud detection features
            
        Returns:
            Dictionary with setup results and domain information
        """
        try:
            self.logger.info(f"Setting up automated fraud detection domain: {domain_name}")
            
            # Create comprehensive domain configuration for fraud detection
            fraud_config = DomainConfig(
                domain_type=DomainType.SPECIALIZED,
                description="Automated financial fraud detection with comprehensive infrastructure",
                version="1.0.0",
                max_memory_mb=2048,
                max_cpu_percent=50.0,
                priority=9,  # High priority for fraud detection
                hidden_layers=[339, 256, 128, 64] if ieee_features else [128, 64, 32],
                activation_function="relu",
                dropout_rate=0.3,
                learning_rate=0.001,
                enable_caching=True,
                cache_size=1000,
                enable_logging=True,
                log_level="INFO",
                shared_foundation_layers=3,
                allow_cross_domain_access=True,
                dependencies=["mathematics", "general"],
                author="Automated Fraud Detection Setup",
                tags=["finance", "fraud", "security", "ml", "risk", "ieee", "automated"]
            )
            
            # Enhanced configuration for fraud detection
            fraud_config.extra_config = {
                'proof_system_enabled': True,
                'feature_count': 339 if ieee_features else 128,
                'fraud_types': ['transaction', 'identity', 'payment', 'behavioral'],
                'automated_setup': True,
                'setup_timestamp': datetime.now().isoformat(),
                'domain_components': {
                    'core_fraud_logic': True,
                    'data_loader': True,
                    'api_interface': True,
                    'proof_system': True,
                    'monitoring': True,
                    'deployment': True
                },
                'ieee_configuration': {
                    'enabled': ieee_features,
                    'feature_columns': ['V1', 'V2', 'V3', 'V4', 'V5'] + [f'V{i}' for i in range(6, 340)] if ieee_features else None,
                    'transaction_features': ['Amount', 'Time'],
                    'categorical_features': ['Class']
                },
                'training_configuration': {
                    'epochs': 50,
                    'batch_size': 256,
                    'learning_rate': 0.001,
                    'validation_split': 0.2,
                    'early_stopping': True,
                    'proof_verification_frequency': 10
                },
                'deployment_configuration': {
                    'deployment_strategies': ['blue_green', 'canary', 'rolling'],
                    'kubernetes_enabled': True,
                    'aws_integration': True,
                    'auto_scaling': True,
                    'high_availability': True
                }
            }
            
            # Register the domain
            if not self.register_domain(domain_name, fraud_config):
                return {
                    'success': False,
                    'error': 'Failed to register fraud detection domain',
                    'domain_name': domain_name
                }
            
            # Create domain isolation
            isolation_success = self.create_domain_isolation(domain_name)
            
            # Get domain info after registration
            domain_info = self.get_domain_info(domain_name)
            
            setup_result = {
                'success': True,
                'domain_name': domain_name,
                'domain_info': domain_info,
                'isolation_created': isolation_success,
                'ieee_features_enabled': ieee_features,
                'automated_setup': True,
                'setup_timestamp': datetime.now().isoformat(),
                'components_configured': [
                    'domain_registration',
                    'domain_isolation',
                    'fraud_core_config',
                    'ieee_feature_config' if ieee_features else 'standard_feature_config',
                    'proof_system_config',
                    'deployment_config'
                ]
            }
            
            # If directory structure creation is requested and path is provided
            if auto_create_structure and domain_path:
                try:
                    structure_result = self._create_fraud_domain_structure(domain_path, domain_name, ieee_features)
                    setup_result.update({
                        'directory_structure_created': structure_result['success'],
                        'directory_structure_details': structure_result
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to create directory structure: {e}")
                    setup_result['directory_structure_created'] = False
                    setup_result['directory_structure_error'] = str(e)
            
            self.logger.info(f"Fraud detection domain '{domain_name}' setup completed successfully")
            return setup_result
            
        except Exception as e:
            self.logger.error(f"Failed to setup fraud detection domain: {e}")
            return {
                'success': False,
                'error': str(e),
                'domain_name': domain_name
            }
    
    def _create_fraud_domain_structure(self, base_path: Path, domain_name: str, ieee_features: bool) -> Dict[str, Any]:
        """
        Create the complete directory structure for fraud detection domain.
        Integrates functionality from fraud_domain_setup.sh script.
        """
        try:
            domain_path = base_path / f"{domain_name}_domain"
            domain_path.mkdir(parents=True, exist_ok=True)
            
            # Create directory structure
            directories = [
                "core", "data", "models", "proof_system", "symbolic", 
                "api", "config", "utils", "tests", "docs", "integration", 
                "monitoring", "deployment", "visualization"
            ]
            
            created_dirs = []
            for dir_name in directories:
                dir_path = domain_path / dir_name
                dir_path.mkdir(exist_ok=True)
                (dir_path / "__init__.py").touch()
                created_dirs.append(str(dir_path))
            
            # Create core files with templates
            files_created = []
            
            # Domain registration file
            registration_file = domain_path / "domain_registration.py"
            registration_content = self._generate_domain_registration_template(domain_name, ieee_features)
            registration_file.write_text(registration_content)
            files_created.append(str(registration_file))
            
            # Core fraud detection file
            core_file = domain_path / "core" / "fraud_core.py"
            core_content = self._generate_fraud_core_template(domain_name, ieee_features)
            core_file.write_text(core_content)
            files_created.append(str(core_file))
            
            # Data loader file
            data_file = domain_path / "data" / "data_loader.py"
            data_content = self._generate_data_loader_template(ieee_features)
            data_file.write_text(data_content)
            files_created.append(str(data_file))
            
            # Requirements file
            req_file = domain_path / "requirements.txt"
            req_content = self._generate_requirements_template(ieee_features)
            req_file.write_text(req_content)
            files_created.append(str(req_file))
            
            return {
                'success': True,
                'domain_path': str(domain_path),
                'directories_created': created_dirs,
                'files_created': files_created,
                'ieee_features_enabled': ieee_features
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create fraud domain structure: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    # Missing method aliases for backward compatibility
    def has_domain(self, domain_name: str) -> bool:
        """Check if domain exists (alias for is_domain_registered)"""
        return self.is_domain_registered(domain_name)
    
    def get_domain(self, domain_name: str) -> Optional[DomainMetadata]:
        """Get domain metadata by name"""
        with self._lock:
            return self._domains.get(domain_name.lower().strip()) if domain_name else None
    
    def activate_domain(self, domain_name: str) -> bool:
        """Activate a domain"""
        if not domain_name or not isinstance(domain_name, str):
            return False
            
        domain_name = domain_name.lower().strip()
        
        with self._lock:
            if domain_name not in self._domains:
                self.logger.error(f"Cannot activate non-existent domain '{domain_name}'")
                return False
            
            metadata = self._domains[domain_name]
            metadata.status = DomainStatus.ACTIVE
            metadata.activation_count += 1
            metadata.last_accessed = datetime.now()
            metadata.last_updated = datetime.now()
            
            self.logger.info(f"Domain '{domain_name}' activated (activation count: {metadata.activation_count})")
            return True
    
    def deactivate_domain(self, domain_name: str) -> bool:
        """Deactivate a domain"""
        if not domain_name or not isinstance(domain_name, str):
            return False
            
        domain_name = domain_name.lower().strip()
        
        with self._lock:
            if domain_name not in self._domains:
                self.logger.error(f"Cannot deactivate non-existent domain '{domain_name}'")
                return False
            
            metadata = self._domains[domain_name]
            metadata.status = DomainStatus.INACTIVE
            metadata.last_updated = datetime.now()
            
            self.logger.info(f"Domain '{domain_name}' deactivated")
            return True
    
    def list_domains(self) -> List[str]:
        """List all domain names (simple version for compatibility)"""
        with self._lock:
            return list(self._domains.keys())
    
    def list_domains_by_status(self, status: DomainStatus) -> List[str]:
        """List domains by status"""
        with self._lock:
            return [name for name, metadata in self._domains.items() 
                   if metadata.status == status]
    
    def list_domains_by_type(self, domain_type: DomainType) -> List[str]:
        """List domains by type"""
        with self._lock:
            return [name for name, metadata in self._domains.items() 
                   if metadata.config.domain_type == domain_type]
    
    def set_domain_state(self, domain_name: str, state_data: Dict[str, Any]) -> bool:
        """Set isolated state for a domain"""
        if not domain_name or not isinstance(domain_name, str):
            return False
            
        domain_name = domain_name.lower().strip()
        
        with self._lock:
            if domain_name not in self._domains:
                self.logger.error(f"Cannot set state for non-existent domain '{domain_name}'")
                return False
            
            self._domain_states[domain_name] = state_data.copy() if state_data else {}
            
            # Log access for audit trail
            if domain_name not in self._domain_access_logs:
                self._domain_access_logs[domain_name] = []
            
            self._domain_access_logs[domain_name].append({
                'action': 'set_state',
                'timestamp': datetime.now().isoformat(),
                'data_size': len(str(state_data))
            })
            
            self.logger.info(f"Domain isolation created for '{domain_name}'")
            return True
    
    def get_domain_state(self, domain_name: str) -> Dict[str, Any]:
        """Get isolated state for a domain"""
        if not domain_name or not isinstance(domain_name, str):
            return {}
            
        domain_name = domain_name.lower().strip()
        
        with self._lock:
            return self._domain_states.get(domain_name, {}).copy()
    
    def set_domain_knowledge(self, domain_name: str, knowledge_data: Dict[str, Any]) -> bool:
        """Set isolated knowledge for a domain"""
        if not domain_name or not isinstance(domain_name, str):
            return False
            
        domain_name = domain_name.lower().strip()
        
        with self._lock:
            if domain_name not in self._domains:
                self.logger.error(f"Cannot set knowledge for non-existent domain '{domain_name}'")
                return False
            
            self._domain_knowledge[domain_name] = knowledge_data.copy() if knowledge_data else {}
            
            # Log access for audit trail
            if domain_name not in self._domain_access_logs:
                self._domain_access_logs[domain_name] = []
            
            self._domain_access_logs[domain_name].append({
                'action': 'set_knowledge',
                'timestamp': datetime.now().isoformat(),
                'data_size': len(str(knowledge_data))
            })
            
            return True
    
    def get_domain_knowledge(self, domain_name: str) -> Dict[str, Any]:
        """Get isolated knowledge for a domain"""
        if not domain_name or not isinstance(domain_name, str):
            return {}
            
        domain_name = domain_name.lower().strip()
        
        with self._lock:
            return self._domain_knowledge.get(domain_name, {}).copy()
    
    def update_domain_statistics(self, domain_name: str, stats: Dict[str, Any]) -> bool:
        """Update domain statistics"""
        if not domain_name or not isinstance(domain_name, str):
            return False
            
        domain_name = domain_name.lower().strip()
        
        with self._lock:
            if domain_name not in self._domains:
                self.logger.error(f"Cannot update statistics for non-existent domain '{domain_name}'")
                return False
            
            metadata = self._domains[domain_name]
            
            # Update specific statistics
            if 'predictions' in stats:
                metadata.total_predictions = stats['predictions']
            if 'accuracy' in stats:
                metadata.average_confidence = stats['accuracy']  
            if 'memory_usage' in stats:
                metadata.resource_usage['memory'] = stats['memory_usage']
            
            metadata.last_updated = datetime.now()
            return True
    
    def save_registry(self) -> bool:
        """Save registry state to persistence file"""
        if not self._persistence_path:
            self.logger.warning("No persistence path configured, cannot save registry")
            return False
        
        try:
            with self._lock:
                # Prepare data for serialization
                registry_data = {
                    'domains': {name: metadata.to_dict() for name, metadata in self._domains.items()},
                    'domain_order': self._domain_order.copy(),
                    'dependencies': {k: list(v) for k, v in self._dependencies.items()},
                    'dependents': {k: list(v) for k, v in self._dependents.items()},
                    'domain_states': self._domain_states.copy(),
                    'domain_knowledge': self._domain_knowledge.copy(),
                    'isolation_metadata': self._isolation_metadata.copy(),
                    'domain_access_logs': self._domain_access_logs.copy(),
                    'total_registrations': self._total_registrations,
                    'total_removals': self._total_removals,
                    'registry_created': self._registry_created.isoformat()
                }
                
                # Save to file
                with open(self._persistence_path, 'w') as f:
                    json.dump(registry_data, f, cls=TensorJSONEncoder, indent=2)
                
                self.logger.info(f"Registry saved to {self._persistence_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")
            return False
    
    def _load_registry(self) -> bool:
        """Load registry state from persistence file"""
        try:
            with open(self._persistence_path, 'r') as f:
                registry_data = json.load(f)
            
            with self._lock:
                # Load domains
                for name, domain_data in registry_data.get('domains', {}).items():
                    metadata = DomainMetadata.from_dict(domain_data)
                    self._domains[name] = metadata
                
                # Load other data
                self._domain_order = registry_data.get('domain_order', [])
                
                deps = registry_data.get('dependencies', {})
                self._dependencies = {k: set(v) for k, v in deps.items()}
                
                deps = registry_data.get('dependents', {})
                self._dependents = {k: set(v) for k, v in deps.items()}
                
                self._domain_states = registry_data.get('domain_states', {})
                self._domain_knowledge = registry_data.get('domain_knowledge', {})
                self._isolation_metadata = registry_data.get('isolation_metadata', {})
                self._domain_access_logs = registry_data.get('domain_access_logs', {})
                
                self._total_registrations = registry_data.get('total_registrations', 0)
                self._total_removals = registry_data.get('total_removals', 0)
                
                if 'registry_created' in registry_data:
                    self._registry_created = datetime.fromisoformat(registry_data['registry_created'])
            
            self.logger.info(f"Registry loaded from {self._persistence_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load registry: {e}")
            return False
    
    def _generate_domain_registration_template(self, domain_name: str, ieee_features: bool) -> str:
        """Generate domain registration template content."""
        return f'''"""
{domain_name.title()} Domain Registration.
Auto-generated domain registration for comprehensive fraud detection.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "independent_core"))

from domain_registry import DomainConfig, DomainType
from brain import Brain


class {domain_name.title().replace('_', '')}DomainRegistration:
    """Handles registration of the {domain_name} domain."""
    
    def __init__(self, brain: Optional[Brain] = None):
        self.brain = brain
        self.logger = logging.getLogger(__name__)
        self.domain_name = "{domain_name}"
        self.domain_config = self._create_domain_config()
    
    def _create_domain_config(self) -> DomainConfig:
        """Create configuration for the {domain_name} domain."""
        return DomainConfig(
            domain_type=DomainType.SPECIALIZED,
            description="{domain_name} with {'IEEE' if ieee_features else 'standard'} features",
            version="1.0.0",
            max_memory_mb=2048,
            max_cpu_percent=40.0,
            priority=9,
            hidden_layers={[339, 256, 128, 64] if ieee_features else [128, 64, 32]},
            activation_function="relu",
            dropout_rate=0.3,
            learning_rate=0.001,
            enable_caching=True,
            cache_size=500,
            enable_logging=True,
            log_level="INFO",
            shared_foundation_layers=3,
            allow_cross_domain_access=True,
            dependencies=["mathematics", "general"],
            author="Automated {domain_name.title()} Setup",
            tags=["finance", "fraud", "security", "ml", "risk"] + (["ieee"] if ieee_features else [])
        )
    
    def register(self) -> bool:
        """Register the {domain_name} domain."""
        try:
            if not self.brain:
                self.logger.error("No Brain instance provided")
                return False
            
            result = self.brain.register_domain(self.domain_name, self.domain_config)
            
            if result.get('success'):
                self.logger.info(f"Successfully registered {{self.domain_name}} domain")
                return True
            else:
                self.logger.error(f"Failed to register domain: {{result.get('error')}}")
                return False
                
        except Exception as e:
            self.logger.error(f"Registration error: {{e}}")
            return False


def register_{domain_name}_domain(brain: Brain) -> bool:
    """Convenience function to register the {domain_name} domain."""
    registration = {domain_name.title().replace('_', '')}DomainRegistration(brain)
    return registration.register()
'''
    
    def _generate_fraud_core_template(self, domain_name: str, ieee_features: bool) -> str:
        """Generate fraud core template content."""
        return f'''"""
Core {domain_name.title()} Detection Logic.
Auto-generated fraud detection orchestrator with {'IEEE' if ieee_features else 'standard'} features.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime


class {domain_name.title().replace('_', '')}DetectionCore:
    """Core {domain_name} detection orchestrator."""
    
    def __init__(self, config_manager=None):
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        self.ieee_features_enabled = {str(ieee_features).lower()}
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize fraud detection components."""
        try:
            self.logger.info("Initializing {domain_name} detection components...")
            
            # Placeholder implementations - ready for enhancement
            self.data_loader = "DataLoader_Ready"
            self.preprocessor = "Preprocessor_Ready" 
            self.ml_predictor = "MLPredictor_Ready"
            self.proof_verifier = "ProofVerifier_Ready"
            
            if self.ieee_features_enabled:
                self.feature_columns = [f'V{{i}}' for i in range(1, 340)]  # V1-V339
                self.logger.info("IEEE fraud detection features enabled (V1-V339)")
            else:
                self.feature_columns = ['amount', 'time', 'merchant_category', 'user_history']
                self.logger.info("Standard fraud detection features enabled")
            
            self.logger.info("Components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {{e}}")
            raise
    
    def detect_fraud(self, transaction: Union[Dict[str, Any], pd.DataFrame], 
                    generate_proof: bool = True) -> Dict[str, Any]:
        """Detect fraud in transaction(s)."""
        try:
            # Convert to DataFrame if needed
            if isinstance(transaction, dict):
                df = pd.DataFrame([transaction])
                single_transaction = True
            else:
                df = transaction
                single_transaction = False
            
            results = []
            for _, row in df.iterrows():
                # Enhanced fraud detection logic
                fraud_score = self._calculate_fraud_score(row)
                
                result = {{
                    'transaction_id': row.get('transaction_id', 'unknown'),
                    'fraud_score': fraud_score,
                    'is_fraud': fraud_score >= 0.7,
                    'risk_level': self._get_risk_level(fraud_score),
                    'explanation': self._generate_explanation(fraud_score, row),
                    'timestamp': datetime.now().isoformat(),
                    'detection_method': 'ieee_enhanced' if self.ieee_features_enabled else 'standard',
                    'features_used': len(self.feature_columns)
                }}
                
                results.append(result)
            
            return results[0] if single_transaction else results
            
        except Exception as e:
            self.logger.error(f"Fraud detection failed: {{e}}")
            raise
    
    def _calculate_fraud_score(self, row: pd.Series) -> float:
        """Calculate fraud score based on available features."""
        score = 0.0
        
        if self.ieee_features_enabled:
            # IEEE feature-based scoring (simplified for template)
            for feature in ['V1', 'V2', 'V3', 'V4', 'V5']:
                if feature in row:
                    # Simplified scoring - enhance with actual ML model
                    value = float(row[feature]) if pd.notna(row[feature]) else 0.0
                    if abs(value) > 2.0:  # Outlier detection
                        score += 0.15
        else:
            # Standard feature-based scoring
            amount = row.get('amount', 0)
            if amount > 5000:
                score += 0.4
            if amount > 10000:
                score += 0.3
                
            hour = pd.to_datetime(row.get('timestamp', datetime.now())).hour
            if hour < 6 or hour > 22:
                score += 0.3
        
        return min(score, 1.0)
    
    def _get_risk_level(self, score: float) -> str:
        """Convert fraud score to risk level."""
        if score >= 0.9: return 'critical'
        elif score >= 0.7: return 'high'  
        elif score >= 0.5: return 'medium'
        elif score >= 0.3: return 'low'
        else: return 'very_low'
    
    def _generate_explanation(self, score: float, row: pd.Series) -> str:
        """Generate human-readable explanation."""
        explanations = []
        
        if self.ieee_features_enabled:
            explanations.append(f"IEEE feature analysis ({{len(self.feature_columns)}} features)")
            if score > 0.7:
                explanations.append("Multiple anomalous feature patterns detected")
        else:
            amount = row.get('amount', 0)
            if amount > 10000:
                explanations.append("Very high transaction amount")
            elif amount > 5000:
                explanations.append("High transaction amount")
                
        if not explanations:
            explanations.append("Normal transaction pattern")
            
        return "; ".join(explanations)
'''
    
    def _generate_data_loader_template(self, ieee_features: bool) -> str:
        """Generate data loader template content."""
        return f'''"""
Data Loader for {'' if not ieee_features else 'IEEE '}Fraud Detection.
Auto-generated data loading and preprocessing with {'IEEE' if ieee_features else 'standard'} feature support.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta


class TransactionDataLoader:
    """Loads and manages transaction data for fraud detection."""
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or Path(__file__).parent
        self.logger = logging.getLogger(__name__)
        self.transaction_cache = {{}}
        self.ieee_features_enabled = {str(ieee_features).lower()}
        
        if self.ieee_features_enabled:
            self.feature_columns = [f'V{{i}}' for i in range(1, 340)] + ['Amount', 'Time']
            self.logger.info("IEEE feature columns initialized (V1-V339, Amount, Time)")
        else:
            self.feature_columns = ['amount', 'timestamp', 'user_id', 'merchant_id', 'transaction_type']
            self.logger.info("Standard feature columns initialized")
    
    def load_transactions(self, source: Union[str, Path, pd.DataFrame],
                         date_range: Optional[Tuple[datetime, datetime]] = None) -> pd.DataFrame:
        """Load transaction data from various sources."""
        try:
            if isinstance(source, pd.DataFrame):
                df = source
            elif isinstance(source, (str, Path)):
                df = self._load_from_file(source)
            else:
                raise ValueError(f"Unsupported source type: {{type(source)}}")
            
            if date_range:
                df = self._filter_by_date(df, date_range)
            
            df = self._validate_transaction_data(df)
            
            self.logger.info(f"Loaded {{len(df)}} transactions with {{'IEEE' if self.ieee_features_enabled else 'standard'}} features")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load transactions: {{e}}")
            raise
    
    def _load_from_file(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load data from file based on extension."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.csv':
            return pd.read_csv(filepath)
        elif filepath.suffix == '.parquet':
            return pd.read_parquet(filepath)
        elif filepath.suffix == '.json':
            return pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file type: {{filepath.suffix}}")
    
    def _validate_transaction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean transaction data."""
        if self.ieee_features_enabled:
            # IEEE dataset validation
            required_columns = ['Time', 'Amount'] + [f'V{{i}}' for i in range(1, 29)]  # Core IEEE columns
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                self.logger.warning(f"Missing IEEE columns: {{missing_cols}}")
                # Fill missing columns with zeros for template
                for col in missing_cols:
                    if col not in df.columns:
                        df[col] = 0.0
        else:
            # Standard validation
            required_columns = ['transaction_id', 'amount', 'timestamp', 'user_id']
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {{missing_cols}}")
        
        return df.dropna(subset=['Amount' if self.ieee_features_enabled else 'amount'])
    
    def generate_sample_data(self, num_transactions: int = 10000) -> pd.DataFrame:
        """Generate sample transaction data for testing."""
        np.random.seed(42)
        
        if self.ieee_features_enabled:
            # Generate IEEE-style synthetic data
            data = {{
                'Time': np.random.randint(0, 172800, num_transactions),  # 0-2 days in seconds
                'Amount': np.random.lognormal(mean=3, sigma=1.5, size=num_transactions)
            }}
            
            # Generate V1-V28 features (PCA transformed)
            for i in range(1, 29):
                data[f'V{{i}}'] = np.random.normal(0, 1, num_transactions)
            
            # Add fraud labels (2% fraudulent)
            fraud_indices = np.random.choice(num_transactions, size=int(num_transactions * 0.02))
            data['Class'] = np.zeros(num_transactions)
            data['Class'][fraud_indices] = 1
            
        else:
            # Generate standard synthetic data
            data = {{
                'transaction_id': [f'TX{{i:06d}}' for i in range(num_transactions)],
                'amount': np.random.lognormal(mean=5, sigma=1.5, size=num_transactions),
                'timestamp': pd.date_range(start='2024-01-01', periods=num_transactions, freq='5min'),
                'user_id': [f'USER{{np.random.randint(1, 1000):04d}}' for _ in range(num_transactions)],
                'merchant_id': [f'MERCH{{np.random.randint(1, 100):03d}}' for _ in range(num_transactions)],
                'transaction_type': np.random.choice(['purchase', 'withdrawal', 'transfer', 'deposit'], size=num_transactions)
            }}
        
        return pd.DataFrame(data)
'''
    
    def _generate_requirements_template(self, ieee_features: bool) -> str:
        """Generate requirements template content."""
        return f'''# Core dependencies for {'IEEE ' if ieee_features else ''}Fraud Detection Domain
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Data processing
pyarrow>=6.0.0

# ML utilities
joblib>=1.1.0

# Validation and testing
pytest>=7.0.0

# API and web
fastapi>=0.75.0
uvicorn>=0.17.0

# Utilities
python-dotenv>=0.19.0
click>=8.0.0

# Development
black>=22.1.0
flake8>=4.0.0

# Enhanced dependencies for {'IEEE ' if ieee_features else ''}features
{"""# IEEE-specific dependencies
scipy>=1.7.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0""" if ieee_features else "# Standard fraud detection dependencies"}

# Proof system dependencies (if using proof verification)
cryptography>=3.4.0
'''

    def __repr__(self) -> str:
        """String representation of registry."""
        return (f"DomainRegistry(domains={len(self._domains)}, "
                f"active={sum(1 for d in self._domains.values() if d.status == DomainStatus.ACTIVE)}, "
                f"total_registrations={self._total_registrations})")
    
    def __len__(self) -> int:
        """Get number of registered domains."""
        return len(self._domains)
    
    def __contains__(self, domain_name: str) -> bool:
        """Check if domain is registered using 'in' operator."""
        return self.is_domain_registered(domain_name)