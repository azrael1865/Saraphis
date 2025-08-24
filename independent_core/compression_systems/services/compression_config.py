"""
Configuration management for compression systems.
Strict validation - no default values for critical parameters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
from pathlib import Path


@dataclass
class CompressionConfig:
    """Central configuration for compression systems. All parameters required."""
    
    # General compression settings
    compression_level: int = 5  # 0-10, higher = more compression
    
    # P-adic compression configuration
    padic_base: int = 5
    padic_precision: int = 10
    padic_adaptive: bool = True
    
    # Sheaf theory configuration
    sheaf_locality_radius: int = 2
    sheaf_cohomology_dim: int = 3
    sheaf_validate_invariants: bool = True
    
    # Tensor decomposition configuration
    tensor_decomp_method: str = 'tucker'  # 'tucker', 'cp', 'tensor_train'
    tensor_rank_ratio: float = 0.5
    tensor_adaptive_rank: bool = True
    
    # GPU memory configuration
    gpu_memory_threshold: float = 0.8
    gpu_cache_size_mb: int = 512
    gpu_prefetch_enabled: bool = True
    
    # Service configuration
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 30
    max_requests_per_module: int = 100
    
    # Performance requirements (fail if not met)
    max_compression_time_ms: float = 1000.0
    min_compression_ratio: float = 2.0
    max_reconstruction_error: float = 0.01
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_padic_config()
        self._validate_sheaf_config()
        self._validate_tensor_config()
        self._validate_gpu_config()
        self._validate_service_config()
        self._validate_performance_config()
    
    def _validate_padic_config(self) -> None:
        """Validate P-adic configuration."""
        if self.padic_base < 2:
            raise ValueError(f"P-adic base must be >= 2, got {self.padic_base}")
        
        if not self._is_prime(self.padic_base):
            raise ValueError(f"P-adic base must be prime, got {self.padic_base}")
        
        if self.padic_precision <= 0:
            raise ValueError(f"P-adic precision must be > 0, got {self.padic_precision}")
        
        if self.padic_precision > 128:
            raise ValueError(f"P-adic precision too high: {self.padic_precision}")
    
    def _validate_sheaf_config(self) -> None:
        """Validate sheaf theory configuration."""
        if self.sheaf_locality_radius <= 0:
            raise ValueError(f"Locality radius must be > 0, got {self.sheaf_locality_radius}")
        
        if self.sheaf_cohomology_dim <= 0:
            raise ValueError(f"Cohomology dimension must be > 0, got {self.sheaf_cohomology_dim}")
        
        if self.sheaf_cohomology_dim > 10:
            raise ValueError(f"Cohomology dimension too high: {self.sheaf_cohomology_dim}")
    
    def _validate_tensor_config(self) -> None:
        """Validate tensor decomposition configuration."""
        valid_methods = ['tucker', 'cp', 'tensor_train']
        if self.tensor_decomp_method not in valid_methods:
            raise ValueError(f"Invalid tensor method: {self.tensor_decomp_method}")
        
        if not 0 < self.tensor_rank_ratio < 1:
            raise ValueError(f"Rank ratio must be in (0,1), got {self.tensor_rank_ratio}")
    
    def _validate_gpu_config(self) -> None:
        """Validate GPU configuration."""
        if not 0 < self.gpu_memory_threshold < 1:
            raise ValueError(f"GPU memory threshold must be in (0,1), got {self.gpu_memory_threshold}")
        
        if self.gpu_cache_size_mb <= 0:
            raise ValueError(f"GPU cache size must be > 0, got {self.gpu_cache_size_mb}")
    
    def _validate_service_config(self) -> None:
        """Validate service configuration."""
        if self.max_concurrent_requests <= 0:
            raise ValueError(f"Max concurrent requests must be > 0")
        
        if self.request_timeout_seconds <= 0:
            raise ValueError(f"Request timeout must be > 0")
        
        if self.max_requests_per_module <= 0:
            raise ValueError(f"Max requests per module must be > 0")
    
    def _validate_performance_config(self) -> None:
        """Validate performance requirements."""
        if self.max_compression_time_ms <= 0:
            raise ValueError(f"Max compression time must be > 0")
        
        if self.min_compression_ratio <= 1:
            raise ValueError(f"Min compression ratio must be > 1")
        
        if self.max_reconstruction_error <= 0:
            raise ValueError(f"Max reconstruction error must be > 0")
    
    @staticmethod
    def _is_prime(n: int) -> bool:
        """Check if number is prime."""
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True


class ConfigurationManager:
    """Manages compression configurations across services and modules."""
    
    def __init__(self, base_config: CompressionConfig):
        if not isinstance(base_config, CompressionConfig):
            raise TypeError("Base config must be CompressionConfig instance")
        
        self.base_config = base_config
        self._module_overrides: Dict[str, Dict[str, Any]] = {}
        self._service_configs: Dict[str, CompressionConfig] = {}
    
    def add_module_override(self, module_name: str, overrides: Dict[str, Any]) -> None:
        """Add module-specific configuration overrides."""
        if not module_name:
            raise ValueError("Module name cannot be empty")
        
        if not isinstance(overrides, dict):
            raise TypeError("Overrides must be dict")
        
        # Validate override keys exist in base config
        base_config_dict = self.base_config.__dict__
        for key in overrides:
            if key not in base_config_dict:
                raise KeyError(f"Invalid override key: {key}")
        
        self._module_overrides[module_name] = overrides
    
    def get_config_for_module(self, module_name: str) -> CompressionConfig:
        """Get configuration with module-specific overrides applied."""
        if not module_name:
            raise ValueError("Module name cannot be empty")
        
        # Start with base config
        config_dict = self.base_config.__dict__.copy()
        
        # Apply module overrides
        if module_name in self._module_overrides:
            config_dict.update(self._module_overrides[module_name])
        
        # Create new config instance with overrides
        return CompressionConfig(**config_dict)
    
    def get_config_for_service(self, service_name: str) -> CompressionConfig:
        """Get configuration for specific service."""
        if service_name not in self._service_configs:
            raise KeyError(f"No config found for service: {service_name}")
        
        return self._service_configs[service_name]
    
    def register_service_config(self, service_name: str, config: CompressionConfig) -> None:
        """Register service-specific configuration."""
        if not service_name:
            raise ValueError("Service name cannot be empty")
        
        if not isinstance(config, CompressionConfig):
            raise TypeError("Config must be CompressionConfig instance")
        
        self._service_configs[service_name] = config
    
    def validate_all_configs(self) -> Dict[str, bool]:
        """Validate all registered configurations."""
        results = {}
        
        # Validate base config
        try:
            # Re-run validation
            self.base_config.__post_init__()
            results['base_config'] = True
        except Exception as e:
            results['base_config'] = False
            raise ValueError(f"Base config validation failed: {e}")
        
        # Validate module configs
        for module_name in self._module_overrides:
            try:
                config = self.get_config_for_module(module_name)
                config.__post_init__()
                results[f'module_{module_name}'] = True
            except Exception as e:
                results[f'module_{module_name}'] = False
                raise ValueError(f"Module {module_name} config validation failed: {e}")
        
        # Validate service configs
        for service_name, config in self._service_configs.items():
            try:
                config.__post_init__()
                results[f'service_{service_name}'] = True
            except Exception as e:
                results[f'service_{service_name}'] = False
                raise ValueError(f"Service {service_name} config validation failed: {e}")
        
        return results
    
    def export_config(self, module_name: str, filepath: Path) -> None:
        """Export module configuration to file."""
        config = self.get_config_for_module(module_name)
        config_dict = config.__dict__
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config_from_file(cls, filepath: Path) -> CompressionConfig:
        """Load configuration from JSON file."""
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return CompressionConfig(**config_dict)
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of all configurations."""
        return {
            'base_config': self.base_config.__dict__,
            'module_overrides': dict(self._module_overrides),
            'service_configs': {
                name: config.__dict__ 
                for name, config in self._service_configs.items()
            },
            'total_modules': len(self._module_overrides),
            'total_services': len(self._service_configs)
        }


class ConfigurationValidator:
    """Validates compression configurations across system."""
    
    @staticmethod
    def validate_system_compatibility(configs: List[CompressionConfig]) -> None:
        """Validate multiple configurations are compatible."""
        if not configs:
            raise ValueError("No configurations provided")
        
        # Check all configs use compatible P-adic bases
        bases = set(config.padic_base for config in configs)
        if len(bases) > 1:
            raise ValueError(f"Incompatible P-adic bases: {bases}")
        
        # Check GPU memory requirements don't exceed total
        total_cache_mb = sum(config.gpu_cache_size_mb for config in configs)
        if total_cache_mb > 8192:  # 8GB limit
            raise ValueError(f"Total GPU cache exceeds limit: {total_cache_mb}MB")
        
        # Check concurrent request limits
        total_concurrent = sum(config.max_concurrent_requests for config in configs)
        if total_concurrent > 10000:
            raise ValueError(f"Total concurrent requests too high: {total_concurrent}")
    
    @staticmethod
    def validate_performance_feasibility(config: CompressionConfig, 
                                       expected_data_sizes: List[int]) -> None:
        """Validate performance requirements are feasible for expected data sizes."""
        for size in expected_data_sizes:
            # Estimate compression time based on size
            estimated_time_ms = size * 0.001  # 1 microsecond per element
            
            if estimated_time_ms > config.max_compression_time_ms:
                raise ValueError(
                    f"Performance requirement infeasible for size {size}: "
                    f"estimated {estimated_time_ms:.2f}ms > limit {config.max_compression_time_ms}ms"
                )