"""
GAC System Configuration Management
Comprehensive configuration for Gradient Ascent Clipping system components
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

class GACMode(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    ADAPTIVE = "adaptive"

class ComponentPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class PIDConfig:
    kp: float = 1.0
    ki: float = 0.1
    kd: float = 0.05
    setpoint: float = 0.0
    integral_windup_limit: float = 100.0
    derivative_smoothing: float = 0.1
    output_limits: tuple = (-10.0, 10.0)
    auto_tune: bool = True
    sample_time: float = 0.1

@dataclass
class MetaLearningConfig:
    learning_rate: float = 0.01
    adaptation_threshold: float = 0.1
    history_limit: int = 1000
    minimum_samples: int = 10
    performance_window: int = 100
    convergence_threshold: float = 0.001
    feature_dimensions: int = 64
    hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.1
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10

@dataclass
class ReinforcementLearningConfig:
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    exploration_strategy: str = "epsilon_greedy"
    q_table_max_size: int = 100000
    state_discretization: Dict[str, int] = field(default_factory=lambda: {
        "performance_buckets": 10,
        "load_buckets": 10,
        "error_buckets": 10
    })
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "performance_improvement": 1.0,
        "stability_bonus": 0.5,
        "efficiency_bonus": 0.3,
        "error_penalty": -2.0
    })
    action_space: List[str] = field(default_factory=lambda: [
        "increase_threshold", "decrease_threshold", "maintain", 
        "boost_component", "reduce_sensitivity", "emergency_stop"
    ])

@dataclass
class ThresholdConfig:
    gradient_magnitude: float = 1.0
    processing_time: float = 5.0
    error_rate: float = 0.05
    memory_usage: float = 0.8
    cpu_usage: float = 0.7
    component_failure_rate: float = 0.1
    system_instability: float = 0.2
    adaptation_frequency: float = 0.05
    emergency_threshold_multiplier: float = 2.0
    auto_adjust: bool = True
    adjustment_sensitivity: float = 0.1

@dataclass
class MonitoringConfig:
    enabled: bool = True
    sampling_interval: float = 1.0
    metrics_retention: int = 3600
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "high_error_rate": 0.1,
        "high_latency": 10.0,
        "memory_pressure": 0.9,
        "component_failures": 5
    })
    notification_channels: List[str] = field(default_factory=lambda: ["log", "metrics"])
    performance_tracking: bool = True
    component_health_checks: bool = True
    automatic_recovery: bool = True
    escalation_levels: Dict[str, int] = field(default_factory=lambda: {
        "warning": 1,
        "error": 2,
        "critical": 3,
        "emergency": 4
    })

@dataclass
class ComponentConfig:
    component_id: str
    priority: ComponentPriority = ComponentPriority.MEDIUM
    enabled: bool = True
    auto_start: bool = True
    restart_on_failure: bool = True
    max_restart_attempts: int = 3
    restart_delay: float = 1.0
    timeout: float = 30.0
    memory_limit: Optional[int] = None
    cpu_limit: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    custom_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemConfig:
    mode: GACMode = GACMode.BALANCED
    max_workers: int = 8
    worker_timeout: float = 30.0
    graceful_shutdown_timeout: float = 10.0
    checkpoint_interval: int = 300
    checkpoint_retention: int = 10
    state_persistence: bool = True
    auto_recovery: bool = True
    performance_optimization: bool = True
    debug_mode: bool = False
    log_level: str = "INFO"

@dataclass
class IntegrationConfig:
    brain_integration: bool = True
    hook_timeout: float = 5.0
    async_processing: bool = True
    batch_processing: bool = True
    batch_size: int = 100
    pipeline_stages: List[str] = field(default_factory=lambda: [
        "preprocessing", "analysis", "processing", "postprocessing"
    ])
    external_apis: Dict[str, Dict[str, str]] = field(default_factory=dict)
    event_driven: bool = True
    real_time_processing: bool = True

@dataclass
class GACConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    pid: PIDConfig = field(default_factory=PIDConfig)
    meta_learning: MetaLearningConfig = field(default_factory=MetaLearningConfig)
    reinforcement_learning: ReinforcementLearningConfig = field(default_factory=ReinforcementLearningConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    components: Dict[str, ComponentConfig] = field(default_factory=dict)
    custom_settings: Dict[str, Any] = field(default_factory=dict)

class GACConfigManager:
    def __init__(self, config_path: Optional[str] = None, auto_load: bool = True):
        self.config_path = Path(config_path) if config_path else Path("./config/gac_config.json")
        self.config = GACConfig()
        
        if auto_load:
            self.load_config()
    
    def load_config(self) -> bool:
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
                return False
            
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            self._load_from_dict(config_data)
            logger.info(f"Configuration loaded from {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def save_config(self, create_backup: bool = True) -> bool:
        try:
            if create_backup and self.config_path.exists():
                backup_path = self.config_path.with_suffix('.backup')
                backup_path.write_text(self.config_path.read_text())
            
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            config_dict = asdict(self.config)
            with open(self.config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _load_from_dict(self, config_data: Dict[str, Any]):
        if "system" in config_data:
            self._load_system_config(config_data["system"])
        if "pid" in config_data:
            self._load_pid_config(config_data["pid"])
        if "meta_learning" in config_data:
            self._load_meta_learning_config(config_data["meta_learning"])
        if "reinforcement_learning" in config_data:
            self._load_rl_config(config_data["reinforcement_learning"])
        if "thresholds" in config_data:
            self._load_threshold_config(config_data["thresholds"])
        if "monitoring" in config_data:
            self._load_monitoring_config(config_data["monitoring"])
        if "integration" in config_data:
            self._load_integration_config(config_data["integration"])
        if "components" in config_data:
            self._load_component_configs(config_data["components"])
        if "custom_settings" in config_data:
            self.config.custom_settings = config_data["custom_settings"]
    
    def _load_system_config(self, data: Dict[str, Any]):
        for key, value in data.items():
            if hasattr(self.config.system, key):
                if key == "mode" and isinstance(value, str):
                    setattr(self.config.system, key, GACMode(value))
                else:
                    setattr(self.config.system, key, value)
    
    def _load_pid_config(self, data: Dict[str, Any]):
        for key, value in data.items():
            if hasattr(self.config.pid, key):
                setattr(self.config.pid, key, value)
    
    def _load_meta_learning_config(self, data: Dict[str, Any]):
        for key, value in data.items():
            if hasattr(self.config.meta_learning, key):
                setattr(self.config.meta_learning, key, value)
    
    def _load_rl_config(self, data: Dict[str, Any]):
        for key, value in data.items():
            if hasattr(self.config.reinforcement_learning, key):
                setattr(self.config.reinforcement_learning, key, value)
    
    def _load_threshold_config(self, data: Dict[str, Any]):
        for key, value in data.items():
            if hasattr(self.config.thresholds, key):
                setattr(self.config.thresholds, key, value)
    
    def _load_monitoring_config(self, data: Dict[str, Any]):
        for key, value in data.items():
            if hasattr(self.config.monitoring, key):
                setattr(self.config.monitoring, key, value)
    
    def _load_integration_config(self, data: Dict[str, Any]):
        for key, value in data.items():
            if hasattr(self.config.integration, key):
                setattr(self.config.integration, key, value)
    
    def _load_component_configs(self, data: Dict[str, Any]):
        for component_id, component_data in data.items():
            component_config = ComponentConfig(component_id=component_id)
            for key, value in component_data.items():
                if hasattr(component_config, key):
                    if key == "priority" and isinstance(value, str):
                        setattr(component_config, key, ComponentPriority(value))
                    else:
                        setattr(component_config, key, value)
            self.config.components[component_id] = component_config
    
    def get_component_config(self, component_id: str) -> Optional[ComponentConfig]:
        return self.config.components.get(component_id)
    
    def add_component_config(self, component_config: ComponentConfig):
        self.config.components[component_config.component_id] = component_config
    
    def remove_component_config(self, component_id: str):
        if component_id in self.config.components:
            del self.config.components[component_id]
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        try:
            self._deep_update(self.config, updates)
            return True
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False
    
    def _deep_update(self, config_obj: Any, updates: Dict[str, Any]):
        for key, value in updates.items():
            if hasattr(config_obj, key):
                current_value = getattr(config_obj, key)
                if isinstance(current_value, dict) and isinstance(value, dict):
                    current_value.update(value)
                elif hasattr(current_value, '__dict__') and isinstance(value, dict):
                    self._deep_update(current_value, value)
                else:
                    setattr(config_obj, key, value)
    
    def validate_config(self) -> tuple[bool, List[str]]:
        errors = []
        
        try:
            if self.config.pid.kp < 0:
                errors.append("PID Kp must be non-negative")
            
            if self.config.meta_learning.learning_rate <= 0 or self.config.meta_learning.learning_rate > 1:
                errors.append("Meta-learning rate must be between 0 and 1")
            
            if self.config.reinforcement_learning.discount_factor < 0 or self.config.reinforcement_learning.discount_factor > 1:
                errors.append("RL discount factor must be between 0 and 1")
            
            if self.config.thresholds.gradient_magnitude <= 0:
                errors.append("Gradient magnitude threshold must be positive")
            
            if self.config.system.max_workers <= 0:
                errors.append("Max workers must be positive")
            
            for component_id, component_config in self.config.components.items():
                if not component_config.component_id:
                    errors.append(f"Component {component_id} missing component_id")
                
                if component_config.timeout <= 0:
                    errors.append(f"Component {component_id} timeout must be positive")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Validation error: {e}")
            return False, errors
    
    def create_preset_config(self, preset: str) -> bool:
        try:
            if preset == "development":
                self.config.system.debug_mode = True
                self.config.system.log_level = "DEBUG"
                self.config.thresholds.auto_adjust = True
                self.config.monitoring.sampling_interval = 0.5
                
            elif preset == "production":
                self.config.system.debug_mode = False
                self.config.system.log_level = "WARNING"
                self.config.thresholds.auto_adjust = False
                self.config.monitoring.sampling_interval = 5.0
                self.config.system.performance_optimization = True
                
            elif preset == "testing":
                self.config.system.debug_mode = True
                self.config.system.log_level = "DEBUG"
                self.config.monitoring.enabled = False
                self.config.system.checkpoint_interval = 60
                
            elif preset == "conservative":
                self.config.system.mode = GACMode.CONSERVATIVE
                self.config.thresholds.gradient_magnitude = 0.5
                self.config.pid.kp = 0.5
                self.config.reinforcement_learning.epsilon = 0.05
                
            elif preset == "aggressive":
                self.config.system.mode = GACMode.AGGRESSIVE
                self.config.thresholds.gradient_magnitude = 2.0
                self.config.pid.kp = 2.0
                self.config.reinforcement_learning.epsilon = 0.2
                
            else:
                logger.error(f"Unknown preset: {preset}")
                return False
            
            logger.info(f"Applied preset configuration: {preset}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply preset {preset}: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        return {
            "system_mode": self.config.system.mode.value,
            "debug_mode": self.config.system.debug_mode,
            "max_workers": self.config.system.max_workers,
            "monitoring_enabled": self.config.monitoring.enabled,
            "component_count": len(self.config.components),
            "brain_integration": self.config.integration.brain_integration,
            "thresholds": {
                "gradient_magnitude": self.config.thresholds.gradient_magnitude,
                "processing_time": self.config.thresholds.processing_time,
                "error_rate": self.config.thresholds.error_rate
            }
        }

def create_default_config() -> GACConfig:
    return GACConfig()

def load_config_from_file(config_path: str) -> Optional[GACConfig]:
    manager = GACConfigManager(config_path, auto_load=True)
    return manager.config if manager.load_config() else None

if __name__ == "__main__":
    config_manager = GACConfigManager("./test_gac_config.json", auto_load=False)
    
    config_manager.create_preset_config("development")
    
    is_valid, errors = config_manager.validate_config()
    if is_valid:
        print("Configuration is valid")
        config_manager.save_config()
        print("Configuration saved")
    else:
        print(f"Configuration validation failed: {errors}")
    
    summary = config_manager.get_config_summary()
    print(f"Configuration summary: {json.dumps(summary, indent=2)}") 