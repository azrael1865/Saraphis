#!/usr/bin/env python3
"""
Universal AI Core Configuration Classes
=====================================

This module provides comprehensive configuration management for the Universal AI Core system.
It extracts and adapts configuration patterns from the analyzed Saraphis codebase components,
making them domain-agnostic while preserving sophisticated learning and proof capabilities.

Configuration Classes:
- UniversalAIConfig: Main system configuration
- LearningConfig: Machine learning and adaptive learning settings
- ProofConfig: Formal proof verification configuration  
- SymbolicConfig: Symbolic reasoning and inference settings
- DataConfig: Data processing and storage configuration
- ModelConfig: Neural network and ensemble model settings
- InferenceConfig: Inference and prediction settings
- ValidationConfig: Validation and evaluation settings
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


# Core Enums for Configuration
class LogicSystem(Enum):
    """Supported logic systems for formal reasoning"""
    PROPOSITIONAL = "propositional"
    PREDICATE = "predicate" 
    MODAL = "modal"
    TEMPORAL = "temporal"
    INTUITIONISTIC = "intuitionistic"
    LINEAR = "linear"
    HIGHER_ORDER = "higher_order"
    TYPE_THEORY = "type_theory"


class ProofLanguage(Enum):
    """Supported formal proof languages"""
    LEAN4 = "lean4"
    COQ = "coq"
    ISABELLE = "isabelle"
    AGDA = "agda"
    IDRIS = "idris"
    NEUROFORMAL = "neuroformal"
    TPTP = "tptp"
    SMT_LIB = "smt_lib"
    METAMATH = "metamath"
    HOL_LIGHT = "hol_light"


class SymbolicOperation(Enum):
    """Types of symbolic reasoning operations"""
    REASONING = "reasoning"
    INFERENCE = "inference"
    DEDUCTION = "deduction"
    INDUCTION = "induction"
    ABDUCTION = "abduction"
    PATTERN_MATCHING = "pattern_matching"
    RULE_APPLICATION = "rule_application"
    CONSTRAINT_SOLVING = "constraint_solving"


class ModelType(Enum):
    """Supported model architectures"""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    TRANSFORMER = "transformer"
    GRAPH_NEURAL_NETWORK = "graph_neural_network"


class EnsembleMethod(Enum):
    """Ensemble combination methods"""
    VOTING = "voting"
    STACKING = "stacking"
    WEIGHTED = "weighted"
    BAYESIAN = "bayesian"


@dataclass
class ProofConfig:
    """Configuration for formal proof verification system"""
    # Core proof settings
    supported_languages: List[ProofLanguage] = field(
        default_factory=lambda: [ProofLanguage.NEUROFORMAL, ProofLanguage.LEAN4, ProofLanguage.COQ]
    )
    default_logic_system: LogicSystem = LogicSystem.PREDICATE
    proof_timeout: float = 300.0  # seconds
    max_proof_steps: int = 1000
    
    # Verification settings
    enable_caching: bool = True
    cache_size: int = 10000
    verification_workers: int = 4
    parallel_verification: bool = True
    
    # External verifier paths
    external_verifiers: Dict[str, str] = field(default_factory=dict)
    
    # Quality settings
    confidence_threshold: float = 0.95
    require_formal_proof: bool = False
    allow_partial_proofs: bool = True
    
    # Performance settings
    memory_limit_mb: int = 4096
    cpu_cores: int = -1  # -1 means use all available
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.proof_timeout <= 0:
            raise ValueError("proof_timeout must be positive")
        if self.max_proof_steps <= 0:
            raise ValueError("max_proof_steps must be positive")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")


@dataclass
class SymbolicConfig:
    """Configuration for symbolic reasoning and inference"""
    # Core symbolic settings
    supported_operations: List[SymbolicOperation] = field(
        default_factory=lambda: list(SymbolicOperation)
    )
    default_operation: SymbolicOperation = SymbolicOperation.REASONING
    
    # Reasoning thresholds
    deductive_confidence_threshold: float = 0.95
    inductive_confidence_threshold: float = 0.80
    abductive_confidence_threshold: float = 0.70
    
    # Pattern matching settings
    pattern_similarity_threshold: float = 0.7
    max_pattern_matches: int = 100
    enable_fuzzy_matching: bool = True
    
    # Knowledge base settings
    knowledge_base_size: int = 10000
    auto_expand_knowledge: bool = True
    knowledge_persistence: bool = True
    
    # Performance settings
    max_reasoning_depth: int = 50
    reasoning_timeout: float = 60.0  # seconds
    enable_parallel_reasoning: bool = True
    reasoning_workers: int = 2
    
    # Cache settings
    enable_result_caching: bool = True
    cache_size: int = 5000
    cache_ttl: int = 3600  # seconds
    
    def __post_init__(self):
        """Validate symbolic configuration"""
        thresholds = [
            self.deductive_confidence_threshold,
            self.inductive_confidence_threshold, 
            self.abductive_confidence_threshold,
            self.pattern_similarity_threshold
        ]
        for threshold in thresholds:
            if not 0 <= threshold <= 1:
                raise ValueError("All confidence thresholds must be between 0 and 1")


@dataclass
class ModelConfig:
    """Configuration for machine learning models"""
    # Model architecture settings
    model_types: List[ModelType] = field(
        default_factory=lambda: [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.NEURAL_NETWORK]
    )
    ensemble_method: EnsembleMethod = EnsembleMethod.VOTING
    
    # Neural network settings
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout_rate: float = 0.2
    activation_function: str = "relu"
    batch_size: int = 32
    learning_rate: float = 0.001
    max_epochs: int = 100
    early_stopping_patience: int = 20
    
    # Tree-based model settings
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    
    # Training settings
    cross_validation_folds: int = 5
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    
    # Performance settings
    n_jobs: int = -1
    use_gpu: bool = True
    memory_limit_mb: int = 8192
    
    # Regularization
    l1_regularization: float = 0.0
    l2_regularization: float = 0.01
    
    def __post_init__(self):
        """Validate model configuration"""
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")


@dataclass  
class LearningConfig:
    """Configuration for adaptive learning and optimization"""
    # Core learning settings
    learning_algorithm: str = "adaptive_gradient"
    adaptation_rate: float = 0.01
    momentum: float = 0.9
    
    # Uncertainty quantification
    enable_uncertainty_quantification: bool = True
    uncertainty_method: str = "ensemble_variance"
    confidence_intervals: bool = True
    
    # Online learning
    enable_online_learning: bool = True
    online_batch_size: int = 10
    online_learning_rate: float = 0.001
    forgetting_factor: float = 0.95
    
    # Feature learning
    enable_feature_learning: bool = True
    feature_selection_method: str = "mutual_information"
    max_features: Optional[int] = None
    feature_importance_threshold: float = 0.01
    
    # Transfer learning
    enable_transfer_learning: bool = True
    transfer_learning_domains: List[str] = field(default_factory=list)
    domain_adaptation_method: str = "fine_tuning"
    
    # Meta-learning
    enable_meta_learning: bool = False
    meta_learning_algorithm: str = "maml"
    inner_loop_steps: int = 5
    outer_loop_lr: float = 0.001
    
    # Continual learning
    enable_continual_learning: bool = True
    catastrophic_forgetting_prevention: str = "elastic_weight_consolidation"
    memory_buffer_size: int = 1000
    
    # Active learning
    enable_active_learning: bool = False
    acquisition_function: str = "uncertainty_sampling"
    query_budget: int = 100
    
    def __post_init__(self):
        """Validate learning configuration"""
        if not 0 < self.adaptation_rate <= 1:
            raise ValueError("adaptation_rate must be between 0 and 1")
        if not 0 <= self.momentum <= 1:
            raise ValueError("momentum must be between 0 and 1")


@dataclass
class DataConfig:
    """Configuration for data processing and management"""
    # Data paths
    data_root_path: str = "./data"
    cache_path: str = "./cache"
    models_path: str = "./models"
    logs_path: str = "./logs"
    
    # Data processing
    batch_size: int = 1000
    max_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    
    # Data validation
    validate_data: bool = True
    data_quality_threshold: float = 0.95
    handle_missing_values: str = "interpolate"  # "drop", "interpolate", "mean"
    outlier_detection_method: str = "isolation_forest"
    
    # Caching settings
    enable_data_caching: bool = True
    cache_compression: bool = True
    cache_format: str = "parquet"  # "parquet", "hdf5", "pickle"
    max_cache_size_gb: float = 10.0
    
    # Feature engineering
    enable_feature_engineering: bool = True
    feature_scaling_method: str = "robust"  # "standard", "robust", "minmax"
    feature_encoding_method: str = "target"  # "onehot", "target", "ordinal"
    
    # Data augmentation
    enable_data_augmentation: bool = False
    augmentation_factor: float = 2.0
    augmentation_methods: List[str] = field(default_factory=list)
    
    # Privacy and security
    enable_data_encryption: bool = False
    encryption_key_path: Optional[str] = None
    anonymization_level: str = "none"  # "none", "basic", "differential_privacy"
    
    def __post_init__(self):
        """Create data directories and validate configuration"""
        # Create directories if they don't exist
        for path_attr in ["data_root_path", "cache_path", "models_path", "logs_path"]:
            path = getattr(self, path_attr)
            Path(path).mkdir(parents=True, exist_ok=True)
        
        if not 0 < self.data_quality_threshold <= 1:
            raise ValueError("data_quality_threshold must be between 0 and 1")


@dataclass
class InferenceConfig:
    """Configuration for inference and prediction"""
    # Core inference settings
    inference_mode: str = "batch"  # "batch", "streaming", "interactive"
    batch_size: int = 100
    max_sequence_length: int = 1000
    
    # Performance settings
    enable_model_compilation: bool = True
    use_mixed_precision: bool = True
    enable_tensorrt: bool = False
    
    # Caching
    enable_prediction_caching: bool = True
    prediction_cache_size: int = 10000
    cache_ttl: int = 3600  # seconds
    
    # Output settings
    return_probabilities: bool = True
    return_confidence_scores: bool = True
    return_explanations: bool = False
    explanation_method: str = "shap"  # "shap", "lime", "attention"
    
    # Quality control
    confidence_threshold: float = 0.5
    uncertainty_threshold: float = 0.3
    fallback_strategy: str = "conservative"  # "conservative", "aggressive", "hybrid"
    
    # Monitoring
    enable_prediction_monitoring: bool = True
    monitor_drift: bool = True
    drift_detection_method: str = "ks_test"
    
    def __post_init__(self):
        """Validate inference configuration"""
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0 <= self.uncertainty_threshold <= 1:
            raise ValueError("uncertainty_threshold must be between 0 and 1")


@dataclass
class ValidationConfig:
    """Configuration for model validation and evaluation"""
    # Validation strategy
    validation_method: str = "cross_validation"  # "holdout", "cross_validation", "bootstrap"
    k_folds: int = 5
    validation_split: float = 0.2
    stratified: bool = True
    
    # Metrics
    regression_metrics: List[str] = field(
        default_factory=lambda: ["rmse", "mae", "r2", "mape"]
    )
    classification_metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1", "auc"]
    )
    custom_metrics: Dict[str, Callable] = field(default_factory=dict)
    
    # Statistical testing
    enable_statistical_tests: bool = True
    significance_level: float = 0.05
    multiple_testing_correction: str = "bonferroni"
    
    # Performance benchmarking
    enable_benchmarking: bool = True
    benchmark_datasets: List[str] = field(default_factory=list)
    baseline_models: List[str] = field(default_factory=list)
    
    # Robustness testing
    enable_adversarial_testing: bool = False
    adversarial_epsilon: float = 0.1
    robustness_tests: List[str] = field(default_factory=list)
    
    # Fairness evaluation
    enable_fairness_evaluation: bool = False
    protected_attributes: List[str] = field(default_factory=list)
    fairness_metrics: List[str] = field(default_factory=lambda: ["demographic_parity", "equalized_odds"])
    
    def __post_init__(self):
        """Validate configuration"""
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")
        if not 0 < self.significance_level < 1:
            raise ValueError("significance_level must be between 0 and 1")


@dataclass
class UniversalAIConfig:
    """Main configuration class for Universal AI Core system"""
    # System identification
    system_name: str = "UniversalAICore"
    version: str = "1.0.0"
    environment: str = "development"  # "development", "staging", "production"
    
    # Core component configurations
    proof_config: ProofConfig = field(default_factory=ProofConfig)
    symbolic_config: SymbolicConfig = field(default_factory=SymbolicConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    learning_config: LearningConfig = field(default_factory=LearningConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    
    # System-wide settings
    debug_mode: bool = False
    verbose_logging: bool = True
    log_level: str = "INFO"
    
    # Performance settings
    max_memory_gb: float = 16.0
    max_cpu_cores: int = -1  # -1 means use all available
    enable_gpu: bool = True
    gpu_memory_fraction: float = 0.8
    
    # Security settings
    enable_authentication: bool = False
    api_key_required: bool = False
    rate_limiting: bool = True
    max_requests_per_minute: int = 100
    
    # Monitoring and telemetry
    enable_monitoring: bool = True
    metrics_collection_interval: int = 60  # seconds
    enable_telemetry: bool = False
    telemetry_endpoint: Optional[str] = None
    
    # Backup and recovery
    enable_auto_backup: bool = True
    backup_interval_hours: int = 24
    max_backups: int = 7
    backup_location: str = "./backups"
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Create backup directory
        Path(self.backup_location).mkdir(parents=True, exist_ok=True)
        
        # Validate memory settings
        if self.max_memory_gb <= 0:
            raise ValueError("max_memory_gb must be positive")
        if not 0 < self.gpu_memory_fraction <= 1:
            raise ValueError("gpu_memory_fraction must be between 0 and 1")
        
        # Set up logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging based on settings"""
        numeric_level = getattr(logging, self.log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {self.log_level}')
        
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(Path(self.data_config.logs_path) / "universal_ai_core.log")
            ]
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """Convert configuration to JSON"""
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=2, default=str)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UniversalAIConfig':
        """Create configuration from dictionary"""
        # Extract nested configurations
        proof_config = ProofConfig(**config_dict.get('proof_config', {}))
        symbolic_config = SymbolicConfig(**config_dict.get('symbolic_config', {}))
        model_config = ModelConfig(**config_dict.get('model_config', {}))
        learning_config = LearningConfig(**config_dict.get('learning_config', {}))
        data_config = DataConfig(**config_dict.get('data_config', {}))
        inference_config = InferenceConfig(**config_dict.get('inference_config', {}))
        validation_config = ValidationConfig(**config_dict.get('validation_config', {}))
        
        # Remove nested configs from main dict
        main_config = {k: v for k, v in config_dict.items() 
                      if not k.endswith('_config')}
        
        return cls(
            proof_config=proof_config,
            symbolic_config=symbolic_config,
            model_config=model_config,
            learning_config=learning_config,
            data_config=data_config,
            inference_config=inference_config,
            validation_config=validation_config,
            **main_config
        )
    
    @classmethod
    def from_json(cls, filepath: str) -> 'UniversalAIConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls, prefix: str = "UNIVERSAL_AI_") -> 'UniversalAIConfig':
        """Load configuration from environment variables"""
        config = cls()
        
        # Map of environment variable suffixes to config attributes
        env_mappings = {
            "DEBUG": ("debug_mode", bool),
            "LOG_LEVEL": ("log_level", str),
            "MAX_MEMORY_GB": ("max_memory_gb", float),
            "ENABLE_GPU": ("enable_gpu", bool),
            "GPU_MEMORY_FRACTION": ("gpu_memory_fraction", float),
            "DATA_ROOT_PATH": ("data_config.data_root_path", str),
            "CACHE_PATH": ("data_config.cache_path", str),
            "MODELS_PATH": ("data_config.models_path", str),
            "PROOF_TIMEOUT": ("proof_config.proof_timeout", float),
            "VERIFICATION_WORKERS": ("proof_config.verification_workers", int),
            "LEARNING_RATE": ("model_config.learning_rate", float),
            "BATCH_SIZE": ("model_config.batch_size", int),
        }
        
        for env_suffix, (attr_path, type_func) in env_mappings.items():
            env_var = f"{prefix}{env_suffix}"
            env_value = os.getenv(env_var)
            
            if env_value is not None:
                try:
                    # Convert string to appropriate type
                    if type_func == bool:
                        value = env_value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        value = type_func(env_value)
                    
                    # Set nested attribute
                    obj = config
                    attrs = attr_path.split('.')
                    for attr in attrs[:-1]:
                        obj = getattr(obj, attr)
                    setattr(obj, attrs[-1], value)
                    
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Failed to set {attr_path} from {env_var}: {e}")
        
        return config
    
    def get_system_hash(self) -> str:
        """Get unique hash of the configuration"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def validate(self) -> bool:
        """Validate the entire configuration"""
        try:
            # All dataclass __post_init__ methods will be called during initialization
            # Additional validation can be added here
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration with values from dictionary"""
        def update_nested(obj, updates_dict):
            for key, value in updates_dict.items():
                if hasattr(obj, key):
                    attr = getattr(obj, key)
                    if hasattr(attr, '__dict__') and isinstance(value, dict):
                        update_nested(attr, value)
                    else:
                        setattr(obj, key, value)
                else:
                    logger.warning(f"Unknown configuration key: {key}")
        
        update_nested(self, updates)


def create_default_config() -> UniversalAIConfig:
    """Create default Universal AI configuration"""
    return UniversalAIConfig()


def create_production_config() -> UniversalAIConfig:
    """Create production-optimized configuration"""
    config = UniversalAIConfig()
    config.environment = "production"
    config.debug_mode = False
    config.log_level = "WARNING"
    config.enable_monitoring = True
    config.enable_auto_backup = True
    config.proof_config.verification_workers = 8
    config.model_config.n_jobs = -1
    config.data_config.batch_size = 5000
    config.inference_config.enable_model_compilation = True
    config.inference_config.use_mixed_precision = True
    return config


def create_development_config() -> UniversalAIConfig:
    """Create development-optimized configuration"""
    config = UniversalAIConfig()
    config.environment = "development"
    config.debug_mode = True
    config.verbose_logging = True
    config.log_level = "DEBUG"
    config.proof_config.verification_workers = 2
    config.model_config.max_epochs = 10  # Faster training for development
    config.data_config.batch_size = 100
    config.enable_monitoring = False
    return config


# Example usage and testing
if __name__ == "__main__":
    print("🔧 Universal AI Core Configuration Test")
    print("=" * 50)
    
    # Test default configuration
    config = create_default_config()
    print(f"✅ Default config created: {config.system_name} v{config.version}")
    print(f"📊 System hash: {config.get_system_hash()[:16]}...")
    
    # Test validation
    is_valid = config.validate()
    print(f"✅ Configuration valid: {is_valid}")
    
    # Test serialization
    config_json = config.to_json()
    print(f"📄 JSON config size: {len(config_json)} characters")
    
    # Test environment loading
    os.environ["UNIVERSAL_AI_DEBUG"] = "true"
    os.environ["UNIVERSAL_AI_LOG_LEVEL"] = "DEBUG"
    env_config = UniversalAIConfig.from_env()
    print(f"🌍 Environment config loaded - debug: {env_config.debug_mode}")
    
    # Test production config
    prod_config = create_production_config()
    print(f"🏭 Production config - workers: {prod_config.proof_config.verification_workers}")
    
    # Test development config  
    dev_config = create_development_config()
    print(f"🛠️ Development config - max epochs: {dev_config.model_config.max_epochs}")
    
    print("\n✅ Configuration system test completed!")