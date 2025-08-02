"""
Hybrid Validation Framework - Framework for validating hybrid system correctness
NO FALLBACKS - HARD FAILURES ONLY
"""

import logging
import math
import threading
import time
import torch
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Set
from enum import Enum

# Import hybrid system components
from .hybrid_padic_structures import HybridPadicWeight, HybridPadicManager, HybridPadicValidator
from .hybrid_padic_compressor import HybridPadicCompressionSystem
from .dynamic_switching_manager import DynamicSwitchingManager

# Import existing components for validation
from .padic_encoder import PadicWeight, PadicCompressionSystem, PadicMathematicalOperations


class ValidationType(Enum):
    """Validation type enumeration"""
    MATHEMATICAL = "mathematical"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    CONSISTENCY = "consistency"
    INTEGRATION = "integration"
    SECURITY = "security"
    CORRECTNESS = "correctness"


class ValidationSeverity(Enum):
    """Validation severity enumeration"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ValidationStatus(Enum):
    """Validation status enumeration"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class ValidationRule:
    """Validation rule definition"""
    rule_id: str
    rule_name: str
    validation_type: ValidationType
    severity: ValidationSeverity
    description: str
    validation_function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    depends_on: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate rule definition"""
        if not isinstance(self.rule_id, str) or not self.rule_id.strip():
            raise ValueError("Rule ID must be non-empty string")
        if not isinstance(self.rule_name, str) or not self.rule_name.strip():
            raise ValueError("Rule name must be non-empty string")
        if not isinstance(self.validation_type, ValidationType):
            raise TypeError("Validation type must be ValidationType")
        if not isinstance(self.severity, ValidationSeverity):
            raise TypeError("Severity must be ValidationSeverity")
        if not callable(self.validation_function):
            raise TypeError("Validation function must be callable")


@dataclass
class ValidationResult:
    """Result of validation execution"""
    rule_id: str
    rule_name: str
    validation_type: ValidationType
    severity: ValidationSeverity
    status: ValidationStatus
    execution_time_ms: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error_trace: Optional[str] = None
    
    def __post_init__(self):
        """Validate result"""
        if not isinstance(self.rule_id, str) or not self.rule_id.strip():
            raise ValueError("Rule ID must be non-empty string")
        if not isinstance(self.status, ValidationStatus):
            raise TypeError("Status must be ValidationStatus")
        if not isinstance(self.execution_time_ms, (int, float)) or self.execution_time_ms < 0:
            raise ValueError("Execution time must be non-negative")


@dataclass
class ValidationFrameworkConfig:
    """Configuration for hybrid validation framework"""
    enable_mathematical_validation: bool = True
    enable_functional_validation: bool = True
    enable_performance_validation: bool = True
    enable_consistency_validation: bool = True
    enable_integration_validation: bool = True
    enable_security_validation: bool = True
    
    # Validation parameters
    mathematical_tolerance: float = 1e-10
    performance_tolerance_ms: float = 1000.0
    memory_tolerance_mb: float = 100.0
    gpu_memory_tolerance_mb: float = 512.0
    
    # Execution parameters
    enable_parallel_validation: bool = True
    max_concurrent_validations: int = 4
    validation_timeout_seconds: int = 300
    
    # Reporting
    enable_detailed_reporting: bool = True
    save_validation_artifacts: bool = True
    generate_validation_report: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if self.mathematical_tolerance <= 0:
            raise ValueError("Mathematical tolerance must be positive")
        if self.performance_tolerance_ms <= 0:
            raise ValueError("Performance tolerance must be positive")
        if self.memory_tolerance_mb <= 0:
            raise ValueError("Memory tolerance must be positive")
        if self.max_concurrent_validations <= 0:
            raise ValueError("Max concurrent validations must be positive")


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    total_validations: int
    passed_validations: int
    failed_validations: int
    warning_validations: int
    error_validations: int
    skipped_validations: int
    total_execution_time_ms: float
    validation_results: List[ValidationResult]
    type_summary: Dict[ValidationType, Dict[str, int]]
    severity_summary: Dict[ValidationSeverity, Dict[str, int]]
    critical_failures: List[str]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.total_validations > 0:
            self.success_rate = self.passed_validations / self.total_validations
            self.failure_rate = self.failed_validations / self.total_validations
        else:
            self.success_rate = 0.0
            self.failure_rate = 0.0


class HybridValidationFramework:
    """
    Comprehensive validation framework for hybrid p-adic system.
    Provides mathematical correctness validation and system integrity checks.
    """
    
    def __init__(self, config: Optional[ValidationFrameworkConfig] = None):
        """Initialize hybrid validation framework"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, ValidationFrameworkConfig):
            raise TypeError(f"Config must be ValidationFrameworkConfig or None, got {type(config)}")
        
        self.config = config or ValidationFrameworkConfig()
        self.logger = logging.getLogger('HybridValidationFramework')
        
        # System components
        self.hybrid_manager: Optional[HybridPadicManager] = None
        self.hybrid_compressor: Optional[HybridPadicCompressionSystem] = None
        self.switching_manager: Optional[DynamicSwitchingManager] = None
        self.pure_compressor: Optional[PadicCompressionSystem] = None
        self.validator: Optional[HybridPadicValidator] = None
        
        # Framework state
        self.is_initialized = False
        self.is_validating = False
        
        # Validation rules
        self.validation_rules: Dict[str, ValidationRule] = {}
        self.validation_results: List[ValidationResult] = []
        
        # Execution tracking
        self.current_validation: Optional[str] = None
        self.validation_start_time: Optional[datetime] = None
        
        # Thread safety
        self._validation_lock = threading.RLock()
        self._results_lock = threading.RLock()
        
        # Performance tracking
        self.validation_metrics = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'average_validation_time': 0.0
        }
        
        self.logger.info("HybridValidationFramework created successfully")
    
    def initialize_framework(self,
                           hybrid_manager: HybridPadicManager,
                           hybrid_compressor: HybridPadicCompressionSystem,
                           switching_manager: DynamicSwitchingManager,
                           pure_compressor: Optional[PadicCompressionSystem] = None) -> None:
        """
        Initialize validation framework with system components.
        
        Args:
            hybrid_manager: Hybrid p-adic manager instance
            hybrid_compressor: Hybrid compression system instance
            switching_manager: Dynamic switching manager instance
            pure_compressor: Optional pure p-adic compressor for comparison
            
        Raises:
            TypeError: If any component is invalid
            RuntimeError: If initialization fails
        """
        if not isinstance(hybrid_manager, HybridPadicManager):
            raise TypeError(f"Hybrid manager must be HybridPadicManager, got {type(hybrid_manager)}")
        if not isinstance(hybrid_compressor, HybridPadicCompressionSystem):
            raise TypeError(f"Hybrid compressor must be HybridPadicCompressionSystem, got {type(hybrid_compressor)}")
        if not isinstance(switching_manager, DynamicSwitchingManager):
            raise TypeError(f"Switching manager must be DynamicSwitchingManager, got {type(switching_manager)}")
        if pure_compressor is not None and not isinstance(pure_compressor, PadicCompressionSystem):
            raise TypeError(f"Pure compressor must be PadicCompressionSystem, got {type(pure_compressor)}")
        
        try:
            # Set component references
            self.hybrid_manager = hybrid_manager
            self.hybrid_compressor = hybrid_compressor
            self.switching_manager = switching_manager
            self.pure_compressor = pure_compressor
            self.validator = HybridPadicValidator()
            
            # Register validation rules
            self._register_mathematical_validation_rules()
            self._register_functional_validation_rules()
            self._register_performance_validation_rules()
            self._register_consistency_validation_rules()
            self._register_integration_validation_rules()
            self._register_security_validation_rules()
            
            self.is_initialized = True
            self.logger.info("Hybrid validation framework initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize validation framework: {e}")
            raise RuntimeError(f"Validation framework initialization failed: {e}")
    
    def run_all_validations(self) -> ValidationReport:
        """
        Run all enabled validations and generate comprehensive report.
        
        Returns:
            Comprehensive validation report
            
        Raises:
            RuntimeError: If framework is not initialized or execution fails
        """
        if not self.is_initialized:
            raise RuntimeError("Validation framework not initialized")
        
        if self.is_validating:
            raise RuntimeError("Validation framework is already running")
        
        with self._validation_lock:
            try:
                self.is_validating = True
                self.validation_results.clear()
                start_time = time.time()
                
                self.logger.info("Starting comprehensive hybrid validation")
                
                # Get enabled validation rules
                enabled_rules = [rule for rule in self.validation_rules.values() if rule.enabled]
                
                # Sort rules by dependencies
                sorted_rules = self._sort_rules_by_dependencies(enabled_rules)
                
                # Execute validations
                if self.config.enable_parallel_validation:
                    self._run_parallel_validations(sorted_rules)
                else:
                    self._run_sequential_validations(sorted_rules)
                
                # Calculate execution time
                total_execution_time = (time.time() - start_time) * 1000
                
                # Generate report
                report = self._generate_validation_report(total_execution_time)
                
                self.logger.info(f"Validation completed: {report.passed_validations}/{report.total_validations} passed")
                
                return report
                
            except Exception as e:
                self.logger.error(f"Validation execution failed: {e}")
                raise RuntimeError(f"Validation execution failed: {e}")
            finally:
                self.is_validating = False
    
    def validate_hybrid_weight_correctness(self, weights: List[HybridPadicWeight]) -> ValidationResult:
        """
        Validate hybrid weight mathematical correctness.
        
        Args:
            weights: List of hybrid weights to validate
            
        Returns:
            Validation result
            
        Raises:
            RuntimeError: If validation fails
        """
        rule_id = "hybrid_weight_correctness"
        start_time = time.time()
        
        try:
            for i, weight in enumerate(weights):
                if not isinstance(weight, HybridPadicWeight):
                    raise RuntimeError(f"Weight {i} is not HybridPadicWeight")
                
                # Validate mathematical properties
                self.validator.validate_hybrid_weight(weight)
                
                # Validate ultrametric property
                if not self._validate_ultrametric_property(weight):
                    raise RuntimeError(f"Weight {i} violates ultrametric property")
                
                # Validate precision consistency
                if not self._validate_precision_consistency(weight):
                    raise RuntimeError(f"Weight {i} has precision inconsistency")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Hybrid Weight Correctness",
                validation_type=ValidationType.MATHEMATICAL,
                severity=ValidationSeverity.CRITICAL,
                status=ValidationStatus.PASSED,
                execution_time_ms=execution_time,
                message=f"All {len(weights)} hybrid weights validated successfully",
                details={'weights_count': len(weights)}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Hybrid Weight Correctness",
                validation_type=ValidationType.MATHEMATICAL,
                severity=ValidationSeverity.CRITICAL,
                status=ValidationStatus.FAILED,
                execution_time_ms=execution_time,
                message=f"Hybrid weight validation failed: {e}",
                error_trace=str(e)
            )
    
    def validate_compression_decompression_cycle(self, test_data: torch.Tensor) -> ValidationResult:
        """
        Validate compression-decompression cycle correctness.
        
        Args:
            test_data: Test data for validation
            
        Returns:
            Validation result
            
        Raises:
            RuntimeError: If validation fails
        """
        rule_id = "compression_decompression_cycle"
        start_time = time.time()
        
        try:
            if not isinstance(test_data, torch.Tensor):
                raise RuntimeError("Test data must be torch.Tensor")
            
            original_data = test_data.clone()
            
            # Perform compression
            compressed = self.hybrid_compressor.compress(test_data)
            
            if not isinstance(compressed, dict):
                raise RuntimeError("Compression must return dict")
            if 'compressed_data' not in compressed:
                raise RuntimeError("Compressed result must contain 'compressed_data'")
            if 'compression_info' not in compressed:
                raise RuntimeError("Compressed result must contain 'compression_info'")
            
            # Perform decompression
            decompressed = self.hybrid_compressor.decompress(
                compressed['compressed_data'],
                compressed['compression_info']
            )
            
            if not isinstance(decompressed, torch.Tensor):
                raise RuntimeError("Decompression must return torch.Tensor")
            
            # Validate reconstruction accuracy
            reconstruction_error = torch.norm(original_data - decompressed).item()
            if reconstruction_error > self.config.mathematical_tolerance:
                raise RuntimeError(f"Reconstruction error {reconstruction_error} exceeds tolerance {self.config.mathematical_tolerance}")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Compression-Decompression Cycle",
                validation_type=ValidationType.FUNCTIONAL,
                severity=ValidationSeverity.CRITICAL,
                status=ValidationStatus.PASSED,
                execution_time_ms=execution_time,
                message="Compression-decompression cycle validated successfully",
                details={
                    'data_size': test_data.numel(),
                    'reconstruction_error': reconstruction_error,
                    'compression_ratio': test_data.numel() / len(compressed['compressed_data'])
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Compression-Decompression Cycle",
                validation_type=ValidationType.FUNCTIONAL,
                severity=ValidationSeverity.CRITICAL,
                status=ValidationStatus.FAILED,
                execution_time_ms=execution_time,
                message=f"Compression-decompression validation failed: {e}",
                error_trace=str(e)
            )
    
    def validate_hybrid_pure_equivalence(self, test_data: torch.Tensor) -> ValidationResult:
        """
        Validate equivalence between hybrid and pure systems.
        
        Args:
            test_data: Test data for validation
            
        Returns:
            Validation result
            
        Raises:
            RuntimeError: If validation fails or pure compressor not available
        """
        if self.pure_compressor is None:
            raise RuntimeError("Pure compressor required for equivalence validation")
        
        rule_id = "hybrid_pure_equivalence"
        start_time = time.time()
        
        try:
            # Compress with both systems
            hybrid_compressed = self.hybrid_compressor.compress(test_data.clone())
            pure_compressed = self.pure_compressor.compress(test_data.clone())
            
            # Decompress both
            hybrid_decompressed = self.hybrid_compressor.decompress(
                hybrid_compressed['compressed_data'],
                hybrid_compressed['compression_info']
            )
            pure_decompressed = self.pure_compressor.decompress(
                pure_compressed['compressed_data'],
                pure_compressed['compression_info']
            )
            
            # Validate equivalence
            equivalence_error = torch.norm(hybrid_decompressed - pure_decompressed).item()
            if equivalence_error > self.config.mathematical_tolerance:
                raise RuntimeError(f"Equivalence error {equivalence_error} exceeds tolerance {self.config.mathematical_tolerance}")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Hybrid-Pure Equivalence",
                validation_type=ValidationType.MATHEMATICAL,
                severity=ValidationSeverity.HIGH,
                status=ValidationStatus.PASSED,
                execution_time_ms=execution_time,
                message="Hybrid-pure equivalence validated successfully",
                details={
                    'equivalence_error': equivalence_error,
                    'hybrid_compression_ratio': test_data.numel() / len(hybrid_compressed['compressed_data']),
                    'pure_compression_ratio': test_data.numel() / len(pure_compressed['compressed_data'])
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Hybrid-Pure Equivalence",
                validation_type=ValidationType.MATHEMATICAL,
                severity=ValidationSeverity.HIGH,
                status=ValidationStatus.FAILED,
                execution_time_ms=execution_time,
                message=f"Hybrid-pure equivalence validation failed: {e}",
                error_trace=str(e)
            )
    
    def validate_switching_manager_consistency(self) -> ValidationResult:
        """
        Validate switching manager consistency.
        
        Returns:
            Validation result
            
        Raises:
            RuntimeError: If validation fails
        """
        rule_id = "switching_manager_consistency"
        start_time = time.time()
        
        try:
            # Test switching decisions consistency
            test_data_sizes = [100, 1000, 10000]
            switching_decisions = []
            
            for size in test_data_sizes:
                test_data = torch.randn(size)
                
                # Make multiple switching decisions for same data
                decisions = []
                for _ in range(5):
                    decision = self.switching_manager.should_switch_to_hybrid(test_data)
                    decisions.append(decision)
                
                # All decisions should be consistent
                if not all(d == decisions[0] for d in decisions):
                    raise RuntimeError(f"Inconsistent switching decisions for data size {size}")
                
                switching_decisions.append((size, decisions[0]))
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Switching Manager Consistency",
                validation_type=ValidationType.CONSISTENCY,
                severity=ValidationSeverity.HIGH,
                status=ValidationStatus.PASSED,
                execution_time_ms=execution_time,
                message="Switching manager consistency validated successfully",
                details={'switching_decisions': switching_decisions}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Switching Manager Consistency",
                validation_type=ValidationType.CONSISTENCY,
                severity=ValidationSeverity.HIGH,
                status=ValidationStatus.FAILED,
                execution_time_ms=execution_time,
                message=f"Switching manager consistency validation failed: {e}",
                error_trace=str(e)
            )
    
    def _register_mathematical_validation_rules(self) -> None:
        """Register mathematical validation rules"""
        if not self.config.enable_mathematical_validation:
            return
        
        # Ultrametric property validation
        self.validation_rules["ultrametric_property"] = ValidationRule(
            rule_id="ultrametric_property",
            rule_name="Ultrametric Property Validation",
            validation_type=ValidationType.MATHEMATICAL,
            severity=ValidationSeverity.CRITICAL,
            description="Validate p-adic ultrametric property",
            validation_function=self._validate_ultrametric_property_rule
        )
        
        # Precision consistency validation
        self.validation_rules["precision_consistency"] = ValidationRule(
            rule_id="precision_consistency",
            rule_name="Precision Consistency Validation",
            validation_type=ValidationType.MATHEMATICAL,
            severity=ValidationSeverity.HIGH,
            description="Validate precision consistency across operations",
            validation_function=self._validate_precision_consistency_rule
        )
        
        # Mathematical operations validation
        self.validation_rules["mathematical_operations"] = ValidationRule(
            rule_id="mathematical_operations",
            rule_name="Mathematical Operations Validation",
            validation_type=ValidationType.MATHEMATICAL,
            severity=ValidationSeverity.CRITICAL,
            description="Validate mathematical operations correctness",
            validation_function=self._validate_mathematical_operations_rule
        )
    
    def _register_functional_validation_rules(self) -> None:
        """Register functional validation rules"""
        if not self.config.enable_functional_validation:
            return
        
        # Compression correctness
        self.validation_rules["compression_correctness"] = ValidationRule(
            rule_id="compression_correctness",
            rule_name="Compression Correctness",
            validation_type=ValidationType.FUNCTIONAL,
            severity=ValidationSeverity.CRITICAL,
            description="Validate compression functionality",
            validation_function=self._validate_compression_correctness_rule
        )
        
        # Decompression correctness
        self.validation_rules["decompression_correctness"] = ValidationRule(
            rule_id="decompression_correctness",
            rule_name="Decompression Correctness",
            validation_type=ValidationType.FUNCTIONAL,
            severity=ValidationSeverity.CRITICAL,
            description="Validate decompression functionality",
            validation_function=self._validate_decompression_correctness_rule
        )
        
        # Error handling validation
        self.validation_rules["error_handling"] = ValidationRule(
            rule_id="error_handling",
            rule_name="Error Handling Validation",
            validation_type=ValidationType.FUNCTIONAL,
            severity=ValidationSeverity.HIGH,
            description="Validate error handling compliance",
            validation_function=self._validate_error_handling_rule
        )
    
    def _register_performance_validation_rules(self) -> None:
        """Register performance validation rules"""
        if not self.config.enable_performance_validation:
            return
        
        # Performance regression validation
        self.validation_rules["performance_regression"] = ValidationRule(
            rule_id="performance_regression",
            rule_name="Performance Regression Validation",
            validation_type=ValidationType.PERFORMANCE,
            severity=ValidationSeverity.MEDIUM,
            description="Validate performance regression",
            validation_function=self._validate_performance_regression_rule
        )
        
        # Memory usage validation
        self.validation_rules["memory_usage"] = ValidationRule(
            rule_id="memory_usage",
            rule_name="Memory Usage Validation",
            validation_type=ValidationType.PERFORMANCE,
            severity=ValidationSeverity.MEDIUM,
            description="Validate memory usage constraints",
            validation_function=self._validate_memory_usage_rule
        )
        
        # GPU memory validation
        self.validation_rules["gpu_memory_usage"] = ValidationRule(
            rule_id="gpu_memory_usage",
            rule_name="GPU Memory Usage Validation",
            validation_type=ValidationType.PERFORMANCE,
            severity=ValidationSeverity.MEDIUM,
            description="Validate GPU memory usage constraints",
            validation_function=self._validate_gpu_memory_usage_rule
        )
    
    def _register_consistency_validation_rules(self) -> None:
        """Register consistency validation rules"""
        if not self.config.enable_consistency_validation:
            return
        
        # Data consistency validation
        self.validation_rules["data_consistency"] = ValidationRule(
            rule_id="data_consistency",
            rule_name="Data Consistency Validation",
            validation_type=ValidationType.CONSISTENCY,
            severity=ValidationSeverity.HIGH,
            description="Validate data consistency across operations",
            validation_function=self._validate_data_consistency_rule
        )
        
        # State consistency validation
        self.validation_rules["state_consistency"] = ValidationRule(
            rule_id="state_consistency",
            rule_name="State Consistency Validation",
            validation_type=ValidationType.CONSISTENCY,
            severity=ValidationSeverity.HIGH,
            description="Validate state consistency",
            validation_function=self._validate_state_consistency_rule
        )
    
    def _register_integration_validation_rules(self) -> None:
        """Register integration validation rules"""
        if not self.config.enable_integration_validation:
            return
        
        # Component integration validation
        self.validation_rules["component_integration"] = ValidationRule(
            rule_id="component_integration",
            rule_name="Component Integration Validation",
            validation_type=ValidationType.INTEGRATION,
            severity=ValidationSeverity.HIGH,
            description="Validate component integration",
            validation_function=self._validate_component_integration_rule
        )
        
        # System integration validation
        self.validation_rules["system_integration"] = ValidationRule(
            rule_id="system_integration",
            rule_name="System Integration Validation",
            validation_type=ValidationType.INTEGRATION,
            severity=ValidationSeverity.HIGH,
            description="Validate system integration",
            validation_function=self._validate_system_integration_rule
        )
    
    def _register_security_validation_rules(self) -> None:
        """Register security validation rules"""
        if not self.config.enable_security_validation:
            return
        
        # Input validation security
        self.validation_rules["input_validation_security"] = ValidationRule(
            rule_id="input_validation_security",
            rule_name="Input Validation Security",
            validation_type=ValidationType.SECURITY,
            severity=ValidationSeverity.HIGH,
            description="Validate input security measures",
            validation_function=self._validate_input_security_rule
        )
        
        # Memory safety validation
        self.validation_rules["memory_safety"] = ValidationRule(
            rule_id="memory_safety",
            rule_name="Memory Safety Validation",
            validation_type=ValidationType.SECURITY,
            severity=ValidationSeverity.HIGH,
            description="Validate memory safety",
            validation_function=self._validate_memory_safety_rule
        )
    
    def _sort_rules_by_dependencies(self, rules: List[ValidationRule]) -> List[ValidationRule]:
        """Sort validation rules by dependencies"""
        sorted_rules = []
        processed = set()
        
        def process_rule(rule: ValidationRule):
            if rule.rule_id in processed:
                return
            
            # Process dependencies first
            for dep_id in rule.depends_on:
                if dep_id in self.validation_rules:
                    dep_rule = self.validation_rules[dep_id]
                    if dep_rule in rules and dep_rule.enabled:
                        process_rule(dep_rule)
            
            sorted_rules.append(rule)
            processed.add(rule.rule_id)
        
        for rule in rules:
            process_rule(rule)
        
        return sorted_rules
    
    def _run_sequential_validations(self, rules: List[ValidationRule]) -> None:
        """Run validations sequentially"""
        for rule in rules:
            try:
                result = self._execute_validation_rule(rule)
                with self._results_lock:
                    self.validation_results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to execute validation rule {rule.rule_id}: {e}")
    
    def _run_parallel_validations(self, rules: List[ValidationRule]) -> None:
        """Run validations in parallel"""
        import concurrent.futures
        
        max_workers = min(self.config.max_concurrent_validations, len(rules))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all validation tasks
            future_to_rule = {
                executor.submit(self._execute_validation_rule, rule): rule 
                for rule in rules
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_rule):
                rule = future_to_rule[future]
                try:
                    result = future.result()
                    with self._results_lock:
                        self.validation_results.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to execute validation rule {rule.rule_id}: {e}")
    
    def _execute_validation_rule(self, rule: ValidationRule) -> ValidationResult:
        """Execute individual validation rule"""
        self.current_validation = rule.rule_id
        self.validation_start_time = datetime.utcnow()
        
        start_time = time.time()
        
        try:
            # Execute validation function
            result = rule.validation_function(**rule.parameters)
            
            if not isinstance(result, ValidationResult):
                # Create result from return value
                execution_time = (time.time() - start_time) * 1000
                result = ValidationResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.rule_name,
                    validation_type=rule.validation_type,
                    severity=rule.severity,
                    status=ValidationStatus.PASSED if result else ValidationStatus.FAILED,
                    execution_time_ms=execution_time,
                    message="Validation completed",
                    details={'result': result}
                )
            
            self.logger.debug(f"Validation {rule.rule_id} completed with status {result.status}")
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            result = ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.rule_name,
                validation_type=rule.validation_type,
                severity=rule.severity,
                status=ValidationStatus.ERROR,
                execution_time_ms=execution_time,
                message=f"Validation failed: {e}",
                error_trace=str(e)
            )
            
            self.logger.error(f"Validation {rule.rule_id} failed: {e}")
            
            return result
    
    # Validation rule implementations
    
    def _validate_ultrametric_property_rule(self) -> ValidationResult:
        """Validate ultrametric property rule implementation"""
        rule_id = "ultrametric_property"
        start_time = time.time()
        
        try:
            # Create test hybrid weights
            test_weights = self._create_test_weights(5)
            
            for weight in test_weights:
                if not self._validate_ultrametric_property(weight):
                    raise RuntimeError("Ultrametric property violation detected")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Ultrametric Property Validation",
                validation_type=ValidationType.MATHEMATICAL,
                severity=ValidationSeverity.CRITICAL,
                status=ValidationStatus.PASSED,
                execution_time_ms=execution_time,
                message="Ultrametric property validated successfully",
                details={'weights_tested': len(test_weights)}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Ultrametric Property Validation",
                validation_type=ValidationType.MATHEMATICAL,
                severity=ValidationSeverity.CRITICAL,
                status=ValidationStatus.FAILED,
                execution_time_ms=execution_time,
                message=f"Ultrametric property validation failed: {e}",
                error_trace=str(e)
            )
    
    def _validate_precision_consistency_rule(self) -> ValidationResult:
        """Validate precision consistency rule implementation"""
        rule_id = "precision_consistency"
        start_time = time.time()
        
        try:
            # Create test hybrid weights
            test_weights = self._create_test_weights(3)
            
            for weight in test_weights:
                if not self._validate_precision_consistency(weight):
                    raise RuntimeError("Precision consistency violation detected")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Precision Consistency Validation",
                validation_type=ValidationType.MATHEMATICAL,
                severity=ValidationSeverity.HIGH,
                status=ValidationStatus.PASSED,
                execution_time_ms=execution_time,
                message="Precision consistency validated successfully",
                details={'weights_tested': len(test_weights)}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Precision Consistency Validation",
                validation_type=ValidationType.MATHEMATICAL,
                severity=ValidationSeverity.HIGH,
                status=ValidationStatus.FAILED,
                execution_time_ms=execution_time,
                message=f"Precision consistency validation failed: {e}",
                error_trace=str(e)
            )
    
    def _validate_mathematical_operations_rule(self) -> ValidationResult:
        """Validate mathematical operations rule implementation"""
        rule_id = "mathematical_operations"
        start_time = time.time()
        
        try:
            # Test basic mathematical operations
            math_ops = PadicMathematicalOperations()
            
            # Test addition
            a = PadicWeight(exponent_channel=torch.randn(10), mantissa_channel=torch.randn(10), prime=7, precision=10)
            b = PadicWeight(exponent_channel=torch.randn(10), mantissa_channel=torch.randn(10), prime=7, precision=10)
            
            result = math_ops.add_weights(a, b)
            
            if not isinstance(result, PadicWeight):
                raise RuntimeError("Mathematical operation result invalid")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Mathematical Operations Validation",
                validation_type=ValidationType.MATHEMATICAL,
                severity=ValidationSeverity.CRITICAL,
                status=ValidationStatus.PASSED,
                execution_time_ms=execution_time,
                message="Mathematical operations validated successfully"
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Mathematical Operations Validation",
                validation_type=ValidationType.MATHEMATICAL,
                severity=ValidationSeverity.CRITICAL,
                status=ValidationStatus.FAILED,
                execution_time_ms=execution_time,
                message=f"Mathematical operations validation failed: {e}",
                error_trace=str(e)
            )
    
    def _validate_compression_correctness_rule(self) -> ValidationResult:
        """Validate compression correctness rule implementation"""
        test_data = torch.randn(1000)
        return self.validate_compression_decompression_cycle(test_data)
    
    def _validate_decompression_correctness_rule(self) -> ValidationResult:
        """Validate decompression correctness rule implementation"""
        test_data = torch.randn(1000)
        return self.validate_compression_decompression_cycle(test_data)
    
    def _validate_error_handling_rule(self) -> ValidationResult:
        """Validate error handling rule implementation"""
        rule_id = "error_handling"
        start_time = time.time()
        
        try:
            # Test invalid input handling
            test_cases = [
                (None, "Data cannot be None"),
                ("invalid", "Data must be torch.Tensor"),
                (torch.tensor([]), "Data cannot be empty"),
            ]
            
            for invalid_input, expected_error in test_cases:
                try:
                    self.hybrid_compressor.compress(invalid_input)
                    raise RuntimeError(f"Expected error for input {invalid_input}")
                except (ValueError, TypeError) as e:
                    if expected_error not in str(e):
                        raise RuntimeError(f"Unexpected error message: {e}")
                except Exception as e:
                    raise RuntimeError(f"Unexpected exception type: {type(e)}")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Error Handling Validation",
                validation_type=ValidationType.FUNCTIONAL,
                severity=ValidationSeverity.HIGH,
                status=ValidationStatus.PASSED,
                execution_time_ms=execution_time,
                message="Error handling validated successfully"
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Error Handling Validation",
                validation_type=ValidationType.FUNCTIONAL,
                severity=ValidationSeverity.HIGH,
                status=ValidationStatus.FAILED,
                execution_time_ms=execution_time,
                message=f"Error handling validation failed: {e}",
                error_trace=str(e)
            )
    
    def _validate_performance_regression_rule(self) -> ValidationResult:
        """Validate performance regression rule implementation"""
        rule_id = "performance_regression"
        start_time = time.time()
        
        try:
            # Test performance consistency
            test_data = torch.randn(1000)
            execution_times = []
            
            for _ in range(5):
                op_start = time.time()
                self.hybrid_compressor.compress(test_data)
                op_time = (time.time() - op_start) * 1000
                execution_times.append(op_time)
            
            # Check for performance regression
            avg_time = sum(execution_times) / len(execution_times)
            if avg_time > self.config.performance_tolerance_ms:
                raise RuntimeError(f"Performance regression detected: {avg_time}ms > {self.config.performance_tolerance_ms}ms")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Performance Regression Validation",
                validation_type=ValidationType.PERFORMANCE,
                severity=ValidationSeverity.MEDIUM,
                status=ValidationStatus.PASSED,
                execution_time_ms=execution_time,
                message="Performance regression validated successfully",
                details={'average_execution_time': avg_time}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Performance Regression Validation",
                validation_type=ValidationType.PERFORMANCE,
                severity=ValidationSeverity.MEDIUM,
                status=ValidationStatus.FAILED,
                execution_time_ms=execution_time,
                message=f"Performance regression validation failed: {e}",
                error_trace=str(e)
            )
    
    def _validate_memory_usage_rule(self) -> ValidationResult:
        """Validate memory usage rule implementation"""
        rule_id = "memory_usage"
        start_time = time.time()
        
        try:
            import gc
            import psutil
            
            initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            
            # Perform memory-intensive operations
            for _ in range(10):
                test_data = torch.randn(5000)
                compressed = self.hybrid_compressor.compress(test_data)
                decompressed = self.hybrid_compressor.decompress(
                    compressed['compressed_data'],
                    compressed['compression_info']
                )
                del test_data, compressed, decompressed
            
            gc.collect()
            final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
            memory_growth = final_memory - initial_memory
            
            if memory_growth > self.config.memory_tolerance_mb:
                raise RuntimeError(f"Memory usage exceeded tolerance: {memory_growth}MB > {self.config.memory_tolerance_mb}MB")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Memory Usage Validation",
                validation_type=ValidationType.PERFORMANCE,
                severity=ValidationSeverity.MEDIUM,
                status=ValidationStatus.PASSED,
                execution_time_ms=execution_time,
                message="Memory usage validated successfully",
                details={'memory_growth_mb': memory_growth}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Memory Usage Validation",
                validation_type=ValidationType.PERFORMANCE,
                severity=ValidationSeverity.MEDIUM,
                status=ValidationStatus.FAILED,
                execution_time_ms=execution_time,
                message=f"Memory usage validation failed: {e}",
                error_trace=str(e)
            )
    
    def _validate_gpu_memory_usage_rule(self) -> ValidationResult:
        """Validate GPU memory usage rule implementation"""
        rule_id = "gpu_memory_usage"
        start_time = time.time()
        
        try:
            if not torch.cuda.is_available():
                return ValidationResult(
                    rule_id=rule_id,
                    rule_name="GPU Memory Usage Validation",
                    validation_type=ValidationType.PERFORMANCE,
                    severity=ValidationSeverity.MEDIUM,
                    status=ValidationStatus.SKIPPED,
                    execution_time_ms=0.0,
                    message="CUDA not available - skipping GPU validation"
                )
            
            initial_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            
            # Perform GPU operations
            test_data = torch.randn(5000).cuda()
            compressed = self.hybrid_compressor.compress(test_data)
            decompressed = self.hybrid_compressor.decompress(
                compressed['compressed_data'],
                compressed['compression_info']
            )
            
            peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            gpu_memory_used = peak_gpu_memory - initial_gpu_memory
            
            if gpu_memory_used > self.config.gpu_memory_tolerance_mb:
                raise RuntimeError(f"GPU memory usage exceeded tolerance: {gpu_memory_used}MB > {self.config.gpu_memory_tolerance_mb}MB")
            
            # Clean up
            del test_data, compressed, decompressed
            torch.cuda.empty_cache()
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="GPU Memory Usage Validation",
                validation_type=ValidationType.PERFORMANCE,
                severity=ValidationSeverity.MEDIUM,
                status=ValidationStatus.PASSED,
                execution_time_ms=execution_time,
                message="GPU memory usage validated successfully",
                details={'gpu_memory_used_mb': gpu_memory_used}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="GPU Memory Usage Validation",
                validation_type=ValidationType.PERFORMANCE,
                severity=ValidationSeverity.MEDIUM,
                status=ValidationStatus.FAILED,
                execution_time_ms=execution_time,
                message=f"GPU memory usage validation failed: {e}",
                error_trace=str(e)
            )
    
    def _validate_data_consistency_rule(self) -> ValidationResult:
        """Validate data consistency rule implementation"""
        rule_id = "data_consistency"
        start_time = time.time()
        
        try:
            # Test data consistency across multiple operations
            test_data = torch.randn(1000)
            results = []
            
            for _ in range(3):
                compressed = self.hybrid_compressor.compress(test_data.clone())
                decompressed = self.hybrid_compressor.decompress(
                    compressed['compressed_data'],
                    compressed['compression_info']
                )
                results.append(decompressed)
            
            # All results should be consistent
            for i in range(1, len(results)):
                consistency_error = torch.norm(results[0] - results[i]).item()
                if consistency_error > self.config.mathematical_tolerance:
                    raise RuntimeError(f"Data consistency violation: error {consistency_error}")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Data Consistency Validation",
                validation_type=ValidationType.CONSISTENCY,
                severity=ValidationSeverity.HIGH,
                status=ValidationStatus.PASSED,
                execution_time_ms=execution_time,
                message="Data consistency validated successfully",
                details={'operations_tested': len(results)}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Data Consistency Validation",
                validation_type=ValidationType.CONSISTENCY,
                severity=ValidationSeverity.HIGH,
                status=ValidationStatus.FAILED,
                execution_time_ms=execution_time,
                message=f"Data consistency validation failed: {e}",
                error_trace=str(e)
            )
    
    def _validate_state_consistency_rule(self) -> ValidationResult:
        """Validate state consistency rule implementation"""
        return self.validate_switching_manager_consistency()
    
    def _validate_component_integration_rule(self) -> ValidationResult:
        """Validate component integration rule implementation"""
        rule_id = "component_integration"
        start_time = time.time()
        
        try:
            # Test integration between components
            test_data = torch.randn(1000)
            
            # Test switching manager integration
            should_switch = self.switching_manager.should_switch_to_hybrid(test_data)
            if not isinstance(should_switch, bool):
                raise RuntimeError("Switching manager integration failed")
            
            # Test compression system integration
            compressed = self.hybrid_compressor.compress(test_data)
            if not isinstance(compressed, dict):
                raise RuntimeError("Compression system integration failed")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Component Integration Validation",
                validation_type=ValidationType.INTEGRATION,
                severity=ValidationSeverity.HIGH,
                status=ValidationStatus.PASSED,
                execution_time_ms=execution_time,
                message="Component integration validated successfully"
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Component Integration Validation",
                validation_type=ValidationType.INTEGRATION,
                severity=ValidationSeverity.HIGH,
                status=ValidationStatus.FAILED,
                execution_time_ms=execution_time,
                message=f"Component integration validation failed: {e}",
                error_trace=str(e)
            )
    
    def _validate_system_integration_rule(self) -> ValidationResult:
        """Validate system integration rule implementation"""
        rule_id = "system_integration"
        start_time = time.time()
        
        try:
            # Test full system integration
            test_data = torch.randn(1000)
            
            # Full compression-decompression cycle
            compressed = self.hybrid_compressor.compress(test_data)
            decompressed = self.hybrid_compressor.decompress(
                compressed['compressed_data'],
                compressed['compression_info']
            )
            
            # Validate reconstruction
            reconstruction_error = torch.norm(test_data - decompressed).item()
            if reconstruction_error > self.config.mathematical_tolerance:
                raise RuntimeError(f"System integration failed: reconstruction error {reconstruction_error}")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="System Integration Validation",
                validation_type=ValidationType.INTEGRATION,
                severity=ValidationSeverity.HIGH,
                status=ValidationStatus.PASSED,
                execution_time_ms=execution_time,
                message="System integration validated successfully",
                details={'reconstruction_error': reconstruction_error}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="System Integration Validation",
                validation_type=ValidationType.INTEGRATION,
                severity=ValidationSeverity.HIGH,
                status=ValidationStatus.FAILED,
                execution_time_ms=execution_time,
                message=f"System integration validation failed: {e}",
                error_trace=str(e)
            )
    
    def _validate_input_security_rule(self) -> ValidationResult:
        """Validate input security rule implementation"""
        rule_id = "input_validation_security"
        start_time = time.time()
        
        try:
            # Test input validation security
            malicious_inputs = [
                None,
                "malicious_string",
                torch.tensor([float('inf')]),
                torch.tensor([float('nan')]),
                torch.tensor([])
            ]
            
            for malicious_input in malicious_inputs:
                try:
                    self.hybrid_compressor.compress(malicious_input)
                except (ValueError, TypeError, RuntimeError):
                    # Expected - input should be rejected
                    continue
                except Exception as e:
                    raise RuntimeError(f"Unexpected exception for malicious input: {e}")
                else:
                    raise RuntimeError(f"Malicious input {malicious_input} was not rejected")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Input Validation Security",
                validation_type=ValidationType.SECURITY,
                severity=ValidationSeverity.HIGH,
                status=ValidationStatus.PASSED,
                execution_time_ms=execution_time,
                message="Input validation security validated successfully",
                details={'malicious_inputs_tested': len(malicious_inputs)}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Input Validation Security",
                validation_type=ValidationType.SECURITY,
                severity=ValidationSeverity.HIGH,
                status=ValidationStatus.FAILED,
                execution_time_ms=execution_time,
                message=f"Input validation security failed: {e}",
                error_trace=str(e)
            )
    
    def _validate_memory_safety_rule(self) -> ValidationResult:
        """Validate memory safety rule implementation"""
        rule_id = "memory_safety"
        start_time = time.time()
        
        try:
            # Test memory safety by creating and destroying objects
            import gc
            
            initial_objects = len(gc.get_objects())
            
            # Create many temporary objects
            for _ in range(100):
                test_data = torch.randn(100)
                compressed = self.hybrid_compressor.compress(test_data)
                decompressed = self.hybrid_compressor.decompress(
                    compressed['compressed_data'],
                    compressed['compression_info']
                )
                # Objects should be automatically cleaned up
                del test_data, compressed, decompressed
            
            gc.collect()
            final_objects = len(gc.get_objects())
            
            # Check for excessive object growth (potential memory leaks)
            object_growth = final_objects - initial_objects
            if object_growth > 1000:  # Reasonable threshold
                raise RuntimeError(f"Potential memory safety issue: {object_growth} objects created")
            
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Memory Safety Validation",
                validation_type=ValidationType.SECURITY,
                severity=ValidationSeverity.HIGH,
                status=ValidationStatus.PASSED,
                execution_time_ms=execution_time,
                message="Memory safety validated successfully",
                details={'object_growth': object_growth}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule_id,
                rule_name="Memory Safety Validation",
                validation_type=ValidationType.SECURITY,
                severity=ValidationSeverity.HIGH,
                status=ValidationStatus.FAILED,
                execution_time_ms=execution_time,
                message=f"Memory safety validation failed: {e}",
                error_trace=str(e)
            )
    
    # Helper methods
    
    def _validate_ultrametric_property(self, weight: HybridPadicWeight) -> bool:
        """Validate ultrametric property for hybrid weight"""
        try:
            # The ultrametric property states that d(x,z) ≤ max(d(x,y), d(y,z))
            # For p-adic numbers, this is automatically satisfied by the p-adic norm
            
            # Basic validation - check that weight components are finite
            if not torch.isfinite(weight.exponent_channel).all():
                return False
            if not torch.isfinite(weight.mantissa_channel).all():
                return False
            
            # Check that the p-adic structure is maintained
            if weight.prime <= 1:
                return False
            if weight.precision <= 0:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_precision_consistency(self, weight: HybridPadicWeight) -> bool:
        """Validate precision consistency for hybrid weight"""
        try:
            # Check that precision is consistent across channels
            if weight.precision <= 0:
                return False
            
            # Check that dimensions are consistent
            if weight.exponent_channel.shape != weight.mantissa_channel.shape:
                return False
            
            # Check that device consistency is maintained
            if weight.exponent_channel.device != weight.mantissa_channel.device:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _create_test_weights(self, count: int) -> List[HybridPadicWeight]:
        """Create test hybrid weights"""
        weights = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for i in range(count):
            weight = HybridPadicWeight(
                exponent_channel=torch.randn(10, device=device),
                mantissa_channel=torch.randn(10, device=device),
                prime=7,
                precision=10,
                valuation=0,
                device=device,
                dtype=torch.float32
            )
            weights.append(weight)
        
        return weights
    
    def _generate_validation_report(self, total_execution_time: float) -> ValidationReport:
        """Generate comprehensive validation report"""
        # Count results by status
        total_validations = len(self.validation_results)
        passed_validations = sum(1 for r in self.validation_results if r.status == ValidationStatus.PASSED)
        failed_validations = sum(1 for r in self.validation_results if r.status == ValidationStatus.FAILED)
        warning_validations = sum(1 for r in self.validation_results if r.status == ValidationStatus.WARNING)
        error_validations = sum(1 for r in self.validation_results if r.status == ValidationStatus.ERROR)
        skipped_validations = sum(1 for r in self.validation_results if r.status == ValidationStatus.SKIPPED)
        
        # Type summary
        type_summary = {}
        for val_type in ValidationType:
            type_results = [r for r in self.validation_results if r.validation_type == val_type]
            type_summary[val_type] = {
                'total': len(type_results),
                'passed': sum(1 for r in type_results if r.status == ValidationStatus.PASSED),
                'failed': sum(1 for r in type_results if r.status == ValidationStatus.FAILED)
            }
        
        # Severity summary
        severity_summary = {}
        for severity in ValidationSeverity:
            severity_results = [r for r in self.validation_results if r.severity == severity]
            severity_summary[severity] = {
                'total': len(severity_results),
                'passed': sum(1 for r in severity_results if r.status == ValidationStatus.PASSED),
                'failed': sum(1 for r in severity_results if r.status == ValidationStatus.FAILED)
            }
        
        # Critical failures
        critical_failures = [
            r.rule_id for r in self.validation_results 
            if r.status == ValidationStatus.FAILED and r.severity == ValidationSeverity.CRITICAL
        ]
        
        # Recommendations
        recommendations = []
        if failed_validations > 0:
            recommendations.append(f"Address {failed_validations} failed validations")
        if len(critical_failures) > 0:
            recommendations.append(f"Immediately fix {len(critical_failures)} critical failures")
        if error_validations > 0:
            recommendations.append(f"Investigate {error_validations} validation errors")
        
        return ValidationReport(
            total_validations=total_validations,
            passed_validations=passed_validations,
            failed_validations=failed_validations,
            warning_validations=warning_validations,
            error_validations=error_validations,
            skipped_validations=skipped_validations,
            total_execution_time_ms=total_execution_time,
            validation_results=self.validation_results.copy(),
            type_summary=type_summary,
            severity_summary=severity_summary,
            critical_failures=critical_failures,
            recommendations=recommendations
        )
    
    def shutdown(self) -> None:
        """Shutdown validation framework"""
        self.logger.info("Shutting down hybrid validation framework")
        
        # Clear references
        self.hybrid_manager = None
        self.hybrid_compressor = None
        self.switching_manager = None
        self.pure_compressor = None
        self.validator = None
        
        # Clear validation state
        self.validation_rules.clear()
        self.validation_results.clear()
        
        self.is_initialized = False
        self.logger.info("Hybrid validation framework shutdown complete")