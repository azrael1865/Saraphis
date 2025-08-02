"""
Brain Specialized Orchestrators Module
Provides comprehensive orchestration capabilities for the Saraphis Brain system
"""

from .brain_orchestrator import (
    BrainOrchestrator,
    OrchestrationTask,
    SystemMetrics,
    ComponentStatus,
    OrchestrationMode,
    SystemState,
    OperationPriority
)

from .brain_decision_engine import (
    BrainDecisionEngine,
    DecisionType,
    DecisionStrategy,
    DecisionConfidence,
    CriteriaType,
    DecisionCriteria,
    DecisionAlternative,
    DecisionContext,
    DecisionResult,
    DecisionMethod,
    WeightedSumMethod,
    TOPSISMethod,
    AHPMethod
)

from .reasoning_orchestrator import (
    ReasoningOrchestrator,
    ReasoningType,
    InferenceRule,
    ReasoningStrategy,
    ConfidenceLevel,
    LogicalStatement,
    InferenceStep,
    ReasoningChain,
    ProofAttempt,
    ReasoningContext,
    ReasoningEngine,
    DeductiveReasoningEngine,
    InductiveReasoningEngine,
    AbductiveReasoningEngine
)

from .neural_orchestrator import (
    NeuralOrchestrator,
    NetworkType,
    TrainingPhase,
    OptimizationStrategy,
    SchedulingStrategy,
    NetworkConfig,
    TrainingConfig,
    ModelState,
    NeuralTask,
    NeuralModel,
    TrainingCoordinator,
    InferenceEngine
)

from .uncertainty_orchestrator import (
    UncertaintyOrchestrator,
    UncertaintyType,
    QuantificationMethod,
    ConfidenceLevel as UncertaintyConfidenceLevel,
    UncertaintyEstimationStrategy,
    UncertaintyEstimate,
    BayesianInference,
    UncertaintyPropagation,
    CalibrationResults,
    CredalSet,
    CategoricalUncertaintyMetrics,
    UncertaintyQuantifier,
    BayesianQuantifier,
    ConformalizedCredalQuantifier,
    DeepDeterministicQuantifier,
    BatchEnsembleQuantifier,
    MonteCarloDropoutQuantifier,
    EnsembleQuantifier,
    BootstrapQuantifier
)

from .domain_orchestrator import (
    DomainOrchestrator,
    DomainType,
    ExpertiseLevel,
    OperationMode,
    ProcessingStrategy,
    DomainExpertise,
    DomainOperation,
    DomainKnowledge,
    DomainExpert,
    MathematicsExpert,
    SportsExpert,
    GeneralExpert
)

__all__ = [
    # Brain Orchestrator
    'BrainOrchestrator',
    'OrchestrationTask',
    'SystemMetrics',
    'ComponentStatus',
    'OrchestrationMode',
    'SystemState',
    'OperationPriority',
    
    # Decision Engine
    'BrainDecisionEngine',
    'DecisionType',
    'DecisionStrategy',
    'DecisionConfidence',
    'CriteriaType',
    'DecisionCriteria',
    'DecisionAlternative',
    'DecisionContext',
    'DecisionResult',
    'DecisionMethod',
    'WeightedSumMethod',
    'TOPSISMethod',
    'AHPMethod',
    
    # Reasoning Orchestrator
    'ReasoningOrchestrator',
    'ReasoningType',
    'InferenceRule',
    'ReasoningStrategy',
    'ConfidenceLevel',
    'LogicalStatement',
    'InferenceStep',
    'ReasoningChain',
    'ProofAttempt',
    'ReasoningContext',
    'ReasoningEngine',
    'DeductiveReasoningEngine',
    'InductiveReasoningEngine',
    'AbductiveReasoningEngine',
    
    # Neural Orchestrator
    'NeuralOrchestrator',
    'NetworkType',
    'TrainingPhase',
    'OptimizationStrategy',
    'SchedulingStrategy',
    'NetworkConfig',
    'TrainingConfig',
    'ModelState',
    'NeuralTask',
    'NeuralModel',
    'TrainingCoordinator',
    'InferenceEngine',
    
    # Uncertainty Orchestrator
    'UncertaintyOrchestrator',
    'UncertaintyType',
    'QuantificationMethod',
    'UncertaintyConfidenceLevel',
    'UncertaintyEstimationStrategy',
    'UncertaintyEstimate',
    'BayesianInference',
    'UncertaintyPropagation',
    'CalibrationResults',
    'CredalSet',
    'CategoricalUncertaintyMetrics',
    'UncertaintyQuantifier',
    'BayesianQuantifier',
    'ConformalizedCredalQuantifier',
    'DeepDeterministicQuantifier',
    'BatchEnsembleQuantifier',
    'MonteCarloDropoutQuantifier',
    'EnsembleQuantifier',
    'BootstrapQuantifier',
    
    # Domain Orchestrator
    'DomainOrchestrator',
    'DomainType',
    'ExpertiseLevel',
    'OperationMode',
    'ProcessingStrategy',
    'DomainExpertise',
    'DomainOperation',
    'DomainKnowledge',
    'DomainExpert',
    'MathematicsExpert',
    'SportsExpert',
    'GeneralExpert'
]

# Module version and metadata
__version__ = "1.0.0"
__author__ = "Saraphis Brain System"
__description__ = "Specialized orchestrators for advanced brain system operations"

# Orchestrator registry for dynamic access
ORCHESTRATOR_REGISTRY = {
    'brain': BrainOrchestrator,
    'decision': BrainDecisionEngine,
    'reasoning': ReasoningOrchestrator,
    'neural': NeuralOrchestrator,
    'uncertainty': UncertaintyOrchestrator,
    'domain': DomainOrchestrator
}

def get_orchestrator(orchestrator_type: str, *args, **kwargs):
    """
    Factory function to create orchestrator instances
    
    Args:
        orchestrator_type: Type of orchestrator ('brain', 'decision', 'reasoning', 'neural', 'uncertainty', 'domain')
        *args, **kwargs: Arguments to pass to orchestrator constructor
    
    Returns:
        Orchestrator instance
    
    Raises:
        ValueError: If orchestrator_type is not recognized
    """
    if orchestrator_type not in ORCHESTRATOR_REGISTRY:
        available_types = ', '.join(ORCHESTRATOR_REGISTRY.keys())
        raise ValueError(f"Unknown orchestrator type: {orchestrator_type}. Available types: {available_types}")
    
    orchestrator_class = ORCHESTRATOR_REGISTRY[orchestrator_type]
    return orchestrator_class(*args, **kwargs)

def list_orchestrators():
    """
    Get list of available orchestrator types
    
    Returns:
        List of available orchestrator type names
    """
    return list(ORCHESTRATOR_REGISTRY.keys())

def get_orchestrator_info(orchestrator_type: str = None):
    """
    Get information about orchestrators
    
    Args:
        orchestrator_type: Specific orchestrator type to get info for (optional)
    
    Returns:
        Dictionary containing orchestrator information
    """
    if orchestrator_type:
        if orchestrator_type not in ORCHESTRATOR_REGISTRY:
            return {"error": f"Unknown orchestrator type: {orchestrator_type}"}
        
        orchestrator_class = ORCHESTRATOR_REGISTRY[orchestrator_type]
        return {
            "type": orchestrator_type,
            "class": orchestrator_class.__name__,
            "module": orchestrator_class.__module__,
            "docstring": orchestrator_class.__doc__
        }
    else:
        # Return info for all orchestrators
        info = {}
        for orc_type, orc_class in ORCHESTRATOR_REGISTRY.items():
            info[orc_type] = {
                "class": orc_class.__name__,
                "module": orc_class.__module__,
                "docstring": orc_class.__doc__
            }
        return info

# Orchestrator capabilities mapping
ORCHESTRATOR_CAPABILITIES = {
    'brain': [
        'system_coordination',
        'task_orchestration',
        'resource_management',
        'component_monitoring',
        'emergency_handling',
        'performance_tracking'
    ],
    'decision': [
        'multi_criteria_analysis',
        'decision_support',
        'alternative_evaluation',
        'consensus_building',
        'sensitivity_analysis',
        'confidence_assessment'
    ],
    'reasoning': [
        'logical_inference',
        'proof_construction',
        'argument_analysis',
        'reasoning_chains',
        'theorem_proving',
        'consistency_checking'
    ],
    'neural': [
        'model_training',
        'inference_coordination',
        'resource_optimization',
        'model_evaluation',
        'neural_architecture_management',
        'training_coordination'
    ],
    'uncertainty': [
        'uncertainty_quantification',
        'confidence_estimation',
        'calibration_analysis',
        'uncertainty_propagation',
        'bayesian_inference',
        'risk_assessment'
    ],
    'domain': [
        'domain_expertise_routing',
        'specialized_processing',
        'knowledge_management',
        'expert_coordination',
        'domain_specific_analysis',
        'cross_domain_integration'
    ]
}

def get_orchestrator_capabilities(orchestrator_type: str = None):
    """
    Get capabilities of orchestrators
    
    Args:
        orchestrator_type: Specific orchestrator type (optional)
    
    Returns:
        Dictionary or list of capabilities
    """
    if orchestrator_type:
        return ORCHESTRATOR_CAPABILITIES.get(orchestrator_type, [])
    else:
        return ORCHESTRATOR_CAPABILITIES

# Integration helpers
def create_orchestrator_suite(brain_instance=None, config=None):
    """
    Create a complete suite of orchestrators
    
    Args:
        brain_instance: Brain instance to pass to orchestrators
        config: Configuration dictionary
    
    Returns:
        Dictionary containing all orchestrator instances
    """
    config = config or {}
    
    suite = {}
    for orc_type in ORCHESTRATOR_REGISTRY:
        try:
            orchestrator_config = config.get(f'{orc_type}_config', {})
            suite[orc_type] = get_orchestrator(orc_type, brain_instance, orchestrator_config)
        except Exception as e:
            print(f"Warning: Failed to create {orc_type} orchestrator: {e}")
            suite[orc_type] = None
    
    return suite

def validate_orchestrator_integration(orchestrator_suite):
    """
    Validate that orchestrators are properly integrated
    
    Args:
        orchestrator_suite: Dictionary of orchestrator instances
    
    Returns:
        Validation report
    """
    report = {
        "status": "valid",
        "issues": [],
        "recommendations": []
    }
    
    # Check that all orchestrators are present
    missing_orchestrators = []
    for orc_type in ORCHESTRATOR_REGISTRY:
        if orc_type not in orchestrator_suite or orchestrator_suite[orc_type] is None:
            missing_orchestrators.append(orc_type)
    
    if missing_orchestrators:
        report["issues"].append(f"Missing orchestrators: {', '.join(missing_orchestrators)}")
        report["status"] = "incomplete"
    
    # Check for brain instance connectivity
    brain_connected = []
    for orc_type, orchestrator in orchestrator_suite.items():
        if orchestrator and hasattr(orchestrator, 'brain') and orchestrator.brain is not None:
            brain_connected.append(orc_type)
    
    if len(brain_connected) < len([o for o in orchestrator_suite.values() if o is not None]):
        report["recommendations"].append("Consider connecting all orchestrators to brain instance for better integration")
    
    return report