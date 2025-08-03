"""
Saraphis Production Multi-Agent Development Framework
Production-ready multi-agent system for parallel development
"""

from .multi_agent_orchestrator import MultiAgentOrchestrator
from .agent_manager import AgentManager
from .agent_coordinator import AgentCoordinator
from .task_distributor import TaskDistributor
from .agent_monitor import AgentMonitor
from .agent_integration_manager import AgentIntegrationManager

# Import specialized agents
from .specialized_agents import (
    BrainOrchestrationAgent,
    ProofSystemAgent,
    UncertaintyAgent,
    TrainingAgent,
    DomainAgent,
    CompressionAgent,
    ProductionAgent,
    WebInterfaceAgent
)

__all__ = [
    'MultiAgentOrchestrator',
    'AgentManager',
    'AgentCoordinator',
    'TaskDistributor',
    'AgentMonitor',
    'AgentIntegrationManager',
    'BrainOrchestrationAgent',
    'ProofSystemAgent',
    'UncertaintyAgent',
    'TrainingAgent',
    'DomainAgent',
    'CompressionAgent',
    'ProductionAgent',
    'WebInterfaceAgent'
]