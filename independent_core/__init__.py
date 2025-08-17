"""
Independent Core Module
Creates import aliases for integration test compatibility.
"""

import sys
from types import ModuleType

# Create gpu_memory_management module aliases for integration test
gpu_memory_management = ModuleType('gpu_memory_management')

# Import from actual locations in compression_systems
from .compression_systems.gpu_memory.smart_pool import SmartPool
from .compression_systems.gpu_memory.auto_swap_manager import AutoSwapManager

# Create smart_pool submodule
smart_pool = ModuleType('smart_pool')
smart_pool.SmartPool = SmartPool
gpu_memory_management.smart_pool = smart_pool

# Create auto_swap submodule  
auto_swap = ModuleType('auto_swap')
auto_swap.AutoSwap = AutoSwapManager  # Map AutoSwapManager to AutoSwap expected by test
gpu_memory_management.auto_swap = auto_swap

# Register gpu_memory_management modules
sys.modules[f'{__name__}.gpu_memory_management'] = gpu_memory_management
sys.modules[f'{__name__}.gpu_memory_management.smart_pool'] = smart_pool
sys.modules[f'{__name__}.gpu_memory_management.auto_swap'] = auto_swap

# Create system_integration module alias
from .compression_systems.system_integration_coordinator import SystemIntegrationCoordinator
system_integration = ModuleType('system_integration')
system_integration.MasterSystemCoordinator = SystemIntegrationCoordinator
sys.modules[f'{__name__}.system_integration'] = system_integration