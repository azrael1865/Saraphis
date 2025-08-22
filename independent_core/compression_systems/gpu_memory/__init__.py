"""
GPU Memory Management System Module - NO FALLBACKS, HARD FAILURES ONLY
Production-ready implementation that fails immediately on any missing component
"""

# Core GPU memory optimizer - REQUIRED, NO FALLBACK
from .gpu_memory_core import (
    GPUMemoryOptimizer,
    GPUMemoryManager,
    MemoryPool,
    StreamManager,
    MemoryOptimizer,
    GPUMemoryBlock,
    MemoryBlock,
    MemoryState,
    CUDAStream,
    GPUOptimizationResult,
    KernelConfig,
    GPUMemoryIntegration
)

# GPU Auto Detector - REQUIRED, NO FALLBACK
from .gpu_auto_detector import (
    GPUAutoDetector,
    GPUSpecs,
    AutoOptimizedConfig,
    ConfigUpdater,
    get_gpu_detector,
    get_config_updater,
    auto_configure_system
)

# DOA Scorer - REQUIRED, NO FALLBACK
from .doa_scorer import (
    DOAScorer,
    DOAScore,
    AccessMetrics,
    AccessPattern,
    SwapPriority
)

# Priority Swapper - REQUIRED, NO FALLBACK
from .priority_swapper import (
    PrioritySwapper,
    SwapOperation,
    SwapBatch,
    SwapDirection,
    SwapStrategy
)

# AutoSwap Manager - REQUIRED, NO FALLBACK
from .auto_swap_manager import (
    AutoSwapManager,
    AutoSwapConfig,
    SwapPolicy,
    MemoryPressureLevel,
    SwapDecision
)

# SmartPool components - REQUIRED, NO FALLBACK
from .smart_pool import (
    SmartPool,
    SmartPoolConfig,
    AllocationRequest,
    SmartPoolStatistics,
    integrate_smartpool_with_gpu_optimizer
)

from .weighted_interval_graph import (
    WeightedIntervalGraphColoring,
    MemoryInterval,
    IntervalNode,
    AllocationStatus
)

from .advanced_memory_pool import (
    AdvancedMemoryPoolManager,
    TieredMemoryPool,
    PooledMemoryBlock,
    PoolTier,
    AllocationStrategy,
    PoolStatistics
)

# CPU Bursting - REQUIRED, NO FALLBACK
from .cpu_bursting_pipeline import (
    CPU_BurstingPipeline,
    CPUBurstingConfig,
    CPUDecompressionEngine,
    DecompressionMode,
    DecompressionTask,
    BurstingStatistics,
    integrate_cpu_bursting
)

# Export all components - ALL REQUIRED
__all__ = [
    # Core GPU memory optimizer
    'GPUMemoryOptimizer',
    'GPUMemoryManager',
    'MemoryPool',
    'StreamManager',
    'MemoryOptimizer',
    'GPUMemoryBlock',
    'MemoryBlock',
    'MemoryState',
    'CUDAStream',
    'GPUOptimizationResult',
    'KernelConfig',
    'GPUMemoryIntegration',
    
    # GPU Auto Detector
    'GPUAutoDetector',
    'GPUSpecs',
    'AutoOptimizedConfig',
    'ConfigUpdater',
    'get_gpu_detector',
    'get_config_updater',
    'auto_configure_system',
    
    # DOA Scorer
    'DOAScorer',
    'DOAScore',
    'AccessMetrics',
    'AccessPattern',
    'SwapPriority',
    
    # Priority Swapper
    'PrioritySwapper',
    'SwapOperation',
    'SwapBatch',
    'SwapDirection',
    'SwapStrategy',
    
    # AutoSwap Manager
    'AutoSwapManager',
    'AutoSwapConfig',
    'SwapPolicy',
    'MemoryPressureLevel',
    'SwapDecision',
    
    # SmartPool system
    'SmartPool',
    'SmartPoolConfig',
    'AllocationRequest',
    'SmartPoolStatistics',
    'integrate_smartpool_with_gpu_optimizer',
    
    # Weighted interval graph coloring
    'WeightedIntervalGraphColoring',
    'MemoryInterval',
    'IntervalNode',
    'AllocationStatus',
    
    # Advanced memory pool manager
    'AdvancedMemoryPoolManager',
    'TieredMemoryPool',
    'PooledMemoryBlock',
    'PoolTier',
    'AllocationStrategy',
    'PoolStatistics',
    
    # CPU Bursting Pipeline
    'CPU_BurstingPipeline',
    'CPUBurstingConfig',
    'CPUDecompressionEngine',
    'DecompressionMode',
    'DecompressionTask',
    'BurstingStatistics',
    'integrate_cpu_bursting',
]

# Module version
__version__ = '2.0.0'  # NO FALLBACKS VERSION

# Module metadata
__author__ = 'Independent Core Development Team'
__description__ = 'GPU memory management - ALL COMPONENTS REQUIRED, NO FALLBACKS'
__status__ = 'Production'
__target_fragmentation_reduction__ = 0.133  # 13.3%
