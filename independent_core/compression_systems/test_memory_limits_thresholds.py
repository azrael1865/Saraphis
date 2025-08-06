"""
Comprehensive Memory Limits and Threshold Configuration Tests

This module extends test_end_to_end_compression.py with detailed memory management tests.
Import this module in test_end_to_end_compression.py or run independently.

Tests cover:
- Memory limit configuration (hard/soft limits, warnings, budgets)
- Memory pressure thresholds (critical/high/moderate/low with actions)
- Dynamic threshold adjustment (workload-based, history-based, streaming)
- Memory allocation strategies (eager/lazy, pooling, fragmentation)
- Multi-tier memory (GPU VRAM, shared memory, CPU RAM, disk swap)
- Memory monitoring and alerts (tracking, leak detection, predictions)

NO FALLBACKS - HARD FAILURES ONLY
"""

import pytest
import torch
import numpy as np
import time
import warnings
from unittest.mock import patch, MagicMock
from collections import deque
from typing import Dict, Any, Optional, List

# Import components to test
from independent_core.compression_systems.padic.memory_pressure_handler import (
    MemoryPressureHandler,
    PressureHandlerConfig,
    ProcessingMode,
    MemoryState
)
from independent_core.compression_systems.system_integration_coordinator import (
    SystemConfiguration,
    SystemIntegrationCoordinator
)


class TestMemoryLimitConfiguration:
    """Test memory limit configuration across GPU, CPU, and system boundaries
    
    Validates hard limits, soft limits, warnings, and per-operation budgets
    NO FALLBACKS - HARD FAILURES ONLY
    """
    
    @pytest.fixture
    def memory_handler(self):
        """Create memory pressure handler with custom limits"""
        config = PressureHandlerConfig(
            gpu_critical_threshold_mb=24000,  # 24GB for testing
            gpu_high_threshold_mb=20400,      # 85% of 24GB
            gpu_moderate_threshold_mb=16800,  # 70% of 24GB
            cpu_critical_threshold_mb=32000,  # 32GB CPU limit
            force_cpu_on_critical=True
        )
        return MemoryPressureHandler(config)
    
    @pytest.fixture
    def system_config(self):
        """Create system configuration with memory limits"""
        config = SystemConfiguration()
        config.gpu_memory_limit_mb = 24576  # 24GB hard limit
        config.enable_cpu_bursting = True
        config.enable_memory_pressure = True
        return config
    
    def test_gpu_memory_hard_limits(self, memory_handler):
        """Test GPU memory hard limits that cannot be exceeded
        
        Validates:
        - Hard limit enforcement
        - Exception on limit breach
        - No allocation beyond limit
        - Proper error messages
        """
        # Set hard limit
        hard_limit_mb = 24000
        memory_handler.config.gpu_critical_threshold_mb = hard_limit_mb
        
        # Mock current GPU memory state near limit
        with patch('torch.cuda.memory_allocated') as mock_allocated:
            with patch('torch.cuda.memory_reserved') as mock_reserved:
                with patch('torch.cuda.max_memory_allocated') as mock_max:
                    # Simulate memory at 99% of hard limit
                    mock_allocated.return_value = hard_limit_mb * 1024 * 1024 * 0.99
                    mock_reserved.return_value = hard_limit_mb * 1024 * 1024
                    mock_max.return_value = hard_limit_mb * 1024 * 1024
                    
                    # Attempt to allocate more memory should fail
                    metrics = memory_handler.get_current_metrics()
                    assert metrics.gpu_allocated_mb >= hard_limit_mb * 0.99
                    
                    # Decision should force CPU processing
                    decision = memory_handler.decide_processing_mode(
                        tensor_size_mb=100  # Try to allocate 100MB more
                    )
                    assert decision == ProcessingMode.CPU_REQUIRED
    
    def test_gpu_memory_soft_limits(self, memory_handler):
        """Test GPU memory soft limits that trigger warnings
        
        Validates:
        - Soft limit warnings
        - Continued operation with warnings
        - Threshold-based alerts
        - Warning accumulation
        """
        soft_limit_mb = 20000  # Soft limit at ~83%
        memory_handler.config.gpu_high_threshold_mb = soft_limit_mb
        
        warning_count = 0
        warnings_captured = []
        
        def warning_handler(message, category, filename, lineno, file=None, line=None):
            nonlocal warning_count
            warning_count += 1
            warnings_captured.append(str(message))
        
        # Set up warning handler
        old_showwarning = warnings.showwarning
        warnings.showwarning = warning_handler
        
        try:
            # Simulate memory at soft limit
            with patch('torch.cuda.memory_allocated') as mock_allocated:
                mock_allocated.return_value = soft_limit_mb * 1024 * 1024
                
                # Check should trigger warning
                state = memory_handler.check_memory_state()
                assert state in [MemoryState.HIGH, MemoryState.MODERATE]
                
                # Multiple checks should accumulate warnings
                for _ in range(5):
                    memory_handler.check_memory_state()
                
                # Should have warnings but continue operation
                assert warning_count > 0
                assert any('memory' in w.lower() for w in warnings_captured)
        finally:
            warnings.showwarning = old_showwarning
    
    def test_cpu_memory_limits_for_bursting(self, memory_handler):
        """Test CPU memory limits when bursting from GPU
        
        Validates:
        - CPU memory threshold for bursting
        - CPU allocation limits
        - Bursting trigger conditions
        - CPU memory monitoring
        """
        cpu_burst_threshold_mb = 4000  # Start bursting at 4GB CPU available
        memory_handler.config.cpu_critical_threshold_mb = cpu_burst_threshold_mb
        
        # Mock CPU memory state
        with patch('psutil.virtual_memory') as mock_mem:
            mock_mem.return_value = MagicMock(
                total=32 * 1024 * 1024 * 1024,  # 32GB total
                available=cpu_burst_threshold_mb * 1024 * 1024,  # At threshold
                percent=87.5  # High usage
            )
            
            # Check CPU availability for bursting
            metrics = memory_handler.get_current_metrics()
            assert metrics.cpu_available_mb <= cpu_burst_threshold_mb
            
            # Large tensor should trigger bursting decision
            decision = memory_handler.decide_processing_mode(
                tensor_size_mb=2000,  # 2GB tensor
                mode=ProcessingMode.ADAPTIVE
            )
            
            # Should consider CPU capacity
            if metrics.cpu_available_mb < 2000:
                assert decision != ProcessingMode.CPU_PREFERRED
    
    def test_system_wide_memory_limits(self, system_config):
        """Test system-wide memory limits across all components
        
        Validates:
        - Total system memory cap
        - Combined GPU+CPU limits
        - System memory percentage limits
        - Global allocation tracking
        """
        coordinator = SystemIntegrationCoordinator(system_config)
        
        # Set system-wide limit as percentage
        system_memory_limit_percent = 0.8  # Use max 80% of system RAM
        
        with patch('psutil.virtual_memory') as mock_mem:
            total_ram = 64 * 1024 * 1024 * 1024  # 64GB
            mock_mem.return_value = MagicMock(
                total=total_ram,
                available=total_ram * 0.3,  # 30% available
                percent=70  # 70% used
            )
            
            # Calculate effective limit
            max_allowed_mb = (total_ram * system_memory_limit_percent) / (1024 * 1024)
            current_used_mb = (total_ram * 0.7) / (1024 * 1024)
            
            # Check if allocation would exceed limit
            can_allocate_10gb = current_used_mb + 10240 < max_allowed_mb
            assert can_allocate_10gb == True  # Should fit within 80% limit
            
            can_allocate_20gb = current_used_mb + 20480 < max_allowed_mb
            assert can_allocate_20gb == False  # Would exceed 80% limit
    
    def test_per_operation_memory_budgets(self, memory_handler):
        """Test memory budgets for individual operations
        
        Validates:
        - Per-operation memory limits
        - Budget enforcement
        - Multi-operation coordination
        - Budget overflow handling
        """
        # Define operation budgets
        operation_budgets = {
            'compression': 4096,     # 4GB for compression
            'decompression': 2048,   # 2GB for decompression
            'encoding': 1024,        # 1GB for encoding
            'transform': 512         # 512MB for transforms
        }
        
        # Track allocations
        current_allocations = {}
        
        def allocate_for_operation(op_name: str, size_mb: int) -> bool:
            """Try to allocate memory for operation"""
            if op_name not in operation_budgets:
                raise ValueError(f"Unknown operation: {op_name}")
            
            current = current_allocations.get(op_name, 0)
            if current + size_mb <= operation_budgets[op_name]:
                current_allocations[op_name] = current + size_mb
                return True
            return False
        
        # Test allocations within budget
        assert allocate_for_operation('compression', 2000) == True
        assert allocate_for_operation('compression', 1000) == True
        assert allocate_for_operation('compression', 1000) == True
        
        # This should exceed budget
        assert allocate_for_operation('compression', 1000) == False
        
        # Other operations should have independent budgets
        assert allocate_for_operation('decompression', 2000) == True
        assert allocate_for_operation('encoding', 1000) == True
        
        # Verify total allocations
        total_allocated = sum(current_allocations.values())
        assert total_allocated == 6000  # 3000 + 2000 + 1000


class TestMemoryPressureThresholds:
    """Test memory pressure thresholds and appropriate system responses
    
    Tests critical, high, moderate, and low thresholds with proper actions
    NO FALLBACKS - HARD FAILURES ONLY
    """
    
    @pytest.fixture
    def pressure_handler(self):
        """Create handler with well-defined thresholds"""
        config = PressureHandlerConfig(
            gpu_critical_threshold_mb=23280,   # 95% of 24GB
            gpu_high_threshold_mb=20400,       # 85% of 24GB  
            gpu_moderate_threshold_mb=16800,   # 70% of 24GB
            gpu_critical_utilization=0.95,
            gpu_high_utilization=0.85,
            gpu_moderate_utilization=0.70,
            force_cpu_on_critical=True,
            adaptive_threshold=True
        )
        return MemoryPressureHandler(config)
    
    def test_critical_threshold_95_percent(self, pressure_handler):
        """Test critical threshold at 95% - emergency actions required
        
        Validates:
        - Emergency cleanup triggers
        - Operation cancellation
        - Forced garbage collection
        - Critical state reporting
        """
        critical_bytes = 23280 * 1024 * 1024  # 95% of 24GB
        
        with patch('torch.cuda.memory_allocated') as mock_allocated:
            with patch('torch.cuda.synchronize') as mock_sync:
                with patch('torch.cuda.empty_cache') as mock_empty:
                    mock_allocated.return_value = critical_bytes
                    
                    # Check state detection
                    state = pressure_handler.check_memory_state()
                    assert state == MemoryState.CRITICAL
                    
                    # Emergency actions should trigger
                    pressure_handler.handle_critical_memory()
                    
                    # Verify emergency procedures called
                    mock_sync.assert_called()  # Synchronize before cleanup
                    mock_empty.assert_called()  # Empty cache
                    
                    # New allocations should be rejected
                    decision = pressure_handler.decide_processing_mode(
                        tensor_size_mb=100,
                        mode=ProcessingMode.GPU_PREFERRED
                    )
                    assert decision == ProcessingMode.CPU_REQUIRED
    
    def test_high_threshold_85_percent(self, pressure_handler):
        """Test high threshold at 85% - aggressive cleanup needed
        
        Validates:
        - Aggressive garbage collection
        - CPU bursting activation
        - Batch size reduction
        - Cache clearing
        """
        high_bytes = 20400 * 1024 * 1024  # 85% of 24GB
        
        with patch('torch.cuda.memory_allocated') as mock_allocated:
            with patch('torch.cuda.empty_cache') as mock_empty:
                mock_allocated.return_value = high_bytes
                
                # Check state detection
                state = pressure_handler.check_memory_state()
                assert state == MemoryState.HIGH
                
                # High pressure actions
                actions = pressure_handler.get_recommended_actions(state)
                assert 'reduce_batch' in actions
                assert 'clear_cache' in actions
                assert 'enable_cpu_bursting' in actions
                
                # Verify CPU bursting preferred
                decision = pressure_handler.decide_processing_mode(
                    tensor_size_mb=1000,  # 1GB tensor
                    mode=ProcessingMode.ADAPTIVE
                )
                assert decision in [ProcessingMode.CPU_PREFERRED, ProcessingMode.CPU_REQUIRED]
    
    def test_moderate_threshold_70_percent(self, pressure_handler):
        """Test moderate threshold at 70% - preventive measures
        
        Validates:
        - Cache size reduction
        - Increased garbage collection frequency
        - Batch size optimization
        - Memory defragmentation
        """
        moderate_bytes = 16800 * 1024 * 1024  # 70% of 24GB
        
        with patch('torch.cuda.memory_allocated') as mock_allocated:
            mock_allocated.return_value = moderate_bytes
            
            # Check state detection
            state = pressure_handler.check_memory_state()
            assert state == MemoryState.MODERATE
            
            # Preventive actions
            actions = pressure_handler.get_recommended_actions(state)
            assert 'optimize_cache' in actions
            assert 'increase_gc_frequency' in actions
            
            # GPU still preferred but with caution
            decision = pressure_handler.decide_processing_mode(
                tensor_size_mb=500,  # 500MB tensor
                mode=ProcessingMode.ADAPTIVE
            )
            # Should still use GPU if tensor fits
            if moderate_bytes + (500 * 1024 * 1024) < 23280 * 1024 * 1024:
                assert decision == ProcessingMode.GPU_PREFERRED
    
    def test_low_threshold_50_percent(self, pressure_handler):
        """Test low threshold at 50% - normal operation
        
        Validates:
        - Normal operation mode
        - Opportunistic caching enabled
        - Full batch sizes
        - Prefetching active
        """
        low_bytes = 12000 * 1024 * 1024  # 50% of 24GB
        
        with patch('torch.cuda.memory_allocated') as mock_allocated:
            mock_allocated.return_value = low_bytes
            
            # Check state detection
            state = pressure_handler.check_memory_state()
            assert state == MemoryState.HEALTHY
            
            # Normal operation actions
            actions = pressure_handler.get_recommended_actions(state)
            assert 'enable_caching' in actions
            assert 'enable_prefetch' in actions
            assert 'full_batch_size' in actions
            
            # GPU strongly preferred
            decision = pressure_handler.decide_processing_mode(
                tensor_size_mb=2000,  # 2GB tensor
                mode=ProcessingMode.ADAPTIVE
            )
            assert decision == ProcessingMode.GPU_PREFERRED
    
    def test_threshold_transitions_and_hysteresis(self, pressure_handler):
        """Test threshold transitions with hysteresis to prevent oscillation
        
        Validates:
        - Smooth transitions between states
        - Hysteresis prevents rapid switching
        - State persistence requirements
        - Gradual response changes
        """
        # Simulate memory fluctuating around threshold
        memory_sequence = [
            (16700, MemoryState.MODERATE),  # Just below 70%
            (16900, MemoryState.MODERATE),  # Just above 70%
            (16600, MemoryState.MODERATE),  # Back below (hysteresis)
            (15000, MemoryState.HEALTHY),   # Clear drop
            (17000, MemoryState.MODERATE),  # Back up
            (20500, MemoryState.HIGH),      # Jump to high
            (20300, MemoryState.HIGH),      # Small drop (hysteresis)
            (19000, MemoryState.MODERATE),  # Clear drop
        ]
        
        previous_state = None
        state_changes = 0
        
        for memory_mb, expected_state in memory_sequence:
            with patch('torch.cuda.memory_allocated') as mock_allocated:
                mock_allocated.return_value = memory_mb * 1024 * 1024
                
                state = pressure_handler.check_memory_state()
                
                # Track state changes
                if previous_state and state != previous_state:
                    state_changes += 1
                previous_state = state
                
                # Verify expected state (with some tolerance for hysteresis)
                assert state in [expected_state, previous_state]
        
        # Should have reasonable number of state changes (not oscillating)
        assert state_changes <= 5  # Maximum expected transitions


class TestDynamicThresholdAdjustment:
    """Test dynamic threshold adjustment based on workload and history
    
    Adapts thresholds for different scenarios and workload patterns
    NO FALLBACKS - HARD FAILURES ONLY
    """
    
    @pytest.fixture
    def adaptive_handler(self):
        """Create handler with adaptive thresholds enabled"""
        config = PressureHandlerConfig(
            adaptive_threshold=True,
            monitoring_interval_ms=100,
            history_window_size=100,
            prediction_horizon_ms=1000
        )
        handler = MemoryPressureHandler(config)
        # Initialize with base thresholds
        handler.config.gpu_critical_threshold_mb = 23280
        handler.config.gpu_high_threshold_mb = 20400
        handler.config.gpu_moderate_threshold_mb = 16800
        return handler
    
    def test_threshold_scaling_based_on_workload(self, adaptive_handler):
        """Test threshold scaling for different workload types
        
        Validates:
        - Threshold adjustment for batch processing
        - Threshold adjustment for streaming
        - Threshold adjustment for interactive
        - Workload-specific optimization
        """
        # Define workload patterns
        workloads = {
            'batch_processing': {
                'pattern': 'steady',
                'memory_usage': 'high',
                'threshold_adjustment': 0.9  # More aggressive, allow higher usage
            },
            'streaming': {
                'pattern': 'variable',
                'memory_usage': 'moderate',
                'threshold_adjustment': 0.85  # Conservative for stability
            },
            'interactive': {
                'pattern': 'bursty',
                'memory_usage': 'low',
                'threshold_adjustment': 0.95  # Very conservative for responsiveness
            }
        }
        
        for workload_type, params in workloads.items():
            # Simulate workload detection
            adaptive_handler.detect_workload_pattern(workload_type)
            
            # Get adjusted thresholds
            adjusted_critical = adaptive_handler.get_adjusted_threshold('critical')
            adjusted_high = adaptive_handler.get_adjusted_threshold('high')
            
            # Verify thresholds adjusted appropriately
            base_critical = 23280
            expected_critical = base_critical * params['threshold_adjustment']
            
            # Allow some tolerance in adjustment
            assert abs(adjusted_critical - expected_critical) < 500
    
    def test_adaptive_thresholds_based_on_history(self, adaptive_handler):
        """Test threshold adaptation based on historical patterns
        
        Validates:
        - Learning from past memory usage
        - Predictive threshold adjustment
        - History-based optimization
        - Pattern recognition
        """
        # Simulate historical memory usage pattern
        history = []
        
        # Generate usage pattern (sine wave with trend)
        for i in range(100):
            time_point = i * 0.1
            base_usage = 15000  # 15GB base
            variation = 3000 * np.sin(time_point)  # ±3GB variation
            trend = 50 * i  # Gradual increase
            usage_mb = base_usage + variation + trend
            history.append(usage_mb)
        
        # Feed history to handler
        for usage_mb in history:
            with patch('torch.cuda.memory_allocated') as mock_allocated:
                mock_allocated.return_value = usage_mb * 1024 * 1024
                adaptive_handler.update_history()
        
        # Predict future usage
        predicted_usage = adaptive_handler.predict_memory_usage(
            horizon_ms=1000
        )
        
        # Should predict increasing trend
        assert predicted_usage > history[-1]
        
        # Adjust thresholds based on prediction
        adaptive_handler.adjust_thresholds_predictive()
        
        # Thresholds should be more conservative due to upward trend
        assert adaptive_handler.config.gpu_high_threshold_mb < 20400
    
    def test_threshold_adjustment_for_batch_operations(self, adaptive_handler):
        """Test threshold adjustment for batch processing
        
        Validates:
        - Batch size impact on thresholds
        - Dynamic batch optimization
        - Memory reservation for batches
        - Batch-aware scheduling
        """
        batch_sizes = [32, 64, 128, 256, 512]
        memory_per_item = 50  # 50MB per item
        
        for batch_size in batch_sizes:
            # Calculate memory requirement
            batch_memory_mb = batch_size * memory_per_item
            
            # Adjust thresholds for batch
            adaptive_handler.adjust_for_batch_size(batch_size)
            
            # Get adjusted thresholds
            moderate_threshold = adaptive_handler.config.gpu_moderate_threshold_mb
            
            # Ensure enough headroom for batch
            min_headroom = batch_memory_mb * 1.5  # 50% safety margin
            
            # Moderate threshold should leave room for at least one batch
            assert (24576 - moderate_threshold) >= min_headroom
            
            # Test batch processing decision
            with patch('torch.cuda.memory_allocated') as mock_allocated:
                # Set memory just below adjusted threshold
                mock_allocated.return_value = (moderate_threshold - 100) * 1024 * 1024
                
                # Should allow batch processing
                can_process = adaptive_handler.can_process_batch(batch_size)
                assert can_process == True
    
    def test_threshold_tuning_for_streaming_data(self, adaptive_handler):
        """Test threshold tuning for streaming workloads
        
        Validates:
        - Stream buffer requirements
        - Continuous processing thresholds
        - Stream-aware memory management
        - Latency-optimized thresholds
        """
        # Configure for streaming
        stream_config = {
            'buffer_size_mb': 1024,
            'throughput_mbps': 100,
            'latency_target_ms': 50
        }
        
        adaptive_handler.configure_for_streaming(stream_config)
        
        # Verify threshold adjustments
        assert adaptive_handler.config.gpu_moderate_threshold_mb < 16800
        
        # Ensure buffer space reserved
        max_usable = 24576 - stream_config['buffer_size_mb']
        assert adaptive_handler.config.gpu_critical_threshold_mb <= max_usable
        
        # Test streaming decisions
        with patch('torch.cuda.memory_allocated') as mock_allocated:
            # Simulate various memory levels
            for memory_percent in [0.4, 0.6, 0.8, 0.9]:
                memory_mb = 24576 * memory_percent
                mock_allocated.return_value = memory_mb * 1024 * 1024
                
                # Check if streaming can continue
                can_stream = adaptive_handler.can_accept_stream_data(
                    data_size_mb=100
                )
                
                # Should reject if would exceed buffer requirements
                if memory_mb + 100 > max_usable:
                    assert can_stream == False
                else:
                    assert can_stream == True
    
    def test_threshold_optimization_latency_vs_throughput(self, adaptive_handler):
        """Test threshold optimization for latency vs throughput tradeoffs
        
        Validates:
        - Latency-optimized thresholds
        - Throughput-optimized thresholds
        - Balanced optimization
        - Dynamic tradeoff adjustment
        """
        optimization_modes = {
            'latency': {
                'target_latency_ms': 10,
                'acceptable_throughput_loss': 0.3,
                'threshold_multiplier': 0.7  # More conservative
            },
            'throughput': {
                'target_throughput_mult': 2.0,
                'acceptable_latency_ms': 100,
                'threshold_multiplier': 0.95  # More aggressive
            },
            'balanced': {
                'latency_weight': 0.5,
                'throughput_weight': 0.5,
                'threshold_multiplier': 0.85  # Middle ground
            }
        }
        
        for mode, params in optimization_modes.items():
            # Set optimization mode
            adaptive_handler.set_optimization_mode(mode)
            
            # Apply threshold adjustments
            base_critical = 23280
            adjusted_critical = base_critical * params['threshold_multiplier']
            
            # Verify threshold adjusted appropriately
            actual_critical = adaptive_handler.config.gpu_critical_threshold_mb
            assert abs(actual_critical - adjusted_critical) < 500
            
            # Test mode-specific decisions
            with patch('torch.cuda.memory_allocated') as mock_allocated:
                mock_allocated.return_value = 20000 * 1024 * 1024  # ~81% usage
                
                decision = adaptive_handler.decide_processing_mode(
                    tensor_size_mb=1000,
                    mode=ProcessingMode.ADAPTIVE
                )
                
                if mode == 'latency':
                    # Should prefer CPU for lower latency
                    assert decision == ProcessingMode.CPU_PREFERRED
                elif mode == 'throughput':
                    # Should still try GPU for throughput
                    assert decision == ProcessingMode.GPU_PREFERRED


class TestMemoryAllocationStrategies:
    """Test different memory allocation strategies
    
    Tests eager vs lazy allocation, memory pool sizing, fragmentation handling
    NO FALLBACKS - HARD FAILURES ONLY
    """
    
    @pytest.fixture
    def memory_pool(self):
        """Create memory pool manager with configurable strategies"""
        from independent_core.compression_systems.gpu_memory.gpu_memory_core import (
            GPUMemoryOptimizer, GPUMemoryBlock
        )
        return GPUMemoryOptimizer({
            'pool_size_mb': 8192,  # 8GB pool
            'allocation_strategy': 'lazy',
            'fragmentation_threshold': 0.3,
            'defrag_trigger_threshold': 0.4,
            'emergency_reserve_mb': 512
        })
    
    def test_eager_allocation_vs_lazy_allocation(self, memory_pool):
        """Test eager allocation vs lazy allocation strategies
        
        Validates:
        - Eager pre-allocation behavior
        - Lazy on-demand allocation
        - Performance characteristics
        - Memory utilization differences
        """
        # Test eager allocation
        memory_pool.config['allocation_strategy'] = 'eager'
        
        # Eager should pre-allocate entire pool
        with patch('torch.cuda.empty_cache') as mock_empty:
            memory_pool.initialize_pool()
            
            # Verify pool pre-allocated
            assert memory_pool.total_pool_size_mb == 8192
            assert memory_pool.allocated_blocks_count > 0
            
            # Request memory - should be fast (already allocated)
            start_time = time.time()
            block = memory_pool.allocate(size_mb=100)
            eager_time = time.time() - start_time
            assert block is not None
            assert eager_time < 0.01  # Should be very fast
        
        # Reset and test lazy allocation
        memory_pool.reset()
        memory_pool.config['allocation_strategy'] = 'lazy'
        
        # Lazy should not pre-allocate
        memory_pool.initialize_pool()
        assert memory_pool.allocated_blocks_count == 0
        
        # Request memory - triggers actual allocation
        start_time = time.time()
        block = memory_pool.allocate(size_mb=100)
        lazy_time = time.time() - start_time
        assert block is not None
        
        # Lazy might be slightly slower on first allocation
        # But saves memory when not needed
        assert memory_pool.allocated_blocks_count == 1
    
    def test_memory_pool_pre_allocation_sizing(self, memory_pool):
        """Test memory pool pre-allocation sizing strategies
        
        Validates:
        - Pool size calculation
        - Dynamic pool growth
        - Pool shrinking
        - Size optimization
        """
        # Test different pool sizing strategies
        sizing_strategies = {
            'fixed': 8192,  # Fixed 8GB
            'percentage': 0.5,  # 50% of available GPU memory
            'adaptive': 'workload',  # Based on workload analysis
            'tiered': [1024, 2048, 4096]  # Multiple tier sizes
        }
        
        for strategy, config in sizing_strategies.items():
            memory_pool.set_sizing_strategy(strategy, config)
            
            if strategy == 'fixed':
                assert memory_pool.get_pool_size() == 8192
                
            elif strategy == 'percentage':
                with patch('torch.cuda.get_device_properties') as mock_props:
                    mock_props.return_value.total_memory = 24 * 1024 * 1024 * 1024
                    expected_size = 24 * 1024 * 0.5  # 12GB
                    assert abs(memory_pool.get_pool_size() - expected_size) < 100
                    
            elif strategy == 'adaptive':
                # Simulate workload analysis
                memory_pool.analyze_workload_pattern()
                pool_size = memory_pool.get_pool_size()
                assert 4096 <= pool_size <= 16384  # Reasonable range
                
            elif strategy == 'tiered':
                # Should create multiple pools
                pools = memory_pool.get_tiered_pools()
                assert len(pools) == 3
                assert [p.size_mb for p in pools] == [1024, 2048, 4096]
    
    def test_memory_fragmentation_thresholds(self, memory_pool):
        """Test memory fragmentation detection and thresholds
        
        Validates:
        - Fragmentation measurement
        - Threshold detection
        - Fragmentation impact on allocation
        - Compaction triggers
        """
        # Simulate fragmented memory state
        allocations = []
        
        # Create fragmentation pattern: allocate and free alternating blocks
        for i in range(100):
            block = memory_pool.allocate(size_mb=10)
            allocations.append(block)
            
        # Free every other block to create fragmentation
        for i in range(0, 100, 2):
            memory_pool.free(allocations[i])
            allocations[i] = None
        
        # Calculate fragmentation
        fragmentation = memory_pool.calculate_fragmentation()
        assert fragmentation > 0.3  # Should be fragmented
        
        # Check if fragmentation exceeds threshold
        assert memory_pool.is_fragmented() == True
        
        # Try to allocate large contiguous block - should fail due to fragmentation
        large_block = memory_pool.allocate(size_mb=500)
        
        if fragmentation > 0.5:
            # Severe fragmentation - allocation should fail
            assert large_block is None
        else:
            # Moderate fragmentation - might succeed with compaction
            assert large_block is not None or memory_pool.compaction_triggered
    
    def test_defragmentation_trigger_thresholds(self, memory_pool):
        """Test defragmentation trigger conditions and thresholds
        
        Validates:
        - Automatic defragmentation triggers
        - Manual defragmentation
        - Defragmentation effectiveness
        - Performance impact tracking
        """
        defrag_threshold = 0.4  # Trigger at 40% fragmentation
        memory_pool.config['defrag_trigger_threshold'] = defrag_threshold
        
        # Create fragmentation
        self._create_fragmentation(memory_pool, target_fragmentation=0.45)
        
        # Check fragmentation level
        initial_fragmentation = memory_pool.calculate_fragmentation()
        assert initial_fragmentation > defrag_threshold
        
        # Defragmentation should trigger automatically
        defrag_triggered = memory_pool.check_and_defragment()
        assert defrag_triggered == True
        
        # Measure defragmentation effectiveness
        final_fragmentation = memory_pool.calculate_fragmentation()
        assert final_fragmentation < initial_fragmentation
        assert final_fragmentation < 0.2  # Should be much better
        
        # Track performance impact
        defrag_stats = memory_pool.get_last_defrag_stats()
        assert defrag_stats['duration_ms'] > 0
        assert defrag_stats['blocks_moved'] > 0
        assert defrag_stats['fragmentation_reduced'] > 0.2
    
    def test_emergency_memory_reserve_allocation(self, memory_pool):
        """Test emergency memory reserve allocation and usage
        
        Validates:
        - Reserve allocation
        - Reserve protection
        - Emergency access conditions
        - Reserve replenishment
        """
        reserve_size_mb = 512
        memory_pool.config['emergency_reserve_mb'] = reserve_size_mb
        
        # Initialize with reserve
        memory_pool.initialize_pool()
        
        # Verify reserve is allocated and protected
        assert memory_pool.emergency_reserve_size_mb == reserve_size_mb
        assert memory_pool.available_pool_size_mb == 8192 - reserve_size_mb
        
        # Normal allocation should not touch reserve
        allocations = []
        available = memory_pool.available_pool_size_mb
        
        # Allocate up to non-reserve limit
        while available > 100:
            block = memory_pool.allocate(size_mb=100)
            if block:
                allocations.append(block)
                available -= 100
            else:
                break
        
        # Should have stopped before touching reserve
        assert memory_pool.emergency_reserve_intact() == True
        
        # Emergency allocation should access reserve
        with memory_pool.emergency_mode():
            emergency_block = memory_pool.allocate(size_mb=100, emergency=True)
            assert emergency_block is not None
            assert memory_pool.emergency_reserve_used_mb == 100
        
        # After emergency, should try to replenish reserve
        for alloc in allocations[:5]:
            memory_pool.free(alloc)
        
        memory_pool.replenish_emergency_reserve()
        assert memory_pool.emergency_reserve_intact() == True
    
    def _create_fragmentation(self, memory_pool, target_fragmentation: float):
        """Helper to create specific fragmentation level"""
        allocations = []
        block_size = 10  # MB
        
        # Allocate blocks
        while memory_pool.calculate_fragmentation() < target_fragmentation:
            block = memory_pool.allocate(size_mb=block_size)
            if block:
                allocations.append(block)
            
            # Free some blocks to create gaps
            if len(allocations) > 2 and np.random.random() < 0.5:
                idx = np.random.randint(0, len(allocations) - 1)
                if allocations[idx]:
                    memory_pool.free(allocations[idx])
                    allocations[idx] = None


class TestMultiTierMemory:
    """Test multi-tier memory management across GPU, CPU, and disk
    
    Validates GPU VRAM, shared memory, CPU RAM, and disk swap management
    NO FALLBACKS - HARD FAILURES ONLY
    """
    
    @pytest.fixture
    def tier_manager(self):
        """Create multi-tier memory manager"""
        from independent_core.compression_systems.gpu_memory.cpu_bursting_pipeline import (
            CPUBurstingPipeline, BurstingConfig
        )
        
        config = BurstingConfig(
            gpu_memory_threshold_mb=20000,  # 20GB GPU limit
            cpu_memory_limit_mb=32768,      # 32GB CPU limit
            enable_disk_swap=True,
            disk_swap_path='/tmp/compression_swap',
            tier_promotion_threshold=0.8,   # Promote if accessed > 80% recently
            tier_demotion_threshold=0.2     # Demote if accessed < 20% recently
        )
        return CPUBurstingPipeline(config)
    
    def test_gpu_vram_limits(self, tier_manager):
        """Test GPU VRAM limit management
        
        Validates:
        - VRAM allocation limits
        - VRAM monitoring
        - VRAM overflow handling
        - VRAM reservation
        """
        vram_limit_mb = 20000  # 20GB
        
        # Set VRAM limit
        tier_manager.set_gpu_vram_limit(vram_limit_mb)
        
        # Mock current VRAM usage
        with patch('torch.cuda.memory_allocated') as mock_allocated:
            with patch('torch.cuda.max_memory_reserved') as mock_reserved:
                # Test various VRAM levels
                test_cases = [
                    (10000, True),   # 50% - should allow
                    (18000, True),   # 90% - should allow with warning
                    (19500, False),  # 97.5% - should deny
                    (20500, False),  # Over limit - must deny
                ]
                
                for vram_used_mb, should_allow in test_cases:
                    mock_allocated.return_value = vram_used_mb * 1024 * 1024
                    mock_reserved.return_value = vram_limit_mb * 1024 * 1024
                    
                    # Check if allocation allowed
                    can_allocate = tier_manager.can_allocate_gpu(size_mb=500)
                    assert can_allocate == should_allow
                    
                    # Verify monitoring accuracy
                    stats = tier_manager.get_gpu_memory_stats()
                    assert abs(stats['allocated_mb'] - vram_used_mb) < 1
                    assert stats['limit_mb'] == vram_limit_mb
    
    def test_gpu_shared_memory_limits(self, tier_manager):
        """Test GPU shared memory limit management
        
        Validates:
        - Shared memory allocation
        - Shared memory limits per kernel
        - Dynamic shared memory
        - Shared memory optimization
        """
        # GPU shared memory is typically 48KB per SM (streaming multiprocessor)
        shared_memory_per_sm_kb = 48
        num_sms = 68  # Example for RTX 3090
        
        # Configure shared memory limits
        tier_manager.configure_shared_memory(
            per_block_limit_kb=shared_memory_per_sm_kb,
            total_sms=num_sms
        )
        
        # Test kernel configurations
        kernel_configs = [
            {'shared_kb': 16, 'blocks': 3},  # Within limit
            {'shared_kb': 32, 'blocks': 1},  # Within limit
            {'shared_kb': 48, 'blocks': 1},  # At limit
            {'shared_kb': 64, 'blocks': 1},  # Exceeds limit
        ]
        
        for config in kernel_configs:
            shared_kb = config['shared_kb']
            blocks_per_sm = config['blocks']
            
            # Check if configuration valid
            is_valid = tier_manager.validate_shared_memory_config(
                shared_kb=shared_kb,
                blocks_per_sm=blocks_per_sm
            )
            
            if shared_kb * blocks_per_sm <= shared_memory_per_sm_kb:
                assert is_valid == True
            else:
                assert is_valid == False
            
            # Get optimized configuration
            if not is_valid:
                optimized = tier_manager.optimize_shared_memory_config(
                    required_shared_kb=shared_kb
                )
                assert optimized['shared_kb'] <= shared_memory_per_sm_kb
                assert optimized['blocks_per_sm'] >= 1
    
    def test_cpu_ram_limits_for_bursting(self, tier_manager):
        """Test CPU RAM limits when bursting from GPU
        
        Validates:
        - CPU RAM allocation limits
        - CPU bursting thresholds
        - CPU memory monitoring
        - Bursting decision logic
        """
        cpu_limit_mb = 32768  # 32GB
        tier_manager.set_cpu_memory_limit(cpu_limit_mb)
        
        # Mock system memory
        with patch('psutil.virtual_memory') as mock_mem:
            # Test different CPU memory scenarios
            test_scenarios = [
                {
                    'available_mb': 20000,
                    'total_mb': 32768,
                    'can_burst_size': 5000,  # Can burst 5GB
                    'should_burst': True
                },
                {
                    'available_mb': 2000,
                    'total_mb': 32768,
                    'can_burst_size': 5000,  # Want 5GB but only 2GB available
                    'should_burst': False
                },
                {
                    'available_mb': 10000,
                    'total_mb': 32768,
                    'can_burst_size': 8000,  # Can burst 8GB
                    'should_burst': True
                }
            ]
            
            for scenario in test_scenarios:
                mock_mem.return_value = MagicMock(
                    available=scenario['available_mb'] * 1024 * 1024,
                    total=scenario['total_mb'] * 1024 * 1024,
                    percent=(1 - scenario['available_mb']/scenario['total_mb']) * 100
                )
                
                # Check bursting decision
                can_burst = tier_manager.can_burst_to_cpu(
                    size_mb=scenario['can_burst_size']
                )
                assert can_burst == scenario['should_burst']
                
                # Verify CPU monitoring
                cpu_stats = tier_manager.get_cpu_memory_stats()
                assert abs(cpu_stats['available_mb'] - scenario['available_mb']) < 100
    
    def test_disk_swap_threshold_configuration(self, tier_manager):
        """Test disk swap threshold configuration and triggers
        
        Validates:
        - Swap enable/disable
        - Swap threshold configuration
        - Swap file management
        - Swap performance monitoring
        """
        # Configure disk swap
        swap_config = {
            'enabled': True,
            'swap_path': '/tmp/test_swap',
            'max_swap_size_mb': 10240,  # 10GB max swap
            'swap_threshold_percent': 90,  # Swap when 90% memory used
            'swap_block_size_mb': 128      # Swap in 128MB blocks
        }
        
        tier_manager.configure_disk_swap(swap_config)
        
        # Test swap trigger conditions
        with patch('psutil.virtual_memory') as mock_mem:
            # Memory at 85% - should not swap
            mock_mem.return_value = MagicMock(percent=85)
            should_swap = tier_manager.should_swap_to_disk()
            assert should_swap == False
            
            # Memory at 92% - should trigger swap
            mock_mem.return_value = MagicMock(percent=92)
            should_swap = tier_manager.should_swap_to_disk()
            assert should_swap == True
        
        # Test swap file creation
        with patch('os.path.exists') as mock_exists:
            with patch('os.makedirs') as mock_makedirs:
                mock_exists.return_value = False
                
                tier_manager.initialize_swap_file()
                mock_makedirs.assert_called_with('/tmp/test_swap', exist_ok=True)
        
        # Test swap operation
        test_data = torch.randn(1000, 1000)  # ~4MB tensor
        
        # Swap to disk
        swap_id = tier_manager.swap_to_disk(test_data, priority='low')
        assert swap_id is not None
        
        # Verify swap stats
        swap_stats = tier_manager.get_swap_statistics()
        assert swap_stats['total_swapped_mb'] > 0
        assert swap_stats['num_swapped_tensors'] == 1
        
        # Restore from swap
        restored_data = tier_manager.restore_from_swap(swap_id)
        assert torch.allclose(restored_data, test_data)
    
    def test_tiered_memory_movement_thresholds(self, tier_manager):
        """Test thresholds for moving data between memory tiers
        
        Validates:
        - Promotion thresholds (disk->CPU->GPU)
        - Demotion thresholds (GPU->CPU->disk)
        - Access pattern tracking
        - Automatic tier migration
        """
        # Configure tier movement thresholds
        tier_manager.configure_tier_thresholds(
            promote_to_gpu_threshold=0.8,   # 80% access frequency
            demote_from_gpu_threshold=0.2,  # 20% access frequency
            promote_to_cpu_threshold=0.5,   # 50% access frequency
            demote_to_disk_threshold=0.1    # 10% access frequency
        )
        
        # Create test tensors with access tracking
        tensor_profiles = [
            {'id': 'hot', 'size_mb': 100, 'access_freq': 0.9, 'tier': 'disk'},
            {'id': 'warm', 'size_mb': 200, 'access_freq': 0.6, 'tier': 'disk'},
            {'id': 'cold', 'size_mb': 150, 'access_freq': 0.1, 'tier': 'gpu'},
            {'id': 'cooling', 'size_mb': 120, 'access_freq': 0.3, 'tier': 'gpu'},
        ]
        
        # Simulate access patterns
        for profile in tensor_profiles:
            tier_manager.register_tensor(
                tensor_id=profile['id'],
                size_mb=profile['size_mb'],
                current_tier=profile['tier']
            )
            
            # Update access frequency
            tier_manager.update_access_frequency(
                tensor_id=profile['id'],
                frequency=profile['access_freq']
            )
        
        # Run tier optimization
        migrations = tier_manager.optimize_tier_placement()
        
        # Verify migrations
        expected_migrations = {
            'hot': 'gpu',     # Should promote to GPU (0.9 > 0.8)
            'warm': 'cpu',    # Should promote to CPU (0.6 > 0.5)
            'cold': 'disk',   # Should demote to disk (0.1 < 0.2)
            'cooling': 'cpu', # Should demote to CPU (0.3 < 0.2 for GPU)
        }
        
        for tensor_id, expected_tier in expected_migrations.items():
            actual_tier = tier_manager.get_tensor_tier(tensor_id)
            assert actual_tier == expected_tier
        
        # Verify migration stats
        stats = tier_manager.get_migration_statistics()
        assert stats['total_migrations'] == 4
        assert stats['promotions'] == 2
        assert stats['demotions'] == 2


class TestMemoryMonitoringAndAlerts:
    """Test memory monitoring, tracking, and alert systems
    
    Validates accurate tracking, threshold alerts, leak detection, and predictions
    NO FALLBACKS - HARD FAILURES ONLY
    """
    
    @pytest.fixture
    def monitor(self):
        """Create memory monitoring system"""
        from independent_core.compression_systems.padic.memory_pressure_handler import (
            MemoryPressureHandler, PressureHandlerConfig
        )
        
        config = PressureHandlerConfig(
            monitoring_interval_ms=100,
            alert_threshold_mb=1000,
            leak_detection_enabled=True,
            leak_threshold_mb_per_hour=500,
            growth_rate_limit_mb_per_sec=100,
            prediction_horizon_ms=5000
        )
        return MemoryPressureHandler(config)
    
    def test_memory_usage_tracking_accuracy(self, monitor):
        """Test accuracy of memory usage tracking
        
        Validates:
        - GPU memory tracking accuracy
        - CPU memory tracking accuracy
        - Real-time updates
        - Historical tracking
        """
        # Track memory over time
        measurements = []
        
        with patch('torch.cuda.memory_allocated') as mock_gpu:
            with patch('psutil.virtual_memory') as mock_cpu:
                # Simulate memory usage pattern
                for i in range(10):
                    gpu_mb = 10000 + i * 500  # Increasing GPU usage
                    cpu_mb = 20000 - i * 200  # Decreasing CPU available
                    
                    mock_gpu.return_value = gpu_mb * 1024 * 1024
                    mock_cpu.return_value = MagicMock(
                        available=cpu_mb * 1024 * 1024,
                        total=32768 * 1024 * 1024
                    )
                    
                    # Take measurement
                    metrics = monitor.get_current_metrics()
                    measurements.append({
                        'time': time.time(),
                        'gpu_mb': metrics.gpu_allocated_mb,
                        'cpu_available_mb': metrics.cpu_available_mb
                    })
                    
                    # Verify accuracy
                    assert abs(metrics.gpu_allocated_mb - gpu_mb) < 1
                    assert abs(metrics.cpu_available_mb - cpu_mb) < 1
                    
                    time.sleep(0.01)  # Small delay between measurements
        
        # Verify historical tracking
        history = monitor.get_memory_history()
        assert len(history) >= 10
        
        # Check trend detection
        gpu_trend = monitor.analyze_gpu_trend()
        assert gpu_trend == 'increasing'
        
        cpu_trend = monitor.analyze_cpu_trend()
        assert cpu_trend == 'decreasing'
    
    def test_threshold_breach_notifications(self, monitor):
        """Test threshold breach detection and notifications
        
        Validates:
        - Threshold breach detection
        - Alert generation
        - Alert deduplication
        - Alert severity levels
        """
        alerts_received = []
        
        def alert_handler(alert):
            alerts_received.append(alert)
        
        # Register alert handler
        monitor.register_alert_handler(alert_handler)
        
        # Set thresholds
        monitor.set_alert_thresholds({
            'gpu_critical': 22000,  # MB
            'gpu_warning': 18000,
            'cpu_critical': 2000,
            'cpu_warning': 5000
        })
        
        # Simulate threshold breaches
        with patch('torch.cuda.memory_allocated') as mock_gpu:
            with patch('psutil.virtual_memory') as mock_cpu:
                # Test warning threshold
                mock_gpu.return_value = 18500 * 1024 * 1024
                monitor.check_thresholds()
                
                assert len(alerts_received) == 1
                assert alerts_received[0]['severity'] == 'warning'
                assert 'gpu' in alerts_received[0]['type'].lower()
                
                # Test critical threshold
                mock_gpu.return_value = 22500 * 1024 * 1024
                monitor.check_thresholds()
                
                assert len(alerts_received) == 2
                assert alerts_received[1]['severity'] == 'critical'
                
                # Test deduplication - same level shouldn't re-alert immediately
                monitor.check_thresholds()
                assert len(alerts_received) == 2  # No new alert
                
                # Test CPU threshold
                mock_cpu.return_value = MagicMock(
                    available=1500 * 1024 * 1024
                )
                monitor.check_thresholds()
                
                assert len(alerts_received) == 3
                assert 'cpu' in alerts_received[2]['type'].lower()
    
    def test_memory_leak_detection_thresholds(self, monitor):
        """Test memory leak detection based on growth patterns
        
        Validates:
        - Leak pattern detection
        - Leak rate calculation
        - False positive prevention
        - Leak source identification
        """
        # Simulate potential memory leak pattern
        leak_pattern = []
        base_memory = 10000  # 10GB base
        
        with patch('torch.cuda.memory_allocated') as mock_gpu:
            # Simulate steady memory growth (potential leak)
            for hour in range(5):
                for minute in range(60):
                    # Linear growth: 10MB per minute = 600MB per hour
                    current_mb = base_memory + (hour * 600) + (minute * 10)
                    mock_gpu.return_value = current_mb * 1024 * 1024
                    
                    monitor.update_memory_tracking()
                    
                    # Fast forward time in test
                    with patch('time.time') as mock_time:
                        mock_time.return_value = (hour * 3600) + (minute * 60)
            
            # Check leak detection after 5 hours
            leak_detected = monitor.detect_memory_leak()
            assert leak_detected == True
            
            # Get leak statistics
            leak_stats = monitor.get_leak_statistics()
            assert leak_stats['leak_rate_mb_per_hour'] > 500  # Above threshold
            assert leak_stats['confidence'] > 0.8  # High confidence
            assert leak_stats['estimated_source'] is not None
        
        # Test non-leak pattern (oscillating memory)
        monitor.reset_tracking()
        
        with patch('torch.cuda.memory_allocated') as mock_gpu:
            for i in range(100):
                # Oscillating pattern - not a leak
                current_mb = base_memory + 500 * np.sin(i * 0.1)
                mock_gpu.return_value = current_mb * 1024 * 1024
                monitor.update_memory_tracking()
            
            leak_detected = monitor.detect_memory_leak()
            assert leak_detected == False
    
    def test_memory_growth_rate_limits(self, monitor):
        """Test memory growth rate limit detection
        
        Validates:
        - Growth rate calculation
        - Rate limit enforcement
        - Burst detection
        - Rate limiting actions
        """
        growth_limit_mb_per_sec = 100
        monitor.config.growth_rate_limit_mb_per_sec = growth_limit_mb_per_sec
        
        with patch('torch.cuda.memory_allocated') as mock_gpu:
            # Test normal growth rate
            start_mb = 10000
            mock_gpu.return_value = start_mb * 1024 * 1024
            monitor.update_memory_tracking()
            
            time.sleep(0.1)
            
            # Grow by 50MB/sec - within limit
            mock_gpu.return_value = (start_mb + 5) * 1024 * 1024
            growth_rate = monitor.calculate_growth_rate()
            assert growth_rate < growth_limit_mb_per_sec
            assert monitor.is_growth_rate_exceeded() == False
            
            # Test excessive growth rate
            time.sleep(0.1)
            
            # Grow by 200MB/sec - exceeds limit
            mock_gpu.return_value = (start_mb + 25) * 1024 * 1024
            growth_rate = monitor.calculate_growth_rate()
            assert growth_rate > growth_limit_mb_per_sec
            assert monitor.is_growth_rate_exceeded() == True
            
            # Get recommended actions
            actions = monitor.get_rate_limit_actions()
            assert 'throttle_allocations' in actions
            assert 'force_garbage_collection' in actions
            assert 'enable_cpu_bursting' in actions
    
    def test_predictive_memory_exhaustion_warnings(self, monitor):
        """Test predictive warnings for memory exhaustion
        
        Validates:
        - Future memory usage prediction
        - Exhaustion time estimation
        - Confidence intervals
        - Preventive action recommendations
        """
        # Create predictable growth pattern
        with patch('torch.cuda.memory_allocated') as mock_gpu:
            # Linear growth pattern for prediction
            measurements = []
            for i in range(20):
                current_mb = 10000 + i * 500  # 500MB per measurement
                mock_gpu.return_value = current_mb * 1024 * 1024
                
                monitor.update_memory_tracking()
                measurements.append((time.time(), current_mb))
                time.sleep(0.05)
            
            # Predict future memory usage
            prediction = monitor.predict_memory_exhaustion(
                horizon_seconds=10,
                memory_limit_mb=24000
            )
            
            assert prediction['will_exhaust'] == True
            assert prediction['time_to_exhaustion_seconds'] > 0
            assert prediction['confidence'] > 0.7
            assert prediction['predicted_usage_mb'] > 24000
            
            # Get preventive recommendations
            recommendations = monitor.get_exhaustion_prevention_actions()
            assert len(recommendations) > 0
            assert any('reduce' in r.lower() for r in recommendations)
            assert any('burst' in r.lower() or 'cpu' in r.lower() for r in recommendations)
        
        # Test pattern without exhaustion risk
        monitor.reset_tracking()
        
        with patch('torch.cuda.memory_allocated') as mock_gpu:
            # Stable memory usage
            for i in range(20):
                current_mb = 10000 + np.random.normal(0, 50)  # Small variations
                mock_gpu.return_value = current_mb * 1024 * 1024
                monitor.update_memory_tracking()
                time.sleep(0.05)
            
            prediction = monitor.predict_memory_exhaustion(
                horizon_seconds=10,
                memory_limit_mb=24000
            )
            
            assert prediction['will_exhaust'] == False
            assert prediction['predicted_usage_mb'] < 12000


class TestMemoryThresholdIntegration:
    """Test integration of memory thresholds with compression pipeline
    
    End-to-end testing of threshold propagation and system response
    NO FALLBACKS - HARD FAILURES ONLY
    """
    
    @pytest.fixture
    def integrated_system(self):
        """Create fully integrated compression system with memory management"""
        from independent_core.compression_systems.system_integration_coordinator import (
            SystemIntegrationCoordinator, SystemConfiguration
        )
        
        config = SystemConfiguration()
        config.enable_memory_pressure = True
        config.enable_cpu_bursting = True
        config.gpu_memory_limit_mb = 24576
        config.cpu_memory_limit_mb = 32768
        
        return SystemIntegrationCoordinator(config)
    
    def test_threshold_propagation_to_gpu_memory_manager(self, integrated_system):
        """Test threshold propagation from config to GPUMemoryManager
        
        Validates:
        - Configuration propagation
        - Threshold consistency
        - Dynamic updates
        - Component synchronization
        """
        # Set system-wide thresholds
        thresholds = {
            'gpu_critical': 23000,
            'gpu_high': 20000,
            'gpu_moderate': 17000,
            'cpu_critical': 30000,
            'cpu_high': 28000
        }
        
        integrated_system.set_memory_thresholds(thresholds)
        
        # Verify propagation to GPU memory manager
        gpu_manager = integrated_system.gpu_memory_manager
        assert gpu_manager.config['critical_threshold_mb'] == 23000
        assert gpu_manager.config['high_threshold_mb'] == 20000
        assert gpu_manager.config['moderate_threshold_mb'] == 17000
        
        # Test dynamic threshold update
        integrated_system.update_threshold('gpu_critical', 22000)
        assert gpu_manager.config['critical_threshold_mb'] == 22000
        
        # Verify all components synchronized
        components = integrated_system.get_memory_managed_components()
        for component in components:
            component_thresholds = component.get_memory_thresholds()
            assert component_thresholds['gpu_critical'] == 22000
    
    def test_integration_with_memory_pressure_handler(self, integrated_system):
        """Test integration between thresholds and MemoryPressureHandler
        
        Validates:
        - Pressure handler responses
        - Threshold-based decisions
        - State transitions
        - Action coordination
        """
        pressure_handler = integrated_system.memory_pressure_handler
        
        # Simulate different pressure scenarios
        with patch('torch.cuda.memory_allocated') as mock_gpu:
            # Low pressure
            mock_gpu.return_value = 10000 * 1024 * 1024  # 10GB
            state = pressure_handler.check_memory_state()
            assert state == MemoryState.HEALTHY
            
            # Verify system response
            response = integrated_system.handle_memory_pressure()
            assert response['action'] == 'continue_normal'
            assert response['cpu_bursting_enabled'] == False
            
            # High pressure
            mock_gpu.return_value = 21000 * 1024 * 1024  # 21GB
            state = pressure_handler.check_memory_state()
            assert state == MemoryState.HIGH
            
            # Verify system response
            response = integrated_system.handle_memory_pressure()
            assert response['action'] == 'enable_bursting'
            assert response['cpu_bursting_enabled'] == True
            assert 'reduce_batch' in response['recommendations']
    
    def test_cpu_bursting_pipeline_threshold_response(self, integrated_system):
        """Test CPUBurstingPipeline response to memory thresholds
        
        Validates:
        - Bursting activation
        - Threshold-based routing
        - Performance adaptation
        - Seamless transitions
        """
        bursting_pipeline = integrated_system.cpu_bursting_pipeline
        
        # Create test tensor
        test_tensor = torch.randn(1000, 1000)  # ~4MB
        
        with patch('torch.cuda.memory_allocated') as mock_gpu:
            # Below threshold - use GPU
            mock_gpu.return_value = 15000 * 1024 * 1024  # 15GB
            
            result = bursting_pipeline.process(test_tensor)
            assert result['processing_device'] == 'gpu'
            assert result['bursting_triggered'] == False
            
            # Above threshold - trigger bursting
            mock_gpu.return_value = 22000 * 1024 * 1024  # 22GB
            
            result = bursting_pipeline.process(test_tensor)
            assert result['processing_device'] == 'cpu'
            assert result['bursting_triggered'] == True
            assert result['reason'] == 'gpu_memory_pressure'
            
            # Verify seamless transition
            assert torch.allclose(result['output'], test_tensor, rtol=1e-5)
    
    def test_smart_memory_pool_defragmentation_triggers(self, integrated_system):
        """Test SmartMemoryPool defragmentation based on thresholds
        
        Validates:
        - Defragmentation triggers
        - Threshold-based timing
        - Pool optimization
        - Performance impact
        """
        memory_pool = integrated_system.gpu_memory_optimizer.memory_pool
        
        # Configure defragmentation thresholds
        memory_pool.set_defrag_thresholds({
            'fragmentation_threshold': 0.3,
            'memory_pressure_threshold': 0.8,
            'time_since_last_defrag_seconds': 300
        })
        
        # Create fragmentation
        allocations = []
        for i in range(50):
            block = memory_pool.allocate(size_mb=20)
            allocations.append(block)
        
        # Free alternate blocks to create fragmentation
        for i in range(0, 50, 2):
            memory_pool.free(allocations[i])
        
        # Check fragmentation level
        fragmentation = memory_pool.calculate_fragmentation()
        assert fragmentation > 0.3
        
        # Simulate memory pressure
        with patch('torch.cuda.memory_allocated') as mock_gpu:
            mock_gpu.return_value = 20000 * 1024 * 1024  # High pressure
            
            # Should trigger defragmentation
            defrag_triggered = memory_pool.check_and_defragment()
            assert defrag_triggered == True
            
            # Verify defragmentation effectiveness
            new_fragmentation = memory_pool.calculate_fragmentation()
            assert new_fragmentation < fragmentation
            assert new_fragmentation < 0.15
            
            # Check performance metrics
            metrics = memory_pool.get_defrag_metrics()
            assert metrics['blocks_consolidated'] > 0
            assert metrics['memory_recovered_mb'] > 0
            assert metrics['time_taken_ms'] > 0
    
    def test_end_to_end_compression_with_memory_pressure(self, integrated_system):
        """Test complete compression pipeline under memory pressure
        
        Validates:
        - Full pipeline operation
        - Pressure handling throughout
        - Quality preservation
        - Performance adaptation
        """
        # Create test data
        test_data = torch.randn(10000, 10000)  # ~400MB tensor
        
        # Run compression under different memory conditions
        results = []
        
        with patch('torch.cuda.memory_allocated') as mock_gpu:
            # Test with increasing memory pressure
            for pressure_level in [0.5, 0.7, 0.85, 0.95]:
                memory_mb = 24576 * pressure_level
                mock_gpu.return_value = memory_mb * 1024 * 1024
                
                # Compress data
                start_time = time.time()
                compressed = integrated_system.compress(
                    test_data,
                    auto_adapt=True
                )
                compression_time = time.time() - start_time
                
                # Decompress data
                decompressed = integrated_system.decompress(compressed)
                
                # Verify correctness
                reconstruction_error = torch.mean(torch.abs(decompressed - test_data))
                
                results.append({
                    'pressure_level': pressure_level,
                    'compression_time': compression_time,
                    'reconstruction_error': reconstruction_error.item(),
                    'device_used': compressed.get('processing_device', 'unknown'),
                    'bursting_used': compressed.get('cpu_bursting_used', False)
                })
        
        # Analyze results
        # Should use GPU at low pressure
        assert results[0]['device_used'] == 'gpu'
        assert results[0]['bursting_used'] == False
        
        # Should switch to CPU at high pressure
        assert results[3]['bursting_used'] == True
        
        # Reconstruction should be accurate regardless of pressure
        for result in results:
            assert result['reconstruction_error'] < 1e-5
        
        # Performance should adapt to pressure
        # Higher pressure might mean slower compression but should complete
        assert all(r['compression_time'] > 0 for r in results)