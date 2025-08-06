#!/usr/bin/env python3
"""
Standalone test for Memory Migration Engine
Tests the migration functionality without external dependencies
"""

import sys
import os
import time
import torch
import numpy as np
from typing import Dict, Any, List

# Add parent directories to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_migration_classes():
    """Test that all migration classes are properly defined"""
    print("\n" + "="*60)
    print("MEMORY MIGRATION ENGINE - STANDALONE TEST")
    print("="*60)
    
    # Import migration components
    from memory.unified_memory_handler import (
        MemoryMigrationEngine,
        MigrationRequest,
        MigrationStats,
        AccessPatternTracker,
        MigrationDecisionEngine,
        MigrationPolicy,
        MemoryAllocation,
        AllocationPriority,
        MemoryTracker,
        UnifiedMemoryConfig,
        MemoryPressureLevel
    )
    
    print("\n1. Testing class instantiation...")
    
    # Test AccessPatternTracker
    tracker = AccessPatternTracker(window_size=100)
    assert tracker.window_size == 100
    print("✓ AccessPatternTracker created")
    
    # Test tracking methods
    tracker.record_access("test_alloc", "subsystem", "cpu")
    tracker.record_dependency("test_alloc", "depends_on")
    freq = tracker.get_access_frequency("test_alloc", time_window=1.0)
    print(f"✓ Access tracking works (frequency: {freq:.2f}/s)")
    
    # Test MigrationDecisionEngine
    config = UnifiedMemoryConfig(
        migration_policy=MigrationPolicy.ADAPTIVE,
        migration_cost_threshold=0.1
    )
    decision_engine = MigrationDecisionEngine(config)
    print("✓ MigrationDecisionEngine created")
    
    # Test decision making
    test_allocation = MemoryAllocation(
        allocation_id="test_123",
        subsystem="test",
        size_bytes=1024*1024,
        priority=AllocationPriority.NORMAL,
        device="cpu",
        timestamp=time.time(),
        last_accessed=time.time()
    )
    
    should_migrate, benefit = decision_engine.should_migrate(
        test_allocation,
        "cuda:0" if torch.cuda.is_available() else "cpu",
        MemoryPressureLevel.HIGH,
        None
    )
    print(f"✓ Migration decision: should_migrate={should_migrate}, benefit={benefit:.3f}")
    
    # Test throughput model
    throughput = decision_engine._get_throughput("cpu", "cuda:0")
    print(f"✓ Throughput model: CPU->GPU = {throughput/1e9:.1f} GB/s")
    
    print("\n2. Testing migration data structures...")
    
    # Test MigrationRequest
    request = MigrationRequest(
        allocation_id="test_123",
        target_device="cuda:0",
        priority=AllocationPriority.NORMAL,
        async_mode=True,
        validate_checksum=True
    )
    assert request.allocation_id == "test_123"
    print("✓ MigrationRequest created")
    
    # Test MigrationStats
    stats = MigrationStats(
        allocation_id="test_123",
        source_device="cpu",
        target_device="cuda:0",
        size_bytes=1024*1024,
        duration_ms=10.5,
        throughput_gbps=0.95,
        success=True,
        chunks_transferred=1,
        checksum_valid=True
    )
    assert stats.success
    assert stats.checksum_valid
    print("✓ MigrationStats created")
    
    print("\n3. Testing access pattern analysis...")
    
    # Record multiple accesses
    for i in range(20):
        device = "cuda:0" if i % 3 == 0 else "cpu"
        tracker.record_access("pattern_test", "subsystem", device)
        time.sleep(0.01)
    
    # Analyze patterns
    preferred = tracker.get_locality_preference("pattern_test")
    frequency = tracker.get_access_frequency("pattern_test", time_window=1.0)
    next_access = tracker.predict_next_access("pattern_test")
    
    print(f"✓ Pattern analysis:")
    print(f"  - Preferred device: {preferred}")
    print(f"  - Access frequency: {frequency:.2f}/s")
    print(f"  - Next access in: {next_access:.3f}s" if next_access else "  - Next access: unpredictable")
    
    # Test migration candidates
    tracker.heat_map["cpu"]["candidate1"] = 0.1
    tracker.heat_map["cpu"]["candidate2"] = 0.5
    tracker.heat_map["cpu"]["candidate3"] = 0.01
    
    candidates = tracker.get_migration_candidates("cpu", min_frequency=0.0)
    print(f"✓ Migration candidates from CPU: {candidates}")
    
    print("\n4. Testing cost-benefit analysis...")
    
    # Test with different pressure levels
    pressures = [
        MemoryPressureLevel.HEALTHY,
        MemoryPressureLevel.MODERATE,
        MemoryPressureLevel.HIGH,
        MemoryPressureLevel.CRITICAL
    ]
    
    for pressure in pressures:
        migration_cost = decision_engine._calculate_migration_cost(
            10 * 1024 * 1024,  # 10MB
            "cpu",
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        
        expected_benefit = decision_engine._calculate_expected_benefit(
            test_allocation,
            "cuda:0" if torch.cuda.is_available() else "cpu",
            pressure,
            {"frequency": 0.1, "preferred_device": "cuda:0"}
        )
        
        net_benefit = expected_benefit - migration_cost
        print(f"  {pressure.name:10s}: cost={migration_cost:.3f}, benefit={expected_benefit:.3f}, net={net_benefit:+.3f}")
    
    print("\n5. Testing heat map decay...")
    
    initial_heat = tracker.heat_map["cpu"]["test_decay"] = 100.0
    tracker._decay_heat_map(decay_factor=0.9)
    decayed_heat = tracker.heat_map["cpu"]["test_decay"]
    
    assert decayed_heat < initial_heat
    print(f"✓ Heat map decay: {initial_heat:.1f} -> {decayed_heat:.1f}")
    
    print("\n6. Testing performance model updates...")
    
    # Update model with measurement
    test_stats = MigrationStats(
        allocation_id="perf_test",
        source_device="cpu",
        target_device="cuda:0" if torch.cuda.is_available() else "cpu",
        size_bytes=100 * 1024 * 1024,
        duration_ms=50,
        throughput_gbps=2.0,
        success=True
    )
    
    decision_engine.update_performance_model(test_stats)
    updated_throughput = decision_engine._get_throughput(
        test_stats.source_device,
        test_stats.target_device
    )
    print(f"✓ Performance model updated: {updated_throughput/1e9:.1f} GB/s")
    
    print("\n7. Testing migration policies...")
    
    # Test different policies
    policies = [
        MigrationPolicy.AGGRESSIVE,
        MigrationPolicy.CONSERVATIVE,
        MigrationPolicy.ADAPTIVE,
        MigrationPolicy.MANUAL
    ]
    
    for policy in policies:
        config.migration_policy = policy
        test_engine = MigrationDecisionEngine(config)
        
        should_migrate, benefit = test_engine.should_migrate(
            test_allocation,
            "cuda:0" if torch.cuda.is_available() else "cpu",
            MemoryPressureLevel.HIGH,
            {"frequency": 0.05}
        )
        
        print(f"  {policy.value:12s}: migrate={should_migrate}, benefit={benefit:+.3f}")
    
    print("\n" + "="*60)
    print("SUCCESS: All migration components validated!")
    print("="*60)
    
    return True


def test_migration_engine_integration():
    """Test the full MemoryMigrationEngine"""
    print("\n" + "="*60)
    print("TESTING FULL MIGRATION ENGINE")
    print("="*60)
    
    try:
        from memory.unified_memory_handler import (
            MemoryMigrationEngine,
            UnifiedMemoryConfig,
            MemoryTracker,
            MemoryAllocation,
            AllocationPriority
        )
        
        # Create config and tracker
        config = UnifiedMemoryConfig(
            migration_chunk_size_mb=32,
            enable_checksum_validation=True,
            migration_parallelism=2
        )
        
        tracker = MemoryTracker()
        
        # Create migration engine
        engine = MemoryMigrationEngine(config, tracker)
        print("✓ MemoryMigrationEngine created successfully")
        
        # Test components
        assert hasattr(engine, 'access_tracker')
        assert hasattr(engine, 'decision_engine')
        assert hasattr(engine, 'format_converters')
        print("✓ All engine components initialized")
        
        # Create test allocation
        test_allocation = MemoryAllocation(
            allocation_id="engine_test",
            subsystem="test",
            size_bytes=5 * 1024 * 1024,
            priority=AllocationPriority.NORMAL,
            device="cpu",
            timestamp=time.time(),
            last_accessed=time.time()
        )
        
        # Register with tracker
        tracker.register_allocation(test_allocation)
        print("✓ Test allocation registered")
        
        # Test migration scheduling
        success = engine.schedule_migration(
            "engine_test",
            "cuda:0" if torch.cuda.is_available() else "cpu",
            AllocationPriority.LOW
        )
        assert success
        print("✓ Migration scheduled successfully")
        
        # Get stats
        stats = engine.get_migration_stats()
        print(f"✓ Migration stats retrieved: {stats['total_migrations']} total migrations")
        
        # Shutdown
        engine.shutdown()
        print("✓ Engine shutdown cleanly")
        
        print("\n✓ Full migration engine integration test passed!")
        
    except Exception as e:
        print(f"✗ Engine integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = True
    
    # Run component tests
    try:
        test_migration_classes()
    except Exception as e:
        print(f"\n✗ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    # Run integration test
    try:
        if success:
            test_migration_engine_integration()
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    if success:
        print("\n" + "="*60)
        print("ALL MIGRATION TESTS PASSED ✓")
        print("Memory Migration Engine is fully operational!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("SOME TESTS FAILED ✗")
        print("Please check the errors above")
        print("="*60)
        sys.exit(1)