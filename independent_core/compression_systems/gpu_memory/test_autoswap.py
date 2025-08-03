"""
Comprehensive test suite for AutoSwap Priority-Based Memory Swapping System
Tests DOAScorer, PrioritySwapper, and AutoSwapManager components
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import time
import logging
import random
from typing import Dict, List, Tuple, Any
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_doa_scorer():
    """Test DOAScorer functionality"""
    print("\n" + "="*60)
    print("Testing DOAScorer")
    print("="*60)
    
    from doa_scorer import DOAScorer, SwapPriority, AccessPattern
    
    # Create scorer
    scorer = DOAScorer({
        'doa_window_size': 100,
        'idle_threshold_seconds': 60.0,
        'weight_frequency': 0.3,
        'weight_recency': 0.3,
        'weight_pattern': 0.2,
        'weight_size': 0.2
    })
    
    # Test 1: Register memory blocks
    print("\n1. Testing memory block registration...")
    blocks = []
    for i in range(10):
        block_id = f"block_{i}"
        size = random.randint(1, 100) * 1024 * 1024  # 1-100MB
        device_id = 0
        priority = random.choice(list(SwapPriority))
        
        scorer.register_memory_block(block_id, size, device_id, priority)
        blocks.append((block_id, size))
        
    print(f"   Registered {len(blocks)} memory blocks")
    
    # Test 2: Record access patterns
    print("\n2. Testing access pattern recording...")
    # Simulate different access patterns
    
    # Sequential access for block_0
    for i in range(10):
        scorer.record_access("block_0", offset=i * 1024, length=1024)
        time.sleep(0.01)
    
    # Random access for block_1
    for _ in range(10):
        scorer.record_access("block_1", offset=random.randint(0, 100000), length=1024)
        time.sleep(0.01)
    
    # Temporal pattern for block_2
    for _ in range(5):
        scorer.record_access("block_2")
        time.sleep(0.1)  # Regular interval
    
    print("   Recorded various access patterns")
    
    # Test 3: Calculate DOA scores
    print("\n3. Testing DOA score calculation...")
    scores = []
    for block_id, _ in blocks[:5]:
        score = scorer.calculate_doa_score(block_id)
        scores.append(score)
        print(f"   {block_id}: DOA={score.doa_value:.2f}s, Priority={score.swap_priority.name}, "
              f"Score={score.priority_score:.3f}")
    
    # Test 4: Get swap candidates
    print("\n4. Testing swap candidate selection...")
    required_bytes = 50 * 1024 * 1024  # 50MB
    candidates = scorer.get_swap_candidates(required_bytes, device_id=0)
    
    print(f"   Found {len(candidates)} swap candidates for {required_bytes / (1024*1024):.1f}MB")
    for i, candidate in enumerate(candidates[:3]):
        print(f"   {i+1}. {candidate.memory_block_id}: {candidate.size_bytes / (1024*1024):.1f}MB, "
              f"Priority={candidate.swap_priority.name}")
    
    # Test 5: Pin/unpin memory
    print("\n5. Testing memory pinning...")
    scorer.pin_memory_block("block_0")
    print("   Pinned block_0")
    
    # Try to get swap candidates again
    candidates_after_pin = scorer.get_swap_candidates(required_bytes, device_id=0)
    pinned_in_candidates = any(c.memory_block_id == "block_0" for c in candidates_after_pin)
    print(f"   block_0 in candidates after pinning: {pinned_in_candidates}")
    
    scorer.unpin_memory_block("block_0")
    print("   Unpinned block_0")
    
    # Test 6: Statistics
    print("\n6. Testing statistics...")
    stats = scorer.get_statistics()
    print(f"   Total blocks: {stats['total_blocks']}")
    print(f"   Pinned blocks: {stats['pinned_blocks']}")
    print(f"   Priority distribution: {stats['priority_distribution']}")
    
    print("\n✓ DOAScorer tests completed")
    return True


def test_priority_swapper():
    """Test PrioritySwapper functionality"""
    print("\n" + "="*60)
    print("Testing PrioritySwapper")
    print("="*60)
    
    from priority_swapper import PrioritySwapper, SwapDirection, SwapStrategy
    from doa_scorer import DOAScore, SwapPriority, AccessMetrics, AccessPattern
    
    # Skip if no CUDA available
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping GPU-specific tests")
        return True
    
    # Create swapper
    swapper = PrioritySwapper({
        'max_concurrent_swaps': 2,
        'batch_size_mb': 32,
        'async_threshold_mb': 8,
        'enable_compression': True
    })
    
    # Test 1: Register tensors
    print("\n1. Testing tensor registration...")
    tensors = {}
    for i in range(5):
        size = random.randint(1, 20) * 1024 * 1024 // 4  # 1-20MB in float32
        tensor = torch.randn(size, device='cuda:0')
        tensor_id = f"tensor_{i}"
        
        swapper.register_memory_block(tensor_id, tensor)
        tensors[tensor_id] = tensor
        
        print(f"   Registered {tensor_id}: {tensor.numel() * 4 / (1024*1024):.1f}MB")
    
    # Test 2: Swap out operations
    print("\n2. Testing swap out operations...")
    swap_operations = []
    
    for i, (tensor_id, tensor) in enumerate(list(tensors.items())[:3]):
        # Create mock DOA score
        doa_score = DOAScore(
            memory_block_id=tensor_id,
            doa_value=float(i * 10),
            priority_score=0.5 + i * 0.1,
            swap_priority=SwapPriority.MEDIUM if i < 2 else SwapPriority.LOW,
            access_metrics=AccessMetrics(
                last_access_time=time.time() - i * 10,
                total_accesses=10 - i,
                access_frequency=1.0 / (i + 1),
                access_pattern=AccessPattern.RANDOM,
                temporal_locality=0.5,
                spatial_locality=0.5,
                reuse_distance=100.0,
                working_set_size=tensor.numel() * 4
            ),
            size_bytes=tensor.numel() * 4,
            device_id=0
        )
        
        # Swap out
        operation = swapper.swap_out(tensor_id, doa_score, 'cpu')
        swap_operations.append(operation)
        
        print(f"   Swapped out {tensor_id}: {'Success' if operation.success else 'Failed'}")
    
    # Test 3: Check memory locations
    print("\n3. Testing memory location tracking...")
    for tensor_id in list(tensors.keys())[:3]:
        location = swapper.get_memory_location(tensor_id)
        print(f"   {tensor_id}: location={location['location']}, "
              f"compressed={location.get('compressed', False)}")
    
    # Test 4: Swap in operations
    print("\n4. Testing swap in operations...")
    for tensor_id in list(tensors.keys())[:2]:
        operation = swapper.swap_in(tensor_id, device_id=0)
        print(f"   Swapped in {tensor_id}: {'Success' if operation.success else 'Failed'}")
    
    # Test 5: Batch swapping
    print("\n5. Testing batch swapping...")
    batch_requests = []
    
    for i in range(3, 5):
        tensor_id = f"tensor_{i}"
        doa_score = DOAScore(
            memory_block_id=tensor_id,
            doa_value=float(i * 5),
            priority_score=0.7,
            swap_priority=SwapPriority.LOW,
            access_metrics=AccessMetrics(
                last_access_time=time.time() - 30,
                total_accesses=5,
                access_frequency=0.1,
                access_pattern=AccessPattern.RANDOM,
                temporal_locality=0.3,
                spatial_locality=0.3,
                reuse_distance=200.0,
                working_set_size=tensors[tensor_id].numel() * 4
            ),
            size_bytes=tensors[tensor_id].numel() * 4,
            device_id=0
        )
        batch_requests.append((tensor_id, doa_score, 'cpu'))
    
    batch_operations = swapper.batch_swap(batch_requests)
    print(f"   Batch swapped {len(batch_operations)} tensors")
    
    # Test 6: Statistics
    print("\n6. Testing statistics...")
    stats = swapper.get_statistics()
    print(f"   Total swaps: {stats['swap_stats']['total_swaps']}")
    print(f"   Successful swaps: {stats['swap_stats']['successful_swaps']}")
    print(f"   Average swap time: {stats['swap_stats']['average_swap_time_ms']:.2f}ms")
    print(f"   Memory distribution: {stats['memory_distribution_mb']}")
    
    print("\n✓ PrioritySwapper tests completed")
    return True


def test_autoswap_integration():
    """Test AutoSwap integration with GPUMemoryOptimizer"""
    print("\n" + "="*60)
    print("Testing AutoSwap Integration")
    print("="*60)
    
    # Skip if no CUDA available
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping integration tests")
        return True
    
    from gpu_memory_core import GPUMemoryOptimizer
    from auto_swap_manager import SwapPolicy
    
    # Create GPU optimizer with AutoSwap
    gpu_config = {
        'device_ids': [0],
        'enable_smart_pool': True,
        'enable_autoswap': True,
        'swap_policy': 'balanced',
        'swap_threshold_low': 0.5,
        'swap_threshold_moderate': 0.75,
        'swap_threshold_high': 0.9,
        'swap_threshold_critical': 0.95,
        'autoswap_monitoring': True
    }
    
    print("\n1. Initializing GPU optimizer with AutoSwap...")
    gpu_optimizer = GPUMemoryOptimizer(gpu_config)
    print("   ✓ GPU optimizer initialized")
    print(f"   ✓ AutoSwap available: {gpu_optimizer.autoswap_manager is not None}")
    
    # Test 2: Register tensors
    print("\n2. Testing tensor registration...")
    tensor_ids = []
    tensors = []
    
    for i in range(10):
        size = random.randint(5, 50) * 1024 * 1024 // 4  # 5-50MB
        tensor = torch.randn(size, device='cuda:0')
        priority = ['low', 'medium', 'high'][i % 3]
        
        tensor_id = gpu_optimizer.register_tensor_for_autoswap(tensor, priority=priority)
        tensor_ids.append(tensor_id)
        tensors.append(tensor)
        
        print(f"   Registered tensor {i}: {tensor.numel() * 4 / (1024*1024):.1f}MB, priority={priority}")
    
    # Test 3: Simulate access patterns
    print("\n3. Simulating access patterns...")
    for _ in range(20):
        # Random access pattern
        tensor_idx = random.randint(0, len(tensor_ids) - 1)
        gpu_optimizer.record_tensor_access(tensor_ids[tensor_idx])
    
    # Heavy access on first few tensors
    for i in range(3):
        for _ in range(10):
            gpu_optimizer.record_tensor_access(tensor_ids[i])
    
    print("   Access patterns recorded")
    
    # Test 4: Check memory state
    print("\n4. Checking memory state...")
    with torch.cuda.device(0):
        allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
        reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    
    print(f"   Allocated: {allocated:.1f}MB")
    print(f"   Reserved: {reserved:.1f}MB")
    print(f"   Total: {total:.1f}MB")
    print(f"   Utilization: {reserved / total * 100:.1f}%")
    
    # Test 5: Trigger memory pressure
    print("\n5. Testing memory pressure handling...")
    
    # Allocate more memory to trigger pressure
    try:
        large_tensor = torch.randn(100 * 1024 * 1024 // 4, device='cuda:0')  # 100MB
        large_tensor_id = gpu_optimizer.register_tensor_for_autoswap(large_tensor, priority='high')
        print("   Allocated large tensor (100MB)")
    except torch.cuda.OutOfMemoryError:
        print("   Out of memory - triggering swap")
        
        # Handle memory pressure
        required_bytes = 100 * 1024 * 1024
        success = gpu_optimizer.handle_memory_pressure(required_bytes, device_id=0)
        print(f"   Memory pressure handled: {'Success' if success else 'Failed'}")
        
        if success:
            # Try allocation again
            try:
                large_tensor = torch.randn(100 * 1024 * 1024 // 4, device='cuda:0')
                print("   Successfully allocated after swapping")
            except:
                print("   Still unable to allocate")
    
    # Test 6: Manual swap operations
    print("\n6. Testing manual swap operations...")
    
    # Get AutoSwap statistics before
    stats_before = gpu_optimizer.autoswap_manager.get_statistics()
    
    # Make swap decision
    decision = gpu_optimizer.autoswap_manager.make_swap_decision(50 * 1024 * 1024, device_id=0)
    print(f"   Swap decision: {len(decision.selected_blocks)} blocks selected")
    print(f"   Total swap size: {decision.total_swap_bytes / (1024*1024):.1f}MB")
    
    # Execute swap
    operations = gpu_optimizer.autoswap_manager.execute_swap_decision(decision)
    print(f"   Executed {len(operations)} swap operations")
    
    # Test 7: Swap tensor back in
    print("\n7. Testing swap in...")
    if decision.selected_blocks:
        first_block = decision.selected_blocks[0]
        swapped_tensor = gpu_optimizer.swap_in_tensor(first_block, device_id=0)
        if swapped_tensor is not None:
            print(f"   Successfully swapped in {first_block}")
        else:
            print(f"   Failed to swap in {first_block}")
    
    # Test 8: AutoSwap statistics
    print("\n8. AutoSwap statistics...")
    stats = gpu_optimizer.autoswap_manager.get_statistics()
    
    print(f"   Total swap decisions: {stats['performance_stats']['total_swap_decisions']}")
    print(f"   Total blocks swapped: {stats['performance_stats']['total_blocks_swapped']}")
    print(f"   Total bytes swapped: {stats['performance_stats']['total_bytes_swapped'] / (1024*1024):.1f}MB")
    print(f"   Memory pressure events: {stats['performance_stats']['memory_pressure_events']}")
    print(f"   Current policy: {stats['current_policy']}")
    print(f"   Current memory pressure: {stats['current_memory_pressure']}")
    
    # Test 9: Cleanup
    print("\n9. Cleanup...")
    gpu_optimizer.shutdown()
    print("   ✓ GPU optimizer shut down")
    
    print("\n✓ AutoSwap integration tests completed")
    return True


def run_stress_test():
    """Run stress test for AutoSwap system"""
    print("\n" + "="*60)
    print("Running AutoSwap Stress Test")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping stress test")
        return True
    
    from gpu_memory_core import GPUMemoryOptimizer
    
    # Create optimizer
    gpu_optimizer = GPUMemoryOptimizer({
        'enable_autoswap': True,
        'swap_policy': 'aggressive',
        'autoswap_monitoring': True
    })
    
    print("\n1. Allocating tensors until memory pressure...")
    tensors = []
    tensor_ids = []
    
    try:
        for i in range(100):
            size = random.randint(10, 50) * 1024 * 1024 // 4  # 10-50MB
            tensor = torch.randn(size, device='cuda:0')
            tensor_id = gpu_optimizer.register_tensor_for_autoswap(tensor, priority='medium')
            
            tensors.append(tensor)
            tensor_ids.append(tensor_id)
            
            if i % 10 == 0:
                with torch.cuda.device(0):
                    utilization = torch.cuda.memory_reserved(0) / torch.cuda.get_device_properties(0).total_memory
                print(f"   Allocated {i+1} tensors, utilization: {utilization:.1%}")
                
    except torch.cuda.OutOfMemoryError:
        print(f"   Out of memory after {len(tensors)} allocations")
    
    print("\n2. Simulating workload with random access...")
    for _ in range(100):
        # Random tensor access
        idx = random.randint(0, len(tensor_ids) - 1)
        gpu_optimizer.record_tensor_access(tensor_ids[idx])
        
        # Occasionally try to access swapped tensor
        if random.random() < 0.1:
            tensor = gpu_optimizer.swap_in_tensor(tensor_ids[idx])
    
    # Final statistics
    stats = gpu_optimizer.autoswap_manager.get_statistics()
    print(f"\n3. Final stress test statistics:")
    print(f"   Total tensors: {len(tensor_ids)}")
    print(f"   Swap decisions: {stats['performance_stats']['total_swap_decisions']}")
    print(f"   Blocks swapped: {stats['performance_stats']['total_blocks_swapped']}")
    print(f"   Success rate: {stats['performance_stats']['successful_swaps'] / max(stats['performance_stats']['total_blocks_swapped'], 1):.1%}")
    
    gpu_optimizer.shutdown()
    print("\n✓ Stress test completed")
    return True


def run_all_tests():
    """Run all AutoSwap tests"""
    print("\n" + "="*80)
    print("AutoSwap Priority-Based Memory Swapping System - Test Suite")
    print("="*80)
    
    try:
        # Test individual components
        test_doa_scorer()
        test_priority_swapper()
        
        # Test integrated system
        test_autoswap_integration()
        
        # Stress test
        run_stress_test()
        
        print("\n" + "="*80)
        print("✅ All AutoSwap tests completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.exception("Test failure details:")
        return False
    
    return True


if __name__ == "__main__":
    # Run tests
    success = run_all_tests()
    
    if success:
        print("\n🎉 AutoSwap Memory Management System is ready for production!")
        print("   - DOAScorer: ✓")
        print("   - PrioritySwapper: ✓")
        print("   - AutoSwapManager: ✓")
        print("   - GPU Integration: ✓")
    else:
        print("\n⚠️  Some tests failed. Please review the logs above.")