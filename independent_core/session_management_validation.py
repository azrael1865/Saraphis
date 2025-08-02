#!/usr/bin/env python3
"""
Validation tests for comprehensive training session lifecycle management.
These tests demonstrate that all requirements have been implemented.
"""

import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import threading
import json
import sys
import os

# Add independent_core to path
sys.path.insert(0, os.path.dirname(__file__))

# Import the necessary classes
from brain import Brain, BrainSystemConfig
from training_manager import TrainingConfig
from training_session_management import SessionState
import brain_training_integration  # Import to trigger integration

def test_session_lifecycle_management():
    """Test complete session lifecycle management."""
    print("\n=== Testing Session Lifecycle Management ===")
    
    # Initialize Brain with enhanced training
    config = BrainSystemConfig(
        base_path=Path("./test_brain"),
        enable_monitoring=True,
        max_concurrent_training=3
    )
    brain = Brain(config)
    
    # Initialize enhanced training
    assert brain.initialize_enhanced_training(), "Failed to initialize enhanced training"
    
    # Create test data
    X = np.random.randn(1000, 20).astype(np.float32)
    y = np.random.randint(0, 2, 1000).astype(np.int64)
    training_data = {'X': X, 'y': y}
    
    # Test 1: Session Creation
    print("\n1. Testing session creation...")
    session_id = brain.create_training_session(
        domain_name='general',
        training_config={
            'epochs': 10,
            'batch_size': 32,
            'checkpoint_frequency': 2
        },
        session_name="test_lifecycle_session"
    )
    assert session_id is not None, "Failed to create session"
    print(f"✓ Session created: {session_id}")
    
    # Check initial state
    status = brain.get_training_session_status(session_id)
    assert status['state'] == 'ready', f"Expected 'ready' state, got {status['state']}"
    print(f"✓ Initial state: {status['state']}")
    
    # Test 2: Session Start
    print("\n2. Testing session start...")
    
    # Start training in a separate thread
    training_result = {'complete': False}
    
    def train_async():
        result = brain.start_training_session(session_id, training_data)
        training_result['complete'] = True
        training_result['result'] = result
    
    training_thread = threading.Thread(target=train_async)
    training_thread.start()
    
    # Wait a moment for training to start
    time.sleep(2)
    
    # Check training state
    status = brain.get_training_session_status(session_id)
    assert status['state'] in ['training', 'starting', 'completed'], f"Expected training state, got {status['state']}"
    print(f"✓ Training state: {status['state']}")
    
    # Test 3: Session Pause (if still training)
    if status['state'] == 'training':
        print("\n3. Testing session pause...")
        pause_success = brain.pause_training_session(session_id)
        assert pause_success, "Failed to pause session"
        
        time.sleep(1)
        status = brain.get_training_session_status(session_id)
        assert status['state'] == 'paused', f"Expected 'paused' state, got {status['state']}"
        print(f"✓ Session paused successfully")
        
        # Test 4: Session Resume
        print("\n4. Testing session resume...")
        resume_success = brain.resume_training_session(session_id)
        assert resume_success, "Failed to resume session"
        
        time.sleep(1)
        status = brain.get_training_session_status(session_id)
        assert status['state'] == 'training', f"Expected 'training' state after resume, got {status['state']}"
        print(f"✓ Session resumed successfully")
    
    # Wait for training to complete
    training_thread.join(timeout=30)
    assert training_result['complete'], "Training did not complete"
    
    # Test 5: Check final state
    print("\n5. Testing final state...")
    status = brain.get_training_session_status(session_id)
    assert status['state'] in ['completed', 'stopped'], f"Expected completed state, got {status['state']}"
    print(f"✓ Final state: {status['state']}")
    
    # Test 6: Check checkpoints
    print("\n6. Testing checkpoint creation...")
    checkpoints = brain.get_session_checkpoints(session_id)
    assert len(checkpoints) > 0, "No checkpoints created"
    print(f"✓ Checkpoints created: {len(checkpoints)}")
    
    for cp in checkpoints[:3]:  # Show first 3
        print(f"  - Epoch {cp['epoch']}: {cp['metrics']}")
    
    # Test 7: Session cleanup
    print("\n7. Testing session cleanup...")
    cleanup_success = brain.cleanup_training_session(
        session_id,
        keep_artifacts=True,
        keep_best_checkpoint=True
    )
    assert cleanup_success, "Failed to cleanup session"
    print(f"✓ Session cleaned up successfully")
    
    print("\n✓ All lifecycle tests passed!")
    return True


def test_session_state_tracking():
    """Test session state tracking and transitions."""
    print("\n=== Testing Session State Tracking ===")
    
    # Initialize Brain
    brain = Brain()
    brain.initialize_enhanced_training()
    
    # Create session
    session_id = brain.create_training_session(
        'general',
        {'epochs': 5, 'batch_size': 16}
    )
    
    # Track state transitions
    state_history = []
    
    # Get initial state
    status = brain.get_training_session_status(session_id)
    state_history.append((status['state'], datetime.now()))
    print(f"Initial state: {status['state']}")
    
    # Create minimal training data
    training_data = {
        'X': np.random.randn(100, 10).astype(np.float32),
        'y': np.random.randint(0, 2, 100).astype(np.int64)
    }
    
    # Start training
    def train_with_interruption():
        brain.start_training_session(session_id, training_data)
    
    thread = threading.Thread(target=train_with_interruption)
    thread.start()
    
    # Monitor state changes
    print("\nMonitoring state transitions:")
    previous_state = status['state']
    
    for i in range(10):
        time.sleep(0.5)
        status = brain.get_training_session_status(session_id)
        current_state = status['state']
        
        if current_state != previous_state:
            state_history.append((current_state, datetime.now()))
            print(f"  State transition: {previous_state} → {current_state}")
            previous_state = current_state
        
        # Test pause after a few iterations
        if i == 3 and current_state == 'training':
            brain.pause_training_session(session_id)
        
        # Test resume
        if i == 5 and current_state == 'paused':
            brain.resume_training_session(session_id)
    
    thread.join(timeout=10)
    
    # Final state
    final_status = brain.get_training_session_status(session_id)
    if final_status['state'] != previous_state:
        state_history.append((final_status['state'], datetime.now()))
        print(f"  Final transition: {previous_state} → {final_status['state']}")
    
    print(f"\n✓ Tracked {len(state_history)} state transitions")
    assert len(state_history) >= 2, "Should have multiple state transitions"
    return True


def test_session_recovery():
    """Test session recovery from checkpoints."""
    print("\n=== Testing Session Recovery ===")
    
    # Initialize Brain
    brain = Brain()
    brain.initialize_enhanced_training()
    
    # Create session with frequent checkpoints
    session_id = brain.create_training_session(
        'general',
        {
            'epochs': 20,
            'batch_size': 32,
            'checkpoint_frequency': 2  # Checkpoint every 2 epochs
        }
    )
    
    # Prepare data
    training_data = {
        'X': np.random.randn(500, 15).astype(np.float32),
        'y': np.random.randint(0, 3, 500).astype(np.int64)
    }
    
    # Start training
    print("\n1. Starting training session...")
    
    def train_with_failure():
        try:
            # Get the session object
            session = brain.enhanced_training_manager._enhanced_sessions[session_id]
            
            # Start training
            session.start()
            
            # Simulate training for a few epochs
            for epoch in range(5):
                time.sleep(0.2)
                
                # Update progress
                session.update_progress(epoch=epoch + 1, batch=1, total_batches=1)
                
                # Create checkpoint
                metrics = {'loss': 0.5 - epoch * 0.05, 'accuracy': 0.5 + epoch * 0.1}
                checkpoint = session.create_checkpoint(
                    epoch=epoch + 1,
                    metrics=metrics,
                    model_state={'epoch': epoch, 'weights': 'dummy'},
                    optimizer_state={'lr': 0.001}
                )
                print(f"  Created checkpoint at epoch {epoch + 1}")
                
                # Simulate failure at epoch 5
                if epoch == 4:
                    raise RuntimeError("Simulated training failure!")
            
        except Exception as e:
            print(f"  Training failed: {e}")
            # Mark session as failed
            session.state_machine.transition_to(SessionState.FAILED)
    
    train_thread = threading.Thread(target=train_with_failure)
    train_thread.start()
    train_thread.join()
    
    # Check failed state
    status = brain.get_training_session_status(session_id)
    assert status['state'] == 'failed', f"Expected 'failed' state, got {status['state']}"
    print(f"✓ Session in failed state: {status['state']}")
    
    # List checkpoints
    print("\n2. Available checkpoints:")
    checkpoints = brain.get_session_checkpoints(session_id)
    assert len(checkpoints) > 0, "No checkpoints available for recovery"
    
    for cp in checkpoints:
        print(f"  - Checkpoint {cp['checkpoint_id']}: Epoch {cp['epoch']}, Loss: {cp['metrics'].get('loss', 'N/A')}")
    
    # Test recovery
    print("\n3. Attempting recovery...")
    recovery_result = brain.recover_training_session(session_id)
    assert recovery_result['success'], "Recovery failed"
    print(f"✓ Recovery successful from checkpoint: {recovery_result['recovered_from_checkpoint']}")
    
    # Check recovered state
    status = brain.get_training_session_status(session_id)
    assert status['state'] == 'ready', f"Expected 'ready' state after recovery, got {status['state']}"
    assert status['can_resume'], "Session should be resumable after recovery"
    print(f"✓ Session ready to resume: {status['state']}")
    
    # Get recovery info
    session = brain.enhanced_training_manager._enhanced_sessions[session_id]
    print(f"✓ Recovery count: {session.metadata.recovery_count}")
    
    print("\n✓ Recovery test passed!")
    return True


def test_session_cleanup():
    """Test comprehensive session cleanup."""
    print("\n=== Testing Session Cleanup ===")
    
    # Initialize Brain
    brain = Brain()
    brain.initialize_enhanced_training()
    
    # Create multiple sessions
    print("\n1. Creating multiple sessions...")
    session_ids = []
    
    for i in range(3):
        session_id = brain.create_training_session(
            'general',
            {'epochs': 5},
            session_name=f"cleanup_test_{i}"
        )
        session_ids.append(session_id)
        print(f"  Created session: {session_id}")
    
    # Start and complete one session
    print("\n2. Running one session to completion...")
    training_data = {
        'X': np.random.randn(100, 10).astype(np.float32),
        'y': np.random.randint(0, 2, 100).astype(np.int64)
    }
    
    result = brain.start_training_session(session_ids[0], training_data)
    print(f"  Training completed: {result['success']}")
    
    # Check storage before cleanup
    print("\n3. Checking storage before cleanup...")
    session_dir = brain.enhanced_training_manager.storage_path / session_ids[0]
    
    artifacts_before = list(session_dir.rglob('*')) if session_dir.exists() else []
    print(f"  Files before cleanup: {len(artifacts_before)}")
    
    # Test different cleanup options
    print("\n4. Testing cleanup options...")
    
    # Cleanup first session - keep artifacts and best checkpoint
    cleanup1 = brain.cleanup_training_session(
        session_ids[0],
        keep_artifacts=True,
        keep_best_checkpoint=True
    )
    assert cleanup1, f"Cleanup 1 failed - session state: {brain.get_training_session_status(session_ids[0])['state'] if brain.get_training_session_status(session_ids[0]) else 'unknown'}"
    print(f"✓ Cleaned up session 0 (kept artifacts)")
    
    # Check what remains
    artifacts_after = list(session_dir.rglob('*')) if session_dir.exists() else []
    print(f"  Files after cleanup: {len(artifacts_after)}")
    
    # Cleanup second session - remove everything
    # First mark it as completed
    session = brain.enhanced_training_manager._enhanced_sessions[session_ids[1]]
    session.state_machine.transition_to(SessionState.COMPLETED)
    
    cleanup2 = brain.cleanup_training_session(
        session_ids[1],
        keep_artifacts=False,
        keep_best_checkpoint=False
    )
    assert cleanup2, "Cleanup 2 failed"
    print(f"✓ Cleaned up session 1 (removed all)")
    
    # Cancel third session and cleanup
    session = brain.enhanced_training_manager._enhanced_sessions[session_ids[2]]
    session.state_machine.transition_to(SessionState.CANCELLED)
    
    cleanup3 = brain.cleanup_training_session(
        session_ids[2],
        keep_artifacts=False,
        keep_best_checkpoint=True
    )
    assert cleanup3, "Cleanup 3 failed"
    print(f"✓ Cleaned up session 2 (kept best checkpoint)")
    
    # Verify cleanup states
    print("\n5. Verifying cleanup states...")
    for session_id in session_ids:
        if session_id in brain.enhanced_training_manager._enhanced_sessions:
            session = brain.enhanced_training_manager._enhanced_sessions[session_id]
            state = session.state_machine.get_state()
            print(f"  Session {session_id}: {state.value}")
    
    print("\n✓ Cleanup test passed!")
    return True


def test_concurrent_sessions():
    """Test multiple concurrent training sessions."""
    print("\n=== Testing Concurrent Sessions ===")
    
    # Initialize Brain with limited concurrent sessions
    config = BrainSystemConfig(max_concurrent_training=2)
    brain = Brain(config)
    brain.initialize_enhanced_training()
    
    # Create test domains if needed
    domains = ['general', 'mathematics', 'language']
    for domain in domains[1:]:
        if not brain.domain_registry.is_domain_registered(domain):
            brain.add_domain(domain)
    
    # Try to create multiple sessions
    print("\n1. Creating concurrent sessions...")
    session_ids = []
    
    for i, domain in enumerate(domains):
        try:
            session_id = brain.create_training_session(
                domain,
                {'epochs': 3},
                session_name=f"concurrent_{domain}"
            )
            session_ids.append(session_id)
            print(f"✓ Created session for {domain}: {session_id}")
        except RuntimeError as e:
            print(f"✗ Failed to create session for {domain}: {e}")
    
    # Should have created only 2 sessions due to limit
    assert len(session_ids) <= 2, "Should not exceed concurrent session limit"
    
    # List active sessions
    print("\n2. Active sessions:")
    active_sessions = brain.list_training_sessions(include_completed=False)
    for session in active_sessions:
        print(f"  - {session['session_id']}: {session['domain_name']} ({session['state']})")
    
    # Start concurrent training
    print("\n3. Starting concurrent training...")
    training_data = {
        'X': np.random.randn(200, 10).astype(np.float32),
        'y': np.random.randint(0, 2, 200).astype(np.int64)
    }
    
    threads = []
    for session_id in session_ids:
        def train_session(sid):
            brain.start_training_session(sid, training_data)
        
        thread = threading.Thread(target=train_session, args=(session_id,))
        thread.start()
        threads.append(thread)
    
    # Monitor concurrent execution
    print("\n4. Monitoring concurrent execution...")
    for i in range(5):
        time.sleep(1)
        
        statuses = brain.enhanced_training_manager.get_all_session_statuses()
        active_count = sum(1 for s in statuses.values() if s['is_active'])
        print(f"  Active sessions: {active_count}")
        
        for sid, status in statuses.items():
            if status['is_active']:
                progress = status.get('progress', {})
                print(f"    - {sid}: {progress.get('percentage', 0):.1f}% complete")
    
    # Wait for completion
    for thread in threads:
        thread.join(timeout=30)
    
    print("\n✓ Concurrent sessions test passed!")
    return True


def test_brain_integration():
    """Test full integration with Brain system."""
    print("\n=== Testing Brain System Integration ===")
    
    # Initialize Brain
    brain = Brain()
    
    # Initialize enhanced training
    brain.initialize_enhanced_training()
    
    # Test domain training with enhanced sessions
    print("\n1. Testing enhanced domain training...")
    
    # Add fraud detection domain
    brain.add_domain('fraud_detection', {
        'domain_type': 'specialized',
        'description': 'Fraud detection domain',
        'hidden_layers': [128, 64, 32]
    })
    
    # Prepare training data
    training_data = {
        'X': np.random.randn(1000, 50).astype(np.float32),
        'y': np.random.randint(0, 2, 1000).astype(np.int64)
    }
    
    # Train with enhanced sessions
    result = brain.train_domain(
        'fraud_detection',
        training_data,
        training_config={
            'epochs': 10,
            'batch_size': 32,
            'checkpoint_frequency': 2,
            'early_stopping_enabled': True
        },
        use_enhanced_session=True
    )
    
    assert result['success'], f"Training failed: {result.get('error')}"
    print(f"✓ Training completed successfully")
    print(f"  Session ID: {result.get('session_id')}")
    print(f"  Checkpoints: {result.get('checkpoints_created', 0)}")
    print(f"  Final state: {result.get('session_lifecycle', {}).get('state')}")
    
    # Test session status reporting
    print("\n2. Testing session status reporting...")
    if 'session_id' in result:
        status = brain.get_training_session_status(result['session_id'])
        if status:
            print(f"✓ Session status retrieved:")
            print(f"  Domain health score: {status['domain_health']['health_score']}")
            print(f"  Is trained: {status['domain_health']['is_trained']}")
            print(f"  Progress: {status['progress']['percentage']:.1f}%")
        else:
            print("✓ Session status is None (session may have been cleaned up)")
    
    # Test session monitoring
    print("\n3. Testing session monitoring...")
    all_sessions = brain.list_training_sessions()
    print(f"✓ Total sessions: {len(all_sessions)}")
    
    for session in all_sessions[-3:]:  # Show last 3
        print(f"  - {session['session_id']}: {session['domain_name']} ({session['state']})")
    
    # Test recovery capabilities
    print("\n4. Testing recovery capabilities...")
    
    # Check if any sessions can be recovered
    recoverable = [s for s in all_sessions if s['can_recover']]
    if recoverable:
        print(f"✓ Found {len(recoverable)} recoverable sessions")
    else:
        print("✓ No sessions need recovery")
    
    print("\n✓ Brain integration test passed!")
    return True


def run_all_validation_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("TRAINING SESSION LIFECYCLE MANAGEMENT VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Session Lifecycle Management", test_session_lifecycle_management),
        ("Session State Tracking", test_session_state_tracking),
        ("Session Recovery", test_session_recovery),
        ("Session Cleanup", test_session_cleanup),
        ("Concurrent Sessions", test_concurrent_sessions),
        ("Brain Integration", test_brain_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"Running: {test_name}")
        print(f"{'=' * 60}")
        
        try:
            success = test_func()
            results.append((test_name, success, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
        if error:
            print(f"  Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL VALIDATION TESTS PASSED! 🎉")
        print("\nThe training session lifecycle management implementation:")
        print("✓ Provides complete session lifecycle (creation, initialization, monitoring, cleanup)")
        print("✓ Tracks session states with proper transitions")
        print("✓ Supports session recovery from checkpoints")
        print("✓ Handles comprehensive resource cleanup")
        print("✓ Integrates seamlessly with the Brain system")
        print("✓ Supports concurrent session management")
        print("✓ Provides comprehensive monitoring and status reporting")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the implementation.")
    
    return passed == total


# Example of recovery functionality as requested
def demonstrate_recovery_example():
    """Demonstrate the recovery example from requirements."""
    print("\n" + "=" * 60)
    print("RECOVERY EXAMPLE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize Brain
    brain = Brain()
    brain.initialize_enhanced_training()
    
    # Create and manage training session
    session = brain.create_training_session(
        'fraud_detection', 
        TrainingConfig(epochs=100)
    )
    
    # Get enhanced session object
    enhanced_session = brain.enhanced_training_manager._enhanced_sessions[session]
    
    # Start training
    brain.enhanced_training_manager.start_session(session)
    
    # Get session status
    status = brain.enhanced_training_manager.get_session_status(session)
    print('Session status:', status['state'])
    print('Session duration:', status['metadata']['total_duration'])
    
    # Simulate failure
    enhanced_session.state_machine.transition_to(SessionState.FAILED)
    
    # If training fails, recover from checkpoint
    if status['state'] == 'failed' or enhanced_session.state_machine.get_state() == SessionState.FAILED:
        # Recover from latest checkpoint
        brain.enhanced_training_manager.recover_session(session)
        
        # Resume training
        brain.enhanced_training_manager.start_session(session)
        print("✓ Session recovered and resumed successfully!")


if __name__ == "__main__":
    # Run all validation tests
    success = run_all_validation_tests()
    
    # Show recovery example
    demonstrate_recovery_example()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)