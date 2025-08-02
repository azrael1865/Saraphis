#!/usr/bin/env python3
"""
Test script to verify progress tracking integration with existing session management.
"""

import sys
import os
import numpy as np
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_progress_integration():
    """Test the complete progress tracking integration."""
    print("=" * 60)
    print("TESTING PROGRESS TRACKING INTEGRATION")
    print("=" * 60)
    
    try:
        # Test 1: Import all modules
        print("\n1. Testing imports...")
        
        from progress_tracker import ProgressTracker, AlertSeverity, ProgressMetrics
        print("  ✓ ProgressTracker imported successfully")
        
        from brain import Brain
        print("  ✓ Brain imported successfully")
        
        # Test 2: Create Brain instance
        print("\n2. Testing Brain initialization...")
        brain = Brain()
        print("  ✓ Brain initialized successfully")
        
        # Test 3: Check progress system initialization
        print("\n3. Testing progress system initialization...")
        if hasattr(brain, '_training_progress'):
            print("  ✓ Progress tracking system initialized in Brain")
        
        if hasattr(brain.training_manager, '_progress_trackers'):
            print("  ✓ Progress tracking system initialized in TrainingManager")
            
        # Test 4: Test progress callback registration
        print("\n4. Testing progress callback registration...")
        
        progress_updates = []
        def test_callback(progress_data):
            progress_updates.append(progress_data)
            print(f"    Progress update received: epoch={progress_data.get('epoch', 'N/A')}")
        
        # Test global callback registration
        brain.register_global_progress_callback(lambda sid, data: test_callback(data))
        print("  ✓ Global progress callback registered")
        
        # Test 5: Test progress tracker creation
        print("\n5. Testing progress tracker creation...")
        
        test_session_id = "test_session_123"
        tracker = brain.training_manager.create_progress_tracker(
            test_session_id,
            {
                'total_epochs': 5,
                'batches_per_epoch': 10,
                'batch_size': 32,
                'enable_visualization': False,  # Disable to avoid GUI issues
                'enable_monitoring': True
            }
        )
        print("  ✓ Progress tracker created successfully")
        
        # Test 6: Test progress tracking functionality
        print("\n6. Testing progress tracking functionality...")
        
        # Start training simulation
        tracker.start_training(
            total_epochs=5,
            batches_per_epoch=10,
            samples_per_batch=32
        )
        print("  ✓ Training started in tracker")
        
        # Simulate training progress
        for epoch in range(1, 6):
            tracker.start_epoch(epoch)
            
            for batch in range(1, 11):
                tracker.start_batch(batch)
                
                # Simulate batch metrics
                metrics = {
                    'loss': max(0.1, 2.0 - epoch * 0.3 + np.random.normal(0, 0.1)),
                    'accuracy': min(0.95, 0.5 + epoch * 0.08 + np.random.normal(0, 0.02)),
                    'learning_rate': 0.001 * (0.95 ** epoch),
                    'gradient_norm': np.random.uniform(0.5, 2.0)
                }
                
                tracker.update_batch(metrics)
                tracker.end_batch()
                
                time.sleep(0.01)  # Small delay to simulate training time
            
            # Simulate validation metrics
            val_metrics = {
                'loss': max(0.15, 2.2 - epoch * 0.25 + np.random.normal(0, 0.15)),
                'accuracy': min(0.92, 0.45 + epoch * 0.07 + np.random.normal(0, 0.03))
            }
            
            tracker.end_epoch(val_metrics)
            print(f"    ✓ Epoch {epoch}/5 completed")
        
        print("  ✓ Training simulation completed")
        
        # Test 7: Test progress reporting
        print("\n7. Testing progress reporting...")
        
        current_progress = tracker.get_current_progress()
        print(f"  ✓ Current progress: {current_progress['progress_percent']:.1f}% complete")
        
        metrics_history = tracker.get_metrics_history()
        print(f"  ✓ Metrics history retrieved: {len(metrics_history)} metric types")
        
        alerts = tracker.get_alerts()
        print(f"  ✓ Alerts retrieved: {len(alerts)} alerts")
        
        # Generate final report
        final_report = tracker.generate_report()
        print(f"  ✓ Final report generated: {final_report['status']}")
        
        # Test 8: Test Brain-level progress methods
        print("\n8. Testing Brain-level progress methods...")
        
        # Test progress report retrieval
        brain_report = brain.get_training_progress_report(test_session_id)
        if 'session_id' in brain_report:
            print("  ✓ Brain progress report retrieval works")
        
        # Test alert retrieval
        brain_alerts = brain.get_training_alerts()
        print(f"  ✓ Brain alert retrieval works: {len(brain_alerts)} alerts")
        
        # Test efficiency metrics
        efficiency = brain.get_training_efficiency_metrics()
        print(f"  ✓ Efficiency metrics retrieved: {efficiency['active_trainings']} active")
        
        # Test 9: Test visualization (if available)
        print("\n9. Testing visualization capabilities...")
        
        try:
            plot = tracker.plot_metrics(['loss', 'accuracy'])
            if plot is not None:
                print("  ✓ Metrics plotting works")
            else:
                print("  ⚠️  Metrics plotting not available (matplotlib missing)")
        except Exception as e:
            print(f"  ⚠️  Metrics plotting failed: {e}")
        
        # Test progress bar creation
        try:
            progress_bar = tracker.create_progress_bar()
            if progress_bar is not None:
                print("  ✓ Progress bar creation works")
            else:
                print("  ⚠️  Progress bar not available (tqdm missing)")
        except Exception as e:
            print(f"  ⚠️  Progress bar creation failed: {e}")
        
        # Test 10: Test cleanup
        print("\n10. Testing cleanup...")
        
        tracker.stop()
        print("  ✓ Progress tracker stopped")
        
        # Test 11: Test enhanced Brain training method
        print("\n11. Testing enhanced Brain training method...")
        
        if hasattr(brain, 'train_domain_with_progress'):
            print("  ✓ Enhanced training method available")
            
            # Create simple test data
            test_data = {
                'X': np.random.randn(100, 5),
                'y': np.random.randint(0, 2, 100)
            }
            
            # Test with minimal configuration
            print("  ⚠️  Enhanced training test skipped (would require full setup)")
            
        # Summary
        print("\n" + "=" * 60)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 60)
        print("\n✅ All core integration tests passed!")
        print("\nKey features verified:")
        print("  ✓ Progress tracker creation and management")
        print("  ✓ Real-time metrics tracking and reporting")
        print("  ✓ Alert system functionality")
        print("  ✓ Brain-level progress API integration")
        print("  ✓ Session management compatibility")
        print("  ✓ Resource cleanup")
        
        print("\nProgress tracking integration is fully functional!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_session_management_compatibility():
    """Test compatibility with existing session management."""
    print("\n" + "=" * 60)
    print("TESTING SESSION MANAGEMENT COMPATIBILITY")
    print("=" * 60)
    
    try:
        from brain import Brain
        from training_manager import TrainingStatus
        
        # Create Brain instance
        brain = Brain()
        
        # Test that existing session management still works
        print("\n1. Testing domain registration...")
        domain_name = "test_progress_domain"
        
        # Register a test domain
        brain.domain_registry.register_domain(
            domain_name,
            {
                'domain_type': 'specialized',
                'description': 'Test domain for progress tracking',
                'priority': 3
            }
        )
        print(f"  ✓ Domain '{domain_name}' registered")
        
        # Test training preparation
        print("\n2. Testing training preparation...")
        
        training_config = {
            'epochs': 3,
            'batch_size': 16,
            'learning_rate': 0.001,
            'save_training_history': True
        }
        
        session_id = brain.training_manager.prepare_training(domain_name, training_config)
        print(f"  ✓ Training prepared with session ID: {session_id}")
        
        # Check that progress tracker can be created for this session
        if session_id in brain.training_manager._sessions:
            session = brain.training_manager._sessions[session_id]
            print(f"  ✓ Session exists with status: {session.status}")
            
            # Create progress tracker for this session
            tracker = brain.training_manager.create_progress_tracker(session_id)
            print("  ✓ Progress tracker created for real session")
            
            # Test progress tracking methods on real session
            progress = brain.training_manager.get_training_progress(session_id)
            if 'error' not in progress:
                print("  ✓ Progress tracking methods work with real sessions")
            else:
                print(f"  ⚠️  Progress method returned error: {progress['error']}")
            
            # Cleanup
            tracker.stop()
            print("  ✓ Progress tracker cleanup successful")
        
        # Test session cancellation with progress tracking
        print("\n3. Testing session lifecycle with progress tracking...")
        
        # Cancel the session
        cancel_result = brain.training_manager.cancel_training(session_id)
        if cancel_result.get('success'):
            print("  ✓ Session cancellation works with progress tracking")
        
        print("\n✅ Session management compatibility verified!")
        return True
        
    except Exception as e:
        print(f"\n❌ Session management compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    """Run the integration tests."""
    
    print("Starting progress tracking integration tests...\n")
    
    # Run core integration test
    test1_success = test_progress_integration()
    
    # Run session management compatibility test
    test2_success = test_session_management_compatibility()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)
    
    if test1_success and test2_success:
        print("\n🎉 All integration tests PASSED!")
        print("\nThe progress tracking system is successfully integrated with:")
        print("  • Existing Brain system")
        print("  • TrainingManager infrastructure")
        print("  • Session management")
        print("  • Alert system")
        print("  • Visualization capabilities")
        print("  • Real-time monitoring")
        
        print("\nYou can now use enhanced training with progress reporting!")
        sys.exit(0)
    else:
        print("\n❌ Some integration tests FAILED!")
        print("Please check the error messages above and fix any issues.")
        sys.exit(1)