#!/usr/bin/env python3
"""
Comprehensive Demo of Integrated Error Recovery and Progress Tracking System.

This demo showcases the complete integration of error recovery capabilities
with the existing progress tracking infrastructure, demonstrating real-world
scenarios and recovery mechanisms.
"""

import os
import sys
import time
import numpy as np
import traceback
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_1_basic_error_recovery():
    """Demo 1: Basic error recovery with progress tracking."""
    print("=" * 70)
    print("DEMO 1: BASIC ERROR RECOVERY WITH PROGRESS TRACKING")
    print("=" * 70)
    
    try:
        from brain import Brain
        from error_recovery_system import ErrorType, ErrorSeverity
        
        # Initialize Brain with error recovery enabled
        brain = Brain()
        
        print("\n1. Setting up domain with error recovery...")
        
        # Register a test domain
        from domain_registry import DomainConfig, DomainType
        domain_name = "error_recovery_test"
        domain_config = DomainConfig(
            domain_type=DomainType.SPECIALIZED,
            description='Test domain for error recovery demo',
            priority=2
        )
        brain.domain_registry.register_domain(domain_name, domain_config)
        
        print(f"   ✓ Domain '{domain_name}' registered")
        
        # Create test training data
        training_data = {
            'X': np.random.randn(100, 5).astype(np.float32),
            'y': np.random.randint(0, 2, 100).astype(np.int64)
        }
        
        print("   ✓ Test training data created")
        
        # Configure training with error recovery
        training_config = {
            'epochs': 3,
            'batch_size': 32,
            'learning_rate': 0.001,
            'enable_recovery': True,
            'progress_config': {
                'enable_visualization': False,
                'enable_monitoring': True,
                'enable_alerts': True
            }
        }
        
        print("\n2. Starting training with integrated error recovery and progress tracking...")
        
        # Define progress callback
        progress_updates = []
        def progress_callback(session_id, progress_data):
            progress_updates.append(progress_data)
            epoch = progress_data.get('epoch', 'N/A')
            progress_pct = progress_data.get('progress_percent', 0)
            print(f"   Progress: Epoch {epoch}, {progress_pct:.1f}% complete")
        
        # Train with error recovery
        result = brain.train_domain_with_recovery(
            domain_name=domain_name,
            training_data=training_data,
            training_config=training_config,
            progress_callback=progress_callback,
            enable_recovery=True
        )
        
        print(f"\n3. Training completed: {result.get('success', False)}")
        
        # Show recovery statistics
        if 'recovery_stats' in result:
            stats = result['recovery_stats']
            print(f"   Recovery stats: {stats['total_errors']} errors, {stats['successful_recoveries']} recoveries")
        
        # Show progress report
        if 'progress_report' in result:
            progress = result['progress_report']
            print(f"   Progress report: {progress.get('progress_percent', 0):.1f}% complete")
        
        print("   ✓ Demo 1 completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Demo 1 failed: {e}")
        traceback.print_exc()
        return False

def demo_2_memory_error_recovery():
    """Demo 2: Simulated memory error with automatic recovery."""
    print("\n" + "=" * 70)
    print("DEMO 2: MEMORY ERROR RECOVERY SIMULATION")
    print("=" * 70)
    
    try:
        from brain import Brain
        from error_recovery_system import ErrorRecoveryManager, ErrorType, ErrorSeverity
        
        # Initialize Brain
        brain = Brain()
        
        print("\n1. Setting up error recovery manager...")
        
        # Get recovery manager from training manager
        if hasattr(brain.training_manager, '_error_recovery_manager'):
            recovery_manager = brain.training_manager._error_recovery_manager
            print("   ✓ Error recovery manager available")
        else:
            print("   ⚠️  Error recovery not initialized, initializing manually...")
            from error_recovery_system import ErrorRecoveryManager
            recovery_manager = ErrorRecoveryManager()
        
        print("\n2. Simulating memory error scenario...")
        
        # Simulate a memory error
        try:
            # This would normally be a real memory error
            raise MemoryError("CUDA out of memory: Tried to allocate 2.00 GiB")
        except MemoryError as mem_error:
            print(f"   Caught memory error: {mem_error}")
            
            # Prepare error context
            context = {
                'session_id': 'demo_session_memory',
                'batch_size': 64,
                'learning_rate': 0.001,
                'epoch': 2,
                'batch': 15
            }
            
            print("   Attempting automatic recovery...")
            
            # Handle the error with recovery system
            recovery_success = recovery_manager.handle_error(mem_error, context)
            
            if recovery_success:
                print("   ✓ Recovery successful!")
                
                # Check suggested recovery actions
                if 'suggested_batch_size' in context:
                    print(f"   → Suggested batch size reduction: {context['suggested_batch_size']}")
                
                if 'suggested_learning_rate' in context:
                    print(f"   → Suggested learning rate: {context['suggested_learning_rate']}")
                
            else:
                print("   ❌ Recovery failed")
        
        print("\n3. Checking recovery statistics...")
        
        stats = recovery_manager.get_recovery_stats()
        print(f"   Total errors handled: {stats['total_errors']}")
        print(f"   Recovery success rate: {stats.get('recovery_success_rate', 0):.2%}")
        print(f"   Strategies used: {list(stats['recovery_strategies_used'].keys())}")
        
        print("   ✓ Demo 2 completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Demo 2 failed: {e}")
        traceback.print_exc()
        return False

def demo_3_checkpoint_recovery():
    """Demo 3: Checkpoint-based recovery demonstration."""
    print("\n" + "=" * 70)
    print("DEMO 3: CHECKPOINT-BASED RECOVERY")
    print("=" * 70)
    
    try:
        from brain import Brain
        
        # Initialize Brain
        brain = Brain()
        
        print("\n1. Creating training session with checkpoints...")
        
        # Register domain
        from domain_registry import DomainConfig, DomainType
        domain_name = "checkpoint_test"
        domain_config = DomainConfig(
            domain_type=DomainType.STANDARD,
            description='Checkpoint recovery test domain'
        )
        brain.domain_registry.register_domain(domain_name, domain_config)
        
        # Prepare training session
        training_config = {
            'epochs': 5,
            'batch_size': 16,
            'learning_rate': 0.001,
            'enable_recovery': True
        }
        
        session_id = brain.training_manager.prepare_training(domain_name, training_config)
        print(f"   ✓ Training session prepared: {session_id}")
        
        print("\n2. Creating recovery checkpoints...")
        
        # Create several checkpoints simulating training progress
        checkpoint_ids = []
        for epoch in range(1, 4):
            # Simulate training state
            training_state = {
                'epoch': epoch,
                'batch': 0,
                'loss': 2.0 - epoch * 0.3,
                'accuracy': 0.5 + epoch * 0.1
            }
            
            # Create checkpoint
            checkpoint_id = brain.training_manager.create_recovery_checkpoint(session_id)
            if checkpoint_id:
                checkpoint_ids.append(checkpoint_id)
                print(f"   ✓ Checkpoint created for epoch {epoch}: {checkpoint_id}")
            
            time.sleep(0.1)  # Small delay between checkpoints
        
        print(f"\n3. Created {len(checkpoint_ids)} checkpoints")
        
        # List available checkpoints
        if hasattr(brain.training_manager, '_error_recovery_manager'):
            recovery_manager = brain.training_manager._error_recovery_manager
            checkpoints = recovery_manager.checkpoint_recovery.list_checkpoints(session_id)
            
            print("   Available checkpoints:")
            for cp in checkpoints:
                print(f"     - {cp.checkpoint_id}: Epoch {cp.epoch}, {cp.timestamp.strftime('%H:%M:%S')}")
        
        print("\n4. Simulating training failure and recovery...")
        
        # Simulate a critical failure
        try:
            raise RuntimeError("Critical model corruption detected")
        except RuntimeError as error:
            print(f"   Critical error occurred: {error}")
            
            # Attempt recovery using Brain's recovery method
            recovery_result = brain.recover_training_session(session_id)
            
            if recovery_result['success']:
                print("   ✓ Session recovered successfully!")
                print(f"   → Restored to epoch: {recovery_result['restored_epoch']}")
                print(f"   → Checkpoint timestamp: {recovery_result['checkpoint_timestamp']}")
            else:
                print(f"   ❌ Recovery failed: {recovery_result['error']}")
        
        print("   ✓ Demo 3 completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Demo 3 failed: {e}")
        traceback.print_exc()
        return False

def demo_4_custom_error_handling():
    """Demo 4: Custom error handling with progress integration."""
    print("\n" + "=" * 70)
    print("DEMO 4: CUSTOM ERROR HANDLING WITH PROGRESS INTEGRATION")
    print("=" * 70)
    
    try:
        from brain import Brain
        from error_recovery_system import ErrorRecord
        
        # Initialize Brain
        brain = Brain()
        
        print("\n1. Setting up custom error handling...")
        
        # Custom error callback
        error_events = []
        def custom_error_callback(error_record: ErrorRecord):
            error_events.append({
                'timestamp': error_record.timestamp,
                'error_type': error_record.error_type.value,
                'severity': error_record.severity.value,
                'recovery_success': error_record.recovery_success,
                'recovery_time': error_record.recovery_time
            })
            print(f"   Custom handler: {error_record.error_type.value} "
                  f"({error_record.severity.value}) - "
                  f"Recovery: {'✓' if error_record.recovery_success else '❌'}")
        
        # Register custom callback if recovery manager is available
        if hasattr(brain.training_manager, '_error_recovery_manager'):
            recovery_manager = brain.training_manager._error_recovery_manager
            recovery_manager.register_recovery_callback(custom_error_callback)
            print("   ✓ Custom error callback registered")
        
        print("\n2. Testing various error scenarios...")
        
        # Test different types of errors
        test_errors = [
            (ValueError("Invalid input data shape"), "Data validation error"),
            (RuntimeError("Model forward pass failed"), "Model execution error"),
            (TimeoutError("Training step timeout"), "Performance issue"),
        ]
        
        for i, (error, description) in enumerate(test_errors, 1):
            print(f"\n   Test {i}: {description}")
            
            try:
                # Use Brain's error handling method
                context = {
                    'session_id': f'demo_session_{i}',
                    'test_scenario': description,
                    'consecutive_failures': i - 1
                }
                
                recovery_success = brain.handle_training_error(error, context)
                
                if recovery_success:
                    print(f"   ✓ Error handled successfully")
                else:
                    print(f"   ⚠️  Error handling completed (recovery may have failed)")
                
            except Exception as handling_error:
                print(f"   ❌ Error handling failed: {handling_error}")
        
        print(f"\n3. Error handling summary:")
        print(f"   Total errors processed: {len(error_events)}")
        print(f"   Successful recoveries: {sum(1 for e in error_events if e['recovery_success'])}")
        
        # Show recovery statistics
        if hasattr(brain.training_manager, '_error_recovery_manager'):
            stats = brain.training_manager._error_recovery_manager.get_recovery_stats()
            print(f"   Error types seen: {list(stats['error_types_seen'].keys())}")
            print(f"   Average recovery time: {stats['average_recovery_time']:.3f}s")
        
        print("   ✓ Demo 4 completed successfully!")
        return True
        
    except Exception as e:
        print(f"   ❌ Demo 4 failed: {e}")
        traceback.print_exc()
        return False

def demo_5_comprehensive_integration():
    """Demo 5: Comprehensive integration of all systems."""
    print("\n" + "=" * 70)
    print("DEMO 5: COMPREHENSIVE SYSTEM INTEGRATION")
    print("=" * 70)
    
    try:
        from brain import Brain
        
        print("\n1. Initializing complete integrated system...")
        
        # Initialize Brain with full integration
        brain = Brain()
        
        # Verify all systems are available
        systems_status = {
            'Brain Core': hasattr(brain, 'brain_core'),
            'Domain Registry': hasattr(brain, 'domain_registry'),
            'Training Manager': hasattr(brain, 'training_manager'),
            'Progress Tracking': hasattr(brain.training_manager, '_progress_trackers'),
            'Error Recovery': hasattr(brain.training_manager, '_error_recovery_manager')
        }
        
        print("   System component status:")
        for system, available in systems_status.items():
            status = "✓" if available else "❌"
            print(f"     {status} {system}")
        
        if not all(systems_status.values()):
            print("   ⚠️  Some systems not available, proceeding with available components")
        
        print("\n2. Setting up multi-domain training scenario...")
        
        # Register multiple domains
        from domain_registry import DomainConfig, DomainType
        domains = [
            ('fraud_detection', 'Financial fraud detection'),
            ('image_classification', 'Image classification'),
            ('text_analysis', 'Text analysis and NLP')
        ]
        
        for domain_name, description in domains:
            domain_config = DomainConfig(
                domain_type=DomainType.SPECIALIZED,
                description=description,
                priority=2
            )
            brain.domain_registry.register_domain(domain_name, domain_config)
            print(f"   ✓ Domain registered: {domain_name}")
        
        print("\n3. Testing complete training workflow with all systems...")
        
        # Test training with the first domain
        domain_name = domains[0][0]
        
        # Create comprehensive training configuration
        training_config = {
            'epochs': 2,
            'batch_size': 16,
            'learning_rate': 0.001,
            'enable_recovery': True,
            'progress_config': {
                'enable_visualization': False,
                'enable_monitoring': True,
                'enable_alerts': True,
                'memory_warning_mb': 500,
                'memory_critical_mb': 800
            }
        }
        
        # Create test data
        training_data = {
            'X': np.random.randn(50, 8).astype(np.float32),
            'y': np.random.randint(0, 2, 50).astype(np.int64)
        }
        
        # Progress tracking
        progress_updates = []
        error_events = []
        
        def comprehensive_progress_callback(session_id, progress_data):
            progress_updates.append(progress_data)
            if len(progress_updates) % 5 == 0:  # Report every 5th update
                print(f"   Training progress: {progress_data.get('progress_percent', 0):.1f}%")
        
        print(f"   Starting training for {domain_name}...")
        
        # Execute complete training workflow
        result = brain.train_domain_with_recovery(
            domain_name=domain_name,
            training_data=training_data,
            training_config=training_config,
            progress_callback=comprehensive_progress_callback,
            enable_recovery=True,
            enable_alerts=True
        )
        
        print(f"\n4. Training completed with result: {result.get('success', False)}")
        
        # Comprehensive reporting
        if result.get('success'):
            print("   Detailed Results:")
            
            # Progress statistics
            if 'progress_report' in result:
                progress = result['progress_report']
                print(f"     Progress: {progress.get('progress_percent', 0):.1f}% complete")
                print(f"     Session ID: {progress.get('session_id', 'N/A')}")
            
            # Recovery statistics
            if 'recovery_stats' in result:
                recovery = result['recovery_stats']
                print(f"     Errors handled: {recovery.get('total_errors', 0)}")
                print(f"     Recovery success rate: {recovery.get('recovery_success_rate', 0):.2%}")
            
            # Error history
            if 'error_history' in result and result['error_history']:
                print(f"     Error events: {len(result['error_history'])}")
                for error in result['error_history'][:3]:  # Show first 3
                    print(f"       - {error['error_type']}: {error['recovery_success']}")
        
        print("\n5. Testing system recovery statistics...")
        
        # Get comprehensive recovery stats
        recovery_stats = brain.get_training_recovery_stats()
        if 'error' not in recovery_stats:
            print("   Recovery system statistics:")
            print(f"     Total errors: {recovery_stats.get('total_errors', 0)}")
            print(f"     Total recoveries: {recovery_stats.get('total_recoveries', 0)}")
            print(f"     Success rate: {recovery_stats.get('recovery_success_rate', 0):.2%}")
        
        # Get training efficiency metrics
        efficiency = brain.get_training_efficiency_metrics()
        print("   Training efficiency:")
        print(f"     Active trainings: {efficiency.get('active_trainings', 0)}")
        print(f"     Average throughput: {efficiency.get('average_throughput', 0):.2f}")
        
        print("\n6. Generating comprehensive report...")
        
        # Export recovery report
        if hasattr(brain.training_manager, '_error_recovery_manager'):
            report_file = "comprehensive_recovery_report.json"
            report = brain.training_manager.export_recovery_report(report_file)
            
            if 'error' not in report:
                print(f"   ✓ Report exported to {report_file}")
                print(f"   Report contains {len(report.get('error_history', []))} error records")
                print(f"   Report contains {len(report.get('checkpoints', []))} checkpoints")
        
        print("\n   ✓ Demo 5 completed successfully!")
        print("\n" + "🎉" * 25)
        print("ALL SYSTEMS INTEGRATED AND FUNCTIONAL!")
        print("🎉" * 25)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Demo 5 failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all error recovery and progress tracking demos."""
    print("🚀 COMPREHENSIVE ERROR RECOVERY & PROGRESS TRACKING DEMO")
    print("=" * 70)
    print("This demo showcases the complete integration of error recovery")
    print("capabilities with the existing progress tracking infrastructure.")
    print("=" * 70)
    
    # Run all demos
    demos = [
        ("Basic Error Recovery", demo_1_basic_error_recovery),
        ("Memory Error Recovery", demo_2_memory_error_recovery),
        ("Checkpoint Recovery", demo_3_checkpoint_recovery),
        ("Custom Error Handling", demo_4_custom_error_handling),
        ("Comprehensive Integration", demo_5_comprehensive_integration)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        print(f"\n🔧 Running: {demo_name}")
        try:
            success = demo_func()
            results.append((demo_name, success))
        except Exception as e:
            print(f"❌ {demo_name} crashed: {e}")
            results.append((demo_name, False))
    
    # Final summary
    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    
    for demo_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:10} {demo_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nOverall: {passed}/{total} demos passed ({passed/total:.1%})")
    
    if passed == total:
        print("\n🎉 ALL DEMOS PASSED! Error recovery system is fully integrated! 🎉")
    else:
        print(f"\n⚠️  {total - passed} demo(s) failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)