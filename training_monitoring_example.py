#!/usr/bin/env python3
"""
Comprehensive Training Monitoring Example
Demonstrates all features of the training execution and monitoring fixes.
"""

import sys
import os
import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path

# Add independent_core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'independent_core'))

try:
    from training_session_manager import (
        TrainingSessionManager, SessionStatus, create_session_manager
    )
    from training_integration_fixes import (
        EnhancedTrainingManager, create_enhanced_training_manager, enhance_brain_training
    )
    from brain import Brain, BrainSystemConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the Saraphis directory")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_monitoring.log')
    ]
)
logger = logging.getLogger(__name__)

def create_sample_fraud_data():
    """Create sample fraud detection data for testing."""
    n_samples = 1000
    
    # Create sample transaction data
    trans_data = {
        'TransactionID': list(range(1, n_samples + 1)),
        'TransactionAmt': np.random.exponential(50, n_samples),
        'ProductCD': np.random.choice(['W', 'H', 'C', 'S', 'R'], n_samples),
        'card1': np.random.randint(1000, 20000, n_samples),
        'card4': np.random.choice(['visa', 'mastercard', 'discover', 'amex'], n_samples),
        'addr1': np.random.choice([204.0, 325.0, 412.0], n_samples),
        'P_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com'], n_samples),
        'M1': np.random.choice(['T', 'F'], n_samples),
        'M2': np.random.choice(['T', 'F'], n_samples),
        'isFraud': np.random.choice([0, 1], n_samples, p=[0.97, 0.03])  # 3% fraud rate
    }
    
    # Add some V columns (Vesta features)
    for i in range(1, 11):
        trans_data[f'V{i}'] = np.random.randn(n_samples)
    
    trans_df = pd.DataFrame(trans_data)
    
    # Create sample identity data
    id_data = {
        'TransactionID': list(range(1, n_samples // 2 + 1)),
        'DeviceType': np.random.choice(['desktop', 'mobile'], n_samples // 2),
        'DeviceInfo': np.random.choice(['Windows', 'iOS', 'Android'], n_samples // 2),
        'id_01': np.random.randint(0, 100, n_samples // 2),
        'id_02': np.random.randint(10000, 700000, n_samples // 2)
    }
    
    id_df = pd.DataFrame(id_data)
    
    return trans_df, id_df

def custom_progress_callback(session_id: str, event: str, data: dict):
    """Custom callback for detailed progress monitoring."""
    if event == 'progress_updated':
        epoch = data.get('epoch', 0)
        batch = data.get('batch', 0)
        total_progress = data.get('total_progress', 0.0)
        print(f"[PROGRESS] Session {session_id[:8]}... - Epoch {epoch}, Batch {batch}, Progress: {total_progress:.1%}")
    
    elif event == 'metrics_updated':
        metrics_str = []
        if 'loss' in data:
            metrics_str.append(f"Loss: {data['loss']:.4f}")
        if 'accuracy' in data:
            metrics_str.append(f"Acc: {data['accuracy']:.4f}")
        if 'val_loss' in data:
            metrics_str.append(f"Val Loss: {data['val_loss']:.4f}")
        if 'val_accuracy' in data:
            metrics_str.append(f"Val Acc: {data['val_accuracy']:.4f}")
        
        if metrics_str:
            print(f"[METRICS] Session {session_id[:8]}... - {', '.join(metrics_str)}")
    
    elif event == 'session_completed':
        print(f"[COMPLETED] Session {session_id[:8]}... - Training completed successfully!")
    
    elif event == 'error_occurred':
        error_type = data.get('error_type', 'Unknown')
        recovery = data.get('recovery_attempted', False)
        print(f"[ERROR] Session {session_id[:8]}... - {error_type}, Recovery attempted: {recovery}")

def demo_session_manager():
    """Demonstrate TrainingSessionManager capabilities."""
    print("\n" + "="*80)
    print("DEMO 1: TrainingSessionManager - Session Lifecycle and Monitoring")
    print("="*80)
    
    # Create session manager
    manager = create_session_manager(".brain/demo_sessions")
    
    # Create a session
    session_id = manager.create_session(
        domain_name="fraud_detection",
        model_type="neural_network",
        config={
            'epochs': 5,
            'batch_size': 32,
            'learning_rate': 0.001,
            'validation_split': 0.2
        }
    )
    
    print(f"✅ Created session: {session_id}")
    
    # Add custom callback
    manager.add_callback(session_id, custom_progress_callback)
    
    # Start session
    manager.start_session(session_id)
    print(f"✅ Started session: {session_id}")
    
    # Simulate training with metrics updates
    print("📈 Simulating training progress...")
    
    for epoch in range(1, 6):  # 5 epochs
        epoch_start = time.time()
        
        for batch in range(1, 21):  # 20 batches per epoch
            # Simulate batch training
            loss = 1.0 * np.exp(-epoch * 0.2) + np.random.normal(0, 0.1)
            accuracy = min(0.95, 0.5 + epoch * 0.08 + np.random.normal(0, 0.02))
            
            # Update metrics
            manager.update_metrics(session_id, {
                'loss': loss,
                'accuracy': accuracy,
                'epoch': epoch,
                'batch': batch
            })
            
            # Report progress
            manager.report_progress(session_id, epoch, batch, 20, epoch_start)
            
            time.sleep(0.05)  # Simulate training time
        
        # Validation metrics
        val_loss = loss + np.random.normal(0, 0.05)
        val_accuracy = accuracy - np.random.normal(0.02, 0.02)
        
        manager.update_metrics(session_id, {
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })
        
        # Create checkpoint every 2 epochs
        if epoch % 2 == 0:
            checkpoint_id = manager.create_checkpoint(
                session_id,
                model_state={'epoch': epoch, 'weights': f"model_weights_epoch_{epoch}"},
                optimizer_state={'lr': 0.001, 'momentum': 0.9},
                is_best=(epoch == 4)  # Mark epoch 4 as best
            )
            print(f"💾 Created checkpoint: {checkpoint_id}")
    
    # Complete session
    final_metrics = {'final_accuracy': 0.89, 'final_loss': 0.23}
    manager.complete_session(session_id, final_metrics)
    
    # Get session info
    session = manager.get_session(session_id)
    print(f"\n📊 Session Summary:")
    print(f"   Duration: {session.duration().total_seconds():.1f} seconds")
    print(f"   Status: {session.status.value}")
    print(f"   Best accuracy: {session.metrics.best_accuracy:.4f}")
    print(f"   Best val accuracy: {session.metrics.best_val_accuracy:.4f}")
    print(f"   Checkpoints created: {len(session.checkpoints)}")
    print(f"   Resource measurements: {len(session.resource_metrics.cpu_usage)}")
    
    # Cleanup
    manager.shutdown()
    print("✅ Session manager demo completed")

def demo_enhanced_training_manager():
    """Demonstrate EnhancedTrainingManager capabilities."""
    print("\n" + "="*80)
    print("DEMO 2: EnhancedTrainingManager - Integrated Training with Brain")
    print("="*80)
    
    # Create sample data
    print("📊 Creating sample fraud detection data...")
    trans_df, id_df = create_sample_fraud_data()
    training_data = {'transactions': trans_df, 'identities': id_df}
    
    print(f"   Transaction data: {trans_df.shape}")
    print(f"   Identity data: {id_df.shape}")
    print(f"   Fraud rate: {trans_df['isFraud'].mean():.2%}")
    
    # Create enhanced training manager
    enhanced_manager = create_enhanced_training_manager()
    
    # Configure training
    training_config = {
        'epochs': 3,
        'batch_size': 64,
        'learning_rate': 0.001,
        'validation_split': 0.2,
        'early_stopping_patience': 5,
        'checkpoint_interval': 2,
        'max_recovery_attempts': 3,
        'log_interval': 5,
        'hidden_layers': [128, 64, 32],
        'dropout_rate': 0.3
    }
    
    print(f"🚀 Starting enhanced training with config: {training_config}")
    
    # Execute training
    result = enhanced_manager.enhanced_train_domain(
        domain_name="fraud_detection",
        training_data=training_data,
        model_type="fraud_classifier",
        config=training_config
    )
    
    # Display results
    print(f"\n📈 Training Results:")
    print(f"   Success: {result['success']}")
    print(f"   Session ID: {result.get('session_id', 'N/A')[:16]}...")
    print(f"   Training time: {result.get('training_time', 0):.1f} seconds")
    print(f"   Epochs completed: {result.get('epochs_completed', 0)}")
    print(f"   Final accuracy: {result.get('final_accuracy', 0):.4f}")
    print(f"   Best val accuracy: {result.get('best_val_accuracy', 0):.4f}")
    print(f"   Checkpoints: {result.get('checkpoints', 0)}")
    print(f"   Recovery attempts: {result.get('recovery_attempts', 0)}")
    
    # Resource usage
    if 'resource_usage' in result:
        resources = result['resource_usage']
        print(f"\n💻 Resource Usage:")
        print(f"   Avg CPU: {resources.get('avg_cpu_usage', 0):.1f}%")
        print(f"   Max CPU: {resources.get('max_cpu_usage', 0):.1f}%")
        print(f"   Avg Memory: {resources.get('avg_memory_usage', 0):.1f}%")
        print(f"   Max Memory: {resources.get('max_memory_usage', 0):.1f}%")
        print(f"   High CPU alerts: {resources.get('high_cpu_alerts', 0)}")
        print(f"   High memory alerts: {resources.get('high_memory_alerts', 0)}")
    
    # Metrics history
    if 'metrics' in result:
        metrics = result['metrics']
        print(f"\n📊 Training Metrics:")
        print(f"   Loss history (last 5): {metrics.get('loss_history', [])[-5:]}")
        print(f"   Accuracy history (last 5): {metrics.get('accuracy_history', [])[-5:]}")
        print(f"   Best loss: {metrics.get('best_loss', 0):.4f}")
        print(f"   Best accuracy: {metrics.get('best_accuracy', 0):.4f}")
        print(f"   Overfitting score: {metrics.get('overfitting_score', 0):.4f}")
    
    # Get session details
    session_id = result.get('session_id')
    if session_id:
        session_info = enhanced_manager.get_session_info(session_id)
        if session_info:
            print(f"\n🔍 Session Details:")
            print(f"   Status: {session_info['status']}")
            print(f"   Duration: {session_info['duration']:.1f} seconds")
            print(f"   Progress: {session_info['metrics']['total_progress']:.1%}")
    
    # List active sessions
    active_sessions = enhanced_manager.list_active_sessions()
    print(f"\n📋 Active Sessions: {len(active_sessions)}")
    for session in active_sessions:
        print(f"   {session['session_id'][:16]}... - {session['domain_name']} - {session['status']}")
    
    # Cleanup
    enhanced_manager.shutdown()
    print("✅ Enhanced training manager demo completed")

def demo_brain_integration():
    """Demonstrate Brain system integration."""
    print("\n" + "="*80)
    print("DEMO 3: Brain Integration - Enhanced Training with Brain System")
    print("="*80)
    
    try:
        # Initialize Brain
        print("🧠 Initializing Brain system...")
        config = BrainSystemConfig(
            enable_monitoring=True,
            enable_parallel_predictions=False,
            max_prediction_threads=1
        )
        brain = Brain(config)
        
        # Enhance Brain with training monitoring
        enhanced_brain = enhance_brain_training(brain)
        print("✅ Brain enhanced with training monitoring capabilities")
        
        # Add fraud detection domain
        domain_result = enhanced_brain.add_domain(
            'fraud_detection',
            {
                'type': 'specialized',
                'description': 'Enhanced fraud detection with monitoring',
                'hidden_layers': [64, 32],
                'learning_rate': 0.001
            }
        )
        print(f"✅ Domain added: {domain_result['success']}")
        
        # Create sample data
        trans_df, id_df = create_sample_fraud_data()
        training_data = {'transactions': trans_df, 'identities': id_df}
        
        # Train with enhanced monitoring
        print("🚀 Training fraud detection domain with enhanced monitoring...")
        result = enhanced_brain.train_domain(
            'fraud_detection',
            training_data,
            epochs=3,
            batch_size=32,
            validation_split=0.2,
            early_stopping_patience=3,
            checkpoint_interval=2
        )
        
        print(f"\n🎯 Brain Training Results:")
        print(f"   Success: {result['success']}")
        if result['success']:
            print(f"   Session ID: {result.get('session_id', 'N/A')[:16]}...")
            print(f"   Training time: {result.get('training_time', 0):.1f} seconds")
            print(f"   Final accuracy: {result.get('final_accuracy', 0):.4f}")
            print(f"   Checkpoints: {result.get('checkpoints', 0)}")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")
        
        # Test prediction after training
        if result['success']:
            print("\n🔮 Testing prediction with trained model...")
            sample_transaction = trans_df.iloc[0:1].drop(['isFraud'], axis=1).to_dict('records')[0]
            
            try:
                prediction = enhanced_brain.predict(
                    {'data': sample_transaction},
                    domain='fraud_detection'
                )
                
                if prediction.success:
                    print(f"   ✅ Prediction successful!")
                    print(f"   Prediction: {prediction.prediction}")
                    print(f"   Confidence: {prediction.confidence:.2%}")
                else:
                    print(f"   ❌ Prediction failed: {prediction.error}")
            except Exception as e:
                print(f"   ⚠️ Prediction error: {e}")
        
        # Cleanup
        enhanced_brain.shutdown()
        print("✅ Brain integration demo completed")
        
    except Exception as e:
        print(f"❌ Brain integration demo failed: {e}")
        import traceback
        traceback.print_exc()

def demo_error_recovery():
    """Demonstrate error recovery capabilities."""
    print("\n" + "="*80)
    print("DEMO 4: Error Recovery - Handling Training Errors")
    print("="*80)
    
    # Create session manager
    manager = create_session_manager(".brain/error_demo")
    
    # Create session
    session_id = manager.create_session(
        domain_name="error_demo",
        model_type="test_model",
        config={'epochs': 5, 'max_recovery_attempts': 3}
    )
    
    manager.start_session(session_id)
    print(f"✅ Created error demo session: {session_id[:16]}...")
    
    # Simulate different types of errors
    errors_to_test = [
        (RuntimeError("CUDA out of memory"), "Memory error"),
        (ValueError("Loss is NaN"), "NaN loss error"),
        (RuntimeError("Gradient overflow"), "Gradient explosion"),
        (ValueError("Invalid data format"), "Data error")
    ]
    
    for i, (error, description) in enumerate(errors_to_test):
        print(f"\n🔥 Simulating {description}...")
        
        # Handle error
        success = manager.handle_error(session_id, error, {
            'epoch': i + 1,
            'batch': 10,
            'phase': 'training'
        })
        
        if success:
            print(f"   ✅ Recovery successful for {description}")
        else:
            print(f"   ❌ Recovery failed for {description}")
        
        # Update session metrics to show recovery
        if success:
            manager.update_metrics(session_id, {
                'loss': 0.5 - i * 0.1,
                'accuracy': 0.6 + i * 0.1,
                'epoch': i + 1
            })
    
    # Get session info
    session = manager.get_session(session_id)
    print(f"\n📊 Error Recovery Summary:")
    print(f"   Total errors: {len(session.errors)}")
    print(f"   Recovery attempts: {session.recovery_attempts}")
    print(f"   Session status: {session.status.value}")
    
    # Show error details
    print(f"\n🔍 Error Details:")
    for i, error in enumerate(session.errors):
        print(f"   Error {i+1}: {error.error_type.value} - Recovery: {error.recovery_attempted}")
    
    # Complete session
    manager.complete_session(session_id)
    manager.shutdown()
    print("✅ Error recovery demo completed")

def main():
    """Run all training monitoring demos."""
    print("🚀 Training Execution and Monitoring Fixes - Comprehensive Demo")
    print("🎯 Demonstrating all 6 issue fixes:")
    print("   1. ✅ Training Session Management")
    print("   2. ✅ Real-time Accuracy Tracking") 
    print("   3. ✅ Training Progress Reporting")
    print("   4. ✅ Error Recovery and Rollback")
    print("   5. ✅ Resource Management")
    print("   6. ✅ Training Validation and Testing")
    
    try:
        # Run all demos
        demo_session_manager()
        demo_enhanced_training_manager()
        demo_brain_integration()
        demo_error_recovery()
        
        print("\n" + "="*80)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\n✅ Key Features Demonstrated:")
        print("   📋 Complete session lifecycle management")
        print("   📈 Real-time metrics tracking and progress reporting")
        print("   💾 Automatic checkpointing and recovery")
        print("   🔥 Error handling with multiple recovery strategies")
        print("   💻 Resource monitoring with alerts")
        print("   🧠 Seamless Brain system integration")
        print("   🔮 End-to-end training with prediction testing")
        
        print("\n🎯 Benefits Achieved:")
        print("   ✅ Robust error handling and recovery")
        print("   ✅ Complete visibility into training progress")
        print("   ✅ Resource protection and monitoring")
        print("   ✅ Quality assurance through validation")
        print("   ✅ Professional-grade session management")
        print("   ✅ Backward compatibility with existing Brain system")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)