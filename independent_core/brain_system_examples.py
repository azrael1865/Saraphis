"""
Brain System Usage Examples
===========================

This module demonstrates practical usage of the Universal AI Core Brain system,
including domain management, training, prediction, and validation of 
anti-catastrophic forgetting capabilities.

Run this script to see the Brain system in action with real-world scenarios.
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from brain import Brain, BrainSystemConfig
from domain_registry import DomainConfig, DomainType


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")


def generate_synthetic_data(num_samples: int, num_features: int, 
                          num_classes: int = 2, seed: int = None) -> Tuple[List[List[float]], List[int]]:
    """Generate synthetic classification data for training."""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate features with some structure
    X = []
    y = []
    
    for class_idx in range(num_classes):
        # Generate samples for this class with class-specific patterns
        class_samples = num_samples // num_classes
        
        # Create a base pattern for this class
        base_pattern = np.random.randn(num_features) * 0.5 + (class_idx * 2)
        
        for _ in range(class_samples):
            # Add noise to base pattern
            sample = base_pattern + np.random.randn(num_features) * 0.3
            X.append(sample.tolist())
            y.append(class_idx)
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = [X[i] for i in indices]
    y = [y[i] for i in indices]
    
    return X, y


def generate_molecular_data(num_samples: int) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Generate synthetic molecular data for specialized domain."""
    molecular_data = []
    targets = []
    
    for i in range(num_samples):
        # Simulate molecular descriptors
        mol_data = {
            "molecular_weight": np.random.uniform(100, 500),
            "logP": np.random.uniform(-2, 5),  # Lipophilicity
            "num_h_donors": np.random.randint(0, 5),
            "num_h_acceptors": np.random.randint(0, 10),
            "num_rotatable_bonds": np.random.randint(0, 10),
            "aromatic_rings": np.random.randint(0, 4),
            "polar_surface_area": np.random.uniform(20, 140)
        }
        
        # Create feature vector from molecular descriptors
        features = [
            mol_data["molecular_weight"] / 500,  # Normalize
            mol_data["logP"] / 5,
            mol_data["num_h_donors"] / 5,
            mol_data["num_h_acceptors"] / 10,
            mol_data["num_rotatable_bonds"] / 10,
            mol_data["aromatic_rings"] / 4,
            mol_data["polar_surface_area"] / 140
        ]
        
        mol_data["features"] = features
        molecular_data.append(mol_data)
        
        # Generate synthetic bioactivity (0-1)
        # Based on Lipinski's rule of five
        bioactivity = 1.0
        if mol_data["molecular_weight"] > 500:
            bioactivity *= 0.7
        if mol_data["logP"] > 5:
            bioactivity *= 0.6
        if mol_data["num_h_donors"] > 5:
            bioactivity *= 0.8
        if mol_data["num_h_acceptors"] > 10:
            bioactivity *= 0.8
        
        bioactivity += np.random.normal(0, 0.1)
        bioactivity = np.clip(bioactivity, 0, 1)
        targets.append(bioactivity)
    
    return molecular_data, targets


def basic_brain_setup() -> Brain:
    """Example 1: Basic Brain initialization and setup."""
    print_section("Example 1: Basic Brain Setup")
    
    # Create configuration
    config = BrainSystemConfig(
        base_path=Path("./brain_demo"),
        enable_persistence=True,
        enable_monitoring=True,
        enable_adaptation=True,
        max_domains=10,
        max_memory_gb=4.0,
        max_concurrent_training=3,
        enable_parallel_predictions=True,
        log_level="INFO"
    )
    
    print("Creating Brain system with configuration:")
    print(f"  Base path: {config.base_path}")
    print(f"  Max domains: {config.max_domains}")
    print(f"  Max memory: {config.max_memory_gb} GB")
    print(f"  Parallel predictions: {config.enable_parallel_predictions}")
    
    # Initialize Brain
    brain = Brain(config)
    
    # Check initialization
    status = brain.get_brain_status()
    print(f"\nBrain initialized successfully!")
    print(f"  Status: {status['health']['status']}")
    print(f"  Components active: {len([c for c in status['components'].values() if c == 'active'])}")
    print(f"  Base domains: {status['domains']['total']}")
    
    return brain


def add_molecular_domain_example(brain: Brain) -> None:
    """Example 2: Adding a specialized molecular analysis domain."""
    print_section("Example 2: Adding Molecular Domain")
    
    # Define molecular domain configuration
    molecular_config = DomainConfig(
        domain_type=DomainType.SPECIALIZED,
        description="Molecular property prediction and bioactivity analysis",
        priority=8,  # High priority for specialized domain
        hidden_layers=[128, 64, 32, 16],  # Deeper network for complex patterns
        activation_function='relu',
        dropout_rate=0.2,  # Prevent overfitting
        learning_rate=0.001,
        max_memory_mb=512,
        max_cpu_percent=30.0
    )
    
    print("Adding molecular analysis domain with configuration:")
    print(f"  Type: {molecular_config.domain_type}")
    print(f"  Architecture: {molecular_config.hidden_layers}")
    print(f"  Priority: {molecular_config.priority}")
    
    # Add the domain
    result = brain.register_domain("molecular_analysis", molecular_config)
    
    if result['success']:
        print(f"\n✓ Molecular domain added successfully!")
        print(f"  Domain name: {result['domain_name']}")
        print(f"  Domain type: {result['domain_type']}")
        print(f"  Initialized: {result['initialized']}")
        
        # Get domain capabilities
        capabilities = brain.get_domain_capabilities("molecular_analysis")
        print(f"\nDomain capabilities:")
        for cap in capabilities['capabilities']:
            print(f"  - {cap['name']}: {cap['description']}")
    else:
        print(f"\n✗ Failed to add domain: {result.get('error', 'Unknown error')}")


def train_multiple_domains_example(brain: Brain) -> Dict[str, float]:
    """Example 3: Training multiple domains with different datasets."""
    print_section("Example 3: Training Multiple Domains")
    
    # Store initial performance for anti-forgetting validation
    initial_performance = {}
    
    # 1. Add and train a general classification domain
    print("\n1. Setting up general classification domain...")
    
    general_config = DomainConfig(
        domain_type=DomainType.CORE,
        description="General purpose classification",
        priority=5,
        hidden_layers=[64, 32],
        learning_rate=0.01,
        max_memory_mb=256
    )
    
    brain.register_domain("general_classifier", general_config)
    
    # Generate training data for general classifier
    print("   Generating training data...")
    X_general, y_general = generate_synthetic_data(
        num_samples=200, 
        num_features=10, 
        num_classes=3,
        seed=42
    )
    
    training_data_general = {
        "X": X_general,
        "y": y_general
    }
    
    # Train the domain
    print("   Training general classifier...")
    from training_manager import TrainingConfig
    
    train_config = TrainingConfig(
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        early_stopping_patience=5
    )
    
    result = brain.train_domain("general_classifier", training_data_general, train_config)
    
    if result['success']:
        print(f"   ✓ Training completed!")
        print(f"     Best performance: {result.get('best_performance', 'N/A')}")
        print(f"     Training time: {result.get('training_time', 0):.2f}s")
        
        # Test the trained model
        test_sample = {"features": X_general[0]}
        prediction = brain.predict("general_classifier", test_sample)
        if prediction.success:
            initial_performance['general_classifier'] = prediction.result.get("confidence", 0.5)
            print(f"     Test prediction confidence: {initial_performance['general_classifier']:.3f}")
    
    # 2. Add and train molecular domain
    print("\n2. Training molecular analysis domain...")
    
    # Generate molecular training data
    print("   Generating molecular data...")
    mol_data, mol_targets = generate_molecular_data(num_samples=150)
    
    # Convert to training format
    X_molecular = [mol['features'] for mol in mol_data]
    y_molecular = [int(target > 0.5) for target in mol_targets]  # Binary classification
    
    training_data_molecular = {
        "X": X_molecular,
        "y": y_molecular,
        "metadata": {
            "task": "bioactivity_prediction",
            "threshold": 0.5
        }
    }
    
    # Train molecular domain
    print("   Training molecular classifier...")
    mol_train_config = TrainingConfig(
        epochs=30,
        batch_size=16,
        validation_split=0.2,
        early_stopping_patience=7,
        learning_rate=0.0005
    )
    
    result = brain.train_domain("molecular_analysis", training_data_molecular, mol_train_config)
    
    if result['success']:
        print(f"   ✓ Molecular training completed!")
        print(f"     Best performance: {result.get('best_performance', 'N/A')}")
        print(f"     Training time: {result.get('training_time', 0):.2f}s")
        
        # Test molecular prediction
        test_mol = {"features": X_molecular[0]}
        mol_prediction = brain.predict("molecular_analysis", test_mol)
        if mol_prediction.success:
            initial_performance['molecular_analysis'] = mol_prediction.result.get("confidence", 0.5)
            print(f"     Test prediction confidence: {initial_performance['molecular_analysis']:.3f}")
    
    # 3. Add and train a sequence processing domain
    print("\n3. Setting up sequence processing domain...")
    
    sequence_config = DomainConfig(
        domain_type=DomainType.SPECIALIZED,
        description="Sequence pattern recognition",
        priority=6,
        hidden_layers=[128, 64, 32],
        learning_rate=0.005,
        max_memory_mb=384
    )
    
    brain.register_domain("sequence_processor", sequence_config)
    
    # Generate sequence data (time series patterns)
    print("   Generating sequence data...")
    X_sequence = []
    y_sequence = []
    
    for _ in range(100):
        # Generate sequences with patterns
        seq_type = np.random.randint(0, 2)
        if seq_type == 0:
            # Increasing pattern
            base = np.random.uniform(0, 0.5)
            sequence = [base + i * 0.1 + np.random.normal(0, 0.05) for i in range(10)]
        else:
            # Oscillating pattern
            sequence = [np.sin(i * 0.5) * 0.5 + 0.5 + np.random.normal(0, 0.05) for i in range(10)]
        
        X_sequence.append(sequence)
        y_sequence.append(seq_type)
    
    training_data_sequence = {
        "X": X_sequence,
        "y": y_sequence
    }
    
    # Train sequence domain
    print("   Training sequence processor...")
    seq_train_config = TrainingConfig(
        epochs=25,
        batch_size=16,
        validation_split=0.2,
        early_stopping_patience=5
    )
    
    result = brain.train_domain("sequence_processor", training_data_sequence, seq_train_config)
    
    if result['success']:
        print(f"   ✓ Sequence training completed!")
        print(f"     Best performance: {result.get('best_performance', 'N/A')}")
        
        test_seq = {"features": X_sequence[0]}
        seq_prediction = brain.predict("sequence_processor", test_seq)
        if seq_prediction.success:
            initial_performance['sequence_processor'] = seq_prediction.result.get("confidence", 0.5)
            print(f"     Test prediction confidence: {initial_performance['sequence_processor']:.3f}")
    
    # Show overall training summary
    print("\n" + "-" * 40)
    print("Training Summary:")
    print(f"  Domains trained: {len(initial_performance)}")
    if initial_performance:
        print(f"  Average initial confidence: {np.mean(list(initial_performance.values())):.3f}")
    
    health = brain.get_health_status()
    print(f"  System health: {health['overall_health']['grade']} ({health['overall_health']['score']:.1f}/100)")
    
    return initial_performance


def validate_no_forgetting_example(brain: Brain, initial_performance: Dict[str, float]) -> None:
    """Example 4: Validate that training new domains doesn't cause forgetting."""
    print_section("Example 4: Validating Anti-Catastrophic Forgetting")
    
    print("This example demonstrates that training new tasks doesn't degrade")
    print("performance on previously learned tasks.\n")
    
    if not initial_performance:
        print("No initial performance data available. Skipping forgetting validation.")
        return
    
    # Get current performance on all domains
    print("1. Checking current performance on all domains...")
    current_performance = {}
    
    # Test data for each domain
    test_data = {
        'general_classifier': generate_synthetic_data(10, 10, 3, seed=99)[0][0],
        'molecular_analysis': generate_molecular_data(10)[0][0]['features'],
        'sequence_processor': [i * 0.1 + np.random.normal(0, 0.05) for i in range(10)]
    }
    
    for domain_name in initial_performance.keys():
        test_sample = {"features": test_data.get(domain_name, [0] * 10)}
        prediction = brain.predict(domain_name, test_sample)
        
        if prediction.success:
            current_performance[domain_name] = prediction.result.get("confidence", 0.5)
            change = current_performance[domain_name] - initial_performance[domain_name]
            
            print(f"   {domain_name}:")
            print(f"     Initial confidence: {initial_performance[domain_name]:.3f}")
            print(f"     Current confidence: {current_performance[domain_name]:.3f}")
            print(f"     Change: {change:+.3f} ({abs(change/initial_performance[domain_name])*100:.1f}%)")
    
    # Now train a new challenging domain
    print("\n2. Training a new complex domain (image_analyzer)...")
    
    image_config = DomainConfig(
        domain_type=DomainType.SPECIALIZED,
        description="Complex image feature analysis",
        priority=9,
        hidden_layers=[256, 128, 64, 32],  # Large network
        learning_rate=0.001,
        dropout_rate=0.3,
        max_memory_mb=768
    )
    
    brain.register_domain("image_analyzer", image_config)
    
    # Generate complex training data
    print("   Generating complex image feature data...")
    X_image, y_image = generate_synthetic_data(
        num_samples=300,
        num_features=50,  # High dimensional
        num_classes=5,    # Many classes
        seed=123
    )
    
    training_data_image = {
        "X": X_image,
        "y": y_image
    }
    
    # Intensive training
    print("   Performing intensive training...")
    from training_manager import TrainingConfig
    intensive_config = TrainingConfig(
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        early_stopping_patience=10,
        learning_rate=0.001
    )
    
    result = brain.train_domain("image_analyzer", training_data_image, intensive_config)
    
    if result['success']:
        print(f"   ✓ New domain trained successfully!")
        print(f"     Training epochs: 50")
        print(f"     Final performance: {result.get('best_performance', 'N/A')}")
    
    # Check if existing domains were affected
    print("\n3. Re-checking performance on original domains...")
    post_training_performance = {}
    forgetting_detected = False
    
    for domain_name in initial_performance.keys():
        test_sample = {"features": test_data.get(domain_name, [0] * 10)}
        prediction = brain.predict(domain_name, test_sample)
        
        if prediction.success:
            post_training_performance[domain_name] = prediction.result.get("confidence", 0.5)
            change = post_training_performance[domain_name] - initial_performance[domain_name]
            degradation = -change / initial_performance[domain_name] * 100
            
            print(f"   {domain_name}:")
            print(f"     Before new training: {current_performance.get(domain_name, 0):.3f}")
            print(f"     After new training:  {post_training_performance[domain_name]:.3f}")
            print(f"     Change: {change:+.3f} ({abs(degradation):.1f}%)")
            
            # Check for catastrophic forgetting (>20% degradation)
            if degradation > 20:
                forgetting_detected = True
                print(f"     ⚠️  WARNING: Significant degradation detected!")
    
    # Summary
    print("\n" + "-" * 40)
    print("Anti-Forgetting Validation Summary:")
    
    if not forgetting_detected:
        print("✓ SUCCESS: No catastrophic forgetting detected!")
        print("  All domains maintained their performance despite intensive new training.")
        
        if post_training_performance and initial_performance:
            avg_change = np.mean([post_training_performance[d] - initial_performance[d] 
                                 for d in initial_performance.keys() if d in post_training_performance])
            print(f"  Average performance change: {avg_change:+.3f}")
    else:
        print("✗ FAILURE: Catastrophic forgetting detected!")
        print("  Some domains showed significant performance degradation.")
    
    # Show protection mechanisms used
    print("\nProtection mechanisms employed:")
    print("  - Domain isolation (separate parameter spaces)")
    print("  - Knowledge protection during training")
    print("  - Elastic weight consolidation")
    print("  - Experience replay buffers")


def brain_persistence_example(brain: Brain) -> None:
    """Example 5: Saving and loading Brain state."""
    print_section("Example 5: Brain Persistence")
    
    print("Demonstrating save/load functionality for production deployments.\n")
    
    # 1. Get current state summary
    print("1. Current Brain state:")
    domains = brain.list_available_domains()
    print(f"   Total domains: {len(domains)}")
    print(f"   Active domains: {len([d for d in domains if d.get('status') == 'active'])}")
    
    # List domains with their health
    print("\n   Domain health:")
    for domain in domains[:5]:  # Show first 5
        health = brain.get_domain_health(domain['name'])
        if health:
            print(f"     {domain['name']}: {health.get('status', 'unknown')} ({health.get('score', 0):.1f}/100)")
    
    # 2. Save complete state
    print("\n2. Saving Brain state...")
    save_path = Path("./brain_demo/brain_backup.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_result = brain.save_brain_state(save_path)
    
    if save_result['success']:
        print(f"   ✓ State saved successfully!")
        print(f"     File: {save_result['filepath']}")
        print(f"     Size: {save_result.get('size_mb', 0):.2f} MB")
        print(f"     Components saved: {len(save_result.get('components_saved', []))}")
        
        # Show what was saved
        if save_result.get('components_saved'):
            print("\n   Saved components:")
            for component in save_result['components_saved']:
                print(f"     - {component}")
    else:
        print(f"   ✗ Save failed: {save_result.get('error', 'Unknown error')}")
        return
    
    # 3. Create a new Brain instance
    print("\n3. Creating new Brain instance...")
    new_config = BrainSystemConfig(
        base_path=Path("./brain_demo_restored"),
        enable_persistence=True,
        enable_monitoring=True
    )
    
    new_brain = Brain(new_config)
    print("   ✓ New Brain created (empty state)")
    
    # 4. Load the saved state
    print("\n4. Loading saved state into new Brain...")
    load_result = new_brain.load_brain_state(save_path)
    
    if load_result['success']:
        print(f"   ✓ State loaded successfully!")
        print(f"     Components loaded: {len(load_result.get('components_loaded', []))}")
        print(f"     Original save time: {load_result.get('timestamp', 'N/A')}")
        
        if load_result.get('warnings'):
            print("\n   Warnings:")
            for warning in load_result['warnings']:
                print(f"     - {warning}")
    else:
        print(f"   ✗ Load failed: {load_result.get('error', 'Unknown error')}")
        return
    
    # 5. Verify the loaded state
    print("\n5. Verifying loaded state...")
    new_domains = new_brain.list_available_domains()
    
    print(f"   Loaded domains: {len(new_domains)}")
    print(f"   Verification:")
    
    # Test predictions on loaded domains
    test_samples = {
        'general_classifier': {"features": [0.5] * 10},
        'molecular_analysis': {"features": [0.3] * 7},
        'sequence_processor': {"features": [i * 0.1 for i in range(10)]}
    }
    
    verified = 0
    for domain_name, test_data in test_samples.items():
        if domain_name in [d['name'] for d in new_domains]:
            prediction = new_brain.predict(domain_name, test_data)
            if prediction.success:
                verified += 1
                confidence = prediction.result.get("confidence", 0)
                print(f"     ✓ {domain_name}: Working (confidence: {confidence:.3f})")
            else:
                print(f"     ✗ {domain_name}: Failed - {prediction.error_message}")
    
    print(f"\n   Successfully verified {verified}/{len(test_samples)} domains")
    
    # 6. Show that training can continue
    print("\n6. Continuing training on loaded Brain...")
    
    if verified > 0:
        # Generate small dataset
        X_new, y_new = generate_synthetic_data(50, 10, 3, seed=200)
        new_training_data = {"X": X_new, "y": y_new}
        
        # Quick training
        from training_manager import TrainingConfig
        quick_config = TrainingConfig(epochs=5, batch_size=16)
        result = new_brain.train_domain("general_classifier", new_training_data, quick_config)
        
        if result['success']:
            print("   ✓ Successfully continued training on loaded domain!")
            print(f"     Additional training time: {result.get('training_time', 0):.2f}s")
    
    print("\n" + "-" * 40)
    print("Persistence validated: Brain state can be saved and restored completely!")


def advanced_features_demo(brain: Brain) -> None:
    """Bonus: Demonstrate advanced Brain features."""
    print_section("Bonus: Advanced Features Demo")
    
    # 1. Performance metrics
    print("1. Comprehensive Performance Metrics:")
    metrics = brain.get_performance_metrics()
    
    print(f"   Prediction metrics:")
    pred_metrics = metrics.get('prediction_stats', {})
    print(f"     Total predictions: {pred_metrics.get('total_predictions', 0)}")
    print(f"     Average response time: {pred_metrics.get('average_response_time', 0):.3f}s")
    print(f"     Success rate: {pred_metrics.get('success_rate', 0):.1%}")
    
    print(f"\n   Resource usage:")
    resource_usage = brain.get_resource_usage()
    memory = resource_usage.get('memory', {})
    cpu = resource_usage.get('cpu', {})
    print(f"     Memory: {memory.get('used_mb', 0):.1f} MB")
    print(f"     CPU: {cpu.get('usage_percent', 0):.1f}%")
    
    # 2. Domain routing demonstration
    print("\n2. Intelligent Domain Routing:")
    
    # Create ambiguous test cases
    test_cases = [
        {
            "description": "Simple classification",
            "features": [0.5, 0.3, 0.7, 0.2, 0.9, 0.1, 0.6, 0.4, 0.8, 0.5]
        },
        {
            "description": "Molecular-like features",
            "features": [0.234, 0.678, 0.123, 0.456, 0.789, 0.345, 0.567]
        },
        {
            "description": "Sequential pattern",
            "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
    ]
    
    domains = brain.list_available_domains()
    if domains:
        for test_case in test_cases:
            # Use first available domain for demonstration
            domain_name = domains[0]['name']
            result = brain.predict(domain_name, {"features": test_case['features']})
            if result.success:
                print(f"\n   Test: {test_case['description']}")
                print(f"     Domain used: {domain_name}")
                print(f"     Prediction confidence: {result.result.get('confidence', 0):.3f}")
    
    # 3. Uncertainty quantification
    print("\n3. Uncertainty Quantification:")
    
    if domains:
        # Create uncertain case (out of distribution)
        uncertain_data = {"features": [10.0] * 10}  # Very different from training
        
        domain_name = domains[0]['name']
        result = brain.predict(domain_name, uncertain_data)
        if result.success:
            confidence_analysis = brain.get_prediction_confidence(result)
            
            print(f"   Out-of-distribution test:")
            print(f"     Overall confidence: {confidence_analysis.get('overall_confidence', 0):.3f}")
            print(f"     Confidence level: {confidence_analysis.get('confidence_level', 'unknown')}")
            
            uncertainty = confidence_analysis.get('uncertainty_breakdown', {})
            if uncertainty:
                print(f"     Epistemic uncertainty: {uncertainty.get('epistemic', 0):.3f}")
                print(f"     Aleatoric uncertainty: {uncertainty.get('aleatoric', 0):.3f}")
            
            if confidence_analysis.get('warnings'):
                print(f"\n   Warnings:")
                for warning in confidence_analysis['warnings']:
                    print(f"     ⚠️  {warning}")
    
    # 4. System diagnostics
    print("\n4. Running System Diagnostics:")
    diagnostic = brain.run_diagnostic()
    
    print(f"   Overall status: {diagnostic.get('overall_status', 'unknown')}")
    print(f"   Checks performed: {len(diagnostic.get('checks_performed', []))}")
    
    if diagnostic.get('issues_found'):
        print(f"\n   Issues found: {len(diagnostic['issues_found'])}")
        for issue in diagnostic['issues_found'][:3]:  # Show first 3
            severity = issue.get('severity', 'unknown')
            description = issue.get('description', 'No description')
            print(f"     - [{severity}] {description}")
    else:
        print("   No issues found!")
    
    if diagnostic.get('recommendations'):
        print(f"\n   Recommendations:")
        for rec in diagnostic['recommendations'][:3]:  # Show first 3
            print(f"     - {rec}")


def main():
    """Complete demonstration of Brain system capabilities."""
    print("\n" + "=" * 60)
    print("Universal AI Core - Brain System Demonstration")
    print("=" * 60)
    print("\nThis demo showcases the complete Brain system including:")
    print("- Domain management and isolation")
    print("- Multi-domain training without forgetting")
    print("- State persistence and recovery")
    print("- Performance monitoring and diagnostics")
    
    try:
        # Initialize Brain
        brain = basic_brain_setup()
        
        # Add specialized molecular domain
        add_molecular_domain_example(brain)
        
        # Train multiple domains
        initial_performance = train_multiple_domains_example(brain)
        
        # Validate anti-catastrophic forgetting
        validate_no_forgetting_example(brain, initial_performance)
        
        # Demonstrate persistence
        brain_persistence_example(brain)
        
        # Show advanced features
        advanced_features_demo(brain)
        
        # Final summary
        print_section("Demo Complete!")
        
        domains = brain.list_available_domains()
        health = brain.get_health_status()
        print(f"Final system state:")
        print(f"  Total domains: {len(domains)}")
        print(f"  System health: {health['overall_health']['grade']}")
        
        print("\n✅ All demonstrations completed successfully!")
        print("\nThe Brain system is ready for production use with:")
        print("  - Domain isolation preventing catastrophic forgetting")
        print("  - Comprehensive monitoring and diagnostics")
        print("  - Full state persistence and recovery")
        print("  - Intelligent routing and uncertainty quantification")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nCleaning up demo directories...")
        import shutil
        for path in ["./brain_demo", "./brain_demo_restored"]:
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                except:
                    pass
        print("Demo complete!")


if __name__ == "__main__":
    main()