"""
GAC System Integration Examples
Demonstrates how to integrate and use the GAC system with the Brain
"""

import asyncio
import logging
import torch
import json
from pathlib import Path
import sys
import os

# Add the parent directory to the path to import Brain and GAC system
sys.path.append(str(Path(__file__).parent.parent))

from brain import Brain, BrainSystemConfig
from gac_system.gradient_ascent_clipping import GACSystem, create_gac_system
from gac_system.gac_config import GACConfigManager
from gac_system.gac_components import (
    create_default_components, 
    create_production_components,
    GradientClippingComponent,
    GradientMonitoringComponent
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_example_config():
    """Create an example GAC configuration"""
    config_manager = GACConfigManager(auto_load=False)
    
    # Configure for development environment
    config_manager.create_preset_config("development")
    
    # Customize some settings
    config_manager.config.thresholds.gradient_magnitude = 2.0
    config_manager.config.monitoring.sampling_interval = 0.5
    config_manager.config.system.max_workers = 4
    
    return config_manager.config

async def basic_gac_integration_example():
    """Basic example of integrating GAC with Brain"""
    print("=" * 60)
    print("Basic GAC Integration Example")
    print("=" * 60)
    
    try:
        # Create Brain system
        brain_config = BrainSystemConfig(
            base_path=Path("./example_brain"),
            max_memory_gb=2.0,
            enable_monitoring=True
        )
        brain = Brain(brain_config)
        
        # Create and integrate GAC system
        gac_config = create_example_config()
        gac_system = GACSystem(gac_config.__dict__)
        
        # Add some example components
        default_components = create_default_components()
        for component in default_components:
            gac_system.register_component(component)
        
        # Integrate with Brain
        success = brain.integrate_gac_system(gac_system)
        print(f"GAC integration success: {success}")
        
        if success:
            # Get GAC status
            gac_status = brain.get_gac_status()
            print(f"GAC Status: {json.dumps(gac_status, indent=2)}")
            
            # Example gradient processing
            test_gradient = torch.randn(50, 50) * 3.0  # Large gradient for demonstration
            print(f"Original gradient norm: {torch.norm(test_gradient).item():.4f}")
            
            processed_gradient = await brain.process_gradient_with_gac(test_gradient)
            print(f"Processed gradient norm: {torch.norm(processed_gradient).item():.4f}")
            
            # Create checkpoint
            checkpoint_success = brain.create_gac_checkpoint("./example_checkpoints")
            print(f"Checkpoint creation success: {checkpoint_success}")
        
    except Exception as e:
        print(f"Error in basic example: {e}")
        import traceback
        traceback.print_exc()

async def advanced_gac_training_example():
    """Advanced example showing GAC integration during training"""
    print("=" * 60)
    print("Advanced GAC Training Integration Example")
    print("=" * 60)
    
    try:
        # Create Brain with GAC integration
        brain = Brain()
        
        # Create production-grade GAC system
        production_components = create_production_components()
        gac_system = create_gac_system()
        
        for component in production_components:
            gac_system.register_component(component)
        
        # Integrate with Brain
        brain.integrate_gac_system(gac_system)
        
        # Register custom training hooks
        def custom_pre_training_hook(domain_name, training_data, config):
            print(f"Custom pre-training hook executed for domain: {domain_name}")
        
        def custom_post_training_hook(domain_name, training_data, result):
            print(f"Custom post-training hook executed for domain: {domain_name}")
            print(f"Training success: {result.get('success', False)}")
        
        def custom_gradient_hook(gradient, context):
            print(f"Custom gradient hook - gradient norm: {torch.norm(gradient).item():.4f}")
        
        def custom_error_hook(error, *args):
            print(f"Custom error hook - error: {error}")
        
        # Register hooks with Brain
        brain.register_gac_hook('pre_training', custom_pre_training_hook)
        brain.register_gac_hook('post_training', custom_post_training_hook)
        brain.register_gac_hook('gradient_update', custom_gradient_hook)
        brain.register_gac_hook('error_callback', custom_error_hook)
        
        # Simulate training with example data
        example_training_data = {
            'input_data': torch.randn(100, 10),
            'target_data': torch.randint(0, 2, (100,))
        }
        
        # Train a domain (this will trigger the GAC hooks)
        training_result = brain.train_domain(
            domain_name='general',
            training_data=example_training_data,
            training_config={
                'epochs': 5,
                'batch_size': 16,
                'learning_rate': 0.01
            }
        )
        
        print(f"Training result: {json.dumps(training_result, indent=2)}")
        
        # Show final GAC system metrics
        final_status = brain.get_gac_status()
        print(f"Final GAC metrics: {json.dumps(final_status['system_metrics'], indent=2)}")
        
    except Exception as e:
        print(f"Error in advanced example: {e}")
        import traceback
        traceback.print_exc()

async def custom_component_example():
    """Example of creating and using custom GAC components"""
    print("=" * 60)
    print("Custom GAC Component Example")
    print("=" * 60)
    
    try:
        from gac_system.gradient_ascent_clipping import GACComponent, ComponentState, EventType
        
        class CustomGradientAnalyzer(GACComponent):
            """Custom component that analyzes gradient patterns"""
            
            def __init__(self, component_id: str = "custom_analyzer"):
                super().__init__(component_id)
                self.gradient_patterns = []
            
            async def process_gradient(self, gradient: torch.Tensor, context: dict) -> torch.Tensor:
                # Analyze gradient patterns
                gradient_norm = torch.norm(gradient).item()
                gradient_mean = torch.mean(gradient).item()
                gradient_std = torch.std(gradient).item()
                
                pattern = {
                    'norm': gradient_norm,
                    'mean': gradient_mean,
                    'std': gradient_std,
                    'sparsity': (gradient.abs() < 1e-6).float().mean().item()
                }
                
                self.gradient_patterns.append(pattern)
                
                # Emit analysis event
                self.emit_event(EventType.PERFORMANCE_METRIC, {
                    'analysis': pattern,
                    'pattern_count': len(self.gradient_patterns)
                })
                
                # Pass gradient through unchanged
                return gradient
            
            def get_component_info(self) -> dict:
                return {
                    'component_type': 'custom_analyzer',
                    'patterns_analyzed': len(self.gradient_patterns),
                    'avg_norm': sum(p['norm'] for p in self.gradient_patterns) / len(self.gradient_patterns) if self.gradient_patterns else 0,
                    'avg_sparsity': sum(p['sparsity'] for p in self.gradient_patterns) / len(self.gradient_patterns) if self.gradient_patterns else 0
                }
        
        # Create GAC system with custom component
        gac_system = create_gac_system()
        
        # Add custom component
        custom_analyzer = CustomGradientAnalyzer()
        gac_system.register_component(custom_analyzer, group="analysis")
        
        # Add some standard components
        clipper = GradientClippingComponent("custom_clipper", {"clip_value": 1.5})
        monitor = GradientMonitoringComponent("custom_monitor")
        
        gac_system.register_component(clipper, group="processing")
        gac_system.register_component(monitor, group="monitoring")
        
        # Start system
        gac_system.start_system()
        
        # Process some test gradients
        print("Processing test gradients...")
        for i in range(5):
            test_gradient = torch.randn(20, 20) * (i + 1)  # Increasing magnitude
            processed = await gac_system.process_gradient(test_gradient)
            print(f"Gradient {i+1}: {torch.norm(test_gradient).item():.4f} -> {torch.norm(processed).item():.4f}")
        
        # Show component status
        component_status = gac_system.get_component_status()
        for comp_id, status in component_status.items():
            print(f"Component {comp_id}: {status['info']}")
        
        # Show system metrics
        metrics = gac_system.get_system_metrics()
        print(f"System processed {metrics['performance_metrics']['total_gradients_processed']} gradients")
        
        gac_system.stop_system()
        
    except Exception as e:
        print(f"Error in custom component example: {e}")
        import traceback
        traceback.print_exc()

async def configuration_management_example():
    """Example of GAC configuration management"""
    print("=" * 60)
    print("GAC Configuration Management Example")
    print("=" * 60)
    
    try:
        from gac_system.gac_config import GACConfigManager
        
        # Create config manager
        config_manager = GACConfigManager("./example_gac_config.json", auto_load=False)
        
        # Apply different presets
        presets = ["development", "production", "conservative", "aggressive"]
        
        for preset in presets:
            print(f"\n--- {preset.upper()} PRESET ---")
            config_manager.create_preset_config(preset)
            
            # Validate configuration
            is_valid, errors = config_manager.validate_config()
            print(f"Configuration valid: {is_valid}")
            if errors:
                print(f"Errors: {errors}")
            
            # Show summary
            summary = config_manager.get_config_summary()
            print(f"Summary: {json.dumps(summary, indent=2)}")
            
            # Save configuration
            config_path = f"./example_config_{preset}.json"
            config_manager.config_path = Path(config_path)
            config_manager.save_config()
            print(f"Configuration saved to: {config_path}")
        
        print("\nConfiguration files created for different presets")
        
    except Exception as e:
        print(f"Error in configuration example: {e}")
        import traceback
        traceback.print_exc()

async def performance_monitoring_example():
    """Example of GAC performance monitoring and metrics"""
    print("=" * 60)
    print("GAC Performance Monitoring Example")
    print("=" * 60)
    
    try:
        # Create GAC system with monitoring
        gac_system = create_gac_system()
        
        # Add monitoring components
        monitor = GradientMonitoringComponent("performance_monitor", {
            'statistics_window': 50,
            'alert_thresholds': {
                'explosion_threshold': 5.0,
                'vanishing_threshold': 1e-5,
                'instability_threshold': 2.0
            }
        })
        
        clipper = GradientClippingComponent("adaptive_clipper", {
            'clip_value': 2.0,
            'adaptive_clipping': True
        })
        
        gac_system.register_component(monitor, group="monitoring")
        gac_system.register_component(clipper, group="processing")
        
        # Set up event monitoring
        events_captured = []
        
        def capture_events(event):
            events_captured.append({
                'type': event.event_type.value,
                'source': event.source_component,
                'data': event.data,
                'timestamp': event.timestamp
            })
        
        # Subscribe to different event types
        from gac_system.gradient_ascent_clipping import EventType
        gac_system.event_bus.subscribe(EventType.GRADIENT_UPDATE, capture_events)
        gac_system.event_bus.subscribe(EventType.SYSTEM_ALERT, capture_events)
        gac_system.event_bus.subscribe(EventType.PERFORMANCE_METRIC, capture_events)
        
        gac_system.start_system()
        
        # Simulate various gradient scenarios
        scenarios = [
            ("Normal gradients", lambda: torch.randn(30, 30) * 0.5),
            ("Large gradients", lambda: torch.randn(30, 30) * 5.0),
            ("Tiny gradients", lambda: torch.randn(30, 30) * 1e-7),
            ("Unstable gradients", lambda: torch.randn(30, 30) * torch.rand(1).item() * 10)
        ]
        
        for scenario_name, gradient_gen in scenarios:
            print(f"\n--- {scenario_name} ---")
            for i in range(10):
                gradient = gradient_gen()
                processed = await gac_system.process_gradient(gradient)
                
                if i % 3 == 0:  # Print every 3rd iteration
                    print(f"  Gradient {i+1}: {torch.norm(gradient).item():.6f} -> {torch.norm(processed).item():.6f}")
        
        # Show captured events
        print(f"\nCaptured {len(events_captured)} events")
        
        # Show events by type
        event_types = {}
        for event in events_captured:
            event_type = event['type']
            if event_type not in event_types:
                event_types[event_type] = 0
            event_types[event_type] += 1
        
        print("Event summary:")
        for event_type, count in event_types.items():
            print(f"  {event_type}: {count}")
        
        # Show system metrics
        final_metrics = gac_system.get_system_metrics()
        print(f"\nFinal system metrics:")
        print(f"  Total gradients processed: {final_metrics['performance_metrics']['total_gradients_processed']}")
        print(f"  Average processing time: {final_metrics['performance_metrics']['average_processing_time']:.6f}s")
        print(f"  Error count: {final_metrics['performance_metrics']['error_count']}")
        print(f"  System uptime: {final_metrics['uptime']:.2f}s")
        
        gac_system.stop_system()
        
    except Exception as e:
        print(f"Error in monitoring example: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run all examples"""
    print("GAC System Integration Examples")
    print("=" * 80)
    
    examples = [
        ("Basic Integration", basic_gac_integration_example),
        ("Advanced Training Integration", advanced_gac_training_example),
        ("Custom Components", custom_component_example),
        ("Configuration Management", configuration_management_example),
        ("Performance Monitoring", performance_monitoring_example)
    ]
    
    for name, example_func in examples:
        try:
            print(f"\n{'='*20} {name} {'='*20}")
            await example_func()
            print(f"{'='*20} {name} Complete {'='*20}\n")
        except Exception as e:
            print(f"Example '{name}' failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("All examples completed!")

if __name__ == "__main__":
    asyncio.run(main())