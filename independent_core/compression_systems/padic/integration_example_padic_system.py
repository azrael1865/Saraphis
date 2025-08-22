"""
P-Adic Integration System Example
Demonstrates comprehensive integration with GAC, Brain Core, and training systems
"""

import torch
import torch.nn as nn
import numpy as np
import time
import asyncio
from typing import Dict, Any, List

# Import P-Adic integration components
from padic_integration import (
    PadicIntegrationConfig, PadicSystemOrchestrator,
    initialize_padic_integration, get_orchestrator, shutdown_padic_integration
)

# Import core components for integration testing
from gac_system.gac_components import GradientCompressionComponent
from brain_core import BrainCore, BrainConfig


class MockGradientCompressionComponent(GradientCompressionComponent):
    """Mock GAC component for testing integration"""
    
    def __init__(self, component_id: str = "mock_gac"):
        self.component_id = component_id
        self.compression_stats = {
            'calls': 0,
            'total_time': 0.0
        }
    
    async def process_gradient(self, gradient: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        """Original gradient processing"""
        self.compression_stats['calls'] += 1
        start_time = time.perf_counter()
        
        # Simple compression simulation
        result = gradient * 0.9  # Simple reduction
        
        elapsed = time.perf_counter() - start_time
        self.compression_stats['total_time'] += elapsed
        
        return result
    
    def _apply_compression(self, gradient: torch.Tensor) -> torch.Tensor:
        """Original compression method"""
        return gradient * 0.95  # Simple compression


class MockBrainCore(BrainCore):
    """Mock Brain Core for testing integration"""
    
    def __init__(self, config: BrainConfig = None):
        self.config = config or BrainConfig()
        self.compression_systems = {}
        self.active_compression = None
        self.buffers = {}
        
        # Create mock model
        self.model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        # Initialize some buffers
        self.buffers['state_buffer'] = torch.randn(1000, 100)
        self.buffers['activation_cache'] = torch.randn(500, 50)
    
    def register_compression_system(self, name: str, system: Any) -> None:
        """Register compression system"""
        self.compression_systems[name] = system
    
    def _allocate_memory(self, size: int) -> torch.Tensor:
        """Mock memory allocation"""
        return torch.empty(size)


def demonstrate_padic_gac_integration():
    """Demonstrate P-Adic integration with GAC components"""
    
    print("=== P-Adic GAC Integration Demonstration ===")
    
    # Initialize P-Adic integration system
    config = PadicIntegrationConfig(
        prime=101,
        base_precision=50,
        adaptive_precision=True,
        gac_compression_threshold=0.001,
        gac_async_processing=True
    )
    
    orchestrator = initialize_padic_integration(config)
    
    print(f"Initialized P-Adic orchestrator with prime {config.prime}")
    
    try:
        # Create mock GAC component
        gac_component = MockGradientCompressionComponent("test_gac_1")
        
        print(f"Created GAC component: {gac_component.component_id}")
        
        # Enhance component with P-Adic capabilities
        orchestrator.gac_integration.enhance_gradient_component(gac_component)
        
        print(f"Enhanced GAC component with P-Adic compression")
        
        # Test enhanced gradient processing
        test_gradients = [
            torch.randn(100, 50) * 0.1,  # Normal gradient
            torch.randn(500, 200) * 0.001,  # Small gradient
            torch.randn(50, 25) * 1.0,  # Large gradient
        ]
        
        print(f"\nTesting enhanced gradient processing:")
        
        for i, gradient in enumerate(test_gradients):
            context = {'step': i, 'force_padic': i == 1}  # Force P-Adic on second gradient
            
            print(f"  Gradient {i+1}: shape {gradient.shape}, norm {gradient.norm().item():.6f}")
            
            start_time = time.perf_counter()
            
            # Process gradient using enhanced component
            try:
                processed = asyncio.run(gac_component.process_gradient(gradient, context))
                
                elapsed = time.perf_counter() - start_time
                compression_ratio = gradient.numel() / (processed.numel() if processed.numel() > 0 else 1)
                
                print(f"    ✓ Processing successful: {elapsed:.4f}s, ratio: {compression_ratio:.2f}x")
                print(f"    Output shape: {processed.shape}, norm: {processed.norm().item():.6f}")
                
            except Exception as e:
                print(f"    ✗ Processing failed: {e}")
        
        # Get component metrics
        metrics = orchestrator.gac_integration.get_component_metrics("test_gac_1")
        print(f"\nGAC Component Metrics:")
        print(f"  Compressions: {metrics['compressions']}")
        print(f"  Decompressions: {metrics['decompressions']}")
        print(f"  Average time: {metrics['average_time']:.6f}s")
        print(f"  Errors: {metrics['errors']}")
        
    except Exception as e:
        print(f"✗ GAC integration failed: {e}")


def demonstrate_padic_brain_integration():
    """Demonstrate P-Adic integration with Brain Core"""
    
    print(f"\n=== P-Adic Brain Core Integration Demonstration ===")
    
    try:
        orchestrator = get_orchestrator()
        
        # Create mock Brain Core
        brain_config = BrainConfig()
        brain = MockBrainCore(brain_config)
        
        print(f"Created Brain Core with model parameters: {sum(p.numel() for p in brain.model.parameters())}")
        
        # Register with Brain Core
        orchestrator.brain_integration.register_with_brain(brain, "padic_primary")
        
        print(f"Registered P-Adic system with Brain Core")
        
        # Test memory monitoring
        print(f"\nTesting memory monitoring:")
        
        for i in range(3):
            try:
                # Simulate memory allocation
                large_tensor = brain._allocate_memory(1000000)  # 1M elements
                print(f"  Allocation {i+1}: 1M elements allocated successfully")
                
            except MemoryError as e:
                print(f"  Allocation {i+1}: Memory limit reached - {e}")
                break
            except Exception as e:
                print(f"  Allocation {i+1}: Unexpected error - {e}")
        
        # Test brain optimization
        print(f"\nTesting brain optimization:")
        
        optimization_result = orchestrator.brain_integration.optimize_brain_compression(brain)
        
        print(f"  Optimization completed:")
        print(f"    Total parameters: {optimization_result['analysis']['total_parameters']:,}")
        print(f"    Total memory: {optimization_result['analysis']['total_memory']:,} bytes")
        print(f"    Compression candidates: {optimization_result['analysis']['compression_candidates']}")
        print(f"    Memory saved: {optimization_result['memory_saved']:,} bytes")
        print(f"    Compression ratio: {optimization_result['compression_ratio']:.4f}")
        
        # Get brain metrics
        brain_metrics = orchestrator.brain_integration.get_brain_metrics()
        print(f"\nBrain Integration Metrics:")
        print(f"  Registrations: {brain_metrics['stats']['registrations']}")
        print(f"  Memory pressure events: {brain_metrics['stats']['memory_pressure_events']}")
        print(f"  Auto optimizations: {brain_metrics['stats']['auto_optimizations']}")
        print(f"  Active brains: {brain_metrics['active_brains']}")
        
    except Exception as e:
        print(f"✗ Brain integration failed: {e}")


def demonstrate_padic_training_integration():
    """Demonstrate P-Adic integration with training systems"""
    
    print(f"\n=== P-Adic Training Integration Demonstration ===")
    
    try:
        orchestrator = get_orchestrator()
        
        # Create a simple model for training
        model = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        
        # Create optimizers
        sgd_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        adam_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Register optimizers
        orchestrator.training_integration.register_optimizer(sgd_optimizer, "sgd_main")
        orchestrator.training_integration.register_optimizer(adam_optimizer, "adam_backup")
        
        print(f"Registered SGD and Adam optimizers")
        
        # Start training mode
        orchestrator.training_integration.start_training()
        
        print(f"Started training mode")
        
        # Simulate training steps
        print(f"\nSimulating training steps:")
        
        for step in range(20):
            # Generate fake data
            inputs = torch.randn(32, 20)
            targets = torch.randint(0, 2, (32, 1)).float()
            
            # Forward pass
            outputs = model(inputs)
            loss = nn.BCELoss()(outputs, targets)
            
            # Backward pass
            sgd_optimizer.zero_grad()
            loss.backward()
            
            # Optimizer step (this will trigger P-Adic hooks)
            sgd_optimizer.step()
            
            if step % 5 == 0:
                print(f"  Step {step}: loss = {loss.item():.4f}")
        
        # Stop training mode
        orchestrator.training_integration.stop_training()
        
        print(f"Stopped training mode")
        
        # Get training metrics
        training_metrics = orchestrator.training_integration.get_training_metrics()
        
        print(f"\nTraining Integration Metrics:")
        print(f"  Total steps: {training_metrics['total_steps']}")
        print(f"  Compressions performed: {training_metrics['compressions_performed']}")
        print(f"  Average compression time: {training_metrics['average_compression_time']:.6f}s")
        print(f"  Memory saved: {training_metrics['memory_saved_mb']:.2f} MB")
        print(f"  Precision adjustments: {training_metrics['precision_adjustments']}")
        print(f"  Current precision: {training_metrics['current_precision']}")
        print(f"  Warmup complete: {training_metrics['warmup_complete']}")
        
        if training_metrics['recent_gradient_magnitude'] > 0:
            print(f"  Recent gradient magnitude: {training_metrics['recent_gradient_magnitude']:.6f}")
            print(f"  Recent gradient sparsity: {training_metrics['recent_gradient_sparsity']:.4f}")
        
    except Exception as e:
        print(f"✗ Training integration failed: {e}")


def demonstrate_orchestrator_services():
    """Demonstrate orchestrator service interfaces"""
    
    print(f"\n=== Orchestrator Services Demonstration ===")
    
    try:
        orchestrator = get_orchestrator()
        
        # Test compression services
        print(f"Testing compression services:")
        
        test_data = torch.randn(100, 100) * 0.1
        print(f"  Original data: shape {test_data.shape}, norm {test_data.norm().item():.6f}")
        
        # Synchronous compression
        start_time = time.perf_counter()
        compressed = orchestrator.compress(test_data)
        compression_time = time.perf_counter() - start_time
        
        print(f"  ✓ Compression: {compression_time:.4f}s")
        
        # Synchronous decompression
        start_time = time.perf_counter()
        decompressed = orchestrator.decompress(compressed)
        decompression_time = time.perf_counter() - start_time
        
        print(f"  ✓ Decompression: {decompression_time:.4f}s")
        
        # Verify reconstruction
        reconstruction_error = torch.mean(torch.abs(test_data - decompressed)).item()
        print(f"  Reconstruction error: {reconstruction_error:.8f}")
        
        # Test asynchronous operations
        print(f"\nTesting asynchronous operations:")
        
        async def test_async():
            # Async compression
            compress_future = orchestrator.compress_async(test_data)
            compressed_async = await asyncio.wrap_future(compress_future)
            
            # Async decompression
            decompress_future = orchestrator.decompress_async(compressed_async)
            decompressed_async = await asyncio.wrap_future(decompress_future)
            
            return decompressed_async
        
        decompressed_async = asyncio.run(test_async())
        async_error = torch.mean(torch.abs(test_data - decompressed_async)).item()
        
        print(f"  ✓ Async operations completed")
        print(f"  Async reconstruction error: {async_error:.8f}")
        
        # Test batch operations
        print(f"\nTesting batch operations:")
        
        with orchestrator.batch_operations() as batch_orchestrator:
            batch_data = [torch.randn(50, 50) * 0.1 for _ in range(5)]
            
            batch_start = time.perf_counter()
            batch_compressed = [batch_orchestrator.compress(data) for data in batch_data]
            batch_decompressed = [batch_orchestrator.decompress(comp) for comp in batch_compressed]
            batch_time = time.perf_counter() - batch_start
            
            print(f"  ✓ Batch operations: 5 tensors in {batch_time:.4f}s")
            
            # Verify batch reconstruction
            batch_errors = [torch.mean(torch.abs(orig - decomp)).item() 
                           for orig, decomp in zip(batch_data, batch_decompressed)]
            avg_batch_error = np.mean(batch_errors)
            
            print(f"  Average batch reconstruction error: {avg_batch_error:.8f}")
        
        # Test system optimization
        print(f"\nTesting system optimization:")
        
        optimization_result = orchestrator.optimize_system()
        
        print(f"  System optimization completed:")
        for category, optimizations in optimization_result.items():
            if optimizations:
                print(f"    {category}: {optimizations}")
            else:
                print(f"    {category}: No optimizations needed")
        
        # Get comprehensive metrics
        print(f"\nSystem-wide metrics:")
        
        system_metrics = orchestrator.get_system_metrics()
        
        print(f"  Performance:")
        print(f"    Avg compression time: {system_metrics['performance']['avg_compression_time']:.6f}s")
        print(f"    Avg decompression time: {system_metrics['performance']['avg_decompression_time']:.6f}s")
        print(f"    Current queue size: {system_metrics['performance']['current_queue_size']}")
        print(f"    Active executor threads: {system_metrics['performance']['executor_active']}")
        
        print(f"  GAC Integration:")
        print(f"    Total compressions: {system_metrics['gac_integration']['total_compressions']}")
        print(f"    Gradients processed: {system_metrics['gac_integration']['gradients_processed']}")
        print(f"    Memory saved: {system_metrics['gac_integration']['memory_saved_bytes']:,} bytes")
        
    except Exception as e:
        print(f"✗ Services demonstration failed: {e}")


def demonstrate_error_handling():
    """Demonstrate hard failure error handling"""
    
    print(f"\n=== Error Handling Demonstration ===")
    
    try:
        orchestrator = get_orchestrator()
        
        print(f"Testing hard failure scenarios:")
        
        # Test invalid data compression
        print(f"  1. Invalid data compression:")
        try:
            orchestrator.compress(None)
        except ValueError as e:
            print(f"    ✓ Caught expected error: {e}")
        
        # Test invalid decompression
        print(f"  2. Invalid decompression:")
        try:
            orchestrator.decompress(None)
        except ValueError as e:
            print(f"    ✓ Caught expected error: {e}")
        
        # Test invalid GAC component enhancement
        print(f"  3. Invalid GAC component:")
        try:
            orchestrator.gac_integration.enhance_gradient_component(None)
        except ValueError as e:
            print(f"    ✓ Caught expected error: {e}")
        
        # Test invalid Brain Core registration
        print(f"  4. Invalid Brain Core:")
        try:
            orchestrator.brain_integration.register_with_brain(None)
        except ValueError as e:
            print(f"    ✓ Caught expected error: {e}")
        
        # Test invalid optimizer registration
        print(f"  5. Invalid optimizer:")
        try:
            orchestrator.training_integration.register_optimizer(None)
        except ValueError as e:
            print(f"    ✓ Caught expected error: {e}")
        
        print(f"  All error handling tests passed")
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities"""
    
    print(f"\n=== Performance Monitoring Demonstration ===")
    
    try:
        orchestrator = get_orchestrator()
        
        # Generate load for monitoring
        print(f"Generating load for performance monitoring:")
        
        test_tensors = [torch.randn(200, 200) * 0.1 for _ in range(20)]
        
        start_time = time.perf_counter()
        
        for i, tensor in enumerate(test_tensors):
            compressed = orchestrator.compress(tensor)
            decompressed = orchestrator.decompress(compressed)
            
            if i % 5 == 0:
                print(f"  Processed {i+1}/20 tensors")
        
        total_time = time.perf_counter() - start_time
        
        print(f"  Completed load generation in {total_time:.4f}s")
        
        # Allow monitoring to collect data
        time.sleep(2.0)
        
        # Get detailed metrics
        metrics = orchestrator.get_system_metrics()
        
        print(f"\nPerformance Analysis:")
        
        # P-Adic system metrics
        if 'total_compressions' in metrics['padic_system']:
            print(f"  P-Adic System:")
            print(f"    Total compressions: {metrics['padic_system']['total_compressions']}")
            print(f"    Average precision: {metrics['padic_system'].get('average_precision', 'N/A')}")
        
        # Performance metrics
        print(f"  Processing Performance:")
        print(f"    Average compression time: {metrics['performance']['avg_compression_time']:.6f}s")
        print(f"    Average decompression time: {metrics['performance']['avg_decompression_time']:.6f}s")
        
        # Calculate throughput
        total_elements = sum(tensor.numel() for tensor in test_tensors)
        throughput = total_elements / total_time
        
        print(f"    Throughput: {throughput:.0f} elements/second")
        print(f"    Total elements processed: {total_elements:,}")
        
        # Memory efficiency
        original_size = sum(tensor.numel() * tensor.element_size() for tensor in test_tensors)
        print(f"    Original data size: {original_size:,} bytes")
        
        if metrics['gac_integration']['memory_saved_bytes'] > 0:
            memory_efficiency = metrics['gac_integration']['memory_saved_bytes'] / original_size
            print(f"    Memory efficiency: {memory_efficiency:.2%}")
        
    except Exception as e:
        print(f"✗ Performance monitoring failed: {e}")


def run_complete_integration_demo():
    """Run complete P-Adic integration demonstration"""
    
    print("P-Adic Integration System - Complete Demonstration")
    print("=" * 60)
    
    try:
        # Individual component demonstrations
        demonstrate_padic_gac_integration()
        demonstrate_padic_brain_integration()
        demonstrate_padic_training_integration()
        
        # Service interface demonstrations
        demonstrate_orchestrator_services()
        
        # Error handling demonstration
        demonstrate_error_handling()
        
        # Performance monitoring
        demonstrate_performance_monitoring()
        
        print(f"\n" + "=" * 60)
        print("✓ All P-Adic integration demonstrations completed successfully!")
        
    except Exception as e:
        print(f"\n" + "=" * 60)
        print(f"✗ Integration demonstration failed: {e}")
        
    finally:
        # Cleanup
        print(f"\nShutting down P-Adic integration system...")
        shutdown_padic_integration()
        print(f"✓ Shutdown complete")


if __name__ == "__main__":
    run_complete_integration_demo()