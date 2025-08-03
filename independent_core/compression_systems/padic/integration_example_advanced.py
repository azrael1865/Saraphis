"""
Advanced P-adic Features Integration Example
Demonstrates Hensel lifting, hierarchical clustering, GPU decompression, and p-adic optimizers
"""

import torch
import numpy as np
import time
from typing import Dict, Any, List

# Import core P-adic components
from independent_core.compression_systems.padic import (
    PadicWeight, PadicCompressionSystem,
    HenselLiftingConfig, HenselLiftingProcessor,
    ClusteringConfig, HierarchicalClusteringManager,
    GPUDecompressionConfig, PadicDecompressionEngine,
    PadicOptimizationManager, PadicAdvancedIntegration
)


def demonstrate_hensel_lifting():
    """Demonstrate Hensel lifting for precision enhancement"""
    
    print("=== Hensel Lifting Demonstration ===")
    
    # Configure Hensel lifting
    hensel_config = HenselLiftingConfig(
        max_iterations=30,
        convergence_tolerance=1e-10,
        damping_factor=0.7,
        adaptive_damping=True,
        enable_validation=True
    )
    
    prime = 7
    base_precision = 5
    
    # Initialize Hensel processor
    hensel_processor = HenselLiftingProcessor(hensel_config, prime, base_precision)
    
    print(f"Initialized Hensel lifting processor for prime {prime}")
    
    # Create sample p-adic weight
    coefficients = [3, 1, 4, 1, 5]  # Sample coefficients
    padic_weight = PadicWeight(
        value=Fraction(sum(coeff * (prime ** i) for i, coeff in enumerate(coefficients)), 1),
        prime=prime,
        precision=base_precision,
        valuation=0,
        digits=coefficients
    )
    
    print(f"Original weight: precision={padic_weight.precision}, digits={padic_weight.digits}")
    
    try:
        # Lift to higher precision
        target_precision = 12
        lifted_weight, lifting_metadata = hensel_processor.lift_to_precision(
            padic_weight, target_precision
        )
        
        print(f"✓ Hensel lifting successful:")
        print(f"  Target precision: {target_precision}")
        print(f"  Final precision: {lifted_weight.precision}")
        print(f"  Total iterations: {lifting_metadata['total_iterations']}")
        print(f"  Lifting time: {lifting_metadata['lifting_time']:.4f}s")
        print(f"  Precision schedule: {lifting_metadata['precision_schedule']}")
        
        # Validate that lifting preserves original information
        reduced_digits = lifted_weight.digits[:base_precision]
        if reduced_digits == coefficients:
            print(f"  ✓ Lifting validation: original information preserved")
        else:
            print(f"  ⚠ Lifting validation: information changed")
            print(f"    Original: {coefficients}")
            print(f"    Reduced:  {reduced_digits}")
        
        # Performance statistics
        stats = hensel_processor.get_lifting_stats()
        print(f"  Performance stats:")
        print(f"    Total lifts: {stats['total_lifts']}")
        print(f"    Average iterations: {stats['average_iterations']:.2f}")
        print(f"    Convergence failures: {stats['convergence_failures']}")
        
    except Exception as e:
        print(f"✗ Hensel lifting failed: {e}")


def demonstrate_hierarchical_clustering():
    """Demonstrate hierarchical clustering with ultrametric distances"""
    
    print(f"\n=== Hierarchical Clustering Demonstration ===")
    
    # Configure clustering
    clustering_config = ClusteringConfig(
        max_cluster_size=500,
        min_cluster_size=5,
        branching_factor=3,
        distance_threshold=1e-4,
        enable_caching=True,
        ultrametric_validation=True,
        cohomology_tracking=True
    )
    
    prime = 5
    
    # Initialize clustering manager
    clustering_manager = HierarchicalClusteringManager(clustering_config, prime)
    
    print(f"Initialized hierarchical clustering for prime {prime}")
    
    # Create sample p-adic weights for clustering
    np.random.seed(42)  # For reproducibility
    weights = []
    
    for i in range(50):  # Create 50 sample weights
        # Generate coefficients with some structure
        base_coeffs = [i % prime, (i * 2) % prime, (i * 3) % prime]
        noise = [np.random.randint(0, prime) for _ in range(2)]
        coefficients = base_coeffs + noise
        
        weight = PadicWeight(
            value=Fraction(sum(coeff * (prime ** i) for i, coeff in enumerate(coefficients)), 1),
            prime=prime,
            precision=len(coefficients),
            valuation=0,
            digits=coefficients
        )
        weights.append(weight)
    
    print(f"Created {len(weights)} sample p-adic weights")
    
    try:
        # Build hierarchical clustering
        root_node, clustering_metadata = clustering_manager.build_hierarchical_clustering(weights)
        
        print(f"✓ Hierarchical clustering successful:")
        print(f"  Input weights: {clustering_metadata['num_weights']}")
        print(f"  Tree depth: {clustering_metadata['tree_depth']}")
        print(f"  Total nodes: {clustering_metadata['total_nodes']}")
        print(f"  Leaf nodes: {clustering_metadata['leaf_nodes']}")
        print(f"  Clustering time: {clustering_metadata['clustering_time']:.4f}s")
        
        # Cache usage statistics
        cache_usage = clustering_metadata['cache_usage']
        print(f"  Cache performance:")
        print(f"    Cache size: {cache_usage['size']}")
        print(f"    Cache hits: {cache_usage['hits']}")
        print(f"    Cache misses: {cache_usage['misses']}")
        
        # Cohomology information
        if clustering_metadata['cohomology_classes']:
            cohomology = clustering_metadata['cohomology_classes']
            print(f"  Cohomology classes:")
            print(f"    Total classes: {cohomology['total_classes']}")
            print(f"    Max class: {cohomology['max_class']}")
        
        # Test cluster finding
        test_weight = weights[0]
        found_cluster = clustering_manager.find_cluster_for_weight(test_weight)
        if found_cluster:
            print(f"  ✓ Cluster finding test: found cluster {found_cluster.id}")
        
        # Performance statistics
        stats = clustering_manager.get_clustering_stats()
        print(f"  Performance stats:")
        print(f"    Total clusterings: {stats['total_clusterings']}")
        print(f"    Nodes created: {stats['total_nodes_created']}")
        print(f"    Ultrametric violations: {stats['ultrametric_violations']}")
        
    except Exception as e:
        print(f"✗ Hierarchical clustering failed: {e}")


def demonstrate_gpu_decompression():
    """Demonstrate GPU-optimized progressive decompression"""
    
    print(f"\n=== GPU Decompression Demonstration ===")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU decompression demonstration")
        return
    
    # Configure GPU decompression
    gpu_config = GPUDecompressionConfig(
        enable_cuda_streams=True,
        num_streams=3,
        batch_size=20,
        memory_pool_size_mb=256,
        enable_progressive_precision=True,
        enable_async_transfer=True
    )
    
    prime = 3
    
    try:
        # Initialize GPU decompression engine
        decompression_engine = PadicDecompressionEngine(gpu_config, prime)
        
        print(f"Initialized GPU decompression engine for prime {prime}")
        print(f"Using {gpu_config.num_streams} CUDA streams")
        
        # Create sample p-adic weights
        weights = []
        for i in range(100):  # Create 100 weights for batch processing
            coefficients = [i % prime, (i * 2) % prime, (i * 3) % prime, (i * 5) % prime]
            weight = PadicWeight(
                coefficients=coefficients,
                prime=prime,
                precision=len(coefficients)
            )
            weights.append(weight)
        
        # Create decompression metadata
        metadata = {
            'original_shape': (10, 10),  # 10x10 tensor
            'dtype': 'torch.float32',
            'device': 'cuda:0'
        }
        
        print(f"Created {len(weights)} weights for GPU decompression")
        
        # Perform progressive decompression
        target_precision = 8
        decompressed_tensor, decompression_metadata = decompression_engine.decompress_progressive(
            weights, target_precision, metadata
        )
        
        print(f"✓ GPU decompression successful:")
        print(f"  Input weights: {decompression_metadata['input_weights']}")
        print(f"  Target precision: {decompression_metadata['target_precision']}")
        print(f"  Precision schedule: {decompression_metadata['precision_schedule']}")
        print(f"  Num batches: {decompression_metadata['num_batches']}")
        print(f"  Streams used: {decompression_metadata['streams_used']}")
        print(f"  Decompression time: {decompression_metadata['decompression_time']:.4f}s")
        print(f"  Throughput: {decompression_metadata['throughput']:.2f} weights/s")
        
        # GPU utilization
        gpu_util = decompression_metadata['gpu_utilization']
        print(f"  GPU utilization:")
        print(f"    Average: {gpu_util['average']:.2%}")
        print(f"    Per stream: {[f'{u:.2%}' for u in gpu_util['per_stream']]}")
        
        # Memory usage
        memory_usage = decompression_metadata['memory_usage']
        print(f"  Memory usage:")
        print(f"    Utilization: {memory_usage['utilization']:.2%}")
        print(f"    Allocated: {memory_usage['allocated_bytes']:,} bytes")
        
        # Validate output tensor
        print(f"  Output tensor:")
        print(f"    Shape: {decompressed_tensor.shape}")
        print(f"    Device: {decompressed_tensor.device}")
        print(f"    Dtype: {decompressed_tensor.dtype}")
        
        # Performance statistics
        stats = decompression_engine.get_decompression_stats()
        print(f"  Performance stats:")
        print(f"    Total decompressions: {stats['total_decompressions']}")
        print(f"    Total weights processed: {stats['total_weights_processed']}")
        print(f"    Average throughput: {stats['average_throughput']:.2f} weights/s")
        
        # Cleanup
        decompression_engine.cleanup()
        
    except Exception as e:
        print(f"✗ GPU decompression failed: {e}")


def demonstrate_padic_optimizers():
    """Demonstrate p-adic optimization algorithms"""
    
    print(f"\n=== P-adic Optimizers Demonstration ===")
    
    prime = 7
    
    # Initialize optimization manager
    optimization_manager = PadicOptimizationManager(prime)
    
    print(f"Initialized p-adic optimization manager for prime {prime}")
    
    # Create sample parameters (p-adic weights)
    params = []
    for i in range(10):  # 10 parameters
        coefficients = [i % prime, (i * 2) % prime, (i * 3) % prime]
        weight = PadicWeight(
            coefficients=coefficients,
            prime=prime,
            precision=len(coefficients)
        )
        params.append(weight)
    
    print(f"Created {len(params)} p-adic parameters")
    
    # Create sample gradients
    gradients = []
    for i in range(len(params)):
        grad_coeffs = [(i + 1) % prime, ((i + 1) * 2) % prime]
        gradient = PadicWeight(
            coefficients=grad_coeffs + [0],  # Pad to match parameter precision
            prime=prime,
            precision=3
        )
        gradients.append(gradient)
    
    print(f"Created {len(gradients)} p-adic gradients")
    
    try:
        # Test SGD optimizer
        print(f"\n1. SGD Optimizer:")
        sgd_name = optimization_manager.create_sgd_optimizer(
            params=params.copy(),  # Use copy to avoid modifying original
            lr=0.01,
            momentum=0.9,
            name="test_sgd"
        )
        
        print(f"   Created SGD optimizer: {sgd_name}")
        
        # Perform optimization steps
        for step in range(5):
            optimization_manager.step(sgd_name, gradients)
        
        sgd_stats = optimization_manager.get_optimizer_stats(sgd_name)
        print(f"   Steps taken: {sgd_stats['steps_taken']}")
        print(f"   Average step time: {sgd_stats['average_step_time']:.6f}s")
        
        # Test Adam optimizer
        print(f"\n2. Adam Optimizer:")
        adam_name = optimization_manager.create_adam_optimizer(
            params=params.copy(),
            lr=0.001,
            betas=(0.9, 0.999),
            name="test_adam"
        )
        
        print(f"   Created Adam optimizer: {adam_name}")
        
        # Perform optimization steps
        for step in range(5):
            optimization_manager.step(adam_name, gradients)
        
        adam_stats = optimization_manager.get_optimizer_stats(adam_name)
        print(f"   Steps taken: {adam_stats['steps_taken']}")
        print(f"   Average step time: {adam_stats['average_step_time']:.6f}s")
        
        # Test RMSprop optimizer
        print(f"\n3. RMSprop Optimizer:")
        rmsprop_name = optimization_manager.create_rmsprop_optimizer(
            params=params.copy(),
            lr=0.01,
            alpha=0.99,
            momentum=0.1,
            name="test_rmsprop"
        )
        
        print(f"   Created RMSprop optimizer: {rmsprop_name}")
        
        # Perform optimization steps
        for step in range(5):
            optimization_manager.step(rmsprop_name, gradients)
        
        rmsprop_stats = optimization_manager.get_optimizer_stats(rmsprop_name)
        print(f"   Steps taken: {rmsprop_stats['steps_taken']}")
        print(f"   Average step time: {rmsprop_stats['average_step_time']:.6f}s")
        
        # List all optimizers
        all_optimizers = optimization_manager.list_optimizers()
        print(f"\n4. All optimizers: {all_optimizers}")
        
        # Get comprehensive stats
        all_stats = optimization_manager.get_all_stats()
        print(f"\n5. Comprehensive statistics:")
        for opt_name, stats in all_stats.items():
            print(f"   {opt_name}: {stats['steps_taken']} steps, {stats['type']} type")
        
    except Exception as e:
        print(f"✗ P-adic optimizers failed: {e}")


def demonstrate_integrated_system():
    """Demonstrate full integration of advanced P-adic features"""
    
    print(f"\n=== Integrated Advanced P-adic System ===")
    
    # Create base compression system
    config = {
        'prime': 5,
        'precision': 6,
        'chunk_size': 1000,
        'gpu_memory_limit_mb': 512,
        'preserve_ultrametric': True,
        'validate_reconstruction': True
    }
    
    compression_system = PadicCompressionSystem(config)
    
    print(f"Created base P-adic compression system")
    
    try:
        # Integrate Hensel lifting
        hensel_config = HenselLiftingConfig(
            max_iterations=20,
            convergence_tolerance=1e-8,
            adaptive_damping=True
        )
        
        hensel_processor = PadicAdvancedIntegration.integrate_hensel_lifting(
            compression_system, hensel_config
        )
        
        print(f"✓ Integrated Hensel lifting processor")
        
        # Integrate hierarchical clustering
        clustering_config = ClusteringConfig(
            max_cluster_size=200,
            min_cluster_size=3,
            branching_factor=4,
            cohomology_tracking=True
        )
        
        clustering_manager = PadicAdvancedIntegration.integrate_hierarchical_clustering(
            compression_system, clustering_config
        )
        
        print(f"✓ Integrated hierarchical clustering manager")
        
        # Integrate GPU decompression (if CUDA available)
        if torch.cuda.is_available():
            gpu_config = GPUDecompressionConfig(
                num_streams=2,
                batch_size=50,
                enable_progressive_precision=True
            )
            
            gpu_engine = PadicAdvancedIntegration.integrate_gpu_decompression(
                compression_system, gpu_config
            )
            
            print(f"✓ Integrated GPU decompression engine")
        else:
            print(f"⚠ CUDA not available, skipping GPU integration")
        
        # Integrate optimization manager
        optimization_manager = PadicAdvancedIntegration.integrate_optimization_manager(
            compression_system
        )
        
        print(f"✓ Integrated optimization manager")
        
        # Test integrated system with sample data
        print(f"\nTesting integrated system:")
        
        # Create test tensor
        test_tensor = torch.randn(5, 5) * 0.1  # Small values for better p-adic representation
        print(f"  Input tensor shape: {test_tensor.shape}")
        
        # Compress using base system
        compressed = compression_system.compress(test_tensor)
        print(f"  ✓ Compression successful")
        
        # Test Hensel lifting on compressed weights
        if hasattr(compression_system, 'hensel_processor'):
            sample_weight = compressed['encoded_data'][0]
            lifted_weight, _ = compression_system.lift_precision(sample_weight, 10)
            print(f"  ✓ Hensel lifting: {sample_weight.precision} -> {lifted_weight.precision}")
        
        # Test clustering on compressed weights
        if hasattr(compression_system, 'clustering_manager'):
            weights_sample = compressed['encoded_data'][:20]  # Use subset for speed
            root_node, _ = compression_system.build_clusters(weights_sample)
            print(f"  ✓ Clustering: {root_node.size()} weights clustered")
        
        # Test optimization
        if hasattr(compression_system, 'optimization_manager'):
            weights_sample = compressed['encoded_data'][:5]  # Small sample for optimization
            gradients_sample = [w.copy() for w in weights_sample]  # Mock gradients
            
            optimizer_name = compression_system.create_optimizer(
                weights_sample, lr=0.01, name="integrated_test"
            )
            compression_system.optimization_manager.step(optimizer_name, gradients_sample)
            print(f"  ✓ Optimization: performed step with {optimizer_name}")
        
        # Decompress
        decompressed = compression_system.decompress(compressed)
        print(f"  ✓ Decompression successful")
        
        # Validate reconstruction
        reconstruction_error = torch.mean(torch.abs(test_tensor - decompressed)).item()
        print(f"  Reconstruction error: {reconstruction_error:.6f}")
        
        if reconstruction_error < 0.01:
            print(f"  ✓ Reconstruction validation passed")
        else:
            print(f"  ⚠ Reconstruction error higher than expected")
        
        print(f"\n✓ Integrated advanced P-adic system demonstration complete!")
        
    except Exception as e:
        print(f"✗ Integrated system failed: {e}")


if __name__ == "__main__":
    # Run all demonstrations
    print("Advanced P-adic Features Integration Examples")
    print("=" * 50)
    
    demonstrate_hensel_lifting()
    demonstrate_hierarchical_clustering()
    demonstrate_gpu_decompression()
    demonstrate_padic_optimizers()
    demonstrate_integrated_system()
    
    print(f"\n" + "=" * 50)
    print("All advanced P-adic demonstrations complete!")