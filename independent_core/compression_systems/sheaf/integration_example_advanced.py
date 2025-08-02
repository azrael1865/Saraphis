"""
Advanced Sheaf Theory Features Integration Example
Demonstrates cellular sheaf construction, restriction maps, cohomology, and reconstruction
"""

import torch
import numpy as np
import time
from typing import Dict, Any, List

# Import core sheaf components
from independent_core.compression_systems.sheaf import (
    CellularSheaf, RestrictionMap, SheafCompressionSystem,
    CellularSheafBuilder, RestrictionMapProcessor, 
    SheafCohomologyCalculator, SheafReconstructionEngine,
    SheafAdvancedIntegration
)


def demonstrate_cellular_sheaf_builder():
    """Demonstrate cellular sheaf construction with different partition strategies"""
    
    print("=== Cellular Sheaf Builder Demonstration ===")
    
    # Create test data - 2D image-like tensor
    torch.manual_seed(42)
    test_data = torch.randn(32, 32) + torch.sin(torch.linspace(0, 4*np.pi, 32)).unsqueeze(0)
    
    print(f"Input data shape: {test_data.shape}")
    
    # Configure sheaf builder
    builder_config = {
        'max_cells': 50,
        'min_cell_size': 20,
        'overlap_threshold': 0.15,
        'dimension_hierarchy': [0, 1, 2]
    }
    
    sheaf_builder = CellularSheafBuilder(builder_config)
    
    # Test different partition strategies
    strategies = ["grid", "adaptive", "hierarchical"]
    
    for strategy in strategies:
        print(f"\n{strategy.upper()} Partition Strategy:")
        
        try:
            start_time = time.time()
            sheaf = sheaf_builder.build_from_data(test_data, strategy)
            build_time = time.time() - start_time
            
            print(f"  ✓ Sheaf construction successful:")
            print(f"    Number of cells: {len(sheaf.cells)}")
            print(f"    Topology connections: {sum(len(neighbors) for neighbors in sheaf.topology.values()) // 2}")
            print(f"    Overlap regions: {len(sheaf.overlap_regions)}")
            print(f"    Build time: {build_time:.4f}s")
            
            # Analyze cell dimensions
            dim_distribution = {}
            for cell, dim in sheaf.cell_dimensions.items():
                dim_distribution[dim] = dim_distribution.get(dim, 0) + 1
            
            print(f"    Cell dimension distribution: {dim_distribution}")
            
            # Analyze section statistics
            section_sizes = [section['data'].numel() for section in sheaf.sections.values()]
            print(f"    Section sizes - min: {min(section_sizes)}, max: {max(section_sizes)}, avg: {np.mean(section_sizes):.1f}")
            
            # Check gluing data quality
            priorities = [gluing['reconstruction_priority'] for gluing in sheaf.gluing_data.values()]
            print(f"    Reconstruction priorities - min: {min(priorities):.3f}, max: {max(priorities):.3f}")
            
        except Exception as e:
            print(f"  ✗ {strategy} partition failed: {e}")
    
    return sheaf  # Return last successful sheaf for further testing


def demonstrate_restriction_map_processor():
    """Demonstrate restriction map computation with consistency validation"""
    
    print(f"\n=== Restriction Map Processor Demonstration ===")
    
    # Create test sheaf
    torch.manual_seed(42)
    test_data = torch.randn(24, 24)
    
    builder = CellularSheafBuilder({'max_cells': 20, 'min_cell_size': 10})
    sheaf = builder.build_from_data(test_data, "grid")
    
    print(f"Base sheaf: {len(sheaf.cells)} cells")
    
    # Configure restriction map processor
    processor_config = {
        'map_types': ['projection', 'interpolation', 'averaging', 'inclusion'],
        'consistency_threshold': 0.90,
        'max_iterations': 50
    }
    
    processor = RestrictionMapProcessor(processor_config)
    
    try:
        start_time = time.time()
        restriction_maps = processor.compute_restriction_maps(sheaf)
        computation_time = time.time() - start_time
        
        print(f"✓ Restriction map computation successful:")
        print(f"  Total restriction maps: {len(restriction_maps)}")
        print(f"  Computation time: {computation_time:.4f}s")
        
        # Analyze map types
        map_types = {}
        for rmap in restriction_maps.values():
            map_type = rmap.restriction_type
            map_types[map_type] = map_types.get(map_type, 0) + 1
        
        print(f"  Map type distribution: {map_types}")
        
        # Test restriction map functionality
        print(f"\n  Testing restriction maps:")
        test_count = 0
        successful_restrictions = 0
        
        for (source_cell, target_cell), rmap in list(restriction_maps.items())[:5]:  # Test first 5
            try:
                source_data = sheaf.sections[source_cell]['data']
                restricted_data = rmap.restriction_func(source_data)
                
                print(f"    {source_cell} -> {target_cell}: {source_data.shape} -> {restricted_data.shape} ({rmap.restriction_type})")
                successful_restrictions += 1
            except Exception as e:
                print(f"    {source_cell} -> {target_cell}: ✗ Failed - {e}")
            
            test_count += 1
        
        print(f"  Successful restrictions: {successful_restrictions}/{test_count}")
        
        # Update sheaf with restriction maps
        sheaf.restriction_maps = restriction_maps
        
        return sheaf
        
    except Exception as e:
        print(f"✗ Restriction map computation failed: {e}")
        return None


def demonstrate_cohomology_calculator():
    """Demonstrate sheaf cohomology computation"""
    
    print(f"\n=== Sheaf Cohomology Calculator Demonstration ===")
    
    # Create test sheaf with restriction maps
    torch.manual_seed(42)
    test_data = torch.randn(20, 20)
    
    builder = CellularSheafBuilder({'max_cells': 15, 'min_cell_size': 8})
    sheaf = builder.build_from_data(test_data, "adaptive")
    
    processor = RestrictionMapProcessor({'consistency_threshold': 0.85})
    restriction_maps = processor.compute_restriction_maps(sheaf)
    sheaf.restriction_maps = restriction_maps
    
    print(f"Sheaf for cohomology: {len(sheaf.cells)} cells, {len(restriction_maps)} restriction maps")
    
    # Configure cohomology calculator
    calculator_config = {
        'max_dimension': 2,
        'numerical_tolerance': 1e-8,
        'use_spectral_sequence': True
    }
    
    calculator = SheafCohomologyCalculator(calculator_config)
    
    try:
        start_time = time.time()
        cohomology_groups = calculator.compute_cohomology(sheaf)
        computation_time = time.time() - start_time
        
        print(f"✓ Cohomology computation successful:")
        print(f"  Computation time: {computation_time:.4f}s")
        
        # Analyze cohomology groups
        for dim, cohom_group in cohomology_groups.items():
            print(f"  H^{dim}: dimension {cohom_group.shape[1]} (rank {cohom_group.shape[0]})")
        
        # Compute Betti numbers
        betti_numbers = calculator.compute_betti_numbers(sheaf)
        print(f"  Betti numbers: {betti_numbers}")
        
        # Compute Euler characteristic
        euler_char = calculator.compute_euler_characteristic(sheaf)
        print(f"  Euler characteristic: {euler_char}")
        
        # Topological analysis
        total_betti = sum(betti_numbers.values())
        print(f"  Total Betti number: {total_betti}")
        
        if total_betti == 0:
            print(f"  ✓ Topologically trivial sheaf")
        else:
            print(f"  ⚠ Non-trivial topology detected")
        
        return cohomology_groups
        
    except Exception as e:
        print(f"✗ Cohomology computation failed: {e}")
        return None


def demonstrate_reconstruction_engine():
    """Demonstrate reconstruction from sheaf data"""
    
    print(f"\n=== Sheaf Reconstruction Engine Demonstration ===")
    
    # Create test data with known structure
    torch.manual_seed(42)
    original_data = torch.zeros(28, 28)
    # Add some structure - diagonal pattern
    for i in range(28):
        for j in range(28):
            original_data[i, j] = np.sin(0.2 * (i + j)) + 0.1 * np.random.randn()
    
    print(f"Original data shape: {original_data.shape}")
    print(f"Original data range: [{original_data.min():.3f}, {original_data.max():.3f}]")
    
    # Build sheaf from original data
    builder = CellularSheafBuilder({'max_cells': 25, 'min_cell_size': 15})
    sheaf = builder.build_from_data(original_data, "grid")
    
    processor = RestrictionMapProcessor()
    restriction_maps = processor.compute_restriction_maps(sheaf)
    sheaf.restriction_maps = restriction_maps
    
    # Test different reconstruction methods
    reconstruction_methods = ['weighted', 'variational', 'spectral', 'hierarchical']
    
    for method in reconstruction_methods:
        print(f"\n{method.upper()} Reconstruction:")
        
        try:
            # Configure reconstruction engine
            engine_config = {
                'reconstruction_method': method,
                'consistency_weight': 0.6,
                'smoothness_weight': 0.3,
                'convergence_threshold': 1e-5,
                'max_iterations': 100
            }
            
            engine = SheafReconstructionEngine(engine_config)
            
            start_time = time.time()
            reconstructed = engine.reconstruct(sheaf, original_data.shape)
            reconstruction_time = time.time() - start_time
            
            # Compute reconstruction error
            mse_error = torch.mean((original_data - reconstructed)**2).item()
            max_error = torch.max(torch.abs(original_data - reconstructed)).item()
            
            print(f"  ✓ Reconstruction successful:")
            print(f"    Reconstruction time: {reconstruction_time:.4f}s")
            print(f"    MSE error: {mse_error:.6f}")
            print(f"    Max absolute error: {max_error:.6f}")
            print(f"    Reconstructed data range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
            
            # Quality assessment
            if mse_error < 0.01:
                print(f"    ✓ Excellent reconstruction quality")
            elif mse_error < 0.1:
                print(f"    ✓ Good reconstruction quality")
            else:
                print(f"    ⚠ Moderate reconstruction quality")
        
        except Exception as e:
            print(f"  ✗ {method} reconstruction failed: {e}")
    
    return reconstructed if 'reconstructed' in locals() else None


def demonstrate_complete_workflow():
    """Demonstrate complete advanced sheaf workflow"""
    
    print(f"\n=== Complete Advanced Sheaf Workflow ===")
    
    # Create complex test data
    torch.manual_seed(42)
    x = torch.linspace(0, 4*np.pi, 40)
    y = torch.linspace(0, 4*np.pi, 40)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Complex pattern with multiple frequency components
    test_data = (torch.sin(X) * torch.cos(Y) + 
                0.5 * torch.sin(2*X + Y) + 
                0.2 * torch.randn(40, 40))
    
    print(f"Complex test data: {test_data.shape}, range [{test_data.min():.3f}, {test_data.max():.3f}]")
    
    # Configure complete workflow
    workflow_config = {
        'builder': {
            'max_cells': 30,
            'min_cell_size': 25,
            'overlap_threshold': 0.2
        },
        'processor': {
            'consistency_threshold': 0.88,
            'max_iterations': 75
        },
        'calculator': {
            'max_dimension': 2,
            'numerical_tolerance': 1e-7
        },
        'engine': {
            'reconstruction_method': 'variational',
            'consistency_weight': 0.7,
            'smoothness_weight': 0.2
        }
    }
    
    try:
        # Create base compression system
        compression_system = SheafCompressionSystem({
            'precision': 1e-6,
            'max_cohomology_dimension': 2,
            'enable_validation': True
        })
        
        # Integrate all advanced features
        integrated_components = SheafAdvancedIntegration.integrate_all_advanced_features(
            compression_system, workflow_config
        )
        
        print(f"✓ Advanced features integrated:")
        for name, component in integrated_components.items():
            print(f"  - {name}: {type(component).__name__}")
        
        # Run complete workflow
        print(f"\nRunning complete workflow:")
        
        start_time = time.time()
        workflow_result = compression_system.complete_sheaf_workflow(test_data, "adaptive")
        total_time = time.time() - start_time
        
        print(f"✓ Complete workflow successful:")
        print(f"  Total processing time: {total_time:.4f}s")
        
        # Analyze results
        sheaf = workflow_result['sheaf']
        cohomology = workflow_result['cohomology']
        reconstructed = workflow_result['reconstructed']
        betti_numbers = workflow_result['betti_numbers']
        euler_char = workflow_result['euler_characteristic']
        
        print(f"  Sheaf structure:")
        print(f"    Cells: {len(sheaf.cells)}")
        print(f"    Restriction maps: {len(workflow_result['restriction_maps'])}")
        print(f"    Overlap regions: {len(sheaf.overlap_regions)}")
        
        print(f"  Topological properties:")
        print(f"    Betti numbers: {betti_numbers}")
        print(f"    Euler characteristic: {euler_char}")
        
        print(f"  Reconstruction quality:")
        mse_error = torch.mean((test_data - reconstructed)**2).item()
        max_error = torch.max(torch.abs(test_data - reconstructed)).item()
        print(f"    MSE error: {mse_error:.6f}")
        print(f"    Max error: {max_error:.6f}")
        
        # Compression analysis
        original_size = test_data.numel() * 4  # float32
        sheaf_size = sum(section['data'].numel() for section in sheaf.sections.values()) * 4
        compression_ratio = original_size / sheaf_size
        
        print(f"  Compression analysis:")
        print(f"    Original size: {original_size:,} bytes")
        print(f"    Sheaf size: {sheaf_size:,} bytes")
        print(f"    Compression ratio: {compression_ratio:.2f}x")
        
        return workflow_result
        
    except Exception as e:
        print(f"✗ Complete workflow failed: {e}")
        return None


def demonstrate_integration_with_existing_systems():
    """Demonstrate integration with existing compression systems"""
    
    print(f"\n=== Integration with Existing Systems ===")
    
    # Create base sheaf compression system
    base_config = {
        'precision': 1e-5,
        'max_cohomology_dimension': 2,
        'enable_validation': True,
        'compression_threshold': 0.1
    }
    
    compression_system = SheafCompressionSystem(base_config)
    
    print(f"Base compression system created")
    
    # Test individual integrations
    print(f"\nTesting individual component integrations:")
    
    try:
        # Test sheaf builder integration
        builder = SheafAdvancedIntegration.integrate_sheaf_builder(
            compression_system, 
            {'max_cells': 20, 'min_cell_size': 10}
        )
        print(f"  ✓ Sheaf builder integrated: {hasattr(compression_system, 'build_sheaf')}")
        
        # Test restriction processor integration
        processor = SheafAdvancedIntegration.integrate_restriction_processor(
            compression_system,
            {'consistency_threshold': 0.9}
        )
        print(f"  ✓ Restriction processor integrated: {hasattr(compression_system, 'compute_restrictions')}")
        
        # Test cohomology calculator integration
        calculator = SheafAdvancedIntegration.integrate_cohomology_calculator(
            compression_system,
            {'max_dimension': 2}
        )
        print(f"  ✓ Cohomology calculator integrated: {hasattr(compression_system, 'compute_cohomology')}")
        
        # Test reconstruction engine integration
        engine = SheafAdvancedIntegration.integrate_reconstruction_engine(
            compression_system,
            {'reconstruction_method': 'weighted'}
        )
        print(f"  ✓ Reconstruction engine integrated: {hasattr(compression_system, 'reconstruct_from_sheaf')}")
        
        # Test workflow integration
        print(f"  ✓ Complete workflow integrated: {hasattr(compression_system, 'complete_sheaf_workflow')}")
        
        # Test workflow on small data
        test_tensor = torch.randn(16, 16)
        
        print(f"\nTesting integrated workflow:")
        start_time = time.time()
        result = compression_system.complete_sheaf_workflow(test_tensor, "grid")
        integration_time = time.time() - start_time
        
        print(f"  ✓ Integrated workflow successful:")
        print(f"    Processing time: {integration_time:.4f}s")
        print(f"    Cells created: {len(result['sheaf'].cells)}")
        print(f"    Restriction maps: {len(result['restriction_maps'])}")
        print(f"    Cohomology computed: {len(result['cohomology'])} dimensions")
        
        # Reconstruction quality
        mse = torch.mean((test_tensor - result['reconstructed'])**2).item()
        print(f"    Reconstruction MSE: {mse:.6f}")
        
    except Exception as e:
        print(f"  ✗ Integration testing failed: {e}")


def demonstrate_performance_analysis():
    """Demonstrate performance characteristics of advanced features"""
    
    print(f"\n=== Performance Analysis ===")
    
    # Test different data sizes
    data_sizes = [(16, 16), (32, 32), (48, 48)]
    
    for size in data_sizes:
        print(f"\nTesting with data size {size}:")
        
        # Generate test data
        torch.manual_seed(42)
        test_data = torch.randn(*size)
        
        # Time each component
        try:
            # Sheaf building
            builder = CellularSheafBuilder({'max_cells': min(30, size[0]*size[1]//20)})
            
            start_time = time.time()
            sheaf = builder.build_from_data(test_data, "grid")
            build_time = time.time() - start_time
            
            # Restriction maps
            processor = RestrictionMapProcessor()
            
            start_time = time.time()
            restriction_maps = processor.compute_restriction_maps(sheaf)
            restriction_time = time.time() - start_time
            
            # Cohomology
            calculator = SheafCohomologyCalculator({'max_dimension': 2})
            
            start_time = time.time()
            cohomology = calculator.compute_cohomology(sheaf)
            cohomology_time = time.time() - start_time
            
            # Reconstruction
            engine = SheafReconstructionEngine({'reconstruction_method': 'weighted'})
            
            start_time = time.time()
            reconstructed = engine.reconstruct(sheaf, test_data.shape)
            reconstruction_time = time.time() - start_time
            
            total_time = build_time + restriction_time + cohomology_time + reconstruction_time
            
            print(f"  ✓ Performance metrics:")
            print(f"    Sheaf building: {build_time:.4f}s ({len(sheaf.cells)} cells)")
            print(f"    Restriction maps: {restriction_time:.4f}s ({len(restriction_maps)} maps)")
            print(f"    Cohomology: {cohomology_time:.4f}s")
            print(f"    Reconstruction: {reconstruction_time:.4f}s")
            print(f"    Total time: {total_time:.4f}s")
            
            # Quality metrics
            mse = torch.mean((test_data - reconstructed)**2).item()
            print(f"    Reconstruction MSE: {mse:.6f}")
            
            # Efficiency metrics
            data_elements = test_data.numel()
            throughput = data_elements / total_time
            print(f"    Throughput: {throughput:.1f} elements/second")
            
        except Exception as e:
            print(f"  ✗ Performance test failed for size {size}: {e}")


if __name__ == "__main__":
    # Run all demonstrations
    print("Advanced Sheaf Theory Features Integration Examples")
    print("=" * 60)
    
    # Individual component demonstrations
    sheaf = demonstrate_cellular_sheaf_builder()
    if sheaf:
        sheaf_with_maps = demonstrate_restriction_map_processor()
        
    demonstrate_cohomology_calculator()
    demonstrate_reconstruction_engine()
    
    # Complete workflow demonstration
    demonstrate_complete_workflow()
    
    # Integration testing
    demonstrate_integration_with_existing_systems()
    
    # Performance analysis
    demonstrate_performance_analysis()
    
    print(f"\n" + "=" * 60)
    print("All advanced sheaf theory demonstrations complete!")