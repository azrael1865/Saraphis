"""
Sheaf Theory Compression System Integration Example
Demonstrates integration with BrainCore and other independent_core components
"""

import numpy as np
from typing import Dict, Any

# Import core components
from independent_core.brain_core import BrainCore
from independent_core.compression_systems.sheaf import SheafCompressionSystem
from independent_core.proof_system.algebraic_rule_enforcer import AlgebraicRuleEnforcer
from independent_core.proof_system.confidence_generator import ConfidenceGenerator


def integrate_sheaf_with_brain_core():
    """Demonstrate integration of Sheaf compression with BrainCore"""
    
    # Initialize BrainCore
    brain_config = {
        'enable_caching': True,
        'max_compression_systems': 10,
        'default_compression': 'sheaf'
    }
    brain = BrainCore(config=brain_config)
    
    # Initialize Sheaf compression system with specific configuration
    sheaf_config = {
        'max_cell_size': 500,
        'min_cell_overlap': 0.2,
        'restriction_type': 'linear',
        'cohomology_degree': 1,
        'enable_validation': True
    }
    sheaf_system = SheafCompressionSystem(config=sheaf_config)
    
    # Register Sheaf system with BrainCore
    brain.register_compression_system('sheaf', sheaf_system)
    brain.active_compression = 'sheaf'
    
    # Example 1: Compress dictionary data
    data_dict = {
        'sensor_1': np.random.randn(100),
        'sensor_2': np.random.randn(100),
        'sensor_3': np.random.randn(100),
        'metadata': {'timestamp': '2025-01-20', 'location': 'lab_a'}
    }
    
    print("Compressing dictionary data...")
    compressed_dict = sheaf_system.compress(data_dict)
    print(f"Compression ratio: {compressed_dict['compression_metadata']['compression_ratio']:.2f}")
    print(f"Number of cells: {compressed_dict['compression_metadata']['num_cells']}")
    print(f"Confidence score: {compressed_dict['confidence_score']:.2f}")
    
    # Decompress and verify
    decompressed_dict = sheaf_system.decompress(compressed_dict)
    print(f"Successfully decompressed dictionary with {len(decompressed_dict)} keys")
    
    # Example 2: Compress 2D array (image-like data)
    image_data = np.random.randn(64, 64)
    
    print("\nCompressing 2D array data...")
    compressed_image = sheaf_system.compress(image_data)
    print(f"Compression ratio: {compressed_image['compression_metadata']['compression_ratio']:.2f}")
    print(f"Number of cells: {compressed_image['compression_metadata']['num_cells']}")
    
    # Access the sheaf structure
    sheaf = compressed_image['sheaf']
    print(f"Base space has {len(sheaf.base_space)} cells")
    print(f"Sheaf has {len(sheaf.restriction_maps)} restriction maps")
    
    # Verify cocycle condition
    if sheaf.verify_cocycle_condition():
        print("✓ Sheaf satisfies cocycle condition")
    
    # Example 3: Compress time series data
    time_series = np.sin(np.linspace(0, 10*np.pi, 1000)) + 0.1*np.random.randn(1000)
    
    print("\nCompressing time series data...")
    compressed_ts = sheaf_system.compress(time_series)
    print(f"Compression ratio: {compressed_ts['compression_metadata']['compression_ratio']:.2f}")
    
    # Demonstrate restriction maps
    sheaf_ts = compressed_ts['sheaf']
    first_cell = list(sheaf_ts.base_space.keys())[0]
    neighbors = list(sheaf_ts.base_space[first_cell])
    
    if neighbors:
        print(f"\nDemonstrating restriction from {first_cell} to {neighbors[0]}")
        restricted_data = sheaf_ts.restrict(first_cell, neighbors[0])
        print(f"Original section shape: {sheaf_ts.sections[first_cell].shape}")
        print(f"Restricted section shape: {restricted_data.shape}")
    
    # Example 4: Custom algebraic rules
    rule_enforcer = sheaf_system.rule_enforcer
    
    # Add custom rule for data validation
    def custom_data_rule(data):
        """Custom rule: data must have positive mean"""
        if isinstance(data, np.ndarray):
            return np.mean(data) > 0
        return True
    
    rule_enforcer.rules['positive_mean'] = custom_data_rule
    
    # Test custom rule
    positive_data = np.abs(np.random.randn(50))
    negative_data = -np.abs(np.random.randn(50))
    
    print("\nTesting custom algebraic rules...")
    print(f"Positive data passes rule: {rule_enforcer.enforce_rule('positive_mean', positive_data)}")
    print(f"Negative data passes rule: {rule_enforcer.enforce_rule('positive_mean', negative_data)}")
    
    # Example 5: Confidence generation for sheaf structures
    confidence_gen = sheaf_system.confidence_generator
    
    # Generate confidence for different compression scenarios
    high_ratio_context = {'compression_ratio': 5.0, 'num_cells': 10}
    low_ratio_context = {'compression_ratio': 1.2, 'num_cells': 50}
    
    print("\nConfidence generation for different scenarios:")
    print(f"High compression ratio confidence: {confidence_gen.sheaf_confidence_maps['default'](sheaf, high_ratio_context):.2f}")
    print(f"Low compression ratio confidence: {confidence_gen.sheaf_confidence_maps['default'](sheaf, low_ratio_context):.2f}")
    
    # Example 6: Working with global sections
    print("\nGlobal sections analysis:")
    if sheaf.global_sections:
        print(f"Number of global sections: {len(sheaf.global_sections)}")
        for section_name, section_data in sheaf.global_sections.items():
            if isinstance(section_data, np.ndarray):
                print(f"  {section_name}: shape {section_data.shape}")
            else:
                print(f"  {section_name}: type {type(section_data).__name__}")
    
    # Example 7: Compute cohomology
    print("\nComputing sheaf cohomology...")
    cochains_0 = sheaf.compute_cochains(0)
    cochains_1 = sheaf.compute_cochains(1)
    
    print(f"0-cochains (sections): {len(cochains_0)}")
    print(f"1-cochains (edge differences): {len(cochains_1)}")
    
    # Store cocycles for analysis
    sheaf.cocycles['degree_1'] = cochains_1
    
    # Example 8: Compression metrics
    print("\nCompression system metrics:")
    for metric_name, metric_value in sheaf_system.compression_metrics.items():
        print(f"  {metric_name}: {metric_value}")
    
    # Example 9: Error handling demonstration
    print("\nDemonstrating NO FALLBACK error handling:")
    
    try:
        # Attempt to compress None data
        sheaf_system.compress(None)
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    try:
        # Attempt to decompress invalid data
        sheaf_system.decompress({'invalid': 'data'})
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    try:
        # Attempt to access non-existent cell
        sheaf.get_section('non_existent_cell')
    except KeyError as e:
        print(f"✓ Correctly raised KeyError: {e}")
    
    # Example 10: Advanced sheaf operations
    print("\nAdvanced sheaf operations:")
    
    # Create custom restriction map
    from independent_core.compression_systems.sheaf import RestrictionMap
    
    def custom_morphism(data):
        """Custom morphism that applies smoothing"""
        if isinstance(data, np.ndarray):
            # Apply simple moving average
            kernel_size = 3
            kernel = np.ones(kernel_size) / kernel_size
            
            if data.ndim == 1:
                return np.convolve(data, kernel, mode='same')
            else:
                return data
        return data
    
    custom_restriction = RestrictionMap(
        source_cell='cell_custom_source',
        target_cell='cell_custom_target',
        morphism_function=custom_morphism,
        metadata={'smoothing_kernel_size': 3}
    )
    
    # Test custom restriction
    test_data = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
    smoothed_data = custom_restriction.apply(test_data)
    print(f"Original data: {test_data}")
    print(f"Smoothed data: {smoothed_data}")
    
    # Compose restriction maps
    if len(sheaf.restriction_maps) >= 2:
        maps = list(sheaf.restriction_maps.values())
        
        # Find two composable maps
        for map1 in maps:
            for map2 in maps:
                if map1.target_cell == map2.source_cell:
                    print(f"\nComposing maps: {map1.source_cell} -> {map1.target_cell} -> {map2.target_cell}")
                    composed_map = map1.compose(map2)
                    print(f"Composed map: {composed_map.source_cell} -> {composed_map.target_cell}")
                    break
            else:
                continue
            break
    
    print("\nSheaf compression system successfully integrated with BrainCore!")
    return brain, sheaf_system


def demonstrate_sheaf_validation():
    """Demonstrate the validation system"""
    from independent_core.compression_systems.sheaf import SheafValidation, CellularSheaf
    
    print("\n=== Sheaf Validation Demonstration ===")
    
    validator = SheafValidation()
    
    # Create a valid sheaf
    base_space = {
        'cell_0': {'cell_1'},
        'cell_1': {'cell_0', 'cell_2'},
        'cell_2': {'cell_1'}
    }
    
    sections = {
        'cell_0': np.array([1, 2, 3]),
        'cell_1': np.array([2, 3, 4]),
        'cell_2': np.array([3, 4, 5])
    }
    
    # Create restriction maps
    from independent_core.compression_systems.sheaf import RestrictionMap
    
    restriction_maps = {}
    for source, neighbors in base_space.items():
        for target in neighbors:
            restriction_maps[(source, target)] = RestrictionMap(
                source_cell=source,
                target_cell=target,
                morphism_function=lambda x: x + 1,  # Simple increment
                metadata={'type': 'increment'}
            )
    
    valid_sheaf = CellularSheaf(
        base_space=base_space,
        sections=sections,
        restriction_maps=restriction_maps
    )
    
    # Validate the sheaf
    try:
        validator.validate_sheaf_structure(valid_sheaf)
        print("✓ Valid sheaf passed validation")
    except ValueError as e:
        print(f"✗ Validation failed: {e}")
    
    # Test invalid sheaf (disconnected)
    disconnected_base = {
        'cell_0': {'cell_1'},
        'cell_1': {'cell_0'},
        'cell_2': set(),  # Isolated cell
        'cell_3': set()   # Another isolated cell
    }
    
    disconnected_sections = {
        'cell_0': np.array([1, 2]),
        'cell_1': np.array([2, 3]),
        'cell_2': np.array([4, 5]),
        'cell_3': np.array([6, 7])
    }
    
    disconnected_maps = {
        ('cell_0', 'cell_1'): RestrictionMap(
            source_cell='cell_0',
            target_cell='cell_1',
            morphism_function=lambda x: x
        ),
        ('cell_1', 'cell_0'): RestrictionMap(
            source_cell='cell_1',
            target_cell='cell_0',
            morphism_function=lambda x: x
        )
    }
    
    try:
        invalid_sheaf = CellularSheaf(
            base_space=disconnected_base,
            sections=disconnected_sections,
            restriction_maps=disconnected_maps
        )
        validator.validate_sheaf_structure(invalid_sheaf)
        print("✗ Invalid sheaf should have failed validation")
    except ValueError as e:
        print(f"✓ Correctly rejected disconnected sheaf: {e}")
    
    print("\nValidation metrics:")
    for metric, value in validator.validation_metrics.items():
        print(f"  {metric}: {value}")


if __name__ == "__main__":
    # Run integration demonstration
    brain, sheaf_system = integrate_sheaf_with_brain_core()
    
    # Run validation demonstration
    demonstrate_sheaf_validation()