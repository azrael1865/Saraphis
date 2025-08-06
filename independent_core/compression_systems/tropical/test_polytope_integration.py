#!/usr/bin/env python3
"""
Integration test demonstrating polytope operations with tropical polynomials
for neural network compression.
"""

import torch
import numpy as np

try:
    from polytope_operations import Polytope, PolytopeOperations, PolytopeSimplification
    from tropical_polynomial import TropicalPolynomial, TropicalMonomial
    from tropical_core import TROPICAL_EPSILON
except ImportError:
    from independent_core.compression_systems.tropical.polytope_operations import (
        Polytope, PolytopeOperations, PolytopeSimplification
    )
    from independent_core.compression_systems.tropical.tropical_polynomial import (
        TropicalPolynomial, TropicalMonomial
    )
    from independent_core.compression_systems.tropical.tropical_core import TROPICAL_EPSILON


def simulate_neural_layer_compression():
    """
    Simulate compressing a neural network layer using polytope operations.
    """
    print("="*60)
    print("Neural Network Layer Compression via Polytope Operations")
    print("="*60)
    
    # Step 1: Create tropical polynomial representing neural layer
    print("\n1. Creating tropical polynomial from neural layer...")
    
    # Simulate weights as tropical monomials
    # In real use, these would come from analyzing actual layer weights
    monomials = []
    for i in range(20):  # 20 neurons
        # Random coefficient (weight magnitude)
        coeff = np.random.randn() * 0.5
        
        # Random sparse exponents (activation pattern)
        exponents = {}
        for j in np.random.choice(5, size=np.random.randint(1, 4), replace=False):
            exponents[int(j)] = np.random.randint(1, 3)
        
        monomials.append(TropicalMonomial(coeff, exponents))
    
    polynomial = TropicalPolynomial(monomials, num_variables=5)
    print(f"  Created polynomial with {len(polynomial.monomials)} monomials")
    
    # Step 2: Extract Newton polytope
    print("\n2. Extracting Newton polytope...")
    newton_vertices = polynomial.newton_polytope()
    polytope = Polytope(newton_vertices, dimension=5, is_bounded=True)
    print(f"  Newton polytope has {polytope.vertices.shape[0]} vertices")
    print(f"  Polytope dimension: {polytope.dimension}")
    
    # Step 3: Analyze polytope structure
    print("\n3. Analyzing polytope structure...")
    
    # Compute volume (relates to model capacity)
    volume = polytope.compute_volume()
    print(f"  Original volume: {volume:.4f}")
    
    # Get edge structure (connectivity of neurons)
    edge_graph = polytope.get_edge_graph()
    total_edges = sum(len(neighbors) for neighbors in edge_graph.values()) // 2
    print(f"  Edge connectivity: {total_edges} edges")
    
    # Step 4: Simplify polytope for compression
    print("\n4. Applying polytope simplification...")
    simplifier = PolytopeSimplification(tolerance=1e-4)
    
    # Remove redundant vertices
    simplified = simplifier.remove_redundant_vertices(polytope)
    print(f"  After removing redundant: {simplified.vertices.shape[0]} vertices")
    
    # Approximate with fewer vertices (compression)
    target_vertices = 8
    compressed = simplifier.approximate_polytope(simplified, max_vertices=target_vertices)
    print(f"  After approximation: {compressed.vertices.shape[0]} vertices")
    
    # Step 5: Compute compression metrics
    print("\n5. Compression metrics:")
    compressed_volume = compressed.compute_volume()
    
    compression_ratio = polytope.vertices.shape[0] / compressed.vertices.shape[0]
    volume_preservation = compressed_volume / max(volume, 1e-10)
    
    print(f"  Vertex compression ratio: {compression_ratio:.2f}x")
    print(f"  Volume preservation: {volume_preservation:.2%}")
    print(f"  Parameters reduced: {polytope.vertices.shape[0]} → {compressed.vertices.shape[0]}")
    
    # Step 6: Project to lower dimensions (feature extraction)
    print("\n6. Projecting to lower dimensions...")
    ops = PolytopeOperations()
    
    # Project to 2D for visualization
    projected_2d = ops.project(compressed, [0, 1])
    print(f"  2D projection: {projected_2d.vertices.shape[0]} vertices")
    area_2d = projected_2d.compute_volume()
    print(f"  2D area: {area_2d:.4f}")
    
    # Project to 3D for analysis
    projected_3d = ops.project(compressed, [0, 1, 2])
    print(f"  3D projection: {projected_3d.vertices.shape[0]} vertices")
    volume_3d = projected_3d.compute_volume()
    print(f"  3D volume: {volume_3d:.4f}")
    
    # Step 7: Analyze integer points (quantization)
    print("\n7. Integer point analysis for quantization...")
    integer_points = simplifier.find_integer_points(projected_3d)
    print(f"  Integer points in 3D projection: {integer_points.shape[0]}")
    
    if integer_points.shape[0] > 0:
        # Compute Ehrhart polynomial (counts integer points in dilations)
        ehrhart_coeffs = simplifier.compute_ehrhart_polynomial(projected_3d)
        print(f"  Ehrhart polynomial coefficients: {[f'{c:.2f}' for c in ehrhart_coeffs]}")
    
    # Step 8: Combine polytopes (layer fusion)
    print("\n8. Simulating layer fusion via Minkowski sum...")
    
    # Create second layer polytope
    vertices2 = torch.randn(5, 5) * 0.3
    polytope2 = Polytope(vertices2, dimension=5, is_bounded=True)
    
    # Compute Minkowski sum (represents combined layer)
    combined = ops.minkowski_sum(compressed, polytope2)
    print(f"  Combined polytope: {combined.vertices.shape[0]} vertices")
    
    # Check intersection (shared representations)
    intersection = ops.intersection(polytope, polytope2)
    if intersection:
        print(f"  Intersection found: {intersection.vertices.shape[0]} vertices")
    else:
        print("  No intersection (independent representations)")
    
    print("\n" + "="*60)
    print("Compression Summary:")
    print(f"  Original complexity: {len(polynomial.monomials)} monomials")
    print(f"  Geometric reduction: {polytope.vertices.shape[0]} → {compressed.vertices.shape[0]} vertices")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Quality preserved: {volume_preservation:.1%}")
    print("="*60)
    
    return compressed, compression_ratio


def test_gpu_acceleration():
    """Test GPU acceleration if available."""
    print("\n" + "="*60)
    print("GPU Acceleration Test")
    print("="*60)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        ops_gpu = PolytopeOperations(device=device)
        ops_cpu = PolytopeOperations(device=torch.device('cpu'))
        
        # Generate test data
        n_points = 5000
        dimension = 10
        points_cpu = torch.randn(n_points, dimension)
        points_gpu = points_cpu.to(device)
        
        # Time CPU computation
        import time
        start = time.time()
        hull_cpu = ops_cpu.convex_hull(points_cpu)
        cpu_time = (time.time() - start) * 1000
        
        # Time GPU computation
        start = time.time()
        hull_gpu = ops_gpu.convex_hull(points_gpu)
        gpu_time = (time.time() - start) * 1000
        
        print(f"\nConvex hull of {n_points} points in {dimension}D:")
        print(f"  CPU time: {cpu_time:.2f}ms")
        print(f"  GPU time: {gpu_time:.2f}ms")
        print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
        
        # Test polytope operations on GPU
        polytope_gpu = hull_gpu.to_device(device)
        test_point = torch.randn(dimension, device=device)
        
        start = time.time()
        contains = polytope_gpu.contains_point(test_point)
        gpu_contains_time = (time.time() - start) * 1000
        
        print(f"\nPoint containment check:")
        print(f"  GPU time: {gpu_contains_time:.4f}ms")
        print(f"  Result: {'Inside' if contains else 'Outside'}")
    else:
        print("\n⚠ GPU not available, skipping GPU tests")
    
    print("="*60)


def test_high_dimensional_compression():
    """Test compression in high dimensions."""
    print("\n" + "="*60)
    print("High-Dimensional Compression Test")
    print("="*60)
    
    dimensions = [10, 25, 50]
    
    for dim in dimensions:
        print(f"\n{dim}-dimensional polytope:")
        
        # Create random polytope
        n_vertices = min(dim * 3, 100)
        vertices = torch.randn(n_vertices, dim) * 0.5
        polytope = Polytope(vertices, dimension=dim, is_bounded=True)
        
        # Simplify
        simplifier = PolytopeSimplification(tolerance=1e-3)
        target = min(dim + 5, n_vertices // 2)
        compressed = simplifier.approximate_polytope(polytope, max_vertices=target)
        
        compression_ratio = n_vertices / compressed.vertices.shape[0]
        print(f"  Vertices: {n_vertices} → {compressed.vertices.shape[0]}")
        print(f"  Compression: {compression_ratio:.2f}x")
        
        # Test operations
        ops = PolytopeOperations()
        
        # Project to 3D
        if dim > 3:
            proj_3d = ops.project(compressed, list(range(3)))
            vol_3d = proj_3d.compute_volume()
            print(f"  3D projection volume: {vol_3d:.4f}")
    
    print("="*60)


if __name__ == "__main__":
    print("Polytope Operations Integration Test")
    print("Demonstrating neural network compression via geometric methods\n")
    
    # Run main compression simulation
    compressed_polytope, ratio = simulate_neural_layer_compression()
    
    # Test GPU acceleration
    test_gpu_acceleration()
    
    # Test high-dimensional cases
    test_high_dimensional_compression()
    
    print("\n✅ Integration test complete!")
    print(f"Achieved {ratio:.2f}x compression with polytope simplification")