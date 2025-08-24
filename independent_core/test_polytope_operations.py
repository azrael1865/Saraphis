"""
Comprehensive test suite for PolytopeOperations component.
Tests polytope creation, geometric operations, validation, and edge cases.
"""

import unittest
import torch
import math
import itertools
import tempfile
from typing import List, Dict, Tuple, Optional
import numpy as np

# Import the components to test
from compression_systems.tropical.polytope_operations import (
    Polytope,
    PolytopeOperations,
    PolytopeSimplification,
    TROPICAL_EPSILON
)


class TestPolytopeCreation(unittest.TestCase):
    """Test Polytope dataclass creation and validation."""
    
    def test_valid_polytope_creation(self):
        """Test creating valid polytopes."""
        # 2D triangle
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        self.assertEqual(P.vertices.shape, torch.Size([3, 2]))
        self.assertEqual(P.dimension, 2)
        self.assertTrue(P.is_bounded)
        self.assertIsNone(P._faces)
        self.assertIsNone(P._volume)
        self.assertIsNone(P._edge_graph)
        self.assertIsNone(P._facets)
    
    def test_1d_polytope(self):
        """Test 1D polytope (interval) creation."""
        vertices = torch.tensor([[0.0], [5.0]])
        P = Polytope(vertices, dimension=1, is_bounded=True)
        
        self.assertEqual(P.vertices.shape, torch.Size([2, 1]))
        self.assertEqual(P.dimension, 1)
    
    def test_3d_polytope(self):
        """Test 3D polytope creation."""
        # Tetrahedron vertices
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        P = Polytope(vertices, dimension=3, is_bounded=True)
        
        self.assertEqual(P.vertices.shape, torch.Size([4, 3]))
        self.assertEqual(P.dimension, 3)
    
    def test_high_dimensional_polytope(self):
        """Test high-dimensional polytope creation."""
        dimension = 10
        vertices = torch.randn(20, dimension)
        P = Polytope(vertices, dimension=dimension, is_bounded=False)
        
        self.assertEqual(P.vertices.shape, torch.Size([20, dimension]))
        self.assertEqual(P.dimension, dimension)
        self.assertFalse(P.is_bounded)
    
    def test_single_vertex_polytope(self):
        """Test polytope with single vertex."""
        vertices = torch.tensor([[1.0, 2.0, 3.0]])
        P = Polytope(vertices, dimension=3, is_bounded=True)
        
        self.assertEqual(P.vertices.shape, torch.Size([1, 3]))


class TestPolytopeValidation(unittest.TestCase):
    """Test Polytope validation in __post_init__."""
    
    def test_invalid_vertex_type(self):
        """Test that non-tensor vertices raise TypeError."""
        with self.assertRaises(TypeError) as context:
            Polytope([[0, 0], [1, 0]], dimension=2, is_bounded=True)
        self.assertIn("Vertices must be torch.Tensor", str(context.exception))
    
    def test_invalid_vertex_dimension(self):
        """Test that 1D or 3D vertex tensors raise ValueError."""
        # 1D tensor
        with self.assertRaises(ValueError) as context:
            Polytope(torch.tensor([1.0, 2.0, 3.0]), dimension=3, is_bounded=True)
        self.assertIn("Vertices must be 2D tensor", str(context.exception))
        
        # 3D tensor
        with self.assertRaises(ValueError) as context:
            Polytope(torch.randn(2, 3, 4), dimension=3, is_bounded=True)
        self.assertIn("Vertices must be 2D tensor", str(context.exception))
    
    def test_empty_vertices(self):
        """Test that empty vertex set raises ValueError."""
        with self.assertRaises(ValueError) as context:
            Polytope(torch.empty(0, 2), dimension=2, is_bounded=True)
        self.assertIn("Polytope must have at least one vertex", str(context.exception))
    
    def test_dimension_mismatch(self):
        """Test that vertex dimension must match polytope dimension."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        with self.assertRaises(ValueError) as context:
            Polytope(vertices, dimension=3, is_bounded=True)
        self.assertIn("Vertex dimension", str(context.exception))
        self.assertIn("doesn't match polytope dimension", str(context.exception))
    
    def test_invalid_dimension_type(self):
        """Test that non-integer dimension raises TypeError."""
        vertices = torch.tensor([[0.0, 0.0]])
        with self.assertRaises(TypeError) as context:
            Polytope(vertices, dimension=2.5, is_bounded=True)
        self.assertIn("Dimension must be int", str(context.exception))
    
    def test_invalid_dimension_value(self):
        """Test that non-positive dimension raises ValueError."""
        vertices = torch.tensor([[0.0]])
        with self.assertRaises(ValueError) as context:
            Polytope(vertices, dimension=0, is_bounded=True)
        self.assertIn("Dimension must be positive", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            Polytope(vertices, dimension=-1, is_bounded=True)
        self.assertIn("Dimension must be positive", str(context.exception))
    
    def test_invalid_is_bounded_type(self):
        """Test that non-boolean is_bounded raises TypeError."""
        vertices = torch.tensor([[0.0, 0.0]])
        with self.assertRaises(TypeError) as context:
            Polytope(vertices, dimension=2, is_bounded="yes")
        self.assertIn("is_bounded must be bool", str(context.exception))
    
    def test_nan_vertices(self):
        """Test that NaN vertices raise ValueError."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, float('nan')]])
        with self.assertRaises(ValueError) as context:
            Polytope(vertices, dimension=2, is_bounded=True)
        self.assertIn("Vertices contain NaN", str(context.exception))
    
    def test_inf_vertices(self):
        """Test that infinite vertices raise ValueError."""
        vertices = torch.tensor([[0.0, 0.0], [float('inf'), 0.0]])
        with self.assertRaises(ValueError) as context:
            Polytope(vertices, dimension=2, is_bounded=True)
        self.assertIn("Vertices contain infinity", str(context.exception))


class TestPolytopeFaceComputation(unittest.TestCase):
    """Test face enumeration methods."""
    
    def test_compute_vertices_as_0_faces(self):
        """Test that 0-faces are the vertices."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        faces = P.compute_faces(0)
        self.assertEqual(len(faces), 3)
        for i, face in enumerate(faces):
            self.assertEqual(face.tolist(), [i])
    
    def test_compute_edges_2d(self):
        """Test edge computation for 2D polytope."""
        # Square
        vertices = torch.tensor([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]
        ])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        edges = P.compute_faces(1)
        self.assertGreaterEqual(len(edges), 4)  # At least 4 edges for square
    
    def test_face_dimension_out_of_range(self):
        """Test that invalid face dimension raises ValueError."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        with self.assertRaises(ValueError) as context:
            P.compute_faces(-1)
        self.assertIn("Face dimension", str(context.exception))
        self.assertIn("out of range", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            P.compute_faces(3)
        self.assertIn("Face dimension", str(context.exception))
    
    def test_face_caching(self):
        """Test that faces are cached after first computation."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        # First computation
        faces1 = P.compute_faces(0)
        self.assertIsNotNone(P._faces)
        self.assertIn(0, P._faces)
        
        # Second computation should use cache
        faces2 = P.compute_faces(0)
        self.assertEqual(len(faces1), len(faces2))


class TestPolytopeContainsPoint(unittest.TestCase):
    """Test point containment checking."""
    
    def test_point_inside_2d_triangle(self):
        """Test points inside 2D triangle."""
        vertices = torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        # Interior point
        self.assertTrue(P.contains_point(torch.tensor([1.0, 0.5])))
        # Centroid
        self.assertTrue(P.contains_point(torch.tensor([1.0, 2.0/3.0])))
    
    def test_point_on_boundary(self):
        """Test points on polytope boundary."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        # Vertex
        self.assertTrue(P.contains_point(torch.tensor([0.0, 0.0])))
        # Edge midpoint
        self.assertTrue(P.contains_point(torch.tensor([0.5, 0.0])))
        self.assertTrue(P.contains_point(torch.tensor([0.5, 0.5])))
    
    def test_point_outside(self):
        """Test points outside polytope."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        self.assertFalse(P.contains_point(torch.tensor([2.0, 0.0])))
        self.assertFalse(P.contains_point(torch.tensor([0.0, 2.0])))
        self.assertFalse(P.contains_point(torch.tensor([-1.0, 0.0])))
        self.assertFalse(P.contains_point(torch.tensor([1.0, 1.0])))
    
    def test_contains_point_invalid_input(self):
        """Test that invalid point inputs raise errors."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        # Non-tensor point
        with self.assertRaises(TypeError) as context:
            P.contains_point([0.5, 0.5])
        self.assertIn("Point must be torch.Tensor", str(context.exception))
        
        # Wrong dimension
        with self.assertRaises(ValueError) as context:
            P.contains_point(torch.tensor([0.5]))
        self.assertIn("Point dimension", str(context.exception))
        self.assertIn("doesn't match polytope dimension", str(context.exception))
    
    def test_contains_point_1d(self):
        """Test point containment in 1D interval."""
        vertices = torch.tensor([[0.0], [5.0]])
        P = Polytope(vertices, dimension=1, is_bounded=True)
        
        self.assertTrue(P.contains_point(torch.tensor([2.5])))
        self.assertTrue(P.contains_point(torch.tensor([0.0])))
        self.assertTrue(P.contains_point(torch.tensor([5.0])))
        self.assertFalse(P.contains_point(torch.tensor([-1.0])))
        self.assertFalse(P.contains_point(torch.tensor([6.0])))


class TestPolytopeVolume(unittest.TestCase):
    """Test volume computation methods."""
    
    def test_1d_volume(self):
        """Test length computation for 1D interval."""
        vertices = torch.tensor([[2.0], [7.0]])
        P = Polytope(vertices, dimension=1, is_bounded=True)
        
        volume = P.compute_volume()
        self.assertAlmostEqual(volume, 5.0, places=6)
    
    def test_2d_area_square(self):
        """Test area computation for 2D square."""
        vertices = torch.tensor([
            [0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]
        ])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        area = P.compute_volume()
        self.assertAlmostEqual(area, 4.0, places=6)
    
    def test_2d_area_triangle(self):
        """Test area computation for 2D triangle."""
        vertices = torch.tensor([[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        area = P.compute_volume()
        self.assertAlmostEqual(area, 6.0, places=6)  # 0.5 * 4 * 3
    
    def test_volume_caching(self):
        """Test that volume is cached after first computation."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        self.assertIsNone(P._volume)
        volume1 = P.compute_volume()
        self.assertIsNotNone(P._volume)
        volume2 = P.compute_volume()
        self.assertEqual(volume1, volume2)


class TestPolytopeGetFacets(unittest.TestCase):
    """Test facet computation methods."""
    
    def test_2d_facets(self):
        """Test facet computation for 2D polytope (edges as facets)."""
        # Triangle
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        facets = P.get_facets()
        self.assertGreaterEqual(len(facets), 3)  # At least 3 edges
        
        for normal, offset in facets:
            self.assertEqual(normal.shape, torch.Size([2]))
            self.assertIsInstance(offset, float)
            # Normal should be unit vector
            self.assertAlmostEqual(torch.norm(normal).item(), 1.0, places=5)
    
    def test_facets_caching(self):
        """Test that facets are cached."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        self.assertIsNone(P._facets)
        facets1 = P.get_facets()
        self.assertIsNotNone(P._facets)
        facets2 = P.get_facets()
        self.assertEqual(len(facets1), len(facets2))


class TestPolytopeEdgeGraph(unittest.TestCase):
    """Test edge graph computation."""
    
    def test_edge_graph_triangle(self):
        """Test edge graph for triangle."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        graph = P.get_edge_graph()
        self.assertIsInstance(graph, dict)
        
        # Triangle should have each vertex connected to 2 others
        for vertex_idx in range(3):
            if vertex_idx in graph:
                self.assertGreaterEqual(len(graph[vertex_idx]), 1)
    
    def test_edge_graph_caching(self):
        """Test that edge graph is cached."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        self.assertIsNone(P._edge_graph)
        graph1 = P.get_edge_graph()
        self.assertIsNotNone(P._edge_graph)
        graph2 = P.get_edge_graph()
        self.assertEqual(graph1, graph2)


class TestPolytopeDeviceOperations(unittest.TestCase):
    """Test device movement operations."""
    
    def test_to_device_cpu(self):
        """Test moving polytope to CPU device."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        P_cpu = P.to_device(torch.device('cpu'))
        self.assertEqual(P_cpu.vertices.device.type, 'cpu')
        self.assertEqual(P_cpu.dimension, P.dimension)
        self.assertEqual(P_cpu.is_bounded, P.is_bounded)
    
    def test_pin_memory(self):
        """Test pinning memory for GPU transfer."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        P_pinned = P.pin_memory()
        self.assertEqual(P_pinned.dimension, P.dimension)
        self.assertEqual(P_pinned.is_bounded, P.is_bounded)


class TestPolytopeOperationsInit(unittest.TestCase):
    """Test PolytopeOperations initialization."""
    
    def test_default_initialization(self):
        """Test default initialization with CPU device."""
        ops = PolytopeOperations()
        self.assertEqual(ops.device.type, 'cpu')
    
    def test_explicit_device(self):
        """Test initialization with explicit device."""
        device = torch.device('cpu')
        ops = PolytopeOperations(device=device)
        self.assertEqual(ops.device, device)
    
    def test_invalid_device_type(self):
        """Test that invalid device type raises TypeError."""
        with self.assertRaises(TypeError) as context:
            PolytopeOperations(device="cpu")
        self.assertIn("Device must be torch.device", str(context.exception))


class TestConvexHull(unittest.TestCase):
    """Test convex hull computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ops = PolytopeOperations()
    
    def test_convex_hull_single_point(self):
        """Test convex hull of single point."""
        points = torch.tensor([[1.0, 2.0]])
        hull = self.ops.convex_hull(points)
        
        self.assertEqual(hull.vertices.shape, torch.Size([1, 2]))
        self.assertEqual(hull.dimension, 2)
        self.assertTrue(hull.is_bounded)
    
    def test_convex_hull_1d(self):
        """Test convex hull in 1D."""
        points = torch.tensor([[1.0], [5.0], [3.0], [2.0], [4.0]])
        hull = self.ops.convex_hull(points)
        
        self.assertEqual(hull.vertices.shape, torch.Size([2, 1]))
        self.assertEqual(hull.dimension, 1)
        # Should only keep min and max
        self.assertAlmostEqual(hull.vertices.min().item(), 1.0)
        self.assertAlmostEqual(hull.vertices.max().item(), 5.0)
    
    def test_convex_hull_2d_triangle(self):
        """Test convex hull of points forming triangle."""
        points = torch.tensor([
            [0.0, 0.0], [2.0, 0.0], [1.0, 2.0],
            [1.0, 0.5], [0.5, 0.5]  # Interior points
        ])
        hull = self.ops.convex_hull(points)
        
        self.assertLessEqual(hull.vertices.shape[0], 3)  # Should only keep triangle vertices
        self.assertEqual(hull.dimension, 2)
    
    def test_convex_hull_2d_square(self):
        """Test convex hull of points forming square."""
        points = torch.tensor([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
            [0.5, 0.5], [0.25, 0.25], [0.75, 0.75]  # Interior points
        ])
        hull = self.ops.convex_hull(points)
        
        self.assertLessEqual(hull.vertices.shape[0], 4)  # Should only keep corners
    
    def test_convex_hull_3d(self):
        """Test convex hull in 3D."""
        # Tetrahedron vertices plus interior points
        points = torch.tensor([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [0.25, 0.25, 0.25]  # Interior point
        ])
        hull = self.ops.convex_hull(points)
        
        self.assertEqual(hull.dimension, 3)
        self.assertTrue(hull.is_bounded)
    
    def test_convex_hull_invalid_input(self):
        """Test that invalid inputs raise appropriate errors."""
        # Non-tensor input
        with self.assertRaises(TypeError) as context:
            self.ops.convex_hull([[0, 0], [1, 0]])
        self.assertIn("Points must be torch.Tensor", str(context.exception))
        
        # Wrong dimension
        with self.assertRaises(ValueError) as context:
            self.ops.convex_hull(torch.tensor([1.0, 2.0, 3.0]))
        self.assertIn("Points must be 2D tensor", str(context.exception))
        
        # Empty points
        with self.assertRaises(ValueError) as context:
            self.ops.convex_hull(torch.empty(0, 2))
        self.assertIn("Cannot compute convex hull of empty", str(context.exception))
        
        # NaN points
        with self.assertRaises(ValueError) as context:
            self.ops.convex_hull(torch.tensor([[0.0, 0.0], [float('nan'), 1.0]]))
        self.assertIn("Points contain NaN", str(context.exception))
        
        # Infinite points
        with self.assertRaises(ValueError) as context:
            self.ops.convex_hull(torch.tensor([[0.0, 0.0], [float('inf'), 1.0]]))
        self.assertIn("Points contain infinity", str(context.exception))
    
    def test_convex_hull_device_transfer(self):
        """Test that points are moved to operations device."""
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        hull = self.ops.convex_hull(points)
        
        self.assertEqual(hull.vertices.device, self.ops.device)


class TestMinkowskiSum(unittest.TestCase):
    """Test Minkowski sum operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ops = PolytopeOperations()
    
    def test_minkowski_sum_squares(self):
        """Test Minkowski sum of two squares."""
        # Unit square
        vertices1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        P = Polytope(vertices1, dimension=2, is_bounded=True)
        
        # Half unit square
        vertices2 = torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
        Q = Polytope(vertices2, dimension=2, is_bounded=True)
        
        sum_polytope = self.ops.minkowski_sum(P, Q)
        
        self.assertEqual(sum_polytope.dimension, 2)
        self.assertTrue(sum_polytope.is_bounded)
        
        # Sum should be larger
        sum_volume = sum_polytope.compute_volume()
        P_volume = P.compute_volume()
        self.assertGreater(sum_volume, P_volume)
    
    def test_minkowski_sum_invalid_input(self):
        """Test that invalid inputs raise errors."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        # Non-polytope argument
        with self.assertRaises(TypeError) as context:
            self.ops.minkowski_sum(P, "not a polytope")
        self.assertIn("Both arguments must be Polytope", str(context.exception))
        
        # Different dimensions
        vertices_3d = torch.tensor([[0.0, 0.0, 0.0]])
        Q = Polytope(vertices_3d, dimension=3, is_bounded=True)
        
        with self.assertRaises(ValueError) as context:
            self.ops.minkowski_sum(P, Q)
        self.assertIn("Polytopes must have same dimension", str(context.exception))


class TestIntersection(unittest.TestCase):
    """Test polytope intersection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ops = PolytopeOperations()
    
    def test_intersection_overlapping_triangles(self):
        """Test intersection of overlapping triangles."""
        vertices1 = torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
        P = Polytope(vertices1, dimension=2, is_bounded=True)
        
        vertices2 = torch.tensor([[1.0, 0.0], [3.0, 0.0], [2.0, 2.0]])
        Q = Polytope(vertices2, dimension=2, is_bounded=True)
        
        intersection = self.ops.intersection(P, Q)
        
        self.assertIsNotNone(intersection)
        self.assertEqual(intersection.dimension, 2)
    
    def test_intersection_disjoint(self):
        """Test intersection of disjoint polytopes."""
        vertices1 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        P = Polytope(vertices1, dimension=2, is_bounded=True)
        
        vertices2 = torch.tensor([[2.0, 2.0], [3.0, 2.0], [2.0, 3.0]])
        Q = Polytope(vertices2, dimension=2, is_bounded=True)
        
        intersection = self.ops.intersection(P, Q)
        
        self.assertIsNone(intersection)
    
    def test_intersection_invalid_input(self):
        """Test that invalid inputs raise errors."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        # Non-polytope argument
        with self.assertRaises(TypeError) as context:
            self.ops.intersection(P, "not a polytope")
        self.assertIn("Both arguments must be Polytope", str(context.exception))
        
        # Different dimensions
        vertices_3d = torch.tensor([[0.0, 0.0, 0.0]])
        Q = Polytope(vertices_3d, dimension=3, is_bounded=True)
        
        with self.assertRaises(ValueError) as context:
            self.ops.intersection(P, Q)
        self.assertIn("Polytopes must have same dimension", str(context.exception))


class TestProjection(unittest.TestCase):
    """Test polytope projection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ops = PolytopeOperations()
    
    def test_project_3d_to_2d(self):
        """Test projecting 3D cube to 2D."""
        # Create 3D unit cube
        vertices = []
        for x in [0, 1]:
            for y in [0, 1]:
                for z in [0, 1]:
                    vertices.append([float(x), float(y), float(z)])
        
        vertices_tensor = torch.tensor(vertices)
        P = Polytope(vertices_tensor, dimension=3, is_bounded=True)
        
        # Project onto xy-plane
        P_xy = self.ops.project(P, [0, 1])
        
        self.assertEqual(P_xy.dimension, 2)
        self.assertTrue(P_xy.is_bounded)
        
        # Should get a square with area ~1
        area = P_xy.compute_volume()
        self.assertAlmostEqual(area, 1.0, places=1)
    
    def test_project_single_dimension(self):
        """Test projecting to single dimension."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        # Project onto x-axis
        P_x = self.ops.project(P, [0])
        
        self.assertEqual(P_x.dimension, 1)
    
    def test_project_invalid_input(self):
        """Test that invalid inputs raise errors."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        # Non-polytope argument
        with self.assertRaises(TypeError) as context:
            self.ops.project("not a polytope", [0])
        self.assertIn("P must be Polytope", str(context.exception))
        
        # Non-list dimensions
        with self.assertRaises(TypeError) as context:
            self.ops.project(P, 0)
        self.assertIn("Dimensions must be list", str(context.exception))
        
        # Empty dimensions
        with self.assertRaises(ValueError) as context:
            self.ops.project(P, [])
        self.assertIn("Must specify at least one dimension", str(context.exception))
        
        # Invalid dimension index
        with self.assertRaises(ValueError) as context:
            self.ops.project(P, [2])
        self.assertIn("Dimension index 2 out of range", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            self.ops.project(P, [-1])
        self.assertIn("Dimension index -1 out of range", str(context.exception))
        
        # Non-integer dimension
        with self.assertRaises(TypeError) as context:
            self.ops.project(P, [0.5])
        self.assertIn("Dimension index must be int", str(context.exception))


class TestNormalFan(unittest.TestCase):
    """Test normal fan computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ops = PolytopeOperations()
    
    def test_normal_fan_triangle(self):
        """Test normal fan of 2D triangle."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        normal_fan = self.ops.compute_normal_fan(P)
        
        self.assertIsInstance(normal_fan, dict)
        self.assertEqual(len(normal_fan), 3)  # One for each vertex
        
        for vertex, normals in normal_fan.items():
            self.assertIsInstance(normals, torch.Tensor)
            self.assertEqual(normals.dim(), 2)
    
    def test_normal_fan_invalid_input(self):
        """Test that non-polytope input raises error."""
        with self.assertRaises(TypeError) as context:
            self.ops.compute_normal_fan("not a polytope")
        self.assertIn("P must be Polytope", str(context.exception))


class TestPolytopeSimplificationInit(unittest.TestCase):
    """Test PolytopeSimplification initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        simplifier = PolytopeSimplification()
        self.assertEqual(simplifier.tolerance, 1e-6)
        self.assertIsInstance(simplifier.ops, PolytopeOperations)
    
    def test_custom_tolerance(self):
        """Test initialization with custom tolerance."""
        simplifier = PolytopeSimplification(tolerance=1e-10)
        self.assertEqual(simplifier.tolerance, 1e-10)
    
    def test_invalid_tolerance_type(self):
        """Test that non-numeric tolerance raises TypeError."""
        with self.assertRaises(TypeError) as context:
            PolytopeSimplification(tolerance="small")
        self.assertIn("Tolerance must be numeric", str(context.exception))
    
    def test_invalid_tolerance_value(self):
        """Test that non-positive tolerance raises ValueError."""
        with self.assertRaises(ValueError) as context:
            PolytopeSimplification(tolerance=0)
        self.assertIn("Tolerance must be positive", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            PolytopeSimplification(tolerance=-1e-6)
        self.assertIn("Tolerance must be positive", str(context.exception))


class TestRemoveRedundantVertices(unittest.TestCase):
    """Test redundant vertex removal."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simplifier = PolytopeSimplification()
    
    def test_remove_interior_points(self):
        """Test removing interior points."""
        vertices = torch.tensor([
            [0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0],  # Corners
            [1.0, 1.0], [0.5, 0.5], [1.5, 1.5]  # Interior points
        ])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        P_simple = self.simplifier.remove_redundant_vertices(P)
        
        self.assertLessEqual(P_simple.vertices.shape[0], 4)  # Should only keep corners
    
    def test_remove_collinear_points(self):
        """Test removing collinear points."""
        vertices = torch.tensor([
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],  # Collinear
            [2.0, 2.0], [0.0, 2.0]
        ])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        P_simple = self.simplifier.remove_redundant_vertices(P)
        
        self.assertLess(P_simple.vertices.shape[0], P.vertices.shape[0])
    
    def test_remove_redundant_invalid_input(self):
        """Test that non-polytope input raises error."""
        with self.assertRaises(TypeError) as context:
            self.simplifier.remove_redundant_vertices("not a polytope")
        self.assertIn("P must be Polytope", str(context.exception))


class TestApproximatePolytope(unittest.TestCase):
    """Test polytope approximation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simplifier = PolytopeSimplification()
    
    def test_approximate_with_fewer_vertices(self):
        """Test approximating polytope with fewer vertices."""
        # Create polytope with many vertices
        n_vertices = 20
        vertices = torch.randn(n_vertices, 2)
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        # Approximate with max 5 vertices
        P_approx = self.simplifier.approximate_polytope(P, max_vertices=5)
        
        self.assertLessEqual(P_approx.vertices.shape[0], 5)
        self.assertEqual(P_approx.dimension, 2)
    
    def test_approximate_already_simple(self):
        """Test that simple polytope is unchanged."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        P_approx = self.simplifier.approximate_polytope(P, max_vertices=5)
        
        self.assertEqual(P_approx.vertices.shape[0], P.vertices.shape[0])
    
    def test_approximate_invalid_input(self):
        """Test that invalid inputs raise errors."""
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        # Non-polytope
        with self.assertRaises(TypeError) as context:
            self.simplifier.approximate_polytope("not a polytope", 5)
        self.assertIn("P must be Polytope", str(context.exception))
        
        # Non-integer max_vertices
        with self.assertRaises(TypeError) as context:
            self.simplifier.approximate_polytope(P, 5.5)
        self.assertIn("max_vertices must be int", str(context.exception))
        
        # Non-positive max_vertices
        with self.assertRaises(ValueError) as context:
            self.simplifier.approximate_polytope(P, 0)
        self.assertIn("max_vertices must be positive", str(context.exception))


class TestFindIntegerPoints(unittest.TestCase):
    """Test integer point finding."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simplifier = PolytopeSimplification()
    
    def test_find_integer_points_2d_triangle(self):
        """Test finding integer points in 2D triangle."""
        vertices = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        integer_points = self.simplifier.find_integer_points(P)
        
        self.assertGreater(integer_points.shape[0], 0)
        
        # Check all points are integers
        self.assertTrue(torch.allclose(integer_points, integer_points.round()))
        
        # Check all points are inside polytope
        for i in range(integer_points.shape[0]):
            self.assertTrue(P.contains_point(integer_points[i]))
    
    def test_find_integer_points_1d(self):
        """Test finding integer points in 1D interval."""
        vertices = torch.tensor([[1.5], [5.7]])
        P = Polytope(vertices, dimension=1, is_bounded=True)
        
        integer_points = self.simplifier.find_integer_points(P)
        
        # Should find 2, 3, 4, 5
        self.assertEqual(integer_points.shape[0], 4)
    
    def test_find_integer_points_empty(self):
        """Test polytope with no integer points."""
        # Small triangle around (0.5, 0.5)
        vertices = torch.tensor([[0.4, 0.4], [0.6, 0.4], [0.5, 0.6]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        integer_points = self.simplifier.find_integer_points(P)
        
        self.assertEqual(integer_points.shape[0], 0)
        self.assertEqual(integer_points.shape[1], 2)  # Correct dimension
    
    def test_find_integer_points_invalid_input(self):
        """Test that non-polytope input raises error."""
        with self.assertRaises(TypeError) as context:
            self.simplifier.find_integer_points("not a polytope")
        self.assertIn("P must be Polytope", str(context.exception))


class TestEhrhartPolynomial(unittest.TestCase):
    """Test Ehrhart polynomial computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simplifier = PolytopeSimplification()
    
    def test_ehrhart_polynomial_2d(self):
        """Test Ehrhart polynomial for 2D polytope."""
        # Unit square
        vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        coeffs = self.simplifier.compute_ehrhart_polynomial(P)
        
        self.assertIsInstance(coeffs, list)
        self.assertEqual(len(coeffs), 3)  # Degree 2 for 2D polytope
    
    def test_ehrhart_polynomial_invalid_input(self):
        """Test that non-polytope input raises error."""
        with self.assertRaises(TypeError) as context:
            self.simplifier.compute_ehrhart_polynomial("not a polytope")
        self.assertIn("P must be Polytope", str(context.exception))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios."""
    
    def test_degenerate_polytope(self):
        """Test handling of degenerate polytopes."""
        ops = PolytopeOperations()
        
        # All points are the same
        points = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        hull = ops.convex_hull(points)
        
        self.assertEqual(hull.vertices.shape[0], 1)
    
    def test_large_polytope(self):
        """Test handling of large polytopes."""
        ops = PolytopeOperations()
        
        # Many random points
        points = torch.randn(1000, 3)
        hull = ops.convex_hull(points)
        
        self.assertEqual(hull.dimension, 3)
        self.assertTrue(hull.is_bounded)
    
    def test_high_dimensional_polytope(self):
        """Test high-dimensional polytope operations."""
        ops = PolytopeOperations()
        
        # 10-dimensional simplex
        dimension = 10
        vertices = torch.eye(dimension + 1, dimension)
        P = Polytope(vertices, dimension=dimension, is_bounded=True)
        
        # Test containment
        point = torch.ones(dimension) * 0.05
        self.assertTrue(P.contains_point(point))
        
        # Test projection
        P_proj = ops.project(P, [0, 1, 2])
        self.assertEqual(P_proj.dimension, 3)
    
    def test_numerical_precision(self):
        """Test handling of numerical precision issues."""
        # Create vertices that are very close
        vertices = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0 + 1e-10, 0.0],  # Very close to previous
            [0.0, 1.0]
        ])
        P = Polytope(vertices, dimension=2, is_bounded=True)
        
        # Should still work without errors
        volume = P.compute_volume()
        self.assertGreater(volume, 0)


class TestPerformance(unittest.TestCase):
    """Test performance requirements."""
    
    def test_convex_hull_performance(self):
        """Test that convex hull meets performance requirements."""
        import time
        ops = PolytopeOperations()
        
        # Test with 10,000 3D points
        points = torch.randn(10000, 3)
        
        start = time.time()
        hull = ops.convex_hull(points)
        elapsed = time.time() - start
        
        # Should complete in under 500ms
        self.assertLess(elapsed, 0.5, f"Convex hull took {elapsed:.3f}s, expected < 0.5s")
    
    def test_containment_performance(self):
        """Test point containment performance."""
        import time
        
        # Create a complex polytope
        vertices = torch.randn(100, 5)
        P = Polytope(vertices, dimension=5, is_bounded=True)
        
        # Test many points
        test_points = torch.randn(1000, 5)
        
        start = time.time()
        for i in range(test_points.shape[0]):
            P.contains_point(test_points[i])
        elapsed = time.time() - start
        
        # Should be reasonably fast
        self.assertLess(elapsed, 5.0, f"Containment checks took {elapsed:.3f}s, expected < 5s")


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)