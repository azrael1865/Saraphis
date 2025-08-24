"""
Comprehensive unit tests for SheafStructures component.
Tests cellular sheaf structures, restriction maps, and topological operations.
"""

import unittest
import numpy as np
from typing import Dict, Any, Set, List, Tuple, Optional
import hashlib
import pickle
import time
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compression_systems.sheaf.sheaf_structures import (
    CellularSheaf,
    RestrictionMap
)


class TestCellularSheaf(unittest.TestCase):
    """Test suite for CellularSheaf class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sheaf = CellularSheaf()
        
        # Common test data
        self.test_cells = ["cell_0", "cell_1", "cell_2", "cell_3"]
        self.test_dimensions = [0, 1, 1, 2]
        self.test_coordinates = [(0, 0), (1, 0), (0, 1), (0.5, 0.5)]
    
    # ============= Initialization and Basic Structure Tests =============
    
    def test_initialization_empty_sheaf(self):
        """Test initialization of empty sheaf"""
        sheaf = CellularSheaf()
        self.assertEqual(len(sheaf.cells), 0)
        self.assertEqual(len(sheaf.topology), 0)
        self.assertEqual(len(sheaf.sections), 0)
        self.assertEqual(len(sheaf.restriction_maps), 0)
        self.assertTrue(sheaf.validate_connectivity())  # Empty sheaf is connected
    
    def test_add_cell_basic(self):
        """Test adding cells to sheaf"""
        self.sheaf.add_cell("cell_0", dimension=0)
        self.assertIn("cell_0", self.sheaf.cells)
        self.assertEqual(self.sheaf.cell_dimensions["cell_0"], 0)
        self.assertIn("cell_0", self.sheaf.topology)
        self.assertEqual(len(self.sheaf.topology["cell_0"]), 0)
    
    def test_add_cell_with_neighbors(self):
        """Test adding cells with neighbor relationships"""
        self.sheaf.add_cell("cell_0", dimension=0)
        self.sheaf.add_cell("cell_1", dimension=1, neighbors={"cell_0"})
        
        self.assertIn("cell_1", self.sheaf.cells)
        self.assertIn("cell_0", self.sheaf.topology["cell_1"])
        # Bidirectional relationship
        self.assertIn("cell_1", self.sheaf.topology["cell_0"])
    
    def test_add_cell_with_coordinates(self):
        """Test adding cells with spatial coordinates"""
        coords = (1.0, 2.0, 3.0)
        self.sheaf.add_cell("cell_0", dimension=2, coordinates=coords)
        
        self.assertIn("cell_0", self.sheaf.cell_coordinates)
        self.assertEqual(self.sheaf.cell_coordinates["cell_0"], coords)
    
    def test_add_cell_invalid_inputs(self):
        """Test adding cells with invalid inputs"""
        # Invalid cell ID type
        with self.assertRaises(TypeError):
            self.sheaf.add_cell(123, dimension=0)
        
        # Empty cell ID
        with self.assertRaises(ValueError):
            self.sheaf.add_cell("", dimension=0)
        
        # Duplicate cell ID
        self.sheaf.add_cell("cell_0", dimension=0)
        with self.assertRaises(ValueError):
            self.sheaf.add_cell("cell_0", dimension=1)
        
        # Invalid dimension
        with self.assertRaises(TypeError):
            self.sheaf.add_cell("cell_1", dimension="one")
        
        with self.assertRaises(ValueError):
            self.sheaf.add_cell("cell_1", dimension=-1)
        
        # Invalid neighbors type
        with self.assertRaises(TypeError):
            self.sheaf.add_cell("cell_2", dimension=0, neighbors=["cell_0"])
        
        # Invalid coordinates
        with self.assertRaises(TypeError):
            self.sheaf.add_cell("cell_3", dimension=0, coordinates="invalid")
    
    def test_remove_cell(self):
        """Test removing cells from sheaf"""
        # Build simple structure
        self.sheaf.add_cell("cell_0", dimension=0)
        self.sheaf.add_cell("cell_1", dimension=1, neighbors={"cell_0"})
        self.sheaf.set_section("cell_0", np.array([1, 2, 3]))
        
        # Remove cell
        self.sheaf.remove_cell("cell_0")
        
        self.assertNotIn("cell_0", self.sheaf.cells)
        self.assertNotIn("cell_0", self.sheaf.topology)
        self.assertNotIn("cell_0", self.sheaf.sections)
        self.assertNotIn("cell_0", self.sheaf.topology["cell_1"])
    
    def test_remove_cell_nonexistent(self):
        """Test removing nonexistent cell"""
        with self.assertRaises(ValueError):
            self.sheaf.remove_cell("nonexistent")
    
    def test_remove_cell_with_restrictions(self):
        """Test removing cell also removes associated restriction maps"""
        self.sheaf.add_cell("cell_0", dimension=0)
        self.sheaf.add_cell("cell_1", dimension=1, neighbors={"cell_0"})
        
        restriction = RestrictionMap("cell_0", "cell_1")
        self.sheaf.add_restriction_map("cell_0", "cell_1", restriction)
        
        self.sheaf.remove_cell("cell_0")
        self.assertEqual(len(self.sheaf.restriction_maps), 0)
    
    # ============= Section Management Tests =============
    
    def test_set_section_basic(self):
        """Test setting section data on cells"""
        self.sheaf.add_cell("cell_0", dimension=0)
        data = np.array([1, 2, 3])
        
        self.sheaf.set_section("cell_0", data)
        self.assertIn("cell_0", self.sheaf.sections)
        np.testing.assert_array_equal(self.sheaf.sections["cell_0"], data)
    
    def test_set_section_invalid_cell(self):
        """Test setting section on nonexistent cell"""
        with self.assertRaises(ValueError):
            self.sheaf.set_section("nonexistent", np.array([1, 2, 3]))
    
    def test_set_section_none_data(self):
        """Test setting None as section data"""
        self.sheaf.add_cell("cell_0", dimension=0)
        with self.assertRaises(ValueError):
            self.sheaf.set_section("cell_0", None)
    
    def test_set_section_type_consistency(self):
        """Test type consistency enforcement for sections"""
        self.sheaf.add_cell("cell_0", dimension=0)
        self.sheaf.add_cell("cell_1", dimension=1)
        
        # Set first section as numpy array
        self.sheaf.set_section("cell_0", np.array([1, 2, 3]))
        
        # Compatible type (different shape but same dtype)
        self.sheaf.set_section("cell_1", np.array([[1, 2], [3, 4]]))
        
        # Add new cell for incompatible type test
        self.sheaf.add_cell("cell_2", dimension=0)
        
        # Incompatible type should raise error
        with self.assertRaises(TypeError):
            self.sheaf.set_section("cell_2", {"key": "value"})
    
    def test_section_numeric_compatibility(self):
        """Test numeric type compatibility for sections"""
        self.sheaf.add_cell("cell_0", dimension=0)
        self.sheaf.add_cell("cell_1", dimension=0)
        
        # Set integer section
        self.sheaf.set_section("cell_0", 42)
        
        # Float should be compatible with int
        self.sheaf.set_section("cell_1", 3.14)
    
    # ============= Restriction Map Tests =============
    
    def test_add_restriction_map_basic(self):
        """Test adding restriction maps between cells"""
        self.sheaf.add_cell("cell_0", dimension=0)
        self.sheaf.add_cell("cell_1", dimension=1, neighbors={"cell_0"})
        
        restriction = RestrictionMap("cell_0", "cell_1")
        self.sheaf.add_restriction_map("cell_0", "cell_1", restriction)
        
        self.assertIn(("cell_0", "cell_1"), self.sheaf.restriction_maps)
        self.assertEqual(self.sheaf.restriction_maps[("cell_0", "cell_1")], restriction)
    
    def test_add_restriction_map_invalid_cells(self):
        """Test adding restriction map with invalid cells"""
        self.sheaf.add_cell("cell_0", dimension=0)
        
        # Source cell doesn't exist
        restriction = RestrictionMap("nonexistent", "cell_0")
        with self.assertRaises(ValueError):
            self.sheaf.add_restriction_map("nonexistent", "cell_0", restriction)
        
        # Target cell doesn't exist  
        restriction = RestrictionMap("cell_0", "nonexistent")
        with self.assertRaises(ValueError):
            self.sheaf.add_restriction_map("cell_0", "nonexistent", restriction)
    
    def test_add_restriction_map_non_neighbors(self):
        """Test adding restriction map between non-neighboring cells"""
        self.sheaf.add_cell("cell_0", dimension=0)
        self.sheaf.add_cell("cell_1", dimension=1)  # Not neighbors
        
        restriction = RestrictionMap("cell_0", "cell_1")
        with self.assertRaises(ValueError):
            self.sheaf.add_restriction_map("cell_0", "cell_1", restriction)
    
    def test_add_restriction_map_mismatched(self):
        """Test adding restriction map with mismatched cells"""
        self.sheaf.add_cell("cell_0", dimension=0)
        self.sheaf.add_cell("cell_1", dimension=1, neighbors={"cell_0"})
        
        # Restriction has different source/target
        restriction = RestrictionMap("cell_1", "cell_0")
        with self.assertRaises(ValueError):
            self.sheaf.add_restriction_map("cell_0", "cell_1", restriction)
    
    def test_get_restriction(self):
        """Test getting restriction maps"""
        self.sheaf.add_cell("cell_0", dimension=0)
        self.sheaf.add_cell("cell_1", dimension=1, neighbors={"cell_0"})
        
        restriction = RestrictionMap("cell_0", "cell_1")
        self.sheaf.add_restriction_map("cell_0", "cell_1", restriction)
        
        retrieved = self.sheaf.get_restriction("cell_0", "cell_1")
        self.assertEqual(retrieved, restriction)
        
        # Non-existent restriction
        self.assertIsNone(self.sheaf.get_restriction("cell_1", "cell_0"))
    
    def test_get_restriction_invalid_cells(self):
        """Test getting restriction with invalid cells"""
        with self.assertRaises(ValueError):
            self.sheaf.get_restriction("nonexistent", "cell_0")
    
    # ============= Overlap Region Tests =============
    
    def test_add_overlap_region(self):
        """Test adding overlap region information"""
        self.sheaf.add_cell("cell_0", dimension=0)
        self.sheaf.add_cell("cell_1", dimension=1)
        
        overlap_info = {"type": "intersection", "size": 10}
        self.sheaf.add_overlap_region("cell_0", "cell_1", overlap_info)
        
        # Check consistent ordering
        self.assertIn(("cell_0", "cell_1"), self.sheaf.overlap_regions)
        self.assertEqual(self.sheaf.overlap_regions[("cell_0", "cell_1")], overlap_info)
    
    def test_add_overlap_region_ordering(self):
        """Test overlap region ordering is consistent"""
        self.sheaf.add_cell("cell_0", dimension=0)
        self.sheaf.add_cell("cell_1", dimension=1)
        
        overlap_info = {"type": "intersection"}
        
        # Add with reversed order
        self.sheaf.add_overlap_region("cell_1", "cell_0", overlap_info)
        
        # Should be stored in consistent order
        self.assertIn(("cell_0", "cell_1"), self.sheaf.overlap_regions)
        self.assertNotIn(("cell_1", "cell_0"), self.sheaf.overlap_regions)
    
    def test_get_overlap_region(self):
        """Test getting overlap region information"""
        self.sheaf.add_cell("cell_0", dimension=0)
        self.sheaf.add_cell("cell_1", dimension=1)
        
        overlap_info = {"type": "intersection"}
        self.sheaf.add_overlap_region("cell_0", "cell_1", overlap_info)
        
        # Get with same order
        retrieved = self.sheaf.get_overlap_region("cell_0", "cell_1")
        self.assertEqual(retrieved, overlap_info)
        
        # Get with reversed order (should still work)
        retrieved = self.sheaf.get_overlap_region("cell_1", "cell_0")
        self.assertEqual(retrieved, overlap_info)
    
    # ============= Topology Tests =============
    
    def test_get_neighbors(self):
        """Test getting cell neighbors"""
        self.sheaf.add_cell("cell_0", dimension=0)
        self.sheaf.add_cell("cell_1", dimension=1, neighbors={"cell_0"})
        self.sheaf.add_cell("cell_2", dimension=1, neighbors={"cell_0"})
        
        neighbors = self.sheaf.get_neighbors("cell_0")
        self.assertEqual(neighbors, {"cell_1", "cell_2"})
        
        # Returns copy, not reference
        neighbors.add("new_cell")
        self.assertNotIn("new_cell", self.sheaf.topology["cell_0"])
    
    def test_get_neighbors_invalid_cell(self):
        """Test getting neighbors of invalid cell"""
        with self.assertRaises(ValueError):
            self.sheaf.get_neighbors("nonexistent")
    
    def test_validate_connectivity_connected(self):
        """Test connectivity validation for connected sheaf"""
        # Create connected structure
        self.sheaf.add_cell("cell_0", dimension=0)
        self.sheaf.add_cell("cell_1", dimension=1, neighbors={"cell_0"})
        self.sheaf.add_cell("cell_2", dimension=1, neighbors={"cell_1"})
        
        self.assertTrue(self.sheaf.validate_connectivity())
    
    def test_validate_connectivity_disconnected(self):
        """Test connectivity validation for disconnected sheaf"""
        # Create disconnected structure
        self.sheaf.add_cell("cell_0", dimension=0)
        self.sheaf.add_cell("cell_1", dimension=1, neighbors={"cell_0"})
        self.sheaf.add_cell("cell_2", dimension=0)  # Isolated
        self.sheaf.add_cell("cell_3", dimension=1, neighbors={"cell_2"})
        
        self.assertFalse(self.sheaf.validate_connectivity())
    
    def test_compute_euler_characteristic(self):
        """Test Euler characteristic computation"""
        # Create simple complex: vertex - edge - vertex
        self.sheaf.add_cell("v0", dimension=0)
        self.sheaf.add_cell("e0", dimension=1)
        self.sheaf.add_cell("v1", dimension=0)
        
        # χ = V - E + F = 2 - 1 + 0 = 1
        chi = self.sheaf.compute_euler_characteristic()
        self.assertEqual(chi, 1)
        
        # Add a face
        self.sheaf.add_cell("f0", dimension=2)
        
        # χ = V - E + F = 2 - 1 + 1 = 2
        chi = self.sheaf.compute_euler_characteristic()
        self.assertEqual(chi, 2)
    
    def test_get_boundary_cells(self):
        """Test boundary cell detection"""
        # Create structure with clear boundary
        self.sheaf.add_cell("center", dimension=2)
        self.sheaf.add_cell("edge1", dimension=1, neighbors={"center"})
        self.sheaf.add_cell("edge2", dimension=1, neighbors={"center"})
        self.sheaf.add_cell("corner", dimension=0, neighbors={"edge1"})
        
        boundary = self.sheaf.get_boundary_cells()
        
        # Cells with fewer neighbors than expected are likely boundary
        self.assertIn("corner", boundary)  # 0-dim with only 1 neighbor
        self.assertIn("edge1", boundary)   # 1-dim with only 2 neighbors
    
    # ============= Statistics Tests =============
    
    def test_get_sheaf_statistics(self):
        """Test comprehensive sheaf statistics"""
        # Build test structure
        self.sheaf.add_cell("cell_0", dimension=0)
        self.sheaf.add_cell("cell_1", dimension=1, neighbors={"cell_0"})
        self.sheaf.add_cell("cell_2", dimension=2, neighbors={"cell_1"})
        
        self.sheaf.set_section("cell_0", np.array([1, 2, 3]))
        self.sheaf.set_section("cell_1", np.array([4, 5, 6]))
        
        restriction = RestrictionMap("cell_0", "cell_1")
        self.sheaf.add_restriction_map("cell_0", "cell_1", restriction)
        
        stats = self.sheaf.get_sheaf_statistics()
        
        self.assertEqual(stats['total_cells'], 3)
        self.assertEqual(stats['total_sections'], 2)
        self.assertEqual(stats['total_restrictions'], 1)
        self.assertTrue(stats['is_connected'])
        self.assertIn('euler_characteristic', stats)
        self.assertIn('dimension_distribution', stats)
        self.assertIn('avg_neighbors', stats)
    
    # ============= Validation Tests =============
    
    def test_validate_structure_invalid_cells(self):
        """Test structure validation with invalid cells"""
        # Manually create invalid structure
        sheaf = CellularSheaf()
        sheaf.cells = {123}  # Invalid type
        
        with self.assertRaises(TypeError):
            sheaf._validate_structure()
    
    def test_validate_structure_invalid_topology(self):
        """Test structure validation with invalid topology"""
        sheaf = CellularSheaf()
        sheaf.cells = {"cell_0"}
        sheaf.topology = {"cell_0": ["cell_1"]}  # Should be set, not list
        
        with self.assertRaises(TypeError):
            sheaf._validate_structure()
    
    def test_validate_structure_unknown_neighbor(self):
        """Test structure validation with unknown neighbor"""
        sheaf = CellularSheaf()
        sheaf.cells = {"cell_0"}
        sheaf.topology = {"cell_0": {"nonexistent"}}
        
        with self.assertRaises(ValueError):
            sheaf._validate_structure()


class TestRestrictionMap(unittest.TestCase):
    """Test suite for RestrictionMap class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.source_cell = "cell_0"
        self.target_cell = "cell_1"
        self.restriction = RestrictionMap(self.source_cell, self.target_cell)
        
        # Common test data
        self.test_array = np.array([1, 2, 3, 4, 5])
        self.test_dict = {"a": 1, "b": 2, "c": 3}
        self.test_list = [1, 2, 3, 4, 5]
    
    # ============= Initialization Tests =============
    
    def test_initialization_basic(self):
        """Test basic restriction map initialization"""
        restriction = RestrictionMap("source", "target")
        self.assertEqual(restriction.source_cell, "source")
        self.assertEqual(restriction.target_cell, "target")
        self.assertEqual(restriction.restriction_type, "default")
        self.assertIsNotNone(restriction.restriction_func)
    
    def test_initialization_with_custom_function(self):
        """Test initialization with custom restriction function"""
        def custom_func(data):
            return data * 2
        
        restriction = RestrictionMap("source", "target", 
                                   restriction_func=custom_func,
                                   restriction_type="custom")
        
        self.assertEqual(restriction.restriction_func, custom_func)
        self.assertEqual(restriction.restriction_type, "custom")
    
    def test_initialization_with_overlap_region(self):
        """Test initialization with overlap region info"""
        overlap = {"slice_indices": (0, 3)}
        restriction = RestrictionMap("source", "target", overlap_region=overlap)
        
        self.assertEqual(restriction.overlap_region, overlap)
    
    def test_initialization_invalid_inputs(self):
        """Test initialization with invalid inputs"""
        # Invalid source cell
        with self.assertRaises(ValueError):
            RestrictionMap("", "target")
        
        with self.assertRaises(ValueError):
            RestrictionMap(None, "target")
        
        # Invalid target cell
        with self.assertRaises(ValueError):
            RestrictionMap("source", "")
        
        # Same source and target
        with self.assertRaises(ValueError):
            RestrictionMap("cell", "cell")
    
    # ============= Default Restriction Tests =============
    
    def test_default_restriction_array(self):
        """Test default restriction for numpy arrays"""
        result = self.restriction.apply(self.test_array)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertLess(len(result), len(self.test_array))
    
    def test_default_restriction_array_with_slice(self):
        """Test array restriction with slice indices"""
        self.restriction.set_overlap_region({"slice_indices": (1, 4)})
        result = self.restriction.apply(self.test_array)
        
        np.testing.assert_array_equal(result, np.array([2, 3, 4]))
    
    def test_default_restriction_array_with_mask(self):
        """Test array restriction with boolean mask"""
        mask = np.array([False, True, True, False, True])
        self.restriction.set_overlap_region({"mask": mask})
        result = self.restriction.apply(self.test_array)
        
        np.testing.assert_array_equal(result, np.array([2, 3, 5]))
    
    def test_default_restriction_2d_array(self):
        """Test default restriction for 2D arrays"""
        array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = self.restriction.apply(array_2d)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 2)
        self.assertLess(result.size, array_2d.size)
    
    def test_default_restriction_dict(self):
        """Test default restriction for dictionaries"""
        result = self.restriction.apply(self.test_dict)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result, self.test_dict)  # Default is shallow copy
    
    def test_default_restriction_dict_with_filter(self):
        """Test dictionary restriction with key filter"""
        self.restriction.set_overlap_region({"key_filter": ["a", "c"]})
        result = self.restriction.apply(self.test_dict)
        
        self.assertEqual(result, {"a": 1, "c": 3})
    
    def test_default_restriction_list(self):
        """Test default restriction for lists"""
        result = self.restriction.apply(self.test_list)
        
        self.assertIsInstance(result, list)
        self.assertLess(len(result), len(self.test_list))
    
    def test_default_restriction_tuple(self):
        """Test default restriction for tuples"""
        test_tuple = (1, 2, 3, 4, 5)
        result = self.restriction.apply(test_tuple)
        
        self.assertIsInstance(result, tuple)
        self.assertLess(len(result), len(test_tuple))
    
    def test_default_restriction_list_with_range(self):
        """Test list restriction with index range"""
        self.restriction.set_overlap_region({"index_range": (1, 3)})
        result = self.restriction.apply(self.test_list)
        
        self.assertEqual(result, [2, 3])
    
    def test_default_restriction_atomic(self):
        """Test default restriction for atomic data"""
        # Integer
        result = self.restriction.apply(42)
        self.assertEqual(result, 42)
        
        # Float
        result = self.restriction.apply(3.14)
        self.assertEqual(result, 3.14)
        
        # String
        result = self.restriction.apply("test")
        self.assertEqual(result, "test")
    
    def test_apply_none_data(self):
        """Test applying restriction to None data"""
        with self.assertRaises(ValueError):
            self.restriction.apply(None)
    
    # ============= Caching Tests =============
    
    def test_apply_caching(self):
        """Test that apply caches results"""
        # First call - cache miss
        result1 = self.restriction.apply(self.test_array)
        self.assertEqual(self.restriction.cache_misses, 1)
        self.assertEqual(self.restriction.cache_hits, 0)
        
        # Second call - cache hit
        result2 = self.restriction.apply(self.test_array)
        self.assertEqual(self.restriction.cache_misses, 1)
        self.assertEqual(self.restriction.cache_hits, 1)
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_cache_clearing_on_overlap_change(self):
        """Test that cache clears when overlap region changes"""
        # Apply and cache
        self.restriction.apply(self.test_array)
        initial_cache_size = len(self.restriction.application_cache)
        self.assertGreater(initial_cache_size, 0)
        
        # Change overlap region
        self.restriction.set_overlap_region({"slice_indices": (0, 2)})
        
        # Cache should be cleared
        self.assertEqual(len(self.restriction.application_cache), 0)
    
    # ============= Compatibility Tests =============
    
    def test_check_compatibility_identical(self):
        """Test compatibility check for identical data"""
        data1 = np.array([1, 2, 3, 4, 5])
        data2 = np.array([1, 2, 3, 4, 5])
        
        self.assertTrue(self.restriction.check_compatibility(data1, data2))
    
    def test_check_compatibility_different(self):
        """Test compatibility check for different data"""
        data1 = np.array([1, 2, 3, 4, 5])
        data2 = np.array([6, 7, 8, 9, 10])
        
        # With default restriction, center portions will be different
        self.assertFalse(self.restriction.check_compatibility(data1, data2))
    
    def test_check_compatibility_none(self):
        """Test compatibility check with None data"""
        with self.assertRaises(ValueError):
            self.restriction.check_compatibility(None, self.test_array)
        
        with self.assertRaises(ValueError):
            self.restriction.check_compatibility(self.test_array, None)
    
    def test_check_compatibility_caching(self):
        """Test that compatibility check uses caching"""
        data1 = np.array([1, 2, 3])
        data2 = np.array([4, 5, 6])
        
        # First check
        result1 = self.restriction.check_compatibility(data1, data2)
        
        # Mark validation time
        self.restriction.last_validation_time = time.time()
        
        # Second check (should use cache)
        initial_cache_size = len(self.restriction.compatibility_cache)
        result2 = self.restriction.check_compatibility(data1, data2)
        
        self.assertEqual(result1, result2)
        self.assertGreaterEqual(len(self.restriction.compatibility_cache), initial_cache_size)
    
    # ============= Data Equality Tests =============
    
    def test_data_equal_arrays(self):
        """Test data equality for arrays"""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 3.0])
        arr3 = np.array([1.0, 2.0, 3.1])
        
        self.assertTrue(self.restriction._data_equal(arr1, arr2))
        self.assertFalse(self.restriction._data_equal(arr1, arr3))
        
        # Different shapes
        arr4 = np.array([[1.0, 2.0, 3.0]])
        self.assertFalse(self.restriction._data_equal(arr1, arr4))
    
    def test_data_equal_dicts(self):
        """Test data equality for dictionaries"""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"a": 1, "b": 2}
        dict3 = {"a": 1, "b": 3}
        
        self.assertTrue(self.restriction._data_equal(dict1, dict2))
        self.assertFalse(self.restriction._data_equal(dict1, dict3))
    
    def test_data_equal_lists(self):
        """Test data equality for lists"""
        list1 = [1, 2, 3]
        list2 = [1, 2, 3]
        list3 = [1, 2, 4]
        
        self.assertTrue(self.restriction._data_equal(list1, list2))
        self.assertFalse(self.restriction._data_equal(list1, list3))
    
    def test_data_equal_mixed_types(self):
        """Test data equality for mixed types"""
        self.assertFalse(self.restriction._data_equal([1, 2, 3], (1, 2, 3)))
        self.assertFalse(self.restriction._data_equal(np.array([1, 2, 3]), [1, 2, 3]))
    
    # ============= Hash Computation Tests =============
    
    def test_compute_data_hash_array(self):
        """Test hash computation for arrays"""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        arr3 = np.array([4, 5, 6])
        
        hash1 = self.restriction._compute_data_hash(arr1)
        hash2 = self.restriction._compute_data_hash(arr2)
        hash3 = self.restriction._compute_data_hash(arr3)
        
        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)
    
    def test_compute_data_hash_dict(self):
        """Test hash computation for dictionaries"""
        dict1 = {"b": 2, "a": 1}  # Different order
        dict2 = {"a": 1, "b": 2}
        dict3 = {"a": 1, "c": 3}
        
        hash1 = self.restriction._compute_data_hash(dict1)
        hash2 = self.restriction._compute_data_hash(dict2)
        hash3 = self.restriction._compute_data_hash(dict3)
        
        self.assertEqual(hash1, hash2)  # Order shouldn't matter
        self.assertNotEqual(hash1, hash3)
    
    def test_compute_data_hash_fallback(self):
        """Test hash computation fallback for unhashable objects"""
        # Create unhashable object
        class UnhashableObject:
            def __init__(self):
                self.data = [1, 2, 3]
        
        obj = UnhashableObject()
        hash_result = self.restriction._compute_data_hash(obj)
        
        self.assertIsInstance(hash_result, str)
        self.assertTrue(hash_result.startswith("obj_"))
    
    # ============= Statistics Tests =============
    
    def test_get_restriction_statistics(self):
        """Test restriction map statistics"""
        # Perform some operations
        self.restriction.apply(self.test_array)
        self.restriction.apply(self.test_array)  # Cache hit
        self.restriction.apply(self.test_dict)   # Cache miss
        
        stats = self.restriction.get_restriction_statistics()
        
        self.assertEqual(stats['source_cell'], self.source_cell)
        self.assertEqual(stats['target_cell'], self.target_cell)
        self.assertEqual(stats['cache_hits'], 1)
        self.assertEqual(stats['cache_misses'], 2)
        self.assertIn('cache_hit_rate', stats)
        self.assertIn('cached_applications', stats)
    
    # ============= Validation Tests =============
    
    def test_validate_restriction_properties(self):
        """Test validation of restriction properties"""
        test_data = np.array([1, 2, 3, 4, 5])
        
        properties = self.restriction.validate_restriction_properties(test_data)
        
        # Well-defined: same input produces same output
        self.assertTrue(properties['well_defined'])
        
        # Type preserving: output has compatible type
        self.assertTrue(properties['type_preserving'])
        
        # Note: idempotent test creates temporary self-restriction which may fail
        self.assertIn('idempotent', properties)
    
    def test_validate_restriction_properties_failure(self):
        """Test validation with data that causes failures"""
        # Create restriction with function that's not well-defined
        counter = [0]
        def unstable_func(data):
            counter[0] += 1
            return data * counter[0]
        
        restriction = RestrictionMap("source", "target", 
                                   restriction_func=unstable_func)
        
        test_data = np.array([1, 2, 3])
        properties = restriction.validate_restriction_properties(test_data)
        
        # Not well-defined since output changes
        self.assertFalse(properties['well_defined'])
    
    # ============= Edge Cases =============
    
    def test_restrict_empty_array(self):
        """Test restricting empty array"""
        empty_array = np.array([])
        
        # Should handle gracefully
        result = self.restriction.apply(empty_array)
        self.assertEqual(len(result), 0)
    
    def test_restrict_single_element(self):
        """Test restricting single element array"""
        single = np.array([42])
        result = self.restriction.apply(single)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 1)
    
    def test_restrict_high_dimensional_array(self):
        """Test restricting high dimensional array"""
        arr_3d = np.random.rand(3, 4, 5)
        result = self.restriction.apply(arr_3d)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertLess(result.size, arr_3d.size)
    
    def test_invalid_slice_indices(self):
        """Test invalid slice indices in overlap region"""
        self.restriction.set_overlap_region({"slice_indices": "invalid"})
        
        with self.assertRaises(ValueError):
            self.restriction.apply(self.test_array)
    
    def test_invalid_mask(self):
        """Test invalid mask in overlap region"""
        self.restriction.set_overlap_region({"mask": "invalid"})
        
        with self.assertRaises(ValueError):
            self.restriction.apply(self.test_array)
    
    def test_invalid_key_filter(self):
        """Test invalid key filter in overlap region"""
        self.restriction.set_overlap_region({"key_filter": "invalid"})
        
        with self.assertRaises(ValueError):
            self.restriction.apply(self.test_dict)
    
    def test_invalid_index_range(self):
        """Test invalid index range in overlap region"""
        self.restriction.set_overlap_region({"index_range": "invalid"})
        
        with self.assertRaises(ValueError):
            self.restriction.apply(self.test_list)


class TestIntegration(unittest.TestCase):
    """Integration tests for sheaf structures working together"""
    
    def test_full_sheaf_workflow(self):
        """Test complete workflow with sheaf and restrictions"""
        # Create sheaf structure
        sheaf = CellularSheaf()
        
        # Add cells
        sheaf.add_cell("vertex_0", dimension=0, coordinates=(0, 0))
        sheaf.add_cell("vertex_1", dimension=0, coordinates=(1, 0))
        sheaf.add_cell("edge_0", dimension=1, neighbors={"vertex_0", "vertex_1"})
        
        # Add sections
        sheaf.set_section("vertex_0", np.array([1.0, 2.0, 3.0]))
        sheaf.set_section("vertex_1", np.array([4.0, 5.0, 6.0]))
        sheaf.set_section("edge_0", np.array([2.5, 3.5, 4.5]))
        
        # Add restrictions
        restriction_v0_e0 = RestrictionMap("vertex_0", "edge_0")
        restriction_v1_e0 = RestrictionMap("vertex_1", "edge_0")
        
        sheaf.add_restriction_map("vertex_0", "edge_0", restriction_v0_e0)
        sheaf.add_restriction_map("vertex_1", "edge_0", restriction_v1_e0)
        
        # Apply restrictions
        v0_data = sheaf.sections["vertex_0"]
        v0_restricted = restriction_v0_e0.apply(v0_data)
        
        self.assertIsInstance(v0_restricted, np.ndarray)
        self.assertLess(len(v0_restricted), len(v0_data))
        
        # Check statistics
        stats = sheaf.get_sheaf_statistics()
        self.assertEqual(stats['total_cells'], 3)
        self.assertEqual(stats['total_sections'], 3)
        self.assertEqual(stats['total_restrictions'], 2)
        self.assertTrue(stats['is_connected'])
    
    def test_complex_topology(self):
        """Test complex topological structure"""
        sheaf = CellularSheaf()
        
        # Create a triangular complex
        vertices = ["v0", "v1", "v2"]
        edges = ["e01", "e12", "e02"]
        face = "f012"
        
        # Add vertices
        for i, v in enumerate(vertices):
            sheaf.add_cell(v, dimension=0, coordinates=(i, 0))
        
        # Add edges connecting vertices
        sheaf.add_cell("e01", dimension=1, neighbors={"v0", "v1"})
        sheaf.add_cell("e12", dimension=1, neighbors={"v1", "v2"})
        sheaf.add_cell("e02", dimension=1, neighbors={"v0", "v2"})
        
        # Add face
        sheaf.add_cell(face, dimension=2, neighbors=set(edges))
        
        # Verify connectivity
        self.assertTrue(sheaf.validate_connectivity())
        
        # Check Euler characteristic: V - E + F = 3 - 3 + 1 = 1
        chi = sheaf.compute_euler_characteristic()
        self.assertEqual(chi, 1)
        
        # Add data and restrictions
        for v in vertices:
            sheaf.set_section(v, np.random.rand(5))
        
        for e in edges:
            sheaf.set_section(e, np.random.rand(3))
        
        sheaf.set_section(face, np.random.rand(2))
        
        # Create restriction from vertices to edges
        for v in vertices:
            neighbors = sheaf.get_neighbors(v)
            for e in neighbors:
                if e in edges:  # If neighbor is an edge
                    restriction = RestrictionMap(v, e)
                    sheaf.add_restriction_map(v, e, restriction)
        
        # Verify structure
        self.assertGreater(len(sheaf.restriction_maps), 0)
        self.assertEqual(len(sheaf.sections), 7)  # 3 vertices + 3 edges + 1 face


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)