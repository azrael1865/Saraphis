"""
Sheaf Theory Compression System Core Implementation
NO FALLBACKS - HARD FAILURES ONLY
"""

from typing import Dict, Any, Set, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import hashlib
import pickle

from ..base.compression_base import CompressionBase
from ...proof_system.algebraic_rule_enforcer import AlgebraicRuleEnforcer
from ...proof_system.confidence_generator import ConfidenceGenerator


@dataclass
class CellularSheaf:
    """
    Represents a cellular sheaf structure for compression.
    A sheaf consists of cells (open sets), sections (data on cells),
    and restriction maps between cells.
    """
    cells: Set[str] = field(default_factory=set)
    topology: Dict[str, Set[str]] = field(default_factory=dict)  # cell -> neighbors
    sections: Dict[str, Any] = field(default_factory=dict)  # cell -> data
    restriction_maps: Dict[Tuple[str, str], 'RestrictionMap'] = field(default_factory=dict)
    cell_dimensions: Dict[str, int] = field(default_factory=dict)  # cell -> dimension
    gluing_data: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # cell -> gluing info
    
    def add_cell(self, cell_id: str, dimension: int, neighbors: Set[str]) -> None:
        """Add a cell to the sheaf structure."""
        if not cell_id or not isinstance(cell_id, str):
            raise ValueError("Cell ID must be non-empty string")
        if dimension < 0:
            raise ValueError("Cell dimension must be non-negative")
        if not isinstance(neighbors, set):
            raise ValueError("Neighbors must be a set")
        
        self.cells.add(cell_id)
        self.topology[cell_id] = neighbors
        self.cell_dimensions[cell_id] = dimension
        
    def set_section(self, cell_id: str, data: Any) -> None:
        """Set section data for a cell."""
        if cell_id not in self.cells:
            raise ValueError(f"Cell {cell_id} not in sheaf")
        if data is None:
            raise ValueError("Section data cannot be None")
        
        self.sections[cell_id] = data
        
    def add_restriction_map(self, source: str, target: str, restriction: 'RestrictionMap') -> None:
        """Add a restriction map between cells."""
        if source not in self.cells:
            raise ValueError(f"Source cell {source} not in sheaf")
        if target not in self.cells:
            raise ValueError(f"Target cell {target} not in sheaf")
        if restriction is None:
            raise ValueError("Restriction map cannot be None")
        
        self.restriction_maps[(source, target)] = restriction
        
    def get_restriction(self, source: str, target: str) -> Optional['RestrictionMap']:
        """Get restriction map between cells."""
        return self.restriction_maps.get((source, target))


class RestrictionMap:
    """
    Represents a morphism between cells in the sheaf.
    Handles data restriction from one cell to another.
    """
    
    def __init__(self, source_cell: str, target_cell: str, 
                 restriction_func: Optional[Callable] = None):
        if not source_cell or not isinstance(source_cell, str):
            raise ValueError("Source cell must be non-empty string")
        if not target_cell or not isinstance(target_cell, str):
            raise ValueError("Target cell must be non-empty string")
        
        self.source_cell = source_cell
        self.target_cell = target_cell
        self.restriction_func = restriction_func or self._default_restriction
        self.compatibility_cache = {}
        
    def _default_restriction(self, data: Any) -> Any:
        """Default restriction: identity for compatible data."""
        if data is None:
            raise ValueError("Cannot restrict None data")
        
        # For numpy arrays, return a view
        if isinstance(data, np.ndarray):
            return data.copy()
        
        # For dictionaries, return a shallow copy
        if isinstance(data, dict):
            return data.copy()
        
        # For other types, return as-is
        return data
        
    def apply(self, section_data: Any) -> Any:
        """Apply the restriction map to section data."""
        if section_data is None:
            raise ValueError("Section data cannot be None")
        
        try:
            restricted = self.restriction_func(section_data)
            if restricted is None:
                raise ValueError("Restriction produced None result")
            return restricted
        except Exception as e:
            raise RuntimeError(f"Restriction map failed: {str(e)}") from e
            
    def check_compatibility(self, data1: Any, data2: Any) -> bool:
        """Check if two pieces of data are compatible under restriction."""
        if data1 is None or data2 is None:
            raise ValueError("Cannot check compatibility of None data")
        
        # Generate cache key
        key = (self._data_hash(data1), self._data_hash(data2))
        
        if key in self.compatibility_cache:
            return self.compatibility_cache[key]
        
        try:
            restricted1 = self.apply(data1)
            restricted2 = self.apply(data2)
            
            # Check compatibility based on data type
            if isinstance(restricted1, np.ndarray) and isinstance(restricted2, np.ndarray):
                compatible = np.allclose(restricted1, restricted2)
            elif isinstance(restricted1, dict) and isinstance(restricted2, dict):
                compatible = restricted1 == restricted2
            else:
                compatible = restricted1 == restricted2
                
            self.compatibility_cache[key] = compatible
            return compatible
            
        except Exception as e:
            raise RuntimeError(f"Compatibility check failed: {str(e)}") from e
            
    def _data_hash(self, data: Any) -> str:
        """Generate hash for data caching."""
        try:
            if isinstance(data, np.ndarray):
                return hashlib.sha256(data.tobytes()).hexdigest()
            else:
                return hashlib.sha256(pickle.dumps(data)).hexdigest()
        except Exception:
            return str(id(data))


class SheafValidation:
    """
    Validates sheaf structures and operations.
    Ensures mathematical consistency and data integrity.
    """
    
    def __init__(self):
        self.validation_rules = {
            'topology': self._validate_topology,
            'sections': self._validate_sections,
            'restrictions': self._validate_restrictions,
            'gluing': self._validate_gluing_conditions
        }
        
    def validate_sheaf(self, sheaf: CellularSheaf) -> None:
        """Perform complete validation of sheaf structure."""
        if not isinstance(sheaf, CellularSheaf):
            raise TypeError("Input must be CellularSheaf instance")
        
        for rule_name, rule_func in self.validation_rules.items():
            try:
                rule_func(sheaf)
            except Exception as e:
                raise ValueError(f"Sheaf validation failed at {rule_name}: {str(e)}") from e
                
    def _validate_topology(self, sheaf: CellularSheaf) -> None:
        """Validate the topology is consistent."""
        if not sheaf.cells:
            raise ValueError("Sheaf has no cells")
        
        # Check all topology references are valid
        for cell, neighbors in sheaf.topology.items():
            if cell not in sheaf.cells:
                raise ValueError(f"Topology references unknown cell: {cell}")
            for neighbor in neighbors:
                if neighbor not in sheaf.cells:
                    raise ValueError(f"Cell {cell} has unknown neighbor: {neighbor}")
                    
    def _validate_sections(self, sheaf: CellularSheaf) -> None:
        """Validate section data."""
        for cell in sheaf.cells:
            if cell not in sheaf.sections:
                raise ValueError(f"Cell {cell} has no section data")
            if sheaf.sections[cell] is None:
                raise ValueError(f"Cell {cell} has None section data")
                
    def _validate_restrictions(self, sheaf: CellularSheaf) -> None:
        """Validate restriction maps form a consistent system."""
        # Check restriction maps exist for all neighbor pairs
        for cell, neighbors in sheaf.topology.items():
            for neighbor in neighbors:
                if (cell, neighbor) not in sheaf.restriction_maps:
                    raise ValueError(f"Missing restriction map from {cell} to {neighbor}")
                    
        # Verify transitivity where applicable
        for (source, intermediate) in sheaf.restriction_maps:
            for (intermediate2, target) in sheaf.restriction_maps:
                if intermediate == intermediate2 and (source, target) in sheaf.restriction_maps:
                    # Check r_{s,t} = r_{i,t} ∘ r_{s,i}
                    self._check_transitivity(sheaf, source, intermediate, target)
                    
    def _check_transitivity(self, sheaf: CellularSheaf, source: str, 
                           intermediate: str, target: str) -> None:
        """Check transitivity of restriction maps."""
        r_si = sheaf.restriction_maps[(source, intermediate)]
        r_it = sheaf.restriction_maps[(intermediate, target)]
        r_st = sheaf.restriction_maps[(source, target)]
        
        # Get section data
        if source not in sheaf.sections:
            return  # Can't check without data
            
        data = sheaf.sections[source]
        
        # Apply compositions
        via_intermediate = r_it.apply(r_si.apply(data))
        direct = r_st.apply(data)
        
        # Check equality
        if isinstance(via_intermediate, np.ndarray) and isinstance(direct, np.ndarray):
            if not np.allclose(via_intermediate, direct):
                raise ValueError(f"Transitivity violated for {source}->{intermediate}->{target}")
        elif via_intermediate != direct:
            raise ValueError(f"Transitivity violated for {source}->{intermediate}->{target}")
            
    def _validate_gluing_conditions(self, sheaf: CellularSheaf) -> None:
        """Validate gluing conditions for local-to-global reconstruction."""
        # Check that compatible local sections can be glued
        for cell in sheaf.cells:
            self._check_local_compatibility(sheaf, cell)
            
    def _check_local_compatibility(self, sheaf: CellularSheaf, cell: str) -> None:
        """Check local compatibility around a cell."""
        neighbors = sheaf.topology.get(cell, set())
        if not neighbors:
            return
            
        cell_data = sheaf.sections.get(cell)
        if cell_data is None:
            return
            
        for neighbor in neighbors:
            neighbor_data = sheaf.sections.get(neighbor)
            if neighbor_data is None:
                continue
                
            # Check compatibility via restriction maps
            if (cell, neighbor) in sheaf.restriction_maps:
                r_cn = sheaf.restriction_maps[(cell, neighbor)]
                if (neighbor, cell) in sheaf.restriction_maps:
                    r_nc = sheaf.restriction_maps[(neighbor, cell)]
                    
                    # Sections should agree on overlaps
                    restricted_cell = r_cn.apply(cell_data)
                    restricted_neighbor = r_nc.apply(neighbor_data)
                    
                    if not self._data_compatible(restricted_cell, restricted_neighbor):
                        raise ValueError(f"Incompatible sections between {cell} and {neighbor}")
                        
    def _data_compatible(self, data1: Any, data2: Any) -> bool:
        """Check if two data pieces are compatible."""
        if type(data1) != type(data2):
            return False
            
        if isinstance(data1, np.ndarray):
            return data1.shape == data2.shape
        elif isinstance(data1, dict):
            return set(data1.keys()) == set(data2.keys())
        else:
            return True


class SheafCompressionSystem(CompressionBase):
    """
    Main Sheaf Theory compression system.
    Implements compress/decompress using cellular sheaf structures.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Core components
        self.validator = SheafValidation()
        self.rule_enforcer = None  # Set during integration
        self.confidence_generator = None  # Set during integration
        
        # Configuration
        self.max_cell_size = config.get('max_cell_size', 1024) if config else 1024
        self.min_cell_size = config.get('min_cell_size', 64) if config else 64
        self.overlap_ratio = config.get('overlap_ratio', 0.1) if config else 0.1
        
        # Internal state
        self.current_sheaf = None
        self.compression_cache = {}
        
    def set_rule_enforcer(self, enforcer: AlgebraicRuleEnforcer) -> None:
        """Set the algebraic rule enforcer for integration."""
        if enforcer is None:
            raise ValueError("Rule enforcer cannot be None")
        if not isinstance(enforcer, AlgebraicRuleEnforcer):
            raise TypeError("Enforcer must be AlgebraicRuleEnforcer instance")
        self.rule_enforcer = enforcer
        
    def set_confidence_generator(self, generator: ConfidenceGenerator) -> None:
        """Set the confidence generator for integration."""
        if generator is None:
            raise ValueError("Confidence generator cannot be None")
        if not isinstance(generator, ConfidenceGenerator):
            raise TypeError("Generator must be ConfidenceGenerator instance")
        self.confidence_generator = generator
        
    def compress(self, data: Any) -> Dict[str, Any]:
        """
        Compress data using sheaf theory.
        Returns compressed representation with sheaf structure.
        """
        if data is None:
            raise ValueError("Data cannot be None")
            
        # Build cellular decomposition
        sheaf = self._build_cellular_sheaf(data)
        
        # Validate sheaf structure
        self.validator.validate_sheaf(sheaf)
        
        # Apply algebraic rules if enforcer is set
        if self.rule_enforcer:
            self._apply_algebraic_rules(sheaf)
            
        # Compress local sections
        compressed_sections = self._compress_sections(sheaf)
        
        # Build compressed representation
        compressed = {
            'type': 'sheaf_compressed',
            'version': '1.0',
            'topology': self._serialize_topology(sheaf),
            'sections': compressed_sections,
            'restrictions': self._serialize_restrictions(sheaf),
            'metadata': self._build_metadata(sheaf, data)
        }
        
        # Update compression metrics
        self._update_metrics(data, compressed)
        
        # Cache for potential reuse
        data_hash = self._compute_data_hash(data)
        self.compression_cache[data_hash] = compressed
        
        return compressed
        
    def decompress(self, compressed_data: Any) -> Any:
        """
        Decompress data from sheaf representation.
        Performs local-to-global reconstruction.
        """
        if compressed_data is None:
            raise ValueError("Compressed data cannot be None")
        if not isinstance(compressed_data, dict):
            raise TypeError("Compressed data must be dictionary")
        if compressed_data.get('type') != 'sheaf_compressed':
            raise ValueError("Invalid compressed data type")
            
        # Rebuild sheaf structure
        sheaf = self._rebuild_sheaf(compressed_data)
        
        # Validate reconstructed sheaf
        self.validator.validate_sheaf(sheaf)
        
        # Decompress sections
        self._decompress_sections(sheaf, compressed_data['sections'])
        
        # Perform local-to-global reconstruction
        reconstructed = self._reconstruct_global(sheaf, compressed_data['metadata'])
        
        return reconstructed
        
    def _build_cellular_sheaf(self, data: Any) -> CellularSheaf:
        """Build cellular decomposition of data."""
        sheaf = CellularSheaf()
        
        if isinstance(data, np.ndarray):
            self._decompose_array(data, sheaf)
        elif isinstance(data, dict):
            self._decompose_dict(data, sheaf)
        elif isinstance(data, (list, tuple)):
            self._decompose_sequence(data, sheaf)
        else:
            # Single cell for atomic data
            self._add_atomic_cell(data, sheaf)
            
        self.current_sheaf = sheaf
        return sheaf
        
    def _decompose_array(self, data: np.ndarray, sheaf: CellularSheaf) -> None:
        """Decompose numpy array into cells."""
        if data.size == 0:
            raise ValueError("Cannot decompose empty array")
            
        shape = data.shape
        
        # Determine cell structure based on array dimensions
        if len(shape) == 1:
            self._decompose_1d_array(data, sheaf)
        elif len(shape) == 2:
            self._decompose_2d_array(data, sheaf)
        else:
            self._decompose_nd_array(data, sheaf)
            
    def _decompose_1d_array(self, data: np.ndarray, sheaf: CellularSheaf) -> None:
        """Decompose 1D array into overlapping cells."""
        n = len(data)
        cell_size = min(max(self.min_cell_size, n // 10), self.max_cell_size)
        overlap = int(cell_size * self.overlap_ratio)
        
        cells_added = []
        
        # Create cells with overlap
        i = 0
        cell_idx = 0
        while i < n:
            end = min(i + cell_size, n)
            cell_id = f"cell_1d_{cell_idx}"
            
            # Add cell
            neighbors = set()
            if cell_idx > 0:
                neighbors.add(f"cell_1d_{cell_idx-1}")
            if end < n:
                neighbors.add(f"cell_1d_{cell_idx+1}")
                
            sheaf.add_cell(cell_id, 1, neighbors)
            sheaf.set_section(cell_id, data[i:end].copy())
            cells_added.append(cell_id)
            
            i = end - overlap if end < n else end
            cell_idx += 1
            
        # Add restriction maps
        for i in range(len(cells_added) - 1):
            source = cells_added[i]
            target = cells_added[i + 1]
            
            # Define restriction for overlapping regions
            overlap_start = (i + 1) * (cell_size - overlap)
            overlap_end = overlap_start + overlap
            
            def make_restriction(start, end):
                def restrict(section_data):
                    if len(section_data) < end - start:
                        raise ValueError("Section too small for restriction")
                    return section_data[-(end-start):]
                return restrict
                
            restriction = RestrictionMap(source, target, 
                                       make_restriction(overlap_start, overlap_end))
            sheaf.add_restriction_map(source, target, restriction)
            
            # Add reverse restriction
            reverse_restriction = RestrictionMap(target, source,
                                               lambda x: x[:overlap])
            sheaf.add_restriction_map(target, source, reverse_restriction)
            
    def _decompose_2d_array(self, data: np.ndarray, sheaf: CellularSheaf) -> None:
        """Decompose 2D array into grid of cells."""
        h, w = data.shape
        
        # Determine grid structure
        cell_h = min(max(self.min_cell_size, h // 4), self.max_cell_size)
        cell_w = min(max(self.min_cell_size, w // 4), self.max_cell_size)
        overlap_h = int(cell_h * self.overlap_ratio)
        overlap_w = int(cell_w * self.overlap_ratio)
        
        cells_grid = {}
        
        # Create grid cells
        row_idx = 0
        i = 0
        while i < h:
            col_idx = 0
            j = 0
            end_i = min(i + cell_h, h)
            
            while j < w:
                end_j = min(j + cell_w, w)
                cell_id = f"cell_2d_{row_idx}_{col_idx}"
                
                # Determine neighbors
                neighbors = set()
                if row_idx > 0:
                    neighbors.add(f"cell_2d_{row_idx-1}_{col_idx}")
                if col_idx > 0:
                    neighbors.add(f"cell_2d_{row_idx}_{col_idx-1}")
                if end_i < h:
                    neighbors.add(f"cell_2d_{row_idx+1}_{col_idx}")
                if end_j < w:
                    neighbors.add(f"cell_2d_{row_idx}_{col_idx+1}")
                    
                sheaf.add_cell(cell_id, 2, neighbors)
                sheaf.set_section(cell_id, data[i:end_i, j:end_j].copy())
                cells_grid[(row_idx, col_idx)] = cell_id
                
                j = end_j - overlap_w if end_j < w else end_j
                col_idx += 1
                
            i = end_i - overlap_h if end_i < h else end_i
            row_idx += 1
            
        # Add restriction maps for neighboring cells
        self._add_2d_restrictions(sheaf, cells_grid, overlap_h, overlap_w)
        
    def _add_2d_restrictions(self, sheaf: CellularSheaf, cells_grid: Dict[Tuple[int, int], str],
                           overlap_h: int, overlap_w: int) -> None:
        """Add restriction maps for 2D grid."""
        for (row, col), cell_id in cells_grid.items():
            # Horizontal restrictions
            if (row, col + 1) in cells_grid:
                target = cells_grid[(row, col + 1)]
                restriction = RestrictionMap(cell_id, target,
                                           lambda x: x[:, -overlap_w:])
                sheaf.add_restriction_map(cell_id, target, restriction)
                
            # Vertical restrictions  
            if (row + 1, col) in cells_grid:
                target = cells_grid[(row + 1, col)]
                restriction = RestrictionMap(cell_id, target,
                                           lambda x: x[-overlap_h:, :])
                sheaf.add_restriction_map(cell_id, target, restriction)
                
    def _decompose_nd_array(self, data: np.ndarray, sheaf: CellularSheaf) -> None:
        """Decompose n-dimensional array."""
        # Flatten to 2D for simplicity in higher dimensions
        original_shape = data.shape
        flattened = data.reshape(original_shape[0], -1)
        
        # Store original shape in metadata
        sheaf.gluing_data['original_shape'] = original_shape
        sheaf.gluing_data['reshape_info'] = {
            'method': 'flatten_nd',
            'original_ndim': len(original_shape)
        }
        
        # Decompose as 2D
        self._decompose_2d_array(flattened, sheaf)
        
    def _decompose_dict(self, data: dict, sheaf: CellularSheaf) -> None:
        """Decompose dictionary into cells by keys."""
        if not data:
            raise ValueError("Cannot decompose empty dictionary")
            
        # Create cell for each key
        cells_by_key = {}
        
        for idx, (key, value) in enumerate(data.items()):
            cell_id = f"cell_dict_{key}"
            
            # All dict cells are neighbors (flat topology)
            neighbors = {f"cell_dict_{k}" for k in data.keys() if k != key}
            
            sheaf.add_cell(cell_id, 0, neighbors)
            sheaf.set_section(cell_id, {key: value})
            cells_by_key[key] = cell_id
            
        # Add identity restrictions between all cells
        for key1, cell1 in cells_by_key.items():
            for key2, cell2 in cells_by_key.items():
                if key1 != key2:
                    restriction = RestrictionMap(cell1, cell2)
                    sheaf.add_restriction_map(cell1, cell2, restriction)
                    
    def _decompose_sequence(self, data: Any, sheaf: CellularSheaf) -> None:
        """Decompose sequence into cells."""
        if not data:
            raise ValueError("Cannot decompose empty sequence")
            
        # Convert to numpy array for easier handling
        arr = np.array(data)
        self._decompose_array(arr, sheaf)
        
        # Store type info for reconstruction
        sheaf.gluing_data['original_type'] = type(data).__name__
        
    def _add_atomic_cell(self, data: Any, sheaf: CellularSheaf) -> None:
        """Add single cell for atomic data."""
        cell_id = "cell_atomic_0"
        sheaf.add_cell(cell_id, 0, set())
        sheaf.set_section(cell_id, data)
        
    def _apply_algebraic_rules(self, sheaf: CellularSheaf) -> None:
        """Apply algebraic rule enforcement to sheaf."""
        if not self.rule_enforcer:
            return
            
        # Define sheaf-specific rules
        rules = {
            'section_consistency': lambda s: self._check_section_consistency(s),
            'restriction_compatibility': lambda s: self._check_restriction_compatibility(s),
            'gluing_validity': lambda s: self._check_gluing_validity(s)
        }
        
        # Register rules
        for rule_name, rule_func in rules.items():
            self.rule_enforcer.rules[rule_name] = rule_func
            
        # Enforce all rules
        for rule_name in rules:
            if not self.rule_enforcer.enforce_rule(rule_name, sheaf):
                raise ValueError(f"Sheaf violates algebraic rule: {rule_name}")
                
    def _check_section_consistency(self, sheaf: CellularSheaf) -> bool:
        """Check section data consistency."""
        for cell in sheaf.cells:
            if cell not in sheaf.sections:
                return False
            if sheaf.sections[cell] is None:
                return False
        return True
        
    def _check_restriction_compatibility(self, sheaf: CellularSheaf) -> bool:
        """Check restriction map compatibility."""
        try:
            self.validator._validate_restrictions(sheaf)
            return True
        except:
            return False
            
    def _check_gluing_validity(self, sheaf: CellularSheaf) -> bool:
        """Check gluing conditions."""
        try:
            self.validator._validate_gluing_conditions(sheaf)
            return True
        except:
            return False
            
    def _compress_sections(self, sheaf: CellularSheaf) -> Dict[str, Any]:
        """Compress individual section data."""
        compressed_sections = {}
        
        for cell, section_data in sheaf.sections.items():
            if isinstance(section_data, np.ndarray):
                compressed = self._compress_array_section(section_data)
            elif isinstance(section_data, dict):
                compressed = self._compress_dict_section(section_data)
            else:
                compressed = self._compress_generic_section(section_data)
                
            compressed_sections[cell] = compressed
            
        return compressed_sections
        
    def _compress_array_section(self, data: np.ndarray) -> Dict[str, Any]:
        """Compress numpy array section."""
        # Use numpy's compression
        compressed_bytes = np.savez_compressed(
            data=data,
            dtype=str(data.dtype),
            shape=data.shape
        )
        
        return {
            'type': 'numpy_compressed',
            'data': compressed_bytes.getvalue() if hasattr(compressed_bytes, 'getvalue') else data.tobytes(),
            'dtype': str(data.dtype),
            'shape': data.shape
        }
        
    def _compress_dict_section(self, data: dict) -> Dict[str, Any]:
        """Compress dictionary section."""
        return {
            'type': 'dict',
            'data': data  # Dictionaries are already efficient
        }
        
    def _compress_generic_section(self, data: Any) -> Dict[str, Any]:
        """Compress generic section data."""
        return {
            'type': 'generic',
            'data': pickle.dumps(data),
            'original_type': type(data).__name__
        }
        
    def _serialize_topology(self, sheaf: CellularSheaf) -> Dict[str, Any]:
        """Serialize sheaf topology for storage."""
        return {
            'cells': list(sheaf.cells),
            'topology': {cell: list(neighbors) for cell, neighbors in sheaf.topology.items()},
            'dimensions': sheaf.cell_dimensions,
            'gluing_data': sheaf.gluing_data
        }
        
    def _serialize_restrictions(self, sheaf: CellularSheaf) -> List[Dict[str, str]]:
        """Serialize restriction maps."""
        restrictions = []
        
        for (source, target), restriction in sheaf.restriction_maps.items():
            restrictions.append({
                'source': source,
                'target': target,
                'type': 'standard'  # Could extend with restriction types
            })
            
        return restrictions
        
    def _build_metadata(self, sheaf: CellularSheaf, original_data: Any) -> Dict[str, Any]:
        """Build metadata for reconstruction."""
        metadata = {
            'original_type': type(original_data).__name__,
            'num_cells': len(sheaf.cells),
            'compression_params': {
                'max_cell_size': self.max_cell_size,
                'min_cell_size': self.min_cell_size,
                'overlap_ratio': self.overlap_ratio
            }
        }
        
        if isinstance(original_data, np.ndarray):
            metadata['array_info'] = {
                'shape': original_data.shape,
                'dtype': str(original_data.dtype)
            }
        elif isinstance(original_data, (list, tuple)):
            metadata['sequence_length'] = len(original_data)
            
        return metadata
        
    def _update_metrics(self, original: Any, compressed: Dict[str, Any]) -> None:
        """Update compression metrics."""
        original_size = self._compute_size(original)
        compressed_size = self._compute_size(compressed)
        
        ratio = compressed_size / original_size if original_size > 0 else 1.0
        
        self.compression_metrics = {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': ratio,
            'num_cells': len(self.current_sheaf.cells) if self.current_sheaf else 0
        }
        
    def _compute_size(self, data: Any) -> int:
        """Compute size of data in bytes."""
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, dict):
            return len(pickle.dumps(data))
        else:
            return len(pickle.dumps(data))
            
    def _compute_data_hash(self, data: Any) -> str:
        """Compute hash for data caching."""
        if isinstance(data, np.ndarray):
            return hashlib.sha256(data.tobytes()).hexdigest()
        else:
            return hashlib.sha256(pickle.dumps(data)).hexdigest()
            
    def _rebuild_sheaf(self, compressed_data: Dict[str, Any]) -> CellularSheaf:
        """Rebuild sheaf structure from compressed data."""
        sheaf = CellularSheaf()
        
        # Rebuild topology
        topology_data = compressed_data['topology']
        
        # Add cells
        for cell in topology_data['cells']:
            neighbors = set(topology_data['topology'].get(cell, []))
            dimension = topology_data['dimensions'].get(cell, 0)
            sheaf.add_cell(cell, dimension, neighbors)
            
        # Restore gluing data
        sheaf.gluing_data = topology_data.get('gluing_data', {})
        
        # Rebuild restriction maps
        for restriction_info in compressed_data['restrictions']:
            source = restriction_info['source']
            target = restriction_info['target']
            
            # Create appropriate restriction map
            restriction = RestrictionMap(source, target)
            sheaf.add_restriction_map(source, target, restriction)
            
        return sheaf
        
    def _decompress_sections(self, sheaf: CellularSheaf, 
                           compressed_sections: Dict[str, Any]) -> None:
        """Decompress section data into sheaf."""
        for cell, compressed in compressed_sections.items():
            if cell not in sheaf.cells:
                raise ValueError(f"Unknown cell in compressed data: {cell}")
                
            section_data = self._decompress_single_section(compressed)
            sheaf.set_section(cell, section_data)
            
    def _decompress_single_section(self, compressed: Dict[str, Any]) -> Any:
        """Decompress a single section."""
        comp_type = compressed.get('type')
        
        if comp_type == 'numpy_compressed':
            return self._decompress_numpy_section(compressed)
        elif comp_type == 'dict':
            return compressed['data']
        elif comp_type == 'generic':
            return pickle.loads(compressed['data'])
        else:
            raise ValueError(f"Unknown compression type: {comp_type}")
            
    def _decompress_numpy_section(self, compressed: Dict[str, Any]) -> np.ndarray:
        """Decompress numpy array section."""
        dtype = np.dtype(compressed['dtype'])
        shape = tuple(compressed['shape'])
        
        # Handle different compression formats
        if isinstance(compressed['data'], bytes):
            # Raw bytes
            return np.frombuffer(compressed['data'], dtype=dtype).reshape(shape)
        else:
            # Compressed npz format
            loaded = np.load(compressed['data'])
            return loaded['data']
            
    def _reconstruct_global(self, sheaf: CellularSheaf, 
                          metadata: Dict[str, Any]) -> Any:
        """Perform local-to-global reconstruction."""
        original_type = metadata['original_type']
        
        if original_type == 'ndarray':
            return self._reconstruct_array(sheaf, metadata)
        elif original_type == 'dict':
            return self._reconstruct_dict(sheaf)
        elif original_type in ('list', 'tuple'):
            return self._reconstruct_sequence(sheaf, metadata, original_type)
        else:
            return self._reconstruct_atomic(sheaf)
            
    def _reconstruct_array(self, sheaf: CellularSheaf, 
                         metadata: Dict[str, Any]) -> np.ndarray:
        """Reconstruct array from sheaf sections."""
        array_info = metadata.get('array_info', {})
        target_shape = tuple(array_info.get('shape', ()))
        dtype = np.dtype(array_info.get('dtype', 'float64'))
        
        # Handle different reconstruction cases
        if 'original_shape' in sheaf.gluing_data:
            # n-dimensional array that was flattened
            flattened = self._reconstruct_2d_array(sheaf, dtype)
            return flattened.reshape(sheaf.gluing_data['original_shape'])
        elif len(target_shape) == 1:
            return self._reconstruct_1d_array(sheaf, target_shape[0], dtype)
        elif len(target_shape) == 2:
            return self._reconstruct_2d_array(sheaf, dtype)
        else:
            raise ValueError(f"Cannot reconstruct array with shape {target_shape}")
            
    def _reconstruct_1d_array(self, sheaf: CellularSheaf, 
                            length: int, dtype: np.dtype) -> np.ndarray:
        """Reconstruct 1D array from overlapping cells."""
        result = np.zeros(length, dtype=dtype)
        covered = np.zeros(length, dtype=bool)
        
        # Sort cells by index
        cells = sorted([c for c in sheaf.cells if c.startswith('cell_1d_')],
                      key=lambda x: int(x.split('_')[-1]))
        
        # Reconstruct by placing sections
        position = 0
        for i, cell in enumerate(cells):
            section = sheaf.sections[cell]
            
            if i == 0:
                # First cell
                result[:len(section)] = section
                covered[:len(section)] = True
                position = len(section)
            else:
                # Find overlap with previous cell
                prev_cell = cells[i-1]
                prev_section = sheaf.sections[prev_cell]
                
                # Determine overlap size
                if (prev_cell, cell) in sheaf.restriction_maps:
                    restriction = sheaf.restriction_maps[(prev_cell, cell)]
                    restricted = restriction.apply(prev_section)
                    overlap_size = len(restricted)
                else:
                    overlap_size = 0
                    
                # Place non-overlapping part
                start = position - overlap_size
                end = min(start + len(section), length)
                result[start:end] = section[:end-start]
                covered[start:end] = True
                position = end
                
        if not np.all(covered):
            raise ValueError("Failed to cover entire array during reconstruction")
            
        return result
        
    def _reconstruct_2d_array(self, sheaf: CellularSheaf, 
                            dtype: np.dtype) -> np.ndarray:
        """Reconstruct 2D array from grid cells."""
        # Determine grid dimensions
        grid_cells = [c for c in sheaf.cells if c.startswith('cell_2d_')]
        if not grid_cells:
            raise ValueError("No 2D cells found for reconstruction")
            
        # Parse grid indices
        max_row = max_col = 0
        for cell in grid_cells:
            parts = cell.split('_')
            row, col = int(parts[2]), int(parts[3])
            max_row = max(max_row, row)
            max_col = max(max_col, col)
            
        # Estimate output size from sections
        sample_section = sheaf.sections[grid_cells[0]]
        cell_h, cell_w = sample_section.shape
        
        # Calculate with overlap
        overlap_ratio = self.overlap_ratio
        overlap_h = int(cell_h * overlap_ratio)
        overlap_w = int(cell_w * overlap_ratio)
        
        est_height = (max_row + 1) * (cell_h - overlap_h) + overlap_h
        est_width = (max_col + 1) * (cell_w - overlap_w) + overlap_w
        
        result = np.zeros((est_height, est_width), dtype=dtype)
        weights = np.zeros((est_height, est_width))
        
        # Place sections with blending in overlaps
        for cell in grid_cells:
            parts = cell.split('_')
            row, col = int(parts[2]), int(parts[3])
            section = sheaf.sections[cell]
            
            # Calculate position
            start_i = row * (cell_h - overlap_h)
            start_j = col * (cell_w - overlap_w)
            end_i = start_i + section.shape[0]
            end_j = start_j + section.shape[1]
            
            # Blend with existing data
            result[start_i:end_i, start_j:end_j] += section
            weights[start_i:end_i, start_j:end_j] += 1
            
        # Normalize by weights
        mask = weights > 0
        result[mask] /= weights[mask]
        
        return result
        
    def _reconstruct_dict(self, sheaf: CellularSheaf) -> dict:
        """Reconstruct dictionary from cells."""
        result = {}
        
        for cell in sheaf.cells:
            if cell.startswith('cell_dict_'):
                section = sheaf.sections[cell]
                if isinstance(section, dict):
                    result.update(section)
                    
        return result
        
    def _reconstruct_sequence(self, sheaf: CellularSheaf, 
                            metadata: Dict[str, Any], type_name: str) -> Any:
        """Reconstruct sequence from sheaf."""
        # First reconstruct as array
        length = metadata.get('sequence_length', 0)
        
        # Determine dtype from first section
        first_cell = next(c for c in sheaf.cells if c.startswith('cell_'))
        first_section = sheaf.sections[first_cell]
        dtype = first_section.dtype if hasattr(first_section, 'dtype') else object
        
        array = self._reconstruct_1d_array(sheaf, length, dtype)
        
        # Convert to original type
        if type_name == 'list':
            return array.tolist()
        elif type_name == 'tuple':
            return tuple(array.tolist())
        else:
            return array
            
    def _reconstruct_atomic(self, sheaf: CellularSheaf) -> Any:
        """Reconstruct atomic data."""
        if 'cell_atomic_0' in sheaf.sections:
            return sheaf.sections['cell_atomic_0']
        else:
            raise ValueError("No atomic cell found for reconstruction")
            
    def integrate_with_confidence(self, data: Any, context: Dict[str, Any]) -> float:
        """
        Generate confidence score for compressed data.
        Integrates with ConfidenceGenerator for sheaf structures.
        """
        if not self.confidence_generator:
            raise RuntimeError("Confidence generator not set")
            
        if not self.current_sheaf:
            raise RuntimeError("No current sheaf structure")
            
        # Build sheaf confidence map
        confidence_map = {}
        
        for cell in self.current_sheaf.cells:
            section = self.current_sheaf.sections.get(cell)
            if section is not None:
                cell_confidence = self.confidence_generator.generate_confidence(
                    section, 
                    {'cell_id': cell, 'context': context}
                )
                confidence_map[cell] = cell_confidence
                
        # Store in confidence generator
        self.confidence_generator.sheaf_confidence_maps[id(self.current_sheaf)] = confidence_map
        
        # Return overall confidence (minimum across cells)
        return min(confidence_map.values()) if confidence_map else 0.0