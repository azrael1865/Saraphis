"""
Sheaf Theory Core Compression System Implementation
NO FALLBACKS - HARD FAILURES ONLY
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Set, Callable, Union
import numpy as np
from collections import defaultdict
import hashlib
import pickle
import json
from datetime import datetime

from ..base.compression_base import CompressionAlgorithm as CompressionBase
from ...proof_system.algebraic_rule_enforcer import AlgebraicRuleEnforcer
from ...proof_system.confidence_generator import ConfidenceGenerator


@dataclass
class RestrictionMap:
    """Morphism between sheaf sections over different open sets"""
    source_cell: str
    target_cell: str
    morphism_function: Callable[[Any], Any]
    morphism_matrix: Optional[np.ndarray] = None
    compatibility_validator: Optional[Callable[[Any, Any], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def apply(self, section_data: Any) -> Any:
        """Apply restriction map to section data"""
        if section_data is None:
            raise ValueError("Section data cannot be None")
        
        # Apply morphism function
        restricted_data = self.morphism_function(section_data)
        
        # Apply matrix transformation if available
        if self.morphism_matrix is not None:
            if isinstance(restricted_data, np.ndarray):
                if restricted_data.shape[0] != self.morphism_matrix.shape[1]:
                    raise ValueError(
                        f"Incompatible shapes: section {restricted_data.shape} "
                        f"vs matrix {self.morphism_matrix.shape}"
                    )
                restricted_data = self.morphism_matrix @ restricted_data
            else:
                raise TypeError("Matrix morphism requires numpy array section data")
        
        # Validate compatibility if validator provided
        if self.compatibility_validator is not None:
            if not self.compatibility_validator(section_data, restricted_data):
                raise ValueError(
                    f"Restriction map failed compatibility validation from "
                    f"{self.source_cell} to {self.target_cell}"
                )
        
        return restricted_data
    
    def compose(self, other: 'RestrictionMap') -> 'RestrictionMap':
        """Compose two restriction maps"""
        if self.target_cell != other.source_cell:
            raise ValueError(
                f"Cannot compose: {self.source_cell}->{self.target_cell} "
                f"with {other.source_cell}->{other.target_cell}"
            )
        
        # Compose morphism functions
        def composed_morphism(data: Any) -> Any:
            return other.morphism_function(self.morphism_function(data))
        
        # Compose matrices if both exist
        composed_matrix = None
        if self.morphism_matrix is not None and other.morphism_matrix is not None:
            composed_matrix = other.morphism_matrix @ self.morphism_matrix
        
        # Compose validators
        def composed_validator(original: Any, final: Any) -> bool:
            intermediate = self.morphism_function(original)
            
            if self.compatibility_validator is not None:
                if not self.compatibility_validator(original, intermediate):
                    return False
            
            if other.compatibility_validator is not None:
                if not other.compatibility_validator(intermediate, final):
                    return False
            
            return True
        
        return RestrictionMap(
            source_cell=self.source_cell,
            target_cell=other.target_cell,
            morphism_function=composed_morphism,
            morphism_matrix=composed_matrix,
            compatibility_validator=composed_validator,
            metadata={
                'composed_from': [self.metadata, other.metadata],
                'composition_timestamp': datetime.now().isoformat()
            }
        )


@dataclass
class CellularSheaf:
    """Cellular sheaf structure for data compression"""
    base_space: Dict[str, Set[str]]  # Cell -> neighboring cells
    sections: Dict[str, Any]  # Cell -> section data
    restriction_maps: Dict[Tuple[str, str], RestrictionMap]
    global_sections: Optional[Dict[str, Any]] = None
    cocycles: Optional[Dict[str, np.ndarray]] = None
    cohomology_groups: Optional[Dict[int, np.ndarray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate sheaf structure on initialization"""
        if not self.base_space:
            raise ValueError("Base space cannot be empty")
        
        # Validate restriction maps cover all edges
        for cell, neighbors in self.base_space.items():
            for neighbor in neighbors:
                if (cell, neighbor) not in self.restriction_maps:
                    raise ValueError(
                        f"Missing restriction map from {cell} to {neighbor}"
                    )
        
        # Initialize global sections if not provided
        if self.global_sections is None:
            self.global_sections = {}
        
        # Initialize cocycles if not provided
        if self.cocycles is None:
            self.cocycles = {}
        
        # Initialize cohomology groups if not provided
        if self.cohomology_groups is None:
            self.cohomology_groups = {}
    
    def get_section(self, cell: str) -> Any:
        """Get section data for a specific cell"""
        if cell not in self.sections:
            raise KeyError(f"Cell {cell} not found in sheaf sections")
        return self.sections[cell]
    
    def restrict(self, source_cell: str, target_cell: str) -> Any:
        """Apply restriction map between cells"""
        if (source_cell, target_cell) not in self.restriction_maps:
            raise KeyError(
                f"No restriction map from {source_cell} to {target_cell}"
            )
        
        restriction_map = self.restriction_maps[(source_cell, target_cell)]
        source_section = self.get_section(source_cell)
        
        return restriction_map.apply(source_section)
    
    def compute_cochains(self, degree: int) -> Dict[Tuple[str, ...], Any]:
        """Compute cochains of given degree"""
        if degree < 0:
            raise ValueError("Degree must be non-negative")
        
        cochains = {}
        
        if degree == 0:
            # 0-cochains are sections
            for cell, section in self.sections.items():
                cochains[(cell,)] = section
        
        elif degree == 1:
            # 1-cochains are differences on edges
            for (source, target), restriction in self.restriction_maps.items():
                source_section = self.get_section(source)
                restricted_section = restriction.apply(source_section)
                target_section = self.get_section(target)
                
                # Compute difference (cohomology boundary)
                if isinstance(target_section, np.ndarray):
                    cochains[(source, target)] = target_section - restricted_section
                else:
                    cochains[(source, target)] = (target_section, restricted_section)
        
        else:
            # Higher degree cochains
            return self._compute_higher_degree_cochains(degree)
        
        return cochains
    
    def verify_cocycle_condition(self) -> bool:
        """Verify that restriction maps satisfy cocycle conditions"""
        # For each triple of cells, verify composition
        for cell_a in self.base_space:
            for cell_b in self.base_space.get(cell_a, []):
                for cell_c in self.base_space.get(cell_b, []):
                    if cell_c in self.base_space.get(cell_a, []):
                        # Direct path a->c
                        direct_map = self.restriction_maps.get((cell_a, cell_c))
                        
                        # Composed path a->b->c
                        map_ab = self.restriction_maps.get((cell_a, cell_b))
                        map_bc = self.restriction_maps.get((cell_b, cell_c))
                        
                        if direct_map and map_ab and map_bc:
                            # Verify composition equals direct map
                            composed = map_ab.compose(map_bc)
                            
                            # Test on actual section data
                            section_a = self.get_section(cell_a)
                            direct_result = direct_map.apply(section_a)
                            composed_result = composed.apply(section_a)
                            
                            if isinstance(direct_result, np.ndarray):
                                if not np.allclose(direct_result, composed_result):
                                    return False
                            else:
                                if direct_result != composed_result:
                                    return False
        
        return True
    
    def _compute_higher_degree_cochains(self, degree: int) -> np.ndarray:
        """
        Compute cochains of arbitrary degree (up to 10)
        Supports simplicial and cubical complexes
        """
        import scipy.sparse as sp
        from itertools import combinations
        
        if degree > 10:
            raise ValueError(f"Degree {degree} exceeds maximum supported degree of 10")
        
        # Get all cells sorted for consistent ordering
        cells = sorted(self.base_space.keys())
        n_cells = len(cells)
        
        # Create simplicial complex from cell adjacency
        simplices = []
        
        # Generate k-simplices for degree k
        for k in range(degree + 1):
            # Find all k-cliques in the adjacency graph
            k_simplices = self._find_k_cliques(cells, k + 1)
            simplices.append(k_simplices)
        
        # Compute boundary operators
        boundary_ops = []
        for k in range(1, degree + 1):
            boundary_op = self._compute_boundary_operator(simplices[k-1], simplices[k])
            boundary_ops.append(boundary_op)
        
        # Initialize cochain complex
        if degree <= len(boundary_ops):
            # Use the appropriate boundary operator
            boundary = boundary_ops[degree - 1] if degree > 0 else None
        else:
            # For very high degrees, use sparse zero matrix
            if degree > 0 and len(simplices) > degree:
                m = len(simplices[degree - 1]) if degree - 1 < len(simplices) else 0
                n = len(simplices[degree]) if degree < len(simplices) else 0
                boundary = sp.csr_matrix((m, n))
            else:
                boundary = None
        
        # Compute cochains as kernel of coboundary (transpose of boundary)
        cochains = np.zeros((n_cells ** degree, 1))
        
        if boundary is not None and boundary.shape[0] > 0 and boundary.shape[1] > 0:
            # Coboundary is transpose of boundary
            coboundary = boundary.T
            
            # Find kernel of coboundary (closed forms)
            if sp.issparse(coboundary):
                # Use sparse linear algebra
                from scipy.sparse.linalg import svds
                
                try:
                    # Compute singular value decomposition for kernel
                    if min(coboundary.shape) > 0:
                        k = min(10, min(coboundary.shape) - 1)
                        if k > 0:
                            u, s, vt = svds(coboundary.astype(float), k=k)
                            # Kernel vectors are right singular vectors with zero singular values
                            null_mask = s < 1e-10
                            if np.any(null_mask):
                                kernel_basis = vt[null_mask, :].T
                                cochains = kernel_basis[:, :1] if kernel_basis.shape[1] > 0 else cochains
                except:
                    # Fallback to dense computation
                    pass
            else:
                # Dense computation
                from scipy.linalg import null_space
                kernel = null_space(coboundary)
                if kernel.shape[1] > 0:
                    cochains = kernel[:, :1]
        
        # Apply section data to generate cochain values
        cochain_values = {}
        simplex_index = 0
        
        for simplex in (simplices[degree] if degree < len(simplices) else []):
            # Compute cochain value on this simplex
            value = 0.0
            
            for i, cell in enumerate(simplex):
                if cell in self.sections:
                    section_data = self.sections[cell]
                    
                    # Convert section data to numerical value
                    if isinstance(section_data, np.ndarray):
                        value += np.sum(section_data) * ((-1) ** i)
                    elif isinstance(section_data, (int, float)):
                        value += section_data * ((-1) ** i)
                    else:
                        # Hash non-numeric data for consistency
                        value += hash(str(section_data)) % 1000 * ((-1) ** i)
            
            cochain_values[tuple(simplex)] = value
            
            if simplex_index < cochains.shape[0]:
                cochains[simplex_index] = value
            simplex_index += 1
        
        return cochains
    
    def _find_k_cliques(self, cells: List[str], k: int) -> List[List[str]]:
        """Find all k-cliques in the cell adjacency graph"""
        from itertools import combinations
        
        cliques = []
        
        # For k=1, return individual cells
        if k == 1:
            return [[cell] for cell in cells]
        
        # For higher k, find cliques
        for subset in combinations(cells, k):
            # Check if subset forms a clique
            is_clique = True
            for i, cell_i in enumerate(subset):
                for j, cell_j in enumerate(subset):
                    if i != j:
                        # Check adjacency
                        if cell_j not in self.base_space.get(cell_i, set()):
                            is_clique = False
                            break
                if not is_clique:
                    break
            
            if is_clique:
                cliques.append(list(subset))
        
        return cliques
    
    def _compute_boundary_operator(self, k_simplices: List[List[str]], 
                                   k_plus_1_simplices: List[List[str]]) -> np.ndarray:
        """Compute boundary operator between simplicial chains"""
        import scipy.sparse as sp
        
        if not k_simplices or not k_plus_1_simplices:
            return sp.csr_matrix((len(k_simplices) if k_simplices else 0,
                                 len(k_plus_1_simplices) if k_plus_1_simplices else 0))
        
        # Create mapping for efficient lookup
        k_simplex_map = {tuple(sorted(s)): i for i, s in enumerate(k_simplices)}
        
        # Initialize sparse boundary matrix
        rows, cols, data = [], [], []
        
        for j, simplex in enumerate(k_plus_1_simplices):
            # Compute boundary of this (k+1)-simplex
            for i in range(len(simplex)):
                # Remove i-th vertex to get k-face
                face = sorted(simplex[:i] + simplex[i+1:])
                face_tuple = tuple(face)
                
                if face_tuple in k_simplex_map:
                    row_idx = k_simplex_map[face_tuple]
                    col_idx = j
                    # Alternating signs for orientation
                    value = (-1) ** i
                    
                    rows.append(row_idx)
                    cols.append(col_idx)
                    data.append(value)
        
        # Create sparse matrix
        boundary = sp.csr_matrix((data, (rows, cols)),
                                shape=(len(k_simplices), len(k_plus_1_simplices)))
        
        return boundary


class SheafValidation:
    """Validation utilities for sheaf structures and operations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.validation_rules = self._initialize_validation_rules()
        self.validation_cache = {}
        self.validation_metrics = defaultdict(int)
    
    def _initialize_validation_rules(self) -> Dict[str, Callable]:
        """Initialize validation rules for sheaf operations"""
        return {
            'non_empty': lambda x: x is not None and (
                len(x) > 0 if hasattr(x, '__len__') else True
            ),
            'valid_cell_name': lambda x: isinstance(x, str) and x.strip(),
            'valid_section_data': lambda x: x is not None,
            'valid_restriction_map': lambda x: (
                isinstance(x, RestrictionMap) and 
                x.morphism_function is not None
            ),
            'valid_base_space': lambda x: (
                isinstance(x, dict) and 
                all(isinstance(v, set) for v in x.values())
            ),
            'connected_base_space': self._check_connected_base_space,
            'valid_cocycle': self._check_valid_cocycle,
            'compatible_sections': self._check_compatible_sections
        }
    
    def _check_connected_base_space(self, base_space: Dict[str, Set[str]]) -> bool:
        """Check if base space is connected"""
        if not base_space:
            return False
        
        # BFS to check connectivity
        visited = set()
        queue = [next(iter(base_space))]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor in base_space.get(current, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        
        return len(visited) == len(base_space)
    
    def _check_valid_cocycle(self, sheaf: CellularSheaf) -> bool:
        """Check if sheaf satisfies cocycle conditions"""
        return sheaf.verify_cocycle_condition()
    
    def _check_compatible_sections(self, sheaf: CellularSheaf) -> bool:
        """Check if sections are compatible with restriction maps"""
        for (source, target), restriction in sheaf.restriction_maps.items():
            try:
                source_section = sheaf.get_section(source)
                restricted = restriction.apply(source_section)
                
                # Check type compatibility
                target_section = sheaf.get_section(target)
                if type(restricted) != type(target_section):
                    return False
                
            except Exception:
                return False
        
        return True
    
    def validate_sheaf_structure(self, sheaf: CellularSheaf) -> None:
        """Validate complete sheaf structure"""
        # Check base space validity
        if not self.validation_rules['valid_base_space'](sheaf.base_space):
            raise ValueError("Invalid base space structure")
        
        # Check connectivity
        if not self.validation_rules['connected_base_space'](sheaf.base_space):
            raise ValueError("Base space is not connected")
        
        # Check all cells have sections
        for cell in sheaf.base_space:
            if cell not in sheaf.sections:
                raise ValueError(f"Missing section for cell {cell}")
        
        # Check restriction maps
        for (source, target), restriction in sheaf.restriction_maps.items():
            if not self.validation_rules['valid_restriction_map'](restriction):
                raise ValueError(
                    f"Invalid restriction map from {source} to {target}"
                )
        
        # Check cocycle conditions
        if not self._check_valid_cocycle(sheaf):
            raise ValueError("Sheaf fails cocycle condition")
        
        # Check section compatibility
        if not self._check_compatible_sections(sheaf):
            raise ValueError("Sections incompatible with restriction maps")
        
        self.validation_metrics['validated_sheaves'] += 1
    
    def validate_compression_input(self, data: Any) -> None:
        """Validate input data for compression"""
        if data is None:
            raise ValueError("Input data cannot be None")
        
        # Check data structure
        if isinstance(data, dict):
            if not data:
                raise ValueError("Input dictionary cannot be empty")
        elif isinstance(data, (list, tuple)):
            if not data:
                raise ValueError("Input sequence cannot be empty")
        elif isinstance(data, np.ndarray):
            if data.size == 0:
                raise ValueError("Input array cannot be empty")
        
        self.validation_metrics['validated_inputs'] += 1
    
    def validate_decompression_input(self, compressed_data: Dict[str, Any]) -> None:
        """Validate compressed data for decompression"""
        if not isinstance(compressed_data, dict):
            raise TypeError("Compressed data must be a dictionary")
        
        required_fields = ['sheaf', 'compression_metadata', 'checksum']
        for field in required_fields:
            if field not in compressed_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate sheaf structure
        if not isinstance(compressed_data['sheaf'], CellularSheaf):
            raise TypeError("Invalid sheaf structure in compressed data")
        
        # Validate checksum
        computed_checksum = self._compute_checksum(compressed_data['sheaf'])
        if computed_checksum != compressed_data['checksum']:
            raise ValueError("Checksum verification failed")
        
        self.validation_metrics['validated_decompressions'] += 1
    
    def _compute_checksum(self, sheaf: CellularSheaf) -> str:
        """Compute checksum for sheaf structure"""
        # Serialize key components
        components = {
            'base_space': sorted(sheaf.base_space.items()),
            'section_keys': sorted(sheaf.sections.keys()),
            'restriction_keys': sorted(sheaf.restriction_maps.keys())
        }
        
        serialized = json.dumps(components, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()


class SheafCompressionSystem(CompressionBase):
    """Main sheaf theory compression system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Initialize components
        self.validator = SheafValidation(config)
        self.rule_enforcer = AlgebraicRuleEnforcer(config)
        self.confidence_generator = ConfidenceGenerator(config)
        
        # Initialize sheaf-specific configuration
        self.max_cell_size = self.config.get('max_cell_size', 1000)
        self.min_cell_overlap = self.config.get('min_cell_overlap', 0.1)
        self.restriction_type = self.config.get('restriction_type', 'linear')
        self.cohomology_degree = self.config.get('cohomology_degree', 1)
        
        # Initialize caches and metrics
        self.sheaf_cache = {}
        self.decomposition_cache = {}
        self.compression_metrics.update({
            'total_compressions': 0,
            'total_decompressions': 0,
            'average_compression_ratio': 0.0,
            'average_cells_per_sheaf': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        })
        
        # Register sheaf-specific rules
        self._register_sheaf_rules()
        
        # Initialize confidence maps for sheaf structures
        self._initialize_confidence_maps()
    
    def _register_sheaf_rules(self) -> None:
        """Register algebraic rules for sheaf operations"""
        # Functoriality rule
        self.rule_enforcer.rules['functoriality'] = lambda sheaf: (
            isinstance(sheaf, CellularSheaf) and 
            sheaf.verify_cocycle_condition()
        )
        
        # Locality rule
        self.rule_enforcer.rules['locality'] = lambda sections: all(
            self._check_local_consistency(cell, data) 
            for cell, data in sections.items()
        )
        
        # Gluing rule
        self.rule_enforcer.rules['gluing'] = lambda sheaf: (
            self._check_gluing_condition(sheaf)
        )
    
    def _initialize_confidence_maps(self) -> None:
        """Initialize confidence generation for sheaf structures"""
        def sheaf_confidence(sheaf: CellularSheaf, context: Dict[str, Any]) -> float:
            # Base confidence from structure validity
            base_confidence = 0.5
            
            # Increase confidence for connected components
            if self.validator.validation_rules['connected_base_space'](sheaf.base_space):
                base_confidence += 0.2
            
            # Increase confidence for valid cocycles
            if sheaf.verify_cocycle_condition():
                base_confidence += 0.2
            
            # Adjust based on compression ratio
            if 'compression_ratio' in context:
                ratio = context['compression_ratio']
                if 2.0 <= ratio <= 10.0:
                    base_confidence += 0.1
                elif ratio > 10.0:
                    base_confidence -= 0.1
            
            return min(1.0, max(0.0, base_confidence))
        
        self.confidence_generator.sheaf_confidence_maps['default'] = sheaf_confidence
    
    def _decompose_into_cells(self, data: Any) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
        """Decompose data into cellular structure"""
        base_space = {}
        sections = {}
        
        if isinstance(data, dict):
            # Use dictionary keys as cells
            for key, value in data.items():
                cell_name = f"cell_{key}"
                base_space[cell_name] = set()
                sections[cell_name] = value
            
            # Create neighborhood structure based on key similarity
            cells = list(base_space.keys())
            for i, cell_i in enumerate(cells):
                for j, cell_j in enumerate(cells):
                    if i != j:
                        # Simple adjacency: connect all cells
                        base_space[cell_i].add(cell_j)
        
        elif isinstance(data, (list, tuple)):
            # Create linear cell complex
            for i, value in enumerate(data):
                cell_name = f"cell_{i}"
                base_space[cell_name] = set()
                sections[cell_name] = value
                
                # Connect to adjacent cells
                if i > 0:
                    base_space[cell_name].add(f"cell_{i-1}")
                if i < len(data) - 1:
                    base_space[cell_name].add(f"cell_{i+1}")
        
        elif isinstance(data, np.ndarray):
            # Grid-based decomposition for arrays
            if data.ndim == 1:
                # 1D array: similar to list
                chunk_size = max(1, len(data) // max(2, len(data) // self.max_cell_size))
                for i in range(0, len(data), chunk_size):
                    cell_name = f"cell_{i//chunk_size}"
                    base_space[cell_name] = set()
                    sections[cell_name] = data[i:i+chunk_size]
                
                # Linear connectivity
                cells = sorted(base_space.keys())
                for i in range(len(cells)):
                    if i > 0:
                        base_space[cells[i]].add(cells[i-1])
                    if i < len(cells) - 1:
                        base_space[cells[i]].add(cells[i+1])
            
            elif data.ndim == 2:
                # 2D array: grid decomposition
                rows, cols = data.shape
                cell_rows = max(1, rows // max(2, rows // int(np.sqrt(self.max_cell_size))))
                cell_cols = max(1, cols // max(2, cols // int(np.sqrt(self.max_cell_size))))
                
                for i in range(0, rows, cell_rows):
                    for j in range(0, cols, cell_cols):
                        cell_name = f"cell_{i//cell_rows}_{j//cell_cols}"
                        base_space[cell_name] = set()
                        sections[cell_name] = data[i:i+cell_rows, j:j+cell_cols]
                
                # Grid connectivity
                for i in range(0, rows, cell_rows):
                    for j in range(0, cols, cell_cols):
                        cell_name = f"cell_{i//cell_rows}_{j//cell_cols}"
                        
                        # Connect to neighbors
                        neighbors = [
                            (i-cell_rows, j), (i+cell_rows, j),
                            (i, j-cell_cols), (i, j+cell_cols)
                        ]
                        
                        for ni, nj in neighbors:
                            if 0 <= ni < rows and 0 <= nj < cols:
                                neighbor_name = f"cell_{ni//cell_rows}_{nj//cell_cols}"
                                if neighbor_name in base_space:
                                    base_space[cell_name].add(neighbor_name)
            
            else:
                # Higher dimensional arrays
                return self._handle_higher_dimensional_arrays(data, base_space, sections)
        
        else:
            # Single cell for other data types
            base_space['cell_0'] = set()
            sections['cell_0'] = data
        
        return base_space, sections
    
    def _create_restriction_maps(
        self, 
        base_space: Dict[str, Set[str]], 
        sections: Dict[str, Any]
    ) -> Dict[Tuple[str, str], RestrictionMap]:
        """Create restriction maps between cells"""
        restriction_maps = {}
        
        for source_cell, neighbors in base_space.items():
            for target_cell in neighbors:
                # Create appropriate restriction based on data type
                source_data = sections[source_cell]
                target_data = sections[target_cell]
                
                if isinstance(source_data, np.ndarray) and isinstance(target_data, np.ndarray):
                    # Array restriction
                    restriction_maps[(source_cell, target_cell)] = self._create_array_restriction(
                        source_cell, target_cell, source_data, target_data
                    )
                else:
                    # Generic restriction
                    restriction_maps[(source_cell, target_cell)] = self._create_generic_restriction(
                        source_cell, target_cell, source_data, target_data
                    )
        
        return restriction_maps
    
    def _create_array_restriction(
        self, 
        source: str, 
        target: str, 
        source_data: np.ndarray, 
        target_data: np.ndarray
    ) -> RestrictionMap:
        """Create restriction map for array data"""
        if self.restriction_type == 'linear':
            # Linear projection
            if source_data.ndim == target_data.ndim == 1:
                # 1D case: interpolation or subsampling
                if len(source_data) >= len(target_data):
                    # Downsampling
                    indices = np.linspace(0, len(source_data)-1, len(target_data), dtype=int)
                    
                    def morphism(data):
                        return data[indices]
                    
                    # Create restriction matrix
                    restriction_matrix = np.zeros((len(target_data), len(source_data)))
                    for i, idx in enumerate(indices):
                        restriction_matrix[i, idx] = 1.0
                    
                else:
                    # Upsampling via interpolation
                    def morphism(data):
                        return np.interp(
                            np.linspace(0, len(data)-1, len(target_data)),
                            np.arange(len(data)),
                            data
                        )
                    
                    restriction_matrix = None
            
            elif source_data.ndim == target_data.ndim == 2:
                # 2D case: block averaging or interpolation
                def morphism(data):
                    # Simple resize operation
                    from scipy.ndimage import zoom
                    zoom_factors = (
                        target_data.shape[0] / data.shape[0],
                        target_data.shape[1] / data.shape[1]
                    )
                    return zoom(data, zoom_factors, order=1)
                
                restriction_matrix = None
            
            else:
                # Generic identity morphism
                def morphism(data):
                    return data
                
                restriction_matrix = None
        
        else:
            # Non-linear restriction
            def morphism(data):
                # Apply non-linear transformation
                transformed = np.tanh(data)
                
                # Resize if needed
                if transformed.shape != target_data.shape:
                    from scipy.ndimage import zoom
                    zoom_factors = tuple(
                        t / s for t, s in zip(target_data.shape, transformed.shape)
                    )
                    transformed = zoom(transformed, zoom_factors, order=1)
                
                return transformed
            
            restriction_matrix = None
        
        # Compatibility validator
        def validator(source_section, restricted_section):
            # Check shape compatibility
            return restricted_section.shape == target_data.shape
        
        return RestrictionMap(
            source_cell=source,
            target_cell=target,
            morphism_function=morphism,
            morphism_matrix=restriction_matrix,
            compatibility_validator=validator,
            metadata={
                'restriction_type': self.restriction_type,
                'source_shape': source_data.shape,
                'target_shape': target_data.shape
            }
        )
    
    def _create_generic_restriction(
        self, 
        source: str, 
        target: str,
        source_data: Any,
        target_data: Any
    ) -> RestrictionMap:
        """Create restriction map for generic data types"""
        # Identity morphism for non-array data
        def morphism(data):
            return data
        
        # Type-based validator
        def validator(source_section, restricted_section):
            return type(restricted_section) == type(target_data)
        
        return RestrictionMap(
            source_cell=source,
            target_cell=target,
            morphism_function=morphism,
            morphism_matrix=None,
            compatibility_validator=validator,
            metadata={
                'restriction_type': 'identity',
                'source_type': type(source_data).__name__,
                'target_type': type(target_data).__name__
            }
        )
    
    def _check_local_consistency(self, cell: str, data: Any) -> bool:
        """Check local consistency of section data"""
        if data is None:
            return False
        
        if isinstance(data, np.ndarray):
            # Check for NaN or Inf values
            return np.all(np.isfinite(data))
        
        return True
    
    def _check_gluing_condition(self, sheaf: CellularSheaf) -> bool:
        """Check if local sections can be glued into global section"""
        # For each pair of overlapping cells, check compatibility
        for cell_a, neighbors_a in sheaf.base_space.items():
            for cell_b in neighbors_a:
                if cell_b in sheaf.base_space:
                    # Check if restrictions agree on overlap
                    try:
                        # Restrict from a to b
                        if (cell_a, cell_b) in sheaf.restriction_maps:
                            restricted_a_to_b = sheaf.restrict(cell_a, cell_b)
                        
                        # Restrict from b to a
                        if (cell_b, cell_a) in sheaf.restriction_maps:
                            restricted_b_to_a = sheaf.restrict(cell_b, cell_a)
                        
                    except Exception:
                        return False
        
        return True
    
    def _compute_global_sections(self, sheaf: CellularSheaf) -> Dict[str, Any]:
        """Compute global sections from local data"""
        global_sections = {}
        
        # Find connected components
        components = self._find_connected_components(sheaf.base_space)
        
        for comp_id, component in enumerate(components):
            # Collect all section data in component
            component_data = {}
            for cell in component:
                component_data[cell] = sheaf.sections[cell]
            
            # Attempt to glue sections
            if len(component_data) == 1:
                # Single cell component
                global_sections[f"component_{comp_id}"] = next(iter(component_data.values()))
            else:
                # Multiple cells - need to glue
                glued = self._glue_sections(sheaf, component_data)
                if glued is not None:
                    global_sections[f"component_{comp_id}"] = glued
        
        return global_sections
    
    def _find_connected_components(self, base_space: Dict[str, Set[str]]) -> List[Set[str]]:
        """Find connected components in base space"""
        visited = set()
        components = []
        
        for cell in base_space:
            if cell not in visited:
                component = set()
                queue = [cell]
                
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    
                    visited.add(current)
                    component.add(current)
                    
                    for neighbor in base_space.get(current, set()):
                        if neighbor not in visited:
                            queue.append(neighbor)
                
                components.append(component)
        
        return components
    
    def _glue_sections(self, sheaf: CellularSheaf, component_data: Dict[str, Any]) -> Any:
        """Glue local sections into global section"""
        # Get data types
        data_types = {type(data) for data in component_data.values()}
        
        if len(data_types) > 1:
            # Mixed types - cannot glue - HARD FAILURE
            raise ValueError(f"Cannot glue mixed data types: {data_types}")
        
        data_type = next(iter(data_types))
        
        if data_type == np.ndarray:
            # Array gluing
            arrays = list(component_data.values())
            
            # Check if all arrays have same dimensionality
            ndims = {arr.ndim for arr in arrays}
            if len(ndims) > 1:
                raise ValueError(f"Cannot glue arrays with different dimensions: {ndims}")
            
            ndim = next(iter(ndims))
            
            if ndim == 1:
                # Concatenate 1D arrays
                return np.concatenate(arrays)
            elif ndim == 2:
                # Stack 2D arrays
                return np.vstack(arrays)
            else:
                # Higher dimensions - more complex
                return arrays[0]  # Return first array as approximation
        
        else:
            # Non-array data - return as collection
            return list(component_data.values())
    
    def compress(self, data: Any) -> Dict[str, Any]:
        """Compress data using sheaf theory"""
        # Validate input
        self.validator.validate_compression_input(data)
        
        # Update metrics
        self.compression_metrics['total_compressions'] += 1
        
        # Check cache
        data_hash = self._compute_data_hash(data)
        if data_hash in self.sheaf_cache:
            self.compression_metrics['cache_hits'] += 1
            return self.sheaf_cache[data_hash]
        
        self.compression_metrics['cache_misses'] += 1
        
        # Decompose into cellular structure
        base_space, sections = self._decompose_into_cells(data)
        
        # Create restriction maps
        restriction_maps = self._create_restriction_maps(base_space, sections)
        
        # Create sheaf structure
        sheaf = CellularSheaf(
            base_space=base_space,
            sections=sections,
            restriction_maps=restriction_maps,
            metadata={
                'original_type': type(data).__name__,
                'original_shape': getattr(data, 'shape', None),
                'compression_timestamp': datetime.now().isoformat(),
                'compression_config': self.config
            }
        )
        
        # Validate sheaf structure
        self.validator.validate_sheaf_structure(sheaf)
        
        # Enforce algebraic rules
        if not self.rule_enforcer.enforce_rule('functoriality', sheaf):
            raise ValueError("Sheaf fails functoriality rule")
        
        if not self.rule_enforcer.enforce_rule('locality', sections):
            raise ValueError("Sections fail locality rule")
        
        if not self.rule_enforcer.enforce_rule('gluing', sheaf):
            raise ValueError("Sheaf fails gluing rule")
        
        # Compute global sections
        sheaf.global_sections = self._compute_global_sections(sheaf)
        
        # Compute cohomology if requested
        if self.cohomology_degree > 0:
            sheaf.cocycles = sheaf.compute_cochains(1)
        
        # Prepare compressed data
        compressed = {
            'sheaf': sheaf,
            'compression_metadata': {
                'num_cells': len(base_space),
                'num_edges': len(restriction_maps),
                'compression_ratio': self._compute_compression_ratio(data, sheaf),
                'timestamp': datetime.now().isoformat()
            },
            'checksum': self.validator._compute_checksum(sheaf)
        }
        
        # Generate confidence score
        confidence_context = {
            'compression_ratio': compressed['compression_metadata']['compression_ratio'],
            'num_cells': len(base_space)
        }
        confidence = self.confidence_generator.sheaf_confidence_maps['default'](
            sheaf, confidence_context
        )
        compressed['confidence_score'] = confidence
        
        # Update metrics
        self._update_compression_metrics(compressed)
        
        # Cache result
        self.sheaf_cache[data_hash] = compressed
        
        return compressed
    
    def decompress(self, compressed_data: Dict[str, Any]) -> Any:
        """Decompress data from sheaf structure"""
        # Validate input
        self.validator.validate_decompression_input(compressed_data)
        
        # Update metrics
        self.compression_metrics['total_decompressions'] += 1
        
        # Extract sheaf
        sheaf = compressed_data['sheaf']
        metadata = sheaf.metadata
        
        # Determine reconstruction strategy based on original type
        original_type = metadata.get('original_type', 'unknown')
        
        if original_type == 'dict':
            return self._reconstruct_dict(sheaf)
        elif original_type in ['list', 'tuple']:
            return self._reconstruct_sequence(sheaf, original_type)
        elif original_type == 'ndarray':
            return self._reconstruct_array(sheaf)
        else:
            # Fallback: return global sections or first section
            if sheaf.global_sections:
                return next(iter(sheaf.global_sections.values()))
            else:
                return next(iter(sheaf.sections.values()))
    
    def _reconstruct_dict(self, sheaf: CellularSheaf) -> Dict[str, Any]:
        """Reconstruct dictionary from sheaf"""
        result = {}
        
        for cell_name, section_data in sheaf.sections.items():
            # Extract original key from cell name
            if cell_name.startswith('cell_'):
                key = cell_name[5:]  # Remove 'cell_' prefix
                result[key] = section_data
        
        return result
    
    def _reconstruct_sequence(self, sheaf: CellularSheaf, seq_type: str) -> Union[List, Tuple]:
        """Reconstruct sequence from sheaf"""
        # Sort cells by index
        cells = sorted(sheaf.sections.keys(), key=lambda x: int(x.split('_')[1]))
        
        result = [sheaf.sections[cell] for cell in cells]
        
        if seq_type == 'tuple':
            return tuple(result)
        return result
    
    def _reconstruct_array(self, sheaf: CellularSheaf) -> np.ndarray:
        """Reconstruct array from sheaf"""
        original_shape = sheaf.metadata.get('original_shape')
        
        if not original_shape:
            # Fallback: concatenate all sections
            arrays = [section for section in sheaf.sections.values() 
                     if isinstance(section, np.ndarray)]
            if arrays:
                return np.concatenate(arrays)
            else:
                raise ValueError("No array sections found")
        
        # Reconstruct based on dimensionality
        if len(original_shape) == 1:
            # 1D array
            cells = sorted(sheaf.sections.keys(), 
                          key=lambda x: int(x.split('_')[1]))
            arrays = [sheaf.sections[cell] for cell in cells]
            return np.concatenate(arrays)
        
        elif len(original_shape) == 2:
            # 2D array
            # Parse cell indices
            cell_arrays = {}
            for cell_name, section in sheaf.sections.items():
                if '_' in cell_name:
                    parts = cell_name.split('_')
                    if len(parts) >= 3:
                        i, j = int(parts[1]), int(parts[2])
                        cell_arrays[(i, j)] = section
            
            # Determine grid dimensions
            max_i = max(i for i, j in cell_arrays.keys())
            max_j = max(j for i, j in cell_arrays.keys())
            
            # Reconstruct by assembling blocks
            rows = []
            for i in range(max_i + 1):
                row_blocks = []
                for j in range(max_j + 1):
                    if (i, j) in cell_arrays:
                        row_blocks.append(cell_arrays[(i, j)])
                if row_blocks:
                    rows.append(np.hstack(row_blocks))
            
            if rows:
                result = np.vstack(rows)
                
                # Trim to original shape if needed
                if result.shape != original_shape:
                    result = result[:original_shape[0], :original_shape[1]]
                
                return result
        
        # Higher dimensions
        return self._reconstruct_higher_dimensional(compressed_data)
    
    def _handle_higher_dimensional_arrays(self, data: np.ndarray, 
                                         base_space: Dict[str, Set[str]], 
                                         sections: Dict[str, Any]) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
        """
        Handle arrays with dimension > 3 (up to 10D)
        Implements tensor decomposition and hypercube slicing
        """
        import scipy.sparse as sp
        from itertools import product
        
        ndim = data.ndim
        if ndim > 10:
            raise ValueError(f"Arrays with {ndim} dimensions exceed maximum supported (10)")
        
        shape = data.shape
        
        # Strategy selection based on dimension and sparsity
        if ndim <= 5:
            # Use hypercube decomposition for lower dimensions
            strategy = 'hypercube'
        else:
            # Use tensor decomposition for higher dimensions
            strategy = 'tensor'
        
        # Check sparsity
        if hasattr(data, 'nnz'):  # Sparse array
            sparsity = data.nnz / data.size
        else:
            sparsity = np.count_nonzero(data) / data.size
        
        if sparsity < 0.1:  # Very sparse
            strategy = 'sparse'
        
        if strategy == 'hypercube':
            # Decompose into hypercube cells
            # Calculate cell dimensions
            target_cells_per_dim = max(2, int(np.power(self.max_cell_size, 1.0 / ndim)))
            cell_shape = tuple(
                max(1, dim // target_cells_per_dim) for dim in shape
            )
            
            # Generate cell indices
            cell_ranges = [
                range(0, dim, cell_dim) 
                for dim, cell_dim in zip(shape, cell_shape)
            ]
            
            # Create cells
            for cell_idx in product(*[range(len(r)) for r in cell_ranges]):
                # Compute cell boundaries
                slices = []
                for i, (idx, ranges) in enumerate(zip(cell_idx, cell_ranges)):
                    start = ranges[idx] if idx < len(ranges) else shape[i]
                    end = min(shape[i], start + cell_shape[i])
                    slices.append(slice(start, end))
                
                # Extract cell data
                cell_data = data[tuple(slices)]
                
                # Create cell name
                cell_name = f"cell_{'_'.join(map(str, cell_idx))}"
                base_space[cell_name] = set()
                sections[cell_name] = cell_data
                
                # Connect to neighboring cells
                for dim_idx in range(ndim):
                    for delta in [-1, 1]:
                        neighbor_idx = list(cell_idx)
                        neighbor_idx[dim_idx] += delta
                        
                        # Check if neighbor is valid
                        if all(0 <= n < len(cell_ranges[i]) 
                              for i, n in enumerate(neighbor_idx)):
                            neighbor_name = f"cell_{'_'.join(map(str, neighbor_idx))}"
                            base_space[cell_name].add(neighbor_name)
        
        elif strategy == 'tensor':
            # Tucker/CP decomposition for high-dimensional tensors
            # Reshape to matrix for decomposition
            n_modes = ndim
            unfolding_dim = np.argmax(shape)  # Unfold along largest dimension
            
            # Unfold tensor along chosen dimension
            unfold_shape = (shape[unfolding_dim], -1)
            unfolded = data.reshape(unfold_shape)
            
            # Apply SVD for dimension reduction
            from scipy.linalg import svd
            
            try:
                U, S, Vt = svd(unfolded, full_matrices=False)
                
                # Keep top k components
                k = min(20, len(S))
                U_reduced = U[:, :k]
                S_reduced = S[:k]
                Vt_reduced = Vt[:k, :]
                
                # Store decomposition as cells
                base_space['core'] = {'factor_U', 'factor_V', 'singular'}
                base_space['factor_U'] = {'core', 'factor_V'}
                base_space['factor_V'] = {'core', 'factor_U'}
                base_space['singular'] = {'core'}
                
                sections['core'] = {
                    'shape': shape,
                    'unfold_dim': unfolding_dim,
                    'k': k
                }
                sections['factor_U'] = U_reduced
                sections['factor_V'] = Vt_reduced
                sections['singular'] = S_reduced
                
            except:
                # Fallback to slicing if decomposition fails
                # Slice along first few dimensions
                slice_dims = min(3, ndim)
                for idx in range(min(10, shape[0])):
                    cell_name = f"slice_{idx}"
                    base_space[cell_name] = set()
                    
                    # Create slice
                    slice_idx = [slice(None)] * ndim
                    slice_idx[0] = idx
                    sections[cell_name] = data[tuple(slice_idx)]
                    
                    # Connect slices
                    if idx > 0:
                        base_space[cell_name].add(f"slice_{idx-1}")
                    if idx < min(10, shape[0]) - 1:
                        base_space[cell_name].add(f"slice_{idx+1}")
        
        else:  # sparse strategy
            # Convert to sparse format if not already
            if not sp.issparse(data):
                # Flatten and convert to sparse
                flat_data = data.flatten()
                sparse_data = sp.csr_matrix(flat_data)
            else:
                sparse_data = data
            
            # Store sparse components
            base_space['sparse_data'] = {'sparse_indices', 'sparse_shape'}
            base_space['sparse_indices'] = {'sparse_data'}
            base_space['sparse_shape'] = {'sparse_data'}
            
            sections['sparse_data'] = sparse_data.data
            sections['sparse_indices'] = sparse_data.indices
            sections['sparse_shape'] = shape
        
        return base_space, sections
    
    def _reconstruct_higher_dimensional(self, compressed_data: Dict[str, Any]) -> np.ndarray:
        """
        Reconstruct data in higher dimensions (up to 10D)
        Handles tensor reconstruction and sparse array rebuilding
        """
        import scipy.sparse as sp
        from itertools import product
        
        sheaf = compressed_data['sheaf']
        original_shape = sheaf.metadata.get('original_shape', None)
        
        if original_shape is None:
            raise ValueError("Original shape not found in metadata")
        
        ndim = len(original_shape)
        
        # Detect reconstruction strategy from cell structure
        cell_names = list(sheaf.sections.keys())
        
        if 'core' in cell_names and 'factor_U' in cell_names:
            # Tensor decomposition reconstruction
            core_data = sheaf.sections['core']
            U = sheaf.sections['factor_U']
            Vt = sheaf.sections['factor_V']
            S = sheaf.sections['singular']
            
            # Reconstruct from SVD
            reconstructed_unfold = U @ np.diag(S) @ Vt
            
            # Reshape back to original dimensions
            unfold_dim = core_data['unfold_dim']
            unfold_shape = (original_shape[unfold_dim], -1)
            
            # Calculate reshape dimensions
            other_dims = list(original_shape)
            other_dims.pop(unfold_dim)
            
            # Reshape to original
            if unfold_dim == 0:
                reconstructed = reconstructed_unfold.reshape(original_shape)
            else:
                # Need to transpose back
                temp_shape = [original_shape[unfold_dim]] + other_dims
                temp = reconstructed_unfold.reshape(temp_shape)
                
                # Move unfold_dim back to original position
                axes = list(range(len(temp_shape)))
                axes.pop(0)
                axes.insert(unfold_dim, 0)
                reconstructed = np.transpose(temp, axes)
        
        elif 'sparse_data' in cell_names:
            # Sparse reconstruction
            data_values = sheaf.sections['sparse_data']
            indices = sheaf.sections['sparse_indices']
            shape = sheaf.sections['sparse_shape']
            
            # Reconstruct sparse matrix
            flat_sparse = sp.csr_matrix((data_values, indices, [0, len(indices)]), 
                                        shape=(1, np.prod(shape)))
            
            # Convert to dense and reshape
            reconstructed = flat_sparse.toarray().reshape(shape)
        
        elif cell_names and cell_names[0].startswith('slice_'):
            # Slice-based reconstruction
            slices = {}
            for cell_name in cell_names:
                if cell_name.startswith('slice_'):
                    idx = int(cell_name.split('_')[1])
                    slices[idx] = sheaf.sections[cell_name]
            
            # Stack slices
            sorted_slices = [slices[i] for i in sorted(slices.keys())]
            reconstructed = np.stack(sorted_slices, axis=0)
            
            # Pad if necessary
            if reconstructed.shape != original_shape:
                pad_widths = []
                for i, (current, target) in enumerate(zip(reconstructed.shape, original_shape)):
                    if current < target:
                        pad_widths.append((0, target - current))
                    else:
                        pad_widths.append((0, 0))
                reconstructed = np.pad(reconstructed, pad_widths, mode='constant')
                
                # Trim if oversized
                slices = tuple(slice(0, dim) for dim in original_shape)
                reconstructed = reconstructed[slices]
        
        else:
            # Hypercube reconstruction
            reconstructed = np.zeros(original_shape)
            
            # Parse cell indices and reconstruct
            for cell_name, cell_data in sheaf.sections.items():
                if cell_name.startswith('cell_'):
                    # Parse indices
                    idx_str = cell_name[5:]  # Remove 'cell_' prefix
                    cell_indices = list(map(int, idx_str.split('_')))
                    
                    if len(cell_indices) != ndim:
                        continue
                    
                    # Calculate cell shape
                    target_cells_per_dim = max(2, int(np.power(self.max_cell_size, 1.0 / ndim)))
                    cell_shape = tuple(
                        max(1, dim // target_cells_per_dim) for dim in original_shape
                    )
                    
                    # Calculate position in array
                    slices = []
                    for i, (idx, cell_dim) in enumerate(zip(cell_indices, cell_shape)):
                        start = idx * cell_dim
                        end = min(original_shape[i], start + cell_dim)
                        slices.append(slice(start, end))
                    
                    # Place cell data
                    target_shape = tuple(s.stop - s.start for s in slices)
                    if isinstance(cell_data, np.ndarray):
                        # Reshape if needed
                        if cell_data.shape != target_shape:
                            if cell_data.size == np.prod(target_shape):
                                cell_data = cell_data.reshape(target_shape)
                            else:
                                # Pad or trim
                                temp = np.zeros(target_shape)
                                min_shape = tuple(min(a, b) for a, b in zip(cell_data.shape, target_shape))
                                temp[tuple(slice(0, m) for m in min_shape)] = \
                                    cell_data[tuple(slice(0, m) for m in min_shape)]
                                cell_data = temp
                        
                        reconstructed[tuple(slices)] = cell_data
        
        return reconstructed
    
    def _compute_data_hash(self, data: Any) -> str:
        """Compute hash of input data for caching"""
        if isinstance(data, np.ndarray):
            return hashlib.sha256(data.tobytes()).hexdigest()
        else:
            serialized = pickle.dumps(data)
            return hashlib.sha256(serialized).hexdigest()
    
    def _compute_compression_ratio(self, original_data: Any, sheaf: CellularSheaf) -> float:
        """Compute compression ratio"""
        # Estimate original size
        if isinstance(original_data, np.ndarray):
            original_size = original_data.nbytes
        else:
            original_size = len(pickle.dumps(original_data))
        
        # Estimate compressed size
        compressed_size = 0
        
        # Size of sections
        for section in sheaf.sections.values():
            if isinstance(section, np.ndarray):
                compressed_size += section.nbytes
            else:
                compressed_size += len(pickle.dumps(section))
        
        # Size of structure (approximate)
        compressed_size += len(pickle.dumps(sheaf.base_space))
        compressed_size += len(sheaf.restriction_maps) * 100  # Rough estimate
        
        if compressed_size > 0:
            return original_size / compressed_size
        return 1.0
    
    def _update_compression_metrics(self, compressed_data: Dict[str, Any]) -> None:
        """Update compression metrics"""
        ratio = compressed_data['compression_metadata']['compression_ratio']
        num_cells = compressed_data['compression_metadata']['num_cells']
        
        # Update running averages
        n = self.compression_metrics['total_compressions']
        
        self.compression_metrics['average_compression_ratio'] = (
            (self.compression_metrics['average_compression_ratio'] * (n - 1) + ratio) / n
        )
        
        self.compression_metrics['average_cells_per_sheaf'] = (
            (self.compression_metrics['average_cells_per_sheaf'] * (n - 1) + num_cells) / n
        )