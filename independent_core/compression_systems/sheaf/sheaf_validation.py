"""
Comprehensive validation for sheaf structures and mathematical properties.
NO FALLBACKS - HARD FAILURES ONLY
"""

from typing import Dict, Any, Set, List, Tuple, Optional, Callable
import numpy as np
import time
from collections import defaultdict, deque

from .sheaf_structures import CellularSheaf, RestrictionMap


class SheafValidation:
    """
    Validates sheaf structures and operations.
    Ensures mathematical consistency, topological validity, and data integrity.
    """
    
    def __init__(self, strict_mode: bool = True):
        """Initialize sheaf validation with configurable strictness"""
        self.strict_mode = strict_mode
        self.validation_rules = {
            'basic_structure': self._validate_basic_structure,
            'topology_consistency': self._validate_topology_consistency,
            'section_completeness': self._validate_section_completeness,
            'restriction_system': self._validate_restriction_system,
            'gluing_conditions': self._validate_gluing_conditions,
            'cohomological_properties': self._validate_cohomological_properties,
            'topological_invariants': self._validate_topological_invariants
        }
        
        # Validation caching for performance
        self.validation_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Statistics tracking
        self.validation_stats = {
            'total_validations': 0,
            'failed_validations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_times': []
        }
    
    def validate_sheaf(self, sheaf: CellularSheaf, rules: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform comprehensive validation of sheaf structure"""
        start_time = time.time()
        
        if not isinstance(sheaf, CellularSheaf):
            raise TypeError(f"Input must be CellularSheaf, got {type(sheaf)}")
        
        # Generate validation key for caching
        validation_key = self._generate_validation_key(sheaf)
        current_time = time.time()
        
        if (validation_key in self.validation_cache and 
            current_time - self.validation_cache[validation_key]['timestamp'] < self.cache_ttl):
            self.validation_stats['cache_hits'] += 1
            return self.validation_cache[validation_key]['result']
        
        self.validation_stats['cache_misses'] += 1
        
        # Select validation rules
        if rules is None:
            rules = list(self.validation_rules.keys())
        
        validation_results = {
            'overall_valid': True,
            'rule_results': {},
            'errors': [],
            'warnings': [],
            'validation_time': 0,
            'sheaf_statistics': sheaf.get_sheaf_statistics()
        }
        
        # Execute validation rules
        for rule_name in rules:
            if rule_name not in self.validation_rules:
                raise ValueError(f"Unknown validation rule: {rule_name}")
            
            try:
                rule_func = self.validation_rules[rule_name]
                rule_result = rule_func(sheaf)
                validation_results['rule_results'][rule_name] = rule_result
                
                if not rule_result.get('valid', False):
                    validation_results['overall_valid'] = False
                    validation_results['errors'].extend(rule_result.get('errors', []))
                
                validation_results['warnings'].extend(rule_result.get('warnings', []))
                
            except Exception as e:
                validation_results['overall_valid'] = False
                error_msg = f"Validation rule '{rule_name}' failed: {e}"
                validation_results['errors'].append(error_msg)
                validation_results['rule_results'][rule_name] = {
                    'valid': False,
                    'error': str(e)
                }
                
                if self.strict_mode:
                    raise ValueError(error_msg) from e
        
        # Finalize results
        validation_time = time.time() - start_time
        validation_results['validation_time'] = validation_time
        
        # Update statistics
        self.validation_stats['total_validations'] += 1
        if not validation_results['overall_valid']:
            self.validation_stats['failed_validations'] += 1
        self.validation_stats['validation_times'].append(validation_time)
        
        # Cache results
        if len(self.validation_cache) < 100:  # Limit cache size
            self.validation_cache[validation_key] = {
                'result': validation_results,
                'timestamp': current_time
            }
        
        # Fail loud if validation failed and in strict mode
        if not validation_results['overall_valid'] and self.strict_mode:
            error_summary = "; ".join(validation_results['errors'])
            raise ValueError(f"Sheaf validation failed: {error_summary}")
        
        return validation_results
    
    def _generate_validation_key(self, sheaf: CellularSheaf) -> str:
        """Generate unique key for sheaf validation caching"""
        key_components = [
            f"cells_{len(sheaf.cells)}",
            f"sections_{len(sheaf.sections)}",
            f"restrictions_{len(sheaf.restriction_maps)}",
            f"overlaps_{len(sheaf.overlap_regions)}"
        ]
        
        # Add hash of cell structure
        cell_hash = hash(frozenset(sheaf.cells))
        key_components.append(f"structure_{abs(cell_hash)}")
        
        return "_".join(key_components)
    
    def _validate_basic_structure(self, sheaf: CellularSheaf) -> Dict[str, Any]:
        """Validate basic sheaf structure requirements"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            # Check non-empty sheaf
            if not sheaf.cells:
                result['errors'].append("Sheaf has no cells")
                result['valid'] = False
                return result
            
            # Validate all cells have required attributes
            for cell in sheaf.cells:
                if cell not in sheaf.topology:
                    result['errors'].append(f"Cell {cell} missing from topology")
                    result['valid'] = False
                
                if cell not in sheaf.cell_dimensions:
                    result['errors'].append(f"Cell {cell} missing dimension information")
                    result['valid'] = False
            
            # Validate topology references
            for cell, neighbors in sheaf.topology.items():
                if cell not in sheaf.cells:
                    result['errors'].append(f"Topology references unknown cell: {cell}")
                    result['valid'] = False
                
                for neighbor in neighbors:
                    if neighbor not in sheaf.cells:
                        result['errors'].append(f"Cell {cell} references unknown neighbor: {neighbor}")
                        result['valid'] = False
            
            # Validate dimensions are consistent
            for cell, dim in sheaf.cell_dimensions.items():
                if not isinstance(dim, int) or dim < 0:
                    result['errors'].append(f"Cell {cell} has invalid dimension: {dim}")
                    result['valid'] = False
            
            # Check topology symmetry
            asymmetric_pairs = []
            for cell, neighbors in sheaf.topology.items():
                for neighbor in neighbors:
                    if neighbor in sheaf.topology and cell not in sheaf.topology[neighbor]:
                        asymmetric_pairs.append((cell, neighbor))
            
            if asymmetric_pairs:
                result['warnings'].append(f"Asymmetric topology relationships: {asymmetric_pairs}")
            
        except Exception as e:
            result['errors'].append(f"Basic structure validation error: {e}")
            result['valid'] = False
        
        return result
    
    def _validate_topology_consistency(self, sheaf: CellularSheaf) -> Dict[str, Any]:
        """Validate topological consistency of the cellular complex"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            # Check connectivity
            if not sheaf.validate_connectivity():
                result['warnings'].append("Sheaf is not connected")
            
            # Validate dimension consistency
            dimension_violations = []
            for cell in sheaf.cells:
                cell_dim = sheaf.cell_dimensions[cell]
                neighbors = sheaf.topology.get(cell, set())
                
                for neighbor in neighbors:
                    neighbor_dim = sheaf.cell_dimensions[neighbor]
                    # Adjacent cells should have dimensions differing by at most 1
                    if abs(cell_dim - neighbor_dim) > 1:
                        dimension_violations.append((cell, cell_dim, neighbor, neighbor_dim))
            
            if dimension_violations:
                result['warnings'].append(f"Dimension violations in adjacency: {dimension_violations}")
            
            # Check for isolated cells
            isolated_cells = [cell for cell, neighbors in sheaf.topology.items() if not neighbors]
            if isolated_cells:
                result['warnings'].append(f"Isolated cells found: {isolated_cells}")
            
            # Validate cellular complex properties
            euler_char = sheaf.compute_euler_characteristic()
            result['euler_characteristic'] = euler_char
            
            # Check for reasonable Euler characteristic bounds
            num_cells = len(sheaf.cells)
            if abs(euler_char) > num_cells:
                result['warnings'].append(f"Unusual Euler characteristic: {euler_char} for {num_cells} cells")
            
        except Exception as e:
            result['errors'].append(f"Topology validation error: {e}")
            result['valid'] = False
        
        return result
    
    def _validate_section_completeness(self, sheaf: CellularSheaf) -> Dict[str, Any]:
        """Validate that all cells have valid section data"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            # Check all cells have sections
            missing_sections = []
            for cell in sheaf.cells:
                if cell not in sheaf.sections:
                    missing_sections.append(cell)
                elif sheaf.sections[cell] is None:
                    missing_sections.append(f"{cell} (None data)")
            
            if missing_sections:
                result['errors'].append(f"Cells missing section data: {missing_sections}")
                result['valid'] = False
            
            # Validate section data types are compatible
            section_types = {}
            for cell, data in sheaf.sections.items():
                data_type = type(data)
                if data_type not in section_types:
                    section_types[data_type] = []
                section_types[data_type].append(cell)
            
            if len(section_types) > 1:
                # Multiple types might be okay, but check compatibility
                type_names = [t.__name__ for t in section_types.keys()]
                result['warnings'].append(f"Multiple section data types: {type_names}")
                
                # Check if types are compatible for restriction operations
                self._check_type_compatibility(sheaf, section_types, result)
            
            # Validate section data integrity
            corrupted_sections = []
            for cell, data in sheaf.sections.items():
                if not self._is_valid_section_data(data):
                    corrupted_sections.append(cell)
            
            if corrupted_sections:
                result['errors'].append(f"Corrupted section data in cells: {corrupted_sections}")
                result['valid'] = False
            
        except Exception as e:
            result['errors'].append(f"Section validation error: {e}")
            result['valid'] = False
        
        return result
    
    def _check_type_compatibility(self, sheaf: CellularSheaf, section_types: Dict[type, List[str]], 
                                result: Dict[str, Any]) -> None:
        """Check if different section data types are compatible"""
        type_list = list(section_types.keys())
        
        for i, type1 in enumerate(type_list):
            for type2 in type_list[i+1:]:
                # Check if restriction maps can handle type conversion
                cells1 = section_types[type1]
                cells2 = section_types[type2]
                
                # Find restriction maps between different types
                cross_type_restrictions = []
                for cell1 in cells1:
                    for cell2 in cells2:
                        if (cell1, cell2) in sheaf.restriction_maps:
                            cross_type_restrictions.append((cell1, cell2, type1, type2))
                
                if cross_type_restrictions:
                    result['warnings'].append(
                        f"Cross-type restrictions found: {cross_type_restrictions}"
                    )
    
    def _is_valid_section_data(self, data: Any) -> bool:
        """Check if section data is valid and not corrupted"""
        try:
            if data is None:
                return False
            
            if isinstance(data, np.ndarray):
                return not (np.isnan(data).any() or np.isinf(data).any())
            
            elif isinstance(data, dict):
                return all(v is not None for v in data.values())
            
            elif isinstance(data, (list, tuple)):
                return len(data) > 0 and all(x is not None for x in data)
            
            else:
                return True
                
        except Exception:
            return False
    
    def _validate_restriction_system(self, sheaf: CellularSheaf) -> Dict[str, Any]:
        """Validate the system of restriction maps forms a consistent structure"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            # Check restriction maps exist for all necessary pairs
            missing_restrictions = []
            for cell in sheaf.cells:
                neighbors = sheaf.topology.get(cell, set())
                for neighbor in neighbors:
                    if (cell, neighbor) not in sheaf.restriction_maps:
                        missing_restrictions.append((cell, neighbor))
            
            if missing_restrictions:
                result['warnings'].append(f"Missing restriction maps: {missing_restrictions}")
            
            # Validate restriction map properties
            invalid_restrictions = []
            for (source, target), restriction in sheaf.restriction_maps.items():
                if not isinstance(restriction, RestrictionMap):
                    invalid_restrictions.append((source, target, "not RestrictionMap"))
                    continue
                
                if restriction.source_cell != source:
                    invalid_restrictions.append((source, target, "source mismatch"))
                
                if restriction.target_cell != target:
                    invalid_restrictions.append((source, target, "target mismatch"))
                
                # Test restriction on actual data
                if source in sheaf.sections and target in sheaf.sections:
                    try:
                        source_data = sheaf.sections[source]
                        restricted_data = restriction.apply(source_data)
                        
                        # Check that restriction produces valid data
                        if not self._is_valid_section_data(restricted_data):
                            invalid_restrictions.append((source, target, "produces invalid data"))
                            
                    except Exception as e:
                        invalid_restrictions.append((source, target, f"application failed: {e}"))
            
            if invalid_restrictions:
                result['errors'].append(f"Invalid restriction maps: {invalid_restrictions}")
                result['valid'] = False
            
            # Check transitivity where applicable
            transitivity_violations = self._check_restriction_transitivity(sheaf)
            if transitivity_violations:
                result['errors'].append(f"Transitivity violations: {transitivity_violations}")
                result['valid'] = False
            
            # Check commutativity in special cases
            commutativity_violations = self._check_restriction_commutativity(sheaf)
            if commutativity_violations:
                result['warnings'].append(f"Commutativity violations: {commutativity_violations}")
            
        except Exception as e:
            result['errors'].append(f"Restriction system validation error: {e}")
            result['valid'] = False
        
        return result
    
    def _check_restriction_transitivity(self, sheaf: CellularSheaf) -> List[Tuple[str, str, str]]:
        """Check transitivity: r_{A,C} = r_{B,C} ∘ r_{A,B} where it should hold"""
        violations = []
        
        for (source, intermediate) in sheaf.restriction_maps:
            for (intermediate2, target) in sheaf.restriction_maps:
                if intermediate == intermediate2 and (source, target) in sheaf.restriction_maps:
                    # Check transitivity: r_{s,t} = r_{i,t} ∘ r_{s,i}
                    try:
                        r_si = sheaf.restriction_maps[(source, intermediate)]
                        r_it = sheaf.restriction_maps[(intermediate, target)]
                        r_st = sheaf.restriction_maps[(source, target)]
                        
                        # Test on actual data if available
                        if source in sheaf.sections:
                            data = sheaf.sections[source]
                            
                            # Compose: r_{i,t} ∘ r_{s,i}
                            via_intermediate = r_it.apply(r_si.apply(data))
                            
                            # Direct: r_{s,t}
                            direct = r_st.apply(data)
                            
                            # Check if results are equal
                            if not self._data_approximately_equal(via_intermediate, direct):
                                violations.append((source, intermediate, target))
                                
                    except Exception:
                        # If we can't test, assume violation
                        violations.append((source, intermediate, target))
        
        return violations
    
    def _check_restriction_commutativity(self, sheaf: CellularSheaf) -> List[Tuple[str, str]]:
        """Check commutativity where applicable (symmetric restrictions)"""
        violations = []
        
        for (source, target) in sheaf.restriction_maps:
            if (target, source) in sheaf.restriction_maps:
                # Both directions exist - check for reasonable commutativity
                try:
                    r_st = sheaf.restriction_maps[(source, target)]
                    r_ts = sheaf.restriction_maps[(target, source)]
                    
                    # Test on data if available
                    if source in sheaf.sections and target in sheaf.sections:
                        source_data = sheaf.sections[source]
                        target_data = sheaf.sections[target]
                        
                        # Apply restrictions both ways
                        restricted_source = r_st.apply(source_data)
                        restricted_target = r_ts.apply(target_data)
                        
                        # Check if they're compatible (for overlapping regions)
                        if not r_st.check_compatibility(source_data, target_data):
                            violations.append((source, target))
                            
                except Exception:
                    violations.append((source, target))
        
        return violations
    
    def _data_approximately_equal(self, data1: Any, data2: Any, tolerance: float = 1e-10) -> bool:
        """Check if two data pieces are approximately equal"""
        try:
            if type(data1) != type(data2):
                return False
            
            if isinstance(data1, np.ndarray):
                if data1.shape != data2.shape:
                    return False
                return np.allclose(data1, data2, rtol=tolerance, atol=tolerance)
            
            elif isinstance(data1, (int, float)):
                return abs(data1 - data2) < tolerance
            
            elif isinstance(data1, dict):
                return data1 == data2
            
            elif isinstance(data1, (list, tuple)):
                if len(data1) != len(data2):
                    return False
                return all(self._data_approximately_equal(x, y, tolerance) 
                          for x, y in zip(data1, data2))
            
            else:
                return data1 == data2
                
        except Exception:
            return False
    
    def _validate_gluing_conditions(self, sheaf: CellularSheaf) -> Dict[str, Any]:
        """Validate gluing conditions for local-to-global reconstruction"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            # Check local compatibility around each cell
            incompatible_regions = []
            
            for cell in sheaf.cells:
                local_violations = self._check_local_gluing(sheaf, cell)
                if local_violations:
                    incompatible_regions.extend(local_violations)
            
            if incompatible_regions:
                result['errors'].append(f"Local gluing violations: {incompatible_regions}")
                result['valid'] = False
            
            # Check global consistency
            global_violations = self._check_global_gluing(sheaf)
            if global_violations:
                result['errors'].append(f"Global gluing violations: {global_violations}")
                result['valid'] = False
            
            # Validate overlap regions are consistent
            overlap_violations = self._check_overlap_consistency(sheaf)
            if overlap_violations:
                result['warnings'].append(f"Overlap inconsistencies: {overlap_violations}")
            
        except Exception as e:
            result['errors'].append(f"Gluing validation error: {e}")
            result['valid'] = False
        
        return result
    
    def _check_local_gluing(self, sheaf: CellularSheaf, cell: str) -> List[str]:
        """Check local compatibility around a specific cell"""
        violations = []
        
        try:
            neighbors = sheaf.topology.get(cell, set())
            if not neighbors or cell not in sheaf.sections:
                return violations
            
            cell_data = sheaf.sections[cell]
            
            for neighbor in neighbors:
                if neighbor not in sheaf.sections:
                    continue
                
                neighbor_data = sheaf.sections[neighbor]
                
                # Check compatibility via restriction maps
                if (cell, neighbor) in sheaf.restriction_maps:
                    r_cn = sheaf.restriction_maps[(cell, neighbor)]
                    
                    try:
                        if not r_cn.check_compatibility(cell_data, neighbor_data):
                            violations.append(f"{cell}-{neighbor}")
                    except Exception:
                        violations.append(f"{cell}-{neighbor} (check failed)")
        
        except Exception as e:
            violations.append(f"{cell} (error: {e})")
        
        return violations
    
    def _check_global_gluing(self, sheaf: CellularSheaf) -> List[str]:
        """Check global gluing consistency across the entire sheaf"""
        violations = []
        
        try:
            # Use a graph traversal to check consistency
            visited = set()
            
            if not sheaf.cells:
                return violations
            
            start_cell = next(iter(sheaf.cells))
            queue = deque([start_cell])
            visited.add(start_cell)
            
            while queue:
                current = queue.popleft()
                neighbors = sheaf.topology.get(current, set())
                
                for neighbor in neighbors:
                    if neighbor in visited:
                        # Check consistency with already visited neighbor
                        if not self._check_consistency_pair(sheaf, current, neighbor):
                            violations.append(f"global_inconsistency_{current}_{neighbor}")
                    else:
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        except Exception as e:
            violations.append(f"global_check_error: {e}")
        
        return violations
    
    def _check_consistency_pair(self, sheaf: CellularSheaf, cell1: str, cell2: str) -> bool:
        """Check consistency between a pair of cells"""
        try:
            if (cell1 not in sheaf.sections or cell2 not in sheaf.sections or
                (cell1, cell2) not in sheaf.restriction_maps):
                return True  # Can't check if data is missing
            
            restriction = sheaf.restriction_maps[(cell1, cell2)]
            data1 = sheaf.sections[cell1]
            data2 = sheaf.sections[cell2]
            
            return restriction.check_compatibility(data1, data2)
            
        except Exception:
            return False
    
    def _check_overlap_consistency(self, sheaf: CellularSheaf) -> List[str]:
        """Check that overlap regions are consistently defined"""
        violations = []
        
        try:
            for (cell1, cell2), overlap_info in sheaf.overlap_regions.items():
                # Check that corresponding restriction maps exist
                if ((cell1, cell2) not in sheaf.restriction_maps and 
                    (cell2, cell1) not in sheaf.restriction_maps):
                    violations.append(f"overlap_without_restriction_{cell1}_{cell2}")
                
                # Validate overlap information format
                if not isinstance(overlap_info, dict):
                    violations.append(f"invalid_overlap_format_{cell1}_{cell2}")
                
                # Check that overlapping cells are actually neighbors
                if (cell2 not in sheaf.topology.get(cell1, set()) and 
                    cell1 not in sheaf.topology.get(cell2, set())):
                    violations.append(f"overlap_non_neighbors_{cell1}_{cell2}")
        
        except Exception as e:
            violations.append(f"overlap_check_error: {e}")
        
        return violations
    
    def _validate_cohomological_properties(self, sheaf: CellularSheaf) -> Dict[str, Any]:
        """Validate cohomological properties of the sheaf"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            # Basic cohomological checks
            
            # Check exactness properties where applicable
            exactness_violations = self._check_exactness_properties(sheaf)
            if exactness_violations:
                result['warnings'].append(f"Exactness violations: {exactness_violations}")
            
            # Check sheaf axioms
            axiom_violations = self._check_sheaf_axioms(sheaf)
            if axiom_violations:
                result['errors'].append(f"Sheaf axiom violations: {axiom_violations}")
                result['valid'] = False
            
            # Compute basic cohomological invariants
            cohom_invariants = self._compute_cohomological_invariants(sheaf)
            result['cohomological_invariants'] = cohom_invariants
            
        except Exception as e:
            result['errors'].append(f"Cohomological validation error: {e}")
            result['valid'] = False
        
        return result
    
    def _check_exactness_properties(self, sheaf: CellularSheaf) -> List[str]:
        """Check exactness properties in restriction sequences"""
        violations = []
        
        # This is a simplified check - full exactness is complex
        try:
            # Check that restriction maps preserve essential structure
            for (source, target), restriction in sheaf.restriction_maps.items():
                if source in sheaf.sections and target in sheaf.sections:
                    source_data = sheaf.sections[source]
                    
                    # Check that restriction doesn't lose essential information
                    restricted = restriction.apply(source_data)
                    
                    # Basic check: restriction should preserve non-triviality
                    if self._is_trivial_data(restricted) and not self._is_trivial_data(source_data):
                        violations.append(f"information_loss_{source}_{target}")
        
        except Exception as e:
            violations.append(f"exactness_check_error: {e}")
        
        return violations
    
    def _is_trivial_data(self, data: Any) -> bool:
        """Check if data is trivial (zero, empty, etc.)"""
        try:
            if data is None:
                return True
            
            if isinstance(data, np.ndarray):
                return np.allclose(data, 0) or data.size == 0
            
            elif isinstance(data, (int, float)):
                return abs(data) < 1e-12
            
            elif isinstance(data, dict):
                return len(data) == 0
            
            elif isinstance(data, (list, tuple)):
                return len(data) == 0
            
            else:
                return False
                
        except Exception:
            return False
    
    def _check_sheaf_axioms(self, sheaf: CellularSheaf) -> List[str]:
        """Check fundamental sheaf axioms"""
        violations = []
        
        try:
            # Axiom 1: Identity axiom - restriction to same cell is identity
            for cell in sheaf.cells:
                if (cell, cell) in sheaf.restriction_maps:
                    if cell in sheaf.sections:
                        data = sheaf.sections[cell]
                        restricted = sheaf.restriction_maps[(cell, cell)].apply(data)
                        if not self._data_approximately_equal(data, restricted):
                            violations.append(f"identity_axiom_{cell}")
            
            # Axiom 2: Uniqueness - if restrictions agree, global section is unique
            # This is complex to check in general, so we do a simplified version
            uniqueness_violations = self._check_uniqueness_axiom(sheaf)
            violations.extend(uniqueness_violations)
            
            # Axiom 3: Existence - if local sections are compatible, global section exists
            # This is also complex, so we check basic cases
            existence_violations = self._check_existence_axiom(sheaf)
            violations.extend(existence_violations)
        
        except Exception as e:
            violations.append(f"axiom_check_error: {e}")
        
        return violations
    
    def _check_uniqueness_axiom(self, sheaf: CellularSheaf) -> List[str]:
        """Check uniqueness axiom in simplified form"""
        violations = []
        
        # Check that if two sections agree on overlaps, they should be equal
        # (simplified version)
        try:
            cells_list = list(sheaf.cells)
            for i, cell1 in enumerate(cells_list):
                for cell2 in cells_list[i+1:]:
                    if (cell1 in sheaf.sections and cell2 in sheaf.sections and
                        cell2 in sheaf.topology.get(cell1, set())):
                        
                        # Check if sections agree on overlap
                        if ((cell1, cell2) in sheaf.restriction_maps and 
                            (cell2, cell1) in sheaf.restriction_maps):
                            
                            r12 = sheaf.restriction_maps[(cell1, cell2)]
                            r21 = sheaf.restriction_maps[(cell2, cell1)]
                            
                            data1 = sheaf.sections[cell1]
                            data2 = sheaf.sections[cell2]
                            
                            if not r12.check_compatibility(data1, data2):
                                violations.append(f"uniqueness_{cell1}_{cell2}")
        
        except Exception as e:
            violations.append(f"uniqueness_check_error: {e}")
        
        return violations
    
    def _check_existence_axiom(self, sheaf: CellularSheaf) -> List[str]:
        """Check existence axiom in simplified form"""
        violations = []
        
        # This is very complex in general - we do basic sanity checks
        try:
            # Check that we can reconstruct global structure from local data
            if not self._test_reconstruction_possibility(sheaf):
                violations.append("reconstruction_impossible")
        
        except Exception as e:
            violations.append(f"existence_check_error: {e}")
        
        return violations
    
    def _test_reconstruction_possibility(self, sheaf: CellularSheaf) -> bool:
        """Test if global reconstruction from local data is possible"""
        try:
            # Simple test: check that restriction maps form a connected system
            if len(sheaf.restriction_maps) == 0:
                return len(sheaf.cells) <= 1
            
            # Check that restriction system covers the topology
            covered_pairs = set(sheaf.restriction_maps.keys())
            topology_pairs = set()
            
            for cell, neighbors in sheaf.topology.items():
                for neighbor in neighbors:
                    topology_pairs.add((cell, neighbor))
            
            # At least some topology should be covered by restrictions
            coverage = len(covered_pairs.intersection(topology_pairs))
            return coverage > 0
            
        except Exception:
            return False
    
    def _compute_cohomological_invariants(self, sheaf: CellularSheaf) -> Dict[str, Any]:
        """Compute basic cohomological invariants"""
        invariants = {}
        
        try:
            # Euler characteristic
            invariants['euler_characteristic'] = sheaf.compute_euler_characteristic()
            
            # Dimension information
            invariants['max_dimension'] = max(sheaf.cell_dimensions.values()) if sheaf.cell_dimensions else 0
            invariants['min_dimension'] = min(sheaf.cell_dimensions.values()) if sheaf.cell_dimensions else 0
            
            # Connectivity information
            invariants['connected_components'] = self._count_connected_components(sheaf)
            invariants['is_connected'] = invariants['connected_components'] == 1
            
            # Restriction map statistics
            invariants['restriction_density'] = len(sheaf.restriction_maps) / max(1, len(sheaf.cells)**2)
            
        except Exception as e:
            invariants['computation_error'] = str(e)
        
        return invariants
    
    def _count_connected_components(self, sheaf: CellularSheaf) -> int:
        """Count connected components of the cellular complex"""
        if not sheaf.cells:
            return 0
        
        visited = set()
        components = 0
        
        for cell in sheaf.cells:
            if cell not in visited:
                components += 1
                # BFS to mark all connected cells
                queue = [cell]
                visited.add(cell)
                
                while queue:
                    current = queue.pop(0)
                    neighbors = sheaf.topology.get(current, set())
                    
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
        
        return components
    
    def _validate_topological_invariants(self, sheaf: CellularSheaf) -> Dict[str, Any]:
        """Validate topological invariants are reasonable"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        
        try:
            stats = sheaf.get_sheaf_statistics()
            
            # Check Euler characteristic bounds
            euler_char = stats['euler_characteristic']
            num_cells = stats['total_cells']
            
            if abs(euler_char) > num_cells:
                result['warnings'].append(f"Unusual Euler characteristic: {euler_char}")
            
            # Check connectivity
            if not stats['is_connected'] and num_cells > 1:
                result['warnings'].append("Sheaf is disconnected")
            
            # Check dimension distribution
            dim_dist = stats['dimension_distribution']
            if len(dim_dist) > 4:  # More than 3D + time
                result['warnings'].append(f"High-dimensional cells present: {dim_dist}")
            
            # Check average connectivity
            avg_neighbors = stats.get('avg_neighbors', 0)
            if avg_neighbors < 1 and num_cells > 1:
                result['warnings'].append(f"Low connectivity: avg {avg_neighbors} neighbors")
            
            if avg_neighbors > 2 * num_cells:
                result['warnings'].append(f"Unusually high connectivity: avg {avg_neighbors} neighbors")
            
        except Exception as e:
            result['errors'].append(f"Topological invariant validation error: {e}")
            result['valid'] = False
        
        return result
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation performance statistics"""
        stats = dict(self.validation_stats)
        
        if stats['validation_times']:
            stats['avg_validation_time'] = np.mean(stats['validation_times'])
            stats['max_validation_time'] = max(stats['validation_times'])
            stats['min_validation_time'] = min(stats['validation_times'])
        else:
            stats['avg_validation_time'] = 0
            stats['max_validation_time'] = 0
            stats['min_validation_time'] = 0
        
        total_cache_requests = stats['cache_hits'] + stats['cache_misses']
        stats['cache_hit_rate'] = stats['cache_hits'] / total_cache_requests if total_cache_requests > 0 else 0
        
        return stats
    
    def clear_validation_cache(self) -> None:
        """Clear validation cache"""
        self.validation_cache.clear()
        self.validation_stats['cache_hits'] = 0
        self.validation_stats['cache_misses'] = 0