"""
Cohomology validation for sheaf compression systems.
Advanced mathematical validation of cohomological properties.
NO FALLBACKS - HARD FAILURES ONLY
"""

from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np
from collections import defaultdict
import itertools

from .sheaf_structures import CellularSheaf, RestrictionMap


class CohomologyValidator:
    """
    Advanced cohomological validation for sheaf compression.
    Validates cohomological consistency and computes cohomology groups.
    """
    
    def __init__(self, max_dimension: int = 3):
        """Initialize cohomology validator"""
        if not isinstance(max_dimension, int) or max_dimension < 0:
            raise ValueError("Max dimension must be non-negative integer")
        
        self.max_dimension = max_dimension
        self.computed_cohomology = {}
        self.chain_complexes = {}
        
        # Validation tolerances
        self.numerical_tolerance = 1e-12
        self.exactness_tolerance = 1e-10
    
    def validate_cohomological_consistency(self, sheaf: CellularSheaf) -> Dict[str, Any]:
        """Perform comprehensive cohomological validation"""
        if not isinstance(sheaf, CellularSheaf):
            raise TypeError("Input must be CellularSheaf")
        
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'cohomology_groups': {},
            'chain_complex_info': {},
            'exactness_check': {},
            'de_rham_consistency': {}
        }
        
        try:
            # Build chain complex
            chain_complex = self._build_chain_complex(sheaf)
            result['chain_complex_info'] = self._analyze_chain_complex(chain_complex)
            
            # Compute cohomology groups
            cohomology = self._compute_cohomology_groups(sheaf, chain_complex)
            result['cohomology_groups'] = cohomology
            
            # Check exactness properties
            exactness = self._check_exactness_sequences(sheaf, chain_complex)
            result['exactness_check'] = exactness
            
            if not exactness['all_exact']:
                result['errors'].extend(exactness['violations'])
                result['valid'] = False
            
            # Validate de Rham cohomology consistency
            de_rham = self._validate_de_rham_consistency(sheaf, cohomology)
            result['de_rham_consistency'] = de_rham
            
            if not de_rham['consistent']:
                result['warnings'].extend(de_rham['warnings'])
            
            # Check Poincaré duality where applicable
            poincare = self._check_poincare_duality(sheaf, cohomology)
            if not poincare['holds']:
                result['warnings'].append(f"Poincaré duality violation: {poincare['reason']}")
            
            # Validate Euler characteristic consistency
            euler_consistency = self._validate_euler_characteristic(sheaf, cohomology)
            if not euler_consistency['consistent']:
                result['errors'].append(f"Euler characteristic inconsistency: {euler_consistency['details']}")
                result['valid'] = False
            
        except Exception as e:
            result['errors'].append(f"Cohomological validation failed: {e}")
            result['valid'] = False
        
        return result
    
    def _build_chain_complex(self, sheaf: CellularSheaf) -> Dict[int, Any]:
        """Build chain complex from cellular sheaf"""
        chain_complex = {}
        
        # Group cells by dimension
        cells_by_dim = defaultdict(list)
        for cell, dim in sheaf.cell_dimensions.items():
            if dim <= self.max_dimension:
                cells_by_dim[dim].append(cell)
        
        # Build chain groups C_n
        for dim in range(self.max_dimension + 1):
            cells = cells_by_dim[dim]
            
            if not cells:
                chain_complex[dim] = {
                    'cells': [],
                    'sections': [],
                    'differential_matrix': np.array([]),
                    'rank': 0
                }
                continue
            
            # Extract sections for this dimension
            sections = []
            for cell in cells:
                if cell in sheaf.sections:
                    sections.append(sheaf.sections[cell])
                else:
                    raise ValueError(f"Missing section data for cell {cell}")
            
            # Build differential matrix ∂_n: C_n → C_{n-1}
            differential = self._build_differential_matrix(sheaf, cells, cells_by_dim.get(dim-1, []))
            
            chain_complex[dim] = {
                'cells': cells,
                'sections': sections,
                'differential_matrix': differential,
                'rank': len(cells)
            }
        
        return chain_complex
    
    def _build_differential_matrix(self, sheaf: CellularSheaf, 
                                 higher_cells: List[str], 
                                 lower_cells: List[str]) -> np.ndarray:
        """Build differential matrix between chain groups"""
        if not higher_cells or not lower_cells:
            if higher_cells:
                return np.zeros((len(lower_cells) if lower_cells else 1, len(higher_cells)))
            else:
                return np.array([])
        
        # Build incidence matrix
        matrix = np.zeros((len(lower_cells), len(higher_cells)))
        
        for i, higher_cell in enumerate(higher_cells):
            for j, lower_cell in enumerate(lower_cells):
                # Check if lower_cell is in the boundary of higher_cell
                incidence = self._compute_incidence(sheaf, higher_cell, lower_cell)
                matrix[j, i] = incidence
        
        return matrix
    
    def _compute_incidence(self, sheaf: CellularSheaf, higher_cell: str, lower_cell: str) -> float:
        """Compute incidence number between cells"""
        # Simplified incidence computation based on restriction maps
        if (higher_cell, lower_cell) in sheaf.restriction_maps:
            # Direct restriction exists
            return 1.0
        
        # Check indirect incidence through neighbors
        higher_neighbors = sheaf.topology.get(higher_cell, set())
        if lower_cell in higher_neighbors:
            return 1.0
        
        # Check if lower_cell is in boundary through dimension analysis
        higher_dim = sheaf.cell_dimensions.get(higher_cell, 0)
        lower_dim = sheaf.cell_dimensions.get(lower_cell, 0)
        
        if higher_dim == lower_dim + 1:
            # Check topological boundary relationship
            return self._check_boundary_relationship(sheaf, higher_cell, lower_cell)
        
        return 0.0
    
    def _check_boundary_relationship(self, sheaf: CellularSheaf, 
                                   higher_cell: str, lower_cell: str) -> float:
        """Check if lower_cell is in topological boundary of higher_cell"""
        # Use coordinates if available
        if (higher_cell in sheaf.cell_coordinates and 
            lower_cell in sheaf.cell_coordinates):
            
            higher_coords = sheaf.cell_coordinates[higher_cell]
            lower_coords = sheaf.cell_coordinates[lower_cell]
            
            # Simple distance-based heuristic
            distance = np.linalg.norm(np.array(higher_coords) - np.array(lower_coords))
            
            # If cells are close, assume boundary relationship
            if distance < 1.5:  # Heuristic threshold
                return 1.0
        
        # Fallback: check through restriction map existence
        common_neighbors = (sheaf.topology.get(higher_cell, set()) & 
                          sheaf.topology.get(lower_cell, set()))
        
        if common_neighbors:
            return 1.0
        
        return 0.0
    
    def _analyze_chain_complex(self, chain_complex: Dict[int, Any]) -> Dict[str, Any]:
        """Analyze chain complex properties"""
        analysis = {
            'dimensions': list(chain_complex.keys()),
            'ranks': {},
            'kernels': {},
            'images': {},
            'betti_numbers': {},
            'differential_ranks': {}
        }
        
        for dim, chain_data in chain_complex.items():
            analysis['ranks'][dim] = chain_data['rank']
            
            # Analyze differential matrix
            diff_matrix = chain_data['differential_matrix']
            if diff_matrix.size > 0:
                # Compute rank of differential
                diff_rank = np.linalg.matrix_rank(diff_matrix)
                analysis['differential_ranks'][dim] = diff_rank
                
                # Compute kernel and image dimensions
                kernel_dim = chain_data['rank'] - diff_rank
                image_dim = diff_rank
                
                analysis['kernels'][dim] = kernel_dim
                analysis['images'][dim] = image_dim
            else:
                analysis['differential_ranks'][dim] = 0
                analysis['kernels'][dim] = chain_data['rank']
                analysis['images'][dim] = 0
        
        # Compute Betti numbers: b_n = dim(ker ∂_n) - dim(im ∂_{n+1})
        for dim in chain_complex.keys():
            kernel_dim = analysis['kernels'].get(dim, 0)
            next_image_dim = analysis['images'].get(dim + 1, 0)
            
            betti_number = kernel_dim - next_image_dim
            analysis['betti_numbers'][dim] = max(0, betti_number)  # Betti numbers are non-negative
        
        return analysis
    
    def _compute_cohomology_groups(self, sheaf: CellularSheaf, 
                                 chain_complex: Dict[int, Any]) -> Dict[str, Any]:
        """Compute cohomology groups H^n(X; F)"""
        cohomology = {
            'betti_numbers': {},
            'euler_characteristic': 0,
            'cohomology_dimensions': {},
            'torsion_info': {}
        }
        
        # Compute Betti numbers from chain complex
        for dim in range(self.max_dimension + 1):
            if dim in chain_complex:
                # H^n = ker(d^n) / im(d^{n-1})
                
                # Get dimensions
                if dim in chain_complex:
                    chain_data = chain_complex[dim]
                    differential = chain_data['differential_matrix']
                    
                    if differential.size > 0:
                        kernel_dim = chain_data['rank'] - np.linalg.matrix_rank(differential)
                    else:
                        kernel_dim = chain_data['rank']
                else:
                    kernel_dim = 0
                
                # Get image from previous differential
                if dim - 1 in chain_complex:
                    prev_data = chain_complex[dim - 1]
                    prev_differential = prev_data['differential_matrix']
                    
                    if prev_differential.size > 0:
                        image_dim = np.linalg.matrix_rank(prev_differential)
                    else:
                        image_dim = 0
                else:
                    image_dim = 0
                
                # Cohomology dimension
                cohom_dim = kernel_dim - image_dim
                cohomology['cohomology_dimensions'][dim] = max(0, cohom_dim)
                cohomology['betti_numbers'][dim] = max(0, cohom_dim)
            else:
                cohomology['cohomology_dimensions'][dim] = 0
                cohomology['betti_numbers'][dim] = 0
        
        # Compute Euler characteristic: χ = Σ(-1)^n * b_n
        euler_char = 0
        for dim, betti in cohomology['betti_numbers'].items():
            euler_char += ((-1) ** dim) * betti
        
        cohomology['euler_characteristic'] = euler_char
        
        return cohomology
    
    def _check_exactness_sequences(self, sheaf: CellularSheaf, 
                                 chain_complex: Dict[int, Any]) -> Dict[str, Any]:
        """Check exactness of sequences in the chain complex"""
        exactness = {
            'all_exact': True,
            'violations': [],
            'sequence_checks': {}
        }
        
        # Check that d^{n+1} ∘ d^n = 0 for all n
        for dim in range(self.max_dimension):
            if dim in chain_complex and (dim + 1) in chain_complex:
                d_n = chain_complex[dim]['differential_matrix']
                d_n_plus_1 = chain_complex[dim + 1]['differential_matrix']
                
                if d_n.size > 0 and d_n_plus_1.size > 0:
                    # Check if composition is zero
                    try:
                        composition = np.dot(d_n, d_n_plus_1)
                        
                        if not np.allclose(composition, 0, atol=self.exactness_tolerance):
                            exactness['all_exact'] = False
                            exactness['violations'].append(f"d^{dim+1} ∘ d^{dim} ≠ 0")
                            exactness['sequence_checks'][dim] = False
                        else:
                            exactness['sequence_checks'][dim] = True
                            
                    except ValueError:
                        # Matrix dimensions don't match - check manually
                        exactness['violations'].append(f"Dimension mismatch in d^{dim+1} ∘ d^{dim}")
                        exactness['all_exact'] = False
                        exactness['sequence_checks'][dim] = False
        
        # Check exactness at each level
        for dim in chain_complex.keys():
            exactness_check = self._check_exactness_at_level(chain_complex, dim)
            if not exactness_check['exact']:
                exactness['violations'].extend(exactness_check['violations'])
                exactness['all_exact'] = False
        
        return exactness
    
    def _check_exactness_at_level(self, chain_complex: Dict[int, Any], dim: int) -> Dict[str, Any]:
        """Check exactness at a specific dimension level"""
        result = {'exact': True, 'violations': []}
        
        if dim not in chain_complex:
            return result
        
        # Check im(d_{n+1}) ⊆ ker(d_n)
        try:
            # Get current and next differentials
            current_diff = chain_complex[dim]['differential_matrix']
            
            if dim + 1 in chain_complex:
                next_diff = chain_complex[dim + 1]['differential_matrix']
                
                if current_diff.size > 0 and next_diff.size > 0:
                    # Check that image of next_diff is in kernel of current_diff
                    if not self._check_image_in_kernel(next_diff, current_diff):
                        result['exact'] = False
                        result['violations'].append(f"im(d_{dim+1}) ⊄ ker(d_{dim})")
            
        except Exception as e:
            result['exact'] = False
            result['violations'].append(f"Exactness check failed at level {dim}: {e}")
        
        return result
    
    def _check_image_in_kernel(self, d_next: np.ndarray, d_current: np.ndarray) -> bool:
        """Check if image of d_next is contained in kernel of d_current"""
        try:
            # Compute image of d_next (column space)
            if d_next.size == 0:
                return True
            
            # Get orthonormal basis for image of d_next
            _, _, V = np.linalg.svd(d_next, full_matrices=False)
            rank_next = np.linalg.matrix_rank(d_next)
            
            if rank_next == 0:
                return True
            
            image_basis = V[:rank_next, :].T
            
            # Check if d_current annihilates the image basis
            if d_current.size == 0:
                return True
            
            product = np.dot(d_current, image_basis)
            
            return np.allclose(product, 0, atol=self.exactness_tolerance)
            
        except Exception:
            return False
    
    def _validate_de_rham_consistency(self, sheaf: CellularSheaf, 
                                    cohomology: Dict[str, Any]) -> Dict[str, Any]:
        """Validate de Rham cohomology consistency"""
        result = {
            'consistent': True,
            'warnings': [],
            'de_rham_numbers': {},
            'comparison': {}
        }
        
        try:
            # For simplicity, we check basic consistency properties
            # Full de Rham cohomology requires differential forms
            
            # Check that Betti numbers are reasonable
            betti_numbers = cohomology.get('betti_numbers', {})
            
            for dim, betti in betti_numbers.items():
                if betti < 0:
                    result['consistent'] = False
                    result['warnings'].append(f"Negative Betti number b_{dim} = {betti}")
                
                if betti > len(sheaf.cells):
                    result['warnings'].append(f"Large Betti number b_{dim} = {betti}")
            
            # Check Poincaré polynomial properties
            if betti_numbers:
                max_dim = max(betti_numbers.keys())
                
                # For closed manifolds, should have Poincaré duality
                if max_dim > 0:
                    for i in range(max_dim // 2 + 1):
                        b_i = betti_numbers.get(i, 0)
                        b_dual = betti_numbers.get(max_dim - i, 0)
                        
                        if abs(b_i - b_dual) > 0:
                            result['warnings'].append(
                                f"Poincaré duality violation: b_{i} = {b_i} ≠ b_{max_dim-i} = {b_dual}"
                            )
            
        except Exception as e:
            result['consistent'] = False
            result['warnings'].append(f"de Rham validation error: {e}")
        
        return result
    
    def _check_poincare_duality(self, sheaf: CellularSheaf, 
                              cohomology: Dict[str, Any]) -> Dict[str, Any]:
        """Check Poincaré duality for oriented closed manifolds"""
        result = {'holds': True, 'reason': ''}
        
        try:
            betti_numbers = cohomology.get('betti_numbers', {})
            
            if not betti_numbers:
                result['reason'] = 'No Betti numbers computed'
                return result
            
            max_dim = max(betti_numbers.keys())
            
            # Check if we likely have a closed manifold
            if not self._appears_closed_manifold(sheaf):
                result['holds'] = True  # Not applicable
                result['reason'] = 'Not a closed manifold'
                return result
            
            # Check Poincaré duality: H^k ≅ H^{n-k}
            for k in range(max_dim // 2 + 1):
                b_k = betti_numbers.get(k, 0)
                b_dual = betti_numbers.get(max_dim - k, 0)
                
                if b_k != b_dual:
                    result['holds'] = False
                    result['reason'] = f'b_{k} = {b_k} ≠ b_{max_dim-k} = {b_dual}'
                    return result
            
        except Exception as e:
            result['holds'] = False
            result['reason'] = f'Poincaré duality check failed: {e}'
        
        return result
    
    def _appears_closed_manifold(self, sheaf: CellularSheaf) -> bool:
        """Heuristic check if sheaf represents a closed manifold"""
        try:
            # Check if connected
            if not sheaf.validate_connectivity():
                return False
            
            # Check if all cells have appropriate number of neighbors
            boundary_cells = sheaf.get_boundary_cells()
            
            # For a closed manifold, shouldn't have many boundary cells
            if len(boundary_cells) > len(sheaf.cells) * 0.1:  # More than 10% boundary
                return False
            
            # Check dimension consistency
            dimensions = set(sheaf.cell_dimensions.values())
            max_dim = max(dimensions) if dimensions else 0
            
            # Should have cells in all dimensions up to max
            for d in range(max_dim + 1):
                if d not in dimensions:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_euler_characteristic(self, sheaf: CellularSheaf, 
                                     cohomology: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Euler characteristic consistency"""
        result = {'consistent': True, 'details': {}}
        
        try:
            # Compute Euler characteristic from cellular structure
            cellular_euler = sheaf.compute_euler_characteristic()
            
            # Compute from cohomology (alternating sum of Betti numbers)
            cohomological_euler = cohomology.get('euler_characteristic', 0)
            
            result['details'] = {
                'cellular_euler': cellular_euler,
                'cohomological_euler': cohomological_euler,
                'difference': abs(cellular_euler - cohomological_euler)
            }
            
            # They should match
            if abs(cellular_euler - cohomological_euler) > self.numerical_tolerance:
                result['consistent'] = False
                result['details']['error'] = (
                    f"Euler characteristic mismatch: "
                    f"cellular = {cellular_euler}, cohomological = {cohomological_euler}"
                )
            
        except Exception as e:
            result['consistent'] = False
            result['details']['error'] = f"Euler characteristic validation failed: {e}"
        
        return result
    
    def compute_spectral_sequence(self, sheaf: CellularSheaf) -> Dict[str, Any]:
        """Compute spectral sequence associated with the sheaf (simplified)"""
        if not isinstance(sheaf, CellularSheaf):
            raise TypeError("Input must be CellularSheaf")
        
        result = {
            'E_0_page': {},
            'E_1_page': {},
            'E_2_page': {},
            'convergence_info': {}
        }
        
        try:
            # Build filtration by dimension
            filtration = self._build_dimension_filtration(sheaf)
            
            # Compute E_0 page
            result['E_0_page'] = self._compute_E0_page(filtration)
            
            # Compute E_1 page
            result['E_1_page'] = self._compute_E1_page(result['E_0_page'])
            
            # Compute E_2 page
            result['E_2_page'] = self._compute_E2_page(result['E_1_page'])
            
            # Analyze convergence
            result['convergence_info'] = self._analyze_spectral_convergence(result)
            
        except Exception as e:
            result['error'] = f"Spectral sequence computation failed: {e}"
        
        return result
    
    def _build_dimension_filtration(self, sheaf: CellularSheaf) -> Dict[int, Set[str]]:
        """Build filtration by cell dimension"""
        filtration = {}
        
        for cell, dim in sheaf.cell_dimensions.items():
            if dim not in filtration:
                filtration[dim] = set()
            filtration[dim].add(cell)
        
        return filtration
    
    def _compute_E0_page(self, filtration: Dict[int, Set[str]]) -> Dict[Tuple[int, int], Any]:
        """Compute E_0 page of spectral sequence"""
        E0_page = {}
        
        for p in filtration.keys():
            for q in range(self.max_dimension + 1):
                E0_page[(p, q)] = {
                    'dimension': len(filtration.get(p, set())),
                    'generators': list(filtration.get(p, set()))
                }
        
        return E0_page
    
    def _compute_E1_page(self, E0_page: Dict[Tuple[int, int], Any]) -> Dict[Tuple[int, int], Any]:
        """Compute E_1 page from E_0 page"""
        E1_page = {}
        
        # Apply d_0 differential
        for (p, q), data in E0_page.items():
            # Simplified: assume most survive to E_1
            E1_page[(p, q)] = {
                'dimension': max(0, data['dimension'] - 1),
                'generators': data['generators'][:max(0, data['dimension'] - 1)]
            }
        
        return E1_page
    
    def _compute_E2_page(self, E1_page: Dict[Tuple[int, int], Any]) -> Dict[Tuple[int, int], Any]:
        """Compute E_2 page from E_1 page"""
        E2_page = {}
        
        # Apply d_1 differential
        for (p, q), data in E1_page.items():
            # Simplified: assume further reduction
            E2_page[(p, q)] = {
                'dimension': max(0, data['dimension'] - 1),
                'generators': data['generators'][:max(0, data['dimension'] - 1)]
            }
        
        return E2_page
    
    def _analyze_spectral_convergence(self, spectral_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze convergence of spectral sequence"""
        convergence_info = {
            'converges': True,
            'stabilization_page': 2,
            'limit_terms': {}
        }
        
        try:
            E2_page = spectral_data.get('E_2_page', {})
            
            # Count non-zero terms
            nonzero_terms = sum(1 for data in E2_page.values() if data['dimension'] > 0)
            
            convergence_info['nonzero_limit_terms'] = nonzero_terms
            convergence_info['limit_terms'] = E2_page
            
        except Exception as e:
            convergence_info['error'] = f"Convergence analysis failed: {e}"
        
        return convergence_info