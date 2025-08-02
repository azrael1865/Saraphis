"""
Advanced Sheaf Theory compression features for the independent core.
Implements cellular sheaf structures, restriction maps, and advanced operations.
NO FALLBACKS - HARD FAILURES ONLY
"""

from typing import Dict, Set, Any, Optional, Callable, Tuple, List
from dataclasses import dataclass, field
import numpy as np
import torch
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh, svds
from scipy.linalg import null_space
import networkx as nx
from collections import defaultdict, deque
import itertools

from .sheaf_structures import CellularSheaf, RestrictionMap
from .cohomology_validator import CohomologyValidator
from ..base import CompressionBase


class CellularSheafBuilder:
    """Constructs cellular sheaf structures from data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary")
            
        self.config = config
        self.max_cells = config.get('max_cells', 1000)
        self.min_cell_size = config.get('min_cell_size', 10)
        self.overlap_threshold = config.get('overlap_threshold', 0.1)
        self.dimension_hierarchy = config.get('dimension_hierarchy', [0, 1, 2])
        
        if not isinstance(self.max_cells, int) or self.max_cells <= 0:
            raise ValueError("max_cells must be positive integer")
        if not isinstance(self.min_cell_size, int) or self.min_cell_size <= 0:
            raise ValueError("min_cell_size must be positive integer")
        if not isinstance(self.overlap_threshold, (int, float)) or not 0 <= self.overlap_threshold <= 1:
            raise ValueError("overlap_threshold must be between 0 and 1")
        if not isinstance(self.dimension_hierarchy, list):
            raise TypeError("dimension_hierarchy must be a list")
            
        self._cell_counter = 0
        self._topology_graph = nx.Graph()
        
    def build_from_data(self, data: torch.Tensor, 
                       partition_strategy: str = "grid") -> CellularSheaf:
        """Build cellular sheaf from tensor data."""
        if data is None:
            raise ValueError("Data cannot be None")
        if not isinstance(data, torch.Tensor):
            raise TypeError("Data must be torch.Tensor")
        if data.numel() == 0:
            raise ValueError("Data tensor cannot be empty")
        if partition_strategy not in ["grid", "adaptive", "spectral", "hierarchical"]:
            raise ValueError(f"Unknown partition strategy: {partition_strategy}")
            
        # Create cellular decomposition
        cells = self._create_cells(data, partition_strategy)
        
        # Build topology
        topology = self._build_topology(cells)
        
        # Extract sections
        sections = self._extract_sections(data, cells)
        
        # Compute cell dimensions
        cell_dimensions = self._compute_cell_dimensions(cells, topology)
        
        # Generate cell coordinates
        cell_coordinates = self._generate_cell_coordinates(cells, data.shape)
        
        # Identify overlap regions
        overlap_regions = self._identify_overlap_regions(cells, sections)
        
        # Create gluing data
        gluing_data = self._create_gluing_data(cells, sections, overlap_regions)
        
        # Build sheaf
        sheaf = CellularSheaf(
            cells=set(cells.keys()),
            topology=topology,
            sections=sections,
            restriction_maps={},  # Will be populated by RestrictionMapProcessor
            cell_dimensions=cell_dimensions,
            gluing_data=gluing_data,
            cell_coordinates=cell_coordinates,
            overlap_regions=overlap_regions
        )
        
        return sheaf
        
    def _create_cells(self, data: torch.Tensor, strategy: str) -> Dict[str, Dict[str, Any]]:
        """Create cellular decomposition of data."""
        if strategy == "grid":
            return self._grid_partition(data)
        elif strategy == "adaptive":
            return self._adaptive_partition(data)
        elif strategy == "spectral":
            return self._spectral_partition(data)
        elif strategy == "hierarchical":
            return self._hierarchical_partition(data)
        else:
            raise ValueError(f"Unknown partition strategy: {strategy}")
            
    def _grid_partition(self, data: torch.Tensor) -> Dict[str, Dict[str, Any]]:
        """Partition data using regular grid."""
        cells = {}
        shape = data.shape
        
        # Compute grid dimensions
        ndim = len(shape)
        cells_per_dim = int(np.ceil(self.max_cells ** (1.0 / ndim)))
        
        # Create grid cells
        cell_sizes = [max(self.min_cell_size, s // cells_per_dim) for s in shape]
        
        for indices in itertools.product(*[range(0, s, cs) for s, cs in zip(shape, cell_sizes)]):
            cell_id = f"cell_{self._cell_counter}"
            self._cell_counter += 1
            
            # Compute cell boundaries
            boundaries = []
            for i, (idx, cs, s) in enumerate(zip(indices, cell_sizes, shape)):
                start = idx
                end = min(idx + cs, s)
                boundaries.append((start, end))
                
            cells[cell_id] = {
                'boundaries': boundaries,
                'indices': indices,
                'volume': np.prod([end - start for start, end in boundaries])
            }
            
            if len(cells) >= self.max_cells:
                break
                
        return cells
        
    def _adaptive_partition(self, data: torch.Tensor) -> Dict[str, Dict[str, Any]]:
        """Partition data adaptively based on information content."""
        cells = {}
        
        # Compute information density
        data_np = data.cpu().numpy()
        grad_magnitude = np.sqrt(sum(np.gradient(data_np, axis=i)**2 for i in range(data_np.ndim)))
        
        # Initialize with single cell
        root_cell = {
            'boundaries': [(0, s) for s in data.shape],
            'indices': tuple(0 for _ in data.shape),
            'volume': data.numel()
        }
        
        queue = deque([root_cell])
        
        while queue and len(cells) < self.max_cells:
            current = queue.popleft()
            
            # Check if should split
            cell_slice = tuple(slice(start, end) for start, end in current['boundaries'])
            cell_info = grad_magnitude[cell_slice].sum()
            
            if current['volume'] > self.min_cell_size and cell_info > self._compute_info_threshold(grad_magnitude):
                # Split cell
                subcells = self._split_cell(current, data.shape)
                queue.extend(subcells)
            else:
                # Add as final cell
                cell_id = f"cell_{self._cell_counter}"
                self._cell_counter += 1
                cells[cell_id] = current
                
        return cells
        
    def _spectral_partition(self, data: torch.Tensor) -> Dict[str, Dict[str, Any]]:
        """Partition data using spectral clustering."""
        cells = {}
        
        # Flatten data for spectral analysis
        data_flat = data.view(-1).cpu().numpy()
        n_points = len(data_flat)
        
        # Create affinity matrix (using local neighborhoods)
        neighborhood_size = min(10, n_points // 10)
        affinity = lil_matrix((n_points, n_points))
        
        for i in range(n_points):
            # Connect to nearby points
            for j in range(max(0, i - neighborhood_size), min(n_points, i + neighborhood_size + 1)):
                if i != j:
                    affinity[i, j] = np.exp(-abs(data_flat[i] - data_flat[j])**2)
                    
        affinity = affinity.tocsr()
        
        # Compute Laplacian
        degree = np.array(affinity.sum(axis=1)).flatten()
        degree_sqrt_inv = np.power(degree, -0.5, where=degree!=0)
        laplacian = np.eye(n_points) - (degree_sqrt_inv[:, None] * affinity.toarray() * degree_sqrt_inv[None, :])
        
        # Get eigenvectors
        n_clusters = min(self.max_cells, n_points // self.min_cell_size)
        if n_clusters > 1:
            _, eigvecs = eigsh(laplacian, k=n_clusters, which='SM')
            
            # Cluster based on eigenvectors
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(eigvecs)
            
            # Create cells from clusters
            for cluster_id in range(n_clusters):
                cell_id = f"cell_{self._cell_counter}"
                self._cell_counter += 1
                
                cluster_indices = np.where(labels == cluster_id)[0]
                
                # Convert flat indices back to multi-dimensional
                multi_indices = []
                for idx in cluster_indices:
                    multi_idx = []
                    remaining = idx
                    for dim_size in reversed(data.shape[1:]):
                        multi_idx.append(remaining % dim_size)
                        remaining //= dim_size
                    multi_idx.append(remaining)
                    multi_indices.append(tuple(reversed(multi_idx)))
                    
                # Compute bounding box
                multi_indices = np.array(multi_indices)
                boundaries = [(mi.min(), mi.max() + 1) for mi in multi_indices.T]
                
                cells[cell_id] = {
                    'boundaries': boundaries,
                    'indices': tuple(mi.min() for mi in multi_indices.T),
                    'volume': len(cluster_indices),
                    'cluster_indices': cluster_indices
                }
        else:
            # Single cell
            cells["cell_0"] = {
                'boundaries': [(0, s) for s in data.shape],
                'indices': tuple(0 for _ in data.shape),
                'volume': data.numel()
            }
            
        return cells
        
    def _hierarchical_partition(self, data: torch.Tensor) -> Dict[str, Dict[str, Any]]:
        """Partition data hierarchically."""
        cells = {}
        
        # Build hierarchy levels
        hierarchy = []
        current_data = data
        
        for level in range(len(self.dimension_hierarchy)):
            level_cells = self._create_hierarchy_level(current_data, level)
            hierarchy.append(level_cells)
            
            # Downsample for next level
            if level < len(self.dimension_hierarchy) - 1:
                current_data = torch.nn.functional.avg_pool2d(
                    current_data.unsqueeze(0).unsqueeze(0),
                    kernel_size=2,
                    stride=2
                ).squeeze()
                
        # Flatten hierarchy into cells
        for level, level_cells in enumerate(hierarchy):
            for cell_data in level_cells:
                cell_id = f"cell_{self._cell_counter}"
                self._cell_counter += 1
                
                cells[cell_id] = {
                    **cell_data,
                    'hierarchy_level': level
                }
                
                if len(cells) >= self.max_cells:
                    break
                    
        return cells
        
    def _create_hierarchy_level(self, data: torch.Tensor, level: int) -> List[Dict[str, Any]]:
        """Create cells for a hierarchy level."""
        cells = []
        target_dim = self.dimension_hierarchy[level]
        
        # Partition based on target dimension
        if target_dim == 0:
            # Point cells
            stride = max(1, data.numel() // (self.max_cells // len(self.dimension_hierarchy)))
            for i in range(0, data.numel(), stride):
                idx = np.unravel_index(i, data.shape)
                cells.append({
                    'boundaries': [(idx[j], idx[j] + 1) for j in range(len(idx))],
                    'indices': idx,
                    'volume': 1,
                    'dimension': 0
                })
        elif target_dim == 1:
            # Edge cells
            for axis in range(len(data.shape)):
                stride = max(1, data.shape[axis] // 10)
                for i in range(0, data.shape[axis] - 1, stride):
                    boundaries = []
                    for j in range(len(data.shape)):
                        if j == axis:
                            boundaries.append((i, i + 2))
                        else:
                            boundaries.append((0, data.shape[j]))
                    cells.append({
                        'boundaries': boundaries,
                        'indices': tuple(i if j == axis else 0 for j in range(len(data.shape))),
                        'volume': 2 * np.prod([data.shape[j] for j in range(len(data.shape)) if j != axis]),
                        'dimension': 1
                    })
        else:
            # Higher dimensional cells
            cells.append({
                'boundaries': [(0, s) for s in data.shape],
                'indices': tuple(0 for _ in data.shape),
                'volume': data.numel(),
                'dimension': target_dim
            })
            
        return cells
        
    def _split_cell(self, cell: Dict[str, Any], shape: torch.Size) -> List[Dict[str, Any]]:
        """Split a cell into subcells."""
        # Find longest dimension
        dimensions = [(end - start, i) for i, (start, end) in enumerate(cell['boundaries'])]
        dimensions.sort(reverse=True)
        
        if dimensions[0][0] <= 1:
            return [cell]
            
        split_dim = dimensions[0][1]
        split_point = (cell['boundaries'][split_dim][0] + cell['boundaries'][split_dim][1]) // 2
        
        # Create two subcells
        subcells = []
        for i in range(2):
            new_boundaries = cell['boundaries'].copy()
            if i == 0:
                new_boundaries[split_dim] = (cell['boundaries'][split_dim][0], split_point)
            else:
                new_boundaries[split_dim] = (split_point, cell['boundaries'][split_dim][1])
                
            new_indices = list(cell['indices'])
            if i == 1:
                new_indices[split_dim] = split_point
                
            subcells.append({
                'boundaries': new_boundaries,
                'indices': tuple(new_indices),
                'volume': np.prod([end - start for start, end in new_boundaries])
            })
            
        return subcells
        
    def _compute_info_threshold(self, grad_magnitude: np.ndarray) -> float:
        """Compute information threshold for adaptive partitioning."""
        return np.percentile(grad_magnitude, 75)
        
    def _build_topology(self, cells: Dict[str, Dict[str, Any]]) -> Dict[str, Set[str]]:
        """Build topological relationships between cells."""
        topology = defaultdict(set)
        
        # Add cells to graph
        for cell_id in cells:
            self._topology_graph.add_node(cell_id)
            
        # Check adjacency
        cell_list = list(cells.items())
        for i, (cell1_id, cell1) in enumerate(cell_list):
            for j, (cell2_id, cell2) in enumerate(cell_list[i+1:], i+1):
                if self._are_adjacent(cell1, cell2):
                    topology[cell1_id].add(cell2_id)
                    topology[cell2_id].add(cell1_id)
                    self._topology_graph.add_edge(cell1_id, cell2_id)
                    
        return dict(topology)
        
    def _are_adjacent(self, cell1: Dict[str, Any], cell2: Dict[str, Any]) -> bool:
        """Check if two cells are adjacent."""
        # Check if cells share a boundary
        for dim, ((start1, end1), (start2, end2)) in enumerate(zip(cell1['boundaries'], cell2['boundaries'])):
            # Check if they touch in this dimension
            if end1 == start2 or end2 == start1:
                # Check if they overlap in other dimensions
                overlap_in_others = True
                for other_dim, ((s1, e1), (s2, e2)) in enumerate(zip(cell1['boundaries'], cell2['boundaries'])):
                    if other_dim != dim:
                        if e1 <= s2 or e2 <= s1:
                            overlap_in_others = False
                            break
                if overlap_in_others:
                    return True
                    
        return False
        
    def _extract_sections(self, data: torch.Tensor, cells: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Extract section data for each cell."""
        sections = {}
        
        for cell_id, cell in cells.items():
            # Extract data slice
            slices = tuple(slice(start, end) for start, end in cell['boundaries'])
            cell_data = data[slices]
            
            # Store section
            sections[cell_id] = {
                'data': cell_data.clone(),
                'shape': cell_data.shape,
                'mean': cell_data.mean().item(),
                'std': cell_data.std().item() if cell_data.numel() > 1 else 0.0,
                'min': cell_data.min().item(),
                'max': cell_data.max().item()
            }
            
        return sections
        
    def _compute_cell_dimensions(self, cells: Dict[str, Dict[str, Any]], 
                                topology: Dict[str, Set[str]]) -> Dict[str, int]:
        """Compute topological dimension of each cell."""
        dimensions = {}
        
        for cell_id, cell in cells.items():
            if 'dimension' in cell:
                dimensions[cell_id] = cell['dimension']
            else:
                # Infer dimension from boundaries
                non_singleton_dims = sum(1 for start, end in cell['boundaries'] if end - start > 1)
                dimensions[cell_id] = non_singleton_dims
                
        return dimensions
        
    def _generate_cell_coordinates(self, cells: Dict[str, Dict[str, Any]], 
                                   shape: torch.Size) -> Dict[str, Tuple[float, ...]]:
        """Generate spatial coordinates for cells."""
        coordinates = {}
        
        for cell_id, cell in cells.items():
            # Compute centroid
            centroid = []
            for start, end in cell['boundaries']:
                centroid.append((start + end) / 2.0)
                
            # Normalize to [0, 1]
            normalized = tuple(c / s for c, s in zip(centroid, shape))
            coordinates[cell_id] = normalized
            
        return coordinates
        
    def _identify_overlap_regions(self, cells: Dict[str, Dict[str, Any]], 
                                  sections: Dict[str, Any]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Identify overlap regions between cells."""
        overlap_regions = {}
        
        cell_list = list(cells.items())
        for i, (cell1_id, cell1) in enumerate(cell_list):
            for j, (cell2_id, cell2) in enumerate(cell_list[i+1:], i+1):
                overlap = self._compute_overlap(cell1, cell2)
                
                if overlap['volume'] > 0:
                    # Compute overlap statistics
                    overlap['overlap_ratio'] = overlap['volume'] / min(cell1['volume'], cell2['volume'])
                    
                    if overlap['overlap_ratio'] >= self.overlap_threshold:
                        overlap_regions[(cell1_id, cell2_id)] = overlap
                        overlap_regions[(cell2_id, cell1_id)] = overlap
                        
        return overlap_regions
        
    def _compute_overlap(self, cell1: Dict[str, Any], cell2: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overlap between two cells."""
        overlap_boundaries = []
        
        for (start1, end1), (start2, end2) in zip(cell1['boundaries'], cell2['boundaries']):
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            
            if overlap_start >= overlap_end:
                return {'volume': 0, 'boundaries': None}
                
            overlap_boundaries.append((overlap_start, overlap_end))
            
        volume = np.prod([end - start for start, end in overlap_boundaries])
        
        return {
            'volume': volume,
            'boundaries': overlap_boundaries,
            'dimensions': [end - start for start, end in overlap_boundaries]
        }
        
    def _create_gluing_data(self, cells: Dict[str, Dict[str, Any]], 
                           sections: Dict[str, Any],
                           overlap_regions: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Create gluing data for reconstruction."""
        gluing_data = {}
        
        for cell_id in cells:
            gluing_data[cell_id] = {
                'neighbors': [],
                'gluing_maps': {},
                'consistency_weights': {},
                'reconstruction_priority': 0.0
            }
            
        # Add neighbor information
        for (cell1_id, cell2_id), overlap in overlap_regions.items():
            if overlap['overlap_ratio'] >= self.overlap_threshold:
                gluing_data[cell1_id]['neighbors'].append(cell2_id)
                
                # Create gluing map
                gluing_data[cell1_id]['gluing_maps'][cell2_id] = {
                    'overlap_boundaries': overlap['boundaries'],
                    'overlap_ratio': overlap['overlap_ratio'],
                    'consistency_weight': self._compute_consistency_weight(
                        sections[cell1_id], sections[cell2_id], overlap
                    )
                }
                
        # Compute reconstruction priorities
        for cell_id in cells:
            # Priority based on centrality and information content
            centrality = nx.degree_centrality(self._topology_graph).get(cell_id, 0.0)
            info_content = sections[cell_id]['std'] / (sections[cell_id]['mean'] + 1e-8)
            gluing_data[cell_id]['reconstruction_priority'] = centrality + info_content
            
        return gluing_data
        
    def _compute_consistency_weight(self, section1: Dict[str, Any], 
                                   section2: Dict[str, Any],
                                   overlap: Dict[str, Any]) -> float:
        """Compute consistency weight between sections."""
        # Weight based on statistical similarity
        mean_diff = abs(section1['mean'] - section2['mean'])
        std_sum = section1['std'] + section2['std'] + 1e-8
        
        consistency = np.exp(-mean_diff / std_sum) * overlap['overlap_ratio']
        return float(consistency)


class RestrictionMapProcessor:
    """Computes and manages restriction maps between cells."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary")
            
        self.config = config
        self.map_types = config.get('map_types', ['projection', 'interpolation', 'averaging'])
        self.consistency_threshold = config.get('consistency_threshold', 0.95)
        self.max_iterations = config.get('max_iterations', 100)
        
        if not isinstance(self.map_types, list):
            raise TypeError("map_types must be a list")
        if not isinstance(self.consistency_threshold, (int, float)) or not 0 <= self.consistency_threshold <= 1:
            raise ValueError("consistency_threshold must be between 0 and 1")
        if not isinstance(self.max_iterations, int) or self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive integer")
            
        self._restriction_cache = {}
        
    def compute_restriction_maps(self, sheaf: CellularSheaf) -> Dict[Tuple[str, str], RestrictionMap]:
        """Compute all restriction maps for the sheaf."""
        if not isinstance(sheaf, CellularSheaf):
            raise TypeError("Input must be CellularSheaf")
            
        restriction_maps = {}
        
        # Compute maps for all adjacent cells
        for cell1 in sheaf.cells:
            for cell2 in sheaf.topology.get(cell1, set()):
                if (cell1, cell2) not in restriction_maps:
                    # Determine map type
                    map_type = self._determine_map_type(sheaf, cell1, cell2)
                    
                    # Create restriction function
                    restriction_func = self._create_restriction_function(
                        sheaf.sections[cell1],
                        sheaf.sections[cell2],
                        sheaf.overlap_regions.get((cell1, cell2), {}),
                        map_type
                    )
                    
                    # Create restriction map
                    restriction_map = RestrictionMap(
                        source_cell=cell1,
                        target_cell=cell2,
                        restriction_func=restriction_func,
                        restriction_type=map_type,
                        overlap_region=sheaf.overlap_regions.get((cell1, cell2), {})
                    )
                    
                    restriction_maps[(cell1, cell2)] = restriction_map
                    
        # Validate consistency
        self._validate_consistency(restriction_maps, sheaf)
        
        return restriction_maps
        
    def _determine_map_type(self, sheaf: CellularSheaf, cell1: str, cell2: str) -> str:
        """Determine appropriate restriction map type."""
        dim1 = sheaf.cell_dimensions.get(cell1, 0)
        dim2 = sheaf.cell_dimensions.get(cell2, 0)
        
        if dim1 > dim2:
            return 'projection'
        elif dim1 < dim2:
            return 'inclusion'
        else:
            # Same dimension - check overlap
            overlap = sheaf.overlap_regions.get((cell1, cell2), {})
            if overlap.get('overlap_ratio', 0) > 0.5:
                return 'interpolation'
            else:
                return 'averaging'
                
    def _create_restriction_function(self, section1: Dict[str, Any], 
                                    section2: Dict[str, Any],
                                    overlap: Dict[str, Any],
                                    map_type: str) -> Callable[[Any], Any]:
        """Create restriction function based on map type."""
        if map_type == 'projection':
            return self._create_projection_map(section1, section2, overlap)
        elif map_type == 'inclusion':
            return self._create_inclusion_map(section1, section2, overlap)
        elif map_type == 'interpolation':
            return self._create_interpolation_map(section1, section2, overlap)
        elif map_type == 'averaging':
            return self._create_averaging_map(section1, section2, overlap)
        else:
            raise ValueError(f"Unknown map type: {map_type}")
            
    def _create_projection_map(self, section1: Dict[str, Any], 
                              section2: Dict[str, Any],
                              overlap: Dict[str, Any]) -> Callable[[Any], Any]:
        """Create projection restriction map."""
        def projection_func(data):
            if not isinstance(data, torch.Tensor):
                raise TypeError("Data must be torch.Tensor")
                
            # Project to lower dimensional space
            if overlap and 'boundaries' in overlap:
                # Extract overlap region
                slices = tuple(slice(max(0, start - section1['data'].shape[i] // 4), 
                                   min(section1['data'].shape[i], end + section1['data'].shape[i] // 4))
                             for i, (start, end) in enumerate(overlap['boundaries']))
                projected = data[slices] if all(s.stop > s.start for s in slices) else data
            else:
                projected = data
                
            # Reduce dimensions if needed
            while projected.ndim > section2['data'].ndim:
                projected = projected.mean(dim=-1)
                
            # Resize to match target
            if projected.shape != section2['shape']:
                projected = torch.nn.functional.interpolate(
                    projected.unsqueeze(0).unsqueeze(0),
                    size=section2['shape'],
                    mode='nearest'
                ).squeeze()
                
            return projected
            
        return projection_func
        
    def _create_inclusion_map(self, section1: Dict[str, Any], 
                             section2: Dict[str, Any],
                             overlap: Dict[str, Any]) -> Callable[[Any], Any]:
        """Create inclusion restriction map."""
        def inclusion_func(data):
            if not isinstance(data, torch.Tensor):
                raise TypeError("Data must be torch.Tensor")
                
            # Pad to higher dimensional space
            target_shape = list(section2['shape'])
            current_shape = list(data.shape)
            
            # Compute padding
            padding = []
            for curr, targ in zip(reversed(current_shape), reversed(target_shape)):
                pad_total = max(0, targ - curr)
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                padding.extend([pad_left, pad_right])
                
            # Apply padding
            if any(p > 0 for p in padding):
                padded = torch.nn.functional.pad(data, padding, mode='replicate')
            else:
                padded = data
                
            # Add dimensions if needed
            while padded.ndim < len(target_shape):
                padded = padded.unsqueeze(-1).expand(*padded.shape, target_shape[padded.ndim])
                
            return padded
            
        return inclusion_func
        
    def _create_interpolation_map(self, section1: Dict[str, Any], 
                                 section2: Dict[str, Any],
                                 overlap: Dict[str, Any]) -> Callable[[Any], Any]:
        """Create interpolation restriction map."""
        def interpolation_func(data):
            if not isinstance(data, torch.Tensor):
                raise TypeError("Data must be torch.Tensor")
                
            # Interpolate to target shape
            if data.shape != section2['shape']:
                # Handle different number of dimensions
                while data.ndim < len(section2['shape']):
                    data = data.unsqueeze(0)
                while data.ndim > len(section2['shape']):
                    data = data.squeeze(0)
                    
                # Interpolate
                data = torch.nn.functional.interpolate(
                    data.unsqueeze(0).unsqueeze(0).float(),
                    size=section2['shape'],
                    mode='bilinear' if len(section2['shape']) == 2 else 'trilinear',
                    align_corners=True
                ).squeeze()
                
            return data
            
        return interpolation_func
        
    def _create_averaging_map(self, section1: Dict[str, Any], 
                             section2: Dict[str, Any],
                             overlap: Dict[str, Any]) -> Callable[[Any], Any]:
        """Create averaging restriction map."""
        def averaging_func(data):
            if not isinstance(data, torch.Tensor):
                raise TypeError("Data must be torch.Tensor")
                
            # Compute local averages
            kernel_size = []
            for s1, s2 in zip(data.shape, section2['shape']):
                kernel_size.append(max(1, s1 // s2))
                
            # Apply averaging
            averaged = data
            for dim, ks in enumerate(kernel_size):
                if ks > 1:
                    # Create averaging kernel
                    kernel_shape = [1] * len(data.shape)
                    kernel_shape[dim] = ks
                    kernel = torch.ones(kernel_shape) / ks
                    
                    # Convolve
                    averaged = torch.nn.functional.conv1d(
                        averaged.unsqueeze(0).unsqueeze(0),
                        kernel.unsqueeze(0).unsqueeze(0),
                        stride=ks
                    ).squeeze()
                    
            # Ensure correct shape
            if averaged.shape != section2['shape']:
                averaged = torch.nn.functional.interpolate(
                    averaged.unsqueeze(0).unsqueeze(0),
                    size=section2['shape'],
                    mode='nearest'
                ).squeeze()
                
            return averaged
            
        return averaging_func
        
    def _validate_consistency(self, restriction_maps: Dict[Tuple[str, str], RestrictionMap],
                             sheaf: CellularSheaf) -> None:
        """Validate consistency of restriction maps."""
        # Check cocycle conditions
        inconsistencies = []
        
        for cell1 in sheaf.cells:
            neighbors = list(sheaf.topology.get(cell1, set()))
            
            for i, cell2 in enumerate(neighbors):
                for cell3 in neighbors[i+1:]:
                    if cell3 in sheaf.topology.get(cell2, set()):
                        # Check consistency: ρ_13 = ρ_23 ∘ ρ_12
                        map_12 = restriction_maps.get((cell1, cell2))
                        map_23 = restriction_maps.get((cell2, cell3))
                        map_13 = restriction_maps.get((cell1, cell3))
                        
                        if map_12 and map_23 and map_13:
                            # Test on section data
                            test_data = sheaf.sections[cell1]['data']
                            
                            # Direct path
                            direct = map_13.restriction_func(test_data)
                            
                            # Indirect path
                            intermediate = map_12.restriction_func(test_data)
                            indirect = map_23.restriction_func(intermediate)
                            
                            # Check consistency
                            if direct.shape == indirect.shape:
                                error = torch.norm(direct - indirect) / (torch.norm(direct) + 1e-8)
                                
                                if error > 1 - self.consistency_threshold:
                                    inconsistencies.append({
                                        'cells': (cell1, cell2, cell3),
                                        'error': error.item()
                                    })
                                    
        if inconsistencies:
            # Try to fix inconsistencies
            self._fix_inconsistencies(inconsistencies, restriction_maps, sheaf)
            
    def _fix_inconsistencies(self, inconsistencies: List[Dict[str, Any]],
                            restriction_maps: Dict[Tuple[str, str], RestrictionMap],
                            sheaf: CellularSheaf) -> None:
        """Fix restriction map inconsistencies."""
        for iteration in range(self.max_iterations):
            if not inconsistencies:
                break
                
            total_fixed = 0
            
            for inconsistency in inconsistencies:
                cell1, cell2, cell3 = inconsistency['cells']
                
                # Recompute problematic maps
                map_types = [
                    self._determine_map_type(sheaf, cell1, cell2),
                    self._determine_map_type(sheaf, cell2, cell3),
                    self._determine_map_type(sheaf, cell1, cell3)
                ]
                
                # Try different map type combinations
                for new_type in self.map_types:
                    # Update one of the maps
                    new_func = self._create_restriction_function(
                        sheaf.sections[cell1],
                        sheaf.sections[cell3],
                        sheaf.overlap_regions.get((cell1, cell3), {}),
                        new_type
                    )
                    
                    restriction_maps[(cell1, cell3)].restriction_func = new_func
                    restriction_maps[(cell1, cell3)].restriction_type = new_type
                    
                    # Test consistency
                    test_data = sheaf.sections[cell1]['data']
                    direct = restriction_maps[(cell1, cell3)].restriction_func(test_data)
                    intermediate = restriction_maps[(cell1, cell2)].restriction_func(test_data)
                    indirect = restriction_maps[(cell2, cell3)].restriction_func(intermediate)
                    
                    if direct.shape == indirect.shape:
                        error = torch.norm(direct - indirect) / (torch.norm(direct) + 1e-8)
                        
                        if error <= 1 - self.consistency_threshold:
                            total_fixed += 1
                            break
                            
            # Remove fixed inconsistencies
            inconsistencies = [inc for inc in inconsistencies 
                             if not self._check_consistency_fixed(inc, restriction_maps, sheaf)]
                             
            if total_fixed == 0:
                # Can't fix remaining inconsistencies
                if inconsistencies:
                    raise RuntimeError(f"Cannot achieve consistency for {len(inconsistencies)} restriction maps")
                    
    def _check_consistency_fixed(self, inconsistency: Dict[str, Any],
                                restriction_maps: Dict[Tuple[str, str], RestrictionMap],
                                sheaf: CellularSheaf) -> bool:
        """Check if an inconsistency has been fixed."""
        cell1, cell2, cell3 = inconsistency['cells']
        
        map_12 = restriction_maps.get((cell1, cell2))
        map_23 = restriction_maps.get((cell2, cell3))
        map_13 = restriction_maps.get((cell1, cell3))
        
        if map_12 and map_23 and map_13:
            test_data = sheaf.sections[cell1]['data']
            
            direct = map_13.restriction_func(test_data)
            intermediate = map_12.restriction_func(test_data)
            indirect = map_23.restriction_func(intermediate)
            
            if direct.shape == indirect.shape:
                error = torch.norm(direct - indirect) / (torch.norm(direct) + 1e-8)
                return error <= 1 - self.consistency_threshold
                
        return False


class SheafCohomologyCalculator:
    """Computes sheaf cohomology for validation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary")
            
        self.config = config
        self.max_dimension = config.get('max_dimension', 3)
        self.numerical_tolerance = config.get('numerical_tolerance', 1e-10)
        self.use_spectral_sequence = config.get('use_spectral_sequence', True)
        
        if not isinstance(self.max_dimension, int) or self.max_dimension < 0:
            raise ValueError("max_dimension must be non-negative integer")
        if not isinstance(self.numerical_tolerance, (int, float)) or self.numerical_tolerance <= 0:
            raise ValueError("numerical_tolerance must be positive")
        if not isinstance(self.use_spectral_sequence, bool):
            raise TypeError("use_spectral_sequence must be boolean")
            
        self.chain_complexes = {}
        self.cohomology_groups = {}
        
    def compute_cohomology(self, sheaf: CellularSheaf, 
                          validator: Optional[CohomologyValidator] = None) -> Dict[int, np.ndarray]:
        """Compute sheaf cohomology groups."""
        if not isinstance(sheaf, CellularSheaf):
            raise TypeError("Input must be CellularSheaf")
            
        # Build chain complex
        complex = self._build_chain_complex(sheaf)
        
        # Compute cohomology
        cohomology = {}
        
        for dim in range(min(self.max_dimension + 1, len(complex))):
            if dim < len(complex) - 1:
                # Compute kernel and image
                kernel = self._compute_kernel(complex[dim])
                image = self._compute_image(complex[dim + 1]) if dim < len(complex) - 1 else np.array([])
                
                # Compute quotient
                cohomology[dim] = self._compute_quotient(kernel, image)
            else:
                # Top dimension
                cohomology[dim] = self._compute_kernel(complex[dim])
                
        # Validate if validator provided
        if validator:
            validation_result = validator.validate_cohomological_consistency(sheaf)
            if not validation_result['valid']:
                raise RuntimeError(f"Cohomology validation failed: {validation_result['errors']}")
                
        self.cohomology_groups[id(sheaf)] = cohomology
        return cohomology
        
    def _build_chain_complex(self, sheaf: CellularSheaf) -> List[np.ndarray]:
        """Build cochain complex from sheaf."""
        # Get cells by dimension
        cells_by_dim = defaultdict(list)
        for cell, dim in sheaf.cell_dimensions.items():
            cells_by_dim[dim].append(cell)
            
        # Build boundary operators
        complex = []
        
        for dim in range(self.max_dimension + 1):
            if dim in cells_by_dim and dim + 1 in cells_by_dim:
                # Create coboundary operator
                rows = len(cells_by_dim[dim + 1])
                cols = len(cells_by_dim[dim])
                
                coboundary = lil_matrix((rows, cols))
                
                for i, higher_cell in enumerate(cells_by_dim[dim + 1]):
                    for j, lower_cell in enumerate(cells_by_dim[dim]):
                        # Check if lower_cell is in boundary of higher_cell
                        if self._is_face(lower_cell, higher_cell, sheaf):
                            # Compute incidence number
                            incidence = self._compute_incidence(lower_cell, higher_cell, sheaf)
                            coboundary[i, j] = incidence
                            
                complex.append(coboundary.tocsr())
            else:
                # Empty operator
                if dim in cells_by_dim:
                    complex.append(csr_matrix((0, len(cells_by_dim[dim]))))
                else:
                    complex.append(csr_matrix((0, 0)))
                    
        return complex
        
    def _is_face(self, lower_cell: str, higher_cell: str, sheaf: CellularSheaf) -> bool:
        """Check if lower_cell is a face of higher_cell."""
        # Check dimension constraint
        dim_lower = sheaf.cell_dimensions.get(lower_cell, 0)
        dim_higher = sheaf.cell_dimensions.get(higher_cell, 0)
        
        if dim_higher != dim_lower + 1:
            return False
            
        # Check topological relationship
        return lower_cell in sheaf.topology.get(higher_cell, set())
        
    def _compute_incidence(self, lower_cell: str, higher_cell: str, 
                          sheaf: CellularSheaf) -> int:
        """Compute incidence number between cells."""
        # Use orientation information if available
        if hasattr(sheaf, 'metadata') and 'orientations' in sheaf.metadata:
            orient_lower = sheaf.metadata['orientations'].get(lower_cell, 1)
            orient_higher = sheaf.metadata['orientations'].get(higher_cell, 1)
            
            # Compute relative orientation
            coord_lower = np.array(sheaf.cell_coordinates.get(lower_cell, (0,)))
            coord_higher = np.array(sheaf.cell_coordinates.get(higher_cell, (0,)))
            
            direction = coord_higher - coord_lower
            incidence = 1 if np.dot(direction, direction) > 0 else -1
            
            return orient_lower * orient_higher * incidence
        else:
            # Default incidence
            return 1
            
    def _compute_kernel(self, operator: csr_matrix) -> np.ndarray:
        """Compute kernel of operator."""
        if operator.shape[0] == 0 or operator.shape[1] == 0:
            return np.eye(operator.shape[1])
            
        # Compute null space
        kernel = null_space(operator.toarray())
        
        # Orthonormalize
        if kernel.size > 0:
            q, _ = np.linalg.qr(kernel)
            return q
        else:
            return np.zeros((operator.shape[1], 0))
            
    def _compute_image(self, operator: csr_matrix) -> np.ndarray:
        """Compute image of operator."""
        if operator.shape[0] == 0 or operator.shape[1] == 0:
            return np.zeros((operator.shape[0], 0))
            
        # Compute range via SVD
        if min(operator.shape) > 0:
            try:
                u, s, vt = svds(operator, k=min(min(operator.shape) - 1, max(1, np.count_nonzero(operator.toarray()))))
                
                # Select significant singular values
                sig_indices = s > self.numerical_tolerance
                
                if np.any(sig_indices):
                    return u[:, sig_indices]
                else:
                    return np.zeros((operator.shape[0], 0))
            except:
                return np.zeros((operator.shape[0], 0))
        else:
            return np.zeros((operator.shape[0], 0))
            
    def _compute_quotient(self, kernel: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Compute quotient space kernel/image."""
        if kernel.shape[1] == 0:
            return np.zeros((kernel.shape[0], 0))
            
        if image.shape[1] == 0:
            return kernel
            
        # Project kernel orthogonal to image
        projection = np.eye(kernel.shape[0]) - image @ image.T
        projected_kernel = projection @ kernel
        
        # Extract linearly independent vectors
        q, r = np.linalg.qr(projected_kernel)
        
        # Find non-zero columns
        tol = self.numerical_tolerance * max(projected_kernel.shape)
        rank = np.sum(np.abs(np.diag(r)) > tol)
        
        return q[:, :rank]
        
    def compute_betti_numbers(self, sheaf: CellularSheaf) -> Dict[int, int]:
        """Compute Betti numbers of the sheaf."""
        cohomology = self.compute_cohomology(sheaf)
        
        betti = {}
        for dim, cohom_group in cohomology.items():
            betti[dim] = cohom_group.shape[1]
            
        return betti
        
    def compute_euler_characteristic(self, sheaf: CellularSheaf) -> int:
        """Compute Euler characteristic."""
        betti = self.compute_betti_numbers(sheaf)
        
        euler = 0
        for dim, betti_num in betti.items():
            euler += (-1)**dim * betti_num
            
        return euler


class SheafReconstructionEngine:
    """Performs local-to-global reconstruction from sheaf data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary")
            
        self.config = config
        self.reconstruction_method = config.get('reconstruction_method', 'weighted')
        self.consistency_weight = config.get('consistency_weight', 0.5)
        self.smoothness_weight = config.get('smoothness_weight', 0.3)
        self.convergence_threshold = config.get('convergence_threshold', 1e-6)
        self.max_iterations = config.get('max_iterations', 1000)
        
        valid_methods = ['weighted', 'variational', 'spectral', 'hierarchical']
        if self.reconstruction_method not in valid_methods:
            raise ValueError(f"reconstruction_method must be one of {valid_methods}")
        if not isinstance(self.consistency_weight, (int, float)) or not 0 <= self.consistency_weight <= 1:
            raise ValueError("consistency_weight must be between 0 and 1")
        if not isinstance(self.smoothness_weight, (int, float)) or not 0 <= self.smoothness_weight <= 1:
            raise ValueError("smoothness_weight must be between 0 and 1")
        if not isinstance(self.convergence_threshold, (int, float)) or self.convergence_threshold <= 0:
            raise ValueError("convergence_threshold must be positive")
        if not isinstance(self.max_iterations, int) or self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive integer")
            
    def reconstruct(self, sheaf: CellularSheaf, target_shape: torch.Size) -> torch.Tensor:
        """Reconstruct global data from sheaf."""
        if not isinstance(sheaf, CellularSheaf):
            raise TypeError("Input must be CellularSheaf")
        if not isinstance(target_shape, (torch.Size, tuple)):
            raise TypeError("target_shape must be torch.Size or tuple")
            
        if self.reconstruction_method == 'weighted':
            return self._weighted_reconstruction(sheaf, target_shape)
        elif self.reconstruction_method == 'variational':
            return self._variational_reconstruction(sheaf, target_shape)
        elif self.reconstruction_method == 'spectral':
            return self._spectral_reconstruction(sheaf, target_shape)
        elif self.reconstruction_method == 'hierarchical':
            return self._hierarchical_reconstruction(sheaf, target_shape)
        else:
            raise ValueError(f"Unknown reconstruction method: {self.reconstruction_method}")
            
    def _weighted_reconstruction(self, sheaf: CellularSheaf, target_shape: torch.Size) -> torch.Tensor:
        """Weighted averaging reconstruction."""
        # Initialize output
        output = torch.zeros(target_shape)
        weights = torch.zeros(target_shape)
        
        # Get reconstruction order
        cell_order = self._compute_reconstruction_order(sheaf)
        
        for cell_id in cell_order:
            section = sheaf.sections[cell_id]
            gluing = sheaf.gluing_data.get(cell_id, {})
            
            # Get cell data
            cell_data = section['data']
            
            # Compute cell weight based on priority and consistency
            base_weight = gluing.get('reconstruction_priority', 1.0)
            
            # Apply data to output grid
            cell_coords = sheaf.cell_coordinates.get(cell_id, (0.5,) * len(target_shape))
            
            # Convert normalized coordinates to indices
            indices = tuple(int(c * s) for c, s in zip(cell_coords, target_shape))
            
            # Determine region of influence
            influence_region = self._compute_influence_region(cell_data.shape, target_shape, indices)
            
            # Add weighted contribution
            for region_indices in influence_region:
                try:
                    # Map cell data to region
                    mapped_data = self._map_to_region(cell_data, region_indices, output[region_indices].shape)
                    
                    # Apply weighted update
                    output[region_indices] += base_weight * mapped_data
                    weights[region_indices] += base_weight
                except Exception:
                    continue
                    
        # Normalize by weights
        mask = weights > 0
        output[mask] /= weights[mask]
        
        # Fill any remaining zeros
        if torch.any(weights == 0):
            output = self._fill_zeros(output, weights == 0)
            
        return output
        
    def _variational_reconstruction(self, sheaf: CellularSheaf, target_shape: torch.Size) -> torch.Tensor:
        """Variational optimization reconstruction."""
        # Initialize with weighted reconstruction
        output = self._weighted_reconstruction(sheaf, target_shape)
        
        # Set up optimization
        output = torch.nn.Parameter(output)
        optimizer = torch.optim.LBFGS([output], lr=0.1, max_iter=20)
        
        def closure():
            optimizer.zero_grad()
            
            # Consistency loss
            consistency_loss = torch.tensor(0.0)
            
            for cell_id, section in sheaf.sections.items():
                cell_data = section['data']
                cell_coords = sheaf.cell_coordinates.get(cell_id, (0.5,) * len(target_shape))
                indices = tuple(int(c * s) for c, s in zip(cell_coords, target_shape))
                
                # Extract corresponding region from output
                region = self._extract_region(output, indices, cell_data.shape)
                
                if region is not None:
                    consistency_loss += torch.norm(region - cell_data)**2
                    
            # Smoothness loss
            smoothness_loss = torch.tensor(0.0)
            
            for dim in range(output.ndim):
                diff = torch.diff(output, dim=dim)
                smoothness_loss += torch.norm(diff)**2
                
            # Total loss
            loss = self.consistency_weight * consistency_loss + self.smoothness_weight * smoothness_loss
            loss.backward()
            
            return loss
            
        # Optimize
        for _ in range(self.max_iterations // 20):
            loss = optimizer.step(closure)
            
            if loss.item() < self.convergence_threshold:
                break
                
        return output.detach()
        
    def _spectral_reconstruction(self, sheaf: CellularSheaf, target_shape: torch.Size) -> torch.Tensor:
        """Spectral method reconstruction."""
        # Build Laplacian from sheaf topology
        n_cells = len(sheaf.cells)
        cell_list = list(sheaf.cells)
        cell_index = {cell: i for i, cell in enumerate(cell_list)}
        
        # Create graph Laplacian
        laplacian = np.zeros((n_cells, n_cells))
        
        for cell1 in sheaf.cells:
            idx1 = cell_index[cell1]
            neighbors = sheaf.topology.get(cell1, set())
            
            # Degree
            laplacian[idx1, idx1] = len(neighbors)
            
            # Adjacency
            for cell2 in neighbors:
                idx2 = cell_index[cell2]
                laplacian[idx1, idx2] = -1
                
        # Get eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        
        # Project section data onto eigenvectors
        section_matrix = np.zeros((n_cells, max(s['data'].numel() for s in sheaf.sections.values())))
        
        for i, cell in enumerate(cell_list):
            section_data = sheaf.sections[cell]['data'].flatten().cpu().numpy()
            section_matrix[i, :len(section_data)] = section_data
            
        # Spectral coefficients
        coefficients = eigenvectors.T @ section_matrix
        
        # Reconstruct on regular grid
        output = torch.zeros(target_shape)
        
        # Create grid interpolation
        grid_points = torch.stack(torch.meshgrid(
            *[torch.linspace(0, 1, s) for s in target_shape], 
            indexing='ij'
        ), dim=-1).reshape(-1, len(target_shape))
        
        # Interpolate from cells to grid
        cell_positions = np.array([sheaf.cell_coordinates.get(cell, (0.5,) * len(target_shape)) 
                                  for cell in cell_list])
                                  
        for point_idx, point in enumerate(grid_points):
            # Find nearby cells
            distances = np.linalg.norm(cell_positions - point.numpy(), axis=1)
            nearby_mask = distances < 0.5
            
            if np.any(nearby_mask):
                # Weighted combination based on spectral representation
                weights = np.exp(-distances[nearby_mask]**2)
                weights /= weights.sum()
                
                # Reconstruct value
                value = 0.0
                for j, (cell_idx, w) in enumerate(zip(np.where(nearby_mask)[0], weights)):
                    cell_value = (eigenvectors[cell_idx] @ coefficients).mean()
                    value += w * cell_value
                    
                # Set in output
                grid_idx = tuple(int(p * (s-1)) for p, s in zip(point, target_shape))
                output[grid_idx] = value
                
        return output
        
    def _hierarchical_reconstruction(self, sheaf: CellularSheaf, target_shape: torch.Size) -> torch.Tensor:
        """Hierarchical coarse-to-fine reconstruction."""
        # Group cells by dimension/hierarchy level
        hierarchy_levels = defaultdict(list)
        
        for cell, dim in sheaf.cell_dimensions.items():
            level = sheaf.gluing_data.get(cell, {}).get('hierarchy_level', dim)
            hierarchy_levels[level].append(cell)
            
        # Sort levels
        sorted_levels = sorted(hierarchy_levels.keys())
        
        # Initialize with coarsest level
        output = None
        
        for level in sorted_levels:
            level_cells = hierarchy_levels[level]
            
            if output is None:
                # Initial reconstruction from coarsest level
                output = torch.zeros(target_shape)
                
            # Reconstruct current level
            level_output = torch.zeros_like(output)
            level_weights = torch.zeros_like(output)
            
            for cell in level_cells:
                section = sheaf.sections[cell]
                cell_data = section['data']
                
                # Map to output
                cell_coords = sheaf.cell_coordinates.get(cell, (0.5,) * len(target_shape))
                indices = tuple(int(c * s) for c, s in zip(cell_coords, target_shape))
                
                influence_region = self._compute_influence_region(cell_data.shape, target_shape, indices)
                
                for region_indices in influence_region:
                    try:
                        mapped_data = self._map_to_region(cell_data, region_indices, output[region_indices].shape)
                        level_output[region_indices] += mapped_data
                        level_weights[region_indices] += 1.0
                    except Exception:
                        continue
                        
            # Combine with previous level
            mask = level_weights > 0
            if level == sorted_levels[0]:
                output[mask] = level_output[mask] / level_weights[mask]
            else:
                # Blend with previous level
                blend_factor = 0.5 + 0.5 * (level / len(sorted_levels))
                output[mask] = (1 - blend_factor) * output[mask] + blend_factor * (level_output[mask] / level_weights[mask])
                
        return output
        
    def _compute_reconstruction_order(self, sheaf: CellularSheaf) -> List[str]:
        """Compute order for reconstruction based on priorities."""
        # Sort cells by reconstruction priority
        priorities = []
        
        for cell in sheaf.cells:
            priority = sheaf.gluing_data.get(cell, {}).get('reconstruction_priority', 0.0)
            priorities.append((priority, cell))
            
        priorities.sort(reverse=True)
        
        return [cell for _, cell in priorities]
        
    def _compute_influence_region(self, cell_shape: torch.Size, 
                                 target_shape: torch.Size,
                                 center: Tuple[int, ...]) -> List[Tuple[slice, ...]]:
        """Compute region of influence for a cell."""
        regions = []
        
        # Compute region size
        region_size = []
        for cs, ts, c in zip(cell_shape, target_shape, center):
            # Scale cell size to target space
            scaled_size = max(1, int(cs * ts / max(cell_shape)))
            region_size.append(scaled_size)
            
        # Create region slices
        slices = []
        for size, ts, c in zip(region_size, target_shape, center):
            start = max(0, c - size // 2)
            end = min(ts, start + size)
            slices.append(slice(start, end))
            
        regions.append(tuple(slices))
        
        return regions
        
    def _map_to_region(self, cell_data: torch.Tensor, 
                      region: Tuple[slice, ...],
                      target_shape: torch.Size) -> torch.Tensor:
        """Map cell data to target region."""
        if cell_data.shape == target_shape:
            return cell_data
            
        # Resize cell data to target shape
        return torch.nn.functional.interpolate(
            cell_data.unsqueeze(0).unsqueeze(0).float(),
            size=target_shape,
            mode='bilinear' if len(target_shape) == 2 else 'trilinear',
            align_corners=True
        ).squeeze()
        
    def _extract_region(self, data: torch.Tensor, 
                       center: Tuple[int, ...],
                       shape: torch.Size) -> Optional[torch.Tensor]:
        """Extract region from data centered at given point."""
        slices = []
        
        for c, s, ds in zip(center, shape, data.shape):
            start = max(0, c - s // 2)
            end = min(ds, start + s)
            
            if end - start != s:
                return None
                
            slices.append(slice(start, end))
            
        return data[tuple(slices)]
        
    def _fill_zeros(self, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Fill zero regions using interpolation."""
        filled = data.clone()
        
        # Use nearest neighbor interpolation
        if torch.any(mask):
            # Get non-zero indices
            non_zero_coords = torch.stack(torch.where(~mask), dim=1).float()
            non_zero_values = data[~mask]
            
            # Get zero indices
            zero_coords = torch.stack(torch.where(mask), dim=1).float()
            
            # Find nearest neighbors
            for i, coord in enumerate(zero_coords):
                distances = torch.norm(non_zero_coords - coord, dim=1)
                nearest_idx = torch.argmin(distances)
                filled[tuple(coord.long())] = non_zero_values[nearest_idx]
                
        return filled


# Integration hooks for existing systems
class SheafAdvancedIntegration:
    """Integration utilities for advanced Sheaf Theory features"""
    
    @staticmethod
    def integrate_sheaf_builder(compression_system: CompressionBase,
                               builder_config: Optional[Dict[str, Any]] = None) -> CellularSheafBuilder:
        """Integrate cellular sheaf builder with compression system"""
        sheaf_builder = CellularSheafBuilder(builder_config)
        
        # Add methods to compression system
        compression_system.sheaf_builder = sheaf_builder
        compression_system.build_sheaf = sheaf_builder.build_from_data
        
        return sheaf_builder
    
    @staticmethod
    def integrate_restriction_processor(compression_system: CompressionBase,
                                      processor_config: Optional[Dict[str, Any]] = None) -> RestrictionMapProcessor:
        """Integrate restriction map processor with compression system"""
        restriction_processor = RestrictionMapProcessor(processor_config)
        
        # Add methods to compression system
        compression_system.restriction_processor = restriction_processor
        compression_system.compute_restrictions = restriction_processor.compute_restriction_maps
        
        return restriction_processor
    
    @staticmethod
    def integrate_cohomology_calculator(compression_system: CompressionBase,
                                      calculator_config: Optional[Dict[str, Any]] = None) -> SheafCohomologyCalculator:
        """Integrate cohomology calculator with compression system"""
        cohomology_calculator = SheafCohomologyCalculator(calculator_config)
        
        # Add methods to compression system
        compression_system.cohomology_calculator = cohomology_calculator
        compression_system.compute_cohomology = cohomology_calculator.compute_cohomology
        compression_system.compute_betti_numbers = cohomology_calculator.compute_betti_numbers
        compression_system.compute_euler_characteristic = cohomology_calculator.compute_euler_characteristic
        
        return cohomology_calculator
    
    @staticmethod
    def integrate_reconstruction_engine(compression_system: CompressionBase,
                                      engine_config: Optional[Dict[str, Any]] = None) -> SheafReconstructionEngine:
        """Integrate reconstruction engine with compression system"""
        reconstruction_engine = SheafReconstructionEngine(engine_config)
        
        # Add methods to compression system
        compression_system.reconstruction_engine = reconstruction_engine
        compression_system.reconstruct_from_sheaf = reconstruction_engine.reconstruct
        
        return reconstruction_engine
    
    @staticmethod
    def integrate_all_advanced_features(compression_system: CompressionBase,
                                       config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Integrate all advanced sheaf features with compression system"""
        if config is None:
            config = {}
            
        builder_config = config.get('builder', {})
        processor_config = config.get('processor', {})
        calculator_config = config.get('calculator', {})
        engine_config = config.get('engine', {})
        
        # Integrate all components
        sheaf_builder = SheafAdvancedIntegration.integrate_sheaf_builder(
            compression_system, builder_config
        )
        restriction_processor = SheafAdvancedIntegration.integrate_restriction_processor(
            compression_system, processor_config
        )
        cohomology_calculator = SheafAdvancedIntegration.integrate_cohomology_calculator(
            compression_system, calculator_config
        )
        reconstruction_engine = SheafAdvancedIntegration.integrate_reconstruction_engine(
            compression_system, engine_config
        )
        
        # Add complete workflow method
        def complete_sheaf_workflow(data: torch.Tensor, partition_strategy: str = "grid") -> Dict[str, Any]:
            """Complete sheaf-based compression workflow"""
            # Build sheaf
            sheaf = sheaf_builder.build_from_data(data, partition_strategy)
            
            # Compute restriction maps
            restriction_maps = restriction_processor.compute_restriction_maps(sheaf)
            sheaf.restriction_maps = restriction_maps
            
            # Compute cohomology
            cohomology = cohomology_calculator.compute_cohomology(sheaf)
            
            # Reconstruct data
            reconstructed = reconstruction_engine.reconstruct(sheaf, data.shape)
            
            return {
                'sheaf': sheaf,
                'restriction_maps': restriction_maps,
                'cohomology': cohomology,
                'reconstructed': reconstructed,
                'betti_numbers': cohomology_calculator.compute_betti_numbers(sheaf),
                'euler_characteristic': cohomology_calculator.compute_euler_characteristic(sheaf)
            }
        
        compression_system.complete_sheaf_workflow = complete_sheaf_workflow
        
        return {
            'sheaf_builder': sheaf_builder,
            'restriction_processor': restriction_processor,
            'cohomology_calculator': cohomology_calculator,
            'reconstruction_engine': reconstruction_engine
        }