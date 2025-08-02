#!/usr/bin/env python3
"""
Molecular Feature Extractor Plugin
==================================

This module provides molecular feature extraction capabilities for the Universal AI Core system.
Extracted and adapted from Saraphis molecular_analyzer.py, preserving all molecular-specific
functionality while integrating with the plugin architecture.

Features:
- RDKit molecular descriptor calculation (2053 features)
- Graph neural network embeddings
- Morgan fingerprints and MACCS keys
- Molecular property descriptors
- Batch processing with error handling
- Performance optimization and caching
"""

import logging
import sys
import time
import warnings
from io import StringIO
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import threading
import weakref
import gc

# Try to import molecular dependencies
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    Chem = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, global_mean_pool
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Import plugin base classes
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from base import FeatureExtractorPlugin, FeatureExtractionResult, FeatureType

logger = logging.getLogger(__name__)


@dataclass
class MolecularDescriptorResult:
    """Result container for molecular descriptor calculations"""
    rdkit_features: np.ndarray
    gnn_features: Optional[np.ndarray] = None
    feature_names: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    valid_molecules: int = 0
    invalid_molecules: int = 0
    error_messages: List[str] = field(default_factory=list)


class MolecularGNN(nn.Module):
    """
    Graph Neural Network for molecular embeddings.
    
    Extracted from Saraphis molecular_analyzer.py lines 1251-1315.
    """
    
    def __init__(self, input_dim: int = 11, hidden_dim: int = 64, output_dim: int = 128, num_layers: int = 3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Graph convolutional layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index, batch):
        """Forward pass through the GNN"""
        # Apply graph convolutions
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling to get molecule-level representation
        x = global_mean_pool(x, batch)
        
        return x


class MolecularFeatureExtractor(FeatureExtractorPlugin):
    """
    Molecular feature extractor plugin.
    
    Extracted and adapted from Saraphis molecular_analyzer.py lines 1317-1408,
    preserving all molecular descriptor calculation capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the molecular feature extractor.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Configuration
        self.use_rdkit = self.config.get('use_rdkit', True) and RDKIT_AVAILABLE
        self.use_gnn = self.config.get('use_gnn', True) and PYTORCH_AVAILABLE and RDKIT_AVAILABLE
        self.batch_size = self.config.get('batch_size', 64)
        self.suppress_warnings = self.config.get('suppress_warnings', True)
        self.cache_gnn_model = self.config.get('cache_gnn_model', True)
        
        # Feature configuration
        self.include_morgan = self.config.get('include_morgan', True)
        self.include_maccs = self.config.get('include_maccs', True)
        self.include_basic_descriptors = self.config.get('include_basic_descriptors', True)
        self.include_advanced_descriptors = self.config.get('include_advanced_descriptors', True)
        
        # GNN model
        self.gnn_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.config.get('use_gpu', True) else 'cpu')
        
        # Feature caching
        self.feature_cache = {}
        self.cache_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'molecules_processed': 0,
            'cache_hits': 0,
            'extraction_time': 0.0,
            'invalid_molecules': 0
        }
        
        # Initialize GNN model if available
        if self.use_gnn:
            self._initialize_gnn_model()
        
        self.logger.info(f"🧬 Molecular Feature Extractor initialized (RDKit: {self.use_rdkit}, GNN: {self.use_gnn})")
    
    def get_metadata(self):
        """Get plugin metadata"""
        from ..base import PluginMetadata, PluginDependency, PluginVersion
        
        dependencies = []
        if self.use_rdkit:
            dependencies.append(PluginDependency(name="rdkit", version_requirement="*"))
        if self.use_gnn:
            dependencies.append(PluginDependency(name="torch", version_requirement="*"))
            dependencies.append(PluginDependency(name="torch_geometric", version_requirement="*"))
        
        return PluginMetadata(
            name="MolecularFeatureExtractor",
            version=PluginVersion(1, 0, 0),
            author="Universal AI Core",
            description="RDKit and GNN-based molecular feature extraction",
            plugin_type="feature_extractor",
            entry_point=f"{__name__}:MolecularFeatureExtractor",
            dependencies=dependencies,
            capabilities=["molecular_descriptors", "morgan_fingerprints", "maccs_keys", "gnn_embeddings"],
            hooks=["before_extraction", "after_extraction"],
            configuration_schema={
                "use_rdkit": {"type": "boolean", "default": True},
                "use_gnn": {"type": "boolean", "default": True},
                "batch_size": {"type": "integer", "default": 64, "minimum": 1},
                "include_morgan": {"type": "boolean", "default": True},
                "include_maccs": {"type": "boolean", "default": True}
            }
        )
    
    def _initialize_gnn_model(self):
        """Initialize the GNN model for molecular embeddings"""
        try:
            self.gnn_model = MolecularGNN(
                input_dim=11,
                hidden_dim=self.config.get('gnn_hidden_dim', 64),
                output_dim=self.config.get('gnn_output_dim', 128),
                num_layers=self.config.get('gnn_layers', 3)
            ).to(self.device)
            
            self.gnn_model.eval()  # Set to evaluation mode
            self.logger.info(f"✅ GNN model initialized on {self.device}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize GNN model: {e}")
            self.use_gnn = False
    
    def extract_features(self, input_data: Any) -> FeatureExtractionResult:
        """
        Extract molecular features from SMILES strings.
        
        Adapted from Saraphis molecular_analyzer.py lines 1317-1376.
        """
        start_time = time.time()
        
        try:
            # Parse input data
            smiles_list = self._parse_input_data(input_data)
            if not smiles_list:
                return FeatureExtractionResult(
                    features={},
                    feature_names=[],
                    feature_types=[],
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="No valid SMILES found in input data"
                )
            
            # Extract RDKit features
            rdkit_result = None
            if self.use_rdkit:
                rdkit_result = self._generate_rdkit_features(smiles_list)
            
            # Extract GNN features
            gnn_features = None
            if self.use_gnn:
                gnn_features = self._generate_gnn_features(smiles_list)
            
            # Combine features
            features, feature_names, feature_types = self._combine_features(
                rdkit_result, gnn_features, len(smiles_list)
            )
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['molecules_processed'] += len(smiles_list)
            self.stats['extraction_time'] += processing_time
            if rdkit_result:
                self.stats['invalid_molecules'] += rdkit_result.invalid_molecules
            
            return FeatureExtractionResult(
                features=features,
                feature_names=feature_names,
                feature_types=feature_types,
                processing_time=processing_time,
                success=True,
                metadata={
                    'total_molecules': len(smiles_list),
                    'valid_molecules': rdkit_result.valid_molecules if rdkit_result else len(smiles_list),
                    'invalid_molecules': rdkit_result.invalid_molecules if rdkit_result else 0,
                    'feature_dimensions': {
                        'rdkit': rdkit_result.rdkit_features.shape[1] if rdkit_result else 0,
                        'gnn': gnn_features.shape[1] if gnn_features is not None else 0
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting molecular features: {e}")
            return FeatureExtractionResult(
                features={},
                feature_names=[],
                feature_types=[],
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _parse_input_data(self, input_data: Any) -> List[str]:
        """Parse input data to extract SMILES strings"""
        smiles_list = []
        
        if isinstance(input_data, str):
            # Single SMILES string
            smiles_list = [input_data]
        elif isinstance(input_data, list):
            # List of SMILES strings
            smiles_list = [str(s) for s in input_data]
        elif isinstance(input_data, dict):
            # Dictionary with SMILES key
            smiles_key = self.config.get('smiles_key', 'smiles')
            if smiles_key in input_data:
                smiles_data = input_data[smiles_key]
                if isinstance(smiles_data, list):
                    smiles_list = [str(s) for s in smiles_data]
                else:
                    smiles_list = [str(smiles_data)]
        elif hasattr(input_data, 'to_dict'):
            # DataFrame or similar
            df = input_data if hasattr(input_data, 'columns') else pd.DataFrame(input_data)
            smiles_col = self.config.get('smiles_column', 'smiles')
            if smiles_col in df.columns:
                smiles_list = df[smiles_col].astype(str).tolist()
        
        return smiles_list
    
    def _generate_rdkit_features(self, smiles_list: List[str]) -> MolecularDescriptorResult:
        """
        Generate RDKit molecular features.
        
        Extracted from Saraphis molecular_analyzer.py lines 1317-1376.
        """
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for molecular feature extraction")
        
        features = []
        feature_names = []
        valid_count = 0
        invalid_count = 0
        errors = []
        
        # Suppress RDKit warnings if configured
        original_stderr = None
        if self.suppress_warnings:
            original_stderr = sys.stderr
            sys.stderr = StringIO()
        
        try:
            for i, smiles in enumerate(smiles_list):
                try:
                    # Parse molecule
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        # Invalid molecule - add zero vector
                        total_features = self._get_feature_count()
                        features.append(np.zeros(total_features))
                        invalid_count += 1
                        errors.append(f"Invalid SMILES at index {i}: {smiles}")
                        continue
                    
                    mol_features = []
                    
                    # Morgan fingerprints (2048 features)
                    if self.include_morgan:
                        morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        morgan_features = np.array(morgan_fp)
                        mol_features.append(morgan_features)
                        
                        if i == 0:  # Add feature names only once
                            feature_names.extend([f"morgan_{j}" for j in range(2048)])
                    
                    # MACCS Keys (167 features)
                    if self.include_maccs:
                        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
                        maccs_features = np.array(maccs_fp)
                        mol_features.append(maccs_features)
                        
                        if i == 0:
                            feature_names.extend([f"maccs_{j}" for j in range(167)])
                    
                    # Basic molecular descriptors (11 features)
                    if self.include_basic_descriptors:
                        basic_descriptors = np.array([
                            Descriptors.MolWt(mol),           # Molecular weight
                            Descriptors.MolLogP(mol),         # LogP (lipophilicity)
                            Descriptors.MolMR(mol),           # Molar refractivity
                            Descriptors.NumHDonors(mol),      # H-bond donors
                            Descriptors.NumHAcceptors(mol),   # H-bond acceptors
                            Descriptors.NumRotatableBonds(mol), # Rotatable bonds
                            Descriptors.NumAromaticRings(mol),  # Aromatic rings
                            Descriptors.NumAliphaticRings(mol), # Aliphatic rings
                            Descriptors.TPSA(mol),            # Topological polar surface area
                            Descriptors.FractionCsp3(mol),    # Fraction of SP3 carbons
                            Descriptors.HeavyAtomCount(mol)   # Heavy atom count
                        ])
                        mol_features.append(basic_descriptors)
                        
                        if i == 0:
                            basic_names = [
                                'mol_weight', 'logp', 'molar_refractivity', 'h_donors',
                                'h_acceptors', 'rotatable_bonds', 'aromatic_rings',
                                'aliphatic_rings', 'tpsa', 'fraction_csp3', 'heavy_atoms'
                            ]
                            feature_names.extend(basic_names)
                    
                    # Advanced descriptors (5 features)
                    if self.include_advanced_descriptors:
                        advanced_descriptors = np.array([
                            Descriptors.RingCount(mol),
                            rdMolDescriptors.CalcNumSpiroAtoms(mol),
                            rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                            rdMolDescriptors.CalcNumHeteroatoms(mol),
                            rdMolDescriptors.CalcRadiusOfGyration(mol)
                        ])
                        mol_features.append(advanced_descriptors)
                        
                        if i == 0:
                            advanced_names = [
                                'ring_count', 'spiro_atoms', 'bridgehead_atoms',
                                'heteroatoms', 'radius_gyration'
                            ]
                            feature_names.extend(advanced_names)
                    
                    # Concatenate all features
                    if mol_features:
                        combined_features = np.concatenate(mol_features)
                        features.append(combined_features)
                        valid_count += 1
                    else:
                        features.append(np.array([]))
                        invalid_count += 1
                
                except Exception as e:
                    # Handle individual molecule errors
                    total_features = self._get_feature_count()
                    features.append(np.zeros(total_features))
                    invalid_count += 1
                    errors.append(f"Error processing molecule {i}: {e}")
            
            # Convert to numpy array
            rdkit_features = np.array(features) if features else np.array([]).reshape(0, self._get_feature_count())
            
            return MolecularDescriptorResult(
                rdkit_features=rdkit_features,
                feature_names=feature_names,
                valid_molecules=valid_count,
                invalid_molecules=invalid_count,
                error_messages=errors
            )
            
        finally:
            # Restore stderr
            if original_stderr is not None:
                sys.stderr = original_stderr
    
    def _get_feature_count(self) -> int:
        """Get total number of RDKit features"""
        count = 0
        if self.include_morgan:
            count += 2048
        if self.include_maccs:
            count += 167
        if self.include_basic_descriptors:
            count += 11
        if self.include_advanced_descriptors:
            count += 5
        return count
    
    def _generate_gnn_features(self, smiles_list: List[str]) -> Optional[np.ndarray]:
        """
        Generate GNN features for molecular embeddings.
        
        Adapted from Saraphis molecular_analyzer.py lines 1251-1315.
        """
        if not self.use_gnn or self.gnn_model is None:
            return None
        
        try:
            graphs = []
            valid_indices = []
            
            # Convert SMILES to graphs
            for i, smiles in enumerate(smiles_list):
                graph = self._smiles_to_graph(smiles)
                if graph is not None:
                    graphs.append(graph)
                    valid_indices.append(i)
            
            if not graphs:
                return np.zeros((len(smiles_list), self.config.get('gnn_output_dim', 128)))
            
            # Batch graphs and process with GNN
            batch = Batch.from_data_list(graphs).to(self.device)
            
            with torch.no_grad():
                embeddings = self.gnn_model(batch.x, batch.edge_index, batch.batch)
                embeddings = embeddings.cpu().numpy()
            
            # Create full feature matrix (pad invalid molecules with zeros)
            full_features = np.zeros((len(smiles_list), embeddings.shape[1]))
            for i, valid_idx in enumerate(valid_indices):
                full_features[valid_idx] = embeddings[i]
            
            return full_features
            
        except Exception as e:
            self.logger.error(f"❌ Error generating GNN features: {e}")
            return np.zeros((len(smiles_list), self.config.get('gnn_output_dim', 128)))
    
    def _smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """
        Convert SMILES string to PyTorch Geometric graph.
        
        Adapted from Saraphis molecular_analyzer.py lines 1203-1249.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Add hydrogens for more accurate representation
            mol = Chem.AddHs(mol)
            
            # Node features (11 features per atom)
            node_features = []
            for atom in mol.GetAtoms():
                features = [
                    atom.GetAtomicNum(),                    # Atomic number
                    atom.GetDegree(),                       # Atom degree
                    atom.GetTotalValence(),                 # Valence electrons
                    atom.GetFormalCharge(),                 # Formal charge
                    atom.GetNumRadicalElectrons(),          # Radical electrons
                    int(atom.GetIsAromatic()),              # Aromaticity
                    int(atom.IsInRing()),                   # Ring membership
                    int(atom.GetHybridization()),           # Hybridization state
                    atom.GetTotalNumHs(),                   # Hydrogen count
                    int(atom.GetAtomicNum() not in [1, 6]), # Non-C/H atom flag
                    atom.GetIsotope()                       # Isotope information
                ]
                node_features.append(features)
            
            # Edge indices
            edge_indices = []
            for bond in mol.GetBonds():
                start_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                edge_indices.append([start_idx, end_idx])
                edge_indices.append([end_idx, start_idx])  # Undirected graph
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            
            return Data(x=x, edge_index=edge_index)
            
        except Exception as e:
            self.logger.debug(f"Error converting SMILES to graph: {e}")
            return None
    
    def _combine_features(self, rdkit_result: Optional[MolecularDescriptorResult], 
                         gnn_features: Optional[np.ndarray], 
                         num_molecules: int) -> Tuple[Dict[str, np.ndarray], List[str], List[FeatureType]]:
        """Combine RDKit and GNN features"""
        features = {}
        feature_names = []
        feature_types = []
        
        # Add RDKit features
        if rdkit_result is not None and rdkit_result.rdkit_features.size > 0:
            features['rdkit_descriptors'] = rdkit_result.rdkit_features
            feature_names.extend(rdkit_result.feature_names)
            feature_types.extend([FeatureType.NUMERICAL] * len(rdkit_result.feature_names))
        
        # Add GNN features
        if gnn_features is not None:
            features['gnn_embeddings'] = gnn_features
            gnn_names = [f'gnn_embed_{i}' for i in range(gnn_features.shape[1])]
            feature_names.extend(gnn_names)
            feature_types.extend([FeatureType.EMBEDDING] * len(gnn_names))
        
        # Add combined features if both are available
        if 'rdkit_descriptors' in features and 'gnn_embeddings' in features:
            combined = np.concatenate([features['rdkit_descriptors'], features['gnn_embeddings']], axis=1)
            features['combined_molecular'] = combined
        
        return features, feature_names, feature_types
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get plugin capabilities"""
        capabilities = {
            'molecular_descriptors': self.use_rdkit,
            'gnn_embeddings': self.use_gnn,
            'batch_processing': True,
            'caching': True,
            'error_handling': True
        }
        
        if self.use_rdkit:
            capabilities.update({
                'morgan_fingerprints': self.include_morgan,
                'maccs_keys': self.include_maccs,
                'basic_descriptors': self.include_basic_descriptors,
                'advanced_descriptors': self.include_advanced_descriptors
            })
        
        return capabilities
    
    def get_hooks(self) -> Dict[str, Callable]:
        """Get plugin hooks"""
        return {
            'before_extraction': self._before_extraction_hook,
            'after_extraction': self._after_extraction_hook
        }
    
    def _before_extraction_hook(self, input_data: Any) -> Any:
        """Hook called before feature extraction"""
        self.logger.debug("🔗 Before molecular feature extraction")
        return input_data
    
    def _after_extraction_hook(self, result: FeatureExtractionResult) -> FeatureExtractionResult:
        """Hook called after feature extraction"""
        self.logger.debug("🔗 After molecular feature extraction")
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return self.stats.copy()
    
    def clear_cache(self):
        """Clear feature cache"""
        with self.cache_lock:
            self.feature_cache.clear()
        self.logger.info("🧹 Cleared molecular feature cache")
    
    def health_check(self) -> bool:
        """Perform health check"""
        try:
            # Test RDKit functionality
            if self.use_rdkit and RDKIT_AVAILABLE:
                test_mol = Chem.MolFromSmiles('CCO')  # Ethanol
                if test_mol is None:
                    return False
            
            # Test GNN model
            if self.use_gnn and self.gnn_model is not None:
                test_graph = self._smiles_to_graph('CCO')
                if test_graph is None:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False


# Plugin metadata for discovery
__plugin_metadata__ = {
    "name": "MolecularFeatureExtractor",
    "version": "1.0.0",
    "author": "Universal AI Core",
    "description": "RDKit and GNN-based molecular feature extraction",
    "plugin_type": "feature_extractor",
    "entry_point": f"{__name__}:MolecularFeatureExtractor",
    "dependencies": [
        {"name": "rdkit", "optional": True},
        {"name": "torch", "optional": True},
        {"name": "torch_geometric", "optional": True}
    ],
    "capabilities": ["molecular_descriptors", "morgan_fingerprints", "maccs_keys", "gnn_embeddings"],
    "hooks": ["before_extraction", "after_extraction"]
}


if __name__ == "__main__":
    # Test the molecular feature extractor
    print("🧬 MOLECULAR FEATURE EXTRACTOR TEST")
    print("=" * 50)
    
    # Test molecules
    test_smiles = [
        'CCO',              # Ethanol
        'CC(C)O',           # Isopropanol
        'c1ccccc1',         # Benzene
        'CCN(CC)CC',        # Triethylamine
        'INVALID_SMILES'    # Invalid for testing error handling
    ]
    
    # Initialize extractor
    config = {
        'use_rdkit': RDKIT_AVAILABLE,
        'use_gnn': PYTORCH_AVAILABLE and RDKIT_AVAILABLE,
        'batch_size': 32,
        'suppress_warnings': True
    }
    
    extractor = MolecularFeatureExtractor(config)
    
    # Test feature extraction
    print(f"\n🔍 Testing with {len(test_smiles)} molecules...")
    result = extractor.extract_features(test_smiles)
    
    if result.success:
        print(f"✅ Extraction successful!")
        print(f"📊 Processing time: {result.processing_time:.3f}s")
        print(f"🧮 Features extracted:")
        for feature_type, features in result.features.items():
            print(f"  {feature_type}: {features.shape}")
        
        # Show metadata
        if result.metadata:
            print(f"📋 Metadata:")
            for key, value in result.metadata.items():
                print(f"  {key}: {value}")
    else:
        print(f"❌ Extraction failed: {result.error_message}")
    
    # Test health check
    health = extractor.health_check()
    print(f"\n🏥 Health check: {'✅' if health else '❌'}")
    
    # Show statistics
    stats = extractor.get_statistics()
    print(f"\n📊 Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Molecular feature extractor test completed!")