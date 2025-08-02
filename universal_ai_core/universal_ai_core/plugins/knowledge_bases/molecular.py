#!/usr/bin/env python3
"""
Molecular Knowledge Base Plugin
==============================

This module provides molecular knowledge base capabilities for the Universal AI Core system.
Extracted and adapted from Saraphis symbolic reasoning patterns, specialized for molecular
knowledge management and chemical intelligence.

Features:
- Molecular property database management
- Chemical reaction knowledge storage
- Drug-target interaction repositories
- QSAR model knowledge base
- Pharmacophore pattern storage
- Molecular similarity searching
- Chemical rule knowledge management
- Bioactivity data integration
"""

import logging
import json
import time
import hashlib
import pickle
import threading
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

# Try to import molecular dependencies
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, DataStructs
    from rdkit.Chem.Fingerprints import FingerprintMols
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Import plugin base classes
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from base import (
    KnowledgeBasePlugin, KnowledgeItem, QueryResult, KnowledgeBaseMetadata,
    KnowledgeType, QueryType, KnowledgeFormat, OperationStatus
)

logger = logging.getLogger(__name__)


@dataclass
class MolecularKnowledgeItem(KnowledgeItem):
    """Extended knowledge item for molecular data"""
    smiles: str = ""
    inchi: str = ""
    molecular_weight: float = 0.0
    logp: float = 0.0
    fingerprint: Optional[np.ndarray] = None
    bioactivity_data: Dict[str, Any] = field(default_factory=dict)
    drug_targets: List[str] = field(default_factory=list)
    chemical_reactions: List[str] = field(default_factory=list)
    
    def calculate_molecular_properties(self):
        """Calculate molecular properties from SMILES"""
        if not self.smiles or not RDKIT_AVAILABLE:
            return
        
        try:
            mol = Chem.MolFromSmiles(self.smiles)
            if mol is not None:
                self.molecular_weight = Descriptors.MolWt(mol)
                self.logp = Descriptors.MolLogP(mol)
                
                # Calculate fingerprint
                fp = FingerprintMols.FingerprintMol(mol)
                self.fingerprint = np.array(fp)
                
                # Generate InChI
                self.inchi = Chem.MolToInchi(mol)
                
        except Exception as e:
            logger.error(f"Error calculating properties for {self.smiles}: {e}")


class MolecularSimilaritySearcher:
    """Molecular similarity search engine"""
    
    def __init__(self):
        self.fingerprint_db = {}
        self.smiles_to_id = {}
        self.logger = logging.getLogger(f"{__name__}.MolecularSimilaritySearcher")
    
    def add_molecule(self, item_id: str, smiles: str, fingerprint: np.ndarray):
        """Add molecule to similarity search index"""
        try:
            self.fingerprint_db[item_id] = fingerprint
            self.smiles_to_id[smiles] = item_id
        except Exception as e:
            self.logger.error(f"Error adding molecule to similarity index: {e}")
    
    def find_similar(self, query_smiles: str, threshold: float = 0.7, max_results: int = 100) -> List[Tuple[str, float]]:
        """Find molecules similar to query SMILES"""
        if not RDKIT_AVAILABLE:
            return []
        
        try:
            # Calculate query fingerprint
            query_mol = Chem.MolFromSmiles(query_smiles)
            if query_mol is None:
                return []
            
            query_fp = FingerprintMols.FingerprintMol(query_mol)
            query_array = np.array(query_fp)
            
            similarities = []
            
            for item_id, fp_array in self.fingerprint_db.items():
                try:
                    # Calculate Tanimoto similarity
                    similarity = self._tanimoto_similarity(query_array, fp_array)
                    
                    if similarity >= threshold:
                        similarities.append((item_id, similarity))
                        
                except Exception as e:
                    self.logger.debug(f"Error calculating similarity for {item_id}: {e}")
            
            # Sort by similarity and limit results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:max_results]
            
        except Exception as e:
            self.logger.error(f"Error in similarity search: {e}")
            return []
    
    def _tanimoto_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Calculate Tanimoto similarity between fingerprints"""
        try:
            intersection = np.sum(fp1 & fp2)
            union = np.sum(fp1 | fp2)
            
            if union == 0:
                return 0.0
            
            return intersection / union
            
        except Exception:
            return 0.0


class ChemicalRuleEngine:
    """Chemical rule knowledge management"""
    
    def __init__(self):
        self.rules = self._initialize_chemical_rules()
        self.rule_applications = defaultdict(list)
        self.logger = logging.getLogger(f"{__name__}.ChemicalRuleEngine")
    
    def _initialize_chemical_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize chemical rules database"""
        return {
            "drug_likeness": {
                "lipinski_rule": {
                    "description": "Lipinski's Rule of Five",
                    "conditions": {
                        "molecular_weight": {"operator": "<=", "value": 500},
                        "logp": {"operator": "<=", "value": 5},
                        "h_bond_donors": {"operator": "<=", "value": 5},
                        "h_bond_acceptors": {"operator": "<=", "value": 10}
                    },
                    "confidence": 0.8,
                    "source": "Lipinski et al. (1997)"
                },
                "veber_rule": {
                    "description": "Veber's rule for oral bioavailability",
                    "conditions": {
                        "rotatable_bonds": {"operator": "<=", "value": 10},
                        "polar_surface_area": {"operator": "<=", "value": 140}
                    },
                    "confidence": 0.75,
                    "source": "Veber et al. (2002)"
                }
            },
            
            "toxicity": {
                "ames_mutagenicity": {
                    "description": "Ames mutagenicity structural alerts",
                    "structural_patterns": [
                        "aromatic_amine",
                        "nitro_group",
                        "epoxide",
                        "alkyl_halide"
                    ],
                    "confidence": 0.7,
                    "source": "Kazius et al. (2005)"
                },
                "hepatotoxicity": {
                    "description": "Hepatotoxicity structural alerts",
                    "structural_patterns": [
                        "acetaminophen_like",
                        "halogenated_aromatic",
                        "reactive_metabolite_former"
                    ],
                    "confidence": 0.6,
                    "source": "Various studies"
                }
            },
            
            "synthetic_chemistry": {
                "click_chemistry": {
                    "description": "Click chemistry reaction patterns",
                    "reaction_patterns": [
                        "azide_alkyne_cycloaddition",
                        "thiol_ene_reaction",
                        "diels_alder_reaction"
                    ],
                    "efficiency": 0.9,
                    "source": "Sharpless et al."
                },
                "buchwald_hartwig": {
                    "description": "Buchwald-Hartwig amination",
                    "reaction_patterns": [
                        "aryl_halide_amine_coupling"
                    ],
                    "efficiency": 0.85,
                    "source": "Buchwald & Hartwig"
                }
            }
        }
    
    def apply_rule(self, rule_category: str, rule_name: str, molecular_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply chemical rule to molecular data"""
        try:
            if rule_category not in self.rules or rule_name not in self.rules[rule_category]:
                return {"applicable": False, "error": "Rule not found"}
            
            rule = self.rules[rule_category][rule_name]
            result = {
                "rule_category": rule_category,
                "rule_name": rule_name,
                "description": rule["description"],
                "applicable": True,
                "passed": True,
                "violations": [],
                "confidence": rule.get("confidence", 1.0)
            }
            
            # Check conditions if present
            if "conditions" in rule:
                for property_name, condition in rule["conditions"].items():
                    if property_name in molecular_data:
                        value = molecular_data[property_name]
                        operator = condition["operator"]
                        threshold = condition["value"]
                        
                        passed = self._evaluate_condition(value, operator, threshold)
                        
                        if not passed:
                            result["passed"] = False
                            result["violations"].append({
                                "property": property_name,
                                "value": value,
                                "condition": f"{operator} {threshold}"
                            })
            
            # Record rule application
            self.rule_applications[f"{rule_category}.{rule_name}"].append({
                "timestamp": datetime.utcnow(),
                "result": result
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying rule {rule_category}.{rule_name}: {e}")
            return {"applicable": False, "error": str(e)}
    
    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate rule condition"""
        if operator == "<=":
            return value <= threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">":
            return value > threshold
        elif operator == "==":
            return abs(value - threshold) < 1e-6
        elif operator == "!=":
            return abs(value - threshold) >= 1e-6
        return False
    
    def get_applicable_rules(self, molecular_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get all applicable rules for given molecular data"""
        applicable_rules = []
        
        for category, rules in self.rules.items():
            for rule_name in rules:
                result = self.apply_rule(category, rule_name, molecular_data)
                if result.get("applicable", False):
                    applicable_rules.append(result)
        
        return applicable_rules


class MolecularKnowledgeBase(KnowledgeBasePlugin):
    """
    Molecular knowledge base plugin.
    
    Provides specialized storage and retrieval for molecular knowledge,
    including chemical structures, properties, reactions, and bioactivity data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the molecular knowledge base"""
        super().__init__(config)
        
        # Configuration
        self.use_rdkit = self.config.get('use_rdkit', True) and RDKIT_AVAILABLE
        self.enable_similarity_search = self.config.get('enable_similarity_search', True)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        
        # Specialized storage
        self.molecular_storage = {}  # item_id -> MolecularKnowledgeItem
        self.smiles_index = {}  # SMILES -> item_id
        self.property_indices = defaultdict(dict)  # property_name -> {value_range -> [item_ids]}
        self.bioactivity_index = defaultdict(list)  # target -> [item_ids]
        
        # Molecular tools
        self.similarity_searcher = MolecularSimilaritySearcher() if self.enable_similarity_search else None
        self.rule_engine = ChemicalRuleEngine()
        
        # Statistics
        self.molecular_stats = {
            'molecules_stored': 0,
            'bioactivity_records': 0,
            'similarity_searches': 0,
            'rule_applications': 0
        }
        
        self.logger.info(f"🧬 Molecular Knowledge Base initialized (RDKit: {self.use_rdkit})")
    
    def _create_metadata(self) -> KnowledgeBaseMetadata:
        """Create metadata for molecular knowledge base"""
        return KnowledgeBaseMetadata(
            name="MolecularKnowledgeBase",
            version="1.0.0",
            author="Universal AI Core",
            description="Specialized knowledge base for molecular and chemical data",
            supported_knowledge_types=[
                KnowledgeType.FACTUAL,
                KnowledgeType.PROCEDURAL,
                KnowledgeType.PATTERN,
                KnowledgeType.RULE
            ],
            supported_formats=[
                KnowledgeFormat.JSON,
                KnowledgeFormat.GRAPH,
                KnowledgeFormat.VECTOR
            ],
            supported_query_types=[
                QueryType.EXACT_MATCH,
                QueryType.FUZZY_SEARCH,
                QueryType.SIMILARITY_SEARCH,
                QueryType.PATTERN_MATCH
            ],
            storage_backend="molecular_specialized",
            indexing_method="molecular_fingerprint",
            vector_dimension=2048,  # Typical fingerprint size
            capabilities=[
                "molecular_similarity_search",
                "chemical_rule_application",
                "bioactivity_data_management",
                "property_indexing",
                "smiles_validation"
            ]
        )
    
    def connect(self) -> bool:
        """Connect to molecular knowledge base"""
        try:
            # Initialize indices
            self._initialize_property_indices()
            self._is_connected = True
            self.logger.info("✅ Connected to molecular knowledge base")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to molecular knowledge base: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from molecular knowledge base"""
        self._is_connected = False
        self.logger.info("🔌 Disconnected from molecular knowledge base")
    
    def store_knowledge(self, item: KnowledgeItem) -> bool:
        """Store molecular knowledge item"""
        try:
            # Convert to molecular knowledge item if needed
            if isinstance(item, KnowledgeItem) and not isinstance(item, MolecularKnowledgeItem):
                mol_item = self._convert_to_molecular_item(item)
            else:
                mol_item = item
            
            # Validate and process molecular data
            if mol_item.smiles and self.use_rdkit:
                mol_item.calculate_molecular_properties()
                
                # Validate SMILES
                mol = Chem.MolFromSmiles(mol_item.smiles)
                if mol is None:
                    self.logger.warning(f"Invalid SMILES: {mol_item.smiles}")
                    return False
            
            # Store in main storage
            self.molecular_storage[mol_item.id] = mol_item
            
            # Update indices
            self._update_indices(mol_item)
            
            # Add to similarity searcher
            if self.similarity_searcher and mol_item.fingerprint is not None:
                self.similarity_searcher.add_molecule(
                    mol_item.id, mol_item.smiles, mol_item.fingerprint
                )
            
            self.molecular_stats['molecules_stored'] += 1
            self._knowledge_count += 1
            
            self.logger.debug(f"📝 Stored molecular knowledge: {mol_item.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error storing molecular knowledge {item.id}: {e}")
            return False
    
    def retrieve_knowledge(self, item_id: str) -> Optional[MolecularKnowledgeItem]:
        """Retrieve molecular knowledge item by ID"""
        return self.molecular_storage.get(item_id)
    
    def query_knowledge(self, query: str, query_type: QueryType = QueryType.FUZZY_SEARCH,
                       max_results: int = 10, **kwargs) -> QueryResult:
        """Query molecular knowledge base"""
        start_time = time.time()
        
        try:
            if query_type == QueryType.EXACT_MATCH:
                results = self._exact_match_search(query, max_results)
            elif query_type == QueryType.FUZZY_SEARCH:
                results = self._fuzzy_search(query, max_results)
            elif query_type == QueryType.SIMILARITY_SEARCH:
                results = self._similarity_search(query, max_results, **kwargs)
            elif query_type == QueryType.PATTERN_MATCH:
                results = self._pattern_match_search(query, max_results, **kwargs)
            else:
                results = []
            
            query_time = time.time() - start_time
            
            return QueryResult(
                items=results,
                query=query,
                query_type=query_type,
                total_results=len(results),
                retrieved_count=len(results),
                query_time=query_time,
                status=OperationStatus.SUCCESS
            )
            
        except Exception as e:
            query_time = time.time() - start_time
            self.logger.error(f"❌ Query error: {e}")
            
            return QueryResult(
                items=[],
                query=query,
                query_type=query_type,
                total_results=0,
                retrieved_count=0,
                query_time=query_time,
                status=OperationStatus.ERROR,
                error_message=str(e)
            )
    
    def _exact_match_search(self, query: str, max_results: int) -> List[MolecularKnowledgeItem]:
        """Exact match search (SMILES, InChI, ID)"""
        results = []
        
        # Search by SMILES
        if query in self.smiles_index:
            item_id = self.smiles_index[query]
            item = self.molecular_storage.get(item_id)
            if item:
                results.append(item)
        
        # Search by ID
        if query in self.molecular_storage:
            results.append(self.molecular_storage[query])
        
        # Search by InChI
        for item in self.molecular_storage.values():
            if item.inchi == query and item not in results:
                results.append(item)
                if len(results) >= max_results:
                    break
        
        return results[:max_results]
    
    def _fuzzy_search(self, query: str, max_results: int) -> List[MolecularKnowledgeItem]:
        """Fuzzy search in content and metadata"""
        results = []
        query_lower = query.lower()
        
        for item in self.molecular_storage.values():
            score = 0
            
            # Search in content
            if hasattr(item, 'content') and query_lower in str(item.content).lower():
                score += 2
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in item.tags):
                score += 3
            
            # Search in metadata
            if query_lower in str(item.metadata).lower():
                score += 1
            
            # Search in drug targets
            if any(query_lower in target.lower() for target in item.drug_targets):
                score += 2
            
            if score > 0:
                results.append((item, score))
        
        # Sort by score and return items
        results.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in results[:max_results]]
    
    def _similarity_search(self, query: str, max_results: int, **kwargs) -> List[MolecularKnowledgeItem]:
        """Molecular similarity search"""
        if not self.similarity_searcher:
            return []
        
        threshold = kwargs.get('similarity_threshold', self.similarity_threshold)
        
        # Find similar molecules
        similar_ids = self.similarity_searcher.find_similar(query, threshold, max_results)
        
        results = []
        for item_id, similarity in similar_ids:
            item = self.molecular_storage.get(item_id)
            if item:
                # Add similarity score to metadata
                item_copy = MolecularKnowledgeItem(**item.__dict__)
                item_copy.metadata = item_copy.metadata.copy()
                item_copy.metadata['similarity_score'] = similarity
                results.append(item_copy)
        
        self.molecular_stats['similarity_searches'] += 1
        return results
    
    def _pattern_match_search(self, query: str, max_results: int, **kwargs) -> List[MolecularKnowledgeItem]:
        """Pattern-based search (substructure, property ranges)"""
        results = []
        
        # Property range search: "property:min-max" or "property:value"
        if ':' in query:
            prop_name, prop_value = query.split(':', 1)
            
            if '-' in prop_value:
                # Range search
                try:
                    min_val, max_val = map(float, prop_value.split('-'))
                    results = self._property_range_search(prop_name, min_val, max_val, max_results)
                except ValueError:
                    pass
            else:
                # Exact value search
                try:
                    value = float(prop_value)
                    results = self._property_exact_search(prop_name, value, max_results)
                except ValueError:
                    pass
        
        # Substructure search (if RDKit available)
        elif self.use_rdkit and len(query) > 3:  # Assume SMARTS pattern
            results = self._substructure_search(query, max_results)
        
        return results[:max_results]
    
    def _property_range_search(self, prop_name: str, min_val: float, max_val: float, max_results: int) -> List[MolecularKnowledgeItem]:
        """Search molecules by property range"""
        results = []
        
        for item in self.molecular_storage.values():
            prop_value = getattr(item, prop_name, None)
            if prop_value is not None and min_val <= prop_value <= max_val:
                results.append(item)
                if len(results) >= max_results:
                    break
        
        return results
    
    def _property_exact_search(self, prop_name: str, value: float, max_results: int) -> List[MolecularKnowledgeItem]:
        """Search molecules by exact property value"""
        results = []
        tolerance = 0.1  # Allow small tolerance for floating point comparison
        
        for item in self.molecular_storage.values():
            prop_value = getattr(item, prop_name, None)
            if prop_value is not None and abs(prop_value - value) <= tolerance:
                results.append(item)
                if len(results) >= max_results:
                    break
        
        return results
    
    def _substructure_search(self, smarts_pattern: str, max_results: int) -> List[MolecularKnowledgeItem]:
        """Substructure search using SMARTS pattern"""
        if not self.use_rdkit:
            return []
        
        try:
            pattern_mol = Chem.MolFromSmarts(smarts_pattern)
            if pattern_mol is None:
                return []
            
            results = []
            
            for item in self.molecular_storage.values():
                if item.smiles:
                    mol = Chem.MolFromSmiles(item.smiles)
                    if mol and mol.HasSubstructMatch(pattern_mol):
                        results.append(item)
                        if len(results) >= max_results:
                            break
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in substructure search: {e}")
            return []
    
    def update_knowledge(self, item_id: str, updated_item: MolecularKnowledgeItem) -> bool:
        """Update molecular knowledge item"""
        try:
            if item_id not in self.molecular_storage:
                return False
            
            old_item = self.molecular_storage[item_id]
            
            # Update indices
            self._remove_from_indices(old_item)
            
            # Recalculate properties if SMILES changed
            if updated_item.smiles != old_item.smiles and self.use_rdkit:
                updated_item.calculate_molecular_properties()
            
            # Store updated item
            updated_item.updated_at = datetime.utcnow()
            updated_item.version = old_item.version + 1
            self.molecular_storage[item_id] = updated_item
            
            # Update indices
            self._update_indices(updated_item)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error updating molecular knowledge {item_id}: {e}")
            return False
    
    def delete_knowledge(self, item_id: str) -> bool:
        """Delete molecular knowledge item"""
        try:
            if item_id not in self.molecular_storage:
                return False
            
            item = self.molecular_storage[item_id]
            
            # Remove from indices
            self._remove_from_indices(item)
            
            # Remove from main storage
            del self.molecular_storage[item_id]
            self._knowledge_count -= 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error deleting molecular knowledge {item_id}: {e}")
            return False
    
    def _convert_to_molecular_item(self, item: KnowledgeItem) -> MolecularKnowledgeItem:
        """Convert generic knowledge item to molecular knowledge item"""
        mol_item = MolecularKnowledgeItem(**item.__dict__)
        
        # Extract molecular data from content if it's a dictionary
        if isinstance(item.content, dict):
            mol_item.smiles = item.content.get('smiles', '')
            mol_item.inchi = item.content.get('inchi', '')
            mol_item.molecular_weight = item.content.get('molecular_weight', 0.0)
            mol_item.logp = item.content.get('logp', 0.0)
            mol_item.bioactivity_data = item.content.get('bioactivity_data', {})
            mol_item.drug_targets = item.content.get('drug_targets', [])
        
        return mol_item
    
    def _initialize_property_indices(self):
        """Initialize property indices"""
        self.property_indices = defaultdict(dict)
        self.bioactivity_index = defaultdict(list)
        self.smiles_index = {}
    
    def _update_indices(self, item: MolecularKnowledgeItem):
        """Update all indices for the item"""
        # SMILES index
        if item.smiles:
            self.smiles_index[item.smiles] = item.id
        
        # Bioactivity index
        for target in item.drug_targets:
            if item.id not in self.bioactivity_index[target]:
                self.bioactivity_index[target].append(item.id)
    
    def _remove_from_indices(self, item: MolecularKnowledgeItem):
        """Remove item from all indices"""
        # SMILES index
        if item.smiles in self.smiles_index:
            del self.smiles_index[item.smiles]
        
        # Bioactivity index
        for target in item.drug_targets:
            if item.id in self.bioactivity_index[target]:
                self.bioactivity_index[target].remove(item.id)
    
    def apply_chemical_rule(self, item_id: str, rule_category: str, rule_name: str) -> Dict[str, Any]:
        """Apply chemical rule to molecular knowledge item"""
        item = self.molecular_storage.get(item_id)
        if not item:
            return {"error": "Item not found"}
        
        # Prepare molecular data for rule application
        molecular_data = {
            'molecular_weight': item.molecular_weight,
            'logp': item.logp,
            # Add more properties as needed
        }
        
        result = self.rule_engine.apply_rule(rule_category, rule_name, molecular_data)
        self.molecular_stats['rule_applications'] += 1
        
        return result
    
    def get_bioactivity_data(self, target: str) -> List[MolecularKnowledgeItem]:
        """Get all molecules with bioactivity data for a specific target"""
        item_ids = self.bioactivity_index.get(target, [])
        return [self.molecular_storage[item_id] for item_id in item_ids if item_id in self.molecular_storage]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get molecular knowledge base statistics"""
        base_stats = super().get_statistics()
        base_stats.update(self.molecular_stats)
        
        base_stats.update({
            "smiles_indexed": len(self.smiles_index),
            "bioactivity_targets": len(self.bioactivity_index),
            "similarity_index_size": len(self.similarity_searcher.fingerprint_db) if self.similarity_searcher else 0
        })
        
        return base_stats
    
    def health_check(self) -> bool:
        """Perform health check"""
        try:
            # Test basic functionality
            if self.use_rdkit:
                # Test SMILES processing
                test_smiles = "CCO"
                mol = Chem.MolFromSmiles(test_smiles)
                if mol is None:
                    return False
            
            # Test indices
            return len(self.molecular_storage) == len(self.smiles_index)
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False


# Plugin metadata for discovery
__plugin_metadata__ = {
    "name": "MolecularKnowledgeBase",
    "version": "1.0.0",
    "author": "Universal AI Core",
    "description": "Specialized knowledge base for molecular and chemical data",
    "plugin_type": "knowledge_base",
    "entry_point": f"{__name__}:MolecularKnowledgeBase",
    "dependencies": [
        {"name": "rdkit", "optional": True}
    ],
    "capabilities": [
        "molecular_similarity_search",
        "chemical_rule_application",
        "bioactivity_data_management",
        "property_indexing",
        "smiles_validation"
    ],
    "hooks": []
}


if __name__ == "__main__":
    # Test the molecular knowledge base
    print("🧬 MOLECULAR KNOWLEDGE BASE TEST")
    print("=" * 50)
    
    # Initialize knowledge base
    config = {
        'use_rdkit': RDKIT_AVAILABLE,
        'enable_similarity_search': True,
        'similarity_threshold': 0.7
    }
    
    mol_kb = MolecularKnowledgeBase(config)
    
    # Connect
    connected = mol_kb.connect()
    print(f"📡 Connected: {'✅' if connected else '❌'}")
    
    # Test molecules
    test_molecules = [
        {
            'id': 'mol1',
            'smiles': 'CCO',  # Ethanol
            'content': {'name': 'Ethanol', 'use': 'Solvent'},
            'tags': ['alcohol', 'solvent'],
            'drug_targets': ['CNS']
        },
        {
            'id': 'mol2', 
            'smiles': 'CC(C)O',  # Isopropanol
            'content': {'name': 'Isopropanol', 'use': 'Antiseptic'},
            'tags': ['alcohol', 'antiseptic'],
            'drug_targets': []
        },
        {
            'id': 'mol3',
            'smiles': 'c1ccccc1',  # Benzene
            'content': {'name': 'Benzene', 'use': 'Industrial'},
            'tags': ['aromatic', 'industrial'],
            'drug_targets': []
        }
    ]
    
    # Store molecules
    print(f"\n📝 Storing {len(test_molecules)} molecules...")
    for mol_data in test_molecules:
        item = MolecularKnowledgeItem(
            id=mol_data['id'],
            content=mol_data['content'],
            knowledge_type=KnowledgeType.FACTUAL,
            format=KnowledgeFormat.JSON,
            tags=mol_data['tags'],
            smiles=mol_data['smiles'],
            drug_targets=mol_data['drug_targets']
        )
        
        success = mol_kb.store_knowledge(item)
        print(f"  {mol_data['id']}: {'✅' if success else '❌'}")
    
    # Test retrieval
    print(f"\n🔍 Testing retrieval...")
    retrieved = mol_kb.retrieve_knowledge('mol1')
    print(f"Retrieved mol1: {'✅' if retrieved else '❌'}")
    
    if retrieved:
        print(f"  SMILES: {retrieved.smiles}")
        print(f"  MW: {retrieved.molecular_weight:.2f}")
        print(f"  LogP: {retrieved.logp:.2f}")
    
    # Test queries
    print(f"\n🔎 Testing queries...")
    
    # Exact match
    result = mol_kb.query_knowledge('CCO', QueryType.EXACT_MATCH)
    print(f"Exact match for 'CCO': {len(result.items)} results")
    
    # Fuzzy search
    result = mol_kb.query_knowledge('alcohol', QueryType.FUZZY_SEARCH)
    print(f"Fuzzy search for 'alcohol': {len(result.items)} results")
    
    # Similarity search (if RDKit available)
    if RDKIT_AVAILABLE:
        result = mol_kb.query_knowledge('CCO', QueryType.SIMILARITY_SEARCH)
        print(f"Similarity search for 'CCO': {len(result.items)} results")
    
    # Test chemical rule application
    if mol_kb.molecular_storage:
        print(f"\n⚖️ Testing chemical rules...")
        item_id = list(mol_kb.molecular_storage.keys())[0]
        rule_result = mol_kb.apply_chemical_rule(item_id, 'drug_likeness', 'lipinski_rule')
        print(f"Lipinski rule application: {'✅' if rule_result.get('applicable') else '❌'}")
        if rule_result.get('applicable'):
            print(f"  Passed: {rule_result.get('passed', False)}")
            print(f"  Violations: {len(rule_result.get('violations', []))}")
    
    # Test health check
    health = mol_kb.health_check()
    print(f"\n🏥 Health check: {'✅' if health else '❌'}")
    
    # Show statistics
    stats = mol_kb.get_statistics()
    print(f"\n📊 Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Disconnect
    mol_kb.disconnect()
    print("\n✅ Molecular knowledge base test completed!")