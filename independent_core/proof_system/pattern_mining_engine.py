"""
Pattern Mining Engine - Mines patterns from proof executions and provides recommendations
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import json
import threading
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


@dataclass
class SequentialPattern:
    """A sequential pattern in strategy executions"""
    sequence: List[str]
    support: int
    confidence: float
    length: int
    
    def __post_init__(self):
        # NO FALLBACKS - HARD FAILURES ONLY
        if not self.sequence or not isinstance(self.sequence, list):
            raise ValueError("Sequence must be non-empty list")
        if not all(isinstance(item, str) for item in self.sequence):
            raise TypeError("All sequence items must be strings")
        if self.support < 0:
            raise ValueError("Support must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.length != len(self.sequence):
            self.length = len(self.sequence)


@dataclass
class AssociationRule:
    """An association rule between strategy elements"""
    antecedent: Set[str]
    consequent: Set[str]
    support: float
    confidence: float
    lift: float
    conviction: float
    
    def __post_init__(self):
        # NO FALLBACKS - HARD FAILURES ONLY
        if not self.antecedent or not isinstance(self.antecedent, set):
            raise ValueError("Antecedent must be non-empty set")
        if not self.consequent or not isinstance(self.consequent, set):
            raise ValueError("Consequent must be non-empty set")
        if not all(isinstance(item, str) for item in self.antecedent):
            raise TypeError("All antecedent items must be strings")
        if not all(isinstance(item, str) for item in self.consequent):
            raise TypeError("All consequent items must be strings")
        if self.antecedent & self.consequent:
            raise ValueError("Antecedent and consequent must be disjoint")
        if not 0.0 <= self.support <= 1.0:
            raise ValueError("Support must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.lift < 0:
            raise ValueError("Lift must be non-negative")
        if self.conviction < 0:
            raise ValueError("Conviction must be non-negative")


@dataclass
class ClusterPattern:
    """A cluster pattern in proof data"""
    cluster_id: int
    centroid: Dict[str, float]
    members: List[Dict[str, Any]]
    radius: float
    density: float
    quality_score: float
    
    def __post_init__(self):
        # NO FALLBACKS - HARD FAILURES ONLY
        if self.cluster_id < 0:
            raise ValueError("Cluster ID must be non-negative")
        if not isinstance(self.centroid, dict):
            raise TypeError("Centroid must be dict")
        if not all(isinstance(v, (int, float)) for v in self.centroid.values()):
            raise TypeError("Centroid values must be numeric")
        if not isinstance(self.members, list):
            raise TypeError("Members must be list")
        if self.radius < 0:
            raise ValueError("Radius must be non-negative")
        if self.density < 0:
            raise ValueError("Density must be non-negative")
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("Quality score must be between 0.0 and 1.0")


@dataclass
class PatternRecommendation:
    """A pattern-based recommendation"""
    pattern_type: str
    pattern_data: Dict[str, Any]
    recommendation: str
    confidence: float
    reasoning: str
    applicable_contexts: List[str]
    
    def __post_init__(self):
        # NO FALLBACKS - HARD FAILURES ONLY
        if not self.pattern_type or not isinstance(self.pattern_type, str):
            raise ValueError("Pattern type must be non-empty string")
        if not isinstance(self.pattern_data, dict):
            raise TypeError("Pattern data must be dict")
        if not self.recommendation or not isinstance(self.recommendation, str):
            raise ValueError("Recommendation must be non-empty string")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if not self.reasoning or not isinstance(self.reasoning, str):
            raise ValueError("Reasoning must be non-empty string")
        if not isinstance(self.applicable_contexts, list):
            raise TypeError("Applicable contexts must be list")


class SequentialPatternMiner:
    """Mines sequential patterns from strategy execution sequences"""
    
    def __init__(self, min_support: int = 2, min_confidence: float = 0.5):
        # NO FALLBACKS - HARD FAILURES ONLY
        if min_support < 1:
            raise ValueError("Min support must be at least 1")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("Min confidence must be between 0.0 and 1.0")
        
        self.min_support = min_support
        self.min_confidence = min_confidence
    
    def mine_patterns(self, sequences: List[List[str]]) -> List[SequentialPattern]:
        """Mine sequential patterns from execution sequences"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not sequences or not isinstance(sequences, list):
            raise ValueError("Sequences must be non-empty list")
        if not all(isinstance(seq, list) for seq in sequences):
            raise TypeError("All sequences must be lists")
        if not all(all(isinstance(item, str) for item in seq) for seq in sequences):
            raise TypeError("All sequence items must be strings")
        
        patterns = []
        
        # Find frequent 1-patterns
        item_counts = Counter()
        for sequence in sequences:
            for item in sequence:
                item_counts[item] += 1
        
        frequent_items = {item for item, count in item_counts.items() 
                         if count >= self.min_support}
        
        if not frequent_items:
            return patterns
        
        # Add 1-patterns
        for item in frequent_items:
            support = item_counts[item]
            confidence = support / len(sequences)
            if confidence >= self.min_confidence:
                patterns.append(SequentialPattern([item], support, confidence, 1))
        
        # Find longer patterns using recursive approach
        current_patterns = [[item] for item in frequent_items]
        length = 1
        
        while current_patterns and length < 10:  # Limit max length
            length += 1
            next_patterns = []
            
            # Generate candidates
            candidates = self._generate_candidates(current_patterns)
            
            # Count support for each candidate
            for candidate in candidates:
                support = self._count_subsequence_support(candidate, sequences)
                if support >= self.min_support:
                    confidence = support / len(sequences)
                    if confidence >= self.min_confidence:
                        patterns.append(SequentialPattern(candidate, support, confidence, length))
                        next_patterns.append(candidate)
            
            current_patterns = next_patterns
        
        # Sort by support (descending)
        patterns.sort(key=lambda p: p.support, reverse=True)
        return patterns
    
    def _generate_candidates(self, frequent_patterns: List[List[str]]) -> List[List[str]]:
        """Generate candidate patterns of length k+1 from frequent patterns of length k"""
        candidates = []
        
        for i, pattern1 in enumerate(frequent_patterns):
            for j, pattern2 in enumerate(frequent_patterns):
                if i <= j:
                    continue
                
                # Try extending pattern1 with last item of pattern2
                if pattern1[1:] == pattern2[:-1]:  # Overlap condition
                    candidate = pattern1 + [pattern2[-1]]
                    if candidate not in candidates:
                        candidates.append(candidate)
                
                # Try extending pattern2 with last item of pattern1
                if pattern2[1:] == pattern1[:-1]:  # Overlap condition
                    candidate = pattern2 + [pattern1[-1]]
                    if candidate not in candidates:
                        candidates.append(candidate)
        
        return candidates
    
    def _count_subsequence_support(self, subsequence: List[str], sequences: List[List[str]]) -> int:
        """Count how many sequences contain the given subsequence"""
        support = 0
        
        for sequence in sequences:
            if self._is_subsequence(subsequence, sequence):
                support += 1
        
        return support
    
    def _is_subsequence(self, subseq: List[str], seq: List[str]) -> bool:
        """Check if subseq is a subsequence of seq"""
        if len(subseq) > len(seq):
            return False
        
        i, j = 0, 0
        while i < len(subseq) and j < len(seq):
            if subseq[i] == seq[j]:
                i += 1
            j += 1
        
        return i == len(subseq)


class AssociationRuleMiner:
    """Mines association rules using Apriori algorithm"""
    
    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.5):
        # NO FALLBACKS - HARD FAILURES ONLY
        if not 0.0 <= min_support <= 1.0:
            raise ValueError("Min support must be between 0.0 and 1.0")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("Min confidence must be between 0.0 and 1.0")
        
        self.min_support = min_support
        self.min_confidence = min_confidence
    
    def mine_rules(self, transactions: List[Set[str]]) -> List[AssociationRule]:
        """Mine association rules from transactions"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not transactions or not isinstance(transactions, list):
            raise ValueError("Transactions must be non-empty list")
        if not all(isinstance(transaction, set) for transaction in transactions):
            raise TypeError("All transactions must be sets")
        if not all(all(isinstance(item, str) for item in transaction) for transaction in transactions):
            raise TypeError("All transaction items must be strings")
        
        # Find frequent itemsets
        frequent_itemsets = self._find_frequent_itemsets(transactions)
        
        # Generate association rules
        rules = []
        for itemset, support in frequent_itemsets.items():
            if len(itemset) < 2:
                continue
            
            # Generate all possible rules for this itemset
            for r in range(1, len(itemset)):
                for antecedent in combinations(itemset, r):
                    antecedent_set = set(antecedent)
                    consequent_set = itemset - antecedent_set
                    
                    # Calculate metrics
                    rule_support = support
                    antecedent_support = frequent_itemsets.get(frozenset(antecedent_set), 0)
                    consequent_support = frequent_itemsets.get(frozenset(consequent_set), 0)
                    
                    if antecedent_support > 0:
                        confidence = rule_support / antecedent_support
                        
                        if confidence >= self.min_confidence:
                            # Calculate lift
                            lift = (rule_support * len(transactions)) / (antecedent_support * consequent_support) if consequent_support > 0 else 0
                            
                            # Calculate conviction
                            if confidence < 1.0:
                                conviction = (1 - consequent_support / len(transactions)) / (1 - confidence)
                            else:
                                conviction = float('inf')
                            
                            rules.append(AssociationRule(
                                antecedent=antecedent_set,
                                consequent=consequent_set,
                                support=rule_support / len(transactions),
                                confidence=confidence,
                                lift=lift,
                                conviction=conviction
                            ))
        
        # Sort by confidence (descending)
        rules.sort(key=lambda r: r.confidence, reverse=True)
        return rules
    
    def _find_frequent_itemsets(self, transactions: List[Set[str]]) -> Dict[frozenset, int]:
        """Find frequent itemsets using Apriori algorithm"""
        frequent_itemsets = {}
        
        # Find frequent 1-itemsets
        item_counts = Counter()
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        
        min_support_count = int(self.min_support * len(transactions))
        
        current_frequent = {}
        for item, count in item_counts.items():
            if count >= min_support_count:
                itemset = frozenset([item])
                current_frequent[itemset] = count
                frequent_itemsets[itemset] = count
        
        # Find frequent k-itemsets
        k = 2
        while current_frequent and k <= 10:  # Limit max size
            candidates = self._generate_candidates_apriori(list(current_frequent.keys()))
            next_frequent = {}
            
            for candidate in candidates:
                count = sum(1 for transaction in transactions if candidate.issubset(transaction))
                if count >= min_support_count:
                    next_frequent[candidate] = count
                    frequent_itemsets[candidate] = count
            
            current_frequent = next_frequent
            k += 1
        
        return frequent_itemsets
    
    def _generate_candidates_apriori(self, frequent_itemsets: List[frozenset]) -> List[frozenset]:
        """Generate candidate itemsets for Apriori"""
        candidates = []
        
        for i, itemset1 in enumerate(frequent_itemsets):
            for j, itemset2 in enumerate(frequent_itemsets):
                if i >= j:
                    continue
                
                # Check if they can be joined
                union = itemset1 | itemset2
                if len(union) == len(itemset1) + 1:  # They differ by exactly one item
                    # Check if all (k-1)-subsets are frequent
                    if self._has_infrequent_subset(union, frequent_itemsets):
                        continue
                    
                    candidates.append(union)
        
        return candidates
    
    def _has_infrequent_subset(self, itemset: frozenset, frequent_itemsets: List[frozenset]) -> bool:
        """Check if itemset has any infrequent (k-1)-subset"""
        for item in itemset:
            subset = itemset - {item}
            if subset not in frequent_itemsets:
                return True
        return False


class ClusteringPatternMiner:
    """Mines clustering patterns from proof execution data"""
    
    def __init__(self, n_clusters: int = 5, max_iterations: int = 100):
        # NO FALLBACKS - HARD FAILURES ONLY
        if n_clusters < 1:
            raise ValueError("Number of clusters must be positive")
        if max_iterations < 1:
            raise ValueError("Max iterations must be positive")
        
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
    
    def mine_patterns(self, execution_data: List[Dict[str, Any]]) -> List[ClusterPattern]:
        """Mine clustering patterns from execution data"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not execution_data or not isinstance(execution_data, list):
            raise ValueError("Execution data must be non-empty list")
        if not all(isinstance(item, dict) for item in execution_data):
            raise TypeError("All execution data items must be dicts")
        
        # Extract features
        features = self._extract_features(execution_data)
        if not features:
            raise ValueError("No features could be extracted from execution data")
        
        # Perform k-means clustering
        clusters = self._kmeans_clustering(features)
        
        # Create cluster patterns
        patterns = []
        for cluster_id, cluster_data in clusters.items():
            if not cluster_data['members']:
                continue
            
            # Calculate cluster quality metrics
            radius = self._calculate_cluster_radius(cluster_data['centroid'], cluster_data['member_features'])
            density = len(cluster_data['members']) / (radius + 1e-8)
            quality_score = self._calculate_cluster_quality(cluster_data)
            
            pattern = ClusterPattern(
                cluster_id=cluster_id,
                centroid=cluster_data['centroid'],
                members=cluster_data['members'],
                radius=radius,
                density=density,
                quality_score=quality_score
            )
            patterns.append(pattern)
        
        # Sort by quality score
        patterns.sort(key=lambda p: p.quality_score, reverse=True)
        return patterns
    
    def _extract_features(self, execution_data: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Extract numerical features from execution data"""
        features = []
        
        for item in execution_data:
            feature_dict = {}
            
            # Extract numerical features
            if 'confidence' in item and isinstance(item['confidence'], (int, float)):
                feature_dict['confidence'] = float(item['confidence'])
            
            if 'execution_time' in item and isinstance(item['execution_time'], (int, float)):
                feature_dict['execution_time'] = float(item['execution_time'])
            
            if 'success' in item and isinstance(item['success'], bool):
                feature_dict['success'] = 1.0 if item['success'] else 0.0
            
            # Extract context features
            if 'context' in item and isinstance(item['context'], dict):
                for key, value in item['context'].items():
                    if isinstance(value, (int, float)):
                        feature_dict[f'context_{key}'] = float(value)
            
            # Extract proof data features
            if 'proof_data' in item and isinstance(item['proof_data'], dict):
                proof_data = item['proof_data']
                
                # Count various elements
                if 'steps' in proof_data and isinstance(proof_data['steps'], list):
                    feature_dict['proof_steps_count'] = float(len(proof_data['steps']))
                
                if 'constraints' in proof_data and isinstance(proof_data['constraints'], list):
                    feature_dict['constraints_count'] = float(len(proof_data['constraints']))
            
            if feature_dict:  # Only add if we extracted some features
                features.append(feature_dict)
        
        return features
    
    def _kmeans_clustering(self, features: List[Dict[str, float]]) -> Dict[int, Dict[str, Any]]:
        """Perform k-means clustering on features"""
        if not features:
            return {}
        
        # Get all feature names
        all_features = set()
        for feature_dict in features:
            all_features.update(feature_dict.keys())
        
        feature_names = sorted(all_features)
        
        # Convert to feature vectors
        feature_vectors = []
        for feature_dict in features:
            vector = [feature_dict.get(name, 0.0) for name in feature_names]
            feature_vectors.append(vector)
        
        feature_vectors = np.array(feature_vectors)
        
        # Normalize features
        means = np.mean(feature_vectors, axis=0)
        stds = np.std(feature_vectors, axis=0)
        stds[stds == 0] = 1.0  # Avoid division by zero
        normalized_features = (feature_vectors - means) / stds
        
        # Initialize centroids randomly
        n_samples, n_features = normalized_features.shape
        effective_clusters = min(self.n_clusters, n_samples)
        
        centroids = np.random.randn(effective_clusters, n_features)
        
        # K-means iterations
        for iteration in range(self.max_iterations):
            # Assign points to clusters
            distances = np.sqrt(((normalized_features[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
            assignments = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(effective_clusters):
                mask = assignments == k
                if np.any(mask):
                    new_centroids[k] = np.mean(normalized_features[mask], axis=0)
                else:
                    new_centroids[k] = centroids[k]  # Keep old centroid if no points assigned
            
            # Check convergence
            if np.allclose(centroids, new_centroids, rtol=1e-4):
                break
            
            centroids = new_centroids
        
        # Create cluster results
        clusters = {}
        for k in range(effective_clusters):
            mask = assignments == k
            cluster_members = [features[i] for i in range(len(features)) if mask[i]]
            
            # Convert centroid back to feature dict
            centroid_dict = {}
            for i, feature_name in enumerate(feature_names):
                centroid_dict[feature_name] = float(centroids[k][i] * stds[i] + means[i])
            
            clusters[k] = {
                'centroid': centroid_dict,
                'members': cluster_members,
                'member_features': normalized_features[mask]
            }
        
        return clusters
    
    def _calculate_cluster_radius(self, centroid: Dict[str, float], member_features: np.ndarray) -> float:
        """Calculate cluster radius (max distance from centroid)"""
        if len(member_features) == 0:
            return 0.0
        
        # Calculate distances from centroid
        distances = np.sqrt(np.sum(member_features ** 2, axis=1))
        return float(np.max(distances))
    
    def _calculate_cluster_quality(self, cluster_data: Dict[str, Any]) -> float:
        """Calculate cluster quality score"""
        members = cluster_data['members']
        if len(members) < 2:
            return 0.0
        
        # Quality based on homogeneity of success rates
        success_rates = [m.get('success', 0.0) for m in members]
        success_std = np.std(success_rates)
        success_homogeneity = 1.0 / (1.0 + success_std)
        
        # Quality based on confidence consistency
        confidences = [m.get('confidence', 0.5) for m in members]
        confidence_std = np.std(confidences)
        confidence_consistency = 1.0 / (1.0 + confidence_std)
        
        # Size factor (larger clusters are generally better)
        size_factor = min(1.0, len(members) / 10.0)
        
        # Combined quality score
        quality = 0.4 * success_homogeneity + 0.4 * confidence_consistency + 0.2 * size_factor
        
        return min(1.0, quality)


class PatternMiningEngine:
    """Main pattern mining engine that coordinates all miners"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        
        # Initialize miners
        sequential_config = config.get('sequential', {})
        self.sequential_miner = SequentialPatternMiner(
            min_support=sequential_config.get('min_support', 2),
            min_confidence=sequential_config.get('min_confidence', 0.5)
        )
        
        association_config = config.get('association', {})
        self.association_miner = AssociationRuleMiner(
            min_support=association_config.get('min_support', 0.1),
            min_confidence=association_config.get('min_confidence', 0.5)
        )
        
        clustering_config = config.get('clustering', {})
        self.clustering_miner = ClusteringPatternMiner(
            n_clusters=clustering_config.get('n_clusters', 5),
            max_iterations=clustering_config.get('max_iterations', 100)
        )
        
        # Pattern storage
        self.patterns_lock = threading.RLock()
        self.sequential_patterns: List[SequentialPattern] = []
        self.association_rules: List[AssociationRule] = []
        self.cluster_patterns: List[ClusterPattern] = []
        
        # Pattern validation
        self.pattern_validators: Dict[str, Callable] = {}
        
        # Recommendation engine
        self.recommendation_threshold = config.get('recommendation_threshold', 0.7)
    
    def mine_all_patterns(self, execution_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mine all types of patterns from execution data"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not execution_data or not isinstance(execution_data, list):
            raise ValueError("Execution data must be non-empty list")
        if not all(isinstance(item, dict) for item in execution_data):
            raise TypeError("All execution data items must be dicts")
        
        results = {}
        
        with self.patterns_lock:
            # Mine sequential patterns
            try:
                sequences = self._extract_strategy_sequences(execution_data)
                if sequences:
                    self.sequential_patterns = self.sequential_miner.mine_patterns(sequences)
                    results['sequential_patterns'] = len(self.sequential_patterns)
                else:
                    self.sequential_patterns = []
                    results['sequential_patterns'] = 0
            except Exception as e:
                logger.error(f"Sequential pattern mining failed: {e}")
                raise RuntimeError(f"Sequential pattern mining failed: {e}")
            
            # Mine association rules
            try:
                transactions = self._extract_strategy_transactions(execution_data)
                if transactions:
                    self.association_rules = self.association_miner.mine_rules(transactions)
                    results['association_rules'] = len(self.association_rules)
                else:
                    self.association_rules = []
                    results['association_rules'] = 0
            except Exception as e:
                logger.error(f"Association rule mining failed: {e}")
                raise RuntimeError(f"Association rule mining failed: {e}")
            
            # Mine cluster patterns
            try:
                self.cluster_patterns = self.clustering_miner.mine_patterns(execution_data)
                results['cluster_patterns'] = len(self.cluster_patterns)
            except Exception as e:
                logger.error(f"Cluster pattern mining failed: {e}")
                raise RuntimeError(f"Cluster pattern mining failed: {e}")
        
        results['total_execution_data'] = len(execution_data)
        results['mining_timestamp'] = time.time()
        
        return results
    
    def get_pattern_recommendations(self, context: Dict[str, Any]) -> List[PatternRecommendation]:
        """Get pattern-based recommendations for given context"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not isinstance(context, dict):
            raise TypeError("Context must be dict")
        
        recommendations = []
        
        with self.patterns_lock:
            # Recommendations from sequential patterns
            recommendations.extend(self._get_sequential_recommendations(context))
            
            # Recommendations from association rules
            recommendations.extend(self._get_association_recommendations(context))
            
            # Recommendations from cluster patterns
            recommendations.extend(self._get_cluster_recommendations(context))
        
        # Sort by confidence
        recommendations.sort(key=lambda r: r.confidence, reverse=True)
        
        # Filter by threshold
        return [r for r in recommendations if r.confidence >= self.recommendation_threshold]
    
    def validate_patterns(self) -> Dict[str, Any]:
        """Validate discovered patterns"""
        validation_results = {}
        
        with self.patterns_lock:
            # Validate sequential patterns
            sequential_valid = 0
            for pattern in self.sequential_patterns:
                if self._validate_sequential_pattern(pattern):
                    sequential_valid += 1
            
            validation_results['sequential'] = {
                'total': len(self.sequential_patterns),
                'valid': sequential_valid,
                'validity_rate': sequential_valid / max(1, len(self.sequential_patterns))
            }
            
            # Validate association rules
            association_valid = 0
            for rule in self.association_rules:
                if self._validate_association_rule(rule):
                    association_valid += 1
            
            validation_results['association'] = {
                'total': len(self.association_rules),
                'valid': association_valid,
                'validity_rate': association_valid / max(1, len(self.association_rules))
            }
            
            # Validate cluster patterns
            cluster_valid = 0
            for pattern in self.cluster_patterns:
                if self._validate_cluster_pattern(pattern):
                    cluster_valid += 1
            
            validation_results['clustering'] = {
                'total': len(self.cluster_patterns),
                'valid': cluster_valid,
                'validity_rate': cluster_valid / max(1, len(self.cluster_patterns))
            }
        
        return validation_results
    
    def detect_pattern_conflicts(self) -> List[Dict[str, Any]]:
        """Detect conflicts between different patterns"""
        conflicts = []
        
        with self.patterns_lock:
            # Check for conflicts between association rules
            for i, rule1 in enumerate(self.association_rules):
                for j, rule2 in enumerate(self.association_rules[i+1:], i+1):
                    if self._rules_conflict(rule1, rule2):
                        conflicts.append({
                            'type': 'association_rule_conflict',
                            'rule1_index': i,
                            'rule2_index': j,
                            'description': f"Rules {i} and {j} have conflicting recommendations"
                        })
            
            # Check for conflicts between sequential patterns and association rules
            for seq_idx, seq_pattern in enumerate(self.sequential_patterns):
                for assoc_idx, assoc_rule in enumerate(self.association_rules):
                    if self._sequential_association_conflict(seq_pattern, assoc_rule):
                        conflicts.append({
                            'type': 'sequential_association_conflict',
                            'sequential_index': seq_idx,
                            'association_index': assoc_idx,
                            'description': f"Sequential pattern {seq_idx} conflicts with association rule {assoc_idx}"
                        })
        
        return conflicts
    
    def export_patterns(self) -> Dict[str, Any]:
        """Export all discovered patterns"""
        with self.patterns_lock:
            return {
                'sequential_patterns': [
                    {
                        'sequence': pattern.sequence,
                        'support': pattern.support,
                        'confidence': pattern.confidence,
                        'length': pattern.length
                    }
                    for pattern in self.sequential_patterns
                ],
                'association_rules': [
                    {
                        'antecedent': list(rule.antecedent),
                        'consequent': list(rule.consequent),
                        'support': rule.support,
                        'confidence': rule.confidence,
                        'lift': rule.lift,
                        'conviction': rule.conviction
                    }
                    for rule in self.association_rules
                ],
                'cluster_patterns': [
                    {
                        'cluster_id': pattern.cluster_id,
                        'centroid': pattern.centroid,
                        'member_count': len(pattern.members),
                        'radius': pattern.radius,
                        'density': pattern.density,
                        'quality_score': pattern.quality_score
                    }
                    for pattern in self.cluster_patterns
                ],
                'export_timestamp': time.time()
            }
    
    def import_patterns(self, pattern_data: Dict[str, Any]) -> None:
        """Import patterns from external data"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not isinstance(pattern_data, dict):
            raise TypeError("Pattern data must be dict")
        
        with self.patterns_lock:
            # Import sequential patterns
            if 'sequential_patterns' in pattern_data:
                self.sequential_patterns = []
                for pattern_dict in pattern_data['sequential_patterns']:
                    pattern = SequentialPattern(
                        sequence=pattern_dict['sequence'],
                        support=pattern_dict['support'],
                        confidence=pattern_dict['confidence'],
                        length=pattern_dict['length']
                    )
                    self.sequential_patterns.append(pattern)
            
            # Import association rules
            if 'association_rules' in pattern_data:
                self.association_rules = []
                for rule_dict in pattern_data['association_rules']:
                    rule = AssociationRule(
                        antecedent=set(rule_dict['antecedent']),
                        consequent=set(rule_dict['consequent']),
                        support=rule_dict['support'],
                        confidence=rule_dict['confidence'],
                        lift=rule_dict['lift'],
                        conviction=rule_dict['conviction']
                    )
                    self.association_rules.append(rule)
            
            # Import cluster patterns
            if 'cluster_patterns' in pattern_data:
                self.cluster_patterns = []
                for cluster_dict in pattern_data['cluster_patterns']:
                    pattern = ClusterPattern(
                        cluster_id=cluster_dict['cluster_id'],
                        centroid=cluster_dict['centroid'],
                        members=[],  # Members not included in export for size reasons
                        radius=cluster_dict['radius'],
                        density=cluster_dict['density'],
                        quality_score=cluster_dict['quality_score']
                    )
                    self.cluster_patterns.append(pattern)
    
    # Helper methods
    
    def _extract_strategy_sequences(self, execution_data: List[Dict[str, Any]]) -> List[List[str]]:
        """Extract strategy execution sequences"""
        sequences = []
        
        # Group by session or time windows
        time_sorted_data = sorted(execution_data, key=lambda x: x.get('start_time', 0))
        
        current_sequence = []
        last_time = 0
        
        for item in time_sorted_data:
            start_time = item.get('start_time', 0)
            strategy_name = item.get('strategy_name', 'unknown')
            
            # Start new sequence if time gap is too large (1 hour)
            if start_time - last_time > 3600 and current_sequence:
                sequences.append(current_sequence)
                current_sequence = []
            
            current_sequence.append(strategy_name)
            last_time = start_time
        
        if current_sequence:
            sequences.append(current_sequence)
        
        return sequences
    
    def _extract_strategy_transactions(self, execution_data: List[Dict[str, Any]]) -> List[Set[str]]:
        """Extract strategy transactions for association rule mining"""
        transactions = []
        
        # Group executions by context similarity
        context_groups = defaultdict(list)
        
        for item in execution_data:
            # Create context signature
            context = item.get('context', {})
            context_keys = tuple(sorted(context.keys()))
            context_groups[context_keys].append(item)
        
        # Create transactions from each context group
        for group in context_groups.values():
            if len(group) >= 2:  # Need at least 2 items for meaningful transaction
                strategy_set = {item.get('strategy_name', 'unknown') for item in group}
                transactions.append(strategy_set)
        
        return transactions
    
    def _get_sequential_recommendations(self, context: Dict[str, Any]) -> List[PatternRecommendation]:
        """Get recommendations from sequential patterns"""
        recommendations = []
        
        current_strategy = context.get('current_strategy')
        if not current_strategy:
            return recommendations
        
        for pattern in self.sequential_patterns:
            if current_strategy in pattern.sequence:
                # Find position of current strategy
                positions = [i for i, s in enumerate(pattern.sequence) if s == current_strategy]
                
                for pos in positions:
                    if pos < len(pattern.sequence) - 1:
                        next_strategy = pattern.sequence[pos + 1]
                        
                        recommendation = PatternRecommendation(
                            pattern_type='sequential',
                            pattern_data={'sequence': pattern.sequence, 'position': pos},
                            recommendation=f"Consider using strategy '{next_strategy}' next",
                            confidence=pattern.confidence * 0.8,  # Slightly reduce confidence
                            reasoning=f"Sequential pattern shows '{next_strategy}' often follows '{current_strategy}'",
                            applicable_contexts=['similar_execution_context']
                        )
                        recommendations.append(recommendation)
        
        return recommendations
    
    def _get_association_recommendations(self, context: Dict[str, Any]) -> List[PatternRecommendation]:
        """Get recommendations from association rules"""
        recommendations = []
        
        current_strategies = set(context.get('used_strategies', []))
        if not current_strategies:
            return recommendations
        
        for rule in self.association_rules:
            if rule.antecedent.issubset(current_strategies):
                for consequent_strategy in rule.consequent:
                    if consequent_strategy not in current_strategies:
                        recommendation = PatternRecommendation(
                            pattern_type='association',
                            pattern_data={
                                'antecedent': list(rule.antecedent),
                                'consequent': list(rule.consequent),
                                'lift': rule.lift
                            },
                            recommendation=f"Consider adding strategy '{consequent_strategy}'",
                            confidence=rule.confidence,
                            reasoning=f"Association rule shows '{consequent_strategy}' is often used with {list(rule.antecedent)}",
                            applicable_contexts=['similar_strategy_combination']
                        )
                        recommendations.append(recommendation)
        
        return recommendations
    
    def _get_cluster_recommendations(self, context: Dict[str, Any]) -> List[PatternRecommendation]:
        """Get recommendations from cluster patterns"""
        recommendations = []
        
        # Extract features from context
        context_features = {}
        if 'confidence' in context:
            context_features['confidence'] = context['confidence']
        if 'execution_time' in context:
            context_features['execution_time'] = context['execution_time']
        
        if not context_features:
            return recommendations
        
        # Find closest cluster
        best_cluster = None
        min_distance = float('inf')
        
        for cluster in self.cluster_patterns:
            distance = 0
            for feature, value in context_features.items():
                if feature in cluster.centroid:
                    distance += (value - cluster.centroid[feature]) ** 2
            
            distance = np.sqrt(distance)
            if distance < min_distance:
                min_distance = distance
                best_cluster = cluster
        
        if best_cluster and best_cluster.quality_score > 0.5:
            recommendation = PatternRecommendation(
                pattern_type='clustering',
                pattern_data={
                    'cluster_id': best_cluster.cluster_id,
                    'distance': min_distance,
                    'quality_score': best_cluster.quality_score
                },
                recommendation=f"Your execution pattern matches cluster {best_cluster.cluster_id}",
                confidence=best_cluster.quality_score * 0.7,
                reasoning=f"Execution characteristics are similar to cluster with quality score {best_cluster.quality_score:.2f}",
                applicable_contexts=['similar_execution_pattern']
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    def _validate_sequential_pattern(self, pattern: SequentialPattern) -> bool:
        """Validate a sequential pattern"""
        # Basic validation
        if pattern.support < 1 or pattern.confidence < 0 or pattern.confidence > 1:
            return False
        
        # Pattern-specific validation
        if len(pattern.sequence) != pattern.length:
            return False
        
        # Custom validators
        for validator in self.pattern_validators.get('sequential', []):
            if not validator(pattern):
                return False
        
        return True
    
    def _validate_association_rule(self, rule: AssociationRule) -> bool:
        """Validate an association rule"""
        # Basic validation
        if not (0 <= rule.support <= 1) or not (0 <= rule.confidence <= 1):
            return False
        
        if rule.lift < 0 or rule.conviction < 0:
            return False
        
        # Custom validators
        for validator in self.pattern_validators.get('association', []):
            if not validator(rule):
                return False
        
        return True
    
    def _validate_cluster_pattern(self, pattern: ClusterPattern) -> bool:
        """Validate a cluster pattern"""
        # Basic validation
        if pattern.radius < 0 or pattern.density < 0:
            return False
        
        if not (0 <= pattern.quality_score <= 1):
            return False
        
        # Custom validators
        for validator in self.pattern_validators.get('clustering', []):
            if not validator(pattern):
                return False
        
        return True
    
    def _rules_conflict(self, rule1: AssociationRule, rule2: AssociationRule) -> bool:
        """Check if two association rules conflict"""
        # Rules conflict if they have overlapping antecedents but contradictory consequents
        if rule1.antecedent & rule2.antecedent:
            # Check if consequents are mutually exclusive (for this simple check)
            if not (rule1.consequent & rule2.consequent) and rule1.confidence > 0.7 and rule2.confidence > 0.7:
                return True
        
        return False
    
    def _sequential_association_conflict(self, seq_pattern: SequentialPattern, assoc_rule: AssociationRule) -> bool:
        """Check if sequential pattern conflicts with association rule"""
        # Very basic conflict detection - could be expanded
        return False  # Placeholder for now
    
    def add_pattern_validator(self, pattern_type: str, validator: Callable) -> None:
        """Add a custom pattern validator"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if not pattern_type or not isinstance(pattern_type, str):
            raise ValueError("Pattern type must be non-empty string")
        if not callable(validator):
            raise TypeError("Validator must be callable")
        
        if pattern_type not in self.pattern_validators:
            self.pattern_validators[pattern_type] = []
        
        self.pattern_validators[pattern_type].append(validator)