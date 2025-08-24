"""
Comprehensive test suite for PatternMiningEngine
Tests all aspects of pattern mining, recommendation generation, and pattern validation
"""

import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Set

from proof_system.pattern_mining_engine import (
    SequentialPattern,
    AssociationRule,
    ClusterPattern,
    PatternRecommendation,
    SequentialPatternMiner,
    AssociationRuleMiner,
    ClusteringPatternMiner,
    PatternMiningEngine
)


class TestSequentialPattern:
    """Test SequentialPattern dataclass"""
    
    def test_valid_sequential_pattern(self):
        """Test creating valid sequential pattern"""
        pattern = SequentialPattern(
            sequence=['strategy1', 'strategy2', 'strategy3'],
            support=10,
            confidence=0.8,
            length=3
        )
        assert pattern.sequence == ['strategy1', 'strategy2', 'strategy3']
        assert pattern.support == 10
        assert pattern.confidence == 0.8
        assert pattern.length == 3
    
    def test_invalid_sequence_empty(self):
        """Test that empty sequence raises ValueError"""
        with pytest.raises(ValueError, match="Sequence must be non-empty list"):
            SequentialPattern([], 5, 0.5, 0)
    
    def test_invalid_sequence_type(self):
        """Test that non-list sequence raises ValueError"""
        with pytest.raises(ValueError, match="Sequence must be non-empty list"):
            SequentialPattern("not_a_list", 5, 0.5, 1)
    
    def test_invalid_sequence_items(self):
        """Test that non-string items raise TypeError"""
        with pytest.raises(TypeError, match="All sequence items must be strings"):
            SequentialPattern([1, 2, 3], 5, 0.5, 3)
    
    def test_negative_support(self):
        """Test that negative support raises ValueError"""
        with pytest.raises(ValueError, match="Support must be non-negative"):
            SequentialPattern(['a'], -1, 0.5, 1)
    
    def test_invalid_confidence_range(self):
        """Test that confidence outside [0, 1] raises ValueError"""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            SequentialPattern(['a'], 5, 1.5, 1)
    
    def test_length_auto_correction(self):
        """Test that length is auto-corrected if incorrect"""
        pattern = SequentialPattern(['a', 'b'], 5, 0.5, 10)
        assert pattern.length == 2


class TestAssociationRule:
    """Test AssociationRule dataclass"""
    
    def test_valid_association_rule(self):
        """Test creating valid association rule"""
        rule = AssociationRule(
            antecedent={'item1', 'item2'},
            consequent={'item3'},
            support=0.3,
            confidence=0.7,
            lift=2.1,
            conviction=3.5
        )
        assert rule.antecedent == {'item1', 'item2'}
        assert rule.consequent == {'item3'}
        assert rule.support == 0.3
        assert rule.confidence == 0.7
        assert rule.lift == 2.1
        assert rule.conviction == 3.5
    
    def test_empty_antecedent(self):
        """Test that empty antecedent raises ValueError"""
        with pytest.raises(ValueError, match="Antecedent must be non-empty set"):
            AssociationRule(set(), {'item1'}, 0.5, 0.7, 1.0, 2.0)
    
    def test_empty_consequent(self):
        """Test that empty consequent raises ValueError"""
        with pytest.raises(ValueError, match="Consequent must be non-empty set"):
            AssociationRule({'item1'}, set(), 0.5, 0.7, 1.0, 2.0)
    
    def test_non_disjoint_sets(self):
        """Test that overlapping antecedent and consequent raise ValueError"""
        with pytest.raises(ValueError, match="Antecedent and consequent must be disjoint"):
            AssociationRule({'item1', 'item2'}, {'item2', 'item3'}, 0.5, 0.7, 1.0, 2.0)
    
    def test_invalid_metrics(self):
        """Test that invalid metrics raise ValueError"""
        with pytest.raises(ValueError, match="Support must be between 0.0 and 1.0"):
            AssociationRule({'item1'}, {'item2'}, -0.1, 0.7, 1.0, 2.0)
        
        with pytest.raises(ValueError, match="Lift must be non-negative"):
            AssociationRule({'item1'}, {'item2'}, 0.5, 0.7, -1.0, 2.0)


class TestClusterPattern:
    """Test ClusterPattern dataclass"""
    
    def test_valid_cluster_pattern(self):
        """Test creating valid cluster pattern"""
        pattern = ClusterPattern(
            cluster_id=1,
            centroid={'feature1': 0.5, 'feature2': 1.2},
            members=[{'data': 'member1'}, {'data': 'member2'}],
            radius=2.5,
            density=0.8,
            quality_score=0.75
        )
        assert pattern.cluster_id == 1
        assert pattern.centroid == {'feature1': 0.5, 'feature2': 1.2}
        assert len(pattern.members) == 2
        assert pattern.radius == 2.5
        assert pattern.density == 0.8
        assert pattern.quality_score == 0.75
    
    def test_negative_cluster_id(self):
        """Test that negative cluster ID raises ValueError"""
        with pytest.raises(ValueError, match="Cluster ID must be non-negative"):
            ClusterPattern(-1, {}, [], 1.0, 0.5, 0.5)
    
    def test_invalid_centroid_type(self):
        """Test that non-dict centroid raises TypeError"""
        with pytest.raises(TypeError, match="Centroid must be dict"):
            ClusterPattern(0, "not_a_dict", [], 1.0, 0.5, 0.5)
    
    def test_non_numeric_centroid_values(self):
        """Test that non-numeric centroid values raise TypeError"""
        with pytest.raises(TypeError, match="Centroid values must be numeric"):
            ClusterPattern(0, {'feature': 'string'}, [], 1.0, 0.5, 0.5)


class TestPatternRecommendation:
    """Test PatternRecommendation dataclass"""
    
    def test_valid_pattern_recommendation(self):
        """Test creating valid pattern recommendation"""
        rec = PatternRecommendation(
            pattern_type='sequential',
            pattern_data={'sequence': ['a', 'b']},
            recommendation='Use strategy B',
            confidence=0.85,
            reasoning='Historical pattern suggests B follows A',
            applicable_contexts=['context1', 'context2']
        )
        assert rec.pattern_type == 'sequential'
        assert rec.pattern_data == {'sequence': ['a', 'b']}
        assert rec.recommendation == 'Use strategy B'
        assert rec.confidence == 0.85
        assert rec.reasoning == 'Historical pattern suggests B follows A'
        assert rec.applicable_contexts == ['context1', 'context2']
    
    def test_invalid_pattern_type(self):
        """Test that invalid pattern type raises ValueError"""
        with pytest.raises(ValueError, match="Pattern type must be non-empty string"):
            PatternRecommendation('', {}, 'rec', 0.5, 'reason', [])


class TestSequentialPatternMiner:
    """Test SequentialPatternMiner"""
    
    def test_initialization(self):
        """Test miner initialization"""
        miner = SequentialPatternMiner(min_support=3, min_confidence=0.6)
        assert miner.min_support == 3
        assert miner.min_confidence == 0.6
    
    def test_invalid_initialization(self):
        """Test invalid initialization parameters"""
        with pytest.raises(ValueError, match="Min support must be at least 1"):
            SequentialPatternMiner(min_support=0)
        
        with pytest.raises(ValueError, match="Min confidence must be between 0.0 and 1.0"):
            SequentialPatternMiner(min_confidence=1.5)
    
    def test_mine_patterns_simple(self):
        """Test mining patterns from simple sequences"""
        miner = SequentialPatternMiner(min_support=2, min_confidence=0.3)
        sequences = [
            ['a', 'b', 'c'],
            ['a', 'b', 'd'],
            ['a', 'c', 'd'],
            ['b', 'c', 'd']
        ]
        
        patterns = miner.mine_patterns(sequences)
        assert len(patterns) > 0
        
        # Check for expected frequent items
        pattern_sequences = [p.sequence for p in patterns]
        assert ['a'] in pattern_sequences
        assert ['b'] in pattern_sequences
        assert ['c'] in pattern_sequences
        assert ['d'] in pattern_sequences
    
    def test_mine_patterns_empty(self):
        """Test mining patterns with empty input"""
        miner = SequentialPatternMiner()
        with pytest.raises(ValueError, match="Sequences must be non-empty list"):
            miner.mine_patterns([])
    
    def test_subsequence_detection(self):
        """Test subsequence detection logic"""
        miner = SequentialPatternMiner()
        assert miner._is_subsequence(['a', 'b'], ['a', 'x', 'b', 'y'])
        assert miner._is_subsequence(['a', 'c'], ['a', 'b', 'c'])
        assert not miner._is_subsequence(['a', 'b'], ['b', 'a'])
        assert not miner._is_subsequence(['a', 'b', 'c'], ['a', 'c'])
    
    def test_candidate_generation(self):
        """Test candidate pattern generation"""
        miner = SequentialPatternMiner()
        frequent = [['a', 'b'], ['b', 'c'], ['a', 'c']]
        candidates = miner._generate_candidates(frequent)
        
        # Should generate patterns like ['a', 'b', 'c']
        assert len(candidates) > 0
        for candidate in candidates:
            assert len(candidate) == 3
    
    def test_support_counting(self):
        """Test support counting for subsequences"""
        miner = SequentialPatternMiner()
        sequences = [
            ['a', 'b', 'c'],
            ['a', 'c', 'b'],
            ['b', 'a', 'c']
        ]
        
        # 'a' followed by 'c' appears in all sequences
        support = miner._count_subsequence_support(['a', 'c'], sequences)
        assert support == 3
        
        # 'a' followed by 'b' followed by 'c' appears only in first
        support = miner._count_subsequence_support(['a', 'b', 'c'], sequences)
        assert support == 1


class TestAssociationRuleMiner:
    """Test AssociationRuleMiner"""
    
    def test_initialization(self):
        """Test miner initialization"""
        miner = AssociationRuleMiner(min_support=0.2, min_confidence=0.7)
        assert miner.min_support == 0.2
        assert miner.min_confidence == 0.7
    
    def test_mine_rules_simple(self):
        """Test mining rules from simple transactions"""
        miner = AssociationRuleMiner(min_support=0.3, min_confidence=0.5)
        transactions = [
            {'milk', 'bread', 'butter'},
            {'milk', 'bread'},
            {'bread', 'butter'},
            {'milk', 'butter'},
            {'milk', 'bread', 'butter', 'cheese'}
        ]
        
        rules = miner.mine_rules(transactions)
        assert len(rules) > 0
        
        # Verify rule structure
        for rule in rules:
            assert isinstance(rule.antecedent, set)
            assert isinstance(rule.consequent, set)
            assert 0 <= rule.support <= 1
            assert 0 <= rule.confidence <= 1
            assert rule.lift >= 0
    
    def test_mine_rules_empty(self):
        """Test mining rules with empty input"""
        miner = AssociationRuleMiner()
        with pytest.raises(ValueError, match="Transactions must be non-empty list"):
            miner.mine_rules([])
    
    def test_frequent_itemsets(self):
        """Test frequent itemset generation"""
        miner = AssociationRuleMiner(min_support=0.4)
        transactions = [
            {'a', 'b'},
            {'a', 'c'},
            {'a', 'b', 'c'},
            {'b', 'c'},
            {'a', 'b', 'c'}
        ]
        
        itemsets = miner._find_frequent_itemsets(transactions)
        assert frozenset(['a']) in itemsets
        assert frozenset(['b']) in itemsets
        assert frozenset(['c']) in itemsets
        
        # With min_support=0.4, need at least 2 occurrences
        assert itemsets[frozenset(['a'])] >= 2
        assert itemsets[frozenset(['b'])] >= 2
        assert itemsets[frozenset(['c'])] >= 2
    
    def test_apriori_candidate_generation(self):
        """Test Apriori candidate generation"""
        miner = AssociationRuleMiner()
        frequent = [frozenset(['a']), frozenset(['b']), frozenset(['c'])]
        candidates = miner._generate_candidates_apriori(frequent)
        
        expected = [frozenset(['a', 'b']), frozenset(['a', 'c']), frozenset(['b', 'c'])]
        for candidate in candidates:
            assert candidate in expected
    
    def test_infrequent_subset_check(self):
        """Test checking for infrequent subsets"""
        miner = AssociationRuleMiner()
        frequent = [frozenset(['a']), frozenset(['b']), frozenset(['a', 'b'])]
        
        # All subsets are frequent
        assert not miner._has_infrequent_subset(frozenset(['a', 'b']), frequent)
        
        # Missing subset
        assert miner._has_infrequent_subset(frozenset(['a', 'b', 'c']), frequent)


class TestClusteringPatternMiner:
    """Test ClusteringPatternMiner"""
    
    def test_initialization(self):
        """Test miner initialization"""
        miner = ClusteringPatternMiner(n_clusters=3, max_iterations=50)
        assert miner.n_clusters == 3
        assert miner.max_iterations == 50
    
    def test_invalid_initialization(self):
        """Test invalid initialization parameters"""
        with pytest.raises(ValueError, match="Number of clusters must be positive"):
            ClusteringPatternMiner(n_clusters=0)
        
        with pytest.raises(ValueError, match="Max iterations must be positive"):
            ClusteringPatternMiner(max_iterations=0)
    
    def test_mine_patterns_simple(self):
        """Test mining patterns from simple execution data"""
        miner = ClusteringPatternMiner(n_clusters=2, max_iterations=10)
        
        execution_data = [
            {'confidence': 0.9, 'execution_time': 1.0, 'success': True},
            {'confidence': 0.8, 'execution_time': 1.2, 'success': True},
            {'confidence': 0.3, 'execution_time': 5.0, 'success': False},
            {'confidence': 0.2, 'execution_time': 4.8, 'success': False},
        ]
        
        patterns = miner.mine_patterns(execution_data)
        assert len(patterns) <= 2
        
        for pattern in patterns:
            assert pattern.cluster_id >= 0
            assert pattern.radius >= 0
            assert pattern.density >= 0
            assert 0 <= pattern.quality_score <= 1
    
    def test_feature_extraction(self):
        """Test feature extraction from execution data"""
        miner = ClusteringPatternMiner()
        
        execution_data = [
            {
                'confidence': 0.8,
                'execution_time': 2.5,
                'success': True,
                'context': {'depth': 3, 'width': 10},
                'proof_data': {
                    'steps': ['step1', 'step2'],
                    'constraints': ['c1', 'c2', 'c3']
                }
            }
        ]
        
        features = miner._extract_features(execution_data)
        assert len(features) == 1
        
        feature = features[0]
        assert feature['confidence'] == 0.8
        assert feature['execution_time'] == 2.5
        assert feature['success'] == 1.0
        assert feature['context_depth'] == 3.0
        assert feature['context_width'] == 10.0
        assert feature['proof_steps_count'] == 2.0
        assert feature['constraints_count'] == 3.0
    
    def test_kmeans_clustering(self):
        """Test k-means clustering algorithm"""
        miner = ClusteringPatternMiner(n_clusters=2, max_iterations=100)
        
        features = [
            {'f1': 1.0, 'f2': 1.0},
            {'f1': 1.5, 'f2': 1.5},
            {'f1': 5.0, 'f2': 5.0},
            {'f1': 5.5, 'f2': 5.5}
        ]
        
        clusters = miner._kmeans_clustering(features)
        assert len(clusters) == 2
        
        # Check that clustering separates the two groups
        for cluster_id, cluster_data in clusters.items():
            assert 'centroid' in cluster_data
            assert 'members' in cluster_data
            assert len(cluster_data['members']) > 0
    
    def test_cluster_quality_calculation(self):
        """Test cluster quality score calculation"""
        miner = ClusteringPatternMiner()
        
        cluster_data = {
            'members': [
                {'success': 1.0, 'confidence': 0.8},
                {'success': 1.0, 'confidence': 0.85},
                {'success': 1.0, 'confidence': 0.82}
            ]
        }
        
        quality = miner._calculate_cluster_quality(cluster_data)
        assert 0 <= quality <= 1
        
        # Homogeneous cluster should have high quality
        assert quality > 0.5


class TestPatternMiningEngine:
    """Test main PatternMiningEngine"""
    
    def test_initialization_default(self):
        """Test engine initialization with default config"""
        engine = PatternMiningEngine()
        assert engine.sequential_miner is not None
        assert engine.association_miner is not None
        assert engine.clustering_miner is not None
        assert engine.recommendation_threshold == 0.7
    
    def test_initialization_custom_config(self):
        """Test engine initialization with custom config"""
        config = {
            'sequential': {'min_support': 5, 'min_confidence': 0.8},
            'association': {'min_support': 0.2, 'min_confidence': 0.6},
            'clustering': {'n_clusters': 10, 'max_iterations': 200},
            'recommendation_threshold': 0.9
        }
        
        engine = PatternMiningEngine(config)
        assert engine.sequential_miner.min_support == 5
        assert engine.sequential_miner.min_confidence == 0.8
        assert engine.association_miner.min_support == 0.2
        assert engine.association_miner.min_confidence == 0.6
        assert engine.clustering_miner.n_clusters == 10
        assert engine.clustering_miner.max_iterations == 200
        assert engine.recommendation_threshold == 0.9
    
    def test_mine_all_patterns(self):
        """Test mining all pattern types"""
        engine = PatternMiningEngine()
        
        execution_data = [
            {
                'strategy_name': 'strategy_a',
                'start_time': 1000,
                'confidence': 0.8,
                'execution_time': 1.5,
                'success': True,
                'context': {'type': 'test'}
            },
            {
                'strategy_name': 'strategy_b',
                'start_time': 1100,
                'confidence': 0.9,
                'execution_time': 1.2,
                'success': True,
                'context': {'type': 'test'}
            },
            {
                'strategy_name': 'strategy_c',
                'start_time': 1200,
                'confidence': 0.7,
                'execution_time': 2.0,
                'success': False,
                'context': {'type': 'test'}
            }
        ]
        
        results = engine.mine_all_patterns(execution_data)
        
        assert 'sequential_patterns' in results
        assert 'association_rules' in results
        assert 'cluster_patterns' in results
        assert 'total_execution_data' in results
        assert 'mining_timestamp' in results
        
        assert results['total_execution_data'] == 3
        assert results['sequential_patterns'] >= 0
        assert results['association_rules'] >= 0
        assert results['cluster_patterns'] >= 0
    
    def test_mine_all_patterns_empty(self):
        """Test mining with empty data"""
        engine = PatternMiningEngine()
        
        with pytest.raises(ValueError, match="Execution data must be non-empty list"):
            engine.mine_all_patterns([])
    
    def test_get_pattern_recommendations(self):
        """Test getting pattern-based recommendations"""
        engine = PatternMiningEngine()
        
        # Manually add some patterns for testing
        engine.sequential_patterns = [
            SequentialPattern(['strategy_a', 'strategy_b'], 5, 0.8, 2)
        ]
        
        engine.association_rules = [
            AssociationRule(
                {'strategy_a'},
                {'strategy_c'},
                0.5,
                0.9,
                2.0,
                3.0
            )
        ]
        
        context = {
            'current_strategy': 'strategy_a',
            'used_strategies': ['strategy_a'],
            'confidence': 0.8,
            'execution_time': 1.5
        }
        
        recommendations = engine.get_pattern_recommendations(context)
        
        # Should get recommendations from sequential patterns and association rules
        assert len(recommendations) > 0
        
        for rec in recommendations:
            assert rec.confidence >= engine.recommendation_threshold
            assert rec.pattern_type in ['sequential', 'association', 'clustering']
    
    def test_validate_patterns(self):
        """Test pattern validation"""
        engine = PatternMiningEngine()
        
        # Add valid patterns
        engine.sequential_patterns = [
            SequentialPattern(['a', 'b'], 5, 0.8, 2),
            SequentialPattern(['c'], 1, 0.5, 1)  # Valid
        ]
        
        engine.association_rules = [
            AssociationRule({'a'}, {'b'}, 0.5, 0.8, 1.5, 2.0),
            AssociationRule({'c'}, {'d'}, 0.95, 0.8, 1.0, 2.0)  # Valid
        ]
        
        validation_results = engine.validate_patterns()
        
        assert 'sequential' in validation_results
        assert 'association' in validation_results
        assert 'clustering' in validation_results
        
        # Both sequential patterns are valid
        assert validation_results['sequential']['valid'] == 2
        assert validation_results['sequential']['total'] == 2
        
        # Both association rules are valid
        assert validation_results['association']['valid'] == 2
        assert validation_results['association']['total'] == 2
    
    def test_detect_pattern_conflicts(self):
        """Test conflict detection between patterns"""
        engine = PatternMiningEngine()
        
        # Add potentially conflicting rules
        engine.association_rules = [
            AssociationRule({'a', 'b'}, {'c'}, 0.5, 0.9, 2.0, 3.0),
            AssociationRule({'a', 'b'}, {'d'}, 0.5, 0.9, 2.0, 3.0)
        ]
        
        conflicts = engine.detect_pattern_conflicts()
        
        # Should detect conflict between rules with same antecedent but different consequent
        assert len(conflicts) > 0
        
        for conflict in conflicts:
            assert 'type' in conflict
            assert 'description' in conflict
    
    def test_export_import_patterns(self):
        """Test exporting and importing patterns"""
        engine1 = PatternMiningEngine()
        
        # Add patterns
        engine1.sequential_patterns = [
            SequentialPattern(['a', 'b'], 5, 0.8, 2)
        ]
        engine1.association_rules = [
            AssociationRule({'a'}, {'b'}, 0.5, 0.8, 1.5, 2.0)
        ]
        engine1.cluster_patterns = [
            ClusterPattern(0, {'f1': 1.0}, [], 2.0, 0.5, 0.7)
        ]
        
        # Export
        exported = engine1.export_patterns()
        
        assert 'sequential_patterns' in exported
        assert 'association_rules' in exported
        assert 'cluster_patterns' in exported
        assert 'export_timestamp' in exported
        
        # Import into new engine
        engine2 = PatternMiningEngine()
        engine2.import_patterns(exported)
        
        assert len(engine2.sequential_patterns) == 1
        assert len(engine2.association_rules) == 1
        assert len(engine2.cluster_patterns) == 1
        
        # Verify imported data
        assert engine2.sequential_patterns[0].sequence == ['a', 'b']
        assert engine2.association_rules[0].antecedent == {'a'}
        assert engine2.cluster_patterns[0].cluster_id == 0
    
    def test_add_pattern_validator(self):
        """Test adding custom pattern validators"""
        engine = PatternMiningEngine()
        
        def custom_validator(pattern):
            return pattern.confidence > 0.5
        
        engine.add_pattern_validator('sequential', custom_validator)
        
        assert 'sequential' in engine.pattern_validators
        assert len(engine.pattern_validators['sequential']) == 1
        
        # Test invalid inputs
        with pytest.raises(ValueError, match="Pattern type must be non-empty string"):
            engine.add_pattern_validator('', custom_validator)
        
        with pytest.raises(TypeError, match="Validator must be callable"):
            engine.add_pattern_validator('test', "not_callable")
    
    def test_extract_strategy_sequences(self):
        """Test extracting strategy sequences from execution data"""
        engine = PatternMiningEngine()
        
        execution_data = [
            {'strategy_name': 'a', 'start_time': 1000},
            {'strategy_name': 'b', 'start_time': 1100},
            {'strategy_name': 'c', 'start_time': 1200},
            {'strategy_name': 'd', 'start_time': 5000},  # Large gap, new sequence
            {'strategy_name': 'e', 'start_time': 5100}
        ]
        
        sequences = engine._extract_strategy_sequences(execution_data)
        
        assert len(sequences) == 2
        assert sequences[0] == ['a', 'b', 'c']
        assert sequences[1] == ['d', 'e']
    
    def test_extract_strategy_transactions(self):
        """Test extracting strategy transactions from execution data"""
        engine = PatternMiningEngine()
        
        execution_data = [
            {'strategy_name': 'a', 'context': {'type': 'test', 'level': 1}},
            {'strategy_name': 'b', 'context': {'type': 'test', 'level': 1}},
            {'strategy_name': 'c', 'context': {'type': 'prod', 'level': 2}},
            {'strategy_name': 'd', 'context': {'type': 'prod', 'level': 2}}
        ]
        
        transactions = engine._extract_strategy_transactions(execution_data)
        
        # Should group by context signature
        assert len(transactions) == 2
        
        for transaction in transactions:
            assert isinstance(transaction, set)
            assert len(transaction) >= 2
    
    def test_sequential_recommendations(self):
        """Test getting recommendations from sequential patterns"""
        engine = PatternMiningEngine()
        
        engine.sequential_patterns = [
            SequentialPattern(['a', 'b', 'c'], 10, 0.9, 3),
            SequentialPattern(['b', 'd'], 5, 0.7, 2)
        ]
        
        context = {'current_strategy': 'b'}
        recommendations = engine._get_sequential_recommendations(context)
        
        assert len(recommendations) > 0
        
        # Should recommend 'c' after 'b' from first pattern
        rec_texts = [r.recommendation for r in recommendations]
        assert any('c' in text for text in rec_texts)
    
    def test_association_recommendations(self):
        """Test getting recommendations from association rules"""
        engine = PatternMiningEngine()
        
        engine.association_rules = [
            AssociationRule({'a', 'b'}, {'c'}, 0.5, 0.9, 2.0, 3.0),
            AssociationRule({'a'}, {'d'}, 0.4, 0.8, 1.8, 2.5)
        ]
        
        context = {'used_strategies': ['a', 'b']}
        recommendations = engine._get_association_recommendations(context)
        
        assert len(recommendations) > 0
        
        # Should recommend 'c' based on first rule
        rec_texts = [r.recommendation for r in recommendations]
        assert any('c' in text for text in rec_texts)
    
    def test_cluster_recommendations(self):
        """Test getting recommendations from cluster patterns"""
        engine = PatternMiningEngine()
        
        engine.cluster_patterns = [
            ClusterPattern(
                0,
                {'confidence': 0.8, 'execution_time': 1.5},
                [],
                2.0,
                0.5,
                0.9
            )
        ]
        
        context = {'confidence': 0.85, 'execution_time': 1.4}
        recommendations = engine._get_cluster_recommendations(context)
        
        assert len(recommendations) > 0
        
        # Should match to cluster 0
        assert recommendations[0].pattern_data['cluster_id'] == 0
    
    def test_thread_safety(self):
        """Test thread safety of pattern operations"""
        engine = PatternMiningEngine()
        results = []
        errors = []
        
        def mine_patterns():
            try:
                data = [
                    {'strategy_name': f'strategy_{i}', 'start_time': i * 100,
                     'confidence': 0.5 + i * 0.1, 'execution_time': 1.0 + i * 0.2,
                     'success': i % 2 == 0, 'context': {'thread': threading.current_thread().name}}
                    for i in range(5)
                ]
                result = engine.mine_all_patterns(data)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=mine_patterns)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 5
    
    def test_pattern_validation_edge_cases(self):
        """Test pattern validation with edge cases"""
        engine = PatternMiningEngine()
        
        # Test sequential pattern validation
        valid_seq = SequentialPattern(['a'], 1, 0.5, 1)
        assert engine._validate_sequential_pattern(valid_seq)
        
        invalid_seq = SequentialPattern(['a'], 0, 0.5, 1)  # support < 1
        assert not engine._validate_sequential_pattern(invalid_seq)
        
        # Test association rule validation
        valid_rule = AssociationRule({'a'}, {'b'}, 0.5, 0.8, 1.0, 2.0)
        assert engine._validate_association_rule(valid_rule)
        
        invalid_rule = AssociationRule({'a'}, {'b'}, 1.5, 0.8, 1.0, 2.0)  # support > 1
        assert not engine._validate_association_rule(invalid_rule)
        
        # Test cluster pattern validation
        valid_cluster = ClusterPattern(0, {}, [], 1.0, 0.5, 0.7)
        assert engine._validate_cluster_pattern(valid_cluster)
        
        invalid_cluster = ClusterPattern(0, {}, [], -1.0, 0.5, 0.7)  # negative radius
        assert not engine._validate_cluster_pattern(invalid_cluster)
    
    def test_rules_conflict_detection(self):
        """Test specific rule conflict detection logic"""
        engine = PatternMiningEngine()
        
        # Conflicting rules: same antecedent, different consequents, high confidence
        rule1 = AssociationRule({'a', 'b'}, {'c'}, 0.5, 0.8, 2.0, 3.0)
        rule2 = AssociationRule({'a', 'b'}, {'d'}, 0.5, 0.8, 2.0, 3.0)
        
        assert engine._rules_conflict(rule1, rule2)
        
        # Non-conflicting: different antecedents
        rule3 = AssociationRule({'e'}, {'c'}, 0.5, 0.8, 2.0, 3.0)
        assert not engine._rules_conflict(rule1, rule3)
        
        # Non-conflicting: low confidence
        rule4 = AssociationRule({'a', 'b'}, {'e'}, 0.5, 0.3, 1.0, 1.5)
        assert not engine._rules_conflict(rule1, rule4)
        
        # Non-conflicting: overlapping consequents
        rule5 = AssociationRule({'a', 'b'}, {'c', 'd'}, 0.5, 0.8, 2.0, 3.0)
        assert not engine._rules_conflict(rule1, rule5)
    
    def test_error_handling_in_mining(self):
        """Test error handling during pattern mining"""
        engine = PatternMiningEngine()
        
        # Test with malformed execution data
        bad_data = [
            {'invalid': 'data'},  # Missing required fields
            None,  # None value
            'not_a_dict'  # Wrong type
        ]
        
        # Should handle type errors
        with pytest.raises(TypeError, match="All execution data items must be dicts"):
            engine.mine_all_patterns(bad_data)
    
    def test_clustering_with_insufficient_data(self):
        """Test clustering with very few data points"""
        engine = PatternMiningEngine({'clustering': {'n_clusters': 5}})
        
        # Only 2 data points but requesting 5 clusters
        execution_data = [
            {'confidence': 0.8, 'execution_time': 1.0},
            {'confidence': 0.3, 'execution_time': 2.0}
        ]
        
        results = engine.mine_all_patterns(execution_data)
        
        # Should adapt to available data
        assert results['cluster_patterns'] <= 2
    
    def test_pattern_sorting(self):
        """Test that patterns are properly sorted"""
        engine = PatternMiningEngine()
        
        # Add patterns with different scores
        engine.sequential_patterns = [
            SequentialPattern(['a'], 5, 0.5, 1),
            SequentialPattern(['b'], 10, 0.8, 1),
            SequentialPattern(['c'], 3, 0.3, 1)
        ]
        
        # Mine to trigger sorting
        execution_data = [{'strategy_name': 'test', 'start_time': 0}]
        engine.mine_all_patterns(execution_data)
        
        # Verify sorted by support (descending)
        supports = [p.support for p in engine.sequential_patterns]
        assert supports == sorted(supports, reverse=True)
    
    def test_recommendation_threshold_filtering(self):
        """Test that recommendations are filtered by threshold"""
        engine = PatternMiningEngine({'recommendation_threshold': 0.8})
        
        # Add patterns with varying confidence
        engine.sequential_patterns = [
            SequentialPattern(['a', 'b'], 10, 0.9, 2),  # Will pass threshold
            SequentialPattern(['c', 'd'], 5, 0.6, 2)    # Won't pass threshold
        ]
        
        context = {'current_strategy': 'a'}
        recommendations = engine.get_pattern_recommendations(context)
        
        # All recommendations should meet threshold
        for rec in recommendations:
            assert rec.confidence >= 0.8
    
    def test_infinity_conviction_handling(self):
        """Test handling of infinite conviction in association rules"""
        miner = AssociationRuleMiner(min_support=0.1, min_confidence=0.5)
        
        # Create scenario where confidence = 1.0 (leads to infinite conviction)
        transactions = [
            {'a', 'b'},
            {'a', 'b'},
            {'a', 'b'},
            {'c'}
        ]
        
        rules = miner.mine_rules(transactions)
        
        # Find rule with confidence = 1.0
        high_conf_rules = [r for r in rules if r.confidence == 1.0]
        if high_conf_rules:
            assert high_conf_rules[0].conviction == float('inf')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])