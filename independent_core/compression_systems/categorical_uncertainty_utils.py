"""
Categorical Uncertainty Quantification Utilities

Provides entropy-based and possibility theory methods for quantifying uncertainty
in categorical data storage and compression systems.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class EntropyType(Enum):
    SHANNON = "shannon"
    RENYI = "renyi"
    MIN_ENTROPY = "min_entropy"
    COLLISION_ENTROPY = "collision_entropy"

@dataclass
class EntropyMetrics:
    """Comprehensive entropy metrics for categorical data"""
    shannon_entropy: float
    renyi_entropy: Dict[float, float]  # alpha -> entropy
    min_entropy: float
    collision_entropy: float
    normalized_entropy: float  # Entropy normalized by log2(num_categories)
    entropy_efficiency: float  # How close to maximum entropy

@dataclass
class PossibilityMetrics:
    """Possibility and necessity measures for categorical data"""
    possibility_measure: Dict[str, float]
    necessity_measure: Dict[str, float]
    possibility_entropy: float
    necessity_entropy: float
    uncertainty_balance: float  # Balance between possibility and necessity

class EntropyCalculator:
    """Calculate various entropy measures for categorical data"""
    
    @staticmethod
    def calculate_shannon_entropy(categories: List[str]) -> float:
        """Calculate Shannon entropy for categorical data"""
        if not categories:
            return 0.0
        
        # Count category frequencies
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Calculate probabilities
        total_count = len(categories)
        probabilities = [count / total_count for count in category_counts.values()]
        
        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        return entropy
    
    @staticmethod
    def calculate_renyi_entropy(categories: List[str], alpha: float) -> float:
        """Calculate Rényi entropy of order alpha"""
        if alpha == 1:
            return EntropyCalculator.calculate_shannon_entropy(categories)
        
        # Count category frequencies
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Calculate probabilities
        total_count = len(categories)
        probabilities = [count / total_count for count in category_counts.values()]
        
        # Calculate Rényi entropy
        if alpha > 0 and alpha != 1:
            sum_p_alpha = sum(p**alpha for p in probabilities if p > 0)
            renyi_entropy = (1 / (1 - alpha)) * np.log2(sum_p_alpha)
            return renyi_entropy
        else:
            return 0.0
    
    @staticmethod
    def calculate_min_entropy(categories: List[str]) -> float:
        """Calculate min-entropy (alpha -> infinity)"""
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        total_count = len(categories)
        max_probability = max(count / total_count for count in category_counts.values())
        
        return -np.log2(max_probability)
    
    @staticmethod
    def calculate_collision_entropy(categories: List[str]) -> float:
        """Calculate collision entropy (alpha = 2)"""
        return EntropyCalculator.calculate_renyi_entropy(categories, alpha=2.0)
    
    @staticmethod
    def calculate_all_entropy_metrics(categories: List[str]) -> EntropyMetrics:
        """Calculate all entropy metrics for categorical data"""
        if not categories:
            return EntropyMetrics(
                shannon_entropy=0.0,
                renyi_entropy={},
                min_entropy=0.0,
                collision_entropy=0.0,
                normalized_entropy=0.0,
                entropy_efficiency=0.0
            )
        
        # Calculate basic entropies
        shannon_entropy = EntropyCalculator.calculate_shannon_entropy(categories)
        min_entropy = EntropyCalculator.calculate_min_entropy(categories)
        collision_entropy = EntropyCalculator.calculate_collision_entropy(categories)
        
        # Calculate Rényi entropy for different alpha values
        alpha_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        renyi_entropy = {}
        for alpha in alpha_values:
            renyi_entropy[alpha] = EntropyCalculator.calculate_renyi_entropy(categories, alpha)
        
        # Calculate normalized metrics
        unique_categories = len(set(categories))
        max_entropy = np.log2(unique_categories) if unique_categories > 1 else 0.0
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0.0
        entropy_efficiency = normalized_entropy
        
        return EntropyMetrics(
            shannon_entropy=shannon_entropy,
            renyi_entropy=renyi_entropy,
            min_entropy=min_entropy,
            collision_entropy=collision_entropy,
            normalized_entropy=normalized_entropy,
            entropy_efficiency=entropy_efficiency
        )

class PossibilityCalculator:
    """Calculate possibility and necessity measures for categorical data"""
    
    @staticmethod
    def calculate_possibility_measure(categories: List[str], 
                                   focal_elements: List[set]) -> Dict[str, float]:
        """Calculate possibility measure for focal elements"""
        # Count category frequencies
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        total_count = len(categories)
        
        # Calculate possibility for each focal element
        possibility_measures = {}
        for element in focal_elements:
            # Possibility is maximum probability of any category in the element
            max_prob = max(category_counts.get(cat, 0) / total_count 
                          for cat in element)
            possibility_measures[str(element)] = max_prob
        
        return possibility_measures
    
    @staticmethod
    def calculate_necessity_measure(categories: List[str],
                                 focal_elements: List[set]) -> Dict[str, float]:
        """Calculate necessity measure for focal elements"""
        # Count category frequencies
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        total_count = len(categories)
        
        # Calculate necessity for each focal element
        necessity_measures = {}
        for element in focal_elements:
            # Necessity is sum of probabilities of categories not in complement
            complement = set(category_counts.keys()) - element
            necessity = 1.0 - sum(category_counts.get(cat, 0) / total_count 
                                 for cat in complement)
            necessity_measures[str(element)] = max(0.0, necessity)
        
        return necessity_measures
    
    @staticmethod
    def calculate_possibility_entropy(possibility_measures: Dict[str, float]) -> float:
        """Calculate entropy of possibility measures"""
        if not possibility_measures:
            return 0.0
        
        # Normalize possibility measures
        total_possibility = sum(possibility_measures.values())
        if total_possibility == 0:
            return 0.0
        
        normalized_measures = {k: v / total_possibility for k, v in possibility_measures.items()}
        
        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in normalized_measures.values() if p > 0)
        return entropy
    
    @staticmethod
    def calculate_necessity_entropy(necessity_measures: Dict[str, float]) -> float:
        """Calculate entropy of necessity measures"""
        if not necessity_measures:
            return 0.0
        
        # Normalize necessity measures
        total_necessity = sum(necessity_measures.values())
        if total_necessity == 0:
            return 0.0
        
        normalized_measures = {k: v / total_necessity for k, v in necessity_measures.items()}
        
        # Calculate entropy
        entropy = -sum(p * np.log2(p) for p in normalized_measures.values() if p > 0)
        return entropy
    
    @staticmethod
    def calculate_all_possibility_metrics(categories: List[str], 
                                       focal_elements: Optional[List[set]] = None) -> PossibilityMetrics:
        """Calculate all possibility and necessity metrics"""
        if not categories:
            return PossibilityMetrics(
                possibility_measure={},
                necessity_measure={},
                possibility_entropy=0.0,
                necessity_entropy=0.0,
                uncertainty_balance=0.0
            )
        
        # Generate focal elements if not provided
        if focal_elements is None:
            unique_categories = set(categories)
            focal_elements = [unique_categories]  # Default to all categories
        
        # Calculate possibility and necessity measures
        possibility_measure = PossibilityCalculator.calculate_possibility_measure(categories, focal_elements)
        necessity_measure = PossibilityCalculator.calculate_necessity_measure(categories, focal_elements)
        
        # Calculate entropies
        possibility_entropy = PossibilityCalculator.calculate_possibility_entropy(possibility_measure)
        necessity_entropy = PossibilityCalculator.calculate_necessity_entropy(necessity_measure)
        
        # Calculate uncertainty balance
        total_possibility = sum(possibility_measure.values())
        total_necessity = sum(necessity_measure.values())
        
        if total_possibility > 0 and total_necessity > 0:
            uncertainty_balance = total_necessity / total_possibility
        else:
            uncertainty_balance = 0.0
        
        return PossibilityMetrics(
            possibility_measure=possibility_measure,
            necessity_measure=necessity_measure,
            possibility_entropy=possibility_entropy,
            necessity_entropy=necessity_entropy,
            uncertainty_balance=uncertainty_balance
        )

class CategoricalUncertaintyAnalyzer:
    """Comprehensive analyzer for categorical uncertainty"""
    
    def __init__(self):
        self.entropy_calculator = EntropyCalculator()
        self.possibility_calculator = PossibilityCalculator()
    
    def analyze_categorical_uncertainty(self, categories: List[str], 
                                     focal_elements: Optional[List[set]] = None) -> Dict[str, Any]:
        """Comprehensive analysis of categorical uncertainty"""
        try:
            # Calculate entropy metrics
            entropy_metrics = EntropyCalculator.calculate_all_entropy_metrics(categories)
            
            # Calculate possibility metrics
            possibility_metrics = PossibilityCalculator.calculate_all_possibility_metrics(
                categories, focal_elements
            )
            
            # Calculate additional uncertainty measures
            unique_categories = len(set(categories))
            total_count = len(categories)
            
            # Category diversity
            category_diversity = unique_categories / total_count if total_count > 0 else 0.0
            
            # Concentration index (Herfindahl index)
            category_counts = {}
            for cat in categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            concentration_index = sum((count / total_count) ** 2 for count in category_counts.values())
            
            # Uncertainty decomposition
            uncertainty_decomposition = {
                'data_uncertainty': entropy_metrics.shannon_entropy,
                'structural_uncertainty': entropy_metrics.normalized_entropy,
                'possibility_uncertainty': possibility_metrics.possibility_entropy,
                'necessity_uncertainty': possibility_metrics.necessity_entropy
            }
            
            return {
                'entropy_metrics': entropy_metrics,
                'possibility_metrics': possibility_metrics,
                'category_diversity': category_diversity,
                'concentration_index': concentration_index,
                'uncertainty_decomposition': uncertainty_decomposition,
                'summary': {
                    'total_categories': total_count,
                    'unique_categories': unique_categories,
                    'shannon_entropy': entropy_metrics.shannon_entropy,
                    'min_entropy': entropy_metrics.min_entropy,
                    'entropy_efficiency': entropy_metrics.entropy_efficiency,
                    'uncertainty_balance': possibility_metrics.uncertainty_balance
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing categorical uncertainty: {e}")
            return {
                'error': str(e),
                'entropy_metrics': EntropyMetrics(0.0, {}, 0.0, 0.0, 0.0, 0.0),
                'possibility_metrics': PossibilityMetrics({}, {}, 0.0, 0.0, 0.0),
                'summary': {}
            }
    
    def compare_uncertainty_methods(self, categories: List[str]) -> Dict[str, float]:
        """Compare different uncertainty quantification methods"""
        try:
            # Calculate metrics for different methods
            entropy_metrics = EntropyCalculator.calculate_all_entropy_metrics(categories)
            possibility_metrics = PossibilityCalculator.calculate_all_possibility_metrics(categories)
            
            # Compare methods
            comparison = {
                'shannon_entropy': entropy_metrics.shannon_entropy,
                'min_entropy': entropy_metrics.min_entropy,
                'collision_entropy': entropy_metrics.collision_entropy,
                'renyi_entropy_2': entropy_metrics.renyi_entropy.get(2.0, 0.0),
                'renyi_entropy_5': entropy_metrics.renyi_entropy.get(5.0, 0.0),
                'possibility_entropy': possibility_metrics.possibility_entropy,
                'necessity_entropy': possibility_metrics.necessity_entropy,
                'entropy_efficiency': entropy_metrics.entropy_efficiency,
                'uncertainty_balance': possibility_metrics.uncertainty_balance
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing uncertainty methods: {e}")
            return {'error': str(e)}

def calculate_categorical_uncertainty(categories: List[str], 
                                   method: str = 'comprehensive') -> Dict[str, Any]:
    """High-level function to calculate categorical uncertainty"""
    analyzer = CategoricalUncertaintyAnalyzer()
    
    if method == 'comprehensive':
        return analyzer.analyze_categorical_uncertainty(categories)
    elif method == 'entropy_only':
        entropy_metrics = EntropyCalculator.calculate_all_entropy_metrics(categories)
        return {'entropy_metrics': entropy_metrics}
    elif method == 'possibility_only':
        possibility_metrics = PossibilityCalculator.calculate_all_possibility_metrics(categories)
        return {'possibility_metrics': possibility_metrics}
    elif method == 'comparison':
        return analyzer.compare_uncertainty_methods(categories)
    else:
        raise ValueError(f"Unknown method: {method}")

def is_categorical_data(data: Any) -> bool:
    """Check if data is categorical"""
    if isinstance(data, (list, np.ndarray)):
        # Check if all elements are strings or have limited unique values
        unique_values = set(str(x) for x in data)
        return len(unique_values) < len(data) * 0.5  # Less than 50% unique values
    return False

def get_optimal_uncertainty_method(categories: List[str]) -> str:
    """Determine optimal uncertainty quantification method for categorical data"""
    if not categories:
        return 'entropy_only'
    
    unique_categories = len(set(categories))
    total_count = len(categories)
    
    # For small datasets, use comprehensive analysis
    if total_count < 100:
        return 'comprehensive'
    
    # For high cardinality, use entropy-based methods
    if unique_categories > total_count * 0.3:
        return 'entropy_only'
    
    # For balanced datasets, use possibility theory
    if 0.1 <= unique_categories / total_count <= 0.3:
        return 'possibility_only'
    
    # Default to comprehensive
    return 'comprehensive' 