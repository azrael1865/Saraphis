"""
Domain Orchestrator - Domain-Specific Operations Coordinator
Manages domain-specific operations, expertise routing, and specialized processing
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock, Lock
import threading
from collections import defaultdict, deque
import json
import traceback
from abc import ABC, abstractmethod
import importlib
import inspect

logger = logging.getLogger(__name__)

class DomainType(Enum):
    MATHEMATICS = "mathematics"
    SCIENCE = "science"
    ENGINEERING = "engineering"
    MEDICINE = "medicine"
    FINANCE = "finance"
    LEGAL = "legal"
    EDUCATION = "education"
    SPORTS = "sports"
    GAMES = "games"
    GENERAL = "general"

class ExpertiseLevel(Enum):
    NOVICE = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    MASTER = 5

class OperationMode(Enum):
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"
    OPTIMIZATION = "optimization"
    SIMULATION = "simulation"
    PREDICTION = "prediction"
    CLASSIFICATION = "classification"
    RECOMMENDATION = "recommendation"

class ProcessingStrategy(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    COLLABORATIVE = "collaborative"
    COMPETITIVE = "competitive"

@dataclass
class DomainExpertise:
    domain: DomainType
    expertise_level: ExpertiseLevel
    specializations: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    confidence_threshold: float = 0.7
    processing_time_avg: float = 0.0
    success_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)

@dataclass
class DomainOperation:
    operation_id: str
    domain: DomainType
    operation_mode: OperationMode
    input_data: Any
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    requirements: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    timeout: float = 60.0
    expected_expertise_level: ExpertiseLevel = ExpertiseLevel.INTERMEDIATE
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None

@dataclass
class DomainKnowledge:
    domain: DomainType
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    concepts: List[str] = field(default_factory=list)
    relationships: Dict[str, List[str]] = field(default_factory=dict)
    rules: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    heuristics: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

class DomainExpert(ABC):
    """Abstract base class for domain experts"""
    
    def __init__(self, domain: DomainType, expertise_level: ExpertiseLevel):
        self.domain = domain
        self.expertise_level = expertise_level
        self.specializations = []
        self.performance_history = deque(maxlen=100)
        self.knowledge_base = {}
    
    @abstractmethod
    def handle_operation(self, operation: DomainOperation) -> Dict[str, Any]:
        """Handle domain-specific operation"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities"""
        pass
    
    @abstractmethod
    def assess_capability(self, operation: DomainOperation) -> float:
        """Assess capability to handle operation (0-1 score)"""
        pass
    
    def update_performance(self, operation: DomainOperation, result: Dict[str, Any]):
        """Update performance metrics"""
        execution_time = (operation.completed_at or time.time()) - (operation.started_at or time.time())
        success = "error" not in result
        
        self.performance_history.append({
            'operation_id': operation.operation_id,
            'execution_time': execution_time,
            'success': success,
            'timestamp': time.time()
        })

class MathematicsExpert(DomainExpert):
    """Mathematics domain expert"""
    
    def __init__(self):
        super().__init__(DomainType.MATHEMATICS, ExpertiseLevel.EXPERT)
        self.specializations = ["algebra", "calculus", "statistics", "geometry", "optimization"]
        self.knowledge_base = {
            "constants": {"pi": 3.14159, "e": 2.71828, "golden_ratio": 1.618},
            "formulas": {
                "quadratic": "(-b ± √(b²-4ac)) / 2a",
                "derivative_power": "d/dx[x^n] = n*x^(n-1)",
                "integral_power": "∫x^n dx = x^(n+1)/(n+1) + C"
            }
        }
    
    def handle_operation(self, operation: DomainOperation) -> Dict[str, Any]:
        """Handle mathematics operations"""
        try:
            mode = operation.operation_mode
            data = operation.input_data
            params = operation.parameters
            
            if mode == OperationMode.ANALYSIS:
                return self._analyze_mathematical_data(data, params)
            elif mode == OperationMode.OPTIMIZATION:
                return self._optimize_mathematical_function(data, params)
            elif mode == OperationMode.EVALUATION:
                return self._evaluate_mathematical_expression(data, params)
            elif mode == OperationMode.SIMULATION:
                return self._simulate_mathematical_model(data, params)
            else:
                return {"error": f"Unsupported operation mode: {mode.value}"}
                
        except Exception as e:
            logger.error(f"Mathematics operation failed: {e}")
            return {"error": str(e)}
    
    def _analyze_mathematical_data(self, data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze mathematical data"""
        if isinstance(data, (list, np.ndarray)):
            data_array = np.array(data)
            
            analysis = {
                "descriptive_statistics": {
                    "mean": float(np.mean(data_array)),
                    "median": float(np.median(data_array)),
                    "std": float(np.std(data_array)),
                    "variance": float(np.var(data_array)),
                    "min": float(np.min(data_array)),
                    "max": float(np.max(data_array)),
                    "range": float(np.max(data_array) - np.min(data_array))
                },
                "distribution_analysis": {
                    "skewness": self._calculate_skewness(data_array),
                    "kurtosis": self._calculate_kurtosis(data_array)
                },
                "correlation_analysis": self._analyze_correlations(data_array) if data_array.ndim > 1 else None
            }
            
            return {"analysis": analysis, "data_type": "numerical"}
        else:
            return {"error": "Unsupported data type for mathematical analysis"}
    
    def _optimize_mathematical_function(self, data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize mathematical function"""
        from scipy.optimize import minimize, minimize_scalar
        
        function_type = params.get("function_type", "quadratic")
        bounds = params.get("bounds")
        method = params.get("method", "BFGS")
        
        if function_type == "quadratic" and isinstance(data, dict):
            # Quadratic function: ax² + bx + c
            a = data.get("a", 1)
            b = data.get("b", 0)
            c = data.get("c", 0)
            
            def quadratic(x):
                return a * x**2 + b * x + c
            
            if bounds:
                result = minimize_scalar(quadratic, bounds=bounds, method='bounded')
            else:
                # Analytical solution for unconstrained quadratic
                optimal_x = -b / (2 * a) if a != 0 else 0
                optimal_value = quadratic(optimal_x)
                result = {
                    'x': optimal_x,
                    'fun': optimal_value,
                    'success': True
                }
            
            return {
                "optimization_result": {
                    "optimal_x": result.get('x', result.get('fun')),
                    "optimal_value": result.get('fun'),
                    "success": result.get('success', True),
                    "method": "analytical" if not bounds else "numerical"
                }
            }
        
        return {"error": "Unsupported function type or data format"}
    
    def _evaluate_mathematical_expression(self, data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate mathematical expressions"""
        if isinstance(data, str):
            # Safe mathematical expression evaluation
            allowed_names = {
                "pi": np.pi,
                "e": np.e,
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "log": np.log,
                "exp": np.exp,
                "sqrt": np.sqrt,
                "abs": abs,
                "pow": pow
            }
            
            # Add variables from parameters
            variables = params.get("variables", {})
            allowed_names.update(variables)
            
            try:
                result = eval(data, {"__builtins__": {}}, allowed_names)
                return {
                    "evaluation_result": {
                        "expression": data,
                        "result": float(result) if isinstance(result, (int, float, np.number)) else str(result),
                        "variables_used": list(variables.keys())
                    }
                }
            except Exception as e:
                return {"error": f"Expression evaluation failed: {e}"}
        
        return {"error": "Expression must be a string"}
    
    def _simulate_mathematical_model(self, data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate mathematical models"""
        model_type = params.get("model_type", "linear")
        time_points = params.get("time_points", 100)
        initial_conditions = params.get("initial_conditions", [1.0])
        
        if model_type == "linear_growth":
            rate = data.get("growth_rate", 0.1) if isinstance(data, dict) else 0.1
            t = np.linspace(0, 10, time_points)
            y = initial_conditions[0] * (1 + rate * t)
            
            return {
                "simulation_result": {
                    "time_points": t.tolist(),
                    "values": y.tolist(),
                    "model_type": model_type,
                    "parameters": {"growth_rate": rate}
                }
            }
        
        elif model_type == "exponential_growth":
            rate = data.get("growth_rate", 0.1) if isinstance(data, dict) else 0.1
            t = np.linspace(0, 10, time_points)
            y = initial_conditions[0] * np.exp(rate * t)
            
            return {
                "simulation_result": {
                    "time_points": t.tolist(),
                    "values": y.tolist(),
                    "model_type": model_type,
                    "parameters": {"growth_rate": rate}
                }
            }
        
        return {"error": f"Unsupported model type: {model_type}"}
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _analyze_correlations(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze correlations in multidimensional data"""
        if data.ndim == 2:
            correlation_matrix = np.corrcoef(data.T)
            return {
                "correlation_matrix": correlation_matrix.tolist(),
                "max_correlation": float(np.max(np.abs(correlation_matrix[correlation_matrix != 1.0]))),
                "mean_correlation": float(np.mean(np.abs(correlation_matrix[correlation_matrix != 1.0])))
            }
        return {}
    
    def get_capabilities(self) -> List[str]:
        return [
            "statistical_analysis",
            "function_optimization",
            "expression_evaluation",
            "mathematical_modeling",
            "numerical_simulation",
            "correlation_analysis"
        ]
    
    def assess_capability(self, operation: DomainOperation) -> float:
        """Assess capability to handle operation"""
        if operation.domain != self.domain:
            return 0.0
        
        # Base capability score
        base_score = 0.8
        
        # Adjust based on operation mode
        mode_scores = {
            OperationMode.ANALYSIS: 0.9,
            OperationMode.OPTIMIZATION: 0.8,
            OperationMode.EVALUATION: 0.9,
            OperationMode.SIMULATION: 0.7,
            OperationMode.PREDICTION: 0.6
        }
        
        mode_score = mode_scores.get(operation.operation_mode, 0.5)
        
        # Adjust based on data complexity
        complexity_score = 1.0
        if hasattr(operation.input_data, '__len__'):
            if len(operation.input_data) > 1000:
                complexity_score = 0.9
            elif len(operation.input_data) > 10000:
                complexity_score = 0.8
        
        return base_score * mode_score * complexity_score

class SportsExpert(DomainExpert):
    """Sports domain expert (specialized for golf)"""
    
    def __init__(self):
        super().__init__(DomainType.SPORTS, ExpertiseLevel.EXPERT)
        self.specializations = ["golf", "performance_analysis", "strategy", "statistics"]
        self.knowledge_base = {
            "golf_rules": {
                "par_values": {"par_3": 3, "par_4": 4, "par_5": 5},
                "scoring": {"birdie": -1, "eagle": -2, "bogey": 1, "double_bogey": 2},
                "equipment": ["driver", "irons", "wedges", "putter"]
            },
            "performance_metrics": ["driving_distance", "fairway_accuracy", "greens_in_regulation", "putting_average"]
        }
    
    def handle_operation(self, operation: DomainOperation) -> Dict[str, Any]:
        """Handle sports operations"""
        try:
            mode = operation.operation_mode
            data = operation.input_data
            params = operation.parameters
            
            if mode == OperationMode.ANALYSIS:
                return self._analyze_sports_performance(data, params)
            elif mode == OperationMode.PREDICTION:
                return self._predict_sports_outcome(data, params)
            elif mode == OperationMode.RECOMMENDATION:
                return self._recommend_strategy(data, params)
            elif mode == OperationMode.EVALUATION:
                return self._evaluate_performance(data, params)
            else:
                return {"error": f"Unsupported operation mode: {mode.value}"}
                
        except Exception as e:
            logger.error(f"Sports operation failed: {e}")
            return {"error": str(e)}
    
    def _analyze_sports_performance(self, data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sports performance data"""
        sport = params.get("sport", "golf")
        
        if sport == "golf" and isinstance(data, dict):
            return self._analyze_golf_performance(data, params)
        elif isinstance(data, (list, np.ndarray)):
            return self._analyze_general_performance(data, params)
        else:
            return {"error": "Unsupported data format for sports analysis"}
    
    def _analyze_golf_performance(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze golf performance data"""
        analysis = {}
        
        # Scoring analysis
        if "scores" in data:
            scores = np.array(data["scores"])
            analysis["scoring"] = {
                "average_score": float(np.mean(scores)),
                "best_score": int(np.min(scores)),
                "worst_score": int(np.max(scores)),
                "consistency": float(np.std(scores)),
                "rounds_analyzed": len(scores)
            }
        
        # Performance metrics analysis
        metrics = ["driving_distance", "fairway_accuracy", "greens_in_regulation", "putting_average"]
        for metric in metrics:
            if metric in data:
                values = np.array(data[metric])
                analysis[metric] = {
                    "average": float(np.mean(values)),
                    "trend": self._calculate_trend(values),
                    "improvement_potential": self._assess_improvement_potential(values, metric)
                }
        
        # Strengths and weaknesses
        analysis["assessment"] = self._assess_golf_strengths_weaknesses(data)
        
        return {"analysis": analysis, "sport": "golf"}
    
    def _predict_sports_outcome(self, data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict sports outcomes"""
        sport = params.get("sport", "golf")
        
        if sport == "golf":
            return self._predict_golf_score(data, params)
        else:
            return {"error": f"Prediction not supported for sport: {sport}"}
    
    def _predict_golf_score(self, data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Predict golf score"""
        if not isinstance(data, dict):
            return {"error": "Golf data must be a dictionary"}
        
        # Simple prediction based on historical performance
        historical_scores = data.get("historical_scores", [])
        if not historical_scores:
            return {"error": "Historical scores required for prediction"}
        
        recent_scores = historical_scores[-5:] if len(historical_scores) >= 5 else historical_scores
        course_difficulty = params.get("course_difficulty", "medium")
        weather_conditions = params.get("weather_conditions", "good")
        
        # Base prediction from recent performance
        base_prediction = np.mean(recent_scores)
        
        # Adjustments
        difficulty_adjustment = {"easy": -2, "medium": 0, "hard": 3}.get(course_difficulty, 0)
        weather_adjustment = {"excellent": -1, "good": 0, "poor": 2, "bad": 4}.get(weather_conditions, 0)
        
        predicted_score = base_prediction + difficulty_adjustment + weather_adjustment
        
        # Confidence based on consistency
        consistency = np.std(recent_scores)
        confidence = max(0.1, 1.0 - (consistency / 10.0))
        
        return {
            "prediction": {
                "predicted_score": round(predicted_score, 1),
                "confidence": round(confidence, 2),
                "range": (round(predicted_score - consistency, 1), round(predicted_score + consistency, 1)),
                "factors": {
                    "base_performance": base_prediction,
                    "course_difficulty": difficulty_adjustment,
                    "weather_conditions": weather_adjustment
                }
            }
        }
    
    def _recommend_strategy(self, data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend sports strategy"""
        sport = params.get("sport", "golf")
        
        if sport == "golf":
            return self._recommend_golf_strategy(data, params)
        else:
            return {"error": f"Strategy recommendation not supported for sport: {sport}"}
    
    def _recommend_golf_strategy(self, data: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend golf strategy"""
        if not isinstance(data, dict):
            return {"error": "Golf data must be a dictionary"}
        
        recommendations = []
        
        # Analyze weaknesses and suggest improvements
        if "fairway_accuracy" in data:
            accuracy = np.mean(data["fairway_accuracy"])
            if accuracy < 0.6:
                recommendations.append({
                    "area": "driving",
                    "recommendation": "Focus on accuracy over distance. Consider using less lofted drivers or fairway woods off the tee.",
                    "priority": "high"
                })
        
        if "greens_in_regulation" in data:
            gir = np.mean(data["greens_in_regulation"])
            if gir < 0.5:
                recommendations.append({
                    "area": "approach_shots",
                    "recommendation": "Work on iron accuracy and course management. Focus on hitting greens rather than pin hunting.",
                    "priority": "high"
                })
        
        if "putting_average" in data:
            putting = np.mean(data["putting_average"])
            if putting > 2.0:
                recommendations.append({
                    "area": "putting",
                    "recommendation": "Practice distance control and read greens more carefully. Focus on lag putting for long putts.",
                    "priority": "medium"
                })
        
        # Course-specific strategy
        course_type = params.get("course_type", "parkland")
        if course_type == "links":
            recommendations.append({
                "area": "course_strategy",
                "recommendation": "Play more conservatively in windy conditions. Use lower ball flight and bump-and-run shots around greens.",
                "priority": "medium"
            })
        
        return {"recommendations": recommendations, "sport": "golf"}
    
    def _calculate_trend(self, values: np.ndarray) -> str:
        """Calculate trend in performance data"""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _assess_improvement_potential(self, values: np.ndarray, metric: str) -> str:
        """Assess improvement potential for a metric"""
        current_avg = np.mean(values)
        
        # Benchmark values for golf metrics
        benchmarks = {
            "driving_distance": {"amateur": 230, "good": 250, "excellent": 280},
            "fairway_accuracy": {"amateur": 0.5, "good": 0.65, "excellent": 0.75},
            "greens_in_regulation": {"amateur": 0.4, "good": 0.55, "excellent": 0.7},
            "putting_average": {"amateur": 2.2, "good": 1.9, "excellent": 1.7}
        }
        
        if metric in benchmarks:
            bench = benchmarks[metric]
            if metric == "putting_average":  # Lower is better
                if current_avg > bench["amateur"]:
                    return "high"
                elif current_avg > bench["good"]:
                    return "medium"
                else:
                    return "low"
            else:  # Higher is better
                if current_avg < bench["amateur"]:
                    return "high"
                elif current_avg < bench["good"]:
                    return "medium"
                else:
                    return "low"
        
        return "unknown"
    
    def _assess_golf_strengths_weaknesses(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess golf strengths and weaknesses"""
        strengths = []
        weaknesses = []
        
        # Analyze each aspect
        if "driving_distance" in data and np.mean(data["driving_distance"]) > 250:
            strengths.append("driving_distance")
        
        if "fairway_accuracy" in data and np.mean(data["fairway_accuracy"]) > 0.65:
            strengths.append("fairway_accuracy")
        elif "fairway_accuracy" in data and np.mean(data["fairway_accuracy"]) < 0.5:
            weaknesses.append("fairway_accuracy")
        
        if "greens_in_regulation" in data and np.mean(data["greens_in_regulation"]) > 0.55:
            strengths.append("greens_in_regulation")
        elif "greens_in_regulation" in data and np.mean(data["greens_in_regulation"]) < 0.4:
            weaknesses.append("greens_in_regulation")
        
        if "putting_average" in data and np.mean(data["putting_average"]) < 1.9:
            strengths.append("putting")
        elif "putting_average" in data and np.mean(data["putting_average"]) > 2.2:
            weaknesses.append("putting")
        
        return {"strengths": strengths, "weaknesses": weaknesses}
    
    def get_capabilities(self) -> List[str]:
        return [
            "performance_analysis",
            "outcome_prediction",
            "strategy_recommendation",
            "weakness_identification",
            "trend_analysis",
            "golf_expertise"
        ]
    
    def assess_capability(self, operation: DomainOperation) -> float:
        """Assess capability to handle operation"""
        if operation.domain != self.domain:
            return 0.0
        
        # Check if it's golf-related
        sport = operation.parameters.get("sport", "")
        golf_keywords = ["golf", "fairway", "green", "putting", "birdie", "eagle", "par"]
        
        # More specific golf detection - check for "golf" explicitly or multiple golf terms
        input_str = str(operation.input_data).lower()
        params_str = str(operation.parameters).lower()
        
        is_golf_related = (sport == "golf" or 
                          "golf" in input_str or
                          "golf" in params_str or
                          sum(keyword in input_str for keyword in golf_keywords) >= 2 or
                          sum(keyword in params_str for keyword in golf_keywords) >= 2)
        
        if is_golf_related:
            return 0.9
        else:
            return 0.6  # General sports capability

class GeneralExpert(DomainExpert):
    """General domain expert for unspecialized operations"""
    
    def __init__(self):
        super().__init__(DomainType.GENERAL, ExpertiseLevel.INTERMEDIATE)
        self.specializations = ["general_analysis", "data_processing", "basic_operations"]
    
    def handle_operation(self, operation: DomainOperation) -> Dict[str, Any]:
        """Handle general operations"""
        try:
            mode = operation.operation_mode
            data = operation.input_data
            
            if mode == OperationMode.ANALYSIS:
                return self._general_analysis(data)
            elif mode == OperationMode.EVALUATION:
                return self._general_evaluation(data)
            else:
                return {"result": f"General handling of {mode.value} operation", "data_type": type(data).__name__}
                
        except Exception as e:
            logger.error(f"General operation failed: {e}")
            return {"error": str(e)}
    
    def _general_analysis(self, data: Any) -> Dict[str, Any]:
        """Perform general data analysis"""
        analysis = {
            "data_type": type(data).__name__,
            "data_size": len(data) if hasattr(data, '__len__') else 1
        }
        
        if isinstance(data, (list, tuple)) and all(isinstance(x, (int, float)) for x in data):
            data_array = np.array(data)
            analysis.update({
                "numeric_analysis": {
                    "mean": float(np.mean(data_array)),
                    "std": float(np.std(data_array)),
                    "min": float(np.min(data_array)),
                    "max": float(np.max(data_array))
                }
            })
        
        return {"analysis": analysis}
    
    def _general_evaluation(self, data: Any) -> Dict[str, Any]:
        """Perform general evaluation"""
        return {
            "evaluation": {
                "data_provided": data is not None,
                "data_type": type(data).__name__,
                "complexity": "simple" if isinstance(data, (int, float, str)) else "complex"
            }
        }
    
    def get_capabilities(self) -> List[str]:
        return ["general_analysis", "data_inspection", "basic_evaluation"]
    
    def assess_capability(self, operation: DomainOperation) -> float:
        """Assess capability to handle operation"""
        # General expert can handle any operation but with lower capability
        return 0.4

class DomainOrchestrator:
    def __init__(self, brain_instance=None, config: Optional[Dict] = None):
        self.brain = brain_instance
        self.config = config or {}
        
        # Core state management
        self._lock = RLock()
        self._domain_experts: Dict[DomainType, List[DomainExpert]] = defaultdict(list)
        self._domain_knowledge: Dict[DomainType, DomainKnowledge] = {}
        self._active_operations: Dict[str, DomainOperation] = {}
        self._completed_operations: Dict[str, DomainOperation] = {}
        self._operation_queue: deque = deque()
        
        # Expert registration and management
        self._expert_registry: Dict[str, DomainExpert] = {}
        self._expert_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Domain routing
        self._routing_strategy = ProcessingStrategy(
            self.config.get('routing_strategy', 'hierarchical')
        )
        
        # Knowledge management
        self._knowledge_update_frequency = self.config.get('knowledge_update_frequency', 3600)  # 1 hour
        self._knowledge_last_updated: Dict[DomainType, float] = {}
        
        # Performance tracking
        self._operation_history: deque = deque(maxlen=1000)
        self._domain_performance_metrics: Dict[DomainType, Dict[str, float]] = defaultdict(dict)
        
        # Initialize default experts
        self._initialize_default_experts()
        
        # Load domain knowledge
        self._load_domain_knowledge()
        
        logger.info("DomainOrchestrator initialized")
    
    def handle_domain_operation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for domain operations"""
        operation_type = parameters.get('operation', 'process')
        
        if operation_type == 'process':
            return self._process_domain_operation(parameters)
        elif operation_type == 'register_expert':
            return self._register_expert(parameters)
        elif operation_type == 'get_domain_status':
            return self._get_domain_status(parameters)
        elif operation_type == 'update_knowledge':
            return self._update_domain_knowledge(parameters)
        elif operation_type == 'analyze_expertise':
            return self._analyze_expertise(parameters)
        elif operation_type == 'get_recommendations':
            return self._get_domain_recommendations(parameters)
        else:
            logger.warning(f"Unknown operation: {operation_type}")
            return {"error": f"Unknown operation: {operation_type}"}
    
    def _process_domain_operation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process a domain-specific operation"""
        try:
            # Create domain operation
            operation = self._create_domain_operation(parameters)
            
            # Route to appropriate expert
            expert = self._route_to_expert(operation)
            
            if not expert:
                return {"error": "No suitable expert found for operation"}
            
            # Store active operation
            with self._lock:
                self._active_operations[operation.operation_id] = operation
            
            # Execute operation
            start_time = time.time()
            operation.started_at = start_time
            
            # Store expert type in operation metadata for performance tracking
            operation.metadata['expert_type'] = type(expert).__name__
            
            try:
                result = expert.handle_operation(operation)
                operation.status = "completed"
                operation.result = result
                
            except Exception as e:
                operation.status = "failed"
                operation.error = str(e)
                result = {"error": str(e)}
            
            operation.completed_at = time.time()
            
            # Update expert performance
            expert.update_performance(operation, result)
            
            # Move to completed operations
            with self._lock:
                if operation.operation_id in self._active_operations:
                    del self._active_operations[operation.operation_id]
                self._completed_operations[operation.operation_id] = operation
            
            # Update performance metrics
            self._update_domain_performance(operation)
            
            # Add to history
            self._operation_history.append({
                'operation_id': operation.operation_id,
                'domain': operation.domain.value,
                'mode': operation.operation_mode.value,
                'execution_time': operation.completed_at - operation.started_at,
                'success': operation.status == "completed",
                'expert_type': type(expert).__name__
            })
            
            logger.info(f"Domain operation {operation.operation_id} completed by {type(expert).__name__}")
            
            return {
                "operation_id": operation.operation_id,
                "domain": operation.domain.value,
                "status": operation.status,
                "result": result,
                "expert_used": type(expert).__name__,
                "execution_time": operation.completed_at - operation.started_at
            }
            
        except Exception as e:
            logger.error(f"Domain operation processing failed: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def _create_domain_operation(self, parameters: Dict[str, Any]) -> DomainOperation:
        """Create domain operation from parameters"""
        # Use time with microseconds and a counter for guaranteed uniqueness
        if not hasattr(self, '_operation_counter'):
            self._operation_counter = 0
        with self._lock:
            self._operation_counter += 1
            counter = self._operation_counter
        operation_id = parameters.get('operation_id', f"domain_op_{int(time.time() * 1000000)}_{counter}")
        domain = DomainType(parameters.get('domain', 'general'))
        operation_mode = OperationMode(parameters.get('operation_mode', 'analysis'))
        input_data = parameters.get('input_data')
        
        return DomainOperation(
            operation_id=operation_id,
            domain=domain,
            operation_mode=operation_mode,
            input_data=input_data,
            parameters=parameters.get('parameters', {}),
            constraints=parameters.get('constraints', {}),
            requirements=parameters.get('requirements', {}),
            priority=parameters.get('priority', 1),
            timeout=parameters.get('timeout', 60.0),
            expected_expertise_level=ExpertiseLevel(parameters.get('expertise_level', 2))
        )
    
    def _route_to_expert(self, operation: DomainOperation) -> Optional[DomainExpert]:
        """Route operation to the most suitable expert"""
        # Get experts for the domain
        domain_experts = self._domain_experts.get(operation.domain, [])
        
        # If no domain-specific experts, try general experts
        if not domain_experts:
            domain_experts = self._domain_experts.get(DomainType.GENERAL, [])
        
        if not domain_experts:
            logger.warning(f"No experts available for domain {operation.domain.value}")
            return None
        
        # Assess capability of each expert
        expert_scores = []
        for expert in domain_experts:
            capability_score = expert.assess_capability(operation)
            
            # Adjust score based on expert performance
            expert_id = f"{type(expert).__name__}_{expert.domain.value}"
            performance = self._expert_performance.get(expert_id, {})
            success_rate = performance.get('success_rate', 0.5)
            avg_time = performance.get('avg_execution_time', 1.0)
            
            # Combine capability with performance (favor higher success rate and lower time)
            adjusted_score = capability_score * success_rate * (1.0 / (1.0 + avg_time))
            
            expert_scores.append((expert, adjusted_score))
        
        # Select expert with highest score
        if expert_scores:
            expert_scores.sort(key=lambda x: x[1], reverse=True)
            selected_expert = expert_scores[0][0]
            
            logger.debug(f"Selected expert {type(selected_expert).__name__} for operation {operation.operation_id}")
            return selected_expert
        
        return None
    
    def _initialize_default_experts(self):
        """Initialize default domain experts"""
        # Mathematics expert
        math_expert = MathematicsExpert()
        self._register_domain_expert(math_expert)
        
        # Sports expert (specialized for golf)
        sports_expert = SportsExpert()
        self._register_domain_expert(sports_expert)
        
        # General expert
        general_expert = GeneralExpert()
        self._register_domain_expert(general_expert)
        
        logger.info("Default domain experts initialized")
    
    def _register_domain_expert(self, expert: DomainExpert):
        """Register a domain expert"""
        with self._lock:
            self._domain_experts[expert.domain].append(expert)
            expert_id = f"{type(expert).__name__}_{expert.domain.value}"
            self._expert_registry[expert_id] = expert
            
            # Initialize performance tracking
            self._expert_performance[expert_id] = {
                'operations_handled': 0,
                'success_rate': 0.0,
                'avg_execution_time': 0.0,
                'last_updated': time.time()
            }
    
    def _register_expert(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new expert"""
        try:
            expert_class_name = parameters.get('expert_class')
            domain = DomainType(parameters.get('domain', 'general'))
            expertise_level = ExpertiseLevel(parameters.get('expertise_level', 2))
            
            if not expert_class_name:
                return {"error": "Expert class name required"}
            
            # This is a simplified registration - in practice would need more sophisticated
            # expert instantiation and validation
            
            return {
                "status": "registration_pending",
                "expert_class": expert_class_name,
                "domain": domain.value,
                "message": "Expert registration requires manual implementation"
            }
            
        except Exception as e:
            logger.error(f"Expert registration failed: {e}")
            return {"error": str(e)}
    
    def _load_domain_knowledge(self):
        """Load domain-specific knowledge"""
        # Initialize domain knowledge for each domain
        for domain_type in DomainType:
            knowledge = DomainKnowledge(domain=domain_type)
            
            if domain_type == DomainType.MATHEMATICS:
                knowledge.concepts = ["algebra", "calculus", "statistics", "geometry", "optimization"]
                knowledge.rules = [
                    "derivative_of_constant_is_zero",
                    "chain_rule_for_derivatives",
                    "fundamental_theorem_of_calculus"
                ]
                knowledge.facts = [
                    "π ≈ 3.14159",
                    "e ≈ 2.71828",
                    "golden_ratio ≈ 1.618"
                ]
                
            elif domain_type == DomainType.SPORTS:
                knowledge.concepts = ["performance", "strategy", "statistics", "training"]
                knowledge.rules = [
                    "golf_par_scoring_system",
                    "lower_score_is_better_in_golf",
                    "consistency_improves_performance"
                ]
                knowledge.facts = [
                    "golf_hole_par_ranges_from_3_to_5",
                    "professional_golf_rounds_are_18_holes",
                    "driving_distance_affects_scoring"
                ]
            
            self._domain_knowledge[domain_type] = knowledge
            self._knowledge_last_updated[domain_type] = time.time()
    
    def _get_domain_status(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get status of domain operations"""
        domain_filter = parameters.get('domain')
        
        with self._lock:
            status = {
                "total_experts": sum(len(experts) for experts in self._domain_experts.values()),
                "active_operations": len(self._active_operations),
                "completed_operations": len(self._completed_operations),
                "domains_available": [domain.value for domain in self._domain_experts.keys()]
            }
            
            # Domain-specific status
            if domain_filter:
                try:
                    domain_type = DomainType(domain_filter)
                    experts = self._domain_experts.get(domain_type, [])
                    status["domain_specific"] = {
                        "domain": domain_filter,
                        "expert_count": len(experts),
                        "expert_types": [type(expert).__name__ for expert in experts],
                        "performance_metrics": self._domain_performance_metrics.get(domain_type, {})
                    }
                except ValueError:
                    status["error"] = f"Invalid domain: {domain_filter}"
            
            # Expert performance summary
            status["expert_performance"] = {}
            for expert_id, performance in self._expert_performance.items():
                status["expert_performance"][expert_id] = {
                    "operations_handled": performance["operations_handled"],
                    "success_rate": performance["success_rate"],
                    "avg_execution_time": performance["avg_execution_time"]
                }
        
        return status
    
    def _update_domain_knowledge(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Update domain knowledge"""
        try:
            domain = DomainType(parameters.get('domain', 'general'))
            knowledge_updates = parameters.get('knowledge_updates', {})
            
            if domain not in self._domain_knowledge:
                self._domain_knowledge[domain] = DomainKnowledge(domain=domain)
            
            knowledge = self._domain_knowledge[domain]
            
            # Update different types of knowledge
            if 'concepts' in knowledge_updates:
                new_concepts = knowledge_updates['concepts']
                knowledge.concepts.extend([c for c in new_concepts if c not in knowledge.concepts])
            
            if 'rules' in knowledge_updates:
                new_rules = knowledge_updates['rules']
                knowledge.rules.extend([r for r in new_rules if r not in knowledge.rules])
            
            if 'facts' in knowledge_updates:
                new_facts = knowledge_updates['facts']
                knowledge.facts.extend([f for f in new_facts if f not in knowledge.facts])
            
            knowledge.last_updated = time.time()
            self._knowledge_last_updated[domain] = time.time()
            
            return {
                "domain": domain.value,
                "status": "updated",
                "concepts_count": len(knowledge.concepts),
                "rules_count": len(knowledge.rules),
                "facts_count": len(knowledge.facts)
            }
            
        except Exception as e:
            logger.error(f"Domain knowledge update failed: {e}")
            return {"error": str(e)}
    
    def _analyze_expertise(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze available expertise"""
        try:
            analysis = {
                "domains_covered": [],
                "expertise_gaps": [],
                "expert_distribution": {},
                "capability_matrix": {}
            }
            
            # Analyze domain coverage
            for domain_type in DomainType:
                experts = self._domain_experts.get(domain_type, [])
                analysis["domains_covered"].append({
                    "domain": domain_type.value,
                    "expert_count": len(experts),
                    "expertise_levels": [expert.expertise_level.value for expert in experts],
                    "specializations": [spec for expert in experts for spec in expert.specializations]
                })
                
                analysis["expert_distribution"][domain_type.value] = len(experts)
                
                # Identify gaps
                if not experts:
                    analysis["expertise_gaps"].append(f"No experts for {domain_type.value}")
                elif all(expert.expertise_level.value < 3 for expert in experts):
                    analysis["expertise_gaps"].append(f"Low expertise level in {domain_type.value}")
            
            # Capability matrix
            for domain_type, experts in self._domain_experts.items():
                capabilities = {}
                for expert in experts:
                    expert_capabilities = expert.get_capabilities()
                    for capability in expert_capabilities:
                        if capability not in capabilities:
                            capabilities[capability] = 0
                        capabilities[capability] += expert.expertise_level.value
                
                analysis["capability_matrix"][domain_type.value] = capabilities
            
            return analysis
            
        except Exception as e:
            logger.error(f"Expertise analysis failed: {e}")
            return {"error": str(e)}
    
    def _get_domain_recommendations(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get recommendations for domain improvements"""
        try:
            recommendations = []
            
            # Analyze operation history for patterns
            recent_operations = list(self._operation_history)[-100:]  # Last 100 operations
            
            if recent_operations:
                # Domain usage analysis
                domain_usage = defaultdict(int)
                domain_success_rates = defaultdict(list)
                
                for op in recent_operations:
                    domain_usage[op['domain']] += 1
                    domain_success_rates[op['domain']].append(op['success'])
                
                # Identify high-usage domains
                total_ops = len(recent_operations)
                for domain, count in domain_usage.items():
                    usage_rate = count / total_ops
                    success_rate = np.mean(domain_success_rates[domain])
                    
                    if usage_rate > 0.2:  # More than 20% of operations
                        recommendations.append({
                            "type": "high_usage_domain",
                            "domain": domain,
                            "message": f"Consider adding more experts for {domain} (usage: {usage_rate:.1%})",
                            "priority": "medium"
                        })
                    
                    if success_rate < 0.8:  # Less than 80% success rate
                        recommendations.append({
                            "type": "low_success_rate",
                            "domain": domain,
                            "message": f"Improve expertise in {domain} (success rate: {success_rate:.1%})",
                            "priority": "high"
                        })
            
            # Knowledge freshness analysis
            current_time = time.time()
            for domain, last_updated in self._knowledge_last_updated.items():
                if current_time - last_updated > self._knowledge_update_frequency:
                    recommendations.append({
                        "type": "knowledge_update",
                        "domain": domain.value,
                        "message": f"Knowledge base for {domain.value} needs updating",
                        "priority": "low"
                    })
            
            # Expert performance analysis
            for expert_id, performance in self._expert_performance.items():
                if performance['operations_handled'] > 10:  # Sufficient data
                    if performance['success_rate'] < 0.7:
                        recommendations.append({
                            "type": "expert_performance",
                            "expert": expert_id,
                            "message": f"Expert {expert_id} has low success rate: {performance['success_rate']:.1%}",
                            "priority": "medium"
                        })
                    
                    if performance['avg_execution_time'] > 5.0:  # More than 5 seconds average
                        recommendations.append({
                            "type": "expert_efficiency",
                            "expert": expert_id,
                            "message": f"Expert {expert_id} has high execution time: {performance['avg_execution_time']:.1f}s",
                            "priority": "low"
                        })
            
            return {
                "recommendations": recommendations,
                "analysis_period": f"Last {len(recent_operations)} operations",
                "total_recommendations": len(recommendations)
            }
            
        except Exception as e:
            logger.error(f"Domain recommendations failed: {e}")
            return {"error": str(e)}
    
    def _update_domain_performance(self, operation: DomainOperation):
        """Update domain performance metrics"""
        domain = operation.domain
        success = operation.status == "completed"
        execution_time = (operation.completed_at or time.time()) - (operation.started_at or time.time())
        
        # Update domain metrics
        metrics = self._domain_performance_metrics[domain]
        
        # Update operation count
        metrics['total_operations'] = metrics.get('total_operations', 0) + 1
        
        # Update success rate
        total_successes = metrics.get('total_successes', 0)
        if success:
            total_successes += 1
        metrics['total_successes'] = total_successes
        metrics['success_rate'] = total_successes / metrics['total_operations']
        
        # Update average execution time
        total_time = metrics.get('total_execution_time', 0.0)
        total_time += execution_time
        metrics['total_execution_time'] = total_time
        metrics['avg_execution_time'] = total_time / metrics['total_operations']
        
        # Update expert performance
        expert_type = operation.metadata.get('expert_type', 'unknown')
        if expert_type != 'unknown':
            expert_id = f"{expert_type}_{domain.value}"
            if expert_id in self._expert_performance:
                expert_perf = self._expert_performance[expert_id]
                expert_perf['operations_handled'] += 1
                
                # Update success rate
                current_successes = expert_perf['success_rate'] * (expert_perf['operations_handled'] - 1)
                if success:
                    current_successes += 1
                expert_perf['success_rate'] = current_successes / expert_perf['operations_handled']
                
                # Update execution time
                current_total_time = expert_perf['avg_execution_time'] * (expert_perf['operations_handled'] - 1)
                current_total_time += execution_time
                expert_perf['avg_execution_time'] = current_total_time / expert_perf['operations_handled']
                
                expert_perf['last_updated'] = time.time()
    
    def get_domain_experts(self, domain: DomainType) -> List[DomainExpert]:
        """Get experts for a specific domain"""
        return self._domain_experts.get(domain, [])
    
    def get_domain_knowledge(self, domain: DomainType) -> Optional[DomainKnowledge]:
        """Get knowledge for a specific domain"""
        return self._domain_knowledge.get(domain)
    
    def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific operation"""
        # Check active operations
        if operation_id in self._active_operations:
            operation = self._active_operations[operation_id]
            return {
                "operation_id": operation_id,
                "status": operation.status,
                "domain": operation.domain.value,
                "mode": operation.operation_mode.value,
                "started_at": operation.started_at,
                "progress": "active"
            }
        
        # Check completed operations
        if operation_id in self._completed_operations:
            operation = self._completed_operations[operation_id]
            return {
                "operation_id": operation_id,
                "status": operation.status,
                "domain": operation.domain.value,
                "mode": operation.operation_mode.value,
                "started_at": operation.started_at,
                "completed_at": operation.completed_at,
                "result": operation.result,
                "error": operation.error
            }
        
        return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        with self._lock:
            report = {
                "summary": {
                    "total_experts": sum(len(experts) for experts in self._domain_experts.values()),
                    "domains_covered": len(self._domain_experts),
                    "active_operations": len(self._active_operations),
                    "completed_operations": len(self._completed_operations),
                    "total_operations_processed": len(self._operation_history)
                },
                "domain_performance": dict(self._domain_performance_metrics),
                "expert_performance": dict(self._expert_performance),
                "recent_operations": list(self._operation_history)[-10:],  # Last 10 operations
                "knowledge_status": {
                    domain.value: {
                        "last_updated": self._knowledge_last_updated.get(domain, 0),
                        "concepts_count": len(knowledge.concepts),
                        "rules_count": len(knowledge.rules),
                        "facts_count": len(knowledge.facts)
                    }
                    for domain, knowledge in self._domain_knowledge.items()
                }
            }
            
            return report