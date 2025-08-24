#!/usr/bin/env python3
"""
Comprehensive test suite for DomainOrchestrator
Tests all domain-specific operations, expert routing, and performance tracking
"""

import unittest
import time
import numpy as np
from unittest.mock import MagicMock, patch, Mock
from typing import Dict, Any, List
from collections import deque
import threading

from orchestrators.domain_orchestrator import (
    DomainOrchestrator, DomainType, ExpertiseLevel, OperationMode,
    ProcessingStrategy, DomainExpertise, DomainOperation, DomainKnowledge,
    DomainExpert, MathematicsExpert, SportsExpert, GeneralExpert
)


class TestEnums(unittest.TestCase):
    """Test enum definitions"""
    
    def test_domain_type_enum(self):
        """Test DomainType enum values"""
        self.assertEqual(DomainType.MATHEMATICS.value, "mathematics")
        self.assertEqual(DomainType.SPORTS.value, "sports")
        self.assertEqual(DomainType.GENERAL.value, "general")
        self.assertIn(DomainType.SCIENCE, DomainType)
        self.assertIn(DomainType.ENGINEERING, DomainType)
    
    def test_expertise_level_enum(self):
        """Test ExpertiseLevel enum values"""
        self.assertEqual(ExpertiseLevel.NOVICE.value, 1)
        self.assertEqual(ExpertiseLevel.EXPERT.value, 4)
        self.assertEqual(ExpertiseLevel.MASTER.value, 5)
        self.assertTrue(ExpertiseLevel.EXPERT.value > ExpertiseLevel.INTERMEDIATE.value)
    
    def test_operation_mode_enum(self):
        """Test OperationMode enum values"""
        self.assertEqual(OperationMode.ANALYSIS.value, "analysis")
        self.assertEqual(OperationMode.OPTIMIZATION.value, "optimization")
        self.assertIn(OperationMode.PREDICTION, OperationMode)
        self.assertIn(OperationMode.SIMULATION, OperationMode)
    
    def test_processing_strategy_enum(self):
        """Test ProcessingStrategy enum values"""
        self.assertEqual(ProcessingStrategy.SEQUENTIAL.value, "sequential")
        self.assertEqual(ProcessingStrategy.PARALLEL.value, "parallel")
        self.assertIn(ProcessingStrategy.HIERARCHICAL, ProcessingStrategy)


class TestDataClasses(unittest.TestCase):
    """Test dataclass definitions"""
    
    def test_domain_expertise_creation(self):
        """Test DomainExpertise dataclass"""
        expertise = DomainExpertise(
            domain=DomainType.MATHEMATICS,
            expertise_level=ExpertiseLevel.EXPERT
        )
        
        self.assertEqual(expertise.domain, DomainType.MATHEMATICS)
        self.assertEqual(expertise.expertise_level, ExpertiseLevel.EXPERT)
        self.assertEqual(expertise.confidence_threshold, 0.7)
        self.assertEqual(expertise.specializations, [])
        self.assertIsInstance(expertise.performance_metrics, dict)
    
    def test_domain_operation_creation(self):
        """Test DomainOperation dataclass"""
        operation = DomainOperation(
            operation_id="test_op_1",
            domain=DomainType.SPORTS,
            operation_mode=OperationMode.ANALYSIS,
            input_data={"test": "data"}
        )
        
        self.assertEqual(operation.operation_id, "test_op_1")
        self.assertEqual(operation.domain, DomainType.SPORTS)
        self.assertEqual(operation.operation_mode, OperationMode.ANALYSIS)
        self.assertEqual(operation.priority, 1)
        self.assertEqual(operation.timeout, 60.0)
        self.assertEqual(operation.status, "pending")
        self.assertIsNone(operation.result)
        self.assertIsNone(operation.error)
    
    def test_domain_knowledge_creation(self):
        """Test DomainKnowledge dataclass"""
        knowledge = DomainKnowledge(
            domain=DomainType.SCIENCE,
            concepts=["physics", "chemistry"]
        )
        
        self.assertEqual(knowledge.domain, DomainType.SCIENCE)
        self.assertEqual(knowledge.concepts, ["physics", "chemistry"])
        self.assertEqual(knowledge.rules, [])
        self.assertIsInstance(knowledge.knowledge_base, dict)
        self.assertIsInstance(knowledge.last_updated, float)


class TestMathematicsExpert(unittest.TestCase):
    """Test MathematicsExpert functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.expert = MathematicsExpert()
    
    def test_initialization(self):
        """Test expert initialization"""
        self.assertEqual(self.expert.domain, DomainType.MATHEMATICS)
        self.assertEqual(self.expert.expertise_level, ExpertiseLevel.EXPERT)
        self.assertIn("algebra", self.expert.specializations)
        self.assertIn("calculus", self.expert.specializations)
        self.assertIn("constants", self.expert.knowledge_base)
    
    def test_handle_analysis_operation(self):
        """Test mathematical analysis operation"""
        operation = DomainOperation(
            operation_id="math_analysis_1",
            domain=DomainType.MATHEMATICS,
            operation_mode=OperationMode.ANALYSIS,
            input_data=[1, 2, 3, 4, 5]
        )
        
        result = self.expert.handle_operation(operation)
        
        self.assertIn("analysis", result)
        self.assertIn("descriptive_statistics", result["analysis"])
        stats = result["analysis"]["descriptive_statistics"]
        self.assertEqual(stats["mean"], 3.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 5.0)
    
    def test_handle_optimization_operation(self):
        """Test mathematical optimization operation"""
        operation = DomainOperation(
            operation_id="math_opt_1",
            domain=DomainType.MATHEMATICS,
            operation_mode=OperationMode.OPTIMIZATION,
            input_data={"a": 1, "b": -2, "c": 1},  # x^2 - 2x + 1 = (x-1)^2
            parameters={"function_type": "quadratic"}
        )
        
        result = self.expert.handle_operation(operation)
        
        self.assertIn("optimization_result", result)
        opt_result = result["optimization_result"]
        self.assertAlmostEqual(opt_result["optimal_x"], 1.0, places=5)
        self.assertAlmostEqual(opt_result["optimal_value"], 0.0, places=5)
        self.assertTrue(opt_result["success"])
    
    def test_handle_evaluation_operation(self):
        """Test mathematical expression evaluation"""
        operation = DomainOperation(
            operation_id="math_eval_1",
            domain=DomainType.MATHEMATICS,
            operation_mode=OperationMode.EVALUATION,
            input_data="2 + 2 * 3",
            parameters={"variables": {}}
        )
        
        result = self.expert.handle_operation(operation)
        
        self.assertIn("evaluation_result", result)
        eval_result = result["evaluation_result"]
        self.assertEqual(eval_result["result"], 8.0)  # 2 + 6 = 8
        self.assertEqual(eval_result["expression"], "2 + 2 * 3")
    
    def test_handle_simulation_operation(self):
        """Test mathematical simulation operation"""
        operation = DomainOperation(
            operation_id="math_sim_1",
            domain=DomainType.MATHEMATICS,
            operation_mode=OperationMode.SIMULATION,
            input_data={"growth_rate": 0.1},
            parameters={
                "model_type": "linear_growth",
                "time_points": 10,
                "initial_conditions": [1.0]
            }
        )
        
        result = self.expert.handle_operation(operation)
        
        self.assertIn("simulation_result", result)
        sim_result = result["simulation_result"]
        self.assertEqual(len(sim_result["time_points"]), 10)
        self.assertEqual(len(sim_result["values"]), 10)
        self.assertEqual(sim_result["model_type"], "linear_growth")
    
    def test_unsupported_operation_mode(self):
        """Test handling of unsupported operation mode"""
        operation = DomainOperation(
            operation_id="math_unsupported",
            domain=DomainType.MATHEMATICS,
            operation_mode=OperationMode.CLASSIFICATION,
            input_data=[1, 2, 3]
        )
        
        result = self.expert.handle_operation(operation)
        
        self.assertIn("error", result)
        self.assertIn("Unsupported", result["error"])
    
    def test_get_capabilities(self):
        """Test expert capabilities listing"""
        capabilities = self.expert.get_capabilities()
        
        self.assertIsInstance(capabilities, list)
        self.assertIn("statistical_analysis", capabilities)
        self.assertIn("function_optimization", capabilities)
        self.assertIn("expression_evaluation", capabilities)
    
    def test_assess_capability(self):
        """Test capability assessment"""
        operation = DomainOperation(
            operation_id="test",
            domain=DomainType.MATHEMATICS,
            operation_mode=OperationMode.ANALYSIS,
            input_data=[1, 2, 3]
        )
        
        score = self.expert.assess_capability(operation)
        
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Wrong domain should return 0
        operation.domain = DomainType.SPORTS
        score = self.expert.assess_capability(operation)
        self.assertEqual(score, 0.0)
    
    def test_update_performance(self):
        """Test performance update"""
        operation = DomainOperation(
            operation_id="test",
            domain=DomainType.MATHEMATICS,
            operation_mode=OperationMode.ANALYSIS,
            input_data=[1, 2, 3],
            started_at=time.time() - 1.0,
            completed_at=time.time()
        )
        
        self.expert.update_performance(operation, {"result": "success"})
        
        self.assertEqual(len(self.expert.performance_history), 1)
        perf = self.expert.performance_history[0]
        self.assertEqual(perf["operation_id"], "test")
        self.assertTrue(perf["success"])


class TestSportsExpert(unittest.TestCase):
    """Test SportsExpert functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.expert = SportsExpert()
    
    def test_initialization(self):
        """Test expert initialization"""
        self.assertEqual(self.expert.domain, DomainType.SPORTS)
        self.assertEqual(self.expert.expertise_level, ExpertiseLevel.EXPERT)
        self.assertIn("golf", self.expert.specializations)
        self.assertIn("golf_rules", self.expert.knowledge_base)
    
    def test_analyze_golf_performance(self):
        """Test golf performance analysis"""
        operation = DomainOperation(
            operation_id="golf_analysis_1",
            domain=DomainType.SPORTS,
            operation_mode=OperationMode.ANALYSIS,
            input_data={
                "scores": [72, 75, 73, 74, 71],
                "driving_distance": [250, 260, 255, 245, 265],
                "fairway_accuracy": [0.6, 0.65, 0.7, 0.55, 0.6],
                "putting_average": [1.8, 1.9, 1.85, 2.0, 1.95]
            },
            parameters={"sport": "golf"}
        )
        
        result = self.expert.handle_operation(operation)
        
        self.assertIn("analysis", result)
        self.assertIn("scoring", result["analysis"])
        self.assertIn("driving_distance", result["analysis"])
        self.assertEqual(result["sport"], "golf")
        
        scoring = result["analysis"]["scoring"]
        self.assertEqual(scoring["best_score"], 71)
        self.assertEqual(scoring["worst_score"], 75)
    
    def test_predict_golf_score(self):
        """Test golf score prediction"""
        operation = DomainOperation(
            operation_id="golf_predict_1",
            domain=DomainType.SPORTS,
            operation_mode=OperationMode.PREDICTION,
            input_data={"historical_scores": [72, 75, 73, 74, 71]},
            parameters={
                "sport": "golf",
                "course_difficulty": "medium",
                "weather_conditions": "good"
            }
        )
        
        result = self.expert.handle_operation(operation)
        
        self.assertIn("prediction", result)
        prediction = result["prediction"]
        self.assertIn("predicted_score", prediction)
        self.assertIn("confidence", prediction)
        self.assertIn("range", prediction)
        self.assertIsInstance(prediction["predicted_score"], (int, float))
        self.assertGreater(prediction["confidence"], 0.0)
        self.assertLessEqual(prediction["confidence"], 1.0)
    
    def test_recommend_golf_strategy(self):
        """Test golf strategy recommendation"""
        operation = DomainOperation(
            operation_id="golf_strategy_1",
            domain=DomainType.SPORTS,
            operation_mode=OperationMode.RECOMMENDATION,
            input_data={
                "fairway_accuracy": [0.4, 0.45, 0.42],  # Low accuracy
                "greens_in_regulation": [0.3, 0.35, 0.32],  # Low GIR
                "putting_average": [2.3, 2.2, 2.4]  # High putting average
            },
            parameters={"sport": "golf", "course_type": "links"}
        )
        
        result = self.expert.handle_operation(operation)
        
        self.assertIn("recommendations", result)
        self.assertEqual(result["sport"], "golf")
        recommendations = result["recommendations"]
        self.assertGreater(len(recommendations), 0)
        
        # Check for specific recommendations based on poor stats
        areas = [rec["area"] for rec in recommendations]
        self.assertIn("driving", areas)  # Due to low fairway accuracy
        self.assertIn("approach_shots", areas)  # Due to low GIR
    
    def test_unsupported_operation_mode(self):
        """Test handling of unsupported operation mode"""
        operation = DomainOperation(
            operation_id="sports_unsupported",
            domain=DomainType.SPORTS,
            operation_mode=OperationMode.OPTIMIZATION,
            input_data={"test": "data"}
        )
        
        result = self.expert.handle_operation(operation)
        
        self.assertIn("error", result)
        self.assertIn("Unsupported", result["error"])
    
    def test_assess_capability_golf(self):
        """Test capability assessment for golf operations"""
        operation = DomainOperation(
            operation_id="test",
            domain=DomainType.SPORTS,
            operation_mode=OperationMode.ANALYSIS,
            input_data={"golf_scores": [72, 73]},
            parameters={"sport": "golf"}
        )
        
        score = self.expert.assess_capability(operation)
        
        self.assertEqual(score, 0.9)  # High score for golf
    
    def test_assess_capability_general_sports(self):
        """Test capability assessment for non-golf sports"""
        operation = DomainOperation(
            operation_id="test",
            domain=DomainType.SPORTS,
            operation_mode=OperationMode.ANALYSIS,
            input_data={"data": [1, 2, 3]},
            parameters={"sport": "basketball"}
        )
        
        score = self.expert.assess_capability(operation)
        
        self.assertEqual(score, 0.6)  # Lower score for non-golf


class TestGeneralExpert(unittest.TestCase):
    """Test GeneralExpert functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.expert = GeneralExpert()
    
    def test_initialization(self):
        """Test expert initialization"""
        self.assertEqual(self.expert.domain, DomainType.GENERAL)
        self.assertEqual(self.expert.expertise_level, ExpertiseLevel.INTERMEDIATE)
        self.assertIn("general_analysis", self.expert.specializations)
    
    def test_general_analysis(self):
        """Test general analysis operation"""
        operation = DomainOperation(
            operation_id="general_1",
            domain=DomainType.GENERAL,
            operation_mode=OperationMode.ANALYSIS,
            input_data=[1, 2, 3, 4, 5]
        )
        
        result = self.expert.handle_operation(operation)
        
        self.assertIn("analysis", result)
        analysis = result["analysis"]
        self.assertEqual(analysis["data_type"], "list")
        self.assertEqual(analysis["data_size"], 5)
        self.assertIn("numeric_analysis", analysis)
    
    def test_general_evaluation(self):
        """Test general evaluation operation"""
        operation = DomainOperation(
            operation_id="general_2",
            domain=DomainType.GENERAL,
            operation_mode=OperationMode.EVALUATION,
            input_data="test string"
        )
        
        result = self.expert.handle_operation(operation)
        
        self.assertIn("evaluation", result)
        evaluation = result["evaluation"]
        self.assertTrue(evaluation["data_provided"])
        self.assertEqual(evaluation["data_type"], "str")
        self.assertEqual(evaluation["complexity"], "simple")
    
    def test_assess_capability(self):
        """Test capability assessment"""
        operation = DomainOperation(
            operation_id="test",
            domain=DomainType.GENERAL,
            operation_mode=OperationMode.ANALYSIS,
            input_data=[1, 2, 3]
        )
        
        score = self.expert.assess_capability(operation)
        
        self.assertEqual(score, 0.4)  # General expert has low base capability


class TestDomainOrchestratorInit(unittest.TestCase):
    """Test DomainOrchestrator initialization"""
    
    def test_basic_initialization(self):
        """Test basic orchestrator initialization"""
        orchestrator = DomainOrchestrator()
        
        self.assertIsNotNone(orchestrator._lock)
        self.assertIsInstance(orchestrator._domain_experts, dict)
        self.assertIsInstance(orchestrator._domain_knowledge, dict)
        self.assertIsInstance(orchestrator._active_operations, dict)
        self.assertIsInstance(orchestrator._operation_queue, deque)
    
    def test_initialization_with_config(self):
        """Test orchestrator initialization with config"""
        config = {
            'routing_strategy': 'parallel',
            'knowledge_update_frequency': 7200
        }
        orchestrator = DomainOrchestrator(config=config)
        
        self.assertEqual(orchestrator.config, config)
        self.assertEqual(orchestrator._knowledge_update_frequency, 7200)
    
    def test_default_experts_initialized(self):
        """Test that default experts are initialized"""
        orchestrator = DomainOrchestrator()
        
        # Check that experts exist for key domains
        self.assertIn(DomainType.MATHEMATICS, orchestrator._domain_experts)
        self.assertIn(DomainType.SPORTS, orchestrator._domain_experts)
        self.assertIn(DomainType.GENERAL, orchestrator._domain_experts)
        
        # Check expert types
        math_experts = orchestrator._domain_experts[DomainType.MATHEMATICS]
        self.assertTrue(any(isinstance(e, MathematicsExpert) for e in math_experts))
    
    def test_domain_knowledge_loaded(self):
        """Test that domain knowledge is loaded"""
        orchestrator = DomainOrchestrator()
        
        # Check knowledge exists for domains
        self.assertIn(DomainType.MATHEMATICS, orchestrator._domain_knowledge)
        self.assertIn(DomainType.SPORTS, orchestrator._domain_knowledge)
        
        # Check knowledge content
        math_knowledge = orchestrator._domain_knowledge[DomainType.MATHEMATICS]
        self.assertIn("algebra", math_knowledge.concepts)
        self.assertIn("calculus", math_knowledge.concepts)


class TestDomainOrchestratorOperations(unittest.TestCase):
    """Test DomainOrchestrator operation handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = DomainOrchestrator()
    
    def test_handle_process_operation(self):
        """Test processing a domain operation"""
        parameters = {
            'operation': 'process',
            'domain': 'mathematics',
            'operation_mode': 'analysis',
            'input_data': [1, 2, 3, 4, 5]
        }
        
        result = self.orchestrator.handle_domain_operation(parameters)
        
        self.assertIn("operation_id", result)
        self.assertIn("status", result)
        self.assertIn("result", result)
        self.assertIn("expert_used", result)
        self.assertEqual(result["domain"], "mathematics")
    
    def test_handle_get_domain_status(self):
        """Test getting domain status"""
        parameters = {'operation': 'get_domain_status'}
        
        result = self.orchestrator.handle_domain_operation(parameters)
        
        self.assertIn("total_experts", result)
        self.assertIn("active_operations", result)
        self.assertIn("domains_available", result)
        self.assertIsInstance(result["total_experts"], int)
        self.assertGreater(result["total_experts"], 0)
    
    def test_handle_get_domain_status_with_filter(self):
        """Test getting domain status with filter"""
        parameters = {
            'operation': 'get_domain_status',
            'domain': 'mathematics'
        }
        
        result = self.orchestrator.handle_domain_operation(parameters)
        
        self.assertIn("domain_specific", result)
        domain_info = result["domain_specific"]
        self.assertEqual(domain_info["domain"], "mathematics")
        self.assertIn("expert_count", domain_info)
        self.assertIn("expert_types", domain_info)
    
    def test_handle_update_knowledge(self):
        """Test updating domain knowledge"""
        parameters = {
            'operation': 'update_knowledge',
            'domain': 'mathematics',
            'knowledge_updates': {
                'concepts': ['topology', 'number_theory'],
                'rules': ['commutative_property'],
                'facts': ['sqrt(2) is irrational']
            }
        }
        
        result = self.orchestrator.handle_domain_operation(parameters)
        
        self.assertEqual(result["status"], "updated")
        self.assertEqual(result["domain"], "mathematics")
        self.assertIn("concepts_count", result)
        
        # Verify knowledge was actually updated
        knowledge = self.orchestrator.get_domain_knowledge(DomainType.MATHEMATICS)
        self.assertIn("topology", knowledge.concepts)
        self.assertIn("commutative_property", knowledge.rules)
    
    def test_handle_analyze_expertise(self):
        """Test analyzing expertise"""
        parameters = {'operation': 'analyze_expertise'}
        
        result = self.orchestrator.handle_domain_operation(parameters)
        
        self.assertIn("domains_covered", result)
        self.assertIn("expertise_gaps", result)
        self.assertIn("expert_distribution", result)
        self.assertIn("capability_matrix", result)
        
        # Check domains covered structure
        for domain_info in result["domains_covered"]:
            self.assertIn("domain", domain_info)
            self.assertIn("expert_count", domain_info)
            self.assertIn("expertise_levels", domain_info)
    
    def test_handle_get_recommendations(self):
        """Test getting domain recommendations"""
        # First, process some operations to generate history
        for i in range(5):
            params = {
                'operation': 'process',
                'domain': 'mathematics',
                'operation_mode': 'analysis',
                'input_data': [i, i+1, i+2]
            }
            self.orchestrator.handle_domain_operation(params)
        
        parameters = {'operation': 'get_recommendations'}
        
        result = self.orchestrator.handle_domain_operation(parameters)
        
        self.assertIn("recommendations", result)
        self.assertIn("analysis_period", result)
        self.assertIn("total_recommendations", result)
        self.assertIsInstance(result["recommendations"], list)
    
    def test_handle_unknown_operation(self):
        """Test handling unknown operation"""
        parameters = {'operation': 'unknown_operation'}
        
        result = self.orchestrator.handle_domain_operation(parameters)
        
        self.assertIn("error", result)
        self.assertIn("Unknown operation", result["error"])


class TestDomainOrchestratorRouting(unittest.TestCase):
    """Test expert routing functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = DomainOrchestrator()
    
    def test_route_to_mathematics_expert(self):
        """Test routing to mathematics expert"""
        operation = DomainOperation(
            operation_id="test",
            domain=DomainType.MATHEMATICS,
            operation_mode=OperationMode.ANALYSIS,
            input_data=[1, 2, 3]
        )
        
        expert = self.orchestrator._route_to_expert(operation)
        
        self.assertIsNotNone(expert)
        self.assertIsInstance(expert, MathematicsExpert)
    
    def test_route_to_sports_expert(self):
        """Test routing to sports expert"""
        operation = DomainOperation(
            operation_id="test",
            domain=DomainType.SPORTS,
            operation_mode=OperationMode.ANALYSIS,
            input_data={"golf": "data"},
            parameters={"sport": "golf"}
        )
        
        expert = self.orchestrator._route_to_expert(operation)
        
        self.assertIsNotNone(expert)
        self.assertIsInstance(expert, SportsExpert)
    
    def test_route_to_general_expert_for_unknown_domain(self):
        """Test routing to general expert for unhandled domain"""
        operation = DomainOperation(
            operation_id="test",
            domain=DomainType.LEGAL,  # No legal expert registered
            operation_mode=OperationMode.ANALYSIS,
            input_data={"data": "test"}
        )
        
        expert = self.orchestrator._route_to_expert(operation)
        
        # Should fall back to general expert
        self.assertIsNotNone(expert)
        self.assertIsInstance(expert, GeneralExpert)
    
    def test_expert_capability_assessment_affects_routing(self):
        """Test that capability assessment affects routing"""
        # Create operation that mathematics expert will score low on
        operation = DomainOperation(
            operation_id="test",
            domain=DomainType.MATHEMATICS,
            operation_mode=OperationMode.RECOMMENDATION,  # Not supported by MathematicsExpert
            input_data=[1, 2, 3]
        )
        
        # Get all experts for the domain
        experts = self.orchestrator._domain_experts[DomainType.MATHEMATICS]
        
        # Assess capabilities
        for expert in experts:
            score = expert.assess_capability(operation)
            # MathematicsExpert should have lower score for RECOMMENDATION
            if isinstance(expert, MathematicsExpert):
                self.assertLess(score, 0.5)


class TestDomainOrchestratorPerformance(unittest.TestCase):
    """Test performance tracking and metrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = DomainOrchestrator()
    
    def test_operation_history_tracking(self):
        """Test that operations are tracked in history"""
        initial_history_len = len(self.orchestrator._operation_history)
        
        # Process an operation
        parameters = {
            'operation': 'process',
            'domain': 'mathematics',
            'operation_mode': 'analysis',
            'input_data': [1, 2, 3]
        }
        
        result = self.orchestrator.handle_domain_operation(parameters)
        
        # Check history was updated
        self.assertEqual(len(self.orchestrator._operation_history), initial_history_len + 1)
        
        # Check history entry
        last_entry = self.orchestrator._operation_history[-1]
        self.assertEqual(last_entry["domain"], "mathematics")
        self.assertEqual(last_entry["mode"], "analysis")
        self.assertTrue(last_entry["success"])
    
    def test_domain_performance_metrics_update(self):
        """Test that domain performance metrics are updated"""
        # Process multiple operations
        for i in range(3):
            parameters = {
                'operation': 'process',
                'domain': 'mathematics',
                'operation_mode': 'analysis',
                'input_data': [i, i+1, i+2]
            }
            self.orchestrator.handle_domain_operation(parameters)
        
        # Check metrics
        metrics = self.orchestrator._domain_performance_metrics[DomainType.MATHEMATICS]
        
        self.assertEqual(metrics["total_operations"], 3)
        self.assertIn("success_rate", metrics)
        self.assertIn("avg_execution_time", metrics)
        self.assertGreater(metrics["success_rate"], 0)
    
    def test_expert_performance_tracking(self):
        """Test that expert performance is tracked"""
        # Process an operation
        parameters = {
            'operation': 'process',
            'domain': 'mathematics',
            'operation_mode': 'analysis',
            'input_data': [1, 2, 3]
        }
        
        self.orchestrator.handle_domain_operation(parameters)
        
        # Check expert performance
        expert_id = "MathematicsExpert_mathematics"
        self.assertIn(expert_id, self.orchestrator._expert_performance)
        
        perf = self.orchestrator._expert_performance[expert_id]
        self.assertGreater(perf["operations_handled"], 0)
    
    def test_get_performance_report(self):
        """Test getting comprehensive performance report"""
        # Process some operations first
        for i in range(2):
            parameters = {
                'operation': 'process',
                'domain': 'mathematics',
                'operation_mode': 'analysis',
                'input_data': [i, i+1]
            }
            self.orchestrator.handle_domain_operation(parameters)
        
        report = self.orchestrator.get_performance_report()
        
        self.assertIn("summary", report)
        self.assertIn("domain_performance", report)
        self.assertIn("expert_performance", report)
        self.assertIn("recent_operations", report)
        self.assertIn("knowledge_status", report)
        
        # Check summary
        summary = report["summary"]
        self.assertGreater(summary["total_experts"], 0)
        self.assertGreater(summary["total_operations_processed"], 0)


class TestDomainOrchestratorConcurrency(unittest.TestCase):
    """Test concurrent operation handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = DomainOrchestrator()
    
    def test_concurrent_operations(self):
        """Test handling multiple concurrent operations"""
        results = []
        errors = []
        
        def process_operation(domain, data):
            try:
                params = {
                    'operation': 'process',
                    'domain': domain,
                    'operation_mode': 'analysis',
                    'input_data': data
                }
                result = self.orchestrator.handle_domain_operation(params)
                results.append(result)
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for i in range(5):
            domain = 'mathematics' if i % 2 == 0 else 'general'
            data = [i, i+1, i+2]
            thread = threading.Thread(target=process_operation, args=(domain, data))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all to complete
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 5)
        
        for result in results:
            self.assertIn("status", result)
            self.assertIn("result", result)
    
    def test_thread_safe_operation_tracking(self):
        """Test that operation tracking is thread-safe"""
        import uuid
        
        def add_to_active(op_num):
            # Use a unique ID for each operation
            op = DomainOperation(
                operation_id=f"op_{op_num}_{uuid.uuid4().hex[:8]}",
                domain=DomainType.GENERAL,
                operation_mode=OperationMode.ANALYSIS,
                input_data="test"
            )
            with self.orchestrator._lock:
                self.orchestrator._active_operations[op.operation_id] = op
        
        threads = []
        for i in range(10):
            thread = threading.Thread(target=add_to_active, args=(i,))
            threads.append(thread)
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All operations should be added
        self.assertEqual(len(self.orchestrator._active_operations), 10)


class TestDomainOrchestratorStatusMethods(unittest.TestCase):
    """Test status and query methods"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = DomainOrchestrator()
    
    def test_get_domain_experts(self):
        """Test getting experts for a domain"""
        experts = self.orchestrator.get_domain_experts(DomainType.MATHEMATICS)
        
        self.assertIsInstance(experts, list)
        self.assertGreater(len(experts), 0)
        self.assertTrue(any(isinstance(e, MathematicsExpert) for e in experts))
    
    def test_get_domain_knowledge(self):
        """Test getting knowledge for a domain"""
        knowledge = self.orchestrator.get_domain_knowledge(DomainType.SPORTS)
        
        self.assertIsNotNone(knowledge)
        self.assertIsInstance(knowledge, DomainKnowledge)
        self.assertEqual(knowledge.domain, DomainType.SPORTS)
        self.assertIn("performance", knowledge.concepts)
    
    def test_get_operation_status_active(self):
        """Test getting status of active operation"""
        # Create and add an active operation
        op = DomainOperation(
            operation_id="test_active",
            domain=DomainType.GENERAL,
            operation_mode=OperationMode.ANALYSIS,
            input_data="test",
            started_at=time.time()
        )
        
        with self.orchestrator._lock:
            self.orchestrator._active_operations["test_active"] = op
        
        status = self.orchestrator.get_operation_status("test_active")
        
        self.assertIsNotNone(status)
        self.assertEqual(status["operation_id"], "test_active")
        self.assertEqual(status["progress"], "active")
    
    def test_get_operation_status_completed(self):
        """Test getting status of completed operation"""
        # Create and add a completed operation
        op = DomainOperation(
            operation_id="test_completed",
            domain=DomainType.GENERAL,
            operation_mode=OperationMode.ANALYSIS,
            input_data="test",
            started_at=time.time() - 1.0,
            completed_at=time.time(),
            status="completed",
            result={"success": True}
        )
        
        with self.orchestrator._lock:
            self.orchestrator._completed_operations["test_completed"] = op
        
        status = self.orchestrator.get_operation_status("test_completed")
        
        self.assertIsNotNone(status)
        self.assertEqual(status["operation_id"], "test_completed")
        self.assertEqual(status["status"], "completed")
        self.assertIn("result", status)
    
    def test_get_operation_status_not_found(self):
        """Test getting status of non-existent operation"""
        status = self.orchestrator.get_operation_status("non_existent")
        
        self.assertIsNone(status)


class TestDomainOrchestratorErrorHandling(unittest.TestCase):
    """Test error handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = DomainOrchestrator()
    
    def test_handle_invalid_domain(self):
        """Test handling invalid domain"""
        parameters = {
            'operation': 'process',
            'domain': 'invalid_domain',
            'operation_mode': 'analysis',
            'input_data': [1, 2, 3]
        }
        
        result = self.orchestrator.handle_domain_operation(parameters)
        
        self.assertIn("error", result)
    
    def test_handle_invalid_operation_mode(self):
        """Test handling invalid operation mode"""
        parameters = {
            'operation': 'process',
            'domain': 'mathematics',
            'operation_mode': 'invalid_mode',
            'input_data': [1, 2, 3]
        }
        
        result = self.orchestrator.handle_domain_operation(parameters)
        
        self.assertIn("error", result)
    
    def test_handle_expert_operation_failure(self):
        """Test handling when expert operation fails"""
        # Create operation that will fail (invalid expression)
        parameters = {
            'operation': 'process',
            'domain': 'mathematics',
            'operation_mode': 'evaluation',
            'input_data': "invalid expression @#$%",
            'parameters': {}
        }
        
        result = self.orchestrator.handle_domain_operation(parameters)
        
        # Should handle the error gracefully
        self.assertIn("result", result)
        self.assertIn("error", result["result"])
    
    def test_update_knowledge_invalid_domain(self):
        """Test updating knowledge with invalid domain"""
        parameters = {
            'operation': 'update_knowledge',
            'domain': 'invalid_domain',
            'knowledge_updates': {'concepts': ['test']}
        }
        
        result = self.orchestrator.handle_domain_operation(parameters)
        
        self.assertIn("error", result)


class TestDomainOrchestratorIntegration(unittest.TestCase):
    """Test integration scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = DomainOrchestrator()
    
    def test_end_to_end_math_operation(self):
        """Test complete end-to-end math operation flow"""
        params = {
            'operation': 'process',
            'domain': 'mathematics',
            'operation_mode': 'optimization',
            'input_data': {'a': 2, 'b': -4, 'c': 2},  # 2x^2 - 4x + 2
            'parameters': {'function_type': 'quadratic'}
        }
        
        result = self.orchestrator.handle_domain_operation(params)
        
        # Verify complete flow
        self.assertEqual(result['status'], 'completed')
        self.assertIn('optimization_result', result['result'])
        self.assertAlmostEqual(result['result']['optimization_result']['optimal_x'], 1.0, places=5)
        
        # Check operation was tracked
        op_id = result['operation_id']
        status = self.orchestrator.get_operation_status(op_id)
        self.assertIsNotNone(status)
        self.assertEqual(status['status'], 'completed')
    
    def test_end_to_end_sports_operation(self):
        """Test complete end-to-end sports operation flow"""
        params = {
            'operation': 'process',
            'domain': 'sports',
            'operation_mode': 'prediction',
            'input_data': {'historical_scores': [72, 73, 74, 71, 75]},
            'parameters': {
                'sport': 'golf',
                'course_difficulty': 'hard',
                'weather_conditions': 'poor'
            }
        }
        
        result = self.orchestrator.handle_domain_operation(params)
        
        # Verify complete flow
        self.assertEqual(result['status'], 'completed')
        self.assertIn('prediction', result['result'])
        self.assertIn('predicted_score', result['result']['prediction'])
        self.assertIn('confidence', result['result']['prediction'])
        
        # Check expert used
        self.assertEqual(result['expert_used'], 'SportsExpert')
    
    def test_multiple_domain_operations_sequence(self):
        """Test processing operations from multiple domains in sequence"""
        domains = ['mathematics', 'sports', 'general']
        results = []
        
        for domain in domains:
            params = {
                'operation': 'process',
                'domain': domain,
                'operation_mode': 'analysis',
                'input_data': [1, 2, 3, 4, 5]
            }
            result = self.orchestrator.handle_domain_operation(params)
            results.append(result)
        
        # All should complete
        for result in results:
            self.assertIn('status', result)
            self.assertEqual(result['status'], 'completed')
        
        # Check different experts were used
        expert_types = [r['expert_used'] for r in results]
        self.assertIn('MathematicsExpert', expert_types)
        self.assertIn('SportsExpert', expert_types)
        self.assertIn('GeneralExpert', expert_types)
    
    def test_knowledge_update_and_retrieval(self):
        """Test updating and retrieving domain knowledge"""
        # Update knowledge
        update_params = {
            'operation': 'update_knowledge',
            'domain': 'mathematics',
            'knowledge_updates': {
                'concepts': ['test_concept_1', 'test_concept_2'],
                'rules': ['test_rule_1'],
                'facts': ['test_fact_1']
            }
        }
        
        update_result = self.orchestrator.handle_domain_operation(update_params)
        self.assertEqual(update_result['status'], 'updated')
        
        # Retrieve and verify
        knowledge = self.orchestrator.get_domain_knowledge(DomainType.MATHEMATICS)
        self.assertIn('test_concept_1', knowledge.concepts)
        self.assertIn('test_concept_2', knowledge.concepts)
        self.assertIn('test_rule_1', knowledge.rules)
        self.assertIn('test_fact_1', knowledge.facts)
    
    def test_performance_metrics_accumulation(self):
        """Test that performance metrics accumulate correctly"""
        # Process multiple operations
        for i in range(5):
            params = {
                'operation': 'process',
                'domain': 'mathematics',
                'operation_mode': 'analysis',
                'input_data': list(range(i, i+5))
            }
            self.orchestrator.handle_domain_operation(params)
        
        # Get performance report
        report = self.orchestrator.get_performance_report()
        
        # Check metrics accumulated
        self.assertIn('domain_performance', report)
        math_perf = report['domain_performance'].get(DomainType.MATHEMATICS)
        self.assertIsNotNone(math_perf)
        self.assertEqual(math_perf['total_operations'], 5)
        self.assertIn('success_rate', math_perf)
        self.assertIn('avg_execution_time', math_perf)


class TestDomainOrchestratorEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = DomainOrchestrator()
    
    def test_empty_input_data(self):
        """Test handling empty input data"""
        parameters = {
            'operation': 'process',
            'domain': 'general',
            'operation_mode': 'analysis',
            'input_data': []
        }
        
        result = self.orchestrator.handle_domain_operation(parameters)
        
        # Should handle empty data gracefully
        self.assertIn("status", result)
    
    def test_very_large_input_data(self):
        """Test handling very large input data"""
        large_data = list(range(10000))
        parameters = {
            'operation': 'process',
            'domain': 'mathematics',
            'operation_mode': 'analysis',
            'input_data': large_data
        }
        
        result = self.orchestrator.handle_domain_operation(parameters)
        
        # Should handle large data
        self.assertIn("status", result)
        self.assertIn("result", result)
    
    def test_none_input_data(self):
        """Test handling None input data"""
        parameters = {
            'operation': 'process',
            'domain': 'general',
            'operation_mode': 'evaluation',
            'input_data': None
        }
        
        result = self.orchestrator.handle_domain_operation(parameters)
        
        # Should handle None gracefully
        self.assertIn("status", result)
    
    def test_operation_history_maxlen(self):
        """Test that operation history respects maxlen"""
        # Process many operations to exceed maxlen (1000)
        for i in range(1005):
            params = {
                'operation': 'process',
                'domain': 'general',
                'operation_mode': 'analysis',
                'input_data': [i]
            }
            self.orchestrator.handle_domain_operation(params)
        
        # History should be capped at maxlen
        self.assertLessEqual(len(self.orchestrator._operation_history), 1000)
    
    def test_duplicate_knowledge_updates(self):
        """Test handling duplicate knowledge updates"""
        parameters = {
            'operation': 'update_knowledge',
            'domain': 'mathematics',
            'knowledge_updates': {
                'concepts': ['algebra', 'algebra', 'calculus'],  # Duplicate algebra
                'rules': ['rule1', 'rule1']  # Duplicate rule
            }
        }
        
        result = self.orchestrator.handle_domain_operation(parameters)
        
        # Should handle duplicates (not add them twice)
        knowledge = self.orchestrator.get_domain_knowledge(DomainType.MATHEMATICS)
        algebra_count = knowledge.concepts.count('algebra')
        self.assertEqual(algebra_count, 1)  # Should only appear once
    
    def test_expert_fallback_to_general(self):
        """Test fallback to general expert for unsupported domains"""
        params = {
            'operation': 'process',
            'domain': 'medicine',  # No medicine expert registered
            'operation_mode': 'analysis',
            'input_data': {'test': 'medical data'}
        }
        
        result = self.orchestrator.handle_domain_operation(params)
        
        # Should use general expert as fallback
        self.assertIn('expert_used', result)
        self.assertEqual(result['expert_used'], 'GeneralExpert')
    
    def test_operation_timeout_handling(self):
        """Test operation timeout parameter is respected"""
        params = {
            'operation': 'process',
            'domain': 'mathematics',
            'operation_mode': 'simulation',
            'input_data': {'growth_rate': 0.1},
            'parameters': {
                'model_type': 'exponential_growth',
                'time_points': 1000  # Large number of points
            },
            'timeout': 10.0  # 10 second timeout
        }
        
        result = self.orchestrator.handle_domain_operation(params)
        
        # Should complete (simulation is fast enough)
        self.assertEqual(result['status'], 'completed')
        self.assertLess(result['execution_time'], 10.0)
    
    def test_complex_nested_input_data(self):
        """Test handling complex nested data structures"""
        complex_data = {
            'level1': {
                'level2': {
                    'values': [1, 2, 3],
                    'metadata': {'type': 'test'}
                },
                'arrays': [[1, 2], [3, 4], [5, 6]]
            }
        }
        
        params = {
            'operation': 'process',
            'domain': 'general',
            'operation_mode': 'evaluation',
            'input_data': complex_data
        }
        
        result = self.orchestrator.handle_domain_operation(params)
        
        # Should handle complex data
        self.assertIn('status', result)
        self.assertEqual(result['status'], 'completed')
    
    def test_special_characters_in_input(self):
        """Test handling special characters in input data"""
        params = {
            'operation': 'process',
            'domain': 'mathematics',
            'operation_mode': 'evaluation',
            'input_data': "2 * pi + e",
            'parameters': {'variables': {}}
        }
        
        result = self.orchestrator.handle_domain_operation(params)
        
        # Should evaluate mathematical constants
        self.assertEqual(result['status'], 'completed')
        self.assertIn('evaluation_result', result['result'])
    
    def test_concurrent_knowledge_updates(self):
        """Test concurrent knowledge updates are thread-safe"""
        import threading
        
        def update_knowledge(domain, concept):
            params = {
                'operation': 'update_knowledge',
                'domain': domain,
                'knowledge_updates': {
                    'concepts': [concept]
                }
            }
            self.orchestrator.handle_domain_operation(params)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=update_knowledge,
                args=('mathematics', f'concurrent_concept_{i}')
            )
            threads.append(thread)
        
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # All concepts should be added
        knowledge = self.orchestrator.get_domain_knowledge(DomainType.MATHEMATICS)
        for i in range(5):
            self.assertIn(f'concurrent_concept_{i}', knowledge.concepts)
    
    def test_invalid_expertise_level(self):
        """Test handling invalid expertise level"""
        params = {
            'operation': 'process',
            'domain': 'mathematics',
            'operation_mode': 'analysis',
            'input_data': [1, 2, 3],
            'expertise_level': 99  # Invalid level
        }
        
        result = self.orchestrator.handle_domain_operation(params)
        
        # Should handle invalid level gracefully
        self.assertIn('error', result)


class TestExpertCapabilities(unittest.TestCase):
    """Test expert capability assessment and routing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = DomainOrchestrator()
    
    def test_math_expert_capability_scores(self):
        """Test mathematics expert capability scoring for different operations"""
        expert = MathematicsExpert()
        
        # High score for analysis
        op_analysis = DomainOperation(
            operation_id="test1",
            domain=DomainType.MATHEMATICS,
            operation_mode=OperationMode.ANALYSIS,
            input_data=[1, 2, 3]
        )
        score = expert.assess_capability(op_analysis)
        self.assertGreater(score, 0.7)
        
        # Lower score for prediction
        op_prediction = DomainOperation(
            operation_id="test2",
            domain=DomainType.MATHEMATICS,
            operation_mode=OperationMode.PREDICTION,
            input_data=[1, 2, 3]
        )
        score = expert.assess_capability(op_prediction)
        self.assertLess(score, 0.7)
        
        # Zero score for wrong domain
        op_wrong = DomainOperation(
            operation_id="test3",
            domain=DomainType.LEGAL,
            operation_mode=OperationMode.ANALYSIS,
            input_data=[1, 2, 3]
        )
        score = expert.assess_capability(op_wrong)
        self.assertEqual(score, 0.0)
    
    def test_sports_expert_golf_specialization(self):
        """Test sports expert specialization in golf"""
        expert = SportsExpert()
        
        # High score for golf
        op_golf = DomainOperation(
            operation_id="test1",
            domain=DomainType.SPORTS,
            operation_mode=OperationMode.ANALYSIS,
            input_data={'golf_scores': [72, 73]},
            parameters={'sport': 'golf'}
        )
        score = expert.assess_capability(op_golf)
        self.assertEqual(score, 0.9)
        
        # Lower score for non-golf
        op_other = DomainOperation(
            operation_id="test2",
            domain=DomainType.SPORTS,
            operation_mode=OperationMode.ANALYSIS,
            input_data={'scores': [100, 95]},
            parameters={'sport': 'basketball'}
        )
        score = expert.assess_capability(op_other)
        self.assertEqual(score, 0.6)
    
    def test_routing_selects_best_expert(self):
        """Test that routing selects the most capable expert"""
        # Create operation that MathematicsExpert handles best
        operation = DomainOperation(
            operation_id="test",
            domain=DomainType.MATHEMATICS,
            operation_mode=OperationMode.OPTIMIZATION,
            input_data={'a': 1, 'b': 2, 'c': 3},
            parameters={'function_type': 'quadratic'}
        )
        
        expert = self.orchestrator._route_to_expert(operation)
        
        self.assertIsNotNone(expert)
        self.assertIsInstance(expert, MathematicsExpert)
    
    def test_expert_performance_affects_routing(self):
        """Test that expert performance history affects routing decisions"""
        # First, establish performance history
        for i in range(3):
            params = {
                'operation': 'process',
                'domain': 'mathematics',
                'operation_mode': 'analysis',
                'input_data': [i, i+1, i+2]
            }
            self.orchestrator.handle_domain_operation(params)
        
        # Check that performance is being tracked
        expert_id = 'MathematicsExpert_mathematics'
        perf = self.orchestrator._expert_performance[expert_id]
        self.assertGreater(perf['operations_handled'], 0)
        self.assertIn('success_rate', perf)
        
        # Create new operation and verify routing considers performance
        operation = DomainOperation(
            operation_id="test",
            domain=DomainType.MATHEMATICS,
            operation_mode=OperationMode.ANALYSIS,
            input_data=[1, 2, 3]
        )
        
        expert = self.orchestrator._route_to_expert(operation)
        self.assertIsNotNone(expert)


if __name__ == '__main__':
    unittest.main()