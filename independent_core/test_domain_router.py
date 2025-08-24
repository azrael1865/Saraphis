"""
Comprehensive test suite for DomainRouter
Tests all functionality including routing strategies, patterns, caching, and edge cases
"""

import unittest
import time
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from domain_router import (
    DomainRouter,
    RoutingStrategy,
    RoutingConfidence,
    RoutingResult,
    DomainPattern,
    RoutingMetrics
)


class MockDomainRegistry:
    """Mock domain registry for testing"""
    
    def __init__(self):
        self.domains = ['general', 'mathematics', 'science', 'programming', 'language', 'custom_domain']
    
    def list_domains(self):
        return self.domains.copy()


class TestDomainRouter(unittest.TestCase):
    """Comprehensive test suite for DomainRouter"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_registry = MockDomainRegistry()
        self.config = {
            'default_strategy': RoutingStrategy.HYBRID.value,
            'confidence_threshold': 0.5,
            'enable_fallback': True,
            'fallback_domain': 'general',
            'cache_max_size': 100,
            'history_size': 50
        }
        self.router = DomainRouter(domain_registry=self.mock_registry, config=self.config)
    
    def test_initialization(self):
        """Test router initialization and configuration"""
        self.assertEqual(self.router.default_strategy, RoutingStrategy.HYBRID)
        self.assertEqual(self.router.confidence_threshold, 0.5)
        self.assertTrue(self.router.enable_fallback)
        self.assertEqual(self.router.fallback_domain, 'general')
        
        # Test default patterns are loaded
        self.assertGreater(len(self.router.domain_patterns), 0)
        self.assertIn('mathematics', self.router.domain_patterns)
        self.assertIn('programming', self.router.domain_patterns)
        
        print("✓ Router initialization test passed")
    
    def test_domain_pattern_operations(self):
        """Test adding, removing, and managing domain patterns"""
        # Test adding a new pattern
        custom_pattern = DomainPattern(
            domain_name="custom_test",
            keywords=["custom", "test"],
            patterns=["custom pattern"],
            priority=8,
            confidence_boost=0.1
        )
        
        self.assertTrue(self.router.add_domain_pattern(custom_pattern))
        self.assertIn("custom_test", self.router.domain_patterns)
        
        # Test removing pattern
        self.assertTrue(self.router.remove_domain_pattern("custom_test"))
        self.assertNotIn("custom_test", self.router.domain_patterns)
        
        # Test removing non-existent pattern
        self.assertFalse(self.router.remove_domain_pattern("non_existent"))
        
        print("✓ Domain pattern operations test passed")
    
    def test_pattern_matching_strategy(self):
        """Test pattern matching routing strategy"""
        # Test mathematical input
        math_result = self.router.route_request(
            "Calculate 2 + 2 * 3",
            strategy=RoutingStrategy.PATTERN_MATCHING
        )
        self.assertEqual(math_result.target_domain, "mathematics")
        self.assertGreater(math_result.confidence_score, 0)
        self.assertEqual(math_result.routing_strategy, RoutingStrategy.PATTERN_MATCHING)
        
        # Test programming input
        prog_result = self.router.route_request(
            "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
            strategy=RoutingStrategy.PATTERN_MATCHING
        )
        self.assertEqual(prog_result.target_domain, "programming")
        
        # Test general input (should fallback)
        general_result = self.router.route_request(
            "hello world",
            strategy=RoutingStrategy.PATTERN_MATCHING
        )
        # Should route to general or use fallback
        self.assertIsNotNone(general_result.target_domain)
        
        print("✓ Pattern matching strategy test passed")
    
    def test_domain_hint_functionality(self):
        """Test domain hint routing"""
        # Test with valid domain hint
        result_with_hint = self.router.route_request(
            "some generic input",
            domain_hint="mathematics"
        )
        # Should check if mathematics pattern matches the input
        self.assertIsNotNone(result_with_hint.target_domain)
        
        # Test with invalid domain hint
        result_invalid_hint = self.router.route_request(
            "some generic input", 
            domain_hint="non_existent_domain"
        )
        self.assertIsNotNone(result_invalid_hint.target_domain)
        
        print("✓ Domain hint functionality test passed")
    
    def test_routing_strategies(self):
        """Test all routing strategies"""
        test_input = "Solve the equation x^2 + 5x + 6 = 0"
        
        strategies = [
            RoutingStrategy.PATTERN_MATCHING,
            RoutingStrategy.HEURISTIC_BASED,
            RoutingStrategy.ML_BASED,
            RoutingStrategy.HYBRID,
            RoutingStrategy.CONFIDENCE_WEIGHTED
        ]
        
        results = {}
        for strategy in strategies:
            result = self.router.route_request(test_input, strategy=strategy)
            results[strategy.value] = result
            
            # All strategies should return valid results
            self.assertIsInstance(result, RoutingResult)
            self.assertIsNotNone(result.target_domain)
            self.assertGreaterEqual(result.confidence_score, 0.0)
            self.assertLessEqual(result.confidence_score, 1.0)
            self.assertEqual(result.routing_strategy, strategy)
        
        print(f"✓ All {len(strategies)} routing strategies test passed")
    
    def test_confidence_levels(self):
        """Test confidence level calculation"""
        # Test confidence level mapping
        confidence_tests = [
            (0.95, RoutingConfidence.VERY_HIGH),
            (0.8, RoutingConfidence.HIGH),
            (0.6, RoutingConfidence.MEDIUM),
            (0.4, RoutingConfidence.LOW),
            (0.2, RoutingConfidence.VERY_LOW)
        ]
        
        for score, expected_level in confidence_tests:
            level = self.router._get_confidence_level(score)
            self.assertEqual(level, expected_level)
        
        print("✓ Confidence levels test passed")
    
    def test_fallback_mechanism(self):
        """Test fallback mechanism for low confidence routing"""
        # Configure low threshold to trigger fallback
        self.router.confidence_threshold = 0.8
        
        # Test with input that should have low confidence
        result = self.router.route_request("very generic unclear input text")
        
        # Should either route to a domain with sufficient confidence or fallback
        self.assertIsNotNone(result.target_domain)
        if result.confidence_score < 0.8:
            self.assertEqual(result.target_domain, self.router.fallback_domain)
            self.assertIn('fallback_applied', result.metadata)
        
        print("✓ Fallback mechanism test passed")
    
    def test_caching_functionality(self):
        """Test routing result caching"""
        test_input = "Calculate the derivative of x^2"
        
        # First request should populate cache
        start_time = time.time()
        result1 = self.router.route_request(test_input)
        first_time = time.time() - start_time
        
        # Second identical request should be faster (cached)
        start_time = time.time()
        result2 = self.router.route_request(test_input)
        second_time = time.time() - start_time
        
        # Results should be identical
        self.assertEqual(result1.target_domain, result2.target_domain)
        self.assertEqual(result1.confidence_score, result2.confidence_score)
        
        # Cache should be populated
        self.assertGreater(len(self.router.pattern_cache), 0)
        
        # Test cache clearing
        self.router.clear_cache()
        self.assertEqual(len(self.router.pattern_cache), 0)
        
        print("✓ Caching functionality test passed")
    
    def test_metrics_tracking(self):
        """Test routing metrics collection"""
        # Reset metrics
        self.router.reset_metrics()
        
        # Perform several routing operations
        test_inputs = [
            "solve equation",
            "def function():",
            "translate this text", 
            "chemical reaction",
            "general question"
        ]
        
        for inp in test_inputs:
            self.router.route_request(inp)
        
        metrics = self.router.get_routing_metrics()
        
        # Verify metrics are tracked
        self.assertEqual(metrics['total_routes'], len(test_inputs))
        self.assertGreaterEqual(metrics['successful_routes'], 0)
        self.assertGreaterEqual(metrics['average_routing_time_ms'], 0)
        self.assertIsInstance(metrics['domain_usage_counts'], dict)
        self.assertIsInstance(metrics['confidence_distribution'], dict)
        
        print("✓ Metrics tracking test passed")
    
    def test_routing_history(self):
        """Test routing history tracking"""
        # Clear history
        self.router.routing_history.clear()
        
        # Make some requests
        test_inputs = ["math problem", "code example", "science question"]
        for inp in test_inputs:
            self.router.route_request(inp)
        
        history = self.router.get_routing_history()
        self.assertEqual(len(history), len(test_inputs))
        
        # Check history structure
        for entry in history:
            self.assertIn('timestamp', entry)
            self.assertIn('input_preview', entry)
            self.assertIn('result', entry)
        
        print("✓ Routing history test passed")
    
    def test_domain_detection_utilities(self):
        """Test utility methods for domain detection"""
        # Test detect_domain
        domain = self.router.detect_domain("Calculate integral of x^2")
        self.assertIsInstance(domain, str)
        self.assertIn(domain, self.router.get_available_domains())
        
        # Test validate_routing
        is_valid = self.router.validate_routing(
            "mathematics", 
            "solve equation x^2 = 4",
            expected_confidence=0.3
        )
        self.assertIsInstance(is_valid, bool)
        
        # Test get_routing_confidence
        confidence = self.router.get_routing_confidence(
            "def hello():", 
            "programming"
        )
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        print("✓ Domain detection utilities test passed")
    
    def test_available_domains(self):
        """Test getting available domains"""
        domains = self.router.get_available_domains()
        
        # Should include default patterns
        expected_domains = ["general", "mathematics", "science", "programming", "language"]
        for domain in expected_domains:
            self.assertIn(domain, domains)
        
        # Should include registry domains
        if self.router.domain_registry:
            registry_domains = self.router.domain_registry.list_domains()
            for domain in registry_domains:
                self.assertIn(domain, domains)
        
        print("✓ Available domains test passed")
    
    def test_pattern_import_export(self):
        """Test pattern import/export functionality"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            # Export patterns
            export_success = self.router.export_patterns(tmp_path)
            self.assertTrue(export_success)
            self.assertTrue(tmp_path.exists())
            
            # Verify export content
            with open(tmp_path, 'r') as f:
                exported_data = json.load(f)
            
            self.assertIsInstance(exported_data, dict)
            self.assertIn('mathematics', exported_data)
            
            # Clear patterns and import
            original_patterns = len(self.router.domain_patterns)
            self.router.domain_patterns.clear()
            
            import_success = self.router.import_patterns(tmp_path)
            self.assertTrue(import_success)
            self.assertGreater(len(self.router.domain_patterns), 0)
            
        finally:
            # Clean up
            if tmp_path.exists():
                tmp_path.unlink()
        
        print("✓ Pattern import/export test passed")
    
    def test_pattern_matching_logic(self):
        """Test DomainPattern matching logic"""
        # Create test pattern
        pattern = DomainPattern(
            domain_name="test_domain",
            keywords=["test", "sample"],
            patterns=["test pattern"],
            regex_patterns=[r'\btest\w*'],
            input_types=["test_type"],
            priority=7,
            confidence_boost=0.1
        )
        
        # Test keyword matching
        matches, confidence = pattern.matches_input("This is a test input", "text")
        self.assertTrue(matches)
        self.assertGreater(confidence, 0)
        
        # Test pattern matching
        matches, confidence = pattern.matches_input("test pattern example", "text")
        self.assertTrue(matches)
        
        # Test regex matching
        matches, confidence = pattern.matches_input("testing regex", "text")
        self.assertTrue(matches)
        
        # Test input type matching
        matches, confidence = pattern.matches_input("any input", "test_type")
        self.assertTrue(matches)
        
        # Test no matches
        matches, confidence = pattern.matches_input("no matching content", "other_type")
        self.assertFalse(matches)
        self.assertEqual(confidence, 0.0)
        
        print("✓ Pattern matching logic test passed")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test empty input
        result = self.router.route_request("")
        self.assertIsNotNone(result.target_domain)
        
        # Test None input
        result = self.router.route_request(None)
        self.assertIsNotNone(result.target_domain)
        
        # Test very long input
        long_input = "x" * 10000
        result = self.router.route_request(long_input)
        self.assertIsNotNone(result.target_domain)
        
        # Test special characters
        special_input = "∫∑∂∆αβγδε π ∞"
        result = self.router.route_request(special_input)
        self.assertIsNotNone(result.target_domain)
        
        # Test with disabled fallback
        original_fallback = self.router.enable_fallback
        self.router.enable_fallback = False
        result = self.router.route_request("unclear input")
        self.assertIsNotNone(result.target_domain)
        self.router.enable_fallback = original_fallback
        
        print("✓ Edge cases test passed")
    
    def test_routing_result_serialization(self):
        """Test RoutingResult to_dict functionality"""
        result = self.router.route_request("test input")
        result_dict = result.to_dict()
        
        # Check all required fields are present
        required_fields = [
            'target_domain', 'confidence_score', 'confidence_level',
            'reasoning', 'alternative_domains', 'routing_strategy',
            'routing_time_ms', 'metadata'
        ]
        
        for field in required_fields:
            self.assertIn(field, result_dict)
        
        # Check field types
        self.assertIsInstance(result_dict['target_domain'], str)
        self.assertIsInstance(result_dict['confidence_score'], (int, float))
        self.assertIsInstance(result_dict['confidence_level'], str)
        self.assertIsInstance(result_dict['reasoning'], str)
        self.assertIsInstance(result_dict['alternative_domains'], list)
        self.assertIsInstance(result_dict['routing_strategy'], str)
        self.assertIsInstance(result_dict['routing_time_ms'], (int, float))
        self.assertIsInstance(result_dict['metadata'], dict)
        
        print("✓ RoutingResult serialization test passed")
    
    def test_optimization_analysis(self):
        """Test routing optimization analysis"""
        # Generate some routing data
        for i in range(50):
            self.router.route_request(f"test input {i}")
        
        optimization_report = self.router.optimize_routing()
        
        # Check report structure
        self.assertIn('current_metrics', optimization_report)
        self.assertIn('optimization_suggestions', optimization_report)
        self.assertIn('cache_efficiency', optimization_report)
        self.assertIn('low_confidence_rate', optimization_report)
        
        # Check metrics
        self.assertIsInstance(optimization_report['optimization_suggestions'], list)
        self.assertGreaterEqual(optimization_report['cache_efficiency'], 0)
        self.assertGreaterEqual(optimization_report['low_confidence_rate'], 0)
        
        print("✓ Optimization analysis test passed")
    
    def test_thread_safety(self):
        """Test thread safety of router operations"""
        import threading
        import queue
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def worker():
            try:
                for i in range(10):
                    result = self.router.route_request(f"thread test {i}")
                    results_queue.put(result.target_domain)
            except Exception as e:
                errors_queue.put(e)
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check for errors
        self.assertTrue(errors_queue.empty(), "Thread safety test failed with errors")
        
        # Check results
        self.assertEqual(results_queue.qsize(), 50)  # 5 threads * 10 requests each
        
        print("✓ Thread safety test passed")
    
    def test_regex_pattern_safety(self):
        """Test regex pattern safety and error handling"""
        # Add pattern with invalid regex
        bad_pattern = DomainPattern(
            domain_name="bad_regex",
            regex_patterns=["[invalid regex"],  # Invalid regex
            keywords=["test"]
        )
        
        # Should not crash when adding
        self.assertTrue(self.router.add_domain_pattern(bad_pattern))
        
        # Should not crash when matching
        matches, confidence = bad_pattern.matches_input("test input", "text")
        # Should still match on keywords despite bad regex
        self.assertTrue(matches)
        
        print("✓ Regex pattern safety test passed")


def run_comprehensive_tests():
    """Run all tests and provide summary"""
    print("=" * 70)
    print("DomainRouter Comprehensive Test Suite")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDomainRouter)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailed Tests:")
        for test, traceback in result.failures:
            print(f"  - {test}")
            print(f"    Failure: {traceback.split('AssertionError:')[-1].strip()[:200]}...")
    
    if result.errors:
        print("\nTests with Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
            print(f"    Error: {traceback.split('Exception:')[-1].strip()[:200]}...")
    
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)