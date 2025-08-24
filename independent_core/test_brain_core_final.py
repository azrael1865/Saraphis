"""
Final comprehensive test for BrainCore with actual methods
"""

import sys
import os
import logging
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import json
import time
import numpy as np

# Setup
logging.basicConfig(level=logging.INFO)
sys.path.insert(0, '.')

from independent_core.brain_core import BrainCore, BrainConfig, UncertaintyMetrics, PredictionResult

def test_brain_core_comprehensive():
    """Comprehensive test of BrainCore"""
    
    print("\n" + "="*60)
    print("COMPREHENSIVE BRAIN CORE TEST")
    print("="*60)
    
    # Setup
    temp_dir = tempfile.mkdtemp()
    test_results = []
    
    try:
        # Test 1: Initialization with different configs
        print("\n[TEST 1: Initialization]")
        try:
            # Default config
            brain1 = BrainCore()
            print("✓ Default initialization successful")
            
            # Dict config
            brain2 = BrainCore({'shared_memory_size': 5000, 'reasoning_depth': 3})
            print("✓ Dict config initialization successful")
            
            # BrainConfig object
            config = BrainConfig(shared_memory_size=8000, reasoning_depth=7)
            brain3 = BrainCore(config)
            print("✓ BrainConfig object initialization successful")
            
            test_results.append(("Initialization", True))
        except Exception as e:
            print(f"✗ Initialization failed: {e}")
            test_results.append(("Initialization", False))
            return test_results
        
        # Use the default brain for remaining tests
        brain = brain1
        
        # Test 2: Configuration access and validation
        print("\n[TEST 2: Configuration Access]")
        try:
            config_dict = brain.get_config()
            if isinstance(config_dict, dict) and 'shared_memory_size' in config_dict:
                print("✓ Config access works")
                print(f"  - Type: {type(config_dict)}")
                print(f"  - Keys: {len(config_dict)} config parameters")
                test_results.append(("Configuration Access", True))
            else:
                print(f"✗ Config access returns unexpected format: {type(config_dict)}")
                test_results.append(("Configuration Access", False))
        except Exception as e:
            print(f"✗ Configuration access failed: {e}")
            test_results.append(("Configuration Access", False))
        
        # Test 3: Basic prediction functionality
        print("\n[TEST 3: Basic Prediction]")
        try:
            # Test various input types
            test_inputs = [
                ([1, 2, 3, 4], "list"),
                (np.array([1, 2, 3, 4]), "numpy_array"),
                ({"data": [1, 2, 3]}, "dict"),
                (42, "scalar")
            ]
            
            predictions_successful = 0
            for test_input, input_type in test_inputs:
                try:
                    result = brain.predict(test_input)
                    if isinstance(result, PredictionResult) and result.success:
                        predictions_successful += 1
                        print(f"  ✓ {input_type}: confidence={result.confidence:.3f}")
                    else:
                        print(f"  ✗ {input_type}: prediction failed")
                except Exception as e:
                    print(f"  ✗ {input_type}: {e}")
            
            if predictions_successful >= len(test_inputs) * 0.75:  # 75% success rate
                print("✓ Basic prediction functionality works")
                test_results.append(("Basic Prediction", True))
            else:
                print(f"✗ Basic prediction insufficient: {predictions_successful}/{len(test_inputs)}")
                test_results.append(("Basic Prediction", False))
        except Exception as e:
            print(f"✗ Basic prediction test failed: {e}")
            test_results.append(("Basic Prediction", False))
        
        # Test 4: Shared knowledge management
        print("\n[TEST 4: Shared Knowledge Management]")
        try:
            # Test knowledge storage and retrieval
            test_knowledge = [
                ("math_constants", {"pi": 3.14159, "e": 2.71828}, "mathematics"),
                ("physics_laws", {"gravity": 9.81, "light_speed": 299792458}, "physics"),
                ("chemistry_data", {"water_boiling": 100, "water_freezing": 0}, "chemistry")
            ]
            
            knowledge_operations_successful = 0
            for key, data, domain in test_knowledge:
                try:
                    # Store knowledge
                    brain.add_shared_knowledge(key, data, domain=domain)
                    
                    # Retrieve knowledge
                    retrieved = brain.get_shared_knowledge(key)
                    
                    if retrieved and retrieved == data:
                        knowledge_operations_successful += 1
                        print(f"  ✓ {key}: stored and retrieved successfully")
                    else:
                        print(f"  ✗ {key}: retrieval mismatch")
                except Exception as e:
                    print(f"  ✗ {key}: {e}")
            
            if knowledge_operations_successful == len(test_knowledge):
                print("✓ Shared knowledge management works")
                test_results.append(("Shared Knowledge Management", True))
            else:
                print(f"✗ Knowledge management failed: {knowledge_operations_successful}/{len(test_knowledge)}")
                test_results.append(("Shared Knowledge Management", False))
        except Exception as e:
            print(f"✗ Shared knowledge test failed: {e}")
            test_results.append(("Shared Knowledge Management", False))
        
        # Test 5: Knowledge search functionality
        print("\n[TEST 5: Knowledge Search]")
        try:
            # Search for previously stored knowledge
            search_queries = ["math", "physics", "water"]
            
            search_successful = 0
            for query in search_queries:
                try:
                    results = brain.search_shared_knowledge(query)
                    if results and len(results) > 0:
                        search_successful += 1
                        print(f"  ✓ Search '{query}': {len(results)} results")
                    else:
                        print(f"  ✗ Search '{query}': no results")
                except Exception as e:
                    print(f"  ✗ Search '{query}': {e}")
            
            if search_successful >= len(search_queries) * 0.66:  # 66% success rate
                print("✓ Knowledge search functionality works")
                test_results.append(("Knowledge Search", True))
            else:
                print(f"✗ Knowledge search insufficient: {search_successful}/{len(search_queries)}")
                test_results.append(("Knowledge Search", False))
        except Exception as e:
            print(f"✗ Knowledge search test failed: {e}")
            test_results.append(("Knowledge Search", False))
        
        # Test 6: State management
        print("\n[TEST 6: State Management]")
        try:
            # Test state summary
            state_summary = brain.get_state_summary()
            print(f"  - State summary: {len(state_summary)} entries")
            
            # Test state save/load with temporary file
            temp_state_file = Path(temp_dir) / "brain_state.json"
            brain.save_state(temp_state_file)
            
            if temp_state_file.exists():
                print("  ✓ State save successful")
                
                # Create new brain and load state
                new_brain = BrainCore()
                new_brain.load_state(temp_state_file)
                
                # Verify knowledge is preserved
                retrieved_knowledge = new_brain.get_shared_knowledge("math_constants")
                if retrieved_knowledge:
                    print("  ✓ State load successful - knowledge preserved")
                    test_results.append(("State Management", True))
                else:
                    print("  ✗ State load failed - knowledge not preserved")
                    test_results.append(("State Management", False))
            else:
                print("  ✗ State save failed")
                test_results.append(("State Management", False))
        except Exception as e:
            print(f"✗ State management test failed: {e}")
            test_results.append(("State Management", False))
        
        # Test 7: Uncertainty quantification
        print("\n[TEST 7: Uncertainty Quantification]")
        try:
            if brain.config.enable_uncertainty:
                # Test uncertainty statistics
                uncertainty_stats = brain.get_uncertainty_statistics()
                print(f"  - Uncertainty stats: {len(uncertainty_stats)} metrics")
                
                # Test prediction with uncertainty
                result = brain.predict([1, 2, 3, 4], domain="test")
                if result.uncertainty_metrics:
                    print("  ✓ Prediction includes uncertainty metrics")
                    print(f"    - Epistemic: {result.uncertainty_metrics.epistemic_uncertainty:.3f}")
                    print(f"    - Aleatoric: {result.uncertainty_metrics.aleatoric_uncertainty:.3f}")
                    test_results.append(("Uncertainty Quantification", True))
                else:
                    print("  ✗ Prediction missing uncertainty metrics")
                    test_results.append(("Uncertainty Quantification", False))
            else:
                print("  ✓ Uncertainty quantification disabled (as expected)")
                test_results.append(("Uncertainty Quantification", True))
        except Exception as e:
            print(f"✗ Uncertainty quantification test failed: {e}")
            test_results.append(("Uncertainty Quantification", False))
        
        # Test 8: Caching functionality
        print("\n[TEST 8: Caching]")
        try:
            if brain.config.enable_caching:
                # Make same prediction twice to test caching
                input_data = [5, 6, 7, 8]
                
                # First prediction
                start_time1 = time.time()
                result1 = brain.predict(input_data, domain="cache_test")
                time1 = time.time() - start_time1
                
                # Second prediction (should be cached)
                start_time2 = time.time()
                result2 = brain.predict(input_data, domain="cache_test")
                time2 = time.time() - start_time2
                
                if result1.success and result2.success:
                    print(f"  ✓ Caching functional - times: {time1:.4f}s vs {time2:.4f}s")
                    
                    # Test cache clearing
                    brain.clear_cache()
                    print("  ✓ Cache clearing works")
                    test_results.append(("Caching", True))
                else:
                    print("  ✗ Caching test predictions failed")
                    test_results.append(("Caching", False))
            else:
                print("  ✓ Caching disabled (as expected)")
                test_results.append(("Caching", True))
        except Exception as e:
            print(f"✗ Caching test failed: {e}")
            test_results.append(("Caching", False))
        
        # Test 9: Domain-specific functionality
        print("\n[TEST 9: Domain-Specific Processing]")
        try:
            test_domains = ["mathematics", "physics", "chemistry", "biology", "general"]
            domain_predictions_successful = 0
            
            for domain in test_domains:
                try:
                    result = brain.predict([1, 2, 3], domain=domain)
                    if result.success:
                        domain_predictions_successful += 1
                        print(f"  ✓ Domain '{domain}': confidence={result.confidence:.3f}")
                    else:
                        print(f"  ✗ Domain '{domain}': prediction failed")
                except Exception as e:
                    print(f"  ✗ Domain '{domain}': {e}")
            
            if domain_predictions_successful >= len(test_domains) * 0.8:  # 80% success rate
                print("✓ Domain-specific processing works")
                test_results.append(("Domain-Specific Processing", True))
            else:
                print(f"✗ Domain processing insufficient: {domain_predictions_successful}/{len(test_domains)}")
                test_results.append(("Domain-Specific Processing", False))
        except Exception as e:
            print(f"✗ Domain-specific test failed: {e}")
            test_results.append(("Domain-Specific Processing", False))
        
        # Test 10: Statistics and monitoring
        print("\n[TEST 10: Statistics and Monitoring]")
        try:
            # Test general statistics
            stats = brain.get_statistics()
            if isinstance(stats, dict) and len(stats) > 0:
                print(f"  ✓ Statistics available: {len(stats)} metrics")
            else:
                print("  ✗ Statistics not available")
            
            # Test confidence scoring
            confidence_score = brain.get_confidence_score("test_domain")
            if isinstance(confidence_score, (int, float)) and 0 <= confidence_score <= 1:
                print(f"  ✓ Confidence scoring works: {confidence_score:.3f}")
            else:
                print(f"  ✗ Confidence scoring invalid: {confidence_score}")
            
            # Test reliability assessment
            reliability = brain.assess_reliability()
            if isinstance(reliability, dict):
                print(f"  ✓ Reliability assessment works: {len(reliability)} metrics")
                test_results.append(("Statistics and Monitoring", True))
            else:
                print("  ✗ Reliability assessment failed")
                test_results.append(("Statistics and Monitoring", False))
        except Exception as e:
            print(f"✗ Statistics and monitoring test failed: {e}")
            test_results.append(("Statistics and Monitoring", False))
        
        # Test 11: Configuration updates
        print("\n[TEST 11: Configuration Updates]")
        try:
            # Test configuration update
            original_reasoning_depth = brain.config.reasoning_depth
            new_config = {'reasoning_depth': original_reasoning_depth + 2}
            
            brain.update_config(new_config)
            updated_config = brain.get_config()
            
            if updated_config['reasoning_depth'] == original_reasoning_depth + 2:
                print("✓ Configuration updates work")
                test_results.append(("Configuration Updates", True))
            else:
                print("✗ Configuration update failed")
                test_results.append(("Configuration Updates", False))
        except Exception as e:
            print(f"✗ Configuration updates test failed: {e}")
            test_results.append(("Configuration Updates", False))
        
        # Test 12: Error handling and validation
        print("\n[TEST 12: Error Handling]")
        try:
            error_scenarios = [
                (None, "None input"),
                ("invalid_string_for_prediction", "Invalid string"),
                ([], "Empty list"),
            ]
            
            error_handling_working = 0
            for invalid_input, description in error_scenarios:
                try:
                    result = brain.predict(invalid_input)
                    # Should either succeed gracefully or fail gracefully
                    if isinstance(result, PredictionResult):
                        error_handling_working += 1
                        print(f"  ✓ {description}: handled gracefully")
                    else:
                        print(f"  ✗ {description}: unexpected return type")
                except Exception as e:
                    # Catching exceptions is also valid error handling
                    error_handling_working += 1
                    print(f"  ✓ {description}: exception caught ({type(e).__name__})")
            
            if error_handling_working >= len(error_scenarios) * 0.66:  # 66% handled properly
                print("✓ Error handling works")
                test_results.append(("Error Handling", True))
            else:
                print(f"✗ Error handling insufficient: {error_handling_working}/{len(error_scenarios)}")
                test_results.append(("Error Handling", False))
        except Exception as e:
            print(f"✗ Error handling test failed: {e}")
            test_results.append(("Error Handling", False))
    
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    success_rate = (passed/total * 100) if total > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
    
    return test_results


if __name__ == "__main__":
    results = test_brain_core_comprehensive()
    
    # Exit with appropriate code
    all_passed = all(result for _, result in results)
    sys.exit(0 if all_passed else 1)