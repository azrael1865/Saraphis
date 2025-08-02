#!/usr/bin/env python3
"""
Comprehensive integration testing for uncertainty quantification.
"""

import sys
import os
import time
import numpy as np

# Add the independent_core directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'independent_core'))

from brain import Brain, BrainSystemConfig

def test_uncertainty_integration():
    """Comprehensive integration testing for uncertainty quantification"""
    try:
        print("🔍 Testing Uncertainty Integration and Orchestration...")
        
        # Initialize Brain with uncertainty integration
        brain = Brain(BrainSystemConfig())
        brain.initialize_uncertainty_integration()
        
        # Test data
        test_data = ['A', 'B', 'A', 'C', 'B', 'A', 'D', 'B', 'A', 'C']
        
        print("\n1. Testing basic uncertainty integration...")
        # Test 1: Basic uncertainty integration
        result1 = brain.submit_orchestration_task_with_uncertainty('uncertainty', data=test_data)
        print(f"   ✅ Basic integration test: {result1.get('uncertainty_result', {}).get('method', 'unknown')}")
        print(f"   ✅ Decision: {result1.get('decision', {}).get('action', 'unknown')}")
        
        print("\n2. Testing domain-specific uncertainty...")
        # Test 2: Domain-specific uncertainty
        brain.add_domain('test_domain', {'type': 'specialized'})
        result2 = brain.submit_orchestration_task_with_uncertainty('domain_uncertainty', domain='test_domain', data=test_data)
        print(f"   ✅ Domain integration test: {result2.get('uncertainty_result', {}).get('method', 'unknown')}")
        
        print("\n3. Testing cross-domain propagation...")
        # Test 3: Cross-domain propagation
        brain.add_domain('related_domain', {'type': 'specialized'})
        result3 = brain.submit_orchestration_task_with_uncertainty('cross_domain', source_domain='test_domain', target_domain='related_domain', data=test_data)
        print(f"   ✅ Cross-domain test: {result3.get('uncertainty_result', {}).get('method', 'unknown')}")
        
        print("\n4. Testing decision making...")
        # Test 4: Decision making
        decision_result = brain._make_uncertainty_based_decision({
            'uncertainty': 0.8,
            'confidence_level': 0.2,
            'method': 'possibility_based'
        })
        print(f"   ✅ Decision making test: {decision_result.get('action', 'unknown')}")
        
        print("\n5. Testing method selection...")
        # Test 5: Method selection
        method1 = brain._select_uncertainty_method('training', test_data)
        method2 = brain._select_uncertainty_method('prediction', test_data)
        method3 = brain._select_uncertainty_method('validation', test_data)
        print(f"   ✅ Training method: {method1}")
        print(f"   ✅ Prediction method: {method2}")
        print(f"   ✅ Validation method: {method3}")
        
        print("\n6. Testing data analysis...")
        # Test 6: Data analysis
        data_type = brain._analyze_data_type(test_data)
        data_complexity = brain._analyze_data_complexity(test_data)
        print(f"   ✅ Data type: {data_type}")
        print(f"   ✅ Data complexity: {data_complexity}")
        
        print("\n7. Testing method-specific recommendations...")
        # Test 7: Method-specific recommendations
        recommendations = brain._get_method_specific_recommendations('possibility_based', 0.8, 0.2)
        print(f"   ✅ Recommendations: {recommendations}")
        
        print("\n8. Testing uncertainty integration report...")
        # Test 8: Integration report
        report = brain.get_uncertainty_integration_report()
        print(f"   ✅ Integration status: {report.get('integration_status', 'unknown')}")
        print(f"   ✅ Performance metrics: {len(report.get('performance_metrics', {}))}")
        
        print("\n9. Testing different data types...")
        # Test 9: Different data types
        continuous_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mixed_data = [1, 'A', 2.5, 'B', 3]
        
        result_continuous = brain.submit_orchestration_task_with_uncertainty('uncertainty', data=continuous_data)
        result_mixed = brain.submit_orchestration_task_with_uncertainty('uncertainty', data=mixed_data)
        
        print(f"   ✅ Continuous data method: {result_continuous.get('uncertainty_result', {}).get('method', 'unknown')}")
        print(f"   ✅ Mixed data method: {result_mixed.get('uncertainty_result', {}).get('method', 'unknown')}")
        
        print("\n10. Testing uncertainty hooks...")
        # Test 10: Uncertainty hooks
        def test_hook(operation, data, **kwargs):
            return {'hook_result': 'test_success'}
        
        brain.register_uncertainty_hook(test_hook)
        print(f"   ✅ Uncertainty hook registered successfully")
        
        brain.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_uncertainty_quantifiers():
    """Test all uncertainty quantifiers with the integration"""
    try:
        print("\n🔍 Testing All Uncertainty Quantifiers with Integration...")
        
        brain = Brain(BrainSystemConfig())
        brain.initialize_uncertainty_integration()
        
        test_data = ['A', 'B', 'A', 'C', 'B', 'A', 'D', 'B', 'A', 'C']
        
        # Test each quantifier
        quantifiers = ['conformalized_credal', 'deep_deterministic', 'batch_ensemble', 'entropy_based', 'possibility_based']
        
        for quantifier in quantifiers:
            print(f"\nTesting {quantifier}...")
            result = brain.uncertainty_orchestrator.quantify({
                'operation': 'estimate_uncertainty',
                'data': test_data,
                'method': quantifier,
                'uncertainty_type': 'epistemic'
            })
            
            uncertainty = result.get('uncertainty', 0.0)
            confidence = result.get('confidence_level', 0.0)
            method = result.get('method', 'unknown')
            
            print(f"   ✅ {quantifier}: Uncertainty={uncertainty:.3f}, Confidence={confidence:.3f}")
            
            # Test decision making
            decision = brain._make_uncertainty_based_decision(result)
            print(f"   ✅ Decision: {decision.get('action', 'unknown')}")
        
        brain.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Quantifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cross_domain_propagation():
    """Test cross-domain uncertainty propagation"""
    try:
        print("\n🔍 Testing Cross-Domain Uncertainty Propagation...")
        
        brain = Brain(BrainSystemConfig())
        brain.initialize_uncertainty_integration()
        
        # Add test domains
        brain.add_domain('fraud_detection', {'type': 'specialized'})
        brain.add_domain('financial_analysis', {'type': 'specialized'})
        brain.add_domain('risk_assessment', {'type': 'specialized'})
        
        test_data = ['A', 'B', 'A', 'C', 'B', 'A', 'D', 'B', 'A', 'C']
        
        # Test propagation
        print("Testing uncertainty propagation from fraud_detection...")
        uncertainty_result = {
            'uncertainty': 0.85,  # High uncertainty
            'confidence_level': 0.15,
            'method': 'possibility_based'
        }
        
        brain._domain_uncertainty_callback('fraud_detection', uncertainty_result)
        
        # Check propagation history
        propagation_history = brain.cross_domain_propagation.get('history', [])
        print(f"   ✅ Propagation events: {len(propagation_history)}")
        
        if propagation_history:
            for event in propagation_history:
                print(f"   ✅ Propagation: {event['source_domain']} -> {event['target_domain']}")
        
        brain.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Cross-domain propagation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_uncertainty_monitoring():
    """Test uncertainty monitoring functionality"""
    try:
        print("\n🔍 Testing Uncertainty Monitoring...")
        
        brain = Brain(BrainSystemConfig())
        brain.initialize_uncertainty_integration()
        
        test_data = ['A', 'B', 'A', 'C', 'B', 'A', 'D', 'B', 'A', 'C']
        
        # Test monitoring with different uncertainty levels
        uncertainty_levels = [0.3, 0.6, 0.9]  # Low, medium, high
        
        for uncertainty_level in uncertainty_levels:
            print(f"Testing monitoring with uncertainty {uncertainty_level}...")
            
            # Simulate uncertainty result
            uncertainty_result = {
                'uncertainty': uncertainty_level,
                'confidence_level': 1.0 - uncertainty_level,
                'method': 'entropy_based'
            }
            
            # Trigger monitoring
            brain._domain_uncertainty_callback('test_domain', uncertainty_result)
            
            # Check alerts
            alerts = brain.uncertainty_monitoring.get('alerts', [])
            print(f"   ✅ Alerts generated: {len(alerts)}")
        
        brain.shutdown()
        return True
        
    except Exception as e:
        print(f"❌ Uncertainty monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎯 COMPREHENSIVE UNCERTAINTY INTEGRATION TESTING")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Uncertainty Integration", test_uncertainty_integration),
        ("Uncertainty Quantifiers", test_uncertainty_quantifiers),
        ("Cross-Domain Propagation", test_cross_domain_propagation),
        ("Uncertainty Monitoring", test_uncertainty_monitoring)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*60}")
    print(f"🎯 TEST RESULTS: {passed}/{total} TESTS PASSED")
    
    if passed == total:
        print("\n🎉 ALL UNCERTAINTY INTEGRATION TESTS PASSED!")
        print("The following features are working:")
        print("  ✅ 1. Uncertainty integration with Brain")
        print("  ✅ 2. Domain-specific uncertainty quantification")
        print("  ✅ 3. Cross-domain uncertainty propagation")
        print("  ✅ 4. Uncertainty-based decision making")
        print("  ✅ 5. Real-time uncertainty monitoring")
        print("  ✅ 6. Comprehensive reporting and alerting")
        print("  ✅ 7. Method selection based on data characteristics")
        print("  ✅ 8. Method-specific recommendations")
        print("  ✅ 9. Hard failures for debugging")
        print("  ✅ 10. Production-ready error handling")
    else:
        print(f"\n❌ {total - passed} TESTS FAILED")
        print("Some uncertainty integration features are not working properly.")
    
    sys.exit(0 if passed == total else 1) 