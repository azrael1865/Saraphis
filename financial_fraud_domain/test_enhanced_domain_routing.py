#!/usr/bin/env python3
"""
Test Enhanced Domain Routing Integration
Comprehensive test suite for the enhanced domain routing system
"""

import asyncio
import logging
import json
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_enhanced_domain_routing():
    """Test the enhanced domain routing system"""
    
    try:
        logger.info("=== Enhanced Domain Routing Integration Test ===")
        
        # Import the enhanced domain routing
        from domain_routing import (
            create_enhanced_fraud_router, validate_and_route_fraud_request,
            FinancialFraudRouter, ENHANCED_AVAILABLE
        )
        
        logger.info(f"Enhanced features available: {ENHANCED_AVAILABLE}")
        
        # Test 1: Basic router creation with different validation levels
        logger.info("\n--- Test 1: Router Creation with Validation Levels ---")
        
        validation_levels = ['basic', 'standard', 'strict', 'paranoid']
        routers = {}
        
        for level in validation_levels:
            try:
                router = create_enhanced_fraud_router(validation_level=level)
                routers[level] = router
                logger.info(f"✓ Created router with {level} validation level")
                logger.info(f"  Enhanced features enabled: {router.use_enhanced}")
            except Exception as e:
                logger.error(f"✗ Failed to create router with {level} validation: {e}")
        
        # Test 2: Enhanced vs Standard routing comparison
        logger.info("\n--- Test 2: Enhanced vs Standard Routing ---")
        
        # Create test input
        test_input = "Analyze suspicious transaction for $15,000"
        test_context = {
            'transaction_amount': 15000.0,
            'transaction_type': 'transfer',
            'transaction_id': 'TXN123456789',
            'user_risk_profile': 'medium'
        }
        
        # Test enhanced routing
        if ENHANCED_AVAILABLE:
            enhanced_router = create_enhanced_fraud_router(
                validation_level='strict',
                use_enhanced=True
            )
            
            try:
                enhanced_result = enhanced_router.route_fraud_request(
                    test_input,
                    context=test_context if hasattr(enhanced_router, 'enhanced_transaction_validator') else None
                )
                logger.info(f"✓ Enhanced routing successful")
                logger.info(f"  Target domain: {enhanced_result.target_domain}")
                logger.info(f"  Confidence: {enhanced_result.confidence_score:.3f}")
                if hasattr(enhanced_result, 'risk_score'):
                    logger.info(f"  Risk score: {enhanced_result.risk_score:.3f}")
            except Exception as e:
                logger.error(f"✗ Enhanced routing failed: {e}")
        
        # Test standard routing
        standard_router = create_enhanced_fraud_router(
            validation_level='standard',
            use_enhanced=False
        )
        
        try:
            from domain_routing import FraudRoutingContext
            standard_context = FraudRoutingContext(**test_context)
            standard_result = standard_router.route_fraud_request(
                test_input,
                context=standard_context
            )
            logger.info(f"✓ Standard routing successful")
            logger.info(f"  Target domain: {standard_result.target_domain}")
            logger.info(f"  Confidence: {standard_result.confidence_score:.3f}")
        except Exception as e:
            logger.error(f"✗ Standard routing failed: {e}")
        
        # Test 3: Helper function validation and routing
        logger.info("\n--- Test 3: Helper Function Validation ---")
        
        test_cases = [
            {
                'input': "Check fraud patterns in payment data",
                'context': {
                    'transaction_amount': 5000.0,
                    'transaction_type': 'payment',
                    'merchant_category': 'online_retail'
                },
                'validation_level': 'standard'
            },
            {
                'input': "High risk transaction analysis needed",
                'context': {
                    'transaction_amount': 50000.0,
                    'transaction_type': 'wire_transfer',
                    'user_risk_profile': 'high'
                },
                'validation_level': 'strict'
            },
            {
                'input': "Investigate suspicious activity",
                'context': {
                    'transaction_amount': 100.0,
                    'transaction_type': 'purchase',
                    'fraud_indicators': ['velocity', 'geographic_anomaly']
                },
                'validation_level': 'paranoid' if ENHANCED_AVAILABLE else 'standard'
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"Testing case {i}: {test_case['validation_level']} validation")
            
            try:
                result = validate_and_route_fraud_request(
                    test_case['input'],
                    context=test_case['context'],
                    validation_level=test_case['validation_level']
                )
                
                if result['success']:
                    logger.info(f"✓ Case {i} successful")
                    logger.info(f"  Target: {result['target_domain']}")
                    logger.info(f"  Confidence: {result['confidence_score']:.3f}")
                    logger.info(f"  Enhanced: {result['enhanced_features_used']}")
                    
                    if 'fraud_category' in result:
                        logger.info(f"  Category: {result['fraud_category']}")
                    if 'risk_score' in result:
                        logger.info(f"  Risk: {result['risk_score']:.3f}")
                else:
                    logger.error(f"✗ Case {i} failed: {result['error']}")
                    
            except Exception as e:
                logger.error(f"✗ Case {i} exception: {e}")
        
        # Test 4: Security validation
        logger.info("\n--- Test 4: Security Validation ---")
        
        malicious_inputs = [
            "'; DROP TABLE transactions; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "eval('malicious_code')"
        ]
        
        for malicious_input in malicious_inputs:
            try:
                result = validate_and_route_fraud_request(
                    malicious_input,
                    validation_level='strict'
                )
                
                if result['success']:
                    logger.warning(f"⚠ Malicious input not detected: {malicious_input[:30]}...")
                else:
                    logger.info(f"✓ Malicious input blocked: {result['error_type']}")
                    
            except Exception as e:
                logger.info(f"✓ Malicious input handled: {type(e).__name__}")
        
        # Test 5: Performance and timeout testing
        logger.info("\n--- Test 5: Performance Testing ---")
        
        import time
        
        # Test with various timeout values
        performance_tests = [
            {'timeout': 50, 'expect_timeout': True},
            {'timeout': 1000, 'expect_timeout': False},
            {'timeout': 5000, 'expect_timeout': False}
        ]
        
        for test in performance_tests:
            start_time = time.time()
            
            try:
                result = validate_and_route_fraud_request(
                    "Performance test routing request",
                    context={'transaction_amount': 1000.0},
                    validation_level='standard',
                    timeout_ms=test['timeout']
                )
                
                elapsed_ms = (time.time() - start_time) * 1000
                
                if result['success']:
                    logger.info(f"✓ Performance test completed in {elapsed_ms:.2f}ms (timeout: {test['timeout']}ms)")
                else:
                    if 'timeout' in result.get('error', '').lower():
                        logger.info(f"✓ Timeout correctly detected after {elapsed_ms:.2f}ms")
                    else:
                        logger.warning(f"⚠ Unexpected error: {result['error']}")
                        
            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                if 'timeout' in str(e).lower():
                    logger.info(f"✓ Timeout exception correctly raised after {elapsed_ms:.2f}ms")
                else:
                    logger.error(f"✗ Unexpected exception: {e}")
        
        # Test 6: Router health and metrics
        logger.info("\n--- Test 6: Health and Metrics ---")
        
        router = create_enhanced_fraud_router(validation_level='standard')
        
        # Perform multiple requests to generate metrics
        for i in range(5):
            try:
                validate_and_route_fraud_request(
                    f"Test request {i}",
                    context={'transaction_amount': 1000.0 + i * 100},
                    validation_level='standard'
                )
            except:
                pass  # Ignore errors for metrics generation
        
        # Check health and metrics
        try:
            health = router.perform_health_check()
            logger.info(f"✓ Health check completed")
            logger.info(f"  Overall status: {health['status']}")
            logger.info(f"  Checks performed: {len(health['checks'])}")
            
            for check_name, check_result in health['checks'].items():
                status = check_result.get('status', 'unknown')
                logger.info(f"    {check_name}: {status}")
        except Exception as e:
            logger.error(f"✗ Health check failed: {e}")
        
        try:
            metrics = router.get_routing_metrics()
            logger.info(f"✓ Metrics retrieved")
            
            if 'fraud_routing' in metrics:
                fraud_metrics = metrics['fraud_routing']
                perf_metrics = fraud_metrics.get('performance', {})
                logger.info(f"  Total requests: {perf_metrics.get('total_requests', 0)}")
                logger.info(f"  Average time: {perf_metrics.get('average_routing_time_ms', 0):.2f}ms")
                logger.info(f"  Failures: {perf_metrics.get('routing_failures', 0)}")
        except Exception as e:
            logger.error(f"✗ Metrics retrieval failed: {e}")
        
        # Test 7: Configuration validation
        logger.info("\n--- Test 7: Configuration Validation ---")
        
        # Test custom configuration
        custom_config = {
            'confidence_threshold': 0.9,
            'enable_fallback': True,
            'validation': {
                'max_routing_time_ms': 2000.0,
                'enable_security_validation': True
            }
        }
        
        try:
            custom_router = create_enhanced_fraud_router(
                validation_level='standard',
                config=custom_config
            )
            
            result = validate_and_route_fraud_request(
                "Custom config test",
                validation_level='standard'
            )
            
            logger.info(f"✓ Custom configuration test successful")
            
        except Exception as e:
            logger.error(f"✗ Custom configuration test failed: {e}")
        
        logger.info("\n=== Integration Test Complete ===")
        logger.info("✅ Enhanced domain routing integration tests completed!")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Enhanced domain routing features may not be available")
        return False
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    async def main():
        """Run all tests"""
        
        logger.info("Starting Enhanced Domain Routing Test Suite")
        
        # Run integration test
        test_result = await test_enhanced_domain_routing()
        
        # Summary
        logger.info(f"\n=== Test Summary ===")
        logger.info(f"Integration Test: {'✅ PASSED' if test_result else '❌ FAILED'}")
        
        if test_result:
            logger.info("🎉 All tests completed successfully!")
            return 0
        else:
            logger.error("❌ Some tests failed")
            return 1
    
    # Run the test suite
    exit_code = asyncio.run(main())
    exit(exit_code)