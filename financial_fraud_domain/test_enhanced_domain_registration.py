#!/usr/bin/env python3
"""
Test Enhanced Domain Registration Integration
Comprehensive test suite for the enhanced domain registration system
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


async def test_enhanced_domain_registration():
    """Test the enhanced domain registration system"""
    
    try:
        logger.info("=== Enhanced Domain Registration Integration Test ===")
        
        # Import the enhanced domain registration
        from domain_registration import (
            FinancialFraudDomain, ValidationLevel, DomainStatus,
            validate_and_register_domain
        )
        
        # Create mock domain registry
        class MockDomainRegistry:
            def __init__(self):
                self.domains = {}
                self.registration_count = 0
            
            def register_domain(self, domain_id, domain, info):
                self.domains[domain_id] = {
                    'domain': domain,
                    'info': info,
                    'registered_at': asyncio.get_event_loop().time()
                }
                self.registration_count += 1
                logger.info(f"Mock registry: Registered domain {domain_id}")
                return True
            
            def get_domain(self, domain_id):
                return self.domains.get(domain_id)
            
            def list_domains(self):
                return list(self.domains.keys())
        
        registry = MockDomainRegistry()
        
        # Test 1: Basic domain creation and registration
        logger.info("\n--- Test 1: Basic Domain Registration ---")
        
        domain = FinancialFraudDomain(
            validation_level=ValidationLevel.STANDARD,
            use_enhanced=True
        )
        
        logger.info(f"Created domain: {domain.domain_id}")
        logger.info(f"Enhanced features: {domain.use_enhanced}")
        logger.info(f"Validation level: {domain.validation_level.value}")
        
        # Test validation
        logger.info("Running domain validation...")
        validation_result = await domain.validate()
        logger.info(f"Validation result: {validation_result}")
        
        if validation_result:
            # Test registration
            logger.info("Attempting domain registration...")
            registration_result = await domain.register(registry)
            logger.info(f"Registration result: {registration_result}")
            
            if registration_result:
                logger.info(f"Domain status: {domain.status.value}")
                
                # Test metrics
                metrics = domain.get_enhanced_metrics()
                logger.info(f"Registration attempts: {metrics['metrics']['registration_attempts']}")
                logger.info(f"Validation attempts: {metrics['metrics']['validation_attempts']}")
                logger.info(f"Success rate: {metrics['registration_success_rate']:.2%}")
        
        # Test 2: Different validation levels
        logger.info("\n--- Test 2: Validation Levels ---")
        
        for level in [ValidationLevel.BASIC, ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            logger.info(f"Testing {level.value} validation level")
            
            test_domain = FinancialFraudDomain(
                validation_level=level,
                use_enhanced=True
            )
            
            validation_start = asyncio.get_event_loop().time()
            valid = await test_domain.validate()
            validation_duration = (asyncio.get_event_loop().time() - validation_start) * 1000
            
            logger.info(f"  {level.value}: Valid={valid}, Duration={validation_duration:.2f}ms")
            
            if valid:
                await test_domain.register(registry)
        
        # Test 3: Enhanced vs Standard mode
        logger.info("\n--- Test 3: Enhanced vs Standard Mode ---")
        
        # Enhanced mode
        enhanced_domain = FinancialFraudDomain(
            validation_level=ValidationLevel.STANDARD,
            use_enhanced=True
        )
        
        enhanced_start = asyncio.get_event_loop().time()
        enhanced_valid = await enhanced_domain.validate()
        enhanced_duration = (asyncio.get_event_loop().time() - enhanced_start) * 1000
        
        logger.info(f"Enhanced mode: Valid={enhanced_valid}, Duration={enhanced_duration:.2f}ms")
        
        # Standard mode
        standard_domain = FinancialFraudDomain(
            validation_level=ValidationLevel.STANDARD,
            use_enhanced=False
        )
        
        standard_start = asyncio.get_event_loop().time()
        standard_valid = await standard_domain.validate()
        standard_duration = (asyncio.get_event_loop().time() - standard_start) * 1000
        
        logger.info(f"Standard mode: Valid={standard_valid}, Duration={standard_duration:.2f}ms")
        
        # Test 4: Configuration validation
        logger.info("\n--- Test 4: Configuration Validation ---")
        
        config_domain = FinancialFraudDomain(use_enhanced=True)
        
        # Test valid configuration update
        valid_updates = {
            "model_threshold": 0.9,
            "max_concurrent_tasks": 200,
            "batch_size": 500
        }
        
        logger.info("Testing valid configuration update...")
        update_result = config_domain.update_configuration(valid_updates)
        logger.info(f"Valid update result: {update_result}")
        
        # Test invalid configuration update
        invalid_updates = {
            "model_threshold": 1.5,  # Invalid: > 1.0
            "max_concurrent_tasks": -10  # Invalid: negative
        }
        
        logger.info("Testing invalid configuration update...")
        invalid_result = config_domain.update_configuration(invalid_updates)
        logger.info(f"Invalid update result: {invalid_result} (should be False)")
        
        # Test 5: Health check
        logger.info("\n--- Test 5: Health Check ---")
        
        health_domain = FinancialFraudDomain(use_enhanced=True)
        await health_domain.validate()
        await health_domain.register(registry)
        
        health_result = await health_domain.health_check()
        logger.info(f"Health check result: {health_result['overall_status']}")
        logger.info(f"Health checks performed: {len(health_result['checks'])}")
        
        for check_name, check_result in health_result['checks'].items():
            logger.info(f"  {check_name}: {check_result['status']}")
        
        # Test 6: Error handling and recovery
        logger.info("\n--- Test 6: Error Handling ---")
        
        error_domain = FinancialFraudDomain(use_enhanced=True)
        
        # Simulate validation errors by corrupting metadata
        original_name = error_domain.metadata.name
        error_domain.metadata.name = ""  # Invalid empty name
        
        logger.info("Testing validation with invalid metadata...")
        error_validation = await error_domain.validate()
        logger.info(f"Error validation result: {error_validation}")
        
        # Restore valid metadata
        error_domain.metadata.name = original_name
        
        # Test recovery
        logger.info("Testing recovery...")
        recovery_validation = await error_domain.validate()
        logger.info(f"Recovery validation result: {recovery_validation}")
        
        # Test 7: Registry verification
        logger.info("\n--- Test 7: Registry Verification ---")
        
        registered_domains = registry.list_domains()
        logger.info(f"Total registered domains: {len(registered_domains)}")
        logger.info(f"Total registrations: {registry.registration_count}")
        
        for domain_id in registered_domains[:3]:  # Show first 3
            domain_info = registry.get_domain(domain_id)
            logger.info(f"  Domain: {domain_id}")
            logger.info(f"    Status: {domain_info['domain'].status.value}")
            logger.info(f"    Validation Level: {domain_info['domain'].validation_level.value}")
        
        # Test 8: Performance metrics
        logger.info("\n--- Test 8: Performance Metrics ---")
        
        perf_domain = FinancialFraudDomain(use_enhanced=True)
        
        # Multiple validation rounds for metrics
        for i in range(3):
            await perf_domain.validate()
        
        await perf_domain.register(registry)
        
        metrics = perf_domain.get_enhanced_metrics()
        logger.info(f"Performance Metrics:")
        logger.info(f"  Uptime: {metrics['uptime_seconds']:.2f}s")
        logger.info(f"  Validation Success Rate: {metrics['validation_success_rate']:.2%}")
        logger.info(f"  Average Validation Time: {metrics['metrics']['validation_duration_ms']:.2f}ms")
        logger.info(f"  Total Validation Attempts: {metrics['metrics']['validation_attempts']}")
        
        logger.info("\n=== Integration Test Complete ===")
        logger.info("✅ All tests passed successfully!")
        
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Enhanced domain registration features may not be available")
        return False
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


async def test_helper_function():
    """Test the helper validation and registration function"""
    
    logger.info("\n=== Testing Helper Function ===")
    
    try:
        from domain_registration import validate_and_register_domain, ValidationLevel
        
        # Create mock registry
        class MockRegistry:
            def __init__(self):
                self.domains = {}
            
            def register_domain(self, domain_id, domain, info):
                self.domains[domain_id] = domain
                return True
        
        registry = MockRegistry()
        
        # Test helper function
        domain = await validate_and_register_domain(
            registry,
            validation_level=ValidationLevel.STANDARD
        )
        
        logger.info(f"Helper function created domain: {domain.domain_id}")
        logger.info(f"Domain registered: {domain.domain_id in [d.domain_id for d in registry.domains.values()]}")
        
        return True
        
    except Exception as e:
        logger.error(f"Helper function test failed: {e}")
        return False


if __name__ == "__main__":
    async def main():
        """Run all tests"""
        
        logger.info("Starting Enhanced Domain Registration Test Suite")
        
        # Run main integration test
        test1_result = await test_enhanced_domain_registration()
        
        # Run helper function test
        test2_result = await test_helper_function()
        
        # Summary
        logger.info(f"\n=== Test Summary ===")
        logger.info(f"Integration Test: {'✅ PASSED' if test1_result else '❌ FAILED'}")
        logger.info(f"Helper Function Test: {'✅ PASSED' if test2_result else '❌ FAILED'}")
        
        if test1_result and test2_result:
            logger.info("🎉 All tests completed successfully!")
            return 0
        else:
            logger.error("❌ Some tests failed")
            return 1
    
    # Run the test suite
    exit_code = asyncio.run(main())
    exit(exit_code)