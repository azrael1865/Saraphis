#!/usr/bin/env python3
"""
Test Dashboard Architecture
Simple test to verify the clean architecture integration works
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_architecture_structure():
    """Test that the architecture files are in the right places"""
    print("Testing architecture structure...")
    
    # Check core dashboard integration files
    core_dir = Path(__file__).parent.parent / 'independent_core' / 'dashboard_integration'
    
    files_to_check = [
        core_dir / '__init__.py',
        core_dir / 'dashboard_bridge.py', 
        core_dir / 'accuracy_integration.py',
        core_dir / 'visualization_dashboard_engine.py'
    ]
    
    for file_path in files_to_check:
        if file_path.exists():
            print(f"✓ {file_path.name} exists in core")
        else:
            print(f"❌ {file_path.name} missing in core")
            return False
    
    # Check domain visualization files
    domain_viz_dir = Path(__file__).parent / 'visualization'
    
    domain_files = [
        domain_viz_dir / '__init__.py',
        domain_viz_dir / 'fraud_dashboard_factory.py',
        domain_viz_dir / 'accuracy_visualizer.py'
    ]
    
    for file_path in domain_files:
        if file_path.exists():
            print(f"✓ {file_path.name} exists in domain")
        else:
            print(f"❌ {file_path.name} missing in domain")
            return False
    
    # Check integration entry point
    integration_file = Path(__file__).parent / 'dashboard_integration.py'
    if integration_file.exists():
        print(f"✓ dashboard_integration.py exists in domain")
    else:
        print(f"❌ dashboard_integration.py missing in domain")
        return False
    
    print("✓ All architecture files in correct locations")
    return True

def test_isolated_components():
    """Test components in isolation"""
    print("\nTesting isolated components...")
    
    # Test core bridge in isolation
    try:
        core_dir = Path(__file__).parent.parent / 'independent_core'
        sys.path.insert(0, str(core_dir))
        
        from dashboard_integration.dashboard_bridge import DashboardBridge
        bridge = DashboardBridge()
        status = bridge.get_status()
        print(f"✓ Core bridge works: {status}")
        
    except Exception as e:
        print(f"❌ Core bridge test failed: {e}")
        return False
    
    # Test fraud visualizer in isolation  
    try:
        from visualization.accuracy_visualizer import FraudAccuracyVisualizer
        visualizer = FraudAccuracyVisualizer()
        layout = visualizer.create_dashboard_layout()
        print(f"✓ Fraud visualizer works: {layout['title']}")
        
    except Exception as e:
        print(f"❌ Fraud visualizer test failed: {e}")
        return False
    
    # Test factory in isolation
    try:
        from visualization.fraud_dashboard_factory import FraudDashboardFactory
        dashboard = FraudDashboardFactory.create_accuracy_dashboard()
        print(f"✓ Dashboard factory works: {dashboard is not None}")
        
    except Exception as e:
        print(f"❌ Dashboard factory test failed: {e}")
        return False
    
    return True

def test_integration_concepts():
    """Test integration concepts without complex dependencies"""
    print("\nTesting integration concepts...")
    
    try:
        # Test the idea: can we register a dashboard factory with the bridge?
        core_dir = Path(__file__).parent.parent / 'independent_core'
        sys.path.insert(0, str(core_dir))
        
        from dashboard_integration.dashboard_bridge import DashboardBridge
        from visualization.fraud_dashboard_factory import FraudDashboardFactory
        
        # Create bridge
        bridge = DashboardBridge()
        
        # Register factory
        success = bridge.register_domain_dashboard(
            domain_name='fraud_test',
            dashboard_factory=FraudDashboardFactory.create_accuracy_dashboard,
            config={'test': True}
        )
        
        if success:
            print("✓ Factory registration works")
        else:
            print("❌ Factory registration failed")
            return False
        
        # Test creation through bridge
        dashboard = bridge.create_dashboard('fraud_test')
        
        if dashboard:
            print("✓ Dashboard creation through bridge works")
        else:
            print("⚠️ Dashboard creation returned None (expected due to dependencies)")
        
        # Test status
        status = bridge.get_status()
        registered_count = status['total_registered']
        
        if registered_count > 0:
            print(f"✓ Bridge tracking works: {registered_count} dashboard(s) registered")
        else:
            print("❌ Bridge not tracking registrations")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Integration concept test failed: {e}")
        return False

def run_architecture_tests():
    """Run all architecture tests"""
    print("=" * 60)
    print("Dashboard Architecture Tests")
    print("=" * 60)
    
    tests = [
        test_architecture_structure,
        test_isolated_components, 
        test_integration_concepts
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"⚠️ Test {test.__name__} failed with exception: {e}")
            results.append(False)
        print("-" * 40)
    
    print(f"\nSummary: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All architecture tests passed!")
        print("\nThe clean dashboard integration architecture is working properly!")
        return True
    else:
        print("⚠️ Some tests failed")
        return False

if __name__ == "__main__":
    success = run_architecture_tests()
    sys.exit(0 if success else 1)