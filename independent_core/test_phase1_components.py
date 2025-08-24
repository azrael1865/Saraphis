#!/usr/bin/env python3
"""
Comprehensive test to identify ALL issues in Phase 1 components.
Tests all configurations, mathematical operations, and data structures.
"""

import sys
import os
import traceback
from typing import Dict, List, Tuple

# Add the independent_core to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_component(name: str, import_path: str, test_func=None) -> Tuple[bool, str]:
    """Test a single component and return status."""
    try:
        # Try to import
        parts = import_path.rsplit('.', 1)
        if len(parts) == 2:
            module = __import__(parts[0], fromlist=[parts[1]])
            component = getattr(module, parts[1])
        else:
            module = __import__(import_path)
            component = module
        
        # Run basic test if provided
        if test_func:
            test_func(component)
        
        return True, "OK"
    except ImportError as e:
        return False, f"Import Error: {e}"
    except AttributeError as e:
        return False, f"Attribute Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def test_gac_config(component):
    """Test GAC configuration."""
    from gac_system.gac_config import GACConfig, GACConfigManager
    config = GACConfig()
    assert config.system.mode is not None
    manager = GACConfigManager(None, auto_load=False)
    manager.config = config
    assert manager.config == config

def test_brain_config(component):
    """Test Brain configuration."""
    from brain_core import BrainConfig
    config = BrainConfig()
    assert config.num_proof_engines > 0

def test_compression_config(component):
    """Test Compression configuration."""
    from compression_systems.services.compression_config import CompressionConfig
    config = CompressionConfig()
    assert config.compression_level >= 0

def test_gac_types(component):
    """Test GAC types."""
    from gac_system.gac_types import ComponentState, EventType, DirectionType
    assert ComponentState.ACTIVE
    assert EventType.GRADIENT_UPDATE
    assert DirectionType.POSITIVE

def test_padic_math(component):
    """Test p-adic mathematical operations."""
    from compression_systems.padic.padic_mathematical_operations import (
        padic_norm, padic_distance, hensel_lift
    )
    import torch
    x = torch.tensor(10.0)
    norm = padic_norm(x, p=5)
    assert norm >= 0

def test_padic_validation(component):
    """Test p-adic validation."""
    from compression_systems.padic.padic_validation import validate_single_weight
    import torch
    weight = torch.tensor(1.0)
    is_valid = validate_single_weight(weight)
    assert isinstance(is_valid, bool)

def test_tropical_algebra(component):
    """Test tropical linear algebra."""
    from compression_systems.tropical.tropical_linear_algebra import (
        TropicalMatrix, tropical_matmul
    )
    import torch
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    C = tropical_matmul(A, B)
    assert C.shape == (2, 2)

def test_padic_weight(component):
    """Test p-adic weight structure."""
    from compression_systems.padic.padic_encoder import PadicWeight
    from fractions import Fraction
    weight = PadicWeight(
        value=Fraction(1),
        prime=5,
        precision=10,
        valuation=0,
        digits=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    assert weight.p_value == 5  # Test the alias property

def test_ultrametric_tree(component):
    """Test ultrametric tree structure."""
    from compression_systems.ultrametric_tree import UltrametricTree
    tree = UltrametricTree()
    tree.insert(5, "value")
    assert tree.search(5) == "value"

# Main test execution
def run_all_tests():
    """Run all Phase 1 component tests."""
    
    print("=" * 80)
    print("PHASE 1 COMPONENT TESTING - COMPREHENSIVE")
    print("=" * 80)
    
    results = {}
    
    # Configuration & Types
    print("\n1. CONFIGURATION & TYPES")
    print("-" * 40)
    
    tests = [
        ("GACConfig", "gac_system.gac_config.GACConfig", test_gac_config),
        ("BrainConfig", "brain_core.BrainConfig", test_brain_config),
        ("DomainConfig", "golf_domain.domain_config.GolfDomainConfig", None),
        ("GACTypes", "gac_system.gac_types", test_gac_types),
        ("CompressionConfig", "compression_systems.services.compression_config.CompressionConfig", test_compression_config),
    ]
    
    for name, import_path, test_func in tests:
        success, msg = test_component(name, import_path, test_func)
        results[name] = (success, msg)
        status = "✅" if success else "❌"
        print(f"{status} {name:20} - {msg}")
    
    # Mathematical Operations
    print("\n2. MATHEMATICAL OPERATIONS")
    print("-" * 40)
    
    tests = [
        ("PadicMathOperations", "compression_systems.padic.padic_mathematical_operations", test_padic_math),
        ("PadicValidation", "compression_systems.padic.padic_validation", test_padic_validation),
        ("TropicalLinearAlgebra", "compression_systems.tropical.tropical_linear_algebra", test_tropical_algebra),
        ("TropicalPolynomial", "compression_systems.tropical.tropical_polynomial.TropicalPolynomial", None),
        ("PolytopeOperations", "compression_systems.tropical.polytope_operations.PolytopeOperations", None),
    ]
    
    for name, import_path, test_func in tests:
        success, msg = test_component(name, import_path, test_func)
        results[name] = (success, msg)
        status = "✅" if success else "❌"
        print(f"{status} {name:20} - {msg}")
    
    # Data Structures
    print("\n3. DATA STRUCTURES")
    print("-" * 40)
    
    tests = [
        ("UncertaintyMetrics", "compression_systems.uncertainty.uncertainty_metrics.UncertaintyQuantification", None),
        ("PadicWeight", "compression_systems.padic.padic_encoder.PadicWeight", test_padic_weight),
        ("SheafStructures", "compression_systems.sheaf.sheaf_structures.CellularSheaf", None),
        ("CSRSparseMatrix", "compression_systems.strategies.csr_sparse_matrix.CSRPadicMatrix", None),
        ("UltrametricTree", "compression_systems.ultrametric_tree.UltrametricTree", test_ultrametric_tree),
    ]
    
    for name, import_path, test_func in tests:
        success, msg = test_component(name, import_path, test_func)
        results[name] = (success, msg)
        status = "✅" if success else "❌"
        print(f"{status} {name:20} - {msg}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total = len(results)
    passed = sum(1 for success, _ in results.values() if success)
    failed = total - passed
    
    print(f"\nTotal Components: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed > 0:
        print("\nFAILED COMPONENTS:")
        for name, (success, msg) in results.items():
            if not success:
                print(f"  - {name}: {msg}")
    
    # List of issues to fix
    print("\n" + "=" * 80)
    print("ACTION ITEMS")
    print("=" * 80)
    
    issues = []
    for name, (success, msg) in results.items():
        if not success:
            if "Import Error" in msg:
                if "No module named" in msg:
                    module = msg.split("'")[1]
                    issues.append(f"Fix import: {name} needs module '{module}'")
                else:
                    issues.append(f"Fix import: {name} - {msg}")
            elif "Attribute Error" in msg:
                issues.append(f"Fix attribute: {name} - {msg}")
            else:
                issues.append(f"Fix error: {name} - {msg}")
    
    if issues:
        print("\nIssues to fix (in order):")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")
    else:
        print("\n✅ All Phase 1 components are working!")
    
    return results, issues

if __name__ == "__main__":
    results, issues = run_all_tests()
    
    # Exit with error code if there are failures
    failed_count = sum(1 for success, _ in results.values() if not success)
    sys.exit(failed_count)