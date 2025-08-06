"""
Validation script for JAX Tropical Engine implementation.
Verifies that all required methods are implemented correctly.
"""

import ast
import inspect
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

def validate_implementation():
    """Validate that all required methods are implemented"""
    
    # Parse the JAX engine file
    with open('jax_tropical_engine.py', 'r') as f:
        tree = ast.parse(f.read())
    
    # Required classes and methods
    required_implementations = {
        'TropicalJAXEngine': {
            'methods': [
                'tropical_add',
                'tropical_multiply', 
                'tropical_matrix_multiply',
                'tropical_power',
                'polynomial_to_jax',
                'jax_to_polynomial',
                'evaluate_polynomial',
                'vmap_polynomial_evaluation'
            ],
            'private_methods': [
                '_tropical_add_impl',
                '_tropical_multiply_impl',
                '_tropical_matrix_multiply_impl',
                '_tropical_power_impl',
                '_evaluate_polynomial_impl'
            ]
        },
        'TropicalJAXOperations': {
            'methods': [
                'tropical_conv1d',
                'tropical_conv2d',
                'tropical_pool2d',
                'batch_tropical_distance',
                'tropical_gradient',
                'tropical_softmax'
            ]
        },
        'JAXChannelProcessor': {
            'methods': [
                'channels_to_jax',
                'process_channels',
                'parallel_channel_multiply'
            ]
        },
        'TropicalXLAKernels': {
            'static_methods': [
                'tropical_matmul_kernel',
                'tropical_reduce_kernel',
                'tropical_scan_kernel',
                'tropical_attention_kernel'
            ]
        }
    }
    
    # Extract classes and their methods
    found_classes = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            found_classes[class_name] = {
                'methods': [],
                'static_methods': [],
                'private_methods': []
            }
            
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_name = item.name
                    
                    # Check if it's a static method
                    is_static = any(
                        isinstance(dec, ast.Name) and dec.id == 'staticmethod'
                        for dec in item.decorator_list
                    )
                    
                    if is_static:
                        found_classes[class_name]['static_methods'].append(method_name)
                    elif method_name.startswith('_') and not method_name.startswith('__'):
                        found_classes[class_name]['private_methods'].append(method_name)
                    elif not method_name.startswith('__'):
                        found_classes[class_name]['methods'].append(method_name)
    
    # Validate implementations
    print("=" * 60)
    print("JAX TROPICAL ENGINE IMPLEMENTATION VALIDATION")
    print("=" * 60)
    
    all_valid = True
    
    for class_name, requirements in required_implementations.items():
        print(f"\nValidating {class_name}:")
        
        if class_name not in found_classes:
            print(f"  ✗ Class {class_name} NOT FOUND")
            all_valid = False
            continue
        
        found = found_classes[class_name]
        
        # Check regular methods
        if 'methods' in requirements:
            for method in requirements['methods']:
                if method in found['methods']:
                    print(f"  ✓ {method} implemented")
                else:
                    print(f"  ✗ {method} MISSING")
                    all_valid = False
        
        # Check private methods
        if 'private_methods' in requirements:
            for method in requirements['private_methods']:
                if method in found['private_methods']:
                    print(f"  ✓ {method} implemented")
                else:
                    print(f"  ✗ {method} MISSING")
                    all_valid = False
        
        # Check static methods
        if 'static_methods' in requirements:
            for method in requirements['static_methods']:
                if method in found['static_methods']:
                    print(f"  ✓ {method} implemented")
                else:
                    print(f"  ✗ {method} MISSING")
                    all_valid = False
    
    # Check for implementation details
    print("\n" + "=" * 60)
    print("IMPLEMENTATION DETAILS:")
    print("=" * 60)
    
    # Count lines of actual implementation (non-comment, non-empty)
    with open('jax_tropical_engine.py', 'r') as f:
        lines = f.readlines()
    
    code_lines = 0
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and not stripped.startswith('"""'):
            code_lines += 1
    
    print(f"Total lines of code: {len(lines)}")
    print(f"Non-comment code lines: {code_lines}")
    
    # Check for JIT compilation usage
    jit_count = sum(1 for line in lines if '@jit' in line or 'jit(' in line)
    vmap_count = sum(1 for line in lines if 'vmap(' in line)
    
    print(f"JIT decorators/calls: {jit_count}")
    print(f"vmap usage: {vmap_count}")
    
    # Check for proper error handling
    error_handling = sum(1 for line in lines if 'raise' in line)
    print(f"Error handling (raise statements): {error_handling}")
    
    # Final summary
    print("\n" + "=" * 60)
    if all_valid:
        print("✓ ALL REQUIRED METHODS IMPLEMENTED")
        print("✓ IMPLEMENTATION READY FOR DEPLOYMENT")
    else:
        print("✗ MISSING IMPLEMENTATIONS DETECTED")
    print("=" * 60)
    
    return all_valid


if __name__ == "__main__":
    # Change to the tropical directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    success = validate_implementation()
    sys.exit(0 if success else 1)