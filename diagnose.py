#!/usr/bin/env python3
"""
Diagnostic - find what's actually in these files
"""

import os
import ast
import sys

def find_classes_in_file(filepath):
    """Find all class definitions in a Python file"""
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        return classes
    except Exception as e:
        return f"Error: {e}"

def find_imports_in_file(filepath):
    """Find all imports in a Python file"""
    try:
        with open(filepath, 'r') as f:
            tree = ast.parse(f.read())
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"from {module} import {alias.name}")
        return imports
    except Exception as e:
        return f"Error: {e}"

# Check padic_compressor.py
padic_compressor_path = "independent_core/compression_systems/padic/padic_compressor.py"

if os.path.exists(padic_compressor_path):
    print(f"=== {padic_compressor_path} ===")
    print("\nClasses defined:")
    classes = find_classes_in_file(padic_compressor_path)
    for c in classes:
        print(f"  - {c}")
    
    print("\nImports:")
    imports = find_imports_in_file(padic_compressor_path)
    for imp in imports[:20]:  # First 20 imports
        print(f"  {imp}")
        if "overflow" in imp.lower():
            print("    ^^ FOUND OVERFLOW IMPORT!")
else:
    print(f"File not found: {padic_compressor_path}")

# Check for overflow patch imports
print("\n=== Searching for overflow patch imports ===")
padic_dir = "independent_core/compression_systems/padic"
for filename in os.listdir(padic_dir):
    if filename.endswith('.py'):
        filepath = os.path.join(padic_dir, filename)
        with open(filepath, 'r') as f:
            content = f.read()
            if 'overflow_patch' in content:
                print(f"  Found in {filename}")
                # Show the line
                for i, line in enumerate(content.split('\n')):
                    if 'overflow_patch' in line:
                        print(f"    Line {i+1}: {line.strip()}")