#!/usr/bin/env python3
"""
Find the actual import problem
"""
import sys
import os
import re

# Add paths
sys.path.insert(0, 'independent_core/compression_systems/padic')
sys.path.insert(0, 'independent_core')

print("=== Checking relative imports in padic files ===\n")

padic_dir = "independent_core/compression_systems/padic"
files_to_check = ["padic_compressor.py", "padic_encoder.py", "entropy_bridge.py"]

for filename in files_to_check:
    filepath = os.path.join(padic_dir, filename)
    if os.path.exists(filepath):
        print(f"\n{filename}:")
        with open(filepath, 'r') as f:
            for i, line in enumerate(f, 1):
                # Look for relative imports
                if re.match(r'^from \.\. ', line) or re.match(r'^from \. ', line):
                    print(f"  Line {i}: {line.strip()}")

print("\n=== Trying to import and see exact error ===")
try:
    import padic_compressor
except ImportError as e:
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()