#!/usr/bin/env python3
import sys

# Read the file
with open('independent_core/compression_systems/padic/entropy_bridge.py', 'r') as f:
    lines = f.readlines()

# Find and fix the problematic imports
new_lines = []
skip_until = None

for i, line in enumerate(lines):
    # Skip the try-except block with relative imports
    if 'from ..encoding.huffman_arithmetic import' in line or 'from ..encoding' in line:
        # Replace with direct imports
        new_lines.append('from encoding.huffman_arithmetic import (\n')
        new_lines.append('    HuffmanEncoder, ArithmeticEncoder, HybridEncoder,\n')
        new_lines.append('    CompressionMetrics\n')
        new_lines.append(')\n')
        skip_until = i + 10  # Skip the next few lines of the import
    elif 'from .padic_logarithmic_encoder' in line:
        new_lines.append('from padic_logarithmic_encoder import LogarithmicPadicWeight\n')
    elif 'from .padic_encoder' in line:
        new_lines.append('from padic_encoder import PadicWeight\n')
    elif skip_until and i < skip_until:
        # Skip lines that are part of the try-except block
        if 'except' in line or 'compression_systems.encoding' in line:
            skip_until = None
    else:
        new_lines.append(line)

# Write back
with open('independent_core/compression_systems/padic/entropy_bridge.py', 'w') as f:
    f.writelines(new_lines)

print("Fixed entropy_bridge.py")
