#!/usr/bin/env python
"""
Standalone test runner for GAC Types that bypasses import issues
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now run the tests directly
if __name__ == "__main__":
    # Import test modules directly, bypassing __init__.py
    import pytest
    
    # Run pytest on the test file
    exit_code = pytest.main(['test_gac_types.py', '-v', '--tb=short', '-p', 'no:cacheprovider'])
    sys.exit(exit_code)