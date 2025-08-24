#!/usr/bin/env python3
"""
Test runner for TropicalPolynomial component tests
"""

import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now run the tests
if __name__ == "__main__":
    from compression_systems.tropical.test_tropical_polynomial import *
    import unittest
    
    # Run tests with verbose output
    unittest.main(argv=[''], verbosity=2, exit=False)