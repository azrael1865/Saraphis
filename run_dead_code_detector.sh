#!/bin/bash

# Saraphis Dead Code Detector Runner
# This script runs the comprehensive dead code analysis for the Saraphis AI system

set -e  # Exit on any error

echo "🔍 Saraphis Dead Code Detector"
echo "================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not found"
    exit 1
fi

# Check if the dead code detector script exists
if [ ! -f "dead_code_detector.py" ]; then
    echo "❌ Error: dead_code_detector.py not found in current directory"
    exit 1
fi

# Check if Saraphis directory exists
if [ ! -d "Saraphis" ]; then
    echo "❌ Error: Saraphis directory not found"
    echo "Please run this script from the root directory of the Saraphis project"
    exit 1
fi

echo "📁 Analyzing Saraphis codebase..."
echo ""

# Run the dead code detector
echo "🚀 Starting analysis..."
python3 dead_code_detector.py --path Saraphis --output dead_code_report.txt --verbose

# Check if analysis was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Analysis completed successfully!"
    echo ""
    echo "📊 Summary:"
    echo "  - Report saved to: dead_code_report.txt"
    echo "  - Check the report for detailed findings"
    echo ""
    echo "🔧 Next steps:"
    echo "  1. Review the dead code report"
    echo "  2. Prioritize high-severity items"
    echo "  3. Remove or implement dead code"
    echo "  4. Re-run analysis to verify cleanup"
    echo ""
else
    echo ""
    echo "❌ Analysis failed. Check the error messages above."
    exit 1
fi

echo "🎯 Dead code detection complete!" 