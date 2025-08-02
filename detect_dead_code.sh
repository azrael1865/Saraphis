#!/bin/bash

# Saraphis Dead Code Detection Script
# Run this script to detect dead code in the Saraphis AI system

echo "🔍 Saraphis Dead Code Detector"
echo "================================"
echo ""

# Check if we're in the right directory
if [ ! -f "dead_code_detector.py" ]; then
    echo "❌ Error: dead_code_detector.py not found"
    echo "Please run this script from the Saraphis directory"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required"
    exit 1
fi

echo "📁 Analyzing Saraphis codebase..."
echo ""

# Run the analysis
echo "🚀 Starting dead code analysis..."
python3 dead_code_detector.py --path . --output dead_code_report.txt

# Check if analysis was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Analysis completed successfully!"
    echo ""
    echo "📊 Results:"
    echo "  - Report saved to: dead_code_report.txt"
    echo "  - Check the report for detailed findings"
    echo ""
    echo "🔧 Next steps:"
    echo "  1. Review the dead code report"
    echo "  2. Prioritize high-severity items"
    echo "  3. Remove or implement dead code"
    echo "  4. Re-run analysis to verify cleanup"
    echo ""
    echo "🎯 Dead code detection complete!"
else
    echo ""
    echo "❌ Analysis failed. Check the error messages above."
    exit 1
fi 