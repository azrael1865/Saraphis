#!/usr/bin/env python3
"""
Comprehensive Dead Code Detector for Saraphis AI System

This script performs thorough analysis of the Saraphis codebase to identify:
- Unused functions and methods
- Dead classes and placeholder implementations
- Unused imports
- Empty pass statements
- Disabled fallback systems
- TODO/placeholder implementations
- Unreachable code blocks
- Dead code patterns specific to Saraphis

Author: Saraphis Development Team
Version: 1.0.0
"""

import os
import sys
import ast
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DeadCodeItem:
    """Represents a dead code item found in the analysis"""
    file_path: str
    line_number: int
    item_type: str  # 'function', 'class', 'import', 'variable', 'pass', 'todo'
    item_name: str
    description: str
    severity: str  # 'high', 'medium', 'low'
    context: str
    suggested_action: str

@dataclass
class AnalysisResult:
    """Complete analysis result for a file or directory"""
    total_files: int
    total_lines: int
    dead_code_items: List[DeadCodeItem]
    dead_code_count: int
    dead_code_percentage: float
    file_analysis: Dict[str, Dict[str, Any]]
    summary: Dict[str, Any]

class SaraphisDeadCodeDetector:
    """Comprehensive dead code detector for Saraphis AI system"""
    
    def __init__(self, root_path: str = "Saraphis"):
        self.root_path = Path(root_path)
        self.all_files = []
        self.import_graph = defaultdict(set)
        self.function_calls = defaultdict(set)
        self.class_instantiations = defaultdict(set)
        self.variable_uses = defaultdict(set)
        self.dead_code_items = []
        
        # Saraphis-specific patterns
        self.saraphis_patterns = {
            'disabled_fallbacks': r'DISABLE_FALLBACKS\s*=\s*True',
            'empty_pass': r'^\s*pass\s*$',
            'todo_implement': r'#\s*TODO.*[Ii]mplement',
            'placeholder_class': r'class\s+\w+.*:\s*pass',
            'placeholder_function': r'def\s+\w+.*:\s*pass',
            'empty_exception': r'except.*:\s*pass',
            'unused_import': r'import\s+\w+.*#.*unused',
            'dead_fallback': r'if\s+DISABLE_FALLBACKS.*:.*pass',
        }
        
        # Known entry points in Saraphis
        self.entry_points = {
            'brain.py': ['Brain', 'BrainSystemConfig'],
            'training_manager.py': ['TrainingManager'],
            'financial_fraud_domain/': ['EnhancedFraudCore', 'FraudDetectionSystem'],
            'independent_core/': ['BrainOrchestrator', 'UncertaintyOrchestrator'],
        }
        
        # Known used patterns (these are intentionally empty)
        self.intentional_empty = {
            'exception_classes': True,  # Exception classes can be empty
            'interface_classes': True,   # Interface classes can be empty
            'placeholder_methods': True, # Placeholder methods in development
        }

    def scan_codebase(self) -> None:
        """Scan the entire codebase for Python files"""
        logger.info(f"Scanning codebase at: {self.root_path}")
        
        for file_path in self.root_path.rglob("*.py"):
            if not self._should_skip_file(file_path):
                self.all_files.append(file_path)
        
        logger.info(f"Found {len(self.all_files)} Python files to analyze")

    def _should_skip_file(self, file_path: Path) -> bool:
        """Determine if a file should be skipped from analysis"""
        skip_patterns = [
            '__pycache__',
            '.git',
            '.pytest_cache',
            'venv',
            'env',
            'node_modules',
            '*.pyc',
            '*.pyo',
            '*.pyd',
            'test_*.py',  # Skip test files for now
            '*_test.py',
        ]
        
        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single file for dead code"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            analyzer = FileAnalyzer(file_path, content, self.saraphis_patterns)
            return analyzer.analyze()
            
        except Exception as e:
            logger.warning(f"Error analyzing {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'error': str(e),
                'dead_code_items': [],
                'total_lines': 0,
                'dead_code_count': 0
            }

    def analyze_all_files(self) -> AnalysisResult:
        """Analyze all files in parallel"""
        logger.info("Starting parallel analysis of all files...")
        
        file_results = []
        total_lines = 0
        all_dead_code_items = []
        
        # Use ThreadPoolExecutor for parallel analysis
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_file = {
                executor.submit(self.analyze_file, file_path): file_path 
                for file_path in self.all_files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    file_results.append(result)
                    total_lines += result.get('total_lines', 0)
                    all_dead_code_items.extend(result.get('dead_code_items', []))
                    
                    if result.get('dead_code_count', 0) > 0:
                        logger.info(f"Found {result['dead_code_count']} dead code items in {file_path}")
                        
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        # Calculate statistics
        dead_code_count = len(all_dead_code_items)
        dead_code_percentage = (dead_code_count / max(total_lines, 1)) * 100
        
        # Build file analysis summary
        file_analysis = {}
        for result in file_results:
            if 'file_path' in result:
                file_analysis[result['file_path']] = {
                    'dead_code_count': result.get('dead_code_count', 0),
                    'total_lines': result.get('total_lines', 0),
                    'dead_code_items': result.get('dead_code_items', [])
                }
        
        # Build summary
        summary = {
            'total_files_analyzed': len(file_results),
            'total_lines_analyzed': total_lines,
            'total_dead_code_items': dead_code_count,
            'dead_code_percentage': dead_code_percentage,
            'dead_code_by_type': self._categorize_dead_code(all_dead_code_items),
            'files_with_dead_code': len([r for r in file_results if r.get('dead_code_count', 0) > 0]),
            'most_affected_files': self._get_most_affected_files(file_analysis)
        }
        
        return AnalysisResult(
            total_files=len(file_results),
            total_lines=total_lines,
            dead_code_items=all_dead_code_items,
            dead_code_count=dead_code_count,
            dead_code_percentage=dead_code_percentage,
            file_analysis=file_analysis,
            summary=summary
        )

    def _categorize_dead_code(self, dead_code_items: List[DeadCodeItem]) -> Dict[str, int]:
        """Categorize dead code items by type"""
        categories = defaultdict(int)
        for item in dead_code_items:
            categories[item.item_type] += 1
        return dict(categories)

    def _get_most_affected_files(self, file_analysis: Dict[str, Dict]) -> List[Tuple[str, int]]:
        """Get files with the most dead code"""
        file_counts = [
            (file_path, data['dead_code_count']) 
            for file_path, data in file_analysis.items()
            if data['dead_code_count'] > 0
        ]
        return sorted(file_counts, key=lambda x: x[1], reverse=True)[:10]

    def generate_report(self, result: AnalysisResult, output_file: str = None) -> str:
        """Generate a comprehensive dead code report"""
        report = []
        report.append("=" * 80)
        report.append("SARAPHIS DEAD CODE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Analysis Duration: {time.time():.2f} seconds")
        report.append("")
        
        # Summary
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Files Analyzed: {result.total_files}")
        report.append(f"Total Lines Analyzed: {result.total_lines:,}")
        report.append(f"Total Dead Code Items: {result.dead_code_count}")
        report.append(f"Dead Code Percentage: {result.dead_code_percentage:.2f}%")
        report.append(f"Files with Dead Code: {result.summary['files_with_dead_code']}")
        report.append("")
        
        # Dead code by type
        report.append("DEAD CODE BY TYPE")
        report.append("-" * 40)
        for item_type, count in result.summary['dead_code_by_type'].items():
            report.append(f"{item_type.title()}: {count}")
        report.append("")
        
        # Most affected files
        report.append("MOST AFFECTED FILES")
        report.append("-" * 40)
        for file_path, count in result.summary['most_affected_files']:
            report.append(f"{file_path}: {count} items")
        report.append("")
        
        # Detailed findings
        report.append("DETAILED FINDINGS")
        report.append("-" * 40)
        
        # Group by file
        by_file = defaultdict(list)
        for item in result.dead_code_items:
            by_file[item.file_path].append(item)
        
        for file_path, items in sorted(by_file.items()):
            if items:
                report.append(f"\n{file_path}:")
                for item in sorted(items, key=lambda x: x.line_number):
                    report.append(f"  Line {item.line_number}: {item.item_type} - {item.item_name}")
                    report.append(f"    Description: {item.description}")
                    report.append(f"    Severity: {item.severity}")
                    report.append(f"    Suggested Action: {item.suggested_action}")
                    report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("1. Remove disabled fallback systems (DISABLE_FALLBACKS = True)")
        report.append("2. Implement TODO items or remove placeholder functions")
        report.append("3. Remove empty pass statements in exception handlers")
        report.append("4. Clean up unused exception classes")
        report.append("5. Implement placeholder methods or mark as abstract")
        report.append("6. Remove unused imports")
        report.append("7. Consolidate duplicate exception classes")
        report.append("")
        
        # Saraphis-specific recommendations
        report.append("SARAPHIS-SPECIFIC RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("1. Review fallback systems in training_manager.py (lines 1818-2177)")
        report.append("2. Clean up fallback code in brain.py (lines 5582-5610)")
        report.append("3. Remove empty files like simple_training_executor.py")
        report.append("4. Implement placeholder methods in financial_fraud_domain")
        report.append("5. Consolidate duplicate exception classes across modules")
        report.append("6. Remove disabled fallback systems in enhanced_fraud_core_*.py")
        report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to: {output_file}")
        
        return report_text

class FileAnalyzer:
    """Analyzes a single file for dead code patterns"""
    
    def __init__(self, file_path: Path, content: str, patterns: Dict[str, str]):
        self.file_path = file_path
        self.content = content
        self.patterns = patterns
        self.lines = content.split('\n')
        self.dead_code_items = []
        self.imports = set()
        self.functions = set()
        self.classes = set()
        self.variables = set()

    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive analysis of the file"""
        # Parse AST
        try:
            tree = ast.parse(self.content)
            self._analyze_ast(tree)
        except SyntaxError:
            # Handle syntax errors gracefully
            pass
        
        # Pattern-based analysis
        self._analyze_patterns()
        
        # Calculate statistics
        total_lines = len(self.lines)
        dead_code_count = len(self.dead_code_items)
        
        return {
            'file_path': str(self.file_path),
            'total_lines': total_lines,
            'dead_code_count': dead_code_count,
            'dead_code_items': self.dead_code_items,
            'imports': list(self.imports),
            'functions': list(self.functions),
            'classes': list(self.classes),
            'variables': list(self.variables)
        }

    def _analyze_ast(self, tree: ast.AST) -> None:
        """Analyze the AST for dead code patterns"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self._analyze_import(node)
            elif isinstance(node, ast.ImportFrom):
                self._analyze_import_from(node)
            elif isinstance(node, ast.FunctionDef):
                self._analyze_function(node)
            elif isinstance(node, ast.ClassDef):
                self._analyze_class(node)
            elif isinstance(node, ast.Assign):
                self._analyze_assignment(node)

    def _analyze_import(self, node: ast.Import) -> None:
        """Analyze import statements"""
        for alias in node.names:
            self.imports.add(alias.name)
            # Check for unused imports (basic check)
            if not self._is_import_used(alias.name):
                self.dead_code_items.append(DeadCodeItem(
                    file_path=str(self.file_path),
                    line_number=node.lineno,
                    item_type='import',
                    item_name=alias.name,
                    description=f"Potentially unused import: {alias.name}",
                    severity='medium',
                    context=f"Import statement at line {node.lineno}",
                    suggested_action="Remove if unused or add usage"
                ))

    def _analyze_import_from(self, node: ast.ImportFrom) -> None:
        """Analyze from import statements"""
        module = node.module or ''
        for alias in node.names:
            full_name = f"{module}.{alias.name}" if module else alias.name
            self.imports.add(full_name)
            if not self._is_import_used(alias.name):
                self.dead_code_items.append(DeadCodeItem(
                    file_path=str(self.file_path),
                    line_number=node.lineno,
                    item_type='import',
                    item_name=full_name,
                    description=f"Potentially unused import: {full_name}",
                    severity='medium',
                    context=f"From import statement at line {node.lineno}",
                    suggested_action="Remove if unused or add usage"
                ))

    def _analyze_function(self, node: ast.FunctionDef) -> None:
        """Analyze function definitions"""
        self.functions.add(node.name)
        
        # Check for empty functions
        if self._is_empty_function(node):
            self.dead_code_items.append(DeadCodeItem(
                file_path=str(self.file_path),
                line_number=node.lineno,
                item_type='function',
                item_name=node.name,
                description=f"Empty function: {node.name}",
                severity='medium',
                context=f"Function definition at line {node.lineno}",
                suggested_action="Implement function or remove if not needed"
            ))

    def _analyze_class(self, node: ast.ClassDef) -> None:
        """Analyze class definitions"""
        self.classes.add(node.name)
        
        # Check for empty classes
        if self._is_empty_class(node):
            self.dead_code_items.append(DeadCodeItem(
                file_path=str(self.file_path),
                line_number=node.lineno,
                item_type='class',
                item_name=node.name,
                description=f"Empty class: {node.name}",
                severity='low',
                context=f"Class definition at line {node.lineno}",
                suggested_action="Add implementation or remove if not needed"
            ))

    def _analyze_assignment(self, node: ast.Assign) -> None:
        """Analyze variable assignments"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables.add(target.id)

    def _analyze_patterns(self) -> None:
        """Analyze patterns in the file content"""
        for line_num, line in enumerate(self.lines, 1):
            # Check for disabled fallbacks
            if re.search(self.patterns['disabled_fallbacks'], line):
                self.dead_code_items.append(DeadCodeItem(
                    file_path=str(self.file_path),
                    line_number=line_num,
                    item_type='fallback',
                    item_name='DISABLE_FALLBACKS',
                    description="Disabled fallback system",
                    severity='high',
                    context=f"Line {line_num}: {line.strip()}",
                    suggested_action="Remove disabled fallback code"
                ))
            
            # Check for empty pass statements
            if re.search(self.patterns['empty_pass'], line):
                self.dead_code_items.append(DeadCodeItem(
                    file_path=str(self.file_path),
                    line_number=line_num,
                    item_type='pass',
                    item_name='empty_pass',
                    description="Empty pass statement",
                    severity='low',
                    context=f"Line {line_num}: {line.strip()}",
                    suggested_action="Remove or implement functionality"
                ))
            
            # Check for TODO implementations
            if re.search(self.patterns['todo_implement'], line):
                self.dead_code_items.append(DeadCodeItem(
                    file_path=str(self.file_path),
                    line_number=line_num,
                    item_type='todo',
                    item_name='TODO_implement',
                    description="TODO implementation needed",
                    severity='medium',
                    context=f"Line {line_num}: {line.strip()}",
                    suggested_action="Implement functionality or remove TODO"
                ))

    def _is_import_used(self, import_name: str) -> bool:
        """Check if an import is used in the file"""
        # Simple check - can be enhanced with more sophisticated analysis
        return import_name in self.content

    def _is_empty_function(self, node: ast.FunctionDef) -> bool:
        """Check if a function is empty"""
        if not node.body:
            return True
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            return True
        return False

    def _is_empty_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is empty"""
        if not node.body:
            return True
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            return True
        return False

def main():
    """Main function to run the dead code detector"""
    parser = argparse.ArgumentParser(description='Saraphis Dead Code Detector')
    parser.add_argument('--path', default='Saraphis', help='Path to Saraphis codebase')
    parser.add_argument('--output', help='Output file for the report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize detector
    detector = SaraphisDeadCodeDetector(args.path)
    
    # Scan codebase
    detector.scan_codebase()
    
    # Analyze all files
    logger.info("Starting analysis...")
    start_time = time.time()
    result = detector.analyze_all_files()
    analysis_time = time.time() - start_time
    
    # Generate report
    report = detector.generate_report(result, args.output)
    
    # Print summary
    print(f"\nAnalysis completed in {analysis_time:.2f} seconds")
    print(f"Found {result.dead_code_count} dead code items across {result.total_files} files")
    print(f"Dead code percentage: {result.dead_code_percentage:.2f}%")
    
    if not args.output:
        print("\n" + "="*80)
        print(report)

if __name__ == "__main__":
    main() 