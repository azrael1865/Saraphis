"""
Comprehensive Documentation Content and Structure Management

Advanced documentation content generation, template management, and structure 
organization for the Saraphis Independent Core system.

Handles all 186 Python files with sophisticated content analysis and generation.
"""

import os
import ast
import inspect
import json
import re
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Union, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import textwrap
from datetime import datetime
import importlib.util
import sys

class ContentType(Enum):
    """Types of documentation content"""
    OVERVIEW = "overview"
    DETAILED_DESCRIPTION = "detailed_description"
    API_REFERENCE = "api_reference"
    USAGE_EXAMPLES = "usage_examples"
    ARCHITECTURAL_NOTES = "architectural_notes"
    CONFIGURATION_OPTIONS = "configuration_options"
    ERROR_HANDLING = "error_handling"
    PERFORMANCE_NOTES = "performance_notes"
    SECURITY_CONSIDERATIONS = "security_considerations"
    INTEGRATION_POINTS = "integration_points"

class DocumentationTemplate(Enum):
    """Documentation templates available"""
    CLASS_TEMPLATE = "class_template"
    FUNCTION_TEMPLATE = "function_template"
    MODULE_TEMPLATE = "module_template"
    API_TEMPLATE = "api_template"
    INTEGRATION_TEMPLATE = "integration_template"
    CONFIGURATION_TEMPLATE = "configuration_template"
    SECURITY_TEMPLATE = "security_template"
    PERFORMANCE_TEMPLATE = "performance_template"

@dataclass
class ContentSection:
    """Individual content section"""
    title: str
    content: str
    content_type: ContentType
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentationStructure:
    """Complete documentation structure for a component"""
    module_name: str
    file_path: str
    overview: str
    sections: List[ContentSection] = field(default_factory=list)
    classes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    functions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    integration_points: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class SourceCodeAnalyzer:
    """Advanced source code analysis for documentation generation"""
    
    def __init__(self):
        self.patterns = {
            'config_pattern': re.compile(r'(\w+)\s*=\s*([^#\n]+)'),
            'class_pattern': re.compile(r'class\s+(\w+)(?:\([^)]*\))?:'),
            'function_pattern': re.compile(r'def\s+(\w+)\s*\([^)]*\):'),
            'import_pattern': re.compile(r'(?:from\s+(\S+)\s+)?import\s+([^#\n]+)'),
            'decorator_pattern': re.compile(r'@(\w+)'),
            'docstring_pattern': re.compile(r'"""([^"]*)"""', re.DOTALL),
            'comment_pattern': re.compile(r'#\s*(.+)'),
            'constant_pattern': re.compile(r'^([A-Z_]+)\s*=\s*(.+)$', re.MULTILINE)
        }
    
    def analyze_file_comprehensive(self, file_path: str) -> DocumentationStructure:
        """Comprehensive analysis of a Python file"""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return self._create_error_structure(file_path, f"Syntax error: {e}")
        
        module_name = self._extract_module_name(file_path)
        overview = self._extract_module_overview(content, tree)
        
        structure = DocumentationStructure(
            module_name=module_name,
            file_path=file_path,
            overview=overview
        )
        
        structure.classes = self._analyze_classes(tree, content)
        structure.functions = self._analyze_functions(tree, content)
        structure.dependencies = self._extract_dependencies(tree)
        structure.configuration = self._extract_configuration(content)
        structure.integration_points = self._identify_integration_points(tree, content)
        structure.examples = self._extract_examples(content)
        structure.metadata = self._collect_metadata(tree, content)
        
        structure.sections = self._generate_content_sections(structure)
        
        return structure
    
    def _create_error_structure(self, file_path: str, error_msg: str) -> DocumentationStructure:
        """Create documentation structure for files with errors"""
        module_name = self._extract_module_name(file_path)
        return DocumentationStructure(
            module_name=module_name,
            file_path=file_path,
            overview=f"Error analyzing file: {error_msg}",
            metadata={"analysis_error": error_msg}
        )
    
    def _extract_module_name(self, file_path: str) -> str:
        """Extract module name from file path"""
        path = Path(file_path)
        if 'independent_core' in path.parts:
            start_idx = path.parts.index('independent_core')
            module_parts = path.parts[start_idx:]
            module_name = '.'.join(module_parts).replace('.py', '')
            return module_name
        return path.stem
    
    def _extract_module_overview(self, content: str, tree: ast.AST) -> str:
        """Extract comprehensive module overview"""
        
        docstring = ast.get_docstring(tree)
        if docstring:
            return docstring.strip()
        
        comments = []
        for line in content.split('\n')[:20]:
            if line.strip().startswith('#'):
                comments.append(line.strip('#').strip())
        
        if comments:
            return '\n'.join(comments)
        
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        overview_parts = []
        if classes:
            overview_parts.append(f"Contains {len(classes)} classes: {', '.join(classes[:5])}")
        if functions:
            overview_parts.append(f"Contains {len(functions)} functions: {', '.join(functions[:5])}")
        
        return '. '.join(overview_parts) if overview_parts else "Module analysis required"
    
    def _analyze_classes(self, tree: ast.AST, content: str) -> Dict[str, Dict[str, Any]]:
        """Comprehensive class analysis"""
        classes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'docstring': ast.get_docstring(node) or "No documentation available",
                    'methods': [],
                    'attributes': [],
                    'inheritance': [base.id for base in node.bases if hasattr(base, 'id')],
                    'decorators': [d.id for d in node.decorator_list if hasattr(d, 'id')],
                    'line_number': node.lineno,
                    'is_abstract': self._is_abstract_class(node),
                    'is_dataclass': self._is_dataclass(node),
                    'complexity_score': self._calculate_class_complexity(node)
                }
                
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = {
                            'name': item.name,
                            'docstring': ast.get_docstring(item) or "No documentation",
                            'args': [arg.arg for arg in item.args.args],
                            'returns': self._extract_return_annotation(item),
                            'decorators': [d.id for d in item.decorator_list if hasattr(d, 'id')],
                            'is_property': any(hasattr(d, 'id') and d.id == 'property' for d in item.decorator_list),
                            'is_classmethod': any(hasattr(d, 'id') and d.id == 'classmethod' for d in item.decorator_list),
                            'is_staticmethod': any(hasattr(d, 'id') and d.id == 'staticmethod' for d in item.decorator_list)
                        }
                        class_info['methods'].append(method_info)
                    
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if hasattr(target, 'id'):
                                class_info['attributes'].append(target.id)
                
                classes[node.name] = class_info
        
        return classes
    
    def _analyze_functions(self, tree: ast.AST, content: str) -> Dict[str, Dict[str, Any]]:
        """Comprehensive function analysis"""
        functions = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not self._is_method(node, tree):
                func_info = {
                    'name': node.name,
                    'docstring': ast.get_docstring(node) or "No documentation available",
                    'args': [arg.arg for arg in node.args.args],
                    'defaults': len(node.args.defaults),
                    'returns': self._extract_return_annotation(node),
                    'decorators': [d.id for d in node.decorator_list if hasattr(d, 'id')],
                    'line_number': node.lineno,
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'is_generator': self._is_generator_function(node),
                    'complexity_score': self._calculate_function_complexity(node),
                    'calls_made': self._extract_function_calls(node)
                }
                
                functions[node.name] = func_info
        
        return functions
    
    def _extract_dependencies(self, tree: ast.AST) -> List[str]:
        """Extract all module dependencies"""
        dependencies = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.add(node.module)
        
        return list(dependencies)
    
    def _extract_configuration(self, content: str) -> Dict[str, Any]:
        """Extract configuration options and constants"""
        config = {}
        
        constant_matches = self.patterns['constant_pattern'].findall(content)
        for name, value in constant_matches:
            try:
                parsed_value = ast.literal_eval(value.strip())
                config[name] = {
                    'value': parsed_value,
                    'type': type(parsed_value).__name__,
                    'raw': value.strip()
                }
            except (ValueError, SyntaxError):
                config[name] = {
                    'value': value.strip(),
                    'type': 'string',
                    'raw': value.strip()
                }
        
        return config
    
    def _identify_integration_points(self, tree: ast.AST, content: str) -> List[str]:
        """Identify system integration points"""
        integration_points = []
        
        integration_indicators = [
            'manager', 'service', 'client', 'adapter', 'connector',
            'handler', 'processor', 'controller', 'gateway', 'bridge'
        ]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if any(indicator in node.name.lower() for indicator in integration_indicators):
                    integration_points.append(f"Class: {node.name}")
            elif isinstance(node, ast.FunctionDef):
                if any(indicator in node.name.lower() for indicator in integration_indicators):
                    integration_points.append(f"Function: {node.name}")
        
        return integration_points
    
    def _extract_examples(self, content: str) -> List[str]:
        """Extract usage examples from comments and docstrings"""
        examples = []
        
        example_indicators = ['example', 'usage', 'sample', 'demo']
        
        in_example = False
        current_example = []
        
        for line in content.split('\n'):
            line_lower = line.lower()
            
            if any(indicator in line_lower for indicator in example_indicators):
                if current_example:
                    examples.append('\n'.join(current_example))
                current_example = [line.strip()]
                in_example = True
            elif in_example:
                if line.strip().startswith('#') or line.strip().startswith('"""'):
                    current_example.append(line.strip())
                elif line.strip() == '':
                    current_example.append('')
                else:
                    if current_example:
                        examples.append('\n'.join(current_example))
                    current_example = []
                    in_example = False
        
        if current_example:
            examples.append('\n'.join(current_example))
        
        return examples
    
    def _collect_metadata(self, tree: ast.AST, content: str) -> Dict[str, Any]:
        """Collect comprehensive metadata about the module"""
        metadata = {
            'total_lines': len(content.split('\n')),
            'code_lines': len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]),
            'comment_lines': len([line for line in content.split('\n') if line.strip().startswith('#')]),
            'docstring_lines': self._count_docstring_lines(content),
            'total_classes': len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]),
            'total_functions': len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
            'imports_count': len([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]),
            'has_main_block': '__name__ == "__main__"' in content,
            'has_async_code': any(isinstance(node, ast.AsyncFunctionDef) for node in ast.walk(tree)),
            'complexity_indicators': self._identify_complexity_indicators(tree, content)
        }
        
        return metadata
    
    def _generate_content_sections(self, structure: DocumentationStructure) -> List[ContentSection]:
        """Generate comprehensive content sections"""
        sections = []
        
        sections.append(ContentSection(
            title="Overview",
            content=structure.overview,
            content_type=ContentType.OVERVIEW,
            priority=1
        ))
        
        if structure.classes:
            class_content = self._generate_class_documentation(structure.classes)
            sections.append(ContentSection(
                title="Classes",
                content=class_content,
                content_type=ContentType.API_REFERENCE,
                priority=2
            ))
        
        if structure.functions:
            function_content = self._generate_function_documentation(structure.functions)
            sections.append(ContentSection(
                title="Functions",
                content=function_content,
                content_type=ContentType.API_REFERENCE,
                priority=3
            ))
        
        if structure.configuration:
            config_content = self._generate_configuration_documentation(structure.configuration)
            sections.append(ContentSection(
                title="Configuration",
                content=config_content,
                content_type=ContentType.CONFIGURATION_OPTIONS,
                priority=4
            ))
        
        if structure.integration_points:
            integration_content = self._generate_integration_documentation(structure.integration_points)
            sections.append(ContentSection(
                title="Integration Points",
                content=integration_content,
                content_type=ContentType.INTEGRATION_POINTS,
                priority=5
            ))
        
        if structure.examples:
            examples_content = '\n\n'.join(structure.examples)
            sections.append(ContentSection(
                title="Examples",
                content=examples_content,
                content_type=ContentType.USAGE_EXAMPLES,
                priority=6
            ))
        
        return sections
    
    def _generate_class_documentation(self, classes: Dict[str, Dict[str, Any]]) -> str:
        """Generate comprehensive class documentation"""
        docs = []
        
        for class_name, class_info in classes.items():
            class_doc = f"### {class_name}\n\n"
            class_doc += f"{class_info['docstring']}\n\n"
            
            if class_info['inheritance']:
                class_doc += f"**Inherits from:** {', '.join(class_info['inheritance'])}\n\n"
            
            if class_info['decorators']:
                class_doc += f"**Decorators:** {', '.join(class_info['decorators'])}\n\n"
            
            if class_info['is_abstract']:
                class_doc += "**Abstract Class:** Yes\n\n"
            
            if class_info['is_dataclass']:
                class_doc += "**Dataclass:** Yes\n\n"
            
            class_doc += f"**Complexity Score:** {class_info['complexity_score']}\n\n"
            
            if class_info['methods']:
                class_doc += "#### Methods\n\n"
                for method in class_info['methods']:
                    class_doc += f"- **{method['name']}({', '.join(method['args'])})**"
                    if method['returns']:
                        class_doc += f" -> {method['returns']}"
                    class_doc += f"\n  {method['docstring']}\n"
                    
                    if method['decorators']:
                        class_doc += f"  *Decorators: {', '.join(method['decorators'])}*\n"
                    class_doc += "\n"
            
            if class_info['attributes']:
                class_doc += "#### Attributes\n\n"
                for attr in class_info['attributes']:
                    class_doc += f"- {attr}\n"
                class_doc += "\n"
            
            docs.append(class_doc)
        
        return '\n'.join(docs)
    
    def _generate_function_documentation(self, functions: Dict[str, Dict[str, Any]]) -> str:
        """Generate comprehensive function documentation"""
        docs = []
        
        for func_name, func_info in functions.items():
            func_doc = f"### {func_name}\n\n"
            func_doc += f"{func_info['docstring']}\n\n"
            
            func_doc += f"**Signature:** `{func_name}({', '.join(func_info['args'])})`"
            if func_info['returns']:
                func_doc += f" -> {func_info['returns']}"
            func_doc += "\n\n"
            
            if func_info['decorators']:
                func_doc += f"**Decorators:** {', '.join(func_info['decorators'])}\n\n"
            
            if func_info['is_async']:
                func_doc += "**Async Function:** Yes\n\n"
            
            if func_info['is_generator']:
                func_doc += "**Generator Function:** Yes\n\n"
            
            func_doc += f"**Complexity Score:** {func_info['complexity_score']}\n\n"
            
            if func_info['calls_made']:
                func_doc += f"**Function Calls:** {', '.join(func_info['calls_made'][:10])}\n\n"
            
            docs.append(func_doc)
        
        return '\n'.join(docs)
    
    def _generate_configuration_documentation(self, config: Dict[str, Any]) -> str:
        """Generate configuration documentation"""
        if not config:
            return "No configuration options found."
        
        config_doc = "### Configuration Options\n\n"
        
        for name, details in config.items():
            config_doc += f"- **{name}**: `{details['value']}` ({details['type']})\n"
        
        return config_doc
    
    def _generate_integration_documentation(self, integration_points: List[str]) -> str:
        """Generate integration points documentation"""
        if not integration_points:
            return "No integration points identified."
        
        integration_doc = "### Integration Points\n\n"
        
        for point in integration_points:
            integration_doc += f"- {point}\n"
        
        return integration_doc
    
    def _is_abstract_class(self, node: ast.ClassDef) -> bool:
        """Check if class is abstract"""
        for base in node.bases:
            if hasattr(base, 'id') and 'ABC' in base.id:
                return True
        return False
    
    def _is_dataclass(self, node: ast.ClassDef) -> bool:
        """Check if class is a dataclass"""
        return any(hasattr(d, 'id') and d.id == 'dataclass' for d in node.decorator_list)
    
    def _is_method(self, node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if function is a method"""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                if node in parent.body:
                    return True
        return False
    
    def _is_generator_function(self, node: ast.FunctionDef) -> bool:
        """Check if function is a generator"""
        for child in ast.walk(node):
            if isinstance(child, ast.Yield) or isinstance(child, ast.YieldFrom):
                return True
        return False
    
    def _extract_return_annotation(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract return type annotation"""
        if node.returns:
            if hasattr(node.returns, 'id'):
                return node.returns.id
            elif hasattr(node.returns, 'attr'):
                return node.returns.attr
        return None
    
    def _extract_function_calls(self, node: ast.FunctionDef) -> List[str]:
        """Extract function calls made within a function"""
        calls = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if hasattr(child.func, 'id'):
                    calls.append(child.func.id)
                elif hasattr(child.func, 'attr'):
                    calls.append(child.func.attr)
        
        return list(set(calls))
    
    def _calculate_class_complexity(self, node: ast.ClassDef) -> int:
        """Calculate complexity score for a class"""
        score = 0
        score += len(node.body)  # Base complexity
        score += len([n for n in node.body if isinstance(n, ast.FunctionDef)]) * 2
        score += len([n for n in ast.walk(node) if isinstance(n, ast.If)]) * 1
        score += len([n for n in ast.walk(node) if isinstance(n, ast.For)]) * 2
        score += len([n for n in ast.walk(node) if isinstance(n, ast.While)]) * 2
        return score
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate complexity score for a function"""
        score = 1  # Base complexity
        score += len([n for n in ast.walk(node) if isinstance(n, ast.If)])
        score += len([n for n in ast.walk(node) if isinstance(n, ast.For)]) * 2
        score += len([n for n in ast.walk(node) if isinstance(n, ast.While)]) * 2
        score += len([n for n in ast.walk(node) if isinstance(n, ast.Try)])
        return score
    
    def _count_docstring_lines(self, content: str) -> int:
        """Count lines containing docstrings"""
        docstring_matches = self.patterns['docstring_pattern'].findall(content)
        total_lines = 0
        for match in docstring_matches:
            total_lines += len(match.split('\n'))
        return total_lines
    
    def _identify_complexity_indicators(self, tree: ast.AST, content: str) -> List[str]:
        """Identify complexity indicators in the code"""
        indicators = []
        
        if len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]) > 5:
            indicators.append("High class count")
        
        if len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]) > 20:
            indicators.append("High function count")
        
        if any(isinstance(n, ast.AsyncFunctionDef) for n in ast.walk(tree)):
            indicators.append("Async programming")
        
        if 'threading' in content or 'multiprocessing' in content:
            indicators.append("Concurrency")
        
        if 'decorator' in content or '@' in content:
            indicators.append("Decorators used")
        
        return indicators

class TemplateManager:
    """Manages documentation templates and formatting"""
    
    def __init__(self):
        self.templates = self._load_default_templates()
    
    def _load_default_templates(self) -> Dict[str, str]:
        """Load default documentation templates"""
        return {
            DocumentationTemplate.CLASS_TEMPLATE.value: """
## {class_name}

{docstring}

### Class Information
- **File:** {file_path}
- **Line:** {line_number}
- **Inheritance:** {inheritance}
- **Decorators:** {decorators}

### Methods
{methods}

### Attributes
{attributes}
""",
            
            DocumentationTemplate.FUNCTION_TEMPLATE.value: """
## {function_name}

{docstring}

### Function Information
- **Signature:** `{signature}`
- **Returns:** {returns}
- **Decorators:** {decorators}
- **Complexity:** {complexity}

### Parameters
{parameters}
""",
            
            DocumentationTemplate.MODULE_TEMPLATE.value: """
# {module_name}

{overview}

## Module Information
- **File Path:** {file_path}
- **Total Classes:** {total_classes}
- **Total Functions:** {total_functions}
- **Dependencies:** {dependencies}

## Contents
{contents}

## Configuration
{configuration}

## Integration Points
{integration_points}

## Examples
{examples}
""",
            
            DocumentationTemplate.API_TEMPLATE.value: """
# API Documentation: {module_name}

## Overview
{overview}

## API Endpoints
{endpoints}

## Data Models
{models}

## Error Handling
{error_handling}

## Authentication
{authentication}

## Rate Limiting
{rate_limiting}
""",
            
            DocumentationTemplate.SECURITY_TEMPLATE.value: """
# Security Documentation: {module_name}

## Security Overview
{security_overview}

## Authentication & Authorization
{auth_details}

## Input Validation
{input_validation}

## Security Measures
{security_measures}

## Compliance
{compliance}

## Security Testing
{security_testing}
"""
        }
    
    def format_template(self, template_type: DocumentationTemplate, **kwargs) -> str:
        """Format a template with provided data"""
        template = self.templates.get(template_type.value, "")
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            return f"Template formatting error: Missing key {e}"

class DocumentationContent:
    """Main documentation content management system"""
    
    def __init__(self):
        self.analyzer = SourceCodeAnalyzer()
        self.template_manager = TemplateManager()
        self.structures: Dict[str, DocumentationStructure] = {}
    
    def analyze_and_structure_file(self, file_path: str) -> DocumentationStructure:
        """Analyze file and create documentation structure"""
        structure = self.analyzer.analyze_file_comprehensive(file_path)
        self.structures[structure.module_name] = structure
        return structure
    
    def generate_comprehensive_content(self, structure: DocumentationStructure) -> Dict[str, str]:
        """Generate all types of documentation content for a structure"""
        content = {}
        
        content['module'] = self._generate_module_documentation(structure)
        content['api'] = self._generate_api_documentation(structure)
        content['architecture'] = self._generate_architecture_documentation(structure)
        content['user_guide'] = self._generate_user_guide(structure)
        content['developer_guide'] = self._generate_developer_guide(structure)
        
        if self._has_security_features(structure):
            content['security'] = self._generate_security_documentation(structure)
        
        if self._has_configuration(structure):
            content['configuration'] = self._generate_configuration_guide(structure)
        
        return content
    
    def _generate_module_documentation(self, structure: DocumentationStructure) -> str:
        """Generate comprehensive module documentation"""
        return self.template_manager.format_template(
            DocumentationTemplate.MODULE_TEMPLATE,
            module_name=structure.module_name,
            overview=structure.overview,
            file_path=structure.file_path,
            total_classes=len(structure.classes),
            total_functions=len(structure.functions),
            dependencies=', '.join(structure.dependencies) if structure.dependencies else 'None',
            contents=self._generate_contents_summary(structure),
            configuration=self._format_configuration(structure.configuration),
            integration_points='\n'.join(structure.integration_points) if structure.integration_points else 'None',
            examples='\n\n'.join(structure.examples) if structure.examples else 'No examples available'
        )
    
    def _generate_api_documentation(self, structure: DocumentationStructure) -> str:
        """Generate API documentation"""
        api_content = f"# API Documentation: {structure.module_name}\n\n"
        api_content += f"{structure.overview}\n\n"
        
        if structure.classes:
            api_content += "## Classes\n\n"
            for class_name, class_info in structure.classes.items():
                api_content += f"### {class_name}\n\n"
                api_content += f"{class_info['docstring']}\n\n"
                
                if class_info['methods']:
                    api_content += "#### Methods\n\n"
                    for method in class_info['methods']:
                        api_content += f"##### {method['name']}\n\n"
                        api_content += f"```python\n{method['name']}({', '.join(method['args'])})\n```\n\n"
                        api_content += f"{method['docstring']}\n\n"
        
        if structure.functions:
            api_content += "## Functions\n\n"
            for func_name, func_info in structure.functions.items():
                api_content += f"### {func_name}\n\n"
                api_content += f"```python\n{func_name}({', '.join(func_info['args'])})\n```\n\n"
                api_content += f"{func_info['docstring']}\n\n"
        
        return api_content
    
    def _generate_architecture_documentation(self, structure: DocumentationStructure) -> str:
        """Generate architecture documentation"""
        arch_content = f"# Architecture: {structure.module_name}\n\n"
        arch_content += f"{structure.overview}\n\n"
        
        arch_content += "## Components Overview\n\n"
        arch_content += f"- **Classes:** {len(structure.classes)}\n"
        arch_content += f"- **Functions:** {len(structure.functions)}\n"
        arch_content += f"- **Dependencies:** {len(structure.dependencies)}\n\n"
        
        if structure.classes:
            arch_content += "## Class Hierarchy\n\n"
            for class_name, class_info in structure.classes.items():
                arch_content += f"- **{class_name}**"
                if class_info['inheritance']:
                    arch_content += f" (inherits from {', '.join(class_info['inheritance'])})"
                arch_content += f"\n  - Methods: {len(class_info['methods'])}\n"
                arch_content += f"  - Complexity: {class_info['complexity_score']}\n"
        
        if structure.integration_points:
            arch_content += "## Integration Points\n\n"
            for point in structure.integration_points:
                arch_content += f"- {point}\n"
        
        return arch_content
    
    def _generate_user_guide(self, structure: DocumentationStructure) -> str:
        """Generate user guide documentation"""
        user_guide = f"# User Guide: {structure.module_name}\n\n"
        user_guide += f"## Introduction\n\n{structure.overview}\n\n"
        
        user_guide += "## Getting Started\n\n"
        user_guide += f"To use {structure.module_name}, import it in your Python code:\n\n"
        user_guide += f"```python\nfrom {structure.module_name} import *\n```\n\n"
        
        if structure.examples:
            user_guide += "## Usage Examples\n\n"
            for i, example in enumerate(structure.examples, 1):
                user_guide += f"### Example {i}\n\n"
                user_guide += f"```python\n{example}\n```\n\n"
        
        if structure.configuration:
            user_guide += "## Configuration\n\n"
            user_guide += self._format_configuration(structure.configuration)
        
        return user_guide
    
    def _generate_developer_guide(self, structure: DocumentationStructure) -> str:
        """Generate developer guide documentation"""
        dev_guide = f"# Developer Guide: {structure.module_name}\n\n"
        dev_guide += f"## Overview\n\n{structure.overview}\n\n"
        
        dev_guide += "## Development Setup\n\n"
        dev_guide += f"File location: `{structure.file_path}`\n\n"
        
        if structure.dependencies:
            dev_guide += "## Dependencies\n\n"
            for dep in structure.dependencies:
                dev_guide += f"- {dep}\n"
            dev_guide += "\n"
        
        dev_guide += "## Code Structure\n\n"
        dev_guide += f"- **Total Lines:** {structure.metadata.get('total_lines', 'Unknown')}\n"
        dev_guide += f"- **Code Lines:** {structure.metadata.get('code_lines', 'Unknown')}\n"
        dev_guide += f"- **Comment Lines:** {structure.metadata.get('comment_lines', 'Unknown')}\n\n"
        
        if structure.metadata.get('complexity_indicators'):
            dev_guide += "## Complexity Indicators\n\n"
            for indicator in structure.metadata['complexity_indicators']:
                dev_guide += f"- {indicator}\n"
            dev_guide += "\n"
        
        return dev_guide
    
    def _generate_security_documentation(self, structure: DocumentationStructure) -> str:
        """Generate security documentation"""
        security_doc = f"# Security: {structure.module_name}\n\n"
        security_doc += "## Security Overview\n\n"
        security_doc += "This module implements security features and considerations.\n\n"
        
        security_classes = [name for name, info in structure.classes.items() 
                          if 'security' in name.lower() or 'auth' in name.lower()]
        
        if security_classes:
            security_doc += "## Security Classes\n\n"
            for class_name in security_classes:
                class_info = structure.classes[class_name]
                security_doc += f"### {class_name}\n\n"
                security_doc += f"{class_info['docstring']}\n\n"
        
        return security_doc
    
    def _generate_configuration_guide(self, structure: DocumentationStructure) -> str:
        """Generate configuration guide"""
        config_guide = f"# Configuration Guide: {structure.module_name}\n\n"
        config_guide += "## Configuration Options\n\n"
        config_guide += self._format_configuration(structure.configuration)
        return config_guide
    
    def _generate_contents_summary(self, structure: DocumentationStructure) -> str:
        """Generate summary of module contents"""
        summary = []
        
        if structure.classes:
            summary.append(f"**Classes ({len(structure.classes)}):** {', '.join(structure.classes.keys())}")
        
        if structure.functions:
            summary.append(f"**Functions ({len(structure.functions)}):** {', '.join(structure.functions.keys())}")
        
        return '\n'.join(summary) if summary else 'No major components identified'
    
    def _format_configuration(self, config: Dict[str, Any]) -> str:
        """Format configuration options"""
        if not config:
            return "No configuration options available."
        
        formatted = []
        for name, details in config.items():
            formatted.append(f"- **{name}**: `{details['value']}` ({details['type']})")
        
        return '\n'.join(formatted)
    
    def _has_security_features(self, structure: DocumentationStructure) -> bool:
        """Check if module has security features"""
        security_indicators = ['security', 'auth', 'encrypt', 'decrypt', 'hash', 'token']
        
        for indicator in security_indicators:
            if indicator in structure.module_name.lower():
                return True
            if any(indicator in class_name.lower() for class_name in structure.classes.keys()):
                return True
            if any(indicator in func_name.lower() for func_name in structure.functions.keys()):
                return True
        
        return False
    
    def _has_configuration(self, structure: DocumentationStructure) -> bool:
        """Check if module has configuration options"""
        return bool(structure.configuration) or 'config' in structure.module_name.lower()

if __name__ == "__main__":
    content_manager = DocumentationContent()
    
    test_file = "/home/will-casterlin/Desktop/Saraphis/independent_core/brain.py"
    structure = content_manager.analyze_and_structure_file(test_file)
    
    content = content_manager.generate_comprehensive_content(structure)
    
    print(f"Generated documentation for {structure.module_name}:")
    for doc_type, doc_content in content.items():
        print(f"\n{doc_type.upper()}:")
        print(doc_content[:500] + "..." if len(doc_content) > 500 else doc_content)