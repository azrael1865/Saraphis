"""
Documentation Generator - Automated Multi-Format Documentation Generation

Advanced automated documentation generation system with multi-format output support
for comprehensive documentation of all 186 Python files in Saraphis Independent Core.

Supports HTML, PDF, Markdown, RST, JSON output with sophisticated templating and styling.
"""

import os
import json
import asyncio
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import subprocess
import shutil
import tempfile
import base64
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import importlib.util

class GeneratorType(Enum):
    """Types of documentation generators"""
    HTML_GENERATOR = "html_generator"
    PDF_GENERATOR = "pdf_generator"
    MARKDOWN_GENERATOR = "markdown_generator"
    RST_GENERATOR = "rst_generator"
    JSON_GENERATOR = "json_generator"
    INTERACTIVE_GENERATOR = "interactive_generator"

class OutputFormat(Enum):
    """Supported output formats"""
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    RST = "rst"
    JSON = "json"
    EPUB = "epub"

class DocumentationTheme(Enum):
    """Available documentation themes"""
    DEFAULT = "default"
    MODERN = "modern"
    TECHNICAL = "technical"
    MINIMAL = "minimal"
    CORPORATE = "corporate"

@dataclass
class GenerationConfig:
    """Configuration for documentation generation"""
    output_format: OutputFormat
    theme: DocumentationTheme = DocumentationTheme.DEFAULT
    include_source_code: bool = True
    include_diagrams: bool = True
    include_metrics: bool = True
    generate_index: bool = True
    compress_output: bool = False
    custom_css: Optional[str] = None
    custom_templates: Dict[str, str] = field(default_factory=dict)
    output_filename: Optional[str] = None

@dataclass
class GenerationResult:
    """Result of documentation generation"""
    success: bool
    output_path: str
    format_type: OutputFormat
    file_size: int = 0
    generation_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class HTMLGenerator:
    """Advanced HTML documentation generator"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.templates = self._load_html_templates()
    
    def _load_html_templates(self) -> Dict[str, str]:
        """Load HTML templates based on theme"""
        base_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>{styles}</style>
    {custom_css}
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>{title}</h1>
            <nav class="navigation">{navigation}</nav>
        </header>
        <main class="content">
            {content}
        </main>
        <footer class="footer">
            <p>Generated on {timestamp} by Saraphis Documentation System</p>
        </footer>
    </div>
    <script>{scripts}</script>
</body>
</html>
"""
        
        module_template = """
<section class="module" id="{module_id}">
    <header class="module-header">
        <h2>{module_name}</h2>
        <div class="module-meta">
            <span class="file-path">{file_path}</span>
            <span class="last-modified">{last_modified}</span>
        </div>
    </header>
    
    <div class="module-overview">
        <h3>Overview</h3>
        <p>{overview}</p>
    </div>
    
    {classes_section}
    {functions_section}
    {configuration_section}
    {examples_section}
</section>
"""
        
        return {
            'base': base_template,
            'module': module_template,
            'styles': self._get_theme_styles(),
            'scripts': self._get_javascript_code()
        }
    
    def _get_theme_styles(self) -> str:
        """Get CSS styles based on selected theme"""
        
        if self.config.theme == DocumentationTheme.MODERN:
            return """
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6; color: #333; background: #f8f9fa;
            }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .header { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;
            }
            .header h1 { font-size: 2.5rem; margin-bottom: 1rem; }
            .navigation { display: flex; gap: 1rem; }
            .navigation a { 
                color: rgba(255,255,255,0.9); text-decoration: none; 
                padding: 0.5rem 1rem; border-radius: 5px; transition: all 0.3s;
            }
            .navigation a:hover { background: rgba(255,255,255,0.2); }
            .module { 
                background: white; border-radius: 10px; padding: 2rem; 
                margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .module-header { border-bottom: 2px solid #eee; padding-bottom: 1rem; margin-bottom: 2rem; }
            .module-header h2 { color: #667eea; font-size: 1.8rem; }
            .module-meta { margin-top: 0.5rem; color: #666; font-size: 0.9rem; }
            .code-block { 
                background: #f8f9fa; border: 1px solid #e9ecef; 
                border-radius: 5px; padding: 1rem; margin: 1rem 0; 
                font-family: 'Consolas', 'Monaco', monospace;
            }
            .footer { text-align: center; margin-top: 3rem; color: #666; }
            """
        
        elif self.config.theme == DocumentationTheme.TECHNICAL:
            return """
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Courier New', monospace; 
                line-height: 1.6; color: #0f0f23; background: #ffffff;
            }
            .container { max-width: 1000px; margin: 0 auto; padding: 20px; }
            .header { 
                border: 2px solid #0f0f23; padding: 1rem; margin-bottom: 2rem;
                background: #0f0f23; color: #cccccc;
            }
            .header h1 { font-size: 1.5rem; font-weight: bold; }
            .module { 
                border: 1px solid #cccccc; padding: 1rem; margin-bottom: 1rem;
            }
            .module-header { border-bottom: 1px solid #cccccc; margin-bottom: 1rem; }
            .module-header h2 { font-size: 1.2rem; font-weight: bold; }
            .code-block { 
                background: #f0f0f0; border: 1px solid #cccccc; 
                padding: 0.5rem; font-family: inherit;
            }
            """
        
        else:  # DEFAULT theme
            return """
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: Arial, sans-serif; line-height: 1.6; 
                color: #333; background: #fff; padding: 20px;
            }
            .container { max-width: 1000px; margin: 0 auto; }
            .header { 
                background: #f4f4f4; padding: 20px; border-radius: 5px; 
                margin-bottom: 20px;
            }
            .header h1 { color: #333; }
            .navigation { margin-top: 15px; }
            .navigation a { 
                color: #0066cc; text-decoration: none; margin-right: 15px;
            }
            .navigation a:hover { text-decoration: underline; }
            .module { 
                border: 1px solid #ddd; padding: 20px; margin-bottom: 20px;
                border-radius: 5px;
            }
            .module-header { border-bottom: 1px solid #eee; padding-bottom: 15px; }
            .module-header h2 { color: #0066cc; }
            .module-meta { color: #666; font-size: 0.9em; margin-top: 5px; }
            .code-block { 
                background: #f8f8f8; border: 1px solid #ddd; 
                padding: 15px; margin: 15px 0; border-radius: 3px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
            .footer { 
                border-top: 1px solid #eee; padding-top: 20px; 
                margin-top: 40px; text-align: center; color: #666;
            }
            """
    
    def _get_javascript_code(self) -> str:
        """Get JavaScript code for interactive features"""
        return """
        // Search functionality
        function initSearch() {
            const searchInput = document.getElementById('search-input');
            if (!searchInput) return;
            
            searchInput.addEventListener('input', function(e) {
                const query = e.target.value.toLowerCase();
                const modules = document.querySelectorAll('.module');
                
                modules.forEach(module => {
                    const content = module.textContent.toLowerCase();
                    if (content.includes(query)) {
                        module.style.display = 'block';
                    } else {
                        module.style.display = 'none';
                    }
                });
            });
        }
        
        // Collapsible sections
        function initCollapsible() {
            const headers = document.querySelectorAll('.collapsible-header');
            headers.forEach(header => {
                header.addEventListener('click', function() {
                    const content = this.nextElementSibling;
                    if (content.style.display === 'none') {
                        content.style.display = 'block';
                        this.classList.add('expanded');
                    } else {
                        content.style.display = 'none';
                        this.classList.remove('expanded');
                    }
                });
            });
        }
        
        // Copy code functionality
        function initCodeCopy() {
            const codeBlocks = document.querySelectorAll('.code-block');
            codeBlocks.forEach(block => {
                const copyBtn = document.createElement('button');
                copyBtn.textContent = 'Copy';
                copyBtn.className = 'copy-btn';
                copyBtn.onclick = () => {
                    navigator.clipboard.writeText(block.textContent);
                    copyBtn.textContent = 'Copied!';
                    setTimeout(() => copyBtn.textContent = 'Copy', 2000);
                };
                block.appendChild(copyBtn);
            });
        }
        
        // Initialize all features
        document.addEventListener('DOMContentLoaded', function() {
            initSearch();
            initCollapsible();
            initCodeCopy();
        });
        """
    
    async def generate(self, documentation_data: Dict[str, Any], output_path: str) -> GenerationResult:
        """Generate HTML documentation"""
        start_time = datetime.now()
        result = GenerationResult(
            success=False,
            output_path=output_path,
            format_type=OutputFormat.HTML
        )
        
        try:
            # Generate navigation
            navigation = self._generate_navigation(documentation_data)
            
            # Generate content sections
            content_sections = []
            for module_name, module_data in documentation_data.items():
                section = self._generate_module_section(module_name, module_data)
                content_sections.append(section)
            
            # Combine all content
            full_content = '\n'.join(content_sections)
            
            # Apply base template
            html_output = self.templates['base'].format(
                title="Saraphis Independent Core Documentation",
                styles=self.templates['styles'],
                custom_css=f"<style>{self.config.custom_css}</style>" if self.config.custom_css else "",
                navigation=navigation,
                content=full_content,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                scripts=self.templates['scripts']
            )
            
            # Write output file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_output)
            
            # Generate additional resources
            await self._generate_resources(os.path.dirname(output_path))
            
            result.success = True
            result.file_size = len(html_output.encode('utf-8'))
            result.generation_time = (datetime.now() - start_time).total_seconds()
            
        except Exception as e:
            result.errors.append(f"HTML generation failed: {str(e)}")
            self.logger.error(f"HTML generation error: {e}")
        
        return result
    
    def _generate_navigation(self, documentation_data: Dict[str, Any]) -> str:
        """Generate navigation menu"""
        nav_items = []
        
        # Add search functionality
        nav_items.append('<input type="text" id="search-input" placeholder="Search documentation..." style="margin-right: 20px; padding: 5px;">')
        
        # Add module links
        for module_name in sorted(documentation_data.keys()):
            safe_name = module_name.replace('.', '_')
            nav_items.append(f'<a href="#{safe_name}">{module_name}</a>')
        
        return '\n'.join(nav_items)
    
    def _generate_module_section(self, module_name: str, module_data: Dict[str, Any]) -> str:
        """Generate HTML section for a module"""
        safe_name = module_name.replace('.', '_')
        
        # Generate classes section
        classes_html = self._generate_classes_html(module_data.get('classes', {}))
        
        # Generate functions section
        functions_html = self._generate_functions_html(module_data.get('functions', {}))
        
        # Generate configuration section
        config_html = self._generate_configuration_html(module_data.get('configuration', {}))
        
        # Generate examples section
        examples_html = self._generate_examples_html(module_data.get('examples', []))
        
        return self.templates['module'].format(
            module_id=safe_name,
            module_name=module_name,
            file_path=module_data.get('file_path', 'Unknown'),
            last_modified=module_data.get('last_modified', 'Unknown'),
            overview=module_data.get('overview', 'No overview available'),
            classes_section=classes_html,
            functions_section=functions_html,
            configuration_section=config_html,
            examples_section=examples_html
        )
    
    def _generate_classes_html(self, classes: Dict[str, Any]) -> str:
        """Generate HTML for classes section"""
        if not classes:
            return ""
        
        html = '<div class="classes-section"><h3 class="collapsible-header">Classes</h3><div class="classes-content">'
        
        for class_name, class_info in classes.items():
            html += f'<div class="class-item"><h4>{class_name}</h4>'
            html += f'<p>{class_info.get("docstring", "No documentation")}</p>'
            
            if class_info.get('methods'):
                html += '<h5>Methods</h5><ul class="methods-list">'
                for method in class_info['methods']:
                    html += f'<li><strong>{method["name"]}</strong>: {method.get("docstring", "No documentation")}</li>'
                html += '</ul>'
            
            html += '</div>'
        
        html += '</div></div>'
        return html
    
    def _generate_functions_html(self, functions: Dict[str, Any]) -> str:
        """Generate HTML for functions section"""
        if not functions:
            return ""
        
        html = '<div class="functions-section"><h3 class="collapsible-header">Functions</h3><div class="functions-content">'
        
        for func_name, func_info in functions.items():
            html += f'<div class="function-item"><h4>{func_name}</h4>'
            html += f'<p>{func_info.get("docstring", "No documentation")}</p>'
            
            if func_info.get('args'):
                html += f'<p><strong>Arguments:</strong> {", ".join(func_info["args"])}</p>'
            
            if func_info.get('returns'):
                html += f'<p><strong>Returns:</strong> {func_info["returns"]}</p>'
            
            html += '</div>'
        
        html += '</div></div>'
        return html
    
    def _generate_configuration_html(self, config: Dict[str, Any]) -> str:
        """Generate HTML for configuration section"""
        if not config:
            return ""
        
        html = '<div class="configuration-section"><h3 class="collapsible-header">Configuration</h3><div class="configuration-content">'
        html += '<table class="config-table"><thead><tr><th>Name</th><th>Value</th><th>Type</th></tr></thead><tbody>'
        
        for name, details in config.items():
            html += f'<tr><td>{name}</td><td><code>{details.get("value", "N/A")}</code></td><td>{details.get("type", "Unknown")}</td></tr>'
        
        html += '</tbody></table></div></div>'
        return html
    
    def _generate_examples_html(self, examples: List[str]) -> str:
        """Generate HTML for examples section"""
        if not examples:
            return ""
        
        html = '<div class="examples-section"><h3 class="collapsible-header">Examples</h3><div class="examples-content">'
        
        for i, example in enumerate(examples, 1):
            html += f'<div class="example-item"><h4>Example {i}</h4>'
            html += f'<div class="code-block">{example}</div></div>'
        
        html += '</div></div>'
        return html
    
    async def _generate_resources(self, output_dir: str):
        """Generate additional resources like CSS, JS files"""
        resources_dir = os.path.join(output_dir, 'resources')
        os.makedirs(resources_dir, exist_ok=True)
        
        # Write separate CSS file if needed
        if self.config.custom_css:
            css_path = os.path.join(resources_dir, 'custom.css')
            with open(css_path, 'w') as f:
                f.write(self.config.custom_css)

class PDFGenerator:
    """Advanced PDF documentation generator"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def generate(self, documentation_data: Dict[str, Any], output_path: str) -> GenerationResult:
        """Generate PDF documentation"""
        start_time = datetime.now()
        result = GenerationResult(
            success=False,
            output_path=output_path,
            format_type=OutputFormat.PDF
        )
        
        try:
            # First generate HTML version
            html_generator = HTMLGenerator(self.config)
            temp_html_path = output_path.replace('.pdf', '_temp.html')
            
            html_result = await html_generator.generate(documentation_data, temp_html_path)
            
            if not html_result.success:
                result.errors.extend(html_result.errors)
                return result
            
            # Convert HTML to PDF using various methods
            pdf_generated = await self._convert_html_to_pdf(temp_html_path, output_path)
            
            if pdf_generated:
                result.success = True
                result.file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
            else:
                result.errors.append("PDF conversion failed")
            
            # Cleanup temporary HTML file
            if os.path.exists(temp_html_path):
                os.remove(temp_html_path)
            
            result.generation_time = (datetime.now() - start_time).total_seconds()
            
        except Exception as e:
            result.errors.append(f"PDF generation failed: {str(e)}")
            self.logger.error(f"PDF generation error: {e}")
        
        return result
    
    async def _convert_html_to_pdf(self, html_path: str, pdf_path: str) -> bool:
        """Convert HTML to PDF using available tools"""
        
        # Try weasyprint first
        try:
            import weasyprint
            html_doc = weasyprint.HTML(filename=html_path)
            html_doc.write_pdf(pdf_path)
            return True
        except ImportError:
            self.logger.warning("weasyprint not available")
        except Exception as e:
            self.logger.error(f"weasyprint conversion failed: {e}")
        
        # Try wkhtmltopdf as fallback
        try:
            result = subprocess.run([
                'wkhtmltopdf', 
                '--page-size', 'A4',
                '--margin-top', '20mm',
                '--margin-right', '20mm',
                '--margin-bottom', '20mm',
                '--margin-left', '20mm',
                html_path, 
                pdf_path
            ], capture_output=True, text=True, timeout=300)
            
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.error(f"wkhtmltopdf conversion failed: {e}")
        
        # Try pandoc as final fallback
        try:
            result = subprocess.run([
                'pandoc', 
                html_path,
                '-o', pdf_path,
                '--pdf-engine=xelatex'
            ], capture_output=True, text=True, timeout=300)
            
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.error(f"pandoc conversion failed: {e}")
        
        return False

class MarkdownGenerator:
    """Advanced Markdown documentation generator"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def generate(self, documentation_data: Dict[str, Any], output_path: str) -> GenerationResult:
        """Generate Markdown documentation"""
        start_time = datetime.now()
        result = GenerationResult(
            success=False,
            output_path=output_path,
            format_type=OutputFormat.MARKDOWN
        )
        
        try:
            markdown_content = self._generate_markdown_content(documentation_data)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            result.success = True
            result.file_size = len(markdown_content.encode('utf-8'))
            result.generation_time = (datetime.now() - start_time).total_seconds()
            
        except Exception as e:
            result.errors.append(f"Markdown generation failed: {str(e)}")
            self.logger.error(f"Markdown generation error: {e}")
        
        return result
    
    def _generate_markdown_content(self, documentation_data: Dict[str, Any]) -> str:
        """Generate comprehensive Markdown content"""
        content = [
            "# Saraphis Independent Core Documentation",
            "",
            f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Table of Contents",
            ""
        ]
        
        # Generate table of contents
        for module_name in sorted(documentation_data.keys()):
            safe_name = module_name.replace('.', '').replace('_', '')
            content.append(f"- [{module_name}](#{safe_name.lower()})")
        
        content.extend(["", "---", ""])
        
        # Generate module documentation
        for module_name in sorted(documentation_data.keys()):
            module_data = documentation_data[module_name]
            content.extend(self._generate_module_markdown(module_name, module_data))
            content.extend(["", "---", ""])
        
        return "\n".join(content)
    
    def _generate_module_markdown(self, module_name: str, module_data: Dict[str, Any]) -> List[str]:
        """Generate Markdown for a single module"""
        safe_name = module_name.replace('.', '').replace('_', '')
        content = [
            f"## {module_name} {{#{safe_name.lower()}}}",
            "",
            f"**File:** `{module_data.get('file_path', 'Unknown')}`",
            "",
            module_data.get('overview', 'No overview available'),
            ""
        ]
        
        # Classes section
        classes = module_data.get('classes', {})
        if classes:
            content.extend(["### Classes", ""])
            for class_name, class_info in classes.items():
                content.extend([
                    f"#### {class_name}",
                    "",
                    class_info.get('docstring', 'No documentation available'),
                    ""
                ])
                
                if class_info.get('methods'):
                    content.extend(["##### Methods", ""])
                    for method in class_info['methods']:
                        args_str = ', '.join(method.get('args', []))
                        content.append(f"- **{method['name']}({args_str})**: {method.get('docstring', 'No documentation')}")
                    content.append("")
        
        # Functions section
        functions = module_data.get('functions', {})
        if functions:
            content.extend(["### Functions", ""])
            for func_name, func_info in functions.items():
                args_str = ', '.join(func_info.get('args', []))
                content.extend([
                    f"#### {func_name}({args_str})",
                    "",
                    func_info.get('docstring', 'No documentation available'),
                    ""
                ])
                
                if func_info.get('returns'):
                    content.extend([f"**Returns:** {func_info['returns']}", ""])
        
        # Configuration section
        config = module_data.get('configuration', {})
        if config:
            content.extend(["### Configuration", ""])
            for name, details in config.items():
                content.append(f"- **{name}**: `{details.get('value', 'N/A')}` ({details.get('type', 'Unknown')})")
            content.append("")
        
        # Examples section
        examples = module_data.get('examples', [])
        if examples:
            content.extend(["### Examples", ""])
            for i, example in enumerate(examples, 1):
                content.extend([
                    f"#### Example {i}",
                    "",
                    "```python",
                    example,
                    "```",
                    ""
                ])
        
        return content

class JSONGenerator:
    """Advanced JSON documentation generator"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def generate(self, documentation_data: Dict[str, Any], output_path: str) -> GenerationResult:
        """Generate JSON documentation"""
        start_time = datetime.now()
        result = GenerationResult(
            success=False,
            output_path=output_path,
            format_type=OutputFormat.JSON
        )
        
        try:
            # Enhance data with metadata
            enhanced_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "generator_version": "1.0.0",
                    "total_modules": len(documentation_data),
                    "generation_config": {
                        "theme": self.config.theme.value,
                        "include_source_code": self.config.include_source_code,
                        "include_diagrams": self.config.include_diagrams,
                        "include_metrics": self.config.include_metrics
                    }
                },
                "documentation": documentation_data,
                "statistics": self._generate_statistics(documentation_data)
            }
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False, default=str)
            
            result.success = True
            result.file_size = os.path.getsize(output_path)
            result.generation_time = (datetime.now() - start_time).total_seconds()
            
        except Exception as e:
            result.errors.append(f"JSON generation failed: {str(e)}")
            self.logger.error(f"JSON generation error: {e}")
        
        return result
    
    def _generate_statistics(self, documentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistics about the documentation"""
        stats = {
            "total_modules": len(documentation_data),
            "total_classes": 0,
            "total_functions": 0,
            "modules_with_examples": 0,
            "modules_with_config": 0,
            "average_classes_per_module": 0,
            "average_functions_per_module": 0
        }
        
        for module_data in documentation_data.values():
            classes = module_data.get('classes', {})
            functions = module_data.get('functions', {})
            
            stats["total_classes"] += len(classes)
            stats["total_functions"] += len(functions)
            
            if module_data.get('examples'):
                stats["modules_with_examples"] += 1
            
            if module_data.get('configuration'):
                stats["modules_with_config"] += 1
        
        if stats["total_modules"] > 0:
            stats["average_classes_per_module"] = stats["total_classes"] / stats["total_modules"]
            stats["average_functions_per_module"] = stats["total_functions"] / stats["total_modules"]
        
        return stats

class DocumentationGenerator:
    """Main documentation generator orchestrator"""
    
    def __init__(self):
        self.generators = {
            OutputFormat.HTML: HTMLGenerator,
            OutputFormat.PDF: PDFGenerator,
            OutputFormat.MARKDOWN: MarkdownGenerator,
            OutputFormat.JSON: JSONGenerator
        }
        self.logger = logging.getLogger(__name__)
    
    async def generate_multi_format(
        self,
        documentation_data: Dict[str, Any],
        output_dir: str,
        formats: List[OutputFormat],
        configs: Optional[Dict[OutputFormat, GenerationConfig]] = None
    ) -> Dict[OutputFormat, GenerationResult]:
        """Generate documentation in multiple formats simultaneously"""
        
        if configs is None:
            configs = {fmt: GenerationConfig(output_format=fmt) for fmt in formats}
        
        results = {}
        
        # Generate all formats in parallel
        tasks = []
        for output_format in formats:
            if output_format not in self.generators:
                self.logger.warning(f"Unsupported format: {output_format}")
                continue
            
            config = configs.get(output_format, GenerationConfig(output_format=output_format))
            generator = self.generators[output_format](config)
            
            output_filename = config.output_filename or f"documentation.{output_format.value}"
            output_path = os.path.join(output_dir, output_filename)
            
            task = asyncio.create_task(
                generator.generate(documentation_data, output_path)
            )
            tasks.append((output_format, task))
        
        # Wait for all generations to complete
        for output_format, task in tasks:
            try:
                result = await task
                results[output_format] = result
            except Exception as e:
                results[output_format] = GenerationResult(
                    success=False,
                    output_path="",
                    format_type=output_format,
                    errors=[f"Generation failed: {str(e)}"]
                )
        
        return results
    
    async def generate_documentation(
        self,
        target: Any,  # DocumentationTarget from production_documentation_system
        doc_type: Any,  # DocumentationType
        level: Any  # DocumentationLevel
    ) -> Optional[str]:
        """Generate documentation for a specific target"""
        
        try:
            # Import the comprehensive documentation system
            from .comprehensive_documentation import DocumentationContent
            
            content_manager = DocumentationContent()
            structure = content_manager.analyze_and_structure_file(target.file_path)
            
            # Generate content based on documentation type
            if hasattr(doc_type, 'value'):
                doc_type_str = doc_type.value
            else:
                doc_type_str = str(doc_type)
            
            content_dict = content_manager.generate_comprehensive_content(structure)
            
            return content_dict.get(doc_type_str, content_dict.get('module', 'No content generated'))
            
        except Exception as e:
            self.logger.error(f"Failed to generate documentation for {target.module_name}: {e}")
            return None
    
    def get_supported_formats(self) -> List[OutputFormat]:
        """Get list of supported output formats"""
        return list(self.generators.keys())
    
    def validate_generation_config(self, config: GenerationConfig) -> List[str]:
        """Validate generation configuration"""
        warnings = []
        
        if config.output_format not in self.generators:
            warnings.append(f"Unsupported output format: {config.output_format}")
        
        if config.custom_css and config.output_format != OutputFormat.HTML:
            warnings.append("Custom CSS only applies to HTML output")
        
        if config.include_diagrams and not shutil.which('dot'):
            warnings.append("Graphviz not found - diagrams will be disabled")
        
        return warnings

if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Test the documentation generator
        generator = DocumentationGenerator()
        
        # Sample documentation data
        sample_data = {
            "independent_core.brain": {
                "file_path": "/path/to/brain.py",
                "overview": "Main brain module for Saraphis",
                "classes": {
                    "Brain": {
                        "docstring": "Main brain class",
                        "methods": [
                            {"name": "think", "args": ["self", "input"], "docstring": "Process input"}
                        ]
                    }
                },
                "functions": {
                    "initialize": {
                        "docstring": "Initialize the brain system",
                        "args": ["config"],
                        "returns": "Brain"
                    }
                },
                "configuration": {
                    "BRAIN_SIZE": {"value": 1024, "type": "int"}
                },
                "examples": ["brain = Brain()", "brain.think('hello')"]
            }
        }
        
        # Generate documentation in multiple formats
        results = await generator.generate_multi_format(
            sample_data,
            "docs/output",
            [OutputFormat.HTML, OutputFormat.MARKDOWN, OutputFormat.JSON]
        )
        
        # Print results
        for format_type, result in results.items():
            print(f"{format_type.value}: {'Success' if result.success else 'Failed'}")
            if result.errors:
                print(f"  Errors: {result.errors}")
    
    asyncio.run(main())