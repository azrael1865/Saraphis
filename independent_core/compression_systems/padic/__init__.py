"""
P-adic Compression Module - Fixed to avoid circular imports
"""

__version__ = '1.0.0'

# Use explicit imports with error handling to avoid circular dependencies
_module_cache = {}

def _safe_import(module_name, attr_name):
    """Safely import to avoid circular dependencies"""
    cache_key = f"{module_name}.{attr_name}"
    if cache_key in _module_cache:
        return _module_cache[cache_key]
    
    try:
        import importlib
        # Use absolute import path
        full_module = f"independent_core.compression_systems.padic.{module_name}"
        module = importlib.import_module(full_module)
        attr = getattr(module, attr_name)
        _module_cache[cache_key] = attr
        return attr
    except (ImportError, AttributeError):
        # Try relative import as fallback
        import sys
        if 'independent_core.compression_systems.padic' in sys.modules:
            parent = sys.modules['independent_core.compression_systems.padic']
            if hasattr(parent, module_name):
                module = getattr(parent, module_name)
                if hasattr(module, attr_name):
                    attr = getattr(module, attr_name)
                    _module_cache[cache_key] = attr
                    return attr
        return None

# Lazy property accessors
@property
def PadicCompressionSystem():
    return _safe_import('padic_compressor', 'PadicCompressionSystem')

@property
def CompressionConfig():
    return _safe_import('padic_compressor', 'CompressionConfig')

@property  
def PadicWeight():
    return _safe_import('padic_encoder', 'PadicWeight')

# For module-level access, use __getattr__
def __getattr__(name):
    lazy_imports = {
        'PadicCompressionSystem': lambda: _safe_import('padic_compressor', 'PadicCompressionSystem'),
        'CompressionConfig': lambda: _safe_import('padic_compressor', 'CompressionConfig'),
        'PadicWeight': lambda: _safe_import('padic_encoder', 'PadicWeight'),
        'PadicValidation': lambda: _safe_import('padic_encoder', 'PadicValidation'),
        'PadicMathematicalOperations': lambda: _safe_import('padic_encoder', 'PadicMathematicalOperations'),
    }
    
    if name in lazy_imports:
        result = lazy_imports[name]()
        if result is not None:
            globals()[name] = result  # Cache for next access
            return result
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'PadicCompressionSystem',
    'CompressionConfig',
    'PadicWeight',
    'PadicValidation',
    'PadicMathematicalOperations',
]
