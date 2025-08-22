"""
Compression Systems Module - Fixed to avoid circular imports
"""

__version__ = '1.0.0'

# Delay padic import to avoid circular dependency
def __getattr__(name):
    """Lazy import attributes on demand"""
    
    # Map of attributes to their source modules
    attr_to_module = {
        'PadicCompressionSystem': 'padic',
        'CompressionConfig': 'padic',
        'PadicWeight': 'padic',
    }
    
    if name in attr_to_module:
        # Import the submodule only when needed
        submodule_name = attr_to_module[name]
        import importlib
        try:
            submodule = importlib.import_module(f'.{submodule_name}', package=__package__)
            attr = getattr(submodule, name)
            # Cache it for future use
            globals()[name] = attr
            return attr
        except (ImportError, AttributeError) as e:
            raise AttributeError(f"Cannot import {name} from {submodule_name}: {e}")
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['PadicCompressionSystem', 'CompressionConfig', 'PadicWeight']
