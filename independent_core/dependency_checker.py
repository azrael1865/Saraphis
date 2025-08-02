#!/usr/bin/env python3
"""
Smart dependency management system with graceful fallbacks.
Provides fallback implementations for missing modules like JWT and TOML.
"""

import sys
import os
import logging
from typing import Dict, Any, Optional, Union
import base64
import json
import hashlib
import time

logger = logging.getLogger(__name__)

class DependencyManager:
    """Manages optional dependencies with fallback implementations."""
    
    def __init__(self):
        self.available = {}
        self.fallbacks = {}
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check which dependencies are available."""
        # Check PyJWT
        try:
            import jwt as pyjwt
            self.available['jwt'] = pyjwt
            logger.debug("PyJWT is available")
        except ImportError:
            self.available['jwt'] = None
            logger.debug("PyJWT not available, will use fallback")
        
        # Check toml
        try:
            import toml
            self.available['toml'] = toml
            logger.debug("toml is available")
        except ImportError:
            self.available['toml'] = None
            logger.debug("toml not available, will use fallback")
        
        # Check yaml
        try:
            import yaml
            self.available['yaml'] = yaml
            logger.debug("PyYAML is available")
        except ImportError:
            self.available['yaml'] = None
            logger.debug("PyYAML not available, will use fallback")
        
        # Check cryptography
        try:
            from cryptography.fernet import Fernet
            self.available['cryptography'] = True
            logger.debug("cryptography is available")
        except ImportError:
            self.available['cryptography'] = None
            logger.debug("cryptography not available, will use fallback")
        
        # Check FastAPI
        try:
            import fastapi
            self.available['fastapi'] = fastapi
            logger.debug("FastAPI is available")
        except ImportError:
            self.available['fastapi'] = None
            logger.debug("FastAPI not available")
    
    def get_dependency_status(self) -> Dict[str, bool]:
        """Get status of all dependencies."""
        return {name: dep is not None for name, dep in self.available.items()}
    
    def print_dependency_report(self):
        """Print a detailed dependency report."""
        print("\n" + "="*50)
        print("DEPENDENCY STATUS REPORT")
        print("="*50)
        
        status = self.get_dependency_status()
        for name, available in status.items():
            status_text = "✓ Available" if available else "✗ Missing (using fallback)"
            print(f"{name:15}: {status_text}")
            
            if not available:
                install_cmd = self._get_install_command(name)
                if install_cmd:
                    print(f"{'':15}  Install: {install_cmd}")
        
        print("\nNote: Fallback implementations have limited functionality")
        print("For full features, install missing dependencies")
        print("="*50)
    
    def _get_install_command(self, dep_name: str) -> Optional[str]:
        """Get pip install command for dependency."""
        commands = {
            'jwt': 'pip install PyJWT',
            'toml': 'pip install toml',
            'yaml': 'pip install PyYAML', 
            'cryptography': 'pip install cryptography',
            'fastapi': 'pip install fastapi'
        }
        return commands.get(dep_name)

# Global dependency manager instance
dependency_manager = DependencyManager()

# JWT Fallback Implementation
class JWTFallback:
    """Basic JWT implementation when PyJWT is not available."""
    
    @staticmethod
    def encode(payload: Dict[str, Any], key: str, algorithm: str = 'HS256') -> str:
        """Encode JWT token with fallback implementation."""
        if algorithm != 'HS256':
            raise NotImplementedError(f"Fallback JWT only supports HS256, got {algorithm}")
        
        # Create header
        header = {
            "typ": "JWT",
            "alg": algorithm
        }
        
        # Encode header and payload
        header_b64 = base64.urlsafe_b64encode(
            json.dumps(header, separators=(',', ':')).encode()
        ).decode().rstrip('=')
        
        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(',', ':')).encode()  
        ).decode().rstrip('=')
        
        # Create signature
        message = f"{header_b64}.{payload_b64}"
        signature = hashlib.sha256(f"{message}{key}".encode()).hexdigest()[:32]
        signature_b64 = base64.urlsafe_b64encode(signature.encode()).decode().rstrip('=')
        
        return f"{message}.{signature_b64}"
    
    @staticmethod
    def decode(token: str, key: str, algorithms: list = None) -> Dict[str, Any]:
        """Decode JWT token with fallback implementation."""
        if algorithms and 'HS256' not in algorithms:
            raise NotImplementedError("Fallback JWT only supports HS256")
        
        try:
            header_b64, payload_b64, signature_b64 = token.split('.')
            
            # Verify signature
            message = f"{header_b64}.{payload_b64}"
            expected_signature = hashlib.sha256(f"{message}{key}".encode()).hexdigest()[:32]
            expected_signature_b64 = base64.urlsafe_b64encode(expected_signature.encode()).decode().rstrip('=')
            
            if signature_b64 != expected_signature_b64:
                raise ValueError("Invalid signature")
            
            # Decode payload
            payload_padded = payload_b64 + '=' * (4 - len(payload_b64) % 4)
            payload_json = base64.urlsafe_b64decode(payload_padded).decode()
            payload = json.loads(payload_json)
            
            # Basic expiration check
            if 'exp' in payload:
                if payload['exp'] < time.time():
                    raise ValueError("Token has expired")
            
            return payload
            
        except Exception as e:
            raise ValueError(f"Invalid token: {e}")

# TOML Fallback Implementation  
class TOMLFallback:
    """Basic TOML implementation when toml package is not available."""
    
    @staticmethod
    def loads(toml_str: str) -> Dict[str, Any]:
        """Parse TOML string with fallback implementation."""
        result = {}
        current_section = result
        section_stack = [result]
        
        for line in toml_str.strip().split('\n'):
            line = line.strip()
            
            if not line or line.startswith('#'):
                continue
            
            # Section headers
            if line.startswith('[') and line.endswith(']'):
                section_name = line[1:-1].strip()
                
                # Handle nested sections
                if '.' in section_name:
                    parts = section_name.split('.')
                    current_section = result
                    for part in parts:
                        if part not in current_section:
                            current_section[part] = {}
                        current_section = current_section[part]
                else:
                    if section_name not in result:
                        result[section_name] = {}
                    current_section = result[section_name]
                continue
            
            # Key-value pairs
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Parse value
                parsed_value = TOMLFallback._parse_value(value)
                current_section[key] = parsed_value
        
        return result
    
    @staticmethod
    def _parse_value(value: str) -> Any:
        """Parse TOML value with basic type detection."""
        value = value.strip()
        
        # String values
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        
        # Boolean values
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Arrays (basic support)
        if value.startswith('[') and value.endswith(']'):
            array_content = value[1:-1].strip()
            if not array_content:
                return []
            
            items = []
            for item in array_content.split(','):
                items.append(TOMLFallback._parse_value(item.strip()))
            return items
        
        # Default to string
        return value
    
    @staticmethod
    def dumps(data: Dict[str, Any]) -> str:
        """Convert dict to TOML string with fallback implementation."""
        lines = []
        
        # Handle top-level key-value pairs
        for key, value in data.items():
            if not isinstance(value, dict):
                lines.append(f"{key} = {TOMLFallback._format_value(value)}")
        
        # Handle sections
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"\n[{key}]")
                for subkey, subvalue in value.items():
                    lines.append(f"{subkey} = {TOMLFallback._format_value(subvalue)}")
        
        return '\n'.join(lines)
    
    @staticmethod
    def _format_value(value: Any) -> str:
        """Format value for TOML output."""
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            formatted_items = [TOMLFallback._format_value(item) for item in value]
            return f"[{', '.join(formatted_items)}]"
        else:
            return f'"{str(value)}"'

# Create module-like objects that use native or fallback implementations
class ModuleProxy:
    """Proxy that uses native implementation if available, otherwise fallback."""
    
    def __init__(self, native_module, fallback_class):
        self.native = native_module
        self.fallback = fallback_class
        
    def __getattr__(self, name):
        if self.native:
            return getattr(self.native, name)
        else:
            return getattr(self.fallback, name)

# Create proxies for JWT and TOML
jwt = ModuleProxy(dependency_manager.available['jwt'], JWTFallback)
toml = ModuleProxy(dependency_manager.available['toml'], TOMLFallback)

# Export dependency report function
def print_dependency_report():
    """Print dependency status report."""
    dependency_manager.print_dependency_report()

def get_dependency_status() -> Dict[str, bool]:
    """Get dependency availability status."""
    return dependency_manager.get_dependency_status()

# YAML Fallback (basic)
class YAMLFallback:
    """Very basic YAML fallback for simple structures."""
    
    @staticmethod
    def safe_load(yaml_str: str) -> Dict[str, Any]:
        """Basic YAML parsing for simple key-value structures."""
        result = {}
        
        for line in yaml_str.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Basic type conversion
                if value.lower() in ('true', 'yes'):
                    value = True
                elif value.lower() in ('false', 'no'):
                    value = False
                elif value.isdigit():
                    value = int(value)
                else:
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                
                result[key] = value
        
        return result
    
    @staticmethod
    def safe_dump(data: Dict[str, Any]) -> str:
        """Basic YAML output."""
        lines = []
        for key, value in data.items():
            if isinstance(value, str):
                lines.append(f"{key}: \"{value}\"")
            else:
                lines.append(f"{key}: {value}")
        return '\n'.join(lines)

# YAML proxy
yaml = ModuleProxy(dependency_manager.available['yaml'], YAMLFallback)

# Encryption fallback (INSECURE - for development only)
class EncryptionFallback:
    """INSECURE fallback encryption for development/testing only."""
    
    @staticmethod
    def generate_key() -> bytes:
        """Generate a simple key."""
        return b"development_key_not_secure_12345678"
    
    @staticmethod
    def encrypt(data: bytes, key: bytes) -> bytes:
        """XOR encryption (INSECURE - development only)."""
        return bytes(a ^ b for a, b in zip(data, (key * (len(data) // len(key) + 1))[:len(data)]))
    
    @staticmethod
    def decrypt(encrypted_data: bytes, key: bytes) -> bytes:
        """XOR decryption (INSECURE - development only)."""
        return EncryptionFallback.encrypt(encrypted_data, key)  # XOR is symmetric

# Export main functions and classes
__all__ = [
    'dependency_manager',
    'jwt', 
    'toml',
    'yaml',
    'print_dependency_report',
    'get_dependency_status',
    'JWTFallback',
    'TOMLFallback', 
    'YAMLFallback',
    'EncryptionFallback'
]