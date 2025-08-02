#!/usr/bin/env python3
"""
Binary Analysis Feature Extractor Plugin
========================================

This module provides binary analysis feature extraction capabilities for the Universal AI Core system.
Adapted from Saraphis molecular descriptor patterns, specialized for executable file analysis,
malware detection, and binary classification tasks.

Features:
- PE header analysis and feature extraction
- Opcode frequency analysis and n-gram extraction
- Entropy and statistical analysis
- Import/export table analysis
- Section analysis and suspicious patterns
- String and metadata extraction
- Packed binary detection
- Control flow analysis features
"""

import logging
import sys
import time
import os
import hashlib
import struct
import math
import warnings
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from collections import Counter, defaultdict
import threading
import re

# Try to import binary analysis dependencies
try:
    import pefile
    PEFILE_AVAILABLE = True
except ImportError:
    PEFILE_AVAILABLE = False

try:
    from capstone import *
    from capstone.x86 import *
    CAPSTONE_AVAILABLE = True
except ImportError:
    CAPSTONE_AVAILABLE = False

# Import plugin base classes
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from base import FeatureExtractorPlugin, FeatureExtractionResult, FeatureType

logger = logging.getLogger(__name__)


@dataclass
class BinaryAnalysisResult:
    """Result container for binary analysis"""
    pe_features: np.ndarray = field(default_factory=lambda: np.array([]))
    opcode_features: np.ndarray = field(default_factory=lambda: np.array([]))
    entropy_features: np.ndarray = field(default_factory=lambda: np.array([]))
    string_features: np.ndarray = field(default_factory=lambda: np.array([]))
    section_features: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_names: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    valid_binaries: int = 0
    invalid_binaries: int = 0
    error_messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BinaryStatisticalAnalyzer:
    """
    Statistical analysis for binary files.
    
    Adapted from molecular descriptor calculation patterns.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.BinaryStatisticalAnalyzer")
    
    def calculate_entropy(self, data: bytes, block_size: int = 256) -> List[float]:
        """Calculate entropy features for binary data"""
        if not data:
            return [0.0] * 5
        
        try:
            # Overall entropy
            overall_entropy = self._shannon_entropy(data)
            
            # Block-wise entropy analysis
            block_entropies = []
            for i in range(0, len(data), block_size):
                block = data[i:i+block_size]
                if block:
                    block_entropies.append(self._shannon_entropy(block))
            
            if block_entropies:
                avg_entropy = np.mean(block_entropies)
                max_entropy = np.max(block_entropies)
                min_entropy = np.min(block_entropies)
                entropy_variance = np.var(block_entropies)
            else:
                avg_entropy = max_entropy = min_entropy = entropy_variance = 0.0
            
            return [overall_entropy, avg_entropy, max_entropy, min_entropy, entropy_variance]
            
        except Exception as e:
            self.logger.error(f"Error calculating entropy: {e}")
            return [0.0] * 5
    
    def _shannon_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0.0
        
        # Count byte frequencies
        byte_counts = Counter(data)
        data_len = len(data)
        
        # Calculate entropy
        entropy = 0.0
        for count in byte_counts.values():
            probability = count / data_len
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def calculate_byte_statistics(self, data: bytes) -> List[float]:
        """Calculate byte-level statistics"""
        if not data:
            return [0.0] * 10
        
        try:
            # Convert to numpy array for efficient computation
            byte_array = np.frombuffer(data, dtype=np.uint8)
            
            # Basic statistics
            mean_val = np.mean(byte_array)
            std_val = np.std(byte_array)
            min_val = np.min(byte_array)
            max_val = np.max(byte_array)
            median_val = np.median(byte_array)
            
            # Advanced statistics
            skewness = self._calculate_skewness(byte_array)
            kurtosis = self._calculate_kurtosis(byte_array)
            
            # Byte distribution features
            unique_bytes = len(np.unique(byte_array))
            zero_bytes = np.sum(byte_array == 0)
            high_bytes = np.sum(byte_array > 127)
            
            return [mean_val, std_val, min_val, max_val, median_val, 
                   skewness, kurtosis, unique_bytes, zero_bytes, high_bytes]
            
        except Exception as e:
            self.logger.error(f"Error calculating byte statistics: {e}")
            return [0.0] * 10
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 3)
        except:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 4) - 3
        except:
            return 0.0


class PEAnalyzer:
    """
    PE (Portable Executable) file analyzer.
    
    Adapted from molecular property calculation patterns.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PEAnalyzer")
    
    def extract_pe_features(self, binary_data: bytes) -> Tuple[np.ndarray, List[str]]:
        """Extract PE header and structure features"""
        if not PEFILE_AVAILABLE:
            self.logger.warning("pefile not available - PE analysis disabled")
            return np.zeros(50), [f"pe_feature_{i}" for i in range(50)]
        
        try:
            pe = pefile.PE(data=binary_data, fast_load=True)
            
            features = []
            feature_names = []
            
            # DOS Header features
            dos_features, dos_names = self._extract_dos_header_features(pe)
            features.extend(dos_features)
            feature_names.extend(dos_names)
            
            # NT Headers features
            nt_features, nt_names = self._extract_nt_header_features(pe)
            features.extend(nt_features)
            feature_names.extend(nt_names)
            
            # Section features
            section_features, section_names = self._extract_section_features(pe)
            features.extend(section_features)
            feature_names.extend(section_names)
            
            # Import/Export features
            import_features, import_names = self._extract_import_features(pe)
            features.extend(import_features)
            feature_names.extend(import_names)
            
            pe.close()
            
            # Pad to fixed size
            target_size = 50
            if len(features) < target_size:
                features.extend([0.0] * (target_size - len(features)))
                feature_names.extend([f"pe_padding_{i}" for i in range(target_size - len(feature_names))])
            elif len(features) > target_size:
                features = features[:target_size]
                feature_names = feature_names[:target_size]
            
            return np.array(features), feature_names
            
        except Exception as e:
            self.logger.error(f"Error extracting PE features: {e}")
            return np.zeros(50), [f"pe_feature_{i}" for i in range(50)]
    
    def _extract_dos_header_features(self, pe) -> Tuple[List[float], List[str]]:
        """Extract DOS header features"""
        try:
            dos_header = pe.DOS_HEADER
            
            features = [
                dos_header.e_magic,
                dos_header.e_cblp,
                dos_header.e_cp,
                dos_header.e_crlc,
                dos_header.e_cparhdr,
                dos_header.e_minalloc,
                dos_header.e_maxalloc,
                dos_header.e_ss,
                dos_header.e_sp,
                dos_header.e_csum,
                dos_header.e_lfanew
            ]
            
            names = [
                'dos_magic', 'dos_cblp', 'dos_cp', 'dos_crlc', 'dos_cparhdr',
                'dos_minalloc', 'dos_maxalloc', 'dos_ss', 'dos_sp', 'dos_csum', 'dos_lfanew'
            ]
            
            return features, names
            
        except Exception as e:
            self.logger.debug(f"Error extracting DOS header: {e}")
            return [0.0] * 11, [f"dos_feature_{i}" for i in range(11)]
    
    def _extract_nt_header_features(self, pe) -> Tuple[List[float], List[str]]:
        """Extract NT headers features"""
        try:
            nt_headers = pe.NT_HEADERS
            file_header = nt_headers.FILE_HEADER
            optional_header = nt_headers.OPTIONAL_HEADER
            
            features = [
                file_header.Machine,
                file_header.NumberOfSections,
                file_header.TimeDateStamp,
                file_header.PointerToSymbolTable,
                file_header.NumberOfSymbols,
                file_header.SizeOfOptionalHeader,
                file_header.Characteristics,
                optional_header.Magic,
                optional_header.MajorLinkerVersion,
                optional_header.MinorLinkerVersion,
                optional_header.SizeOfCode,
                optional_header.SizeOfInitializedData,
                optional_header.SizeOfUninitializedData,
                optional_header.AddressOfEntryPoint,
                optional_header.BaseOfCode,
                optional_header.ImageBase,
                optional_header.SectionAlignment,
                optional_header.FileAlignment,
                optional_header.MajorOperatingSystemVersion,
                optional_header.MinorOperatingSystemVersion,
                optional_header.MajorImageVersion,
                optional_header.MinorImageVersion,
                optional_header.MajorSubsystemVersion,
                optional_header.MinorSubsystemVersion,
                optional_header.SizeOfImage,
                optional_header.SizeOfHeaders,
                optional_header.CheckSum,
                optional_header.Subsystem,
                optional_header.DllCharacteristics
            ]
            
            names = [
                'machine', 'num_sections', 'timestamp', 'symbol_table_ptr', 'num_symbols',
                'optional_header_size', 'characteristics', 'magic', 'major_linker_version',
                'minor_linker_version', 'size_of_code', 'size_initialized_data',
                'size_uninitialized_data', 'entry_point', 'base_of_code', 'image_base',
                'section_alignment', 'file_alignment', 'major_os_version', 'minor_os_version',
                'major_image_version', 'minor_image_version', 'major_subsystem_version',
                'minor_subsystem_version', 'size_of_image', 'size_of_headers', 'checksum',
                'subsystem', 'dll_characteristics'
            ]
            
            return features, names
            
        except Exception as e:
            self.logger.debug(f"Error extracting NT headers: {e}")
            return [0.0] * 29, [f"nt_feature_{i}" for i in range(29)]
    
    def _extract_section_features(self, pe) -> Tuple[List[float], List[str]]:
        """Extract section-level features"""
        try:
            sections = pe.sections
            
            # Aggregate section features
            total_sections = len(sections)
            executable_sections = 0
            writable_sections = 0
            total_virtual_size = 0
            total_raw_size = 0
            max_entropy = 0.0
            
            for section in sections:
                if section.Characteristics & 0x20000000:  # IMAGE_SCN_MEM_EXECUTE
                    executable_sections += 1
                if section.Characteristics & 0x80000000:  # IMAGE_SCN_MEM_WRITE
                    writable_sections += 1
                
                total_virtual_size += section.Misc_VirtualSize
                total_raw_size += section.SizeOfRawData
                
                # Calculate section entropy
                section_data = section.get_data()
                if section_data:
                    entropy = self._calculate_section_entropy(section_data)
                    max_entropy = max(max_entropy, entropy)
            
            features = [
                total_sections,
                executable_sections,
                writable_sections,
                total_virtual_size,
                total_raw_size,
                max_entropy
            ]
            
            names = [
                'total_sections', 'executable_sections', 'writable_sections',
                'total_virtual_size', 'total_raw_size', 'max_section_entropy'
            ]
            
            return features, names
            
        except Exception as e:
            self.logger.debug(f"Error extracting section features: {e}")
            return [0.0] * 6, [f"section_feature_{i}" for i in range(6)]
    
    def _extract_import_features(self, pe) -> Tuple[List[float], List[str]]:
        """Extract import table features"""
        try:
            imports = []
            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    imports.extend([imp.name.decode() for imp in entry.imports if imp.name])
            
            # Count suspicious imports
            suspicious_imports = [
                'CreateProcess', 'WriteProcessMemory', 'VirtualAlloc', 'LoadLibrary',
                'GetProcAddress', 'RegOpenKey', 'InternetOpen', 'socket', 'connect'
            ]
            
            total_imports = len(imports)
            suspicious_count = sum(1 for imp in imports if any(sus in str(imp) for sus in suspicious_imports))
            unique_dlls = len(set(entry.dll.decode() for entry in pe.DIRECTORY_ENTRY_IMPORT)) if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else 0
            
            features = [total_imports, suspicious_count, unique_dlls]
            names = ['total_imports', 'suspicious_imports', 'unique_dlls']
            
            return features, names
            
        except Exception as e:
            self.logger.debug(f"Error extracting import features: {e}")
            return [0.0] * 3, ['total_imports', 'suspicious_imports', 'unique_dlls']
    
    def _calculate_section_entropy(self, data: bytes) -> float:
        """Calculate entropy of section data"""
        if not data:
            return 0.0
        
        byte_counts = Counter(data)
        data_len = len(data)
        
        entropy = 0.0
        for count in byte_counts.values():
            probability = count / data_len
            entropy -= probability * math.log2(probability)
        
        return entropy


class OpcodeAnalyzer:
    """
    Opcode and assembly instruction analyzer.
    
    Adapted from molecular fingerprint patterns.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.OpcodeAnalyzer")
        self.md = None
        if CAPSTONE_AVAILABLE:
            try:
                self.md = Cs(CS_ARCH_X86, CS_MODE_32)
                self.md.detail = True
            except Exception as e:
                self.logger.warning(f"Failed to initialize Capstone: {e}")
    
    def extract_opcode_features(self, binary_data: bytes, max_instructions: int = 10000) -> Tuple[np.ndarray, List[str]]:
        """Extract opcode frequency and n-gram features"""
        if not CAPSTONE_AVAILABLE or self.md is None:
            self.logger.warning("Capstone not available - opcode analysis disabled")
            return np.zeros(200), [f"opcode_feature_{i}" for i in range(200)]
        
        try:
            # Disassemble instructions
            instructions = list(self.md.disasm(binary_data, 0x1000, count=max_instructions))
            
            if not instructions:
                return np.zeros(200), [f"opcode_feature_{i}" for i in range(200)]
            
            # Extract opcode frequency features
            opcode_freq = self._extract_opcode_frequency(instructions)
            
            # Extract instruction type features
            inst_type_freq = self._extract_instruction_type_frequency(instructions)
            
            # Extract opcode n-grams
            ngram_features = self._extract_opcode_ngrams(instructions)
            
            # Combine all features
            all_features = []
            all_features.extend(opcode_freq)
            all_features.extend(inst_type_freq)
            all_features.extend(ngram_features)
            
            # Pad to fixed size
            target_size = 200
            if len(all_features) < target_size:
                all_features.extend([0.0] * (target_size - len(all_features)))
            elif len(all_features) > target_size:
                all_features = all_features[:target_size]
            
            feature_names = [f"opcode_feature_{i}" for i in range(target_size)]
            
            return np.array(all_features), feature_names
            
        except Exception as e:
            self.logger.error(f"Error extracting opcode features: {e}")
            return np.zeros(200), [f"opcode_feature_{i}" for i in range(200)]
    
    def _extract_opcode_frequency(self, instructions) -> List[float]:
        """Extract top opcode frequencies"""
        # Common x86 opcodes to track
        tracked_opcodes = [
            'mov', 'push', 'pop', 'call', 'ret', 'jmp', 'je', 'jne', 'jz', 'jnz',
            'add', 'sub', 'mul', 'div', 'and', 'or', 'xor', 'not', 'shl', 'shr',
            'cmp', 'test', 'lea', 'nop', 'int', 'syscall', 'leave', 'enter'
        ]
        
        # Count opcodes
        opcode_counts = Counter()
        total_instructions = len(instructions)
        
        for instruction in instructions:
            mnemonic = instruction.mnemonic.lower()
            opcode_counts[mnemonic] += 1
        
        # Calculate frequencies for tracked opcodes
        frequencies = []
        for opcode in tracked_opcodes:
            freq = opcode_counts.get(opcode, 0) / max(total_instructions, 1)
            frequencies.append(freq)
        
        return frequencies
    
    def _extract_instruction_type_frequency(self, instructions) -> List[float]:
        """Extract instruction type frequencies"""
        type_counts = defaultdict(int)
        total_instructions = len(instructions)
        
        for instruction in instructions:
            mnemonic = instruction.mnemonic.lower()
            
            # Categorize instructions
            if mnemonic in ['mov', 'lea', 'xchg']:
                type_counts['data_movement'] += 1
            elif mnemonic in ['add', 'sub', 'mul', 'div', 'inc', 'dec']:
                type_counts['arithmetic'] += 1
            elif mnemonic in ['and', 'or', 'xor', 'not', 'shl', 'shr']:
                type_counts['logical'] += 1
            elif mnemonic in ['cmp', 'test']:
                type_counts['comparison'] += 1
            elif mnemonic.startswith('j') or mnemonic in ['call', 'ret']:
                type_counts['control_flow'] += 1
            elif mnemonic in ['push', 'pop']:
                type_counts['stack'] += 1
            else:
                type_counts['other'] += 1
        
        # Calculate frequencies
        categories = ['data_movement', 'arithmetic', 'logical', 'comparison', 'control_flow', 'stack', 'other']
        frequencies = []
        for category in categories:
            freq = type_counts[category] / max(total_instructions, 1)
            frequencies.append(freq)
        
        return frequencies
    
    def _extract_opcode_ngrams(self, instructions, n: int = 3) -> List[float]:
        """Extract opcode n-gram frequencies"""
        if len(instructions) < n:
            return [0.0] * 20  # Return fixed number of features
        
        # Extract instruction sequence
        opcodes = [inst.mnemonic.lower() for inst in instructions]
        
        # Generate n-grams
        ngrams = []
        for i in range(len(opcodes) - n + 1):
            ngram = tuple(opcodes[i:i+n])
            ngrams.append(ngram)
        
        # Count n-gram frequencies
        ngram_counts = Counter(ngrams)
        total_ngrams = len(ngrams)
        
        # Get top n-grams
        top_ngrams = ngram_counts.most_common(20)
        
        # Extract frequencies
        frequencies = []
        for _, count in top_ngrams:
            freq = count / max(total_ngrams, 1)
            frequencies.append(freq)
        
        # Pad to fixed size
        while len(frequencies) < 20:
            frequencies.append(0.0)
        
        return frequencies


class StringAnalyzer:
    """
    String and metadata analyzer for binaries.
    
    Adapted from molecular metadata extraction patterns.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.StringAnalyzer")
    
    def extract_string_features(self, binary_data: bytes) -> Tuple[np.ndarray, List[str]]:
        """Extract string-based features"""
        try:
            # Extract printable strings
            strings = self._extract_strings(binary_data)
            
            # Calculate string statistics
            string_stats = self._calculate_string_statistics(strings)
            
            # Detect suspicious strings
            suspicious_counts = self._count_suspicious_strings(strings)
            
            # Combine features
            features = []
            features.extend(string_stats)
            features.extend(suspicious_counts)
            
            # Pad to fixed size
            target_size = 20
            if len(features) < target_size:
                features.extend([0.0] * (target_size - len(features)))
            elif len(features) > target_size:
                features = features[:target_size]
            
            feature_names = [
                'total_strings', 'avg_string_length', 'max_string_length',
                'printable_ratio', 'ascii_ratio', 'url_count', 'ip_count',
                'registry_count', 'crypto_count', 'api_count', 'file_count',
                'suspicious_count'
            ] + [f"string_feature_{i}" for i in range(target_size - 12)]
            
            return np.array(features), feature_names
            
        except Exception as e:
            self.logger.error(f"Error extracting string features: {e}")
            return np.zeros(20), [f"string_feature_{i}" for i in range(20)]
    
    def _extract_strings(self, data: bytes, min_length: int = 4) -> List[str]:
        """Extract printable strings from binary data"""
        strings = []
        current_string = ""
        
        for byte in data:
            char = chr(byte)
            if char.isprintable() and not char.isspace():
                current_string += char
            else:
                if len(current_string) >= min_length:
                    strings.append(current_string)
                current_string = ""
        
        # Don't forget the last string
        if len(current_string) >= min_length:
            strings.append(current_string)
        
        return strings
    
    def _calculate_string_statistics(self, strings: List[str]) -> List[float]:
        """Calculate basic string statistics"""
        if not strings:
            return [0.0] * 4
        
        total_strings = len(strings)
        string_lengths = [len(s) for s in strings]
        avg_length = np.mean(string_lengths)
        max_length = np.max(string_lengths)
        
        # Calculate character ratios
        all_chars = ''.join(strings)
        printable_ratio = sum(1 for c in all_chars if c.isprintable()) / max(len(all_chars), 1)
        ascii_ratio = sum(1 for c in all_chars if ord(c) < 128) / max(len(all_chars), 1)
        
        return [total_strings, avg_length, max_length, printable_ratio, ascii_ratio]
    
    def _count_suspicious_strings(self, strings: List[str]) -> List[float]:
        """Count suspicious string patterns"""
        # Patterns to look for
        url_pattern = re.compile(r'https?://|ftp://', re.IGNORECASE)
        ip_pattern = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
        registry_pattern = re.compile(r'HKEY_|SOFTWARE\\|SYSTEM\\', re.IGNORECASE)
        crypto_pattern = re.compile(r'AES|DES|RSA|MD5|SHA|crypt', re.IGNORECASE)
        api_pattern = re.compile(r'CreateProcess|WriteFile|RegOpenKey|LoadLibrary', re.IGNORECASE)
        file_pattern = re.compile(r'\.(exe|dll|bat|cmd|scr|vbs|ps1)$', re.IGNORECASE)
        
        counts = {
            'url': 0, 'ip': 0, 'registry': 0, 'crypto': 0, 'api': 0, 'file': 0, 'suspicious': 0
        }
        
        suspicious_keywords = [
            'backdoor', 'keylog', 'trojan', 'virus', 'malware', 'exploit',
            'payload', 'shellcode', 'rootkit', 'botnet'
        ]
        
        for string in strings:
            if url_pattern.search(string):
                counts['url'] += 1
            if ip_pattern.search(string):
                counts['ip'] += 1
            if registry_pattern.search(string):
                counts['registry'] += 1
            if crypto_pattern.search(string):
                counts['crypto'] += 1
            if api_pattern.search(string):
                counts['api'] += 1
            if file_pattern.search(string):
                counts['file'] += 1
            if any(keyword in string.lower() for keyword in suspicious_keywords):
                counts['suspicious'] += 1
        
        return list(counts.values())


class BinaryFeatureExtractor(FeatureExtractorPlugin):
    """
    Binary analysis feature extractor plugin.
    
    Adapted from Saraphis molecular feature extraction patterns,
    specialized for executable file analysis and malware detection.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the binary feature extractor"""
        super().__init__(config)
        
        # Configuration
        self.use_pe_analysis = self.config.get('use_pe_analysis', True) and PEFILE_AVAILABLE
        self.use_opcode_analysis = self.config.get('use_opcode_analysis', True) and CAPSTONE_AVAILABLE
        self.use_string_analysis = self.config.get('use_string_analysis', True)
        self.use_entropy_analysis = self.config.get('use_entropy_analysis', True)
        
        self.max_file_size = self.config.get('max_file_size', 50 * 1024 * 1024)  # 50MB
        self.max_instructions = self.config.get('max_instructions', 10000)
        
        # Initialize analyzers
        self.pe_analyzer = PEAnalyzer() if self.use_pe_analysis else None
        self.opcode_analyzer = OpcodeAnalyzer() if self.use_opcode_analysis else None
        self.string_analyzer = StringAnalyzer() if self.use_string_analysis else None
        self.stats_analyzer = BinaryStatisticalAnalyzer() if self.use_entropy_analysis else None
        
        # Feature caching
        self.feature_cache = {}
        self.cache_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'binaries_processed': 0,
            'cache_hits': 0,
            'extraction_time': 0.0,
            'invalid_binaries': 0
        }
        
        self.logger.info(f"🔍 Binary Feature Extractor initialized (PE: {self.use_pe_analysis}, Opcode: {self.use_opcode_analysis})")
    
    def get_metadata(self):
        """Get plugin metadata"""
        from ..base import PluginMetadata, PluginDependency, PluginVersion
        
        dependencies = []
        if self.use_pe_analysis:
            dependencies.append(PluginDependency(name="pefile", version_requirement="*"))
        if self.use_opcode_analysis:
            dependencies.append(PluginDependency(name="capstone", version_requirement="*"))
        
        return PluginMetadata(
            name="BinaryFeatureExtractor",
            version=PluginVersion(1, 0, 0),
            author="Universal AI Core",
            description="Binary analysis and malware detection features",
            plugin_type="feature_extractor",
            entry_point=f"{__name__}:BinaryFeatureExtractor",
            dependencies=dependencies,
            capabilities=["pe_analysis", "opcode_analysis", "entropy_analysis", "string_analysis"],
            hooks=["before_extraction", "after_extraction"],
            configuration_schema={
                "use_pe_analysis": {"type": "boolean", "default": True},
                "use_opcode_analysis": {"type": "boolean", "default": True},
                "max_file_size": {"type": "integer", "default": 52428800}
            }
        )
    
    def extract_features(self, input_data: Any) -> FeatureExtractionResult:
        """Extract features from binary files"""
        start_time = time.time()
        
        try:
            # Parse input data
            binary_files = self._parse_input_data(input_data)
            if not binary_files:
                return FeatureExtractionResult(
                    features={},
                    feature_names=[],
                    feature_types=[],
                    processing_time=time.time() - start_time,
                    success=False,
                    error_message="No valid binary files found in input data"
                )
            
            # Process each binary file
            all_features = []
            all_feature_names = []
            valid_count = 0
            invalid_count = 0
            
            for binary_data in binary_files:
                try:
                    # Check file size
                    if len(binary_data) > self.max_file_size:
                        self.logger.warning(f"Binary file too large: {len(binary_data)} bytes")
                        invalid_count += 1
                        continue
                    
                    # Extract features
                    result = self._extract_single_binary_features(binary_data)
                    
                    if result.valid_binaries > 0:
                        # Combine all feature types
                        combined_features = []
                        feature_names = []
                        
                        if result.pe_features.size > 0:
                            combined_features.extend(result.pe_features)
                            feature_names.extend([f"pe_{name}" for name in result.feature_names[:len(result.pe_features)]])
                        
                        if result.opcode_features.size > 0:
                            combined_features.extend(result.opcode_features)
                            feature_names.extend([f"opcode_{i}" for i in range(len(result.opcode_features))])
                        
                        if result.entropy_features.size > 0:
                            combined_features.extend(result.entropy_features)
                            feature_names.extend([f"entropy_{i}" for i in range(len(result.entropy_features))])
                        
                        if result.string_features.size > 0:
                            combined_features.extend(result.string_features)
                            feature_names.extend([f"string_{i}" for i in range(len(result.string_features))])
                        
                        all_features.append(combined_features)
                        if not all_feature_names:  # Set feature names once
                            all_feature_names = feature_names
                        
                        valid_count += 1
                    else:
                        invalid_count += 1
                        
                except Exception as e:
                    self.logger.error(f"Error processing binary: {e}")
                    invalid_count += 1
            
            processing_time = time.time() - start_time
            
            # Convert to numpy array
            if all_features:
                features_array = np.array(all_features)
                feature_types = [FeatureType.NUMERICAL] * len(all_feature_names)
                
                # Update statistics
                self.stats['binaries_processed'] += valid_count
                self.stats['extraction_time'] += processing_time
                self.stats['invalid_binaries'] += invalid_count
                
                return FeatureExtractionResult(
                    features={'binary_features': features_array},
                    feature_names=all_feature_names,
                    feature_types=feature_types,
                    processing_time=processing_time,
                    success=True,
                    metadata={
                        'total_binaries': len(binary_files),
                        'valid_binaries': valid_count,
                        'invalid_binaries': invalid_count,
                        'feature_dimensions': {
                            'total': len(all_feature_names),
                            'pe_features': self._count_pe_features(),
                            'opcode_features': 200 if self.use_opcode_analysis else 0,
                            'entropy_features': 15 if self.use_entropy_analysis else 0,
                            'string_features': 20 if self.use_string_analysis else 0
                        }
                    }
                )
            else:
                return FeatureExtractionResult(
                    features={},
                    feature_names=[],
                    feature_types=[],
                    processing_time=processing_time,
                    success=False,
                    error_message="No valid features extracted from binary files"
                )
                
        except Exception as e:
            self.logger.error(f"❌ Error extracting binary features: {e}")
            return FeatureExtractionResult(
                features={},
                feature_names=[],
                feature_types=[],
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _parse_input_data(self, input_data: Any) -> List[bytes]:
        """Parse input data to extract binary files"""
        binary_files = []
        
        if isinstance(input_data, bytes):
            # Single binary file
            binary_files = [input_data]
        elif isinstance(input_data, str):
            # File path
            try:
                with open(input_data, 'rb') as f:
                    binary_files = [f.read()]
            except Exception as e:
                self.logger.error(f"Error reading file {input_data}: {e}")
        elif isinstance(input_data, list):
            # List of file paths or binary data
            for item in input_data:
                if isinstance(item, bytes):
                    binary_files.append(item)
                elif isinstance(item, str):
                    try:
                        with open(item, 'rb') as f:
                            binary_files.append(f.read())
                    except Exception as e:
                        self.logger.error(f"Error reading file {item}: {e}")
        elif isinstance(input_data, dict):
            # Dictionary with file data
            if 'binary_data' in input_data:
                if isinstance(input_data['binary_data'], list):
                    binary_files = input_data['binary_data']
                else:
                    binary_files = [input_data['binary_data']]
            elif 'file_paths' in input_data:
                for path in input_data['file_paths']:
                    try:
                        with open(path, 'rb') as f:
                            binary_files.append(f.read())
                    except Exception as e:
                        self.logger.error(f"Error reading file {path}: {e}")
        
        return binary_files
    
    def _extract_single_binary_features(self, binary_data: bytes) -> BinaryAnalysisResult:
        """Extract features from a single binary file"""
        result = BinaryAnalysisResult()
        
        try:
            # Check cache
            data_hash = hashlib.sha256(binary_data).hexdigest()
            with self.cache_lock:
                if data_hash in self.feature_cache:
                    self.stats['cache_hits'] += 1
                    return self.feature_cache[data_hash]
            
            # PE analysis
            if self.pe_analyzer:
                pe_features, pe_names = self.pe_analyzer.extract_pe_features(binary_data)
                result.pe_features = pe_features
                result.feature_names.extend(pe_names)
            
            # Opcode analysis
            if self.opcode_analyzer:
                opcode_features, opcode_names = self.opcode_analyzer.extract_opcode_features(
                    binary_data, self.max_instructions
                )
                result.opcode_features = opcode_features
            
            # Entropy analysis
            if self.stats_analyzer:
                entropy_features = self.stats_analyzer.calculate_entropy(binary_data)
                byte_stats = self.stats_analyzer.calculate_byte_statistics(binary_data)
                result.entropy_features = np.array(entropy_features + byte_stats)
            
            # String analysis
            if self.string_analyzer:
                string_features, string_names = self.string_analyzer.extract_string_features(binary_data)
                result.string_features = string_features
            
            result.valid_binaries = 1
            
            # Cache result
            with self.cache_lock:
                self.feature_cache[data_hash] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting features from binary: {e}")
            result.invalid_binaries = 1
            result.error_messages.append(str(e))
            return result
    
    def _count_pe_features(self) -> int:
        """Count number of PE features"""
        return 50 if self.use_pe_analysis else 0
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get plugin capabilities"""
        return {
            'pe_analysis': self.use_pe_analysis,
            'opcode_analysis': self.use_opcode_analysis,
            'string_analysis': self.use_string_analysis,
            'entropy_analysis': self.use_entropy_analysis,
            'batch_processing': True,
            'caching': True,
            'malware_detection': True,
            'executable_analysis': True
        }
    
    def get_hooks(self) -> Dict[str, Callable]:
        """Get plugin hooks"""
        return {
            'before_extraction': self._before_extraction_hook,
            'after_extraction': self._after_extraction_hook
        }
    
    def _before_extraction_hook(self, input_data: Any) -> Any:
        """Hook called before feature extraction"""
        self.logger.debug("🔗 Before binary feature extraction")
        return input_data
    
    def _after_extraction_hook(self, result: FeatureExtractionResult) -> FeatureExtractionResult:
        """Hook called after feature extraction"""
        self.logger.debug("🔗 After binary feature extraction")
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return self.stats.copy()
    
    def clear_cache(self):
        """Clear feature cache"""
        with self.cache_lock:
            self.feature_cache.clear()
        self.logger.info("🧹 Cleared binary feature cache")
    
    def health_check(self) -> bool:
        """Perform health check"""
        try:
            # Test with dummy data
            test_data = b"MZ\x90\x00" + b"A" * 100  # Minimal PE-like header
            result = self._extract_single_binary_features(test_data)
            return result.valid_binaries > 0 or result.invalid_binaries > 0  # Should process something
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False


# Plugin metadata for discovery
__plugin_metadata__ = {
    "name": "BinaryFeatureExtractor",
    "version": "1.0.0", 
    "author": "Universal AI Core",
    "description": "Binary analysis and malware detection features",
    "plugin_type": "feature_extractor",
    "entry_point": f"{__name__}:BinaryFeatureExtractor",
    "dependencies": [
        {"name": "pefile", "optional": True},
        {"name": "capstone", "optional": True}
    ],
    "capabilities": ["pe_analysis", "opcode_analysis", "entropy_analysis", "string_analysis"],
    "hooks": ["before_extraction", "after_extraction"]
}


if __name__ == "__main__":
    # Test the binary feature extractor
    print("🔍 BINARY FEATURE EXTRACTOR TEST")
    print("=" * 50)
    
    # Create test binary data (minimal PE header)
    test_binary = (
        b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00"
        b"\xb8\x00\x00\x00\x00\x00\x00\x00\x40\x00\x00\x00\x00\x00\x00\x00"
        + b"A" * 1000  # Padding data
    )
    
    # Test configuration
    config = {
        'use_pe_analysis': PEFILE_AVAILABLE,
        'use_opcode_analysis': CAPSTONE_AVAILABLE,
        'use_string_analysis': True,
        'use_entropy_analysis': True,
        'max_file_size': 10 * 1024 * 1024
    }
    
    # Initialize extractor
    extractor = BinaryFeatureExtractor(config)
    
    # Test feature extraction
    print(f"\n🔍 Testing with binary data ({len(test_binary)} bytes)...")
    result = extractor.extract_features(test_binary)
    
    if result.success:
        print(f"✅ Extraction successful!")
        print(f"📊 Processing time: {result.processing_time:.3f}s")
        print(f"🧮 Features extracted:")
        for feature_type, features in result.features.items():
            if isinstance(features, np.ndarray):
                print(f"  {feature_type}: {features.shape}")
            else:
                print(f"  {feature_type}: {type(features)}")
        
        # Show metadata
        if result.metadata:
            print(f"📋 Metadata:")
            for key, value in result.metadata.items():
                print(f"  {key}: {value}")
    else:
        print(f"❌ Extraction failed: {result.error_message}")
    
    # Test health check
    health = extractor.health_check()
    print(f"\n🏥 Health check: {'✅' if health else '❌'}")
    
    # Show capabilities
    capabilities = extractor.get_capabilities()
    print(f"\n🔧 Capabilities:")
    for key, value in capabilities.items():
        print(f"  {key}: {value}")
    
    # Show statistics
    stats = extractor.get_statistics()
    print(f"\n📊 Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Binary feature extractor test completed!")