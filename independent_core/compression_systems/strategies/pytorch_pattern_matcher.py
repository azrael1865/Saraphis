"""
Pure PyTorch Pattern Matching for Compression System
Replaces JAX-based pattern matching with optimized PyTorch operations
PRODUCTION READY - NO PLACEHOLDERS - HARD FAILURES ONLY
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PatternMatchResult:
    """Result from pattern matching operation"""
    pattern_indices: torch.Tensor
    position_indices: torch.Tensor
    confidence_scores: Optional[torch.Tensor] = None
    match_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            'pattern_indices': self.pattern_indices.cpu().numpy().tolist(),
            'position_indices': self.position_indices.cpu().numpy().tolist(),
            'match_count': self.match_count
        }
        if self.confidence_scores is not None:
            result['confidence_scores'] = self.confidence_scores.cpu().numpy().tolist()
        return result


class PyTorchPatternMatcher(nn.Module):
    """
    Pure PyTorch pattern matching using unfold for sliding windows.
    Optimized for GPU acceleration with torch.compile support.
    """
    
    def __init__(self, 
                 compile_mode: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32):
        """
        Initialize pattern matcher with optional compilation.
        
        Args:
            compile_mode: Enable torch.compile for JIT optimization
            device: Target device (cuda/cpu), auto-detected if None
            dtype: Data type for operations
        """
        super().__init__()
        self.compile_mode = compile_mode
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.dtype = dtype
        
        # Compile critical methods if torch.compile is available
        if compile_mode and hasattr(torch, 'compile'):
            try:
                # Configure dynamic shapes for torch.compile
                import torch._dynamo as dynamo
                dynamo.config.capture_dynamic_output_shape_ops = True
                
                self._find_patterns_impl = torch.compile(
                    self._find_patterns_impl, 
                    fullgraph=False,  # Allow graph breaks for dynamic operations
                    mode='default'  # Use default mode for better compatibility
                )
                logger.info("Pattern matcher compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile pattern matcher: {e}")
                self.compile_mode = False
    
    def _find_patterns_impl(self, 
                            data: torch.Tensor, 
                            patterns: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core pattern matching implementation using unfold.
        
        Args:
            data: Input tensor of shape (batch_size, seq_len) or (seq_len,)
            patterns: Pattern tensor of shape (num_patterns, pattern_len)
            
        Returns:
            Tuple of (pattern_indices, position_indices)
        """
        # Ensure 2D data tensor
        if data.dim() == 1:
            data = data.unsqueeze(0)
        
        batch_size, seq_len = data.shape
        num_patterns, pattern_len = patterns.shape
        
        # Validate inputs
        if seq_len < pattern_len:
            return torch.tensor([], dtype=torch.long, device=self.device), \
                   torch.tensor([], dtype=torch.long, device=self.device)
        
        # Use unfold for efficient sliding window extraction
        # Shape: (batch_size, num_windows, pattern_len)
        windows = data.unfold(dimension=1, size=pattern_len, step=1)
        num_windows = windows.shape[1]
        
        # Reshape for broadcasting
        # windows: (batch_size, num_windows, 1, pattern_len)
        # patterns: (1, 1, num_patterns, pattern_len)
        windows = windows.unsqueeze(2)
        patterns_expanded = patterns.unsqueeze(0).unsqueeze(0)
        
        # Compare all windows with all patterns
        # Shape: (batch_size, num_windows, num_patterns)
        matches = (windows == patterns_expanded).all(dim=-1)
        
        # Find matching indices
        batch_idx, pos_idx, pattern_idx = torch.where(matches)
        
        # For single batch, return position and pattern indices
        if batch_size == 1:
            return pattern_idx, pos_idx
        
        # For multiple batches, include batch index
        return torch.stack([batch_idx, pattern_idx], dim=1), pos_idx
    
    def find_patterns(self, 
                     data: torch.Tensor, 
                     patterns: torch.Tensor) -> PatternMatchResult:
        """
        Find exact pattern matches in data.
        
        Args:
            data: Input tensor to search
            patterns: Patterns to find
            
        Returns:
            PatternMatchResult containing match information
        """
        # Move to device and ensure correct dtype
        data = data.to(device=self.device, dtype=self.dtype)
        patterns = patterns.to(device=self.device, dtype=self.dtype)
        
        pattern_indices, position_indices = self._find_patterns_impl(data, patterns)
        
        return PatternMatchResult(
            pattern_indices=pattern_indices,
            position_indices=position_indices,
            match_count=len(pattern_indices)
        )
    
    def find_patterns_batched(self, 
                             data: torch.Tensor, 
                             patterns: torch.Tensor,
                             batch_size: int = 32) -> Dict[str, torch.Tensor]:
        """
        Batched pattern finding with memory efficiency.
        
        Args:
            data: Input tensor
            patterns: Patterns to find
            batch_size: Number of patterns to process at once
            
        Returns:
            Dictionary with pattern statistics and indices
        """
        data = data.to(device=self.device, dtype=self.dtype)
        patterns = patterns.to(device=self.device, dtype=self.dtype)
        
        if data.dim() == 1:
            data = data.unsqueeze(0)
        
        num_patterns = patterns.shape[0]
        all_pattern_indices = []
        all_position_indices = []
        pattern_counts = torch.zeros(num_patterns, dtype=torch.long, device=self.device)
        
        # Process patterns in batches to manage memory
        for i in range(0, num_patterns, batch_size):
            batch_patterns = patterns[i:min(i+batch_size, num_patterns)]
            pattern_idx, pos_idx = self._find_patterns_impl(data, batch_patterns)
            
            # Adjust pattern indices for batch offset
            if len(pattern_idx) > 0:
                pattern_idx = pattern_idx + i
                all_pattern_indices.append(pattern_idx)
                all_position_indices.append(pos_idx)
                
                # Count occurrences per pattern
                for j in range(len(batch_patterns)):
                    pattern_counts[i + j] = (pattern_idx == i + j).sum()
        
        # Concatenate results
        if all_pattern_indices:
            pattern_indices = torch.cat(all_pattern_indices)
            position_indices = torch.cat(all_position_indices)
        else:
            pattern_indices = torch.tensor([], dtype=torch.long, device=self.device)
            position_indices = torch.tensor([], dtype=torch.long, device=self.device)
        
        return {
            'pattern_indices': pattern_indices,
            'position_indices': position_indices,
            'pattern_counts': pattern_counts,
            'total_matches': len(pattern_indices),
            'patterns_with_matches': (pattern_counts > 0).sum().item()
        }
    
    def find_approximate_patterns(self, 
                                 data: torch.Tensor, 
                                 patterns: torch.Tensor, 
                                 tolerance: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find patterns with tolerance for approximate matching.
        
        Args:
            data: Input tensor
            patterns: Patterns to find
            tolerance: Maximum relative difference for match
            
        Returns:
            Tuple of (pattern_indices, position_indices, confidence_scores)
        """
        data = data.to(device=self.device, dtype=self.dtype)
        patterns = patterns.to(device=self.device, dtype=self.dtype)
        
        if data.dim() == 1:
            data = data.unsqueeze(0)
        
        batch_size, seq_len = data.shape
        num_patterns, pattern_len = patterns.shape
        
        if seq_len < pattern_len:
            return (torch.tensor([], dtype=torch.long, device=self.device),
                   torch.tensor([], dtype=torch.long, device=self.device),
                   torch.tensor([], dtype=self.dtype, device=self.device))
        
        # Extract windows
        windows = data.unfold(dimension=1, size=pattern_len, step=1)
        num_windows = windows.shape[1]
        
        # Compute relative differences
        windows_expanded = windows.unsqueeze(2)  # (batch, windows, 1, pattern_len)
        patterns_expanded = patterns.unsqueeze(0).unsqueeze(0)  # (1, 1, patterns, pattern_len)
        
        # Compute element-wise differences
        diff = torch.abs(windows_expanded - patterns_expanded)
        
        # Normalize by pattern magnitude (avoid division by zero)
        pattern_norm = torch.abs(patterns_expanded) + 1e-10
        relative_diff = diff / pattern_norm
        
        # Check if all elements are within tolerance
        matches_mask = (relative_diff <= tolerance).all(dim=-1)
        
        # Compute confidence scores (1 - mean relative difference)
        confidence = 1.0 - relative_diff.mean(dim=-1)
        confidence = torch.where(matches_mask, confidence, torch.zeros_like(confidence))
        
        # Find matching indices
        batch_idx, pos_idx, pattern_idx = torch.where(matches_mask)
        
        # Get corresponding confidence scores
        confidence_scores = confidence[batch_idx, pos_idx, pattern_idx]
        
        if batch_size == 1:
            return pattern_idx, pos_idx, confidence_scores
        
        return torch.stack([batch_idx, pattern_idx], dim=1), pos_idx, confidence_scores
    
    def find_recurring_patterns(self, 
                               data: torch.Tensor,
                               min_length: int = 3,
                               max_length: int = 32,
                               min_occurrences: int = 2) -> Dict[str, Any]:
        """
        Automatically detect recurring patterns in data.
        
        Args:
            data: Input tensor to analyze
            min_length: Minimum pattern length
            max_length: Maximum pattern length
            min_occurrences: Minimum number of occurrences
            
        Returns:
            Dictionary with discovered patterns and statistics
        """
        data = data.to(device=self.device, dtype=self.dtype)
        
        if data.dim() > 1:
            data = data.flatten()
        
        discovered_patterns = []
        pattern_stats = []
        
        for length in range(min_length, min(max_length + 1, len(data) // 2)):
            # Extract all possible patterns of this length
            windows = data.unfold(0, length, 1)
            
            # Find unique patterns
            unique_patterns, inverse_indices = torch.unique(windows, dim=0, return_inverse=True)
            
            # Count occurrences
            for i, pattern in enumerate(unique_patterns):
                count = (inverse_indices == i).sum().item()
                if count >= min_occurrences:
                    positions = torch.where(inverse_indices == i)[0]
                    
                    # Check if positions are sufficiently spaced (not overlapping)
                    if len(positions) > 1:
                        gaps = positions[1:] - positions[:-1]
                        if gaps.min() >= length:  # Non-overlapping
                            discovered_patterns.append(pattern)
                            pattern_stats.append({
                                'pattern': pattern.cpu().numpy(),
                                'length': length,
                                'occurrences': count,
                                'positions': positions.cpu().numpy(),
                                'compression_ratio': len(data) / (length + count * 2)  # Rough estimate
                            })
        
        # Sort by compression potential
        pattern_stats.sort(key=lambda x: x['compression_ratio'], reverse=True)
        
        return {
            'num_patterns': len(discovered_patterns),
            'patterns': discovered_patterns[:10] if discovered_patterns else [],  # Top 10
            'statistics': pattern_stats[:10] if pattern_stats else [],
            'total_elements': len(data),
            'potential_compression': max([s['compression_ratio'] for s in pattern_stats]) if pattern_stats else 1.0
        }
    
    def compute_pattern_entropy(self, 
                               data: torch.Tensor,
                               pattern_length: int = 4) -> float:
        """
        Compute entropy based on pattern distribution.
        
        Args:
            data: Input tensor
            pattern_length: Length of patterns to consider
            
        Returns:
            Entropy value (lower means more compressible)
        """
        data = data.to(device=self.device, dtype=self.dtype)
        
        if data.dim() > 1:
            data = data.flatten()
        
        if len(data) < pattern_length:
            return float('inf')
        
        # Extract patterns
        windows = data.unfold(0, pattern_length, 1)
        
        # Find unique patterns and counts
        unique_patterns, counts = torch.unique(windows, dim=0, return_counts=True)
        
        # Compute probabilities
        probs = counts.float() / counts.sum()
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
        
        return entropy.item()
    
    def find_hierarchical_patterns(self,
                                  data: torch.Tensor,
                                  levels: List[int] = [2, 4, 8, 16]) -> Dict[str, Any]:
        """
        Find patterns at multiple hierarchical levels.
        
        Args:
            data: Input tensor
            levels: Pattern lengths to analyze
            
        Returns:
            Dictionary with multi-level pattern analysis
        """
        data = data.to(device=self.device, dtype=self.dtype)
        
        if data.dim() > 1:
            data = data.flatten()
        
        hierarchy = {}
        
        for level in levels:
            if len(data) < level:
                continue
            
            # Extract patterns at this level
            windows = data.unfold(0, level, 1)
            unique_patterns, inverse = torch.unique(windows, dim=0, return_inverse=True)
            counts = torch.bincount(inverse)
            
            # Find most common patterns
            top_k = min(10, len(unique_patterns))
            top_counts, top_indices = torch.topk(counts, top_k)
            
            hierarchy[f'level_{level}'] = {
                'num_unique': len(unique_patterns),
                'top_patterns': unique_patterns[top_indices].cpu().numpy(),
                'top_counts': top_counts.cpu().numpy(),
                'entropy': self.compute_pattern_entropy(data, level),
                'redundancy': 1.0 - (len(unique_patterns) / len(windows))
            }
        
        return hierarchy
    
    def forward(self, data: torch.Tensor, patterns: torch.Tensor) -> PatternMatchResult:
        """
        Forward pass for nn.Module compatibility.
        
        Args:
            data: Input tensor
            patterns: Patterns to match
            
        Returns:
            PatternMatchResult
        """
        return self.find_patterns(data, patterns)