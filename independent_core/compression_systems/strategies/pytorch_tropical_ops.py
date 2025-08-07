"""
PyTorch Implementation of Tropical Mathematics Operations
Replaces JAX tropical operations with pure PyTorch implementations
PRODUCTION READY - NO PLACEHOLDERS - HARD FAILURES ONLY
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any
import logging
import math

logger = logging.getLogger(__name__)

# Tropical constants matching tropical_core.py
TROPICAL_ZERO = -1e38
TROPICAL_EPSILON = 1e-10


class PyTorchTropicalOps:
    """
    PyTorch implementation of tropical mathematics operations.
    Uses max-plus algebra: addition is max(a,b), multiplication is a+b.
    """
    
    def __init__(self, 
                 device: Optional[torch.device] = None,
                 dtype: torch.dtype = torch.float32,
                 compile_mode: bool = True):
        """
        Initialize tropical operations module.
        
        Args:
            device: Target device (cuda/cpu)
            dtype: Data type for operations
            compile_mode: Enable torch.compile optimization
        """
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.dtype = dtype
        self.compile_mode = compile_mode
        
        # Compile critical operations if available
        if compile_mode and hasattr(torch, 'compile'):
            try:
                self.tropical_add = torch.compile(self._tropical_add_impl, fullgraph=True)
                self.tropical_multiply = torch.compile(self._tropical_multiply_impl, fullgraph=True)
                self.tropical_matmul = torch.compile(self._tropical_matmul_impl, fullgraph=True)
                logger.info("Tropical operations compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile tropical ops: {e}")
                self.tropical_add = self._tropical_add_impl
                self.tropical_multiply = self._tropical_multiply_impl
                self.tropical_matmul = self._tropical_matmul_impl
        else:
            self.tropical_add = self._tropical_add_impl
            self.tropical_multiply = self._tropical_multiply_impl
            self.tropical_matmul = self._tropical_matmul_impl
    
    def _tropical_add_impl(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Tropical addition: max(a, b)
        In max-plus algebra, addition corresponds to taking the maximum.
        
        Args:
            a: First tensor
            b: Second tensor (must be broadcastable with a)
            
        Returns:
            Element-wise maximum of a and b
        """
        # Handle tropical zero (-infinity)
        a_finite = torch.where(a > TROPICAL_ZERO, a, torch.full_like(a, TROPICAL_ZERO))
        b_finite = torch.where(b > TROPICAL_ZERO, b, torch.full_like(b, TROPICAL_ZERO))
        
        return torch.maximum(a_finite, b_finite)
    
    def _tropical_multiply_impl(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Tropical multiplication: a + b
        In max-plus algebra, multiplication corresponds to addition.
        
        Args:
            a: First tensor
            b: Second tensor (must be broadcastable with a)
            
        Returns:
            Element-wise sum of a and b
        """
        # Check for tropical zeros
        a_is_zero = a <= TROPICAL_ZERO
        b_is_zero = b <= TROPICAL_ZERO
        
        # If either operand is tropical zero, result is tropical zero
        result = a + b
        result = torch.where(a_is_zero | b_is_zero, 
                           torch.full_like(result, TROPICAL_ZERO), 
                           result)
        
        # Overflow protection
        result = torch.clamp(result, min=TROPICAL_ZERO, max=1e38)
        
        return result
    
    def _tropical_matmul_impl(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Tropical matrix multiplication.
        C[i,j] = max_k(A[i,k] + B[k,j])
        
        Args:
            A: Matrix of shape (m, n)
            B: Matrix of shape (n, p)
            
        Returns:
            Result matrix of shape (m, p)
        """
        m, n = A.shape
        n2, p = B.shape
        
        if n != n2:
            raise ValueError(f"Matrix dimensions incompatible: {A.shape} x {B.shape}")
        
        # Efficient implementation using broadcasting
        # Expand dimensions for broadcasting
        A_expanded = A.unsqueeze(2)  # Shape: (m, n, 1)
        B_expanded = B.unsqueeze(0)  # Shape: (1, n, p)
        
        # Tropical multiplication (addition in standard arithmetic)
        products = A_expanded + B_expanded  # Shape: (m, n, p)
        
        # Handle tropical zeros
        products = torch.where(
            (A_expanded <= TROPICAL_ZERO) | (B_expanded <= TROPICAL_ZERO),
            torch.full_like(products, TROPICAL_ZERO),
            products
        )
        
        # Tropical addition (max operation)
        result = products.max(dim=1)[0]  # Shape: (m, p)
        
        return result
    
    def tropical_conv1d(self, 
                       input: torch.Tensor, 
                       weight: torch.Tensor, 
                       stride: int = 1, 
                       padding: int = 0) -> torch.Tensor:
        """
        Tropical 1D convolution using max-plus algebra.
        
        Args:
            input: Input tensor of shape (batch, channels, length)
            weight: Convolution kernel of shape (out_channels, in_channels, kernel_size)
            stride: Stride for convolution
            padding: Padding for convolution
            
        Returns:
            Convolved tensor
        """
        batch_size, in_channels, length = input.shape
        out_channels, in_channels2, kernel_size = weight.shape
        
        if in_channels != in_channels2:
            raise ValueError(f"Channel mismatch: input has {in_channels}, weight has {in_channels2}")
        
        # Apply padding if needed
        if padding > 0:
            input = F.pad(input, (padding, padding), value=TROPICAL_ZERO)
            length = length + 2 * padding
        
        # Calculate output length
        out_length = (length - kernel_size) // stride + 1
        
        # Initialize output
        output = torch.full(
            (batch_size, out_channels, out_length), 
            TROPICAL_ZERO, 
            device=input.device, 
            dtype=input.dtype
        )
        
        # Perform tropical convolution
        for i in range(out_length):
            start_idx = i * stride
            end_idx = start_idx + kernel_size
            
            # Extract input window
            input_window = input[:, :, start_idx:end_idx]  # (batch, in_channels, kernel_size)
            
            # Compute tropical convolution for this position
            for oc in range(out_channels):
                # Tropical multiplication (addition)
                products = input_window + weight[oc].unsqueeze(0)  # (batch, in_channels, kernel_size)
                
                # Tropical addition across channels and kernel (max)
                output[:, oc, i] = products.flatten(1).max(dim=1)[0]
        
        return output
    
    def tropical_polynomial_eval(self, 
                                coeffs: torch.Tensor, 
                                x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate tropical polynomial.
        p(x) = max_i(coeffs[i] + i*x)
        
        Args:
            coeffs: Polynomial coefficients of shape (degree+1,) or (batch, degree+1)
            x: Points to evaluate at, shape () or (batch,) or (batch, n_points)
            
        Returns:
            Polynomial values at x
        """
        # Handle different input shapes
        if coeffs.dim() == 1:
            coeffs = coeffs.unsqueeze(0)  # Add batch dimension
        if x.dim() == 0:
            x = x.unsqueeze(0)  # Make it 1D
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        
        batch_size, degree_plus_one = coeffs.shape
        batch_size_x, n_points = x.shape if x.dim() == 2 else (x.shape[0], 1)
        
        # Create degree tensor
        degrees = torch.arange(degree_plus_one, device=coeffs.device, dtype=coeffs.dtype)
        
        # Reshape for broadcasting
        coeffs = coeffs.unsqueeze(2)  # (batch, degree+1, 1)
        degrees = degrees.unsqueeze(0).unsqueeze(2)  # (1, degree+1, 1)
        x = x.unsqueeze(1)  # (batch, 1, n_points)
        
        # Compute tropical polynomial: max_i(coeffs[i] + i*x)
        terms = coeffs + degrees * x  # (batch, degree+1, n_points)
        
        # Tropical addition (max over degrees)
        result = terms.max(dim=1)[0]  # (batch, n_points)
        
        # Squeeze if needed
        if n_points == 1:
            result = result.squeeze(-1)
        
        return result
    
    def tropical_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Compute tropical distance between tensors.
        d(a,b) = max_i |a_i - b_i|
        
        Args:
            a: First tensor
            b: Second tensor
            
        Returns:
            Tropical distance
        """
        diff = torch.abs(a - b)
        
        # Handle tropical zeros
        valid_mask = (a > TROPICAL_ZERO) & (b > TROPICAL_ZERO)
        diff = torch.where(valid_mask, diff, torch.zeros_like(diff))
        
        return diff.max()
    
    def tropical_inner_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Tropical inner product: max_i(a_i + b_i)
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Tropical inner product
        """
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
        
        # Tropical multiplication (addition)
        products = self.tropical_multiply(a, b)
        
        # Tropical sum (max)
        return products.max()
    
    def tropical_power(self, A: torch.Tensor, n: int) -> torch.Tensor:
        """
        Compute tropical matrix power A^n.
        
        Args:
            A: Square matrix
            n: Power (must be non-negative)
            
        Returns:
            A^n in tropical arithmetic
        """
        if A.dim() != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"Expected square matrix, got shape {A.shape}")
        if n < 0:
            raise ValueError(f"Power must be non-negative, got {n}")
        
        if n == 0:
            # Tropical identity matrix (0 on diagonal, -inf elsewhere)
            size = A.shape[0]
            result = torch.full_like(A, TROPICAL_ZERO)
            result.diagonal().fill_(0)
            return result
        
        result = A.clone()
        for _ in range(n - 1):
            result = self.tropical_matmul(result, A)
        
        return result
    
    def tropical_trace(self, A: torch.Tensor) -> torch.Tensor:
        """
        Compute tropical trace: max_i A[i,i]
        
        Args:
            A: Square matrix
            
        Returns:
            Tropical trace
        """
        if A.dim() != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"Expected square matrix, got shape {A.shape}")
        
        return A.diagonal().max()
    
    def tropical_determinant(self, A: torch.Tensor) -> torch.Tensor:
        """
        Compute tropical determinant (permanent in tropical arithmetic).
        
        Args:
            A: Square matrix
            
        Returns:
            Tropical determinant
        """
        n = A.shape[0]
        if A.shape != (n, n):
            raise ValueError(f"Expected square matrix, got shape {A.shape}")
        
        if n == 1:
            return A[0, 0]
        if n == 2:
            # max(a00 + a11, a01 + a10)
            return torch.max(A[0, 0] + A[1, 1], A[0, 1] + A[1, 0])
        
        # For larger matrices, use recursive definition
        # This is computationally expensive for large matrices
        result = torch.tensor(TROPICAL_ZERO, device=A.device, dtype=A.dtype)
        
        # Generate all permutations (expensive for large n)
        import itertools
        for perm in itertools.permutations(range(n)):
            term = torch.tensor(0.0, device=A.device, dtype=A.dtype)
            for i, j in enumerate(perm):
                term = term + A[i, j]
            result = torch.max(result, term)
        
        return result
    
    def tropical_solve(self, A: torch.Tensor, b: torch.Tensor, 
                      max_iter: int = 100) -> torch.Tensor:
        """
        Solve tropical linear system Ax = b using iterative method.
        
        Args:
            A: Coefficient matrix of shape (n, n)
            b: Right-hand side vector of shape (n,)
            max_iter: Maximum iterations
            
        Returns:
            Solution vector x
        """
        n = A.shape[0]
        if A.shape != (n, n):
            raise ValueError(f"Expected square matrix, got shape {A.shape}")
        if b.shape != (n,):
            raise ValueError(f"Expected vector of length {n}, got shape {b.shape}")
        
        # Initialize solution
        x = torch.zeros(n, device=A.device, dtype=A.dtype)
        
        # Iterative tropical Jacobi method
        for iteration in range(max_iter):
            x_new = torch.full_like(x, TROPICAL_ZERO)
            
            for i in range(n):
                # Compute x_i = max_j(b_i - A[i,j] + x_j) for j != i
                for j in range(n):
                    if i != j:
                        term = b[i] - A[i, j] + x[j]
                        x_new[i] = torch.max(x_new[i], term)
            
            # Check convergence
            if torch.allclose(x, x_new, atol=TROPICAL_EPSILON):
                break
            
            x = x_new
        
        return x
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to configured device."""
        return tensor.to(device=self.device, dtype=self.dtype)
    
    def create_tropical_tensor(self, 
                              data: Union[List, torch.Tensor, float],
                              shape: Optional[Tuple] = None) -> torch.Tensor:
        """
        Create a tropical tensor with proper initialization.
        
        Args:
            data: Input data (list, tensor, or scalar)
            shape: Optional shape for tensor creation
            
        Returns:
            Tropical tensor
        """
        if isinstance(data, torch.Tensor):
            tensor = data.clone()
        elif isinstance(data, (int, float)):
            if shape is None:
                tensor = torch.tensor(data)
            else:
                tensor = torch.full(shape, data)
        else:
            tensor = torch.tensor(data)
        
        # Move to device and dtype
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        
        # Replace infinities with tropical zero
        tensor = torch.where(torch.isinf(tensor) & (tensor < 0), 
                           torch.full_like(tensor, TROPICAL_ZERO), 
                           tensor)
        
        return tensor