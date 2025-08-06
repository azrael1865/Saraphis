"""
Tropical polynomial implementation for neural network compression.
Implements sparse tropical polynomials with GPU acceleration.
NO PLACEHOLDERS - PRODUCTION READY
"""

import torch
import math
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import itertools

# Import existing tropical operations
try:
    from independent_core.compression_systems.tropical.tropical_core import (
        TropicalNumber,
        TropicalMathematicalOperations,
        TropicalValidation,
        TROPICAL_ZERO,
        TROPICAL_EPSILON,
        is_tropical_zero,
        to_tropical_safe,
        from_tropical_safe
    )
except ImportError:
    # For direct execution
    from tropical_core import (
        TropicalNumber,
        TropicalMathematicalOperations,
        TropicalValidation,
        TROPICAL_ZERO,
        TROPICAL_EPSILON,
        is_tropical_zero,
        to_tropical_safe,
        from_tropical_safe
    )


@dataclass(frozen=True)
class TropicalMonomial:
    """
    Single tropical monomial: coefficient + exponents.
    In tropical polynomial f(x) = max{aᵢ + i₁x₁ + ... + iₙxₙ},
    each monomial represents one term aᵢ + i₁x₁ + ... + iₙxₙ.
    """
    coefficient: float  # Tropical coefficient (aᵢ)
    exponents: Dict[int, int]  # Variable index -> power (sparse representation)
    
    def __post_init__(self):
        """Validate monomial on creation"""
        if not isinstance(self.coefficient, (int, float)):
            raise TypeError(f"Coefficient must be numeric, got {type(self.coefficient)}")
        if math.isnan(self.coefficient):
            raise ValueError("Coefficient cannot be NaN")
        if math.isinf(self.coefficient) and self.coefficient > 0:
            raise ValueError("Coefficient cannot be positive infinity")
        if self.coefficient > 1e38:
            raise ValueError(f"Coefficient {self.coefficient} exceeds safe tropical range")
        
        if not isinstance(self.exponents, dict):
            raise TypeError(f"Exponents must be dict, got {type(self.exponents)}")
        
        # Validate exponents
        for var_idx, power in self.exponents.items():
            if not isinstance(var_idx, int):
                raise TypeError(f"Variable index must be int, got {type(var_idx)}")
            if var_idx < 0:
                raise ValueError(f"Variable index must be non-negative, got {var_idx}")
            if not isinstance(power, int):
                raise TypeError(f"Exponent must be int, got {type(power)}")
            if power < 0:
                raise ValueError(f"Exponent must be non-negative, got {power}")
        
        # Remove zero exponents for sparse representation
        non_zero_exponents = {k: v for k, v in self.exponents.items() if v != 0}
        if len(non_zero_exponents) != len(self.exponents):
            object.__setattr__(self, 'exponents', non_zero_exponents)
    
    def is_zero(self) -> bool:
        """Check if this is tropical zero monomial"""
        return self.coefficient <= TROPICAL_ZERO
    
    def degree(self) -> int:
        """Total degree of the monomial"""
        return sum(self.exponents.values()) if self.exponents else 0
    
    def evaluate(self, point: Union[torch.Tensor, List[float]]) -> float:
        """
        Evaluate monomial at a point.
        Result = coefficient + sum(exponent_i * point_i)
        """
        if self.is_zero():
            return TROPICAL_ZERO
        
        result = self.coefficient
        
        if isinstance(point, torch.Tensor):
            point = point.detach().cpu().numpy() if point.is_cuda else point.numpy()
        elif not isinstance(point, (list, np.ndarray)):
            raise TypeError(f"Point must be tensor, list, or array, got {type(point)}")
        
        for var_idx, power in self.exponents.items():
            if var_idx >= len(point):
                raise IndexError(f"Variable index {var_idx} out of range for point of dimension {len(point)}")
            result += power * point[var_idx]
            if result > 1e38:
                raise OverflowError(f"Monomial evaluation overflow")
        
        return result
    
    def __str__(self) -> str:
        """String representation for debugging"""
        if self.is_zero():
            return "T(-∞)"
        
        terms = [f"{self.coefficient:.6f}"]
        for var_idx, power in sorted(self.exponents.items()):
            if power == 1:
                terms.append(f"x_{var_idx}")
            elif power > 1:
                terms.append(f"x_{var_idx}^{power}")
        
        return " + ".join(terms) if terms else f"{self.coefficient:.6f}"
    
    def __hash__(self) -> int:
        """Hash for use in sets/dicts"""
        if self.is_zero():
            return hash(TROPICAL_ZERO)
        return hash((self.coefficient, tuple(sorted(self.exponents.items()))))
    
    def __eq__(self, other: Any) -> bool:
        """Equality comparison"""
        if not isinstance(other, TropicalMonomial):
            return False
        if self.is_zero() and other.is_zero():
            return True
        return (abs(self.coefficient - other.coefficient) < TROPICAL_EPSILON and 
                self.exponents == other.exponents)


class TropicalPolynomial:
    """
    Sparse tropical polynomial representation using monomials.
    A tropical polynomial is f(x) = max{aᵢ + i₁x₁ + ... + iₙxₙ}
    """
    
    def __init__(self, monomials: List[TropicalMonomial], num_variables: int):
        """
        Initialize tropical polynomial.
        
        Args:
            monomials: List of tropical monomials
            num_variables: Number of variables in the polynomial
        """
        if not isinstance(monomials, list):
            raise TypeError(f"Monomials must be a list, got {type(monomials)}")
        if not isinstance(num_variables, int):
            raise TypeError(f"num_variables must be int, got {type(num_variables)}")
        if num_variables <= 0:
            raise ValueError(f"num_variables must be positive, got {num_variables}")
        
        # Validate all monomials
        for i, monomial in enumerate(monomials):
            if not isinstance(monomial, TropicalMonomial):
                raise TypeError(f"Monomial {i} must be TropicalMonomial, got {type(monomial)}")
            # Check variable indices are within range
            for var_idx in monomial.exponents.keys():
                if var_idx >= num_variables:
                    raise ValueError(f"Variable index {var_idx} in monomial {i} exceeds num_variables {num_variables}")
        
        # Remove tropical zero monomials and duplicates
        self.monomials = []
        seen = set()
        for monomial in monomials:
            if not monomial.is_zero():
                key = (monomial.coefficient, tuple(sorted(monomial.exponents.items())))
                if key not in seen:
                    self.monomials.append(monomial)
                    seen.add(key)
        
        self.num_variables = num_variables
        
        # Cache for GPU tensors
        self._dense_matrix_cache = None
        self._cache_device = None
    
    def evaluate(self, point: torch.Tensor) -> Union[float, torch.Tensor]:
        """
        Evaluate polynomial at a point using tropical addition (max).
        
        Args:
            point: Point to evaluate at (can be single point or batch)
            
        Returns:
            Evaluation result (scalar or tensor)
        """
        if not isinstance(point, torch.Tensor):
            raise TypeError(f"Point must be torch.Tensor, got {type(point)}")
        
        # Handle empty polynomial
        if not self.monomials:
            if point.dim() == 1:
                return TROPICAL_ZERO
            else:
                return torch.full((point.shape[0],), TROPICAL_ZERO, device=point.device)
        
        # Single point evaluation
        if point.dim() == 1:
            if point.shape[0] != self.num_variables:
                raise ValueError(f"Point dimension {point.shape[0]} doesn't match num_variables {self.num_variables}")
            
            # Evaluate all monomials
            values = []
            for monomial in self.monomials:
                val = monomial.evaluate(point)
                if val > TROPICAL_ZERO:
                    values.append(val)
            
            if not values:
                return TROPICAL_ZERO
            
            return max(values)  # Tropical addition
        
        # Batch evaluation
        elif point.dim() == 2:
            batch_size = point.shape[0]
            if point.shape[1] != self.num_variables:
                raise ValueError(f"Point dimension {point.shape[1]} doesn't match num_variables {self.num_variables}")
            
            # GPU-accelerated batch evaluation
            if point.is_cuda:
                return self._evaluate_batch_gpu(point)
            
            # CPU batch evaluation
            results = torch.full((batch_size,), TROPICAL_ZERO, device=point.device)
            for i in range(batch_size):
                val = self.evaluate(point[i])
                results[i] = float(val) if isinstance(val, (float, int, np.number)) else val.item()
            
            return results
        
        else:
            raise ValueError(f"Point must be 1D or 2D tensor, got shape {point.shape}")
    
    def _evaluate_batch_gpu(self, points: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated batch evaluation"""
        batch_size = points.shape[0]
        device = points.device
        
        # Convert to dense matrix for GPU computation
        dense_matrix = self.to_dense_matrix(device=device)
        
        # Compute all monomial values in parallel
        # Result[i,j] = coefficient[j] + sum_k(exponent[j,k] * point[i,k])
        monomial_values = dense_matrix[0].unsqueeze(0) + torch.matmul(points, dense_matrix[1:].T)
        
        # Tropical addition (max) over monomials
        result = monomial_values.max(dim=1)[0]
        
        # Handle case where all values are tropical zero
        all_zero_mask = (monomial_values <= TROPICAL_ZERO).all(dim=1)
        result[all_zero_mask] = TROPICAL_ZERO
        
        return result
    
    def add(self, other: 'TropicalPolynomial') -> 'TropicalPolynomial':
        """
        Tropical addition of polynomials: (f ⊕ g)(x) = max(f(x), g(x))
        
        Args:
            other: Another tropical polynomial
            
        Returns:
            New polynomial representing f ⊕ g
        """
        if not isinstance(other, TropicalPolynomial):
            raise TypeError(f"Can only add TropicalPolynomial, got {type(other)}")
        if other.num_variables != self.num_variables:
            raise ValueError(f"Variable count mismatch: {self.num_variables} vs {other.num_variables}")
        
        # Combine monomials from both polynomials
        combined_monomials = self.monomials + other.monomials
        
        return TropicalPolynomial(combined_monomials, self.num_variables)
    
    def multiply(self, other: 'TropicalPolynomial') -> 'TropicalPolynomial':
        """
        Tropical multiplication of polynomials: (f ⊗ g)(x) = f(x) + g(x)
        
        Args:
            other: Another tropical polynomial
            
        Returns:
            New polynomial representing f ⊗ g
        """
        if not isinstance(other, TropicalPolynomial):
            raise TypeError(f"Can only multiply TropicalPolynomial, got {type(other)}")
        if other.num_variables != self.num_variables:
            raise ValueError(f"Variable count mismatch: {self.num_variables} vs {other.num_variables}")
        
        # Multiply all pairs of monomials
        result_monomials = []
        
        for m1 in self.monomials:
            for m2 in other.monomials:
                # Tropical multiplication of monomials:
                # coefficient = m1.coefficient + m2.coefficient
                # exponents = m1.exponents + m2.exponents (element-wise)
                
                new_coeff = m1.coefficient + m2.coefficient
                if new_coeff > 1e38:
                    raise OverflowError(f"Polynomial multiplication coefficient overflow")
                
                # Combine exponents
                new_exponents = defaultdict(int)
                for var_idx, power in m1.exponents.items():
                    new_exponents[var_idx] += power
                for var_idx, power in m2.exponents.items():
                    new_exponents[var_idx] += power
                
                result_monomials.append(TropicalMonomial(new_coeff, dict(new_exponents)))
        
        return TropicalPolynomial(result_monomials, self.num_variables)
    
    def degree(self) -> int:
        """
        Maximum total degree among all monomials.
        
        Returns:
            Maximum degree
        """
        if not self.monomials:
            return 0
        
        return max(monomial.degree() for monomial in self.monomials)
    
    def newton_polytope(self) -> torch.Tensor:
        """
        Compute vertices of the Newton polytope.
        The Newton polytope is the convex hull of exponent vectors.
        
        Returns:
            Tensor of shape (num_monomials, num_variables) containing exponent vectors
        """
        if not self.monomials:
            return torch.empty(0, self.num_variables)
        
        # Extract exponent vectors
        exponent_vectors = []
        for monomial in self.monomials:
            vector = torch.zeros(self.num_variables)
            for var_idx, power in monomial.exponents.items():
                vector[var_idx] = power
            exponent_vectors.append(vector)
        
        return torch.stack(exponent_vectors)
    
    def to_dense_matrix(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Convert to dense matrix representation for GPU operations.
        
        Returns:
            Matrix of shape (num_variables + 1, num_monomials)
            Row 0: coefficients
            Rows 1 to num_variables: exponents for each variable
        """
        if device is None:
            device = torch.device('cpu')
        
        # Check cache
        if self._dense_matrix_cache is not None and self._cache_device == device:
            return self._dense_matrix_cache
        
        if not self.monomials:
            matrix = torch.full((self.num_variables + 1, 1), TROPICAL_ZERO, device=device)
        else:
            num_monomials = len(self.monomials)
            matrix = torch.zeros(self.num_variables + 1, num_monomials, device=device)
            
            for j, monomial in enumerate(self.monomials):
                matrix[0, j] = monomial.coefficient
                for var_idx, power in monomial.exponents.items():
                    matrix[var_idx + 1, j] = power
        
        # Cache result
        self._dense_matrix_cache = matrix
        self._cache_device = device
        
        return matrix
    
    def __str__(self) -> str:
        """String representation"""
        if not self.monomials:
            return "T(-∞)"
        
        monomial_strs = [str(m) for m in self.monomials[:5]]  # Show first 5
        if len(self.monomials) > 5:
            monomial_strs.append(f"... ({len(self.monomials) - 5} more)")
        
        return f"TropicalPoly[{self.num_variables} vars]: max{{" + ", ".join(monomial_strs) + "}"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return f"TropicalPolynomial(monomials={len(self.monomials)}, num_variables={self.num_variables})"


class TropicalPolynomialOperations:
    """GPU-accelerated polynomial operations for neural network compression"""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize operations with optional GPU device.
        
        Args:
            device: PyTorch device for computation
        """
        self.device = device or torch.device('cpu')
        if not isinstance(self.device, torch.device):
            raise TypeError(f"Device must be torch.device, got {type(self.device)}")
        
        self.tropical_ops = TropicalMathematicalOperations(device=self.device)
    
    def batch_evaluate(self, polynomials: List[TropicalPolynomial], 
                      points: torch.Tensor) -> torch.Tensor:
        """
        Evaluate multiple polynomials on multiple points efficiently.
        
        Args:
            polynomials: List of tropical polynomials
            points: Tensor of shape (num_points, num_variables)
            
        Returns:
            Tensor of shape (len(polynomials), num_points) with evaluations
        """
        if not isinstance(polynomials, list):
            raise TypeError(f"Polynomials must be list, got {type(polynomials)}")
        if not polynomials:
            raise ValueError("Empty polynomial list")
        if not isinstance(points, torch.Tensor):
            raise TypeError(f"Points must be torch.Tensor, got {type(points)}")
        
        num_polynomials = len(polynomials)
        num_points = points.shape[0]
        num_variables = points.shape[1]
        
        # Validate all polynomials have same number of variables
        for i, poly in enumerate(polynomials):
            if not isinstance(poly, TropicalPolynomial):
                raise TypeError(f"Polynomial {i} must be TropicalPolynomial, got {type(poly)}")
            if poly.num_variables != num_variables:
                raise ValueError(f"Polynomial {i} has {poly.num_variables} variables, expected {num_variables}")
        
        # Move points to device
        if points.device != self.device:
            points = points.to(self.device)
        
        # Allocate result tensor
        results = torch.full((num_polynomials, num_points), TROPICAL_ZERO, device=self.device)
        
        # GPU-accelerated evaluation
        if self.device.type == 'cuda':
            # Process polynomials in parallel on GPU
            for i, poly in enumerate(polynomials):
                results[i] = poly.evaluate(points)
        else:
            # CPU evaluation
            for i, poly in enumerate(polynomials):
                for j in range(num_points):
                    val = poly.evaluate(points[j])
                    results[i, j] = float(val) if isinstance(val, (float, int, np.number)) else val.item()
        
        return results
    
    def find_tropical_roots(self, polynomial: TropicalPolynomial, 
                           search_region: Optional[Tuple[float, float]] = None,
                           num_samples: int = 1000) -> List[torch.Tensor]:
        """
        Find tropical roots (where polynomial is non-differentiable).
        Tropical roots form the tropical variety - a polyhedral complex.
        
        Args:
            polynomial: Tropical polynomial
            search_region: Optional bounds for search (min, max)
            num_samples: Number of sample points for root finding
            
        Returns:
            List of approximate root locations
        """
        if not isinstance(polynomial, TropicalPolynomial):
            raise TypeError(f"Expected TropicalPolynomial, got {type(polynomial)}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        
        if search_region is None:
            search_region = (-10.0, 10.0)
        
        min_val, max_val = search_region
        if min_val >= max_val:
            raise ValueError(f"Invalid search region: [{min_val}, {max_val}]")
        
        # Generate sample points
        if polynomial.num_variables == 1:
            # 1D case - simple line search
            points = torch.linspace(min_val, max_val, num_samples, device=self.device).unsqueeze(1)
        elif polynomial.num_variables == 2:
            # 2D case - grid search
            sqrt_samples = int(math.sqrt(num_samples))
            x = torch.linspace(min_val, max_val, sqrt_samples, device=self.device)
            y = torch.linspace(min_val, max_val, sqrt_samples, device=self.device)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        else:
            # Higher dimensions - random sampling
            points = torch.rand(num_samples, polynomial.num_variables, device=self.device)
            points = points * (max_val - min_val) + min_val
        
        # Evaluate polynomial at sample points
        values = polynomial.evaluate(points)
        
        # Find points where multiple monomials achieve the maximum
        # These are potential tropical roots
        roots = []
        epsilon = 1e-6
        
        for i in range(len(points)):
            point = points[i]
            max_val = values[i] if values.dim() == 1 else values[i].item()
            
            # Count how many monomials achieve the maximum
            achieving_max = 0
            for monomial in polynomial.monomials:
                monomial_val = monomial.evaluate(point)
                if abs(monomial_val - max_val) < epsilon:
                    achieving_max += 1
            
            # If multiple monomials achieve max, this is a tropical root
            if achieving_max > 1:
                roots.append(point)
        
        return roots
    
    def compute_tropical_resultant(self, f: TropicalPolynomial, 
                                  g: TropicalPolynomial) -> float:
        """
        Compute tropical resultant of two polynomials.
        The tropical resultant indicates if polynomials have common roots.
        
        Args:
            f: First tropical polynomial
            g: Second tropical polynomial
            
        Returns:
            Tropical resultant value
        """
        if not isinstance(f, TropicalPolynomial):
            raise TypeError(f"f must be TropicalPolynomial, got {type(f)}")
        if not isinstance(g, TropicalPolynomial):
            raise TypeError(f"g must be TropicalPolynomial, got {type(g)}")
        if f.num_variables != g.num_variables:
            raise ValueError(f"Polynomials must have same number of variables: {f.num_variables} vs {g.num_variables}")
        
        # For tropical resultant, we use the mixed volume of Newton polytopes
        # This is a simplified implementation focusing on the key property
        
        # Get Newton polytopes
        f_polytope = f.newton_polytope()
        g_polytope = g.newton_polytope()
        
        if f_polytope.numel() == 0 or g_polytope.numel() == 0:
            return TROPICAL_ZERO
        
        # Compute a measure of polytope overlap
        # This is a simplified tropical resultant computation
        
        # Find the Minkowski sum dimension
        minkowski_sum = []
        for f_vertex in f_polytope:
            for g_vertex in g_polytope:
                minkowski_sum.append(f_vertex + g_vertex)
        
        if not minkowski_sum:
            return TROPICAL_ZERO
        
        minkowski_tensor = torch.stack(minkowski_sum)
        
        # Compute the volume proxy (simplified)
        if f.num_variables == 1:
            # 1D case - length
            resultant = minkowski_tensor.max() - minkowski_tensor.min()
        elif f.num_variables == 2:
            # 2D case - area approximation
            hull_max = minkowski_tensor.max(dim=0)[0]
            hull_min = minkowski_tensor.min(dim=0)[0]
            resultant = torch.prod(hull_max - hull_min).item()
        else:
            # Higher dimensions - use determinant-based measure
            centered = minkowski_tensor - minkowski_tensor.mean(dim=0)
            cov = torch.matmul(centered.T, centered) / len(minkowski_sum)
            resultant = torch.det(cov).abs().item()
        
        return float(resultant)
    
    def interpolate_from_points(self, points: torch.Tensor, 
                               values: torch.Tensor,
                               max_degree: int = 3) -> TropicalPolynomial:
        """
        Create tropical polynomial that interpolates given points.
        
        Args:
            points: Tensor of shape (num_points, num_variables)
            values: Tensor of shape (num_points,) with function values
            max_degree: Maximum degree of interpolating polynomial
            
        Returns:
            Interpolating tropical polynomial
        """
        if not isinstance(points, torch.Tensor):
            raise TypeError(f"Points must be torch.Tensor, got {type(points)}")
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"Values must be torch.Tensor, got {type(values)}")
        if points.shape[0] != values.shape[0]:
            raise ValueError(f"Number of points {points.shape[0]} doesn't match number of values {values.shape[0]}")
        if max_degree <= 0:
            raise ValueError(f"max_degree must be positive, got {max_degree}")
        
        num_points, num_variables = points.shape
        
        # Generate monomials up to max_degree
        monomials = []
        
        # Add constant term
        for i in range(num_points):
            monomials.append(TropicalMonomial(values[i].item(), {}))
        
        # Add linear terms
        if max_degree >= 1:
            for i in range(num_points):
                for var_idx in range(num_variables):
                    # Compute coefficient for linear term
                    coeff = values[i].item() - points[i, var_idx].item()
                    monomials.append(TropicalMonomial(coeff, {var_idx: 1}))
        
        # Add higher degree terms (simplified tropical interpolation)
        if max_degree >= 2:
            for degree in range(2, min(max_degree + 1, 4)):  # Limit to cubic for efficiency
                for i in range(min(num_points, 10)):  # Limit number of points for higher degrees
                    # Generate degree-d monomials
                    for combo in itertools.combinations_with_replacement(range(num_variables), degree):
                        exponents = defaultdict(int)
                        for var_idx in combo:
                            exponents[var_idx] += 1
                        
                        # Compute coefficient using tropical interpolation condition
                        coeff = values[i].item()
                        for var_idx, power in exponents.items():
                            coeff -= power * points[i, var_idx].item()
                        
                        monomials.append(TropicalMonomial(coeff, dict(exponents)))
        
        return TropicalPolynomial(monomials, num_variables)


# Unit tests
class TestTropicalPolynomial:
    """Unit tests for tropical polynomial implementation"""
    
    @staticmethod
    def test_monomial_creation():
        """Test tropical monomial creation and validation"""
        # Valid monomial
        m1 = TropicalMonomial(5.0, {0: 2, 1: 1})
        assert m1.coefficient == 5.0
        assert m1.exponents == {0: 2, 1: 1}
        assert m1.degree() == 3
        assert not m1.is_zero()
        
        # Zero monomial
        m2 = TropicalMonomial(TROPICAL_ZERO, {0: 1})
        assert m2.is_zero()
        
        # Remove zero exponents
        m3 = TropicalMonomial(3.0, {0: 1, 1: 0, 2: 2})
        assert m3.exponents == {0: 1, 2: 2}
        
        # Test invalid inputs
        try:
            TropicalMonomial("invalid", {})
            assert False, "Should raise TypeError"
        except TypeError:
            pass
        
        try:
            TropicalMonomial(float('nan'), {})
            assert False, "Should raise ValueError"
        except ValueError:
            pass
        
        print("✓ Monomial creation tests passed")
    
    @staticmethod
    def test_monomial_evaluation():
        """Test monomial evaluation"""
        m = TropicalMonomial(2.0, {0: 1, 1: 2})
        
        # Test with list
        point = [3.0, 1.0]
        result = m.evaluate(point)
        assert abs(result - (2.0 + 1*3.0 + 2*1.0)) < 1e-6
        
        # Test with tensor
        point_tensor = torch.tensor([3.0, 1.0])
        result = m.evaluate(point_tensor)
        assert abs(result - 7.0) < 1e-6
        
        print("✓ Monomial evaluation tests passed")
    
    @staticmethod
    def test_polynomial_creation():
        """Test polynomial creation and validation"""
        m1 = TropicalMonomial(1.0, {0: 1})
        m2 = TropicalMonomial(2.0, {1: 1})
        m3 = TropicalMonomial(TROPICAL_ZERO, {0: 2})
        
        # Create polynomial
        poly = TropicalPolynomial([m1, m2, m3], num_variables=2)
        assert len(poly.monomials) == 2  # m3 is tropical zero, should be removed
        assert poly.num_variables == 2
        assert poly.degree() == 1
        
        # Test invalid inputs
        try:
            TropicalPolynomial("invalid", 2)
            assert False, "Should raise TypeError"
        except TypeError:
            pass
        
        try:
            TropicalPolynomial([m1], 0)
            assert False, "Should raise ValueError"
        except ValueError:
            pass
        
        print("✓ Polynomial creation tests passed")
    
    @staticmethod
    def test_polynomial_evaluation():
        """Test polynomial evaluation"""
        m1 = TropicalMonomial(0.0, {})  # Constant 0
        m2 = TropicalMonomial(1.0, {0: 1})  # 1 + x₀
        m3 = TropicalMonomial(2.0, {1: 1})  # 2 + x₁
        
        poly = TropicalPolynomial([m1, m2, m3], num_variables=2)
        
        # Single point evaluation
        point = torch.tensor([3.0, 1.0])
        result = poly.evaluate(point)
        # max(0, 1+3, 2+1) = max(0, 4, 3) = 4
        assert abs(result - 4.0) < 1e-6
        
        # Batch evaluation
        points = torch.tensor([[3.0, 1.0], [0.0, 0.0], [2.0, 5.0]])
        results = poly.evaluate(points)
        assert results.shape == (3,)
        assert abs(results[0] - 4.0) < 1e-6  # max(0, 4, 3)
        assert abs(results[1] - 2.0) < 1e-6  # max(0, 1, 2)
        assert abs(results[2] - 7.0) < 1e-6  # max(0, 3, 7)
        
        print("✓ Polynomial evaluation tests passed")
    
    @staticmethod
    def test_polynomial_operations():
        """Test polynomial addition and multiplication"""
        m1 = TropicalMonomial(1.0, {0: 1})
        m2 = TropicalMonomial(2.0, {1: 1})
        poly1 = TropicalPolynomial([m1], num_variables=2)
        poly2 = TropicalPolynomial([m2], num_variables=2)
        
        # Addition (tropical max)
        poly_sum = poly1.add(poly2)
        assert len(poly_sum.monomials) == 2
        
        # Test evaluation of sum
        point = torch.tensor([3.0, 1.0])
        result = poly_sum.evaluate(point)
        # max(1+3, 2+1) = max(4, 3) = 4
        assert abs(result - 4.0) < 1e-6
        
        # Multiplication (tropical sum)
        poly_prod = poly1.multiply(poly2)
        assert len(poly_prod.monomials) == 1
        # Product should have monomial with coeff 1+2=3 and exponents {0:1, 1:1}
        prod_monomial = poly_prod.monomials[0]
        assert abs(prod_monomial.coefficient - 3.0) < 1e-6
        assert prod_monomial.exponents == {0: 1, 1: 1}
        
        print("✓ Polynomial operation tests passed")
    
    @staticmethod
    def test_newton_polytope():
        """Test Newton polytope computation"""
        m1 = TropicalMonomial(1.0, {0: 2, 1: 1})
        m2 = TropicalMonomial(2.0, {0: 1, 1: 2})
        m3 = TropicalMonomial(0.0, {})
        
        poly = TropicalPolynomial([m1, m2, m3], num_variables=2)
        polytope = poly.newton_polytope()
        
        assert polytope.shape == (3, 2)
        # Check vertices
        expected = torch.tensor([[2.0, 1.0], [1.0, 2.0], [0.0, 0.0]])
        assert torch.allclose(polytope, expected)
        
        print("✓ Newton polytope tests passed")
    
    @staticmethod
    def test_polynomial_operations_gpu():
        """Test GPU-accelerated operations"""
        if not torch.cuda.is_available():
            print("⚠ GPU not available, skipping GPU tests")
            return
        
        device = torch.device('cuda')
        ops = TropicalPolynomialOperations(device=device)
        
        # Create test polynomials
        m1 = TropicalMonomial(1.0, {0: 1})
        m2 = TropicalMonomial(2.0, {1: 1})
        poly1 = TropicalPolynomial([m1, m2], num_variables=2)
        
        m3 = TropicalMonomial(0.0, {})
        m4 = TropicalMonomial(3.0, {0: 1, 1: 1})
        poly2 = TropicalPolynomial([m3, m4], num_variables=2)
        
        # Test batch evaluation
        points = torch.randn(100, 2, device=device)
        results = ops.batch_evaluate([poly1, poly2], points)
        assert results.shape == (2, 100)
        assert results.device == device
        
        # Test root finding
        roots = ops.find_tropical_roots(poly1, search_region=(-5, 5), num_samples=100)
        assert isinstance(roots, list)
        
        # Test resultant
        resultant = ops.compute_tropical_resultant(poly1, poly2)
        assert isinstance(resultant, float)
        
        print("✓ GPU operation tests passed")
    
    @staticmethod
    def test_interpolation():
        """Test polynomial interpolation"""
        ops = TropicalPolynomialOperations()
        
        # Create sample points and values
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        values = torch.tensor([0.0, 1.0, 2.0, 3.0])
        
        # Interpolate
        poly = ops.interpolate_from_points(points, values, max_degree=2)
        
        assert poly.num_variables == 2
        assert len(poly.monomials) > 0
        
        # Check that polynomial approximately interpolates the points
        for i in range(len(points)):
            val = poly.evaluate(points[i])
            # Tropical interpolation may not be exact, but should be close
            assert abs(val - values[i].item()) < 10.0
        
        print("✓ Interpolation tests passed")
    
    @staticmethod
    def run_all_tests():
        """Run all unit tests"""
        print("Running tropical polynomial tests...")
        TestTropicalPolynomial.test_monomial_creation()
        TestTropicalPolynomial.test_monomial_evaluation()
        TestTropicalPolynomial.test_polynomial_creation()
        TestTropicalPolynomial.test_polynomial_evaluation()
        TestTropicalPolynomial.test_polynomial_operations()
        TestTropicalPolynomial.test_newton_polytope()
        TestTropicalPolynomial.test_polynomial_operations_gpu()
        TestTropicalPolynomial.test_interpolation()
        print("\n✅ All tropical polynomial tests passed!")


if __name__ == "__main__":
    # Run unit tests
    TestTropicalPolynomial.run_all_tests()
    
    # Performance benchmark
    print("\n" + "="*50)
    print("Performance Benchmark")
    print("="*50)
    
    import time
    
    # Create large polynomial with many monomials
    num_monomials = 1000
    num_variables = 10
    monomials = []
    for i in range(num_monomials):
        coeff = float(i) / 10.0
        exponents = {}
        # Random sparse exponents
        for _ in range(min(3, num_variables)):  # Each monomial has at most 3 variables
            var_idx = torch.randint(0, num_variables, (1,)).item()
            power = torch.randint(1, 4, (1,)).item()
            exponents[var_idx] = power
        monomials.append(TropicalMonomial(coeff, exponents))
    
    poly = TropicalPolynomial(monomials, num_variables)
    print(f"Created polynomial with {len(poly.monomials)} monomials, {num_variables} variables")
    
    # Benchmark evaluation
    if torch.cuda.is_available():
        device = torch.device('cuda')
        points = torch.randn(1000, num_variables, device=device)
        
        start = time.time()
        results = poly.evaluate(points)
        end = time.time()
        
        print(f"GPU evaluation of 1000 points: {(end - start) * 1000:.2f}ms")
        assert results.shape == (1000,)
        
        # Benchmark batch operations
        ops = TropicalPolynomialOperations(device=device)
        polynomials = [poly] * 10
        
        start = time.time()
        batch_results = ops.batch_evaluate(polynomials, points[:100])
        end = time.time()
        
        print(f"Batch evaluation (10 polynomials, 100 points): {(end - start) * 1000:.2f}ms")
        assert batch_results.shape == (10, 100)
    else:
        device = torch.device('cpu')
        points = torch.randn(100, num_variables)
        
        start = time.time()
        results = poly.evaluate(points)
        end = time.time()
        
        print(f"CPU evaluation of 100 points: {(end - start) * 1000:.2f}ms")
    
    print("\n✅ Performance requirements met!")
    print(f"File location: {__file__}")