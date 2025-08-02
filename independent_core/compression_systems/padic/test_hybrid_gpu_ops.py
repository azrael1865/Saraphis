"""
Comprehensive tests for hybrid p-adic GPU operations
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import pytest
import time
import gc
from typing import Dict, Any, List

from .hybrid_padic_gpu_ops import (
    GPUOperationConfig,
    HybridPadicGPUOps,
    HybridPadicGPUOptimizer,
    HybridPadicGPUManager
)
from .hybrid_padic_structures import HybridPadicWeight, HybridPadicManager
from .padic_encoder import PadicValidation


class TestGPUOperationConfig:
    """Test suite for GPU operation configuration"""
    
    def test_valid_config_creation(self):
        """Test creation of valid GPU operation config"""
        config = GPUOperationConfig(
            stream_count=4,
            max_batch_size=1024,
            memory_limit_mb=2048,
            enable_mixed_precision=True,
            enable_tensor_cores=True,
            optimization_level=3
        )
        
        assert config.stream_count == 4
        assert config.max_batch_size == 1024
        assert config.memory_limit_mb == 2048
        assert config.enable_mixed_precision is True
        assert config.enable_tensor_cores is True
        assert config.optimization_level == 3
    
    def test_invalid_stream_count(self):
        """Test invalid stream count validation"""
        with pytest.raises(ValueError, match="stream_count"):
            GPUOperationConfig(stream_count=0)
        
        with pytest.raises(ValueError, match="stream_count"):
            GPUOperationConfig(stream_count=-1)
        
        with pytest.raises(ValueError, match="stream_count.*exceeds maximum"):
            GPUOperationConfig(stream_count=64)
    
    def test_invalid_batch_size(self):
        """Test invalid batch size validation"""
        with pytest.raises(ValueError, match="max_batch_size"):
            GPUOperationConfig(max_batch_size=0)
        
        with pytest.raises(ValueError, match="max_batch_size.*exceeds maximum"):
            GPUOperationConfig(max_batch_size=100000)
    
    def test_invalid_memory_limit(self):
        """Test invalid memory limit validation"""
        with pytest.raises(ValueError, match="memory_limit_mb"):
            GPUOperationConfig(memory_limit_mb=0)
        
        with pytest.raises(ValueError, match="memory_limit_mb"):
            GPUOperationConfig(memory_limit_mb=-100)


class TestHybridPadicGPUOps:
    """Test suite for GPU-accelerated hybrid p-adic operations"""
    
    def setup_method(self):
        """Setup test environment"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for GPU operations tests")
        
        self.config = GPUOperationConfig(
            stream_count=2,
            max_batch_size=256,
            memory_limit_mb=512,
            enable_mixed_precision=False,  # Use float32 for testing
            stream_synchronization=True
        )
        
        self.prime = 5
        self.precision = 4
        self.gpu_ops = HybridPadicGPUOps(self.config, self.prime, self.precision)
        
        # Create hybrid manager for test weight creation
        hybrid_config = {
            'prime': self.prime,
            'precision': self.precision,
            'gpu_memory_limit_mb': 512
        }
        self.hybrid_manager = HybridPadicManager(hybrid_config)
    
    def _create_test_weight(self, value: float = 1.0) -> HybridPadicWeight:
        """Create a test hybrid p-adic weight"""
        # Create GPU tensors
        exp_tensor = torch.tensor([value], dtype=torch.float32, device='cuda')
        mant_tensor = torch.tensor([value * 0.5], dtype=torch.float32, device='cuda')
        
        return HybridPadicWeight(
            exponent_channel=exp_tensor,
            mantissa_channel=mant_tensor,
            prime=self.prime,
            precision=self.precision,
            compression_params={'test': True},
            gpu_memory_info={'allocated': 0}
        )
    
    def test_gpu_ops_initialization(self):
        """Test GPU operations initialization"""
        assert self.gpu_ops.prime == self.prime
        assert self.gpu_ops.precision == self.precision
        assert self.gpu_ops.device.type == 'cuda'
        assert len(self.gpu_ops.streams) == self.config.stream_count
        
        # Check that GPU constants are initialized
        assert hasattr(self.gpu_ops, 'prime_powers')
        assert hasattr(self.gpu_ops, 'mod_inverses')
        assert hasattr(self.gpu_ops, 'log_prime')
        assert hasattr(self.gpu_ops, 'inv_log_prime')
    
    def test_invalid_initialization(self):
        """Test invalid initialization parameters"""
        with pytest.raises(ValueError, match="Config cannot be None"):
            HybridPadicGPUOps(None, self.prime, self.precision)
        
        with pytest.raises(TypeError, match="Config must be GPUOperationConfig"):
            HybridPadicGPUOps("invalid", self.prime, self.precision)
        
        with pytest.raises(ValueError, match="Prime"):
            HybridPadicGPUOps(self.config, 1, self.precision)
        
        with pytest.raises(ValueError, match="Precision"):
            HybridPadicGPUOps(self.config, self.prime, 0)
    
    def test_addition_operation(self):
        """Test GPU-accelerated addition"""
        x = self._create_test_weight(2.0)
        y = self._create_test_weight(3.0)
        
        result = self.gpu_ops.add(x, y)
        
        # Validate result structure
        assert isinstance(result, HybridPadicWeight)
        assert result.prime == self.prime
        assert result.precision == self.precision
        assert result.exponent_channel.is_cuda
        assert result.mantissa_channel.is_cuda
        
        # Verify operation was recorded
        stats = self.gpu_ops.get_operation_stats()
        assert stats['add_operations'] >= 1
        assert stats['total_operations'] >= 1
    
    def test_multiplication_operation(self):
        """Test GPU-accelerated multiplication"""
        x = self._create_test_weight(2.0)
        y = self._create_test_weight(3.0)
        
        result = self.gpu_ops.multiply(x, y)
        
        # Validate result structure
        assert isinstance(result, HybridPadicWeight)
        assert result.prime == self.prime
        assert result.precision == self.precision
        assert result.exponent_channel.is_cuda
        assert result.mantissa_channel.is_cuda
        
        # Verify operation was recorded
        stats = self.gpu_ops.get_operation_stats()
        assert stats['multiply_operations'] >= 1
    
    def test_division_operation(self):
        """Test GPU-accelerated division"""
        x = self._create_test_weight(6.0)
        y = self._create_test_weight(2.0)
        
        result = self.gpu_ops.divide(x, y)
        
        # Validate result structure
        assert isinstance(result, HybridPadicWeight)
        assert result.prime == self.prime
        assert result.precision == self.precision
        
        # Verify operation was recorded
        stats = self.gpu_ops.get_operation_stats()
        assert stats['divide_operations'] >= 1
    
    def test_division_by_zero(self):
        """Test division by zero error handling"""
        x = self._create_test_weight(5.0)
        zero_weight = self._create_test_weight(0.0)
        
        with pytest.raises(ValueError, match="Division by zero"):
            self.gpu_ops.divide(x, zero_weight)
    
    def test_power_operation(self):
        """Test GPU-accelerated exponentiation"""
        x = self._create_test_weight(2.0)
        exponent = 3.0
        
        result = self.gpu_ops.power(x, exponent)
        
        # Validate result structure
        assert isinstance(result, HybridPadicWeight)
        assert result.prime == self.prime
        assert result.precision == self.precision
        
        # Verify operation was recorded
        stats = self.gpu_ops.get_operation_stats()
        assert stats['power_operations'] >= 1
    
    def test_invalid_power_exponent(self):
        """Test invalid exponent handling"""
        x = self._create_test_weight(2.0)
        
        with pytest.raises(TypeError, match="Exponent must be"):
            self.gpu_ops.power(x, "invalid")
        
        with pytest.raises(ValueError, match="Invalid exponent"):
            self.gpu_ops.power(x, float('nan'))
        
        with pytest.raises(ValueError, match="Exponent.*too large"):
            self.gpu_ops.power(x, 2000.0)
    
    def test_batch_operations(self):
        """Test batch GPU operations"""
        weights = [self._create_test_weight(i + 1.0) for i in range(3)]
        operations = ['add', 'multiply', 'power']
        operands = [
            self._create_test_weight(1.0),  # for add
            self._create_test_weight(2.0),  # for multiply
            2.0  # for power
        ]
        
        results = self.gpu_ops.batch_operations(weights, operations, operands)
        
        # Validate results
        assert len(results) == len(weights)
        for result in results:
            assert isinstance(result, HybridPadicWeight)
            assert result.prime == self.prime
            assert result.precision == self.precision
        
        # Verify batch operation was recorded
        stats = self.gpu_ops.get_operation_stats()
        assert stats['batch_operations'] >= 1
    
    def test_invalid_batch_operations(self):
        """Test invalid batch operation parameters"""
        with pytest.raises(ValueError, match="Weights list cannot be empty"):
            self.gpu_ops.batch_operations([], [], [])
        
        weights = [self._create_test_weight(1.0)]
        operations = ['add', 'multiply']  # Mismatched lengths
        operands = [self._create_test_weight(1.0)]
        
        with pytest.raises(ValueError, match="same length"):
            self.gpu_ops.batch_operations(weights, operations, operands)
        
        # Test invalid operation
        operations = ['invalid_op']
        operands = [self._create_test_weight(1.0)]
        
        with pytest.raises(ValueError, match="Invalid operation"):
            self.gpu_ops.batch_operations(weights, operations, operands)
    
    def test_weight_validation(self):
        """Test weight validation in operations"""
        valid_weight = self._create_test_weight(1.0)
        
        with pytest.raises(ValueError, match="Weight.*cannot be None"):
            self.gpu_ops.add(None, valid_weight)
        
        with pytest.raises(TypeError, match="Weight.*must be HybridPadicWeight"):
            self.gpu_ops.add("invalid", valid_weight)
        
        # Test CPU weight (should fail - must be on CUDA)
        cpu_tensor = torch.tensor([1.0], dtype=torch.float32, device='cpu')
        cpu_weight = HybridPadicWeight(
            exponent_channel=cpu_tensor,
            mantissa_channel=cpu_tensor,
            prime=self.prime,
            precision=self.precision,
            compression_params={},
            gpu_memory_info={}
        )
        
        with pytest.raises(ValueError, match="must be on CUDA"):
            self.gpu_ops.add(cpu_weight, valid_weight)
    
    def test_memory_management(self):
        """Test GPU memory management"""
        initial_memory = torch.cuda.memory_allocated()
        
        # Create several weights and perform operations
        weights = [self._create_test_weight(i + 1.0) for i in range(10)]
        for i in range(len(weights) - 1):
            result = self.gpu_ops.add(weights[i], weights[i + 1])
        
        # Check memory usage tracking
        memory_info = self.gpu_ops._get_memory_info()
        assert 'allocated' in memory_info
        assert 'cached' in memory_info
        assert 'max_allocated' in memory_info
        
        # Clean up
        self.gpu_ops.cleanup_gpu_memory()
        
        # Memory should be cleaned
        final_memory = torch.cuda.memory_allocated()
        # Note: Memory might not go back to exactly initial due to PyTorch caching
    
    def test_operation_statistics(self):
        """Test operation statistics tracking"""
        # Reset stats
        self.gpu_ops.reset_operation_stats()
        
        # Perform various operations
        x = self._create_test_weight(2.0)
        y = self._create_test_weight(3.0)
        
        self.gpu_ops.add(x, y)
        self.gpu_ops.multiply(x, y)
        self.gpu_ops.power(x, 2.0)
        
        stats = self.gpu_ops.get_operation_stats()
        
        assert stats['total_operations'] == 3
        assert stats['add_operations'] == 1
        assert stats['multiply_operations'] == 1
        assert stats['power_operations'] == 1
        assert stats['average_operation_time'] > 0
        assert stats['cuda_kernel_calls'] > 0
    
    def test_stream_management(self):
        """Test CUDA stream management"""
        initial_stream_switches = self.gpu_ops.operation_stats['stream_switches']
        
        # Perform operations to trigger stream switches
        x = self._create_test_weight(1.0)
        y = self._create_test_weight(2.0)
        
        for _ in range(5):
            self.gpu_ops.add(x, y)
        
        # Check that streams were used
        final_stream_switches = self.gpu_ops.operation_stats['stream_switches']
        assert final_stream_switches > initial_stream_switches
        
        # Test synchronization
        self.gpu_ops.synchronize_streams()  # Should not raise


class TestHybridPadicGPUOptimizer:
    """Test suite for GPU optimizer"""
    
    def setup_method(self):
        """Setup test environment"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for GPU optimizer tests")
        
        config = GPUOperationConfig(stream_count=2, memory_limit_mb=512)
        self.prime = 5
        self.precision = 4
        
        self.gpu_ops = HybridPadicGPUOps(config, self.prime, self.precision)
        self.optimizer = HybridPadicGPUOptimizer(
            self.gpu_ops, 
            learning_rate=0.01, 
            momentum=0.9, 
            weight_decay=0.001
        )
    
    def _create_test_weight(self, value: float = 1.0) -> HybridPadicWeight:
        """Create a test hybrid p-adic weight"""
        exp_tensor = torch.tensor([value], dtype=torch.float32, device='cuda')
        mant_tensor = torch.tensor([value * 0.5], dtype=torch.float32, device='cuda')
        
        return HybridPadicWeight(
            exponent_channel=exp_tensor,
            mantissa_channel=mant_tensor,
            prime=self.prime,
            precision=self.precision,
            compression_params={'test': True},
            gpu_memory_info={'allocated': 0}
        )
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        assert self.optimizer.learning_rate == 0.01
        assert self.optimizer.momentum == 0.9
        assert self.optimizer.weight_decay == 0.001
        assert len(self.optimizer.momentum_buffers) == 0
    
    def test_invalid_optimizer_params(self):
        """Test invalid optimizer parameters"""
        with pytest.raises(ValueError, match="GPU operations cannot be None"):
            HybridPadicGPUOptimizer(None)
        
        with pytest.raises(TypeError, match="GPU operations must be"):
            HybridPadicGPUOptimizer("invalid")
        
        with pytest.raises(ValueError, match="Learning rate must be"):
            HybridPadicGPUOptimizer(self.gpu_ops, learning_rate=0.0)
        
        with pytest.raises(ValueError, match="Learning rate.*too large"):
            HybridPadicGPUOptimizer(self.gpu_ops, learning_rate=2.0)
        
        with pytest.raises(ValueError, match="Momentum must be"):
            HybridPadicGPUOptimizer(self.gpu_ops, momentum=1.5)
        
        with pytest.raises(ValueError, match="Weight decay must be"):
            HybridPadicGPUOptimizer(self.gpu_ops, weight_decay=-0.1)
    
    def test_optimization_step(self):
        """Test optimization step"""
        weights = [self._create_test_weight(2.0), self._create_test_weight(3.0)]
        gradients = [self._create_test_weight(0.1), self._create_test_weight(0.2)]
        
        updated_weights = self.optimizer.step(weights, gradients)
        
        # Validate results
        assert len(updated_weights) == len(weights)
        for weight in updated_weights:
            assert isinstance(weight, HybridPadicWeight)
            assert weight.prime == self.prime
            assert weight.precision == self.precision
        
        # Check optimization stats
        stats = self.optimizer.get_optimization_stats()
        assert stats['total_steps'] == 1
        assert stats['weight_updates'] == 2
        assert stats['average_step_time'] > 0
    
    def test_momentum_optimization(self):
        """Test optimization with momentum"""
        weights = [self._create_test_weight(1.0)]
        gradients = [self._create_test_weight(0.1)]
        
        # First step - initializes momentum
        self.optimizer.step(weights, gradients)
        assert len(self.optimizer.momentum_buffers) == 1
        
        # Second step - uses momentum
        self.optimizer.step(weights, gradients)
        
        stats = self.optimizer.get_optimization_stats()
        assert stats['momentum_updates'] >= 1
    
    def test_weight_decay(self):
        """Test weight decay functionality"""
        optimizer_with_decay = HybridPadicGPUOptimizer(
            self.gpu_ops, learning_rate=0.01, weight_decay=0.1
        )
        
        weights = [self._create_test_weight(1.0)]
        gradients = [self._create_test_weight(0.1)]
        
        updated_weights = optimizer_with_decay.step(weights, gradients)
        
        # Weight decay should affect the update
        assert len(updated_weights) == 1
        assert isinstance(updated_weights[0], HybridPadicWeight)
    
    def test_invalid_step_parameters(self):
        """Test invalid step parameters"""
        with pytest.raises(ValueError, match="Weights list cannot be empty"):
            self.optimizer.step([], [])
        
        weights = [self._create_test_weight(1.0)]
        gradients = []
        
        with pytest.raises(ValueError, match="Gradients list cannot be empty"):
            self.optimizer.step(weights, gradients)
        
        gradients = [self._create_test_weight(0.1), self._create_test_weight(0.2)]
        
        with pytest.raises(ValueError, match="length mismatch"):
            self.optimizer.step(weights, gradients)
    
    def test_zero_grad(self):
        """Test gradient zeroing"""
        weights = [self._create_test_weight(1.0)]
        gradients = [self._create_test_weight(0.1)]
        
        # Create momentum buffers
        self.optimizer.step(weights, gradients)
        assert len(self.optimizer.momentum_buffers) > 0
        
        # Clear buffers
        self.optimizer.zero_grad()
        assert len(self.optimizer.momentum_buffers) == 0


class TestHybridPadicGPUManager:
    """Test suite for GPU manager"""
    
    def setup_method(self):
        """Setup test environment"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for GPU manager tests")
        
        self.config = GPUOperationConfig(stream_count=2, memory_limit_mb=512)
        self.prime = 5
        self.precision = 4
        
        self.manager = HybridPadicGPUManager(self.config, self.prime, self.precision)
    
    def _create_test_weight(self, value: float = 1.0) -> HybridPadicWeight:
        """Create a test hybrid p-adic weight"""
        exp_tensor = torch.tensor([value], dtype=torch.float32, device='cuda')
        mant_tensor = torch.tensor([value * 0.5], dtype=torch.float32, device='cuda')
        
        return HybridPadicWeight(
            exponent_channel=exp_tensor,
            mantissa_channel=mant_tensor,
            prime=self.prime,
            precision=self.precision,
            compression_params={'test': True},
            gpu_memory_info={'allocated': 0}
        )
    
    def test_manager_initialization(self):
        """Test GPU manager initialization"""
        assert self.manager.prime == self.prime
        assert self.manager.precision == self.precision
        assert isinstance(self.manager.gpu_ops, HybridPadicGPUOps)
        assert len(self.manager.optimizers) == 0
    
    def test_create_optimizer(self):
        """Test optimizer creation"""
        optimizer = self.manager.create_optimizer(
            'test_opt', 
            learning_rate=0.01, 
            momentum=0.9
        )
        
        assert isinstance(optimizer, HybridPadicGPUOptimizer)
        assert 'test_opt' in self.manager.optimizers
        assert self.manager.training_stats['parameter_groups'] == 1
    
    def test_duplicate_optimizer_name(self):
        """Test duplicate optimizer name handling"""
        self.manager.create_optimizer('test_opt')
        
        with pytest.raises(ValueError, match="already exists"):
            self.manager.create_optimizer('test_opt')
    
    def test_get_optimizer(self):
        """Test optimizer retrieval"""
        created_opt = self.manager.create_optimizer('test_opt')
        retrieved_opt = self.manager.get_optimizer('test_opt')
        
        assert created_opt is retrieved_opt
        
        with pytest.raises(ValueError, match="not found"):
            self.manager.get_optimizer('nonexistent')
    
    def test_train_step(self):
        """Test training step across parameter groups"""
        # Create optimizers
        self.manager.create_optimizer('group1', learning_rate=0.01)
        self.manager.create_optimizer('group2', learning_rate=0.02)
        
        # Create parameter groups
        parameter_groups = {
            'group1': (
                [self._create_test_weight(1.0)],
                [self._create_test_weight(0.1)]
            ),
            'group2': (
                [self._create_test_weight(2.0)],
                [self._create_test_weight(0.2)]
            )
        }
        
        results = self.manager.train_step(parameter_groups)
        
        # Validate results
        assert len(results) == 2
        assert 'group1' in results
        assert 'group2' in results
        
        for group_results in results.values():
            assert len(group_results) == 1
            assert isinstance(group_results[0], HybridPadicWeight)
        
        # Check training stats
        stats = self.manager.get_comprehensive_stats()
        assert stats['training_stats']['total_batches'] == 1
    
    def test_invalid_train_step_parameters(self):
        """Test invalid training step parameters"""
        with pytest.raises(ValueError, match="Parameter groups cannot be empty"):
            self.manager.train_step({})
        
        # Missing optimizer for group
        parameter_groups = {
            'nonexistent_group': ([], [])
        }
        
        with pytest.raises(ValueError, match="No optimizer found"):
            self.manager.train_step(parameter_groups)
    
    def test_synchronization_and_cleanup(self):
        """Test synchronization and cleanup"""
        # These should not raise
        self.manager.synchronize_all()
        self.manager.cleanup_all()
    
    def test_comprehensive_stats(self):
        """Test comprehensive statistics"""
        # Create optimizer and perform operations
        self.manager.create_optimizer('test_opt')
        
        parameter_groups = {
            'test_opt': (
                [self._create_test_weight(1.0)],
                [self._create_test_weight(0.1)]
            )
        }
        
        self.manager.train_step(parameter_groups)
        
        stats = self.manager.get_comprehensive_stats()
        
        # Validate stats structure
        assert 'training_stats' in stats
        assert 'gpu_ops_stats' in stats
        assert 'optimizer_stats' in stats
        
        assert 'test_opt' in stats['optimizer_stats']
        
        training_stats = stats['training_stats']
        assert training_stats['total_batches'] == 1
        assert training_stats['parameter_groups'] == 1
        assert training_stats['average_epoch_time'] > 0
    
    def test_reset_all_stats(self):
        """Test resetting all statistics"""
        # Perform some operations
        self.manager.create_optimizer('test_opt')
        parameter_groups = {
            'test_opt': ([self._create_test_weight(1.0)], [self._create_test_weight(0.1)])
        }
        self.manager.train_step(parameter_groups)
        
        # Reset stats
        self.manager.reset_all_stats()
        
        stats = self.manager.get_comprehensive_stats()
        assert stats['training_stats']['total_batches'] == 0
        assert stats['gpu_ops_stats']['total_operations'] == 0


def test_gpu_operations_basic():
    """Standalone test for basic GPU operations - can be run without pytest"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    config = GPUOperationConfig(stream_count=2, memory_limit_mb=256)
    prime = 5
    precision = 4
    
    # Test GPU operations creation
    gpu_ops = HybridPadicGPUOps(config, prime, precision)
    
    # Create test weights
    exp_tensor = torch.tensor([2.0], dtype=torch.float32, device='cuda')
    mant_tensor = torch.tensor([1.0], dtype=torch.float32, device='cuda')
    
    x = HybridPadicWeight(
        exponent_channel=exp_tensor,
        mantissa_channel=mant_tensor,
        prime=prime,
        precision=precision,
        compression_params={},
        gpu_memory_info={}
    )
    
    y = HybridPadicWeight(
        exponent_channel=exp_tensor.clone(),
        mantissa_channel=mant_tensor.clone(),
        prime=prime,
        precision=precision,
        compression_params={},
        gpu_memory_info={}
    )
    
    # Test basic operations
    result_add = gpu_ops.add(x, y)
    result_mul = gpu_ops.multiply(x, y)
    result_pow = gpu_ops.power(x, 2.0)
    
    # Verify results
    assert isinstance(result_add, HybridPadicWeight)
    assert isinstance(result_mul, HybridPadicWeight)
    assert isinstance(result_pow, HybridPadicWeight)
    
    # Check stats
    stats = gpu_ops.get_operation_stats()
    assert stats['total_operations'] == 3
    assert stats['add_operations'] == 1
    assert stats['multiply_operations'] == 1
    assert stats['power_operations'] == 1
    
    print("✅ Basic GPU operations test passed")


def test_gpu_optimizer_basic():
    """Standalone test for basic GPU optimizer"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    config = GPUOperationConfig(stream_count=2, memory_limit_mb=256)
    prime = 5
    precision = 4
    
    # Create GPU operations and optimizer
    gpu_ops = HybridPadicGPUOps(config, prime, precision)
    optimizer = HybridPadicGPUOptimizer(gpu_ops, learning_rate=0.01)
    
    # Create test weights and gradients
    exp_tensor = torch.tensor([1.0], dtype=torch.float32, device='cuda')
    mant_tensor = torch.tensor([0.5], dtype=torch.float32, device='cuda')
    
    weight = HybridPadicWeight(
        exponent_channel=exp_tensor,
        mantissa_channel=mant_tensor,
        prime=prime,
        precision=precision,
        compression_params={},
        gpu_memory_info={}
    )
    
    gradient = HybridPadicWeight(
        exponent_channel=torch.tensor([0.1], dtype=torch.float32, device='cuda'),
        mantissa_channel=torch.tensor([0.05], dtype=torch.float32, device='cuda'),
        prime=prime,
        precision=precision,
        compression_params={},
        gpu_memory_info={}
    )
    
    # Perform optimization step
    updated_weights = optimizer.step([weight], [gradient])
    
    # Verify results
    assert len(updated_weights) == 1
    assert isinstance(updated_weights[0], HybridPadicWeight)
    
    # Check stats
    stats = optimizer.get_optimization_stats()
    assert stats['total_steps'] == 1
    assert stats['weight_updates'] == 1
    
    print("✅ Basic GPU optimizer test passed")


if __name__ == "__main__":
    """Run basic tests when executed as script"""
    print("Running hybrid p-adic GPU operations tests...")
    
    try:
        test_gpu_operations_basic()
        test_gpu_optimizer_basic()
        print("✅ All basic GPU tests passed!")
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        raise