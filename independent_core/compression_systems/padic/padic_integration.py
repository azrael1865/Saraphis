"""
P-Adic Integration System for Independent Core
Integrates P-Adic compression with GAC, Brain Core, and training systems
NO FALLBACKS - HARD FAILURES ONLY
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import queue
import weakref
from contextlib import contextmanager
import gc

from ...gac_system.gac_components import GradientCompressionComponent, GACComponent
from ...brain import Brain as BrainCore, BrainSystemConfig as BrainConfig
from ..gpu_memory import GPUMemoryOptimizer
from .padic_compressor import PadicCompressionSystem
from .padic_advanced import (
    HenselLiftingProcessor, HierarchicalClusteringManager,
    PadicDecompressionEngine, PadicOptimizationManager,
    HenselLiftingConfig, ClusteringConfig, GPUDecompressionConfig
)
from .padic_service_config import PadicServiceConfiguration, PadicIntegrationConfig





class PadicGACIntegration:
    """Integrates P-Adic compression with GAC components"""
    
    def __init__(self, config: PadicIntegrationConfig, padic_system: PadicCompressionSystem, gpu_optimizer=None):
        if config is None:
            raise ValueError("Configuration cannot be None")
        if padic_system is None:
            raise ValueError("P-Adic system cannot be None")
            
        self.config = config
        self.padic_system = padic_system
        self.gpu_optimizer = gpu_optimizer
        
        # Component registry
        self.enhanced_components: Dict[str, GradientCompressionComponent] = {}
        self.component_stats: Dict[str, Dict[str, Any]] = {}
        
        # Async processing
        self.processing_queue = asyncio.Queue(maxsize=config.orchestrator_queue_size)
        self.result_futures: Dict[str, Future] = {}
        
        # Performance tracking
        self.gac_stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'average_compression_time': 0.0,
            'average_decompression_time': 0.0,
            'compression_ratio_achieved': 0.0,
            'gradients_processed': 0,
            'memory_saved_bytes': 0
        }
        
        self._lock = threading.RLock()
        self._shutdown = False
        
    def enhance_gradient_component(self, component: GradientCompressionComponent) -> None:
        """Enhance a GAC gradient compression component with P-Adic capabilities"""
        if component is None:
            raise ValueError("Component cannot be None")
        if not isinstance(component, GradientCompressionComponent):
            raise TypeError("Component must be GradientCompressionComponent")
            
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Integration is shut down")
                
            component_id = component.component_id
            if component_id in self.enhanced_components:
                raise ValueError(f"Component {component_id} already enhanced")
            
            # Inject P-Adic compression
            self._inject_padic_compression(component)
            
            # Register component
            self.enhanced_components[component_id] = component
            self.component_stats[component_id] = {
                'compressions': 0,
                'decompressions': 0,
                'total_time': 0.0,
                'errors': 0,
                'last_compression_ratio': 1.0
            }
    
    def _inject_padic_compression(self, component: GradientCompressionComponent) -> None:
        """Inject P-Adic compression into gradient component"""
        original_process = component.process_gradient
        original_compress = getattr(component, '_apply_compression', None)
        
        async def padic_process_gradient(gradient: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
            """Enhanced gradient processing with P-Adic compression"""
            if gradient is None:
                raise ValueError("Gradient cannot be None")
            if not isinstance(gradient, torch.Tensor):
                raise TypeError("Gradient must be torch.Tensor")
                
            start_time = time.perf_counter()
            
            try:
                # Optimize GPU memory if available
                if self.gpu_optimizer and gradient.device.type == 'cuda':
                    with self.gpu_optimizer.optimize_memory_context():
                        # Check if P-Adic compression should be used
                        if self._should_use_padic(gradient, context):
                            compressed = await self._async_compress_gradient(gradient, component.component_id)
                            result = await self._async_decompress_gradient(compressed, component.component_id)
                        else:
                            # Use original compression
                            result = await original_process(gradient, context)
                else:
                    # Check if P-Adic compression should be used
                    if self._should_use_padic(gradient, context):
                        compressed = await self._async_compress_gradient(gradient, component.component_id)
                        result = await self._async_decompress_gradient(compressed, component.component_id)
                    else:
                        # Use original compression
                        result = await original_process(gradient, context)
                
                # Update stats
                elapsed = time.perf_counter() - start_time
                self._update_component_stats(component.component_id, 'compression', elapsed)
                
                return result
                
            except Exception as e:
                self._update_component_stats(component.component_id, 'error', 0)
                raise RuntimeError(f"P-Adic gradient processing failed: {str(e)}") from e
        
        def padic_apply_compression(gradient: torch.Tensor) -> torch.Tensor:
            """Synchronous P-Adic compression"""
            if gradient is None:
                raise ValueError("Gradient cannot be None")
                
            try:
                compressed = self.padic_system.compress(gradient)
                result = self.padic_system.decompress(compressed)
                return result
            except Exception as e:
                raise RuntimeError(f"P-Adic compression failed: {str(e)}") from e
        
        # Replace methods
        component.process_gradient = padic_process_gradient
        component._apply_compression = padic_apply_compression
        component.padic_compression = self.padic_system
    
    def _should_use_padic(self, gradient: torch.Tensor, context: Dict[str, Any]) -> bool:
        """Determine if P-Adic compression should be used"""
        if gradient.numel() < 100:
            return False
            
        # Check gradient magnitude
        grad_norm = gradient.norm().item()
        if grad_norm < self.config.gac_compression_threshold:
            return False
            
        # Check context hints
        force_padic = context.get('force_padic', False)
        if force_padic:
            return True
            
        # Check memory pressure
        if torch.cuda.is_available():
            memory_free = torch.cuda.mem_get_info()[0]
            if memory_free < self.config.brain_memory_limit // 4:
                return True
                
        return True
    
    async def _async_compress_gradient(self, gradient: torch.Tensor, component_id: str) -> Dict[str, Any]:
        """Asynchronously compress gradient"""
        if self.config.gac_async_processing:
            future = Future()
            await self.processing_queue.put((gradient, component_id, 'compress', future))
            return await asyncio.wrap_future(future)
        else:
            return self.padic_system.compress(gradient)
    
    async def _async_decompress_gradient(self, compressed: Dict[str, Any], component_id: str) -> torch.Tensor:
        """Asynchronously decompress gradient"""
        if self.config.gac_async_processing:
            future = Future()
            await self.processing_queue.put((compressed, component_id, 'decompress', future))
            return await asyncio.wrap_future(future)
        else:
            return self.padic_system.decompress(compressed)
    
    def _update_component_stats(self, component_id: str, operation: str, elapsed: float) -> None:
        """Update component statistics"""
        with self._lock:
            if component_id not in self.component_stats:
                return
                
            stats = self.component_stats[component_id]
            
            if operation == 'compression':
                stats['compressions'] += 1
                stats['total_time'] += elapsed
                self.gac_stats['total_compressions'] += 1
                self.gac_stats['gradients_processed'] += 1
            elif operation == 'decompression':
                stats['decompressions'] += 1
                stats['total_time'] += elapsed
                self.gac_stats['total_decompressions'] += 1
            elif operation == 'error':
                stats['errors'] += 1
    
    def get_component_metrics(self, component_id: str) -> Dict[str, Any]:
        """Get metrics for a specific component"""
        with self._lock:
            if component_id not in self.component_stats:
                raise ValueError(f"Component {component_id} not found")
                
            stats = self.component_stats[component_id].copy()
            if stats['compressions'] > 0:
                stats['average_time'] = stats['total_time'] / stats['compressions']
            else:
                stats['average_time'] = 0.0
                
            return stats
    
    def shutdown(self) -> None:
        """Shutdown GAC integration"""
        with self._lock:
            self._shutdown = True
            self.enhanced_components.clear()
            self.component_stats.clear()


class PadicBrainIntegration:
    """Integrates P-Adic compression with Brain Core"""
    
    def __init__(self, config: PadicIntegrationConfig, padic_system: PadicCompressionSystem):
        if config is None:
            raise ValueError("Configuration cannot be None")
        if padic_system is None:
            raise ValueError("P-Adic system cannot be None")
            
        self.config = config
        self.padic_system = padic_system
        
        # Brain Core references
        self.brain_cores: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self.compression_handles: Dict[str, Any] = {}
        
        # Memory management
        self.memory_tracker = {
            'allocated': 0,
            'peak': 0,
            'compressions_triggered': 0,
            'memory_saved': 0
        }
        
        # Performance tracking
        self.brain_stats = {
            'registrations': 0,
            'compressions': 0,
            'decompressions': 0,
            'auto_optimizations': 0,
            'memory_pressure_events': 0
        }
        
        self._lock = threading.RLock()
        self._shutdown = False
    
    def register_with_brain(self, brain: BrainCore, system_name: str = "padic") -> None:
        """Register P-Adic system with Brain Core"""
        if brain is None:
            raise ValueError("Brain Core cannot be None")
        if not isinstance(brain, BrainCore):
            raise TypeError("Must be BrainCore instance")
            
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Integration is shut down")
                
            # Register compression system
            brain.register_compression_system(system_name, self.padic_system)
            
            # Set as active if configured
            if self.config.brain_auto_register:
                brain.active_compression = system_name
                
            # Register brain core
            brain_id = id(brain)
            self.brain_cores[brain_id] = brain
            
            # Setup memory monitoring
            self._setup_memory_monitoring(brain)
            
            # Setup P-Adic specific integration
            brain.padic_integration = self
            
            self.brain_stats['registrations'] += 1
    
    def _setup_memory_monitoring(self, brain: BrainCore) -> None:
        """Setup memory monitoring for Brain Core"""
        original_allocate = getattr(brain, '_allocate_memory', None)
        
        def monitored_allocate(size: int) -> Any:
            """Monitor memory allocations"""
            if size <= 0:
                raise ValueError("Size must be positive")
                
            with self._lock:
                # Check memory limit
                if self.memory_tracker['allocated'] + size > self.config.brain_memory_limit:
                    # Trigger compression
                    self._trigger_memory_compression(brain)
                    
                    # Re-check after compression
                    if self.memory_tracker['allocated'] + size > self.config.brain_memory_limit:
                        raise MemoryError(f"Memory limit exceeded: {size} bytes requested")
                
                # Allocate
                result = original_allocate(size) if original_allocate else None
                
                # Update tracking
                self.memory_tracker['allocated'] += size
                self.memory_tracker['peak'] = max(self.memory_tracker['peak'], 
                                                  self.memory_tracker['allocated'])
                
                return result
        
        if original_allocate:
            brain._allocate_memory = monitored_allocate
    
    def _trigger_memory_compression(self, brain: BrainCore) -> None:
        """Trigger compression due to memory pressure"""
        self.memory_tracker['compressions_triggered'] += 1
        self.brain_stats['memory_pressure_events'] += 1
        
        # Find compressible tensors in brain
        compressible = self._find_compressible_tensors(brain)
        
        for tensor_ref, tensor in compressible:
            if tensor.numel() > 1000:  # Only compress larger tensors
                try:
                    compressed = self.padic_system.compress(tensor)
                    saved = tensor.numel() * tensor.element_size() - self._estimate_compressed_size(compressed)
                    
                    if saved > 0:
                        # Store compressed version
                        self.compression_handles[tensor_ref] = compressed
                        self.memory_tracker['memory_saved'] += saved
                        self.memory_tracker['allocated'] -= saved
                        
                        # Clear original tensor
                        tensor.data = torch.empty(0)
                        
                except Exception as e:
                    # Hard failure on compression error
                    raise RuntimeError(f"Memory compression failed: {str(e)}") from e
    
    def _find_compressible_tensors(self, brain: BrainCore) -> List[Tuple[str, torch.Tensor]]:
        """Find tensors that can be compressed"""
        compressible = []
        
        # Check model parameters
        if hasattr(brain, 'model') and hasattr(brain.model, 'parameters'):
            for name, param in brain.model.named_parameters():
                if param.requires_grad and param.numel() > 1000:
                    compressible.append((f"param_{name}", param.data))
                    
        # Check buffers
        if hasattr(brain, 'buffers'):
            for name, buffer in brain.buffers.items():
                if isinstance(buffer, torch.Tensor) and buffer.numel() > 1000:
                    compressible.append((f"buffer_{name}", buffer))
                    
        return compressible
    
    def _estimate_compressed_size(self, compressed: Dict[str, Any]) -> int:
        """Estimate size of compressed data"""
        size = 0
        
        # Coefficients
        if 'coefficients' in compressed and isinstance(compressed['coefficients'], torch.Tensor):
            size += compressed['coefficients'].numel() * compressed['coefficients'].element_size()
            
        # Metadata
        size += 1000  # Rough estimate for metadata
        
        return size
    
    def optimize_brain_compression(self, brain: BrainCore) -> Dict[str, Any]:
        """Optimize compression settings for Brain Core"""
        if brain is None:
            raise ValueError("Brain Core cannot be None")
            
        brain_id = id(brain)
        if brain_id not in self.brain_cores:
            raise ValueError("Brain Core not registered")
            
        with self._lock:
            # Analyze current state
            analysis = self._analyze_brain_state(brain)
            
            # Optimize compression parameters
            optimizations = self._compute_optimizations(analysis)
            
            # Apply optimizations
            self._apply_optimizations(brain, optimizations)
            
            self.brain_stats['auto_optimizations'] += 1
            
            return {
                'analysis': analysis,
                'optimizations': optimizations,
                'memory_saved': self.memory_tracker['memory_saved'],
                'compression_ratio': self._calculate_compression_ratio()
            }
    
    def _analyze_brain_state(self, brain: BrainCore) -> Dict[str, Any]:
        """Analyze Brain Core state for optimization"""
        analysis = {
            'total_parameters': 0,
            'total_memory': 0,
            'compression_candidates': 0,
            'current_compression_ratio': 0.0
        }
        
        # Count parameters
        if hasattr(brain, 'model'):
            for param in brain.model.parameters():
                analysis['total_parameters'] += param.numel()
                analysis['total_memory'] += param.numel() * param.element_size()
                
        # Check existing compressions
        analysis['compression_candidates'] = len(self._find_compressible_tensors(brain))
        
        if analysis['total_memory'] > 0:
            analysis['current_compression_ratio'] = (
                self.memory_tracker['memory_saved'] / analysis['total_memory']
            )
            
        return analysis
    
    def _compute_optimizations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compute optimization parameters"""
        optimizations = {
            'recommended_precision': self.config.base_precision,
            'recommended_prime': self.config.prime,
            'enable_adaptive': True,
            'compression_threshold': 0.1
        }
        
        # Adjust precision based on compression ratio
        if analysis['current_compression_ratio'] < 0.3:
            optimizations['recommended_precision'] = max(
                self.config.precision_min,
                self.config.base_precision - self.config.precision_step
            )
        elif analysis['current_compression_ratio'] > 0.7:
            optimizations['recommended_precision'] = min(
                self.config.precision_max,
                self.config.base_precision + self.config.precision_step
            )
            
        return optimizations
    
    def _apply_optimizations(self, brain: BrainCore, optimizations: Dict[str, Any]) -> None:
        """Apply optimizations to Brain Core"""
        # Update P-Adic system configuration
        if 'recommended_precision' in optimizations:
            self.padic_system.precision = optimizations['recommended_precision']
            
        # Update compression threshold
        if 'compression_threshold' in optimizations:
            self.padic_system.config['compression_threshold'] = optimizations['compression_threshold']
            
    def _calculate_compression_ratio(self) -> float:
        """Calculate overall compression ratio"""
        if self.memory_tracker['allocated'] == 0:
            return 0.0
            
        return self.memory_tracker['memory_saved'] / (
            self.memory_tracker['allocated'] + self.memory_tracker['memory_saved']
        )
    
    def get_brain_metrics(self) -> Dict[str, Any]:
        """Get Brain Core integration metrics"""
        with self._lock:
            return {
                'stats': self.brain_stats.copy(),
                'memory': self.memory_tracker.copy(),
                'active_brains': len(self.brain_cores),
                'compression_ratio': self._calculate_compression_ratio()
            }
    
    def shutdown(self) -> None:
        """Shutdown Brain integration"""
        with self._lock:
            self._shutdown = True
            self.brain_cores.clear()
            self.compression_handles.clear()


class PadicTrainingIntegration:
    """Integrates P-Adic compression with training systems"""
    
    def __init__(self, config: PadicIntegrationConfig, 
                 padic_system: PadicCompressionSystem,
                 optimization_manager: PadicOptimizationManager):
        if config is None:
            raise ValueError("Configuration cannot be None")
        if padic_system is None:
            raise ValueError("P-Adic system cannot be None")
        if optimization_manager is None:
            raise ValueError("Optimization manager cannot be None")
            
        self.config = config
        self.padic_system = padic_system
        self.optimization_manager = optimization_manager
        
        # Training state
        self.training_step = 0
        self.is_training = False
        self.warmup_complete = False
        
        # Optimizer tracking
        self.registered_optimizers: Dict[str, Any] = {}
        self.optimizer_hooks: Dict[str, List[Callable]] = {}
        
        # Gradient history for adaptive compression
        self.gradient_history = {
            'magnitudes': [],
            'sparsities': [],
            'compression_ratios': []
        }
        
        # Performance tracking
        self.training_stats = {
            'total_steps': 0,
            'compressions_performed': 0,
            'decompressions_performed': 0,
            'average_compression_time': 0.0,
            'memory_saved_mb': 0.0,
            'precision_adjustments': 0
        }
        
        self._lock = threading.RLock()
        self._shutdown = False
    
    def register_optimizer(self, optimizer: torch.optim.Optimizer, name: str = "default") -> None:
        """Register optimizer for P-Adic integration"""
        if optimizer is None:
            raise ValueError("Optimizer cannot be None")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("Must be torch optimizer")
            
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Integration is shut down")
                
            if name in self.registered_optimizers:
                raise ValueError(f"Optimizer {name} already registered")
                
            # Register optimizer
            self.registered_optimizers[name] = optimizer
            self.optimizer_hooks[name] = []
            
            # Setup hooks
            self._setup_optimizer_hooks(optimizer, name)
            
            # Register with optimization manager
            self.optimization_manager.register_optimizer(name, optimizer)
    
    def _setup_optimizer_hooks(self, optimizer: torch.optim.Optimizer, name: str) -> None:
        """Setup hooks for optimizer integration"""
        # Pre-step hook
        def pre_step_hook(optimizer, *args, **kwargs):
            if not self.is_training:
                return
                
            self.training_step += 1
            
            # Check if we should compress this step
            if self._should_compress_step():
                self._compress_optimizer_state(optimizer, name)
                
        # Post-step hook
        def post_step_hook(optimizer, *args, **kwargs):
            if not self.is_training:
                return
                
            # Monitor gradients if configured
            if self.config.training_monitor_gradients:
                self._monitor_gradients(optimizer)
                
            # Check for adaptive adjustments
            if self.config.adaptive_precision and self.warmup_complete:
                self._check_adaptive_adjustments()
                
        # Register hooks
        self.optimizer_hooks[name].extend([pre_step_hook, post_step_hook])
        
        # Monkey-patch step method
        original_step = optimizer.step
        
        def hooked_step(closure=None):
            pre_step_hook(optimizer)
            result = original_step(closure)
            post_step_hook(optimizer)
            return result
            
        optimizer.step = hooked_step
    
    def _should_compress_step(self) -> bool:
        """Determine if compression should occur this step"""
        if self.training_step < self.config.training_warmup_steps:
            return False
            
        if not self.warmup_complete:
            self.warmup_complete = True
            
        return self.training_step % self.config.training_interval == 0
    
    def _compress_optimizer_state(self, optimizer: torch.optim.Optimizer, name: str) -> None:
        """Compress optimizer state"""
        start_time = time.perf_counter()
        memory_before = self._get_optimizer_memory(optimizer)
        
        compressed_states = {}
        
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                param_id = id(p)
                state = optimizer.state.get(p, {})
                
                # Compress momentum/velocity
                if 'momentum_buffer' in state:
                    compressed = self.padic_system.compress(state['momentum_buffer'])
                    compressed_states[f"{param_id}_momentum"] = compressed
                    state['momentum_buffer'] = self.padic_system.decompress(compressed)
                    
                # Compress Adam states
                if 'exp_avg' in state:
                    compressed = self.padic_system.compress(state['exp_avg'])
                    compressed_states[f"{param_id}_exp_avg"] = compressed
                    state['exp_avg'] = self.padic_system.decompress(compressed)
                    
                if 'exp_avg_sq' in state:
                    compressed = self.padic_system.compress(state['exp_avg_sq'])
                    compressed_states[f"{param_id}_exp_avg_sq"] = compressed
                    state['exp_avg_sq'] = self.padic_system.decompress(compressed)
                    
        # Update statistics
        elapsed = time.perf_counter() - start_time
        memory_after = self._get_optimizer_memory(optimizer)
        memory_saved = max(0, memory_before - memory_after) / (1024 * 1024)  # MB
        
        self.training_stats['compressions_performed'] += len(compressed_states)
        self.training_stats['memory_saved_mb'] += memory_saved
        self.training_stats['average_compression_time'] = (
            (self.training_stats['average_compression_time'] * 
             (self.training_stats['compressions_performed'] - len(compressed_states)) +
             elapsed * len(compressed_states)) / 
            self.training_stats['compressions_performed']
        )
    
    def _monitor_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """Monitor gradient statistics"""
        total_norm = 0.0
        total_sparsity = 0.0
        param_count = 0
        
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad = p.grad.data
                    total_norm += grad.norm().item() ** 2
                    total_sparsity += (grad.abs() < 1e-6).float().mean().item()
                    param_count += 1
                    
        if param_count > 0:
            avg_norm = (total_norm / param_count) ** 0.5
            avg_sparsity = total_sparsity / param_count
            
            # Update history
            self.gradient_history['magnitudes'].append(avg_norm)
            self.gradient_history['sparsities'].append(avg_sparsity)
            
            # Keep history bounded
            max_history = self.config.monitor_history_size
            if len(self.gradient_history['magnitudes']) > max_history:
                self.gradient_history['magnitudes'] = self.gradient_history['magnitudes'][-max_history:]
                self.gradient_history['sparsities'] = self.gradient_history['sparsities'][-max_history:]
    
    def _check_adaptive_adjustments(self) -> None:
        """Check if adaptive precision adjustments are needed"""
        if len(self.gradient_history['magnitudes']) < 100:
            return
            
        # Analyze recent gradient behavior
        recent_mags = self.gradient_history['magnitudes'][-100:]
        recent_sparsity = self.gradient_history['sparsities'][-100:]
        
        avg_magnitude = np.mean(recent_mags)
        avg_sparsity = np.mean(recent_sparsity)
        mag_variance = np.var(recent_mags)
        
        # Determine if adjustment is needed
        adjust_precision = False
        new_precision = self.padic_system.precision
        
        if avg_magnitude < 1e-4 and avg_sparsity > 0.9:
            # Very small, sparse gradients - can use lower precision
            new_precision = max(self.config.precision_min, 
                                new_precision - self.config.precision_step)
            adjust_precision = True
            
        elif avg_magnitude > 1.0 or mag_variance > 0.1:
            # Large or variable gradients - need higher precision
            new_precision = min(self.config.precision_max,
                                new_precision + self.config.precision_step)
            adjust_precision = True
            
        if adjust_precision and new_precision != self.padic_system.precision:
            self.padic_system.precision = new_precision
            self.training_stats['precision_adjustments'] += 1
    
    def _get_optimizer_memory(self, optimizer: torch.optim.Optimizer) -> int:
        """Estimate optimizer memory usage"""
        total_memory = 0
        
        for group in optimizer.param_groups:
            for p in group['params']:
                # Parameter memory
                total_memory += p.numel() * p.element_size()
                
                # State memory
                state = optimizer.state.get(p, {})
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        total_memory += value.numel() * value.element_size()
                        
        return total_memory
    
    def start_training(self) -> None:
        """Start training mode"""
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Integration is shut down")
            self.is_training = True
            self.training_stats['total_steps'] = self.training_step
    
    def stop_training(self) -> None:
        """Stop training mode"""
        with self._lock:
            self.is_training = False
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training integration metrics"""
        with self._lock:
            metrics = self.training_stats.copy()
            
            # Add gradient statistics
            if self.gradient_history['magnitudes']:
                metrics['recent_gradient_magnitude'] = np.mean(self.gradient_history['magnitudes'][-100:])
                metrics['recent_gradient_sparsity'] = np.mean(self.gradient_history['sparsities'][-100:])
            else:
                metrics['recent_gradient_magnitude'] = 0.0
                metrics['recent_gradient_sparsity'] = 0.0
                
            metrics['current_precision'] = self.padic_system.precision
            metrics['warmup_complete'] = self.warmup_complete
            
            return metrics
    
    def shutdown(self) -> None:
        """Shutdown training integration"""
        with self._lock:
            self._shutdown = True
            self.is_training = False
            self.registered_optimizers.clear()
            self.optimizer_hooks.clear()


class PadicSystemOrchestrator:
    """Orchestrates the overall P-Adic system integration"""
    
    def __init__(self, config: Optional[PadicIntegrationConfig] = None, 
                 service_config: Optional[PadicServiceConfiguration] = None,
                 gpu_optimizer: Optional[GPUMemoryOptimizer] = None):
        self.config = config or PadicIntegrationConfig()
        self.service_config = service_config or PadicServiceConfiguration()
        self.gpu_optimizer = gpu_optimizer
        
        # Register configuration change callback
        self.service_config.register_change_callback(self._on_config_change)
        
        # Initialize P-Adic components
        self._initialize_padic_components()
        
        # Initialize integration components
        self.gac_integration = PadicGACIntegration(self.config, self.padic_system, self.gpu_optimizer)
        self.brain_integration = PadicBrainIntegration(self.config, self.padic_system)
        self.training_integration = PadicTrainingIntegration(
            self.config, self.padic_system, self.optimization_manager
        )
        
        # Processing infrastructure
        self.executor = ThreadPoolExecutor(max_workers=self.config.orchestrator_threads)
        self.processing_queue = queue.Queue(maxsize=self.config.orchestrator_queue_size)
        
        # Service registry for external modules
        self.services = {
            'compress': self.compress,
            'decompress': self.decompress,
            'compress_async': self.compress_async,
            'decompress_async': self.decompress_async,
            'optimize': self.optimize_system,
            'get_metrics': self.get_system_metrics
        }
        
        # Performance monitoring
        self.performance_monitor = self._create_performance_monitor()
        
        # System state
        self._shutdown = False
        self._processing_thread = None
        self._monitor_thread = None
        
        # Start background threads
        self._start_background_threads()
    
    def _initialize_padic_components(self) -> None:
        """Initialize core P-Adic components"""
        # P-Adic compression system
        padic_config = {
            'prime': self.config.prime,
            'precision': self.config.base_precision,
            'gpu_acceleration': torch.cuda.is_available(),
            'adaptive_precision': self.config.adaptive_precision,
            'compression_threshold': self.config.gac_compression_threshold
        }
        self.padic_system = PadicCompressionSystem(padic_config)
        
        # Advanced components
        hensel_config = HenselLiftingConfig(
            max_iterations=100,
            tolerance=1e-10,
            adaptive_damping=True
        )
        self.hensel_processor = HenselLiftingProcessor(
            hensel_config, self.config.prime, self.config.base_precision
        )
        
        clustering_config = ClusteringConfig(
            max_depth=10,
            max_cluster_size=1000,
            distance_threshold=0.1
        )
        self.clustering_manager = HierarchicalClusteringManager(
            clustering_config, self.config.prime
        )
        
        gpu_config = GPUDecompressionConfig(
            batch_size=self.config.gac_batch_size,
            max_streams=4,
            memory_pool_size=1024 * 1024 * 1024  # 1GB
        )
        self.decompression_engine = PadicDecompressionEngine(
            gpu_config, self.config.prime
        )
        
        self.optimization_manager = PadicOptimizationManager(self.config.prime)
    
    def _create_performance_monitor(self) -> Dict[str, Any]:
        """Create performance monitoring structure"""
        return {
            'compression_times': [],
            'decompression_times': [],
            'queue_sizes': [],
            'memory_usage': [],
            'gpu_utilization': [],
            'timestamp': []
        }
    
    def _start_background_threads(self) -> None:
        """Start background processing threads"""
        # Processing thread
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self._processing_thread.start()
        
        # Monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitor_thread.start()
    
    def _processing_loop(self) -> None:
        """Main processing loop for async operations"""
        while not self._shutdown:
            try:
                # Get item from queue with timeout
                item = self.processing_queue.get(timeout=1.0)
                
                if item is None:  # Shutdown signal
                    break
                    
                data, operation, future = item
                
                try:
                    if operation == 'compress':
                        result = self.padic_system.compress(data)
                    elif operation == 'decompress':
                        result = self.padic_system.decompress(data)
                    else:
                        raise ValueError(f"Unknown operation: {operation}")
                        
                    future.set_result(result)
                    
                except Exception as e:
                    future.set_exception(e)
                    
            except queue.Empty:
                continue
            except Exception as e:
                # Log but continue processing
                print(f"Processing error: {str(e)}")
    
    def _monitoring_loop(self) -> None:
        """Monitoring loop for performance tracking"""
        while not self._shutdown:
            try:
                time.sleep(self.config.monitor_interval)
                
                # Collect metrics
                timestamp = time.time()
                
                # Queue size
                queue_size = self.processing_queue.qsize()
                self.performance_monitor['queue_sizes'].append(queue_size)
                
                # Memory usage
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                    self.performance_monitor['memory_usage'].append(memory_used)
                    
                    # GPU utilization
                    gpu_util = torch.cuda.utilization()
                    self.performance_monitor['gpu_utilization'].append(gpu_util)
                else:
                    self.performance_monitor['memory_usage'].append(0)
                    self.performance_monitor['gpu_utilization'].append(0)
                    
                self.performance_monitor['timestamp'].append(timestamp)
                
                # Trim old data
                max_size = self.config.monitor_history_size
                for key in self.performance_monitor:
                    if len(self.performance_monitor[key]) > max_size:
                        self.performance_monitor[key] = self.performance_monitor[key][-max_size:]
                        
            except Exception as e:
                print(f"Monitoring error: {str(e)}")
    
    def compress(self, data: torch.Tensor) -> Dict[str, Any]:
        """Synchronous compression"""
        if self._shutdown:
            raise RuntimeError("System is shut down")
        if data is None:
            raise ValueError("Data cannot be None")
            
        start_time = time.perf_counter()
        
        # Use GPU optimization if available
        if self.gpu_optimizer and data.device.type == 'cuda':
            with self.gpu_optimizer.optimize_memory_context():
                result = self.padic_system.compress(data)
        else:
            result = self.padic_system.compress(data)
            
        elapsed = time.perf_counter() - start_time
        self.performance_monitor['compression_times'].append(elapsed)
        
        return result
    
    def decompress(self, compressed: Dict[str, Any]) -> torch.Tensor:
        """Synchronous decompression"""
        if self._shutdown:
            raise RuntimeError("System is shut down")
        if compressed is None:
            raise ValueError("Compressed data cannot be None")
            
        start_time = time.perf_counter()
        result = self.padic_system.decompress(compressed)
        elapsed = time.perf_counter() - start_time
        
        self.performance_monitor['decompression_times'].append(elapsed)
        
        return result
    
    def compress_async(self, data: torch.Tensor) -> Future:
        """Asynchronous compression"""
        if self._shutdown:
            raise RuntimeError("System is shut down")
        if data is None:
            raise ValueError("Data cannot be None")
            
        future = Future()
        
        try:
            self.processing_queue.put((data, 'compress', future), 
                                      timeout=self.config.orchestrator_timeout)
        except queue.Full:
            future.set_exception(RuntimeError("Processing queue full"))
            
        return future
    
    def decompress_async(self, compressed: Dict[str, Any]) -> Future:
        """Asynchronous decompression"""
        if self._shutdown:
            raise RuntimeError("System is shut down")
        if compressed is None:
            raise ValueError("Compressed data cannot be None")
            
        future = Future()
        
        try:
            self.processing_queue.put((compressed, 'decompress', future),
                                      timeout=self.config.orchestrator_timeout)
        except queue.Full:
            future.set_exception(RuntimeError("Processing queue full"))
            
        return future
    
    def optimize_system(self) -> Dict[str, Any]:
        """Optimize the entire P-Adic system"""
        if self._shutdown:
            raise RuntimeError("System is shut down")
            
        optimizations = {
            'padic_system': {},
            'integrations': {},
            'performance': {}
        }
        
        # Analyze system performance
        if self.performance_monitor['compression_times']:
            avg_compression = np.mean(self.performance_monitor['compression_times'][-100:])
            if avg_compression > self.config.performance_threshold_ms / 1000:
                # Compression too slow - reduce precision
                new_precision = max(self.config.precision_min,
                                    self.padic_system.precision - self.config.precision_step)
                self.padic_system.precision = new_precision
                optimizations['padic_system']['precision_reduced'] = new_precision
                
        # Check memory pressure
        if self.performance_monitor['memory_usage']:
            recent_memory = np.mean(self.performance_monitor['memory_usage'][-10:])
            if recent_memory > self.config.brain_memory_limit / (1024 * 1024) * 0.8:
                # High memory usage - trigger compressions
                optimizations['integrations']['memory_compression_triggered'] = True
                
        # Optimize thread pool
        queue_sizes = self.performance_monitor['queue_sizes'][-100:]
        if queue_sizes and np.mean(queue_sizes) > self.config.orchestrator_queue_size * 0.5:
            # High queue usage - may need more threads
            optimizations['performance']['high_queue_usage'] = True
            
        return optimizations
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            'padic_system': self.padic_system.get_metrics(),
            'gac_integration': self.gac_integration.gac_stats,
            'brain_integration': self.brain_integration.get_brain_metrics(),
            'training_integration': self.training_integration.get_training_metrics(),
            'performance': {
                'avg_compression_time': np.mean(self.performance_monitor['compression_times'][-100:])
                if self.performance_monitor['compression_times'] else 0.0,
                'avg_decompression_time': np.mean(self.performance_monitor['decompression_times'][-100:])
                if self.performance_monitor['decompression_times'] else 0.0,
                'current_queue_size': self.processing_queue.qsize(),
                'executor_active': len(self.executor._threads)
            }
        }
    
    def get_service(self, service_name: str) -> Callable:
        """Get service interface for external modules"""
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")
            
        return self.services[service_name]
    
    @contextmanager
    def batch_operations(self):
        """Context manager for batch operations"""
        # Increase batch size temporarily
        original_batch = self.config.gac_batch_size
        self.config.gac_batch_size *= 4
        
        try:
            yield self
        finally:
            self.config.gac_batch_size = original_batch
    
    def _on_config_change(self, key: str, old_value: Any, new_value: Any) -> None:
        """Handle configuration changes"""
        try:
            logger.info(f"Configuration changed: {key} = {new_value} (was {old_value})")
            
            # Handle specific configuration changes
            if key.startswith('padic.'):
                self._handle_padic_config_change(key, new_value)
            elif key.startswith('service.'):
                self._handle_service_config_change(key, new_value)
            elif key.startswith('integration.'):
                self._handle_integration_config_change(key, new_value)
            elif key.startswith('monitoring.'):
                self._handle_monitoring_config_change(key, new_value)
                
        except Exception as e:
            logger.error(f"Failed to handle configuration change {key}: {str(e)}")
    
    def _handle_padic_config_change(self, key: str, value: Any) -> None:
        """Handle P-adic specific configuration changes"""
        if key == 'padic.default_precision':
            # Update precision in existing components
            if hasattr(self, 'padic_system'):
                self.padic_system.precision = value
        elif key == 'padic.compression_ratio':
            # Update compression settings
            if hasattr(self, 'padic_system'):
                self.padic_system.compression_ratio = value
        elif key.startswith('padic.gpu_decompression.'):
            # Update GPU settings
            self._update_gpu_settings()
    
    def _handle_service_config_change(self, key: str, value: Any) -> None:
        """Handle service configuration changes"""
        if key == 'service.max_batch_size':
            # Update batch sizes across components
            self.config.gac_batch_size = value
        elif key == 'service.max_concurrent_requests':
            # Potentially resize thread pool (requires restart)
            logger.warning(f"Changed {key} requires system restart to take effect")
    
    def _handle_integration_config_change(self, key: str, value: Any) -> None:
        """Handle integration configuration changes"""
        if key == 'integration.brain_core_enabled':
            if value and not hasattr(self, 'brain_integration'):
                # Initialize brain integration if enabled
                self.brain_integration = PadicBrainIntegration(self.config, self.padic_system)
            elif not value and hasattr(self, 'brain_integration'):
                # Shutdown brain integration if disabled
                self.brain_integration.shutdown()
                delattr(self, 'brain_integration')
        elif key == 'integration.gpu_memory_enabled':
            self._update_gpu_settings()
    
    def _handle_monitoring_config_change(self, key: str, value: Any) -> None:
        """Handle monitoring configuration changes"""
        if key == 'monitoring.health_check_interval':
            # Update monitoring intervals
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.update_interval(value)
    
    def _update_gpu_settings(self) -> None:
        """Update GPU settings from current configuration"""
        if hasattr(self, 'gpu_decompression_engine'):
            gpu_config = self.service_config.get_gpu_config()
            self.gpu_decompression_engine.update_config(gpu_config)
    
    def get_service_configuration(self) -> PadicServiceConfiguration:
        """Get the current service configuration"""
        return self.service_config
    
    def update_service_configuration(self, updates: Dict[str, Any]) -> bool:
        """Update service configuration"""
        return self.service_config.update(updates)
    
    def shutdown(self) -> None:
        """Shutdown the orchestrator"""
        self._shutdown = True
        
        # Signal processing thread
        self.processing_queue.put(None)
        
        # Wait for threads
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            
        # Shutdown components
        self.gac_integration.shutdown()
        self.brain_integration.shutdown()
        self.training_integration.shutdown()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Clear GPU resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        gc.collect()


# Module-level instance for easy access
_orchestrator: Optional[PadicSystemOrchestrator] = None


def initialize_padic_integration(config: Optional[PadicIntegrationConfig] = None,
                                service_config: Optional[PadicServiceConfiguration] = None,
                                gpu_optimizer: Optional[GPUMemoryOptimizer] = None) -> PadicSystemOrchestrator:
    """Initialize the P-Adic integration system"""
    global _orchestrator
    
    if _orchestrator is not None:
        raise RuntimeError("P-Adic integration already initialized")
        
    _orchestrator = PadicSystemOrchestrator(config, service_config, gpu_optimizer)
    return _orchestrator


def get_orchestrator() -> PadicSystemOrchestrator:
    """Get the current orchestrator instance"""
    if _orchestrator is None:
        raise RuntimeError("P-Adic integration not initialized")
        
    return _orchestrator


def shutdown_padic_integration() -> None:
    """Shutdown the P-Adic integration system"""
    global _orchestrator
    
    if _orchestrator is not None:
        _orchestrator.shutdown()
        _orchestrator = None