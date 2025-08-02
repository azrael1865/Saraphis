"""
Neural Orchestrator - Neural Network Coordination System
Manages neural network operations, training coordination, and model orchestration
"""

import asyncio
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock, Lock, Event
import threading
from collections import defaultdict, deque
import json
import traceback
from abc import ABC, abstractmethod
import weakref

logger = logging.getLogger(__name__)

class NetworkType(Enum):
    FEEDFORWARD = "feedforward"
    CONVOLUTIONAL = "convolutional"
    RECURRENT = "recurrent"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    AUTOENCODER = "autoencoder"
    GAN = "gan"
    VAE = "variational_autoencoder"

class TrainingPhase(Enum):
    INITIALIZATION = "initialization"
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    FINE_TUNING = "fine_tuning"
    INFERENCE = "inference"
    COMPLETED = "completed"
    FAILED = "failed"

class OptimizationStrategy(Enum):
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    MOMENTUM = "momentum"
    ADAPTIVE = "adaptive"

class SchedulingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    PRIORITY_BASED = "priority_based"
    RESOURCE_AWARE = "resource_aware"
    PERFORMANCE_BASED = "performance_based"
    ADAPTIVE = "adaptive"

@dataclass
class NetworkConfig:
    network_id: str
    network_type: NetworkType
    architecture: Dict[str, Any]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    hidden_layers: List[int] = field(default_factory=list)
    activation_function: str = "relu"
    dropout_rate: float = 0.1
    batch_norm: bool = True
    device: str = "auto"
    precision: str = "float32"
    initialization: str = "xavier"

@dataclass
class TrainingConfig:
    training_id: str
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    optimizer: OptimizationStrategy = OptimizationStrategy.ADAM
    loss_function: str = "mse"
    metrics: List[str] = field(default_factory=lambda: ["loss", "accuracy"])
    validation_split: float = 0.2
    early_stopping: bool = True
    patience: int = 10
    checkpoint_frequency: int = 10
    regularization: Dict[str, float] = field(default_factory=dict)
    scheduling: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelState:
    model_id: str
    network_config: NetworkConfig
    training_config: Optional[TrainingConfig] = None
    phase: TrainingPhase = TrainingPhase.INITIALIZATION
    epoch: int = 0
    iteration: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    best_metrics: Dict[str, float] = field(default_factory=dict)
    convergence_status: str = "training"
    resource_usage: Dict[str, float] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NeuralTask:
    task_id: str
    model_id: str
    operation: str  # "train", "inference", "evaluate", "fine_tune"
    priority: int = 1  # 1=low, 5=high
    data: Optional[Any] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 3600.0  # 1 hour default
    retry_count: int = 3
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None

class NeuralModel(nn.Module):
    """Base neural model class with orchestration support"""
    
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config
        self.model_id = config.network_id
        self.network_type = config.network_type
        self._build_network()
        self._initialize_weights()
    
    def _build_network(self):
        """Build the neural network architecture"""
        if self.network_type == NetworkType.FEEDFORWARD:
            self._build_feedforward()
        elif self.network_type == NetworkType.CONVOLUTIONAL:
            self._build_convolutional()
        elif self.network_type == NetworkType.LSTM:
            self._build_lstm()
        elif self.network_type == NetworkType.TRANSFORMER:
            self._build_transformer()
        else:
            self._build_custom()
    
    def _build_feedforward(self):
        """Build feedforward network"""
        layers = []
        input_size = np.prod(self.config.input_shape)
        
        # Hidden layers
        prev_size = input_size
        for hidden_size in self.config.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if self.config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            if self.config.activation_function == "relu":
                layers.append(nn.ReLU())
            elif self.config.activation_function == "tanh":
                layers.append(nn.Tanh())
            elif self.config.activation_function == "sigmoid":
                layers.append(nn.Sigmoid())
            
            if self.config.dropout_rate > 0:
                layers.append(nn.Dropout(self.config.dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        output_size = np.prod(self.config.output_shape)
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def _build_convolutional(self):
        """Build convolutional network"""
        # Simplified CNN implementation
        conv_layers = []
        
        # Assume input is image-like
        in_channels = self.config.input_shape[0] if len(self.config.input_shape) > 2 else 1
        
        # Conv layers
        for i, out_channels in enumerate([32, 64, 128]):
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.config.input_shape)
            conv_output = self.conv_layers(dummy_input)
            flattened_size = conv_output.numel()
        
        # Fully connected layers
        fc_layers = []
        prev_size = flattened_size
        for hidden_size in self.config.hidden_layers:
            fc_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(self.config.dropout_rate)
            ])
            prev_size = hidden_size
        
        output_size = np.prod(self.config.output_shape)
        fc_layers.append(nn.Linear(prev_size, output_size))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        self.flatten = nn.Flatten()
    
    def _build_lstm(self):
        """Build LSTM network"""
        input_size = self.config.input_shape[-1]
        hidden_size = self.config.hidden_layers[0] if self.config.hidden_layers else 128
        num_layers = len(self.config.hidden_layers) if self.config.hidden_layers else 2
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.config.dropout_rate if num_layers > 1 else 0
        )
        
        output_size = np.prod(self.config.output_shape)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        if self.config.dropout_rate > 0:
            self.dropout = nn.Dropout(self.config.dropout_rate)
    
    def _build_transformer(self):
        """Build transformer network"""
        # Simplified transformer implementation
        d_model = self.config.hidden_layers[0] if self.config.hidden_layers else 512
        nhead = self.config.architecture.get('nhead', 8)
        num_layers = len(self.config.hidden_layers) if self.config.hidden_layers else 6
        
        self.embedding = nn.Linear(self.config.input_shape[-1], d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=self.config.dropout_rate
        )
        
        output_size = np.prod(self.config.output_shape)
        self.output_layer = nn.Linear(d_model, output_size)
    
    def _build_custom(self):
        """Build custom network based on architecture config"""
        # Fallback to simple feedforward
        self._build_feedforward()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.initialization == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                elif self.config.initialization == "kaiming":
                    nn.init.kaiming_uniform_(module.weight)
                elif self.config.initialization == "normal":
                    nn.init.normal_(module.weight, 0, 0.01)
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, mode='fan_out')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass"""
        if self.network_type == NetworkType.FEEDFORWARD:
            x = x.view(x.size(0), -1)  # Flatten
            return self.network(x)
        
        elif self.network_type == NetworkType.CONVOLUTIONAL:
            x = self.conv_layers(x)
            x = self.flatten(x)
            return self.fc_layers(x)
        
        elif self.network_type == NetworkType.LSTM:
            lstm_out, _ = self.lstm(x)
            # Use last output
            last_output = lstm_out[:, -1, :]
            if hasattr(self, 'dropout'):
                last_output = self.dropout(last_output)
            return self.output_layer(last_output)
        
        elif self.network_type == NetworkType.TRANSFORMER:
            x = self.embedding(x)
            x = self.transformer(x, x)  # Self-attention
            # Use mean pooling
            x = x.mean(dim=1)
            return self.output_layer(x)
        
        else:
            return self.network(x)

class TrainingCoordinator:
    """Coordinates training processes for multiple models"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.active_trainings: Dict[str, Dict[str, Any]] = {}
        self.training_lock = Lock()
        
    def start_training(self, model_id: str, training_config: TrainingConfig, data_loader: Any) -> bool:
        """Start training for a model"""
        try:
            with self.training_lock:
                if model_id in self.active_trainings:
                    logger.warning(f"Training already active for model {model_id}")
                    return False
                
                # Get model
                model = self.orchestrator.get_model(model_id)
                if not model:
                    logger.error(f"Model {model_id} not found")
                    return False
                
                # Setup training state
                training_state = {
                    'model': model,
                    'config': training_config,
                    'data_loader': data_loader,
                    'optimizer': self._create_optimizer(model, training_config),
                    'loss_function': self._create_loss_function(training_config),
                    'scheduler': self._create_scheduler(training_config),
                    'epoch': 0,
                    'best_loss': float('inf'),
                    'patience_counter': 0,
                    'metrics_history': defaultdict(list)
                }
                
                self.active_trainings[model_id] = training_state
                
                # Update model state
                model_state = self.orchestrator.get_model_state(model_id)
                model_state.phase = TrainingPhase.TRAINING
                model_state.training_config = training_config
                
                logger.info(f"Training started for model {model_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start training for model {model_id}: {e}")
            return False
    
    def training_step(self, model_id: str) -> Dict[str, Any]:
        """Perform one training step"""
        if model_id not in self.active_trainings:
            return {"error": "Training not active"}
        
        training_state = self.active_trainings[model_id]
        model = training_state['model']
        config = training_state['config']
        optimizer = training_state['optimizer']
        loss_function = training_state['loss_function']
        data_loader = training_state['data_loader']
        
        try:
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(data_loader):
                # Move to device
                device = next(model.parameters()).device
                data, target = data.to(device), target.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(data)
                loss = loss_function(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update iteration count
                model_state = self.orchestrator.get_model_state(model_id)
                model_state.iteration += 1
            
            # Calculate average loss
            avg_loss = epoch_loss / max(1, num_batches)
            
            # Update epoch
            training_state['epoch'] += 1
            model_state = self.orchestrator.get_model_state(model_id)
            model_state.epoch = training_state['epoch']
            model_state.metrics['loss'] = avg_loss
            
            # Store metrics
            training_state['metrics_history']['loss'].append(avg_loss)
            
            # Early stopping check
            if config.early_stopping:
                if avg_loss < training_state['best_loss']:
                    training_state['best_loss'] = avg_loss
                    training_state['patience_counter'] = 0
                    model_state.best_metrics['loss'] = avg_loss
                else:
                    training_state['patience_counter'] += 1
                
                if training_state['patience_counter'] >= config.patience:
                    model_state.convergence_status = "converged"
                    model_state.phase = TrainingPhase.COMPLETED
                    return {"status": "converged", "loss": avg_loss, "epoch": training_state['epoch']}
            
            # Learning rate scheduling
            if training_state['scheduler']:
                training_state['scheduler'].step()
            
            # Check if training complete
            if training_state['epoch'] >= config.epochs:
                model_state.phase = TrainingPhase.COMPLETED
                model_state.convergence_status = "completed"
                return {"status": "completed", "loss": avg_loss, "epoch": training_state['epoch']}
            
            return {"status": "training", "loss": avg_loss, "epoch": training_state['epoch']}
            
        except Exception as e:
            logger.error(f"Training step failed for model {model_id}: {e}")
            model_state = self.orchestrator.get_model_state(model_id)
            model_state.phase = TrainingPhase.FAILED
            return {"error": str(e)}
    
    def _create_optimizer(self, model: nn.Module, config: TrainingConfig) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        if config.optimizer == OptimizationStrategy.ADAM:
            return optim.Adam(model.parameters(), lr=config.learning_rate)
        elif config.optimizer == OptimizationStrategy.ADAMW:
            return optim.AdamW(model.parameters(), lr=config.learning_rate)
        elif config.optimizer == OptimizationStrategy.SGD:
            return optim.SGD(model.parameters(), lr=config.learning_rate)
        elif config.optimizer == OptimizationStrategy.RMSPROP:
            return optim.RMSprop(model.parameters(), lr=config.learning_rate)
        else:
            return optim.Adam(model.parameters(), lr=config.learning_rate)
    
    def _create_loss_function(self, config: TrainingConfig) -> nn.Module:
        """Create loss function based on configuration"""
        if config.loss_function == "mse":
            return nn.MSELoss()
        elif config.loss_function == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif config.loss_function == "bce":
            return nn.BCELoss()
        elif config.loss_function == "l1":
            return nn.L1Loss()
        else:
            return nn.MSELoss()
    
    def _create_scheduler(self, config: TrainingConfig) -> Optional[Any]:
        """Create learning rate scheduler"""
        if 'scheduler' not in config.scheduling:
            return None
        
        scheduler_type = config.scheduling['scheduler']
        if scheduler_type == "step":
            step_size = config.scheduling.get('step_size', 10)
            gamma = config.scheduling.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(None, step_size=step_size, gamma=gamma)  # Will set optimizer later
        elif scheduler_type == "exponential":
            gamma = config.scheduling.get('gamma', 0.95)
            return optim.lr_scheduler.ExponentialLR(None, gamma=gamma)
        
        return None

class InferenceEngine:
    """Handles model inference operations"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.inference_cache: Dict[str, Any] = {}
        self.cache_lock = Lock()
    
    def run_inference(self, model_id: str, input_data: torch.Tensor, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """Run inference on a model"""
        try:
            model = self.orchestrator.get_model(model_id)
            if not model:
                return {"error": f"Model {model_id} not found"}
            
            model.eval()
            device = next(model.parameters()).device
            
            # Move input to device
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data)
            input_data = input_data.to(device)
            
            # Run inference
            with torch.no_grad():
                if batch_size and input_data.size(0) > batch_size:
                    # Batch processing
                    outputs = []
                    for i in range(0, input_data.size(0), batch_size):
                        batch = input_data[i:i+batch_size]
                        batch_output = model(batch)
                        outputs.append(batch_output)
                    output = torch.cat(outputs, dim=0)
                else:
                    output = model(input_data)
            
            # Update model state
            model_state = self.orchestrator.get_model_state(model_id)
            model_state.phase = TrainingPhase.INFERENCE
            model_state.last_updated = time.time()
            
            return {
                "output": output.cpu().numpy(),
                "shape": output.shape,
                "device": str(device),
                "inference_time": time.time() - model_state.last_updated
            }
            
        except Exception as e:
            logger.error(f"Inference failed for model {model_id}: {e}")
            return {"error": str(e)}
    
    def batch_inference(self, model_id: str, data_loader: Any) -> Dict[str, Any]:
        """Run batch inference"""
        try:
            model = self.orchestrator.get_model(model_id)
            if not model:
                return {"error": f"Model {model_id} not found"}
            
            model.eval()
            device = next(model.parameters()).device
            
            all_outputs = []
            all_targets = []
            total_time = 0.0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(data_loader):
                    start_time = time.time()
                    
                    data = data.to(device)
                    output = model(data)
                    
                    all_outputs.append(output.cpu())
                    if target is not None:
                        all_targets.append(target)
                    
                    total_time += time.time() - start_time
            
            # Concatenate results
            final_output = torch.cat(all_outputs, dim=0)
            final_targets = torch.cat(all_targets, dim=0) if all_targets else None
            
            return {
                "outputs": final_output.numpy(),
                "targets": final_targets.numpy() if final_targets is not None else None,
                "total_time": total_time,
                "batches_processed": len(all_outputs),
                "throughput": len(all_outputs) / max(total_time, 1e-6)
            }
            
        except Exception as e:
            logger.error(f"Batch inference failed for model {model_id}: {e}")
            return {"error": str(e)}

class NeuralOrchestrator:
    def __init__(self, brain_instance=None, config: Optional[Dict] = None):
        self.brain = brain_instance
        self.config = config or {}
        
        # Core state management
        self._lock = RLock()
        self._models: Dict[str, NeuralModel] = {}
        self._model_states: Dict[str, ModelState] = {}
        self._task_queue: deque = deque()
        self._active_tasks: Dict[str, NeuralTask] = {}
        self._completed_tasks: Dict[str, NeuralTask] = {}
        
        # Coordinators and engines
        self.training_coordinator = TrainingCoordinator(self)
        self.inference_engine = InferenceEngine(self)
        
        # Resource management
        self._device_manager = self._initialize_device_manager()
        self._memory_monitor = self._initialize_memory_monitor()
        self._resource_lock = Lock()
        
        # Scheduling
        self._scheduler_strategy = SchedulingStrategy(
            self.config.get('scheduler_strategy', 'resource_aware')
        )
        self._scheduler_thread = None
        self._scheduler_running = False
        
        # Performance tracking
        self._performance_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._model_performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Model registry
        self._model_registry: Dict[str, Dict[str, Any]] = {}
        
        # GPU optimization integration
        self._gpu_optimizer = None
        if hasattr(self.brain, 'gpu_memory_optimizer'):
            self._gpu_optimizer = self.brain.gpu_memory_optimizer
        
        logger.info("NeuralOrchestrator initialized")
    
    def coordinate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Main coordination entry point"""
        operation = parameters.get('operation', 'train')
        
        if operation == 'create_model':
            return self._create_model(parameters)
        elif operation == 'train':
            return self._coordinate_training(parameters)
        elif operation == 'inference':
            return self._coordinate_inference(parameters)
        elif operation == 'evaluate':
            return self._coordinate_evaluation(parameters)
        elif operation == 'fine_tune':
            return self._coordinate_fine_tuning(parameters)
        elif operation == 'schedule_task':
            return self._schedule_task(parameters)
        elif operation == 'get_model_status':
            return self._get_model_status(parameters)
        elif operation == 'optimize_resources':
            return self._optimize_resources(parameters)
        elif operation == 'get_performance_report':
            return self._get_performance_report(parameters)
        else:
            logger.warning(f"Unknown operation: {operation}")
            return {"error": f"Unknown operation: {operation}"}
    
    def _create_model(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new neural model"""
        try:
            # Parse model configuration
            model_id = parameters.get('model_id', f"model_{int(time.time())}")
            network_type = NetworkType(parameters.get('network_type', 'feedforward'))
            architecture = parameters.get('architecture', {})
            input_shape = tuple(parameters.get('input_shape', (784,)))
            output_shape = tuple(parameters.get('output_shape', (10,)))
            
            # Create network configuration
            network_config = NetworkConfig(
                network_id=model_id,
                network_type=network_type,
                architecture=architecture,
                input_shape=input_shape,
                output_shape=output_shape,
                hidden_layers=parameters.get('hidden_layers', [128, 64]),
                activation_function=parameters.get('activation', 'relu'),
                dropout_rate=parameters.get('dropout_rate', 0.1),
                batch_norm=parameters.get('batch_norm', True),
                device=parameters.get('device', 'auto'),
                initialization=parameters.get('initialization', 'xavier')
            )
            
            # Create model
            model = NeuralModel(network_config)
            
            # Determine device
            device = self._select_device(network_config.device)
            model = model.to(device)
            
            # Create model state
            model_state = ModelState(
                model_id=model_id,
                network_config=network_config,
                phase=TrainingPhase.INITIALIZATION
            )
            
            # Store model and state
            with self._lock:
                self._models[model_id] = model
                self._model_states[model_id] = model_state
                self._model_registry[model_id] = {
                    'created_at': time.time(),
                    'network_type': network_type.value,
                    'parameters': sum(p.numel() for p in model.parameters()),
                    'device': str(device)
                }
            
            # GPU memory optimization
            if self._gpu_optimizer and device.type == 'cuda':
                self._gpu_optimizer.prepare_for_computation(
                    computation_type='model_creation',
                    model_id=model_id,
                    memory_estimate=self._estimate_model_memory(model)
                )
            
            logger.info(f"Model {model_id} created successfully on {device}")
            
            return {
                "model_id": model_id,
                "status": "created",
                "network_type": network_type.value,
                "parameters": self._model_registry[model_id]['parameters'],
                "device": str(device),
                "memory_usage": self._get_model_memory_usage(model_id)
            }
            
        except Exception as e:
            logger.error(f"Model creation failed: {e}")
            traceback.print_exc()
            return {"error": str(e)}
    
    def _coordinate_training(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate model training"""
        try:
            model_id = parameters.get('model_id')
            if not model_id or model_id not in self._models:
                return {"error": "Model not found"}
            
            # Create training configuration
            training_config = TrainingConfig(
                training_id=f"train_{model_id}_{int(time.time())}",
                batch_size=parameters.get('batch_size', 32),
                learning_rate=parameters.get('learning_rate', 0.001),
                epochs=parameters.get('epochs', 100),
                optimizer=OptimizationStrategy(parameters.get('optimizer', 'adam')),
                loss_function=parameters.get('loss_function', 'mse'),
                metrics=parameters.get('metrics', ['loss', 'accuracy']),
                validation_split=parameters.get('validation_split', 0.2),
                early_stopping=parameters.get('early_stopping', True),
                patience=parameters.get('patience', 10),
                checkpoint_frequency=parameters.get('checkpoint_frequency', 10),
                regularization=parameters.get('regularization', {}),
                scheduling=parameters.get('scheduling', {})
            )
            
            # Get or create data loader
            data_loader = parameters.get('data_loader')
            if not data_loader:
                # Create dummy data loader for demonstration
                data_loader = self._create_dummy_data_loader(model_id, training_config.batch_size)
            
            # GPU optimization for training
            if self._gpu_optimizer:
                model = self._models[model_id]
                device = next(model.parameters()).device
                if device.type == 'cuda':
                    self._gpu_optimizer.prepare_for_training(
                        model_id=model_id,
                        batch_size=training_config.batch_size,
                        sequence_length=getattr(training_config, 'sequence_length', 100)
                    )
            
            # Start training
            training_started = self.training_coordinator.start_training(
                model_id, training_config, data_loader
            )
            
            if not training_started:
                return {"error": "Failed to start training"}
            
            # Create training task
            task = NeuralTask(
                task_id=f"train_task_{int(time.time())}",
                model_id=model_id,
                operation="train",
                priority=parameters.get('priority', 3),
                parameters={'training_config': training_config}
            )
            
            # Schedule task
            self._schedule_neural_task(task)
            
            return {
                "training_id": training_config.training_id,
                "task_id": task.task_id,
                "status": "started",
                "epochs": training_config.epochs,
                "batch_size": training_config.batch_size,
                "optimizer": training_config.optimizer.value
            }
            
        except Exception as e:
            logger.error(f"Training coordination failed: {e}")
            return {"error": str(e)}
    
    def _coordinate_inference(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate model inference"""
        try:
            model_id = parameters.get('model_id')
            if not model_id or model_id not in self._models:
                return {"error": "Model not found"}
            
            input_data = parameters.get('input_data')
            if input_data is None:
                return {"error": "Input data required"}
            
            batch_size = parameters.get('batch_size')
            
            # GPU optimization for inference
            if self._gpu_optimizer:
                model = self._models[model_id]
                device = next(model.parameters()).device
                if device.type == 'cuda':
                    self._gpu_optimizer.optimize_for_operation(
                        operation_type='inference',
                        model_id=model_id,
                        batch_size=batch_size or 1
                    )
            
            # Run inference
            result = self.inference_engine.run_inference(model_id, input_data, batch_size)
            
            if 'error' in result:
                return result
            
            # Update performance metrics
            self._update_inference_metrics(model_id, result)
            
            return {
                "model_id": model_id,
                "inference_result": result,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Inference coordination failed: {e}")
            return {"error": str(e)}
    
    def _coordinate_evaluation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate model evaluation"""
        try:
            model_id = parameters.get('model_id')
            if not model_id or model_id not in self._models:
                return {"error": "Model not found"}
            
            test_data_loader = parameters.get('test_data_loader')
            if not test_data_loader:
                # Create dummy test data
                test_data_loader = self._create_dummy_data_loader(model_id, 32)
            
            # Run batch inference for evaluation
            result = self.inference_engine.batch_inference(model_id, test_data_loader)
            
            if 'error' in result:
                return result
            
            # Calculate evaluation metrics
            evaluation_metrics = self._calculate_evaluation_metrics(result)
            
            # Update model state
            model_state = self._model_states[model_id]
            model_state.phase = TrainingPhase.TESTING
            model_state.metrics.update(evaluation_metrics)
            
            return {
                "model_id": model_id,
                "evaluation_metrics": evaluation_metrics,
                "throughput": result.get('throughput', 0),
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Evaluation coordination failed: {e}")
            return {"error": str(e)}
    
    def _coordinate_fine_tuning(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate model fine-tuning"""
        try:
            model_id = parameters.get('model_id')
            if not model_id or model_id not in self._models:
                return {"error": "Model not found"}
            
            # Create fine-tuning configuration (similar to training but with lower learning rate)
            training_config = TrainingConfig(
                training_id=f"finetune_{model_id}_{int(time.time())}",
                batch_size=parameters.get('batch_size', 16),  # Smaller batch size
                learning_rate=parameters.get('learning_rate', 0.0001),  # Lower learning rate
                epochs=parameters.get('epochs', 10),  # Fewer epochs
                optimizer=OptimizationStrategy(parameters.get('optimizer', 'adam')),
                loss_function=parameters.get('loss_function', 'mse'),
                early_stopping=True,
                patience=5
            )
            
            # Update model state
            model_state = self._model_states[model_id]
            model_state.phase = TrainingPhase.FINE_TUNING
            
            # Start fine-tuning (similar to training)
            data_loader = parameters.get('data_loader') or self._create_dummy_data_loader(model_id, training_config.batch_size)
            
            training_started = self.training_coordinator.start_training(
                model_id, training_config, data_loader
            )
            
            if not training_started:
                return {"error": "Failed to start fine-tuning"}
            
            return {
                "training_id": training_config.training_id,
                "status": "started",
                "phase": "fine_tuning",
                "epochs": training_config.epochs
            }
            
        except Exception as e:
            logger.error(f"Fine-tuning coordination failed: {e}")
            return {"error": str(e)}
    
    def _schedule_task(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Schedule a neural task"""
        try:
            task = NeuralTask(
                task_id=parameters.get('task_id', f"task_{int(time.time())}"),
                model_id=parameters.get('model_id', ''),
                operation=parameters.get('operation', 'inference'),
                priority=parameters.get('priority', 1),
                parameters=parameters.get('task_parameters', {}),
                dependencies=parameters.get('dependencies', []),
                timeout=parameters.get('timeout', 3600.0)
            )
            
            self._schedule_neural_task(task)
            
            return {
                "task_id": task.task_id,
                "status": "scheduled",
                "priority": task.priority,
                "queue_position": len(self._task_queue)
            }
            
        except Exception as e:
            logger.error(f"Task scheduling failed: {e}")
            return {"error": str(e)}
    
    def _schedule_neural_task(self, task: NeuralTask):
        """Schedule a neural task based on strategy"""
        with self._lock:
            if self._scheduler_strategy == SchedulingStrategy.PRIORITY_BASED:
                # Insert based on priority
                inserted = False
                for i, existing_task in enumerate(self._task_queue):
                    if task.priority > existing_task.priority:
                        self._task_queue.insert(i, task)
                        inserted = True
                        break
                if not inserted:
                    self._task_queue.append(task)
            
            elif self._scheduler_strategy == SchedulingStrategy.RESOURCE_AWARE:
                # Consider resource availability
                self._task_queue.append(task)  # Simplified for now
            
            else:  # ROUND_ROBIN or default
                self._task_queue.append(task)
            
            # Start scheduler if not running
            if not self._scheduler_running:
                self._start_scheduler()
    
    def _start_scheduler(self):
        """Start the task scheduler thread"""
        if self._scheduler_running:
            return
        
        self._scheduler_running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="neural_scheduler",
            daemon=True
        )
        self._scheduler_thread.start()
        logger.debug("Neural task scheduler started")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self._scheduler_running:
            try:
                # Process next task
                task = None
                with self._lock:
                    if self._task_queue:
                        task = self._task_queue.popleft()
                
                if task:
                    self._execute_task(task)
                else:
                    time.sleep(0.1)  # Short sleep when no tasks
                    
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(1)
    
    def _execute_task(self, task: NeuralTask):
        """Execute a neural task"""
        try:
            task.started_at = time.time()
            task.status = "running"
            
            with self._lock:
                self._active_tasks[task.task_id] = task
            
            # Route task based on operation
            if task.operation == "train":
                result = self._execute_training_task(task)
            elif task.operation == "inference":
                result = self._execute_inference_task(task)
            elif task.operation == "evaluate":
                result = self._execute_evaluation_task(task)
            else:
                result = {"error": f"Unknown operation: {task.operation}"}
            
            # Complete task
            task.completed_at = time.time()
            task.result = result
            task.status = "completed" if "error" not in result else "failed"
            
            with self._lock:
                if task.task_id in self._active_tasks:
                    del self._active_tasks[task.task_id]
                self._completed_tasks[task.task_id] = task
            
            logger.debug(f"Task {task.task_id} completed with status: {task.status}")
            
        except Exception as e:
            task.error = str(e)
            task.status = "failed"
            task.completed_at = time.time()
            logger.error(f"Task {task.task_id} failed: {e}")
    
    def _execute_training_task(self, task: NeuralTask) -> Dict[str, Any]:
        """Execute a training task"""
        # Perform one training step
        return self.training_coordinator.training_step(task.model_id)
    
    def _execute_inference_task(self, task: NeuralTask) -> Dict[str, Any]:
        """Execute an inference task"""
        parameters = task.parameters
        input_data = parameters.get('input_data')
        batch_size = parameters.get('batch_size')
        
        return self.inference_engine.run_inference(task.model_id, input_data, batch_size)
    
    def _execute_evaluation_task(self, task: NeuralTask) -> Dict[str, Any]:
        """Execute an evaluation task"""
        parameters = task.parameters
        test_data_loader = parameters.get('test_data_loader')
        
        return self.inference_engine.batch_inference(task.model_id, test_data_loader)
    
    def _get_model_status(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get status of a model"""
        model_id = parameters.get('model_id')
        if not model_id:
            # Return status of all models
            return self._get_all_models_status()
        
        if model_id not in self._models:
            return {"error": "Model not found"}
        
        model_state = self._model_states[model_id]
        model_info = self._model_registry.get(model_id, {})
        
        return {
            "model_id": model_id,
            "phase": model_state.phase.value,
            "epoch": model_state.epoch,
            "iteration": model_state.iteration,
            "metrics": model_state.metrics,
            "best_metrics": model_state.best_metrics,
            "convergence_status": model_state.convergence_status,
            "resource_usage": model_state.resource_usage,
            "created_at": model_info.get('created_at'),
            "parameters": model_info.get('parameters'),
            "device": model_info.get('device')
        }
    
    def _get_all_models_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        with self._lock:
            models_status = {}
            for model_id in self._models:
                model_state = self._model_states[model_id]
                models_status[model_id] = {
                    "phase": model_state.phase.value,
                    "epoch": model_state.epoch,
                    "metrics": model_state.metrics,
                    "convergence_status": model_state.convergence_status
                }
        
        return {
            "models": models_status,
            "total_models": len(self._models),
            "active_tasks": len(self._active_tasks),
            "queued_tasks": len(self._task_queue)
        }
    
    def _optimize_resources(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resource usage"""
        try:
            optimization_results = {}
            
            # GPU memory optimization
            if self._gpu_optimizer:
                gpu_result = self._gpu_optimizer.optimize_memory_context()
                optimization_results['gpu_optimization'] = gpu_result
            
            # Model memory optimization
            memory_freed = self._optimize_model_memory()
            optimization_results['memory_freed'] = memory_freed
            
            # Task queue optimization
            queue_optimized = self._optimize_task_queue()
            optimization_results['queue_optimized'] = queue_optimized
            
            return {
                "status": "completed",
                "optimizations": optimization_results,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Resource optimization failed: {e}")
            return {"error": str(e)}
    
    def _optimize_model_memory(self) -> int:
        """Optimize model memory usage"""
        memory_freed = 0
        
        with self._lock:
            # Remove unused models from GPU
            for model_id, model in list(self._models.items()):
                model_state = self._model_states[model_id]
                
                # If model hasn't been used recently and is not training
                time_since_update = time.time() - model_state.last_updated
                if (time_since_update > 3600 and  # 1 hour
                    model_state.phase not in [TrainingPhase.TRAINING, TrainingPhase.FINE_TUNING]):
                    
                    # Move to CPU if on GPU
                    device = next(model.parameters()).device
                    if device.type == 'cuda':
                        model.cpu()
                        memory_freed += 1
                        logger.info(f"Moved model {model_id} to CPU to free GPU memory")
        
        return memory_freed
    
    def _optimize_task_queue(self) -> bool:
        """Optimize task queue ordering"""
        try:
            with self._lock:
                if len(self._task_queue) > 1:
                    # Re-sort queue based on current strategy
                    tasks = list(self._task_queue)
                    self._task_queue.clear()
                    
                    # Sort by priority and creation time
                    tasks.sort(key=lambda t: (-t.priority, t.created_at))
                    self._task_queue.extend(tasks)
                    
                    return True
            return False
            
        except Exception as e:
            logger.error(f"Task queue optimization failed: {e}")
            return False
    
    def _get_performance_report(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        with self._lock:
            total_models = len(self._models)
            active_models = sum(1 for state in self._model_states.values() 
                              if state.phase in [TrainingPhase.TRAINING, TrainingPhase.INFERENCE])
            
            # Calculate average metrics
            avg_metrics = {}
            for model_id, state in self._model_states.items():
                for metric, value in state.metrics.items():
                    if metric not in avg_metrics:
                        avg_metrics[metric] = []
                    avg_metrics[metric].append(value)
            
            for metric in avg_metrics:
                avg_metrics[metric] = np.mean(avg_metrics[metric])
            
            return {
                "total_models": total_models,
                "active_models": active_models,
                "completed_tasks": len(self._completed_tasks),
                "active_tasks": len(self._active_tasks),
                "queued_tasks": len(self._task_queue),
                "average_metrics": avg_metrics,
                "performance_metrics": dict(self._performance_metrics),
                "resource_usage": self._get_resource_usage(),
                "scheduler_strategy": self._scheduler_strategy.value,
                "gpu_optimization_active": self._gpu_optimizer is not None
            }
    
    def _initialize_device_manager(self) -> Dict[str, Any]:
        """Initialize device management"""
        device_info = {
            "cpu_available": True,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "default_device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        if device_info["cuda_available"]:
            device_info["cuda_devices"] = []
            for i in range(device_info["cuda_device_count"]):
                device_info["cuda_devices"].append({
                    "device_id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    "memory_available": torch.cuda.get_device_properties(i).total_memory
                })
        
        return device_info
    
    def _initialize_memory_monitor(self) -> Dict[str, Any]:
        """Initialize memory monitoring"""
        return {
            "enabled": True,
            "update_interval": 5.0,
            "memory_threshold": 0.9,
            "cleanup_threshold": 0.95
        }
    
    def _select_device(self, device_preference: str) -> torch.device:
        """Select appropriate device for model"""
        if device_preference == "auto":
            if torch.cuda.is_available():
                # Select GPU with most available memory
                best_gpu = 0
                max_memory = 0
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    available = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                    if available > max_memory:
                        max_memory = available
                        best_gpu = i
                return torch.device(f"cuda:{best_gpu}")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device_preference)
    
    def _estimate_model_memory(self, model: nn.Module) -> int:
        """Estimate model memory usage in bytes"""
        total_params = sum(p.numel() for p in model.parameters())
        param_bytes = total_params * 4  # 4 bytes per float32
        
        # Add estimated activation memory (rough estimate)
        activation_multiplier = 2  # Forward + backward pass
        total_memory = param_bytes * activation_multiplier
        
        return total_memory
    
    def _get_model_memory_usage(self, model_id: str) -> Dict[str, Any]:
        """Get memory usage for a model"""
        if model_id not in self._models:
            return {"error": "Model not found"}
        
        model = self._models[model_id]
        device = next(model.parameters()).device
        
        memory_info = {
            "device": str(device),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "parameter_memory": sum(p.numel() * p.element_size() for p in model.parameters())
        }
        
        if device.type == 'cuda':
            torch.cuda.set_device(device)
            memory_info.update({
                "gpu_allocated": torch.cuda.memory_allocated(device),
                "gpu_cached": torch.cuda.memory_reserved(device),
                "gpu_max_allocated": torch.cuda.max_memory_allocated(device)
            })
        
        return memory_info
    
    def _create_dummy_data_loader(self, model_id: str, batch_size: int) -> Any:
        """Create dummy data loader for testing"""
        model_state = self._model_states[model_id]
        input_shape = model_state.network_config.input_shape
        output_shape = model_state.network_config.output_shape
        
        # Create dummy data
        class DummyDataset:
            def __init__(self, size=1000):
                self.size = size
                self.input_shape = input_shape
                self.output_shape = output_shape
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                x = torch.randn(*self.input_shape)
                y = torch.randn(*self.output_shape)
                return x, y
        
        dataset = DummyDataset()
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def _calculate_evaluation_metrics(self, inference_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate evaluation metrics from inference results"""
        metrics = {}
        
        outputs = inference_result.get('outputs')
        targets = inference_result.get('targets')
        
        if outputs is not None and targets is not None:
            # Calculate MSE
            mse = np.mean((outputs - targets) ** 2)
            metrics['mse'] = float(mse)
            
            # Calculate MAE
            mae = np.mean(np.abs(outputs - targets))
            metrics['mae'] = float(mae)
            
            # Calculate R2 score (for regression)
            ss_res = np.sum((targets - outputs) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            metrics['r2'] = float(r2)
        
        metrics['inference_time'] = inference_result.get('total_time', 0)
        metrics['throughput'] = inference_result.get('throughput', 0)
        
        return metrics
    
    def _update_inference_metrics(self, model_id: str, inference_result: Dict[str, Any]):
        """Update inference performance metrics"""
        inference_time = inference_result.get('inference_time', 0)
        
        if model_id not in self._performance_metrics:
            self._performance_metrics[model_id] = {}
        
        # Update inference time metrics
        metrics = self._performance_metrics[model_id]
        if 'inference_times' not in metrics:
            metrics['inference_times'] = deque(maxlen=100)
        
        metrics['inference_times'].append(inference_time)
        metrics['avg_inference_time'] = np.mean(metrics['inference_times'])
        metrics['last_inference_time'] = inference_time
    
    def _get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage"""
        resource_usage = {
            "cpu_models": 0,
            "gpu_models": 0,
            "total_gpu_memory": 0,
            "allocated_gpu_memory": 0
        }
        
        for model_id, model in self._models.items():
            device = next(model.parameters()).device
            if device.type == 'cuda':
                resource_usage["gpu_models"] += 1
            else:
                resource_usage["cpu_models"] += 1
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                resource_usage["total_gpu_memory"] += props.total_memory
                resource_usage["allocated_gpu_memory"] += torch.cuda.memory_allocated(i)
        
        return resource_usage
    
    def get_model(self, model_id: str) -> Optional[NeuralModel]:
        """Get model by ID"""
        return self._models.get(model_id)
    
    def get_model_state(self, model_id: str) -> Optional[ModelState]:
        """Get model state by ID"""
        return self._model_states.get(model_id)
    
    def shutdown(self):
        """Shutdown the orchestrator"""
        logger.info("Shutting down NeuralOrchestrator")
        
        # Stop scheduler
        self._scheduler_running = False
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=5.0)
        
        # GPU cleanup
        if self._gpu_optimizer:
            self._gpu_optimizer.cleanup_after_training()
        
        # Clear models and states
        with self._lock:
            self._models.clear()
            self._model_states.clear()
            self._task_queue.clear()
            self._active_tasks.clear()
        
        logger.info("NeuralOrchestrator shutdown complete")