"""
Dynamic Switching Manager - Main orchestrator for dynamic switching between hybrid and pure p-adic compression
NO FALLBACKS - HARD FAILURES ONLY
"""

import asyncio
import logging
import threading
import time
import torch
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

# Import GAC system components
from gac_system.direction_state import DirectionStateManager, DirectionState
from gac_system.enhanced_bounder import EnhancedGradientBounder

# Import performance optimizer
from performance_optimizer import PerformanceOptimizer

# Import hybrid p-adic compression system
from .hybrid_padic_compressor import HybridPadicCompressionSystem, HybridPadicIntegrationManager
from .hybrid_padic_structures import HybridPadicWeight, HybridPadicManager


class CompressionMode(Enum):
    """Compression mode enumeration"""
    HYBRID = "hybrid"
    PURE_PADIC = "pure_padic"
    AUTO = "auto"


class SwitchingTrigger(Enum):
    """Switching trigger enumeration"""
    GRADIENT_DIRECTION = "gradient_direction"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    MEMORY_PRESSURE = "memory_pressure"
    DATA_SIZE = "data_size"
    ERROR_RATE = "error_rate"
    MANUAL = "manual"
    AUTOMATIC = "automatic"


@dataclass
class SwitchingEvent:
    """Record of a switching event"""
    event_id: str
    timestamp: datetime
    from_mode: CompressionMode
    to_mode: CompressionMode
    trigger: SwitchingTrigger
    data_shape: Tuple[int, ...]
    decision_confidence: float
    switching_time_ms: float
    performance_impact: float
    success: bool
    error_message: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate switching event"""
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("Event ID must be non-empty string")
        if not isinstance(self.timestamp, datetime):
            raise TypeError("Timestamp must be datetime")
        if not isinstance(self.from_mode, CompressionMode):
            raise TypeError("From mode must be CompressionMode")
        if not isinstance(self.to_mode, CompressionMode):
            raise TypeError("To mode must be CompressionMode")
        if not isinstance(self.trigger, SwitchingTrigger):
            raise TypeError("Trigger must be SwitchingTrigger")
        if not isinstance(self.decision_confidence, (int, float)) or not 0 <= self.decision_confidence <= 1:
            raise ValueError("Decision confidence must be float in [0, 1]")
        if not isinstance(self.switching_time_ms, (int, float)) or self.switching_time_ms < 0:
            raise ValueError("Switching time must be non-negative number")


@dataclass
class SwitchingConfig:
    """Configuration for dynamic switching system"""
    # Switching behavior
    enable_dynamic_switching: bool = True
    default_mode: CompressionMode = CompressionMode.AUTO
    auto_switch_confidence_threshold: float = 0.7
    manual_override_enabled: bool = True
    
    # Performance thresholds
    hybrid_performance_threshold: float = 0.8  # Switch to hybrid if performance below this
    pure_performance_threshold: float = 0.6   # Switch to pure if performance below this
    performance_measurement_window: int = 10  # Number of recent operations to consider
    
    # Data size thresholds
    hybrid_data_size_threshold: int = 1000    # Use hybrid for data > 1000 elements
    pure_data_size_threshold: int = 100       # Use pure for data < 100 elements
    
    # Memory thresholds
    gpu_memory_pressure_threshold: float = 0.85  # Switch if GPU memory > 85%
    system_memory_threshold_mb: int = 1024       # System memory threshold
    
    # Error rate thresholds
    max_reconstruction_error: float = 1e-6
    error_rate_threshold: float = 0.05  # 5% error rate threshold
    
    # Gradient analysis
    enable_gradient_analysis: bool = True
    gradient_stability_threshold: float = 0.1
    gradient_oscillation_threshold: float = 0.3
    
    # Switching timing
    min_switching_interval_ms: float = 100.0  # Minimum time between switches
    switching_cooldown_ms: float = 1000.0     # Cooldown after switching
    max_switches_per_minute: int = 60
    
    # Analytics and monitoring
    enable_switching_analytics: bool = True
    analytics_retention_hours: int = 24
    enable_performance_prediction: bool = True
    
    def __post_init__(self):
        """Validate switching configuration"""
        if not isinstance(self.auto_switch_confidence_threshold, (int, float)) or not 0 <= self.auto_switch_confidence_threshold <= 1:
            raise ValueError("Auto switch confidence threshold must be float in [0, 1]")
        if not isinstance(self.hybrid_performance_threshold, (int, float)) or not 0 <= self.hybrid_performance_threshold <= 1:
            raise ValueError("Hybrid performance threshold must be float in [0, 1]")
        if not isinstance(self.pure_performance_threshold, (int, float)) or not 0 <= self.pure_performance_threshold <= 1:
            raise ValueError("Pure performance threshold must be float in [0, 1]")
        if not isinstance(self.hybrid_data_size_threshold, int) or self.hybrid_data_size_threshold <= 0:
            raise ValueError("Hybrid data size threshold must be positive int")
        if not isinstance(self.gpu_memory_pressure_threshold, (int, float)) or not 0 <= self.gpu_memory_pressure_threshold <= 1:
            raise ValueError("GPU memory pressure threshold must be float in [0, 1]")
        if not isinstance(self.min_switching_interval_ms, (int, float)) or self.min_switching_interval_ms < 0:
            raise ValueError("Min switching interval must be non-negative number")


@dataclass
class SwitchingAnalytics:
    """Analytics data for switching decisions"""
    total_switches: int = 0
    successful_switches: int = 0
    failed_switches: int = 0
    switches_by_trigger: Dict[SwitchingTrigger, int] = field(default_factory=lambda: defaultdict(int))
    switches_by_mode: Dict[Tuple[CompressionMode, CompressionMode], int] = field(default_factory=lambda: defaultdict(int))
    average_switching_time_ms: float = 0.0
    average_decision_confidence: float = 0.0
    performance_improvement_ratio: float = 0.0
    last_analytics_update: Optional[datetime] = None
    
    def update_switching_metrics(self, event: SwitchingEvent):
        """Update analytics with switching event"""
        self.total_switches += 1
        self.last_analytics_update = datetime.utcnow()
        
        if event.success:
            self.successful_switches += 1
        else:
            self.failed_switches += 1
        
        self.switches_by_trigger[event.trigger] += 1
        self.switches_by_mode[(event.from_mode, event.to_mode)] += 1
        
        # Update averages
        if self.total_switches > 1:
            old_time_avg = self.average_switching_time_ms
            self.average_switching_time_ms = (
                (old_time_avg * (self.total_switches - 1) + event.switching_time_ms) / self.total_switches
            )
            
            old_conf_avg = self.average_decision_confidence
            self.average_decision_confidence = (
                (old_conf_avg * (self.total_switches - 1) + event.decision_confidence) / self.total_switches
            )
        else:
            self.average_switching_time_ms = event.switching_time_ms
            self.average_decision_confidence = event.decision_confidence


class DynamicSwitchingManager:
    """
    Main orchestrator for dynamic switching between hybrid and pure p-adic compression.
    Coordinates with GAC system, performance optimizer, and compression systems for intelligent switching decisions.
    """
    
    def __init__(self, config: Optional[SwitchingConfig] = None):
        """Initialize dynamic switching manager"""
        # NO FALLBACKS - HARD FAILURES ONLY
        if config is not None and not isinstance(config, SwitchingConfig):
            raise TypeError(f"Config must be SwitchingConfig or None, got {type(config)}")
        
        self.config = config or SwitchingConfig()
        self.logger = logging.getLogger('DynamicSwitchingManager')
        
        # Switching state
        self.is_initialized = False
        self.current_mode = self.config.default_mode
        self.last_switch_time: Optional[datetime] = None
        self.switching_enabled = self.config.enable_dynamic_switching
        
        # Component integrations
        self.direction_state_manager: Optional[DirectionStateManager] = None
        self.performance_optimizer: Optional[PerformanceOptimizer] = None
        self.gradient_bounder: Optional[EnhancedGradientBounder] = None
        self.hybrid_compression_system: Optional[HybridPadicCompressionSystem] = None
        
        # Switching analytics and history
        self.analytics = SwitchingAnalytics()
        self.switching_history: deque = deque(maxlen=1000)  # Last 1000 switching events
        self.performance_history: deque = deque(maxlen=self.config.performance_measurement_window)
        self.recent_switches: deque = deque(maxlen=100)  # Recent switches for rate limiting
        
        # Thread safety
        self._switching_lock = threading.RLock()
        self._analytics_lock = threading.RLock()
        self._performance_lock = threading.RLock()
        
        # Background monitoring
        self._monitoring_enabled = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        self.logger.info("DynamicSwitchingManager created successfully")
    
    def initialize_switching_system(self, 
                                  direction_manager: DirectionStateManager,
                                  performance_optimizer: PerformanceOptimizer,
                                  hybrid_compression_system: HybridPadicCompressionSystem,
                                  gradient_bounder: Optional[EnhancedGradientBounder] = None) -> None:
        """
        Initialize switching system with required components.
        
        Args:
            direction_manager: GAC direction state manager
            performance_optimizer: Performance optimizer instance
            hybrid_compression_system: Hybrid compression system
            gradient_bounder: Optional gradient bounder
            
        Raises:
            RuntimeError: If initialization fails
        """
        if self.is_initialized:
            return
        
        with self._switching_lock:
            try:
                # Validate and store component references
                if not isinstance(direction_manager, DirectionStateManager):
                    raise TypeError(f"Direction manager must be DirectionStateManager, got {type(direction_manager)}")
                if not isinstance(performance_optimizer, PerformanceOptimizer):
                    raise TypeError(f"Performance optimizer must be PerformanceOptimizer, got {type(performance_optimizer)}")
                if not isinstance(hybrid_compression_system, HybridPadicCompressionSystem):
                    raise TypeError(f"Hybrid compression system must be HybridPadicCompressionSystem, got {type(hybrid_compression_system)}")
                
                self.direction_state_manager = direction_manager
                self.performance_optimizer = performance_optimizer
                self.hybrid_compression_system = hybrid_compression_system
                self.gradient_bounder = gradient_bounder
                
                # Initialize decision engine and performance monitor
                from .switching_decision_engine import SwitchingDecisionEngine
                from .switching_performance_monitor import SwitchingPerformanceMonitor
                
                self.decision_engine = SwitchingDecisionEngine(self.config)
                self.decision_engine.initialize_decision_engine(
                    direction_manager=direction_manager,
                    performance_optimizer=performance_optimizer,
                    gradient_bounder=gradient_bounder
                )
                
                self.performance_monitor = SwitchingPerformanceMonitor(self.config)
                self.performance_monitor.initialize_performance_monitor(performance_optimizer)
                
                # Start background monitoring if enabled
                if self.config.enable_switching_analytics:
                    self._start_background_monitoring()
                
                self.is_initialized = True
                self.logger.info("Dynamic switching system initialized successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize switching system: {e}")
                raise RuntimeError(f"Switching system initialization failed: {e}")
    
    def should_switch_to_hybrid(self, data: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, float, SwitchingTrigger]:
        """
        Determine if compression should switch to hybrid mode.
        
        Args:
            data: Input tensor data
            context: Optional context information
            
        Returns:
            Tuple of (should_switch, confidence, trigger)
            
        Raises:
            RuntimeError: If switching system not initialized
            ValueError: If data is invalid
        """
        if not self.is_initialized:
            raise RuntimeError("Switching system not initialized")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Data must be torch.Tensor, got {type(data)}")
        if data.numel() == 0:
            raise ValueError("Data tensor cannot be empty")
        
        context = context or {}
        
        with self._switching_lock:
            # Check if switching is enabled
            if not self.switching_enabled:
                return False, 0.0, SwitchingTrigger.MANUAL
            
            # Check if already in hybrid mode
            if self.current_mode == CompressionMode.HYBRID:
                return False, 0.0, SwitchingTrigger.AUTOMATIC
            
            # Check switching rate limits
            if not self._can_switch_now():
                return False, 0.0, SwitchingTrigger.AUTOMATIC
            
            # Use decision engine for analysis
            decision_result = self.decision_engine.analyze_switching_criteria(data, context)
            
            # Check specific triggers for hybrid switching
            should_switch = False
            confidence = 0.0
            trigger = SwitchingTrigger.AUTOMATIC
            
            # Data size trigger
            if data.numel() > self.config.hybrid_data_size_threshold:
                should_switch = True
                confidence = max(confidence, 0.8)
                trigger = SwitchingTrigger.DATA_SIZE
            
            # Performance trigger
            if self._should_switch_for_performance(CompressionMode.HYBRID):
                should_switch = True
                confidence = max(confidence, decision_result.get('performance_confidence', 0.0))
                trigger = SwitchingTrigger.PERFORMANCE_THRESHOLD
            
            # Memory pressure trigger (favor hybrid for large data)
            if self._is_memory_pressure() and data.numel() > 500:
                should_switch = True
                confidence = max(confidence, 0.7)
                trigger = SwitchingTrigger.MEMORY_PRESSURE
            
            # Gradient direction trigger
            if context.get('gradients') is not None:
                gradient_confidence = self.decision_engine.evaluate_gradient_direction(context['gradients'])
                if gradient_confidence > 0.6:  # Stable/oscillating gradients favor hybrid
                    should_switch = True
                    confidence = max(confidence, gradient_confidence)
                    trigger = SwitchingTrigger.GRADIENT_DIRECTION
            
            # Apply confidence threshold
            if confidence < self.config.auto_switch_confidence_threshold:
                should_switch = False
            
            return should_switch, confidence, trigger
    
    def should_switch_to_pure(self, data: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, float, SwitchingTrigger]:
        """
        Determine if compression should switch to pure p-adic mode.
        
        Args:
            data: Input tensor data
            context: Optional context information
            
        Returns:
            Tuple of (should_switch, confidence, trigger)
            
        Raises:
            RuntimeError: If switching system not initialized
            ValueError: If data is invalid
        """
        if not self.is_initialized:
            raise RuntimeError("Switching system not initialized")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Data must be torch.Tensor, got {type(data)}")
        if data.numel() == 0:
            raise ValueError("Data tensor cannot be empty")
        
        context = context or {}
        
        with self._switching_lock:
            # Check if switching is enabled
            if not self.switching_enabled:
                return False, 0.0, SwitchingTrigger.MANUAL
            
            # Check if already in pure mode
            if self.current_mode == CompressionMode.PURE_PADIC:
                return False, 0.0, SwitchingTrigger.AUTOMATIC
            
            # Check switching rate limits
            if not self._can_switch_now():
                return False, 0.0, SwitchingTrigger.AUTOMATIC
            
            # Use decision engine for analysis
            decision_result = self.decision_engine.analyze_switching_criteria(data, context)
            
            # Check specific triggers for pure switching
            should_switch = False
            confidence = 0.0
            trigger = SwitchingTrigger.AUTOMATIC
            
            # Data size trigger
            if data.numel() < self.config.pure_data_size_threshold:
                should_switch = True
                confidence = max(confidence, 0.8)
                trigger = SwitchingTrigger.DATA_SIZE
            
            # Performance trigger
            if self._should_switch_for_performance(CompressionMode.PURE_PADIC):
                should_switch = True
                confidence = max(confidence, decision_result.get('performance_confidence', 0.0))
                trigger = SwitchingTrigger.PERFORMANCE_THRESHOLD
            
            # Memory pressure trigger (favor pure for small data)
            if self._is_memory_pressure() and data.numel() <= 500:
                should_switch = True
                confidence = max(confidence, 0.7)
                trigger = SwitchingTrigger.MEMORY_PRESSURE
            
            # Error rate trigger
            if self._has_high_error_rate():
                should_switch = True
                confidence = max(confidence, 0.6)
                trigger = SwitchingTrigger.ERROR_RATE
            
            # Gradient direction trigger
            if context.get('gradients') is not None:
                gradient_confidence = self.decision_engine.evaluate_gradient_direction(context['gradients'])
                if gradient_confidence < 0.4:  # Unstable gradients might favor pure
                    should_switch = True
                    confidence = max(confidence, 1.0 - gradient_confidence)
                    trigger = SwitchingTrigger.GRADIENT_DIRECTION
            
            # Apply confidence threshold
            if confidence < self.config.auto_switch_confidence_threshold:
                should_switch = False
            
            return should_switch, confidence, trigger
    
    def execute_switch(self, compression_system: HybridPadicCompressionSystem, target_mode: CompressionMode) -> SwitchingEvent:
        """
        Execute compression mode switch.
        
        Args:
            compression_system: Compression system to switch
            target_mode: Target compression mode
            
        Returns:
            Switching event record
            
        Raises:
            RuntimeError: If switch execution fails
            ValueError: If parameters are invalid
        """
        if not isinstance(compression_system, HybridPadicCompressionSystem):
            raise TypeError(f"Compression system must be HybridPadicCompressionSystem, got {type(compression_system)}")
        if not isinstance(target_mode, CompressionMode):
            raise TypeError(f"Target mode must be CompressionMode, got {type(target_mode)}")
        if target_mode == CompressionMode.AUTO:
            raise ValueError("Cannot switch to AUTO mode directly")
        
        start_time = time.time()
        event_id = str(uuid.uuid4())
        from_mode = self.current_mode
        
        with self._switching_lock:
            try:
                # Create switching event
                switch_event = SwitchingEvent(
                    event_id=event_id,
                    timestamp=datetime.utcnow(),
                    from_mode=from_mode,
                    to_mode=target_mode,
                    trigger=SwitchingTrigger.MANUAL,  # Will be updated by caller
                    data_shape=(0,),  # Will be updated by caller
                    decision_confidence=1.0,  # Will be updated by caller
                    switching_time_ms=0.0,  # Will be calculated
                    performance_impact=0.0,  # Will be calculated later
                    success=False  # Will be updated
                )
                
                # Execute the switch
                if target_mode == CompressionMode.HYBRID:
                    # Enable hybrid compression
                    compression_system.enable_hybrid = True
                    compression_system.force_hybrid = False
                elif target_mode == CompressionMode.PURE_PADIC:
                    # Disable hybrid compression
                    compression_system.enable_hybrid = False
                    compression_system.force_hybrid = False
                else:
                    raise ValueError(f"Invalid target mode: {target_mode}")
                
                # Update current mode
                self.current_mode = target_mode
                self.last_switch_time = datetime.utcnow()
                
                # Calculate switching time
                switching_time_ms = (time.time() - start_time) * 1000
                switch_event.switching_time_ms = switching_time_ms
                switch_event.success = True
                
                # Add to recent switches for rate limiting
                self.recent_switches.append(datetime.utcnow())
                
                # Record switching event
                self.switching_history.append(switch_event)
                
                # Update analytics
                with self._analytics_lock:
                    self.analytics.update_switching_metrics(switch_event)
                
                # Monitor performance impact
                if self.performance_monitor:
                    self.performance_monitor.monitor_switching_performance(switch_event)
                
                self.logger.info(f"Successfully switched from {from_mode.value} to {target_mode.value} in {switching_time_ms:.2f}ms")
                
                return switch_event
                
            except Exception as e:
                # Record failed switching event
                switching_time_ms = (time.time() - start_time) * 1000
                switch_event.switching_time_ms = switching_time_ms
                switch_event.success = False
                switch_event.error_message = str(e)
                
                self.switching_history.append(switch_event)
                
                with self._analytics_lock:
                    self.analytics.update_switching_metrics(switch_event)
                
                self.logger.error(f"Failed to switch from {from_mode.value} to {target_mode.value}: {e}")
                raise RuntimeError(f"Switch execution failed: {e}")
    
    def get_switching_analytics(self) -> Dict[str, Any]:
        """
        Get comprehensive switching analytics.
        
        Returns:
            Dictionary containing switching analytics
        """
        with self._analytics_lock:
            current_time = datetime.utcnow()
            
            return {
                'summary': {
                    'total_switches': self.analytics.total_switches,
                    'successful_switches': self.analytics.successful_switches,
                    'failed_switches': self.analytics.failed_switches,
                    'success_rate': (
                        self.analytics.successful_switches / max(1, self.analytics.total_switches)
                    ),
                    'average_switching_time_ms': self.analytics.average_switching_time_ms,
                    'average_decision_confidence': self.analytics.average_decision_confidence
                },
                'switching_patterns': {
                    'switches_by_trigger': dict(self.analytics.switches_by_trigger),
                    'switches_by_mode_transition': {
                        f"{from_mode.value}_to_{to_mode.value}": count
                        for (from_mode, to_mode), count in self.analytics.switches_by_mode.items()
                    }
                },
                'current_state': {
                    'current_mode': self.current_mode.value,
                    'switching_enabled': self.switching_enabled,
                    'last_switch_time': self.last_switch_time.isoformat() if self.last_switch_time else None,
                    'time_since_last_switch_seconds': (
                        (current_time - self.last_switch_time).total_seconds() 
                        if self.last_switch_time else None
                    )
                },
                'recent_activity': {
                    'recent_switches_count': len(self.recent_switches),
                    'switches_in_last_minute': len([
                        s for s in self.recent_switches 
                        if (current_time - s).total_seconds() < 60
                    ]),
                    'can_switch_now': self._can_switch_now()
                },
                'performance': {
                    'performance_improvement_ratio': self.analytics.performance_improvement_ratio,
                    'recent_performance_trend': self._get_recent_performance_trend()
                },
                'configuration': {
                    'auto_switch_confidence_threshold': self.config.auto_switch_confidence_threshold,
                    'hybrid_data_size_threshold': self.config.hybrid_data_size_threshold,
                    'pure_data_size_threshold': self.config.pure_data_size_threshold,
                    'min_switching_interval_ms': self.config.min_switching_interval_ms,
                    'max_switches_per_minute': self.config.max_switches_per_minute
                },
                'last_update': current_time.isoformat()
            }
    
    def update_switching_config(self, config: SwitchingConfig) -> None:
        """
        Update switching configuration.
        
        Args:
            config: New switching configuration
            
        Raises:
            TypeError: If config is invalid
        """
        if not isinstance(config, SwitchingConfig):
            raise TypeError(f"Config must be SwitchingConfig, got {type(config)}")
        
        with self._switching_lock:
            old_config = self.config
            self.config = config
            
            # Update switching enabled state
            self.switching_enabled = config.enable_dynamic_switching
            
            # Update component configurations if available
            if self.decision_engine:
                self.decision_engine.update_configuration(config)
            
            if self.performance_monitor:
                self.performance_monitor.update_configuration(config)
            
            self.logger.info(f"Switching configuration updated")
            self.logger.debug(f"Config changes: enable_dynamic_switching={old_config.enable_dynamic_switching}->{config.enable_dynamic_switching}")
    
    def _can_switch_now(self) -> bool:
        """Check if switching is allowed based on rate limits and cooldowns"""
        current_time = datetime.utcnow()
        
        # Check minimum interval since last switch
        if self.last_switch_time:
            time_since_last = (current_time - self.last_switch_time).total_seconds() * 1000
            if time_since_last < self.config.min_switching_interval_ms:
                return False
        
        # Check switches per minute limit
        minute_ago = current_time - timedelta(minutes=1)
        recent_switches_count = len([
            s for s in self.recent_switches 
            if s >= minute_ago
        ])
        
        if recent_switches_count >= self.config.max_switches_per_minute:
            return False
        
        return True
    
    def _should_switch_for_performance(self, target_mode: CompressionMode) -> bool:
        """Check if should switch based on performance metrics"""
        if not self.performance_history:
            return False
        
        recent_performance = list(self.performance_history)[-5:]  # Last 5 operations
        avg_performance = sum(recent_performance) / len(recent_performance)
        
        if target_mode == CompressionMode.HYBRID:
            return avg_performance < self.config.hybrid_performance_threshold
        elif target_mode == CompressionMode.PURE_PADIC:
            return avg_performance < self.config.pure_performance_threshold
        
        return False
    
    def _is_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            usage_ratio = allocated / total
            return usage_ratio > self.config.gpu_memory_pressure_threshold
        
        return False
    
    def _has_high_error_rate(self) -> bool:
        """Check if reconstruction error rate is high"""
        # This would need to be implemented with actual error tracking
        # For now, return False as placeholder
        return False
    
    def _get_recent_performance_trend(self) -> str:
        """Get recent performance trend direction"""
        if len(self.performance_history) < 3:
            return "insufficient_data"
        
        recent = list(self.performance_history)[-5:]
        if len(recent) < 3:
            return "insufficient_data"
        
        # Simple trend analysis
        first_half = recent[:len(recent)//2]
        second_half = recent[len(recent)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg * 1.05:
            return "improving"
        elif second_avg < first_avg * 0.95:
            return "declining"
        else:
            return "stable"
    
    def _start_background_monitoring(self) -> None:
        """Start background monitoring thread"""
        if self._monitoring_enabled:
            return
        
        self._monitoring_enabled = True
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._background_monitoring_loop,
            name="SwitchingMonitor",
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("Background switching monitoring started")
    
    def _background_monitoring_loop(self) -> None:
        """Background monitoring loop"""
        while not self._stop_monitoring.wait(30.0):  # Check every 30 seconds
            try:
                # Clean up old data
                self._cleanup_old_data()
                
                # Update analytics
                self._update_background_analytics()
                
            except Exception as e:
                self.logger.error(f"Error in background monitoring: {e}")
    
    def _cleanup_old_data(self) -> None:
        """Clean up old analytics data"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.config.analytics_retention_hours)
        
        # Clean up recent switches
        self.recent_switches = deque([
            s for s in self.recent_switches if s >= cutoff_time
        ], maxlen=100)
    
    def _update_background_analytics(self) -> None:
        """Update background analytics"""
        with self._analytics_lock:
            self.analytics.last_analytics_update = datetime.utcnow()
    
    def shutdown(self) -> None:
        """Shutdown dynamic switching manager"""
        self.logger.info("Shutting down dynamic switching manager")
        
        # Stop background monitoring
        if self._monitoring_enabled:
            self._monitoring_enabled = False
            self._stop_monitoring.set()
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=5.0)
        
        # Shutdown components
        if self.decision_engine:
            self.decision_engine.shutdown()
        
        if self.performance_monitor:
            self.performance_monitor.shutdown()
        
        self.is_initialized = False
        self.logger.info("Dynamic switching manager shutdown complete")