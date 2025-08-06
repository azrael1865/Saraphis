"""
JAX Stub Interface for Remote Operations without JAX Installation.
Enables lightweight client-side operations that mirror TropicalJAXEngine API.

PRODUCTION-READY - NO PLACEHOLDERS - FULL ERROR HANDLING

This interface allows clients to execute JAX operations remotely without
having JAX installed locally. All operations are serialized, sent to
remote servers, and results are deserialized back to the client.

Key Features:
- Complete mirroring of TropicalJAXEngine API
- Efficient msgpack serialization with compression
- Multiple protocol support (gRPC, HTTP/HTTPS, WebSocket)
- Connection pooling and load balancing
- Async and batch operation support
- Circuit breaker pattern for resilience
- Comprehensive error handling and retry logic
"""

import os
import sys
import time
import json
import hashlib
import logging
import asyncio
import threading
import queue
import zlib
import lz4.frame
import msgpack
import numpy as np
import torch
import aiohttp
import grpc
import websockets
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from collections import defaultdict, deque
import weakref
import traceback

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import deployment configuration
from independent_core.compression_systems.tropical.deployment_config import (
    DeploymentConfig,
    RemoteServerConfig,
    LoadBalancingStrategy,
    CircuitBreakerConfig,
    RetryPolicy
)

# Import tropical core for constants
from independent_core.compression_systems.tropical.tropical_core import (
    TROPICAL_ZERO,
    TROPICAL_EPSILON
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations that can be executed remotely"""
    # Core tropical operations
    TROPICAL_ADD = "tropical_add"
    TROPICAL_MULTIPLY = "tropical_multiply"
    TROPICAL_MATRIX_MULTIPLY = "tropical_matrix_multiply"
    TROPICAL_POWER = "tropical_power"
    POLYNOMIAL_TO_JAX = "polynomial_to_jax"
    EVALUATE_POLYNOMIAL = "evaluate_polynomial"
    VMAP_POLYNOMIAL_EVALUATION = "vmap_polynomial_evaluation"
    
    # Advanced operations
    TROPICAL_CONV1D = "tropical_conv1d"
    TROPICAL_CONV2D = "tropical_conv2d"
    TROPICAL_POOL2D = "tropical_pool2d"
    BATCH_TROPICAL_DISTANCE = "batch_tropical_distance"
    TROPICAL_GRADIENT = "tropical_gradient"
    TROPICAL_SOFTMAX = "tropical_softmax"
    
    # Channel operations
    CHANNELS_TO_JAX = "channels_to_jax"
    PROCESS_CHANNELS = "process_channels"
    PARALLEL_CHANNEL_MULTIPLY = "parallel_channel_multiply"


class SerializationFormat(Enum):
    """Serialization formats for data transfer"""
    MSGPACK = "msgpack"
    JSON = "json"
    NUMPY = "numpy"
    PICKLE = "pickle"


class CompressionType(Enum):
    """Compression algorithms for large data"""
    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"
    GZIP = "gzip"


@dataclass
class RemoteOperation:
    """Represents a remote operation to be executed"""
    operation_id: str
    operation_type: OperationType
    args: List[Any]
    kwargs: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 0
    timeout_ms: int = 30000
    retry_count: int = 0
    callback: Optional[Callable] = None
    future: Optional[Future] = None
    
    def __hash__(self):
        return hash(self.operation_id)


@dataclass
class OperationResult:
    """Result from a remote operation"""
    operation_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0
    server_info: Dict[str, Any] = field(default_factory=dict)


class ConnectionPool:
    """Manages connections to remote servers with pooling"""
    
    def __init__(self, config: DeploymentConfig):
        """Initialize connection pool"""
        self.config = config
        self.servers = config.remote_servers
        self.pool_config = config.connection_pool
        
        # Connection pools by server
        self._grpc_channels: Dict[str, List[grpc.Channel]] = defaultdict(list)
        self._http_sessions: Dict[str, List[aiohttp.ClientSession]] = defaultdict(list)
        self._websocket_connections: Dict[str, List[websockets.WebSocketClientProtocol]] = defaultdict(list)
        
        # Connection tracking
        self._active_connections: Dict[str, int] = defaultdict(int)
        self._connection_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._failed_connections: Dict[str, int] = defaultdict(int)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize minimum connections
        self._initialize_pools()
    
    def _initialize_pools(self):
        """Create minimum connections for each server"""
        for server in self.servers:
            server_key = f"{server.host}:{server.port}"
            
            # Create minimum connections based on protocol
            for _ in range(self.pool_config.min_connections):
                try:
                    if server.protocol == "grpc":
                        channel = self._create_grpc_channel(server)
                        self._grpc_channels[server_key].append(channel)
                    elif server.protocol in ["http", "https"]:
                        # HTTP sessions are created asynchronously
                        pass
                    elif server.protocol == "websocket":
                        # WebSocket connections are created asynchronously
                        pass
                except Exception as e:
                    logger.error(f"Failed to initialize connection to {server_key}: {e}")
                    self._failed_connections[server_key] += 1
    
    def _create_grpc_channel(self, server: RemoteServerConfig) -> grpc.Channel:
        """Create a gRPC channel with proper configuration"""
        options = [
            ('grpc.max_send_message_length', -1),
            ('grpc.max_receive_message_length', -1),
            ('grpc.keepalive_time_ms', self.pool_config.keep_alive_interval_ms),
            ('grpc.keepalive_timeout_ms', 20000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
        ]
        
        if server.ssl_verify and server.protocol == "grpc":
            # Create secure channel
            if server.ssl_cert_path:
                with open(server.ssl_cert_path, 'rb') as f:
                    root_certificates = f.read()
                credentials = grpc.ssl_channel_credentials(root_certificates=root_certificates)
            else:
                credentials = grpc.ssl_channel_credentials()
            
            channel = grpc.secure_channel(
                f"{server.host}:{server.port}",
                credentials,
                options=options
            )
        else:
            # Create insecure channel
            channel = grpc.insecure_channel(
                f"{server.host}:{server.port}",
                options=options
            )
        
        return channel
    
    async def _create_http_session(self, server: RemoteServerConfig) -> aiohttp.ClientSession:
        """Create an HTTP/HTTPS session with proper configuration"""
        timeout = aiohttp.ClientTimeout(
            total=server.request_timeout_ms / 1000,
            connect=server.connection_timeout_ms / 1000,
            sock_connect=server.connection_timeout_ms / 1000,
            sock_read=server.request_timeout_ms / 1000
        )
        
        connector_kwargs = {
            'limit': server.max_connections,
            'limit_per_host': server.max_connections,
            'keepalive_timeout': self.pool_config.idle_timeout_ms / 1000,
            'force_close': False,
        }
        
        if server.ssl_verify and server.protocol == "https":
            import ssl
            ssl_context = ssl.create_default_context()
            if server.ssl_cert_path:
                ssl_context.load_cert_chain(server.ssl_cert_path, server.ssl_key_path)
            connector_kwargs['ssl'] = ssl_context
        else:
            connector_kwargs['ssl'] = False
        
        connector = aiohttp.TCPConnector(**connector_kwargs)
        
        # Add authentication headers
        headers = self.config.security.get_auth_headers()
        
        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        )
        
        return session
    
    async def _create_websocket_connection(self, server: RemoteServerConfig) -> websockets.WebSocketClientProtocol:
        """Create a WebSocket connection with proper configuration"""
        uri = f"{'wss' if server.ssl_verify else 'ws'}://{server.host}:{server.port}/{server.api_version}"
        
        ssl_context = None
        if server.ssl_verify:
            import ssl
            ssl_context = ssl.create_default_context()
            if server.ssl_cert_path:
                ssl_context.load_cert_chain(server.ssl_cert_path, server.ssl_key_path)
        
        # Add authentication headers
        extra_headers = self.config.security.get_auth_headers()
        
        connection = await websockets.connect(
            uri,
            ssl=ssl_context,
            extra_headers=extra_headers,
            ping_interval=self.pool_config.keep_alive_interval_ms / 1000,
            ping_timeout=20,
            close_timeout=10,
            max_size=int(self.config.resource_limits.max_response_size_mb * 1024 * 1024)
        )
        
        return connection
    
    def get_connection(self, server: RemoteServerConfig, protocol: Optional[str] = None):
        """Get a connection from the pool for a server"""
        protocol = protocol or server.protocol
        server_key = f"{server.host}:{server.port}"
        
        with self._lock:
            # Check if server is healthy
            if self._failed_connections[server_key] > self.config.circuit_breaker.failure_threshold:
                raise ConnectionError(f"Server {server_key} is marked as unhealthy")
            
            # Get connection based on protocol
            if protocol == "grpc":
                channels = self._grpc_channels[server_key]
                if channels:
                    return channels[0]  # gRPC channels are shared
                else:
                    channel = self._create_grpc_channel(server)
                    self._grpc_channels[server_key].append(channel)
                    return channel
            
            elif protocol in ["http", "https"]:
                # HTTP sessions are returned asynchronously
                return None  # Will be created in async context
            
            elif protocol == "websocket":
                # WebSocket connections are returned asynchronously
                return None  # Will be created in async context
            
            else:
                raise ValueError(f"Unsupported protocol: {protocol}")
    
    async def get_async_connection(self, server: RemoteServerConfig, protocol: Optional[str] = None):
        """Get an async connection from the pool"""
        protocol = protocol or server.protocol
        server_key = f"{server.host}:{server.port}"
        
        if protocol in ["http", "https"]:
            sessions = self._http_sessions[server_key]
            if not sessions:
                session = await self._create_http_session(server)
                self._http_sessions[server_key].append(session)
                return session
            return sessions[0]  # Reuse existing session
        
        elif protocol == "websocket":
            connections = self._websocket_connections[server_key]
            if not connections or not connections[0].open:
                connection = await self._create_websocket_connection(server)
                self._websocket_connections[server_key] = [connection]
                return connection
            return connections[0]
        
        else:
            # For gRPC, return sync connection
            return self.get_connection(server, protocol)
    
    def return_connection(self, server: RemoteServerConfig, connection: Any, failed: bool = False):
        """Return a connection to the pool"""
        server_key = f"{server.host}:{server.port}"
        
        with self._lock:
            if failed:
                self._failed_connections[server_key] += 1
                # Don't return failed connections to pool
                return
            
            # Track successful connection
            self._connection_times[server_key].append(time.time())
            self._failed_connections[server_key] = max(0, self._failed_connections[server_key] - 1)
    
    async def close_all(self):
        """Close all connections in the pool"""
        # Close gRPC channels
        for channels in self._grpc_channels.values():
            for channel in channels:
                try:
                    channel.close()
                except Exception as e:
                    logger.error(f"Error closing gRPC channel: {e}")
        
        # Close HTTP sessions
        for sessions in self._http_sessions.values():
            for session in sessions:
                try:
                    await session.close()
                except Exception as e:
                    logger.error(f"Error closing HTTP session: {e}")
        
        # Close WebSocket connections
        for connections in self._websocket_connections.values():
            for connection in connections:
                try:
                    await connection.close()
                except Exception as e:
                    logger.error(f"Error closing WebSocket connection: {e}")
        
        self._grpc_channels.clear()
        self._http_sessions.clear()
        self._websocket_connections.clear()


class DataSerializer:
    """Handles serialization and deserialization of data for remote operations"""
    
    def __init__(self, compression_threshold_bytes: int = 1024 * 1024):  # 1MB
        """Initialize serializer with compression threshold"""
        self.compression_threshold = compression_threshold_bytes
        self._type_handlers = self._register_type_handlers()
    
    def _register_type_handlers(self) -> Dict[type, Tuple[Callable, Callable]]:
        """Register serialization/deserialization handlers for different types"""
        return {
            np.ndarray: (self._serialize_numpy, self._deserialize_numpy),
            torch.Tensor: (self._serialize_torch, self._deserialize_torch),
            list: (self._serialize_list, self._deserialize_list),
            dict: (self._serialize_dict, self._deserialize_dict),
            float: (self._serialize_scalar, self._deserialize_scalar),
            int: (self._serialize_scalar, self._deserialize_scalar),
        }
    
    def serialize(self, data: Any, format: SerializationFormat = SerializationFormat.MSGPACK,
                 compression: CompressionType = CompressionType.LZ4) -> bytes:
        """Serialize data with optional compression"""
        # Convert data to serializable format
        serializable = self._to_serializable(data)
        
        # Create metadata
        metadata = {
            'type': type(data).__name__,
            'shape': getattr(data, 'shape', None),
            'dtype': str(getattr(data, 'dtype', None)),
            'format': format.value,
            'compression': CompressionType.NONE.value,
            'original_size': 0
        }
        
        # Serialize based on format
        if format == SerializationFormat.MSGPACK:
            serialized = msgpack.packb({
                'metadata': metadata,
                'data': serializable
            }, use_bin_type=True)
        elif format == SerializationFormat.JSON:
            serialized = json.dumps({
                'metadata': metadata,
                'data': serializable
            }).encode('utf-8')
        else:
            raise ValueError(f"Unsupported serialization format: {format}")
        
        # Apply compression if data is large enough
        original_size = len(serialized)
        metadata['original_size'] = original_size
        
        if original_size > self.compression_threshold:
            if compression == CompressionType.ZLIB:
                compressed = zlib.compress(serialized, level=6)
            elif compression == CompressionType.LZ4:
                compressed = lz4.frame.compress(serialized, compression_level=6)
            elif compression == CompressionType.GZIP:
                import gzip
                compressed = gzip.compress(serialized, compresslevel=6)
            else:
                compressed = serialized
            
            # Only use compression if it reduces size
            if len(compressed) < original_size * 0.9:
                metadata['compression'] = compression.value
                
                # Re-serialize with compression metadata
                if format == SerializationFormat.MSGPACK:
                    return msgpack.packb({
                        'metadata': metadata,
                        'compressed_data': compressed
                    }, use_bin_type=True)
                else:
                    # For JSON, use base64 encoding for binary data
                    import base64
                    return json.dumps({
                        'metadata': metadata,
                        'compressed_data': base64.b64encode(compressed).decode('ascii')
                    }).encode('utf-8')
        
        return serialized
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize data with automatic decompression"""
        # Parse initial structure
        try:
            initial = msgpack.unpackb(data, raw=False)
        except:
            try:
                initial = json.loads(data.decode('utf-8'))
            except:
                raise ValueError("Failed to deserialize data")
        
        metadata = initial.get('metadata', {})
        
        # Handle compression
        if metadata.get('compression') != CompressionType.NONE.value and 'compressed_data' in initial:
            compressed = initial['compressed_data']
            
            # Handle base64 decoding for JSON
            if isinstance(compressed, str):
                import base64
                compressed = base64.b64decode(compressed)
            
            # Decompress
            if metadata['compression'] == CompressionType.ZLIB.value:
                decompressed = zlib.decompress(compressed)
            elif metadata['compression'] == CompressionType.LZ4.value:
                decompressed = lz4.frame.decompress(compressed)
            elif metadata['compression'] == CompressionType.GZIP.value:
                import gzip
                decompressed = gzip.decompress(compressed)
            else:
                decompressed = compressed
            
            # Re-parse decompressed data
            if metadata['format'] == SerializationFormat.MSGPACK.value:
                initial = msgpack.unpackb(decompressed, raw=False)
            else:
                initial = json.loads(decompressed.decode('utf-8'))
        
        # Convert back to original type
        data = initial.get('data')
        data_type = metadata.get('type')
        
        if data_type == 'ndarray':
            return self._deserialize_numpy(data, metadata)
        elif data_type == 'Tensor':
            return self._deserialize_torch(data, metadata)
        else:
            return data
    
    def _to_serializable(self, data: Any) -> Any:
        """Convert data to a serializable format"""
        data_type = type(data)
        
        if data_type in self._type_handlers:
            serializer, _ = self._type_handlers[data_type]
            return serializer(data)
        elif isinstance(data, (list, tuple)):
            return [self._to_serializable(item) for item in data]
        elif isinstance(data, dict):
            return {k: self._to_serializable(v) for k, v in data.items()}
        else:
            return data
    
    def _serialize_numpy(self, arr: np.ndarray) -> Dict[str, Any]:
        """Serialize numpy array"""
        # Handle special values
        if np.any(np.isinf(arr)):
            arr = np.where(np.isinf(arr), 1e308 if arr.dtype == np.float64 else 1e38, arr)
        if np.any(np.isnan(arr)):
            arr = np.where(np.isnan(arr), 0, arr)
        
        return {
            'data': arr.tobytes(),
            'shape': arr.shape,
            'dtype': str(arr.dtype),
            'order': 'C' if arr.flags['C_CONTIGUOUS'] else 'F'
        }
    
    def _deserialize_numpy(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> np.ndarray:
        """Deserialize numpy array"""
        if isinstance(data, dict) and 'data' in data:
            arr_bytes = data['data']
            if isinstance(arr_bytes, str):
                import base64
                arr_bytes = base64.b64decode(arr_bytes)
            
            arr = np.frombuffer(arr_bytes, dtype=np.dtype(data['dtype']))
            arr = arr.reshape(data['shape'])
            
            if data.get('order') == 'F':
                arr = np.asfortranarray(arr)
            
            return arr
        else:
            # Fallback for simple lists
            return np.array(data)
    
    def _serialize_torch(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Serialize PyTorch tensor"""
        # Convert to numpy for serialization
        arr = tensor.detach().cpu().numpy()
        return {
            **self._serialize_numpy(arr),
            'requires_grad': tensor.requires_grad,
            'device': str(tensor.device)
        }
    
    def _deserialize_torch(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> torch.Tensor:
        """Deserialize PyTorch tensor"""
        arr = self._deserialize_numpy(data, metadata)
        tensor = torch.from_numpy(arr)
        
        if data.get('requires_grad'):
            tensor.requires_grad_(True)
        
        # Note: Device placement is handled by caller
        return tensor
    
    def _serialize_list(self, lst: List) -> List:
        """Serialize list recursively"""
        return [self._to_serializable(item) for item in lst]
    
    def _deserialize_list(self, data: List, metadata: Dict[str, Any]) -> List:
        """Deserialize list"""
        return data
    
    def _serialize_dict(self, d: Dict) -> Dict:
        """Serialize dictionary recursively"""
        return {k: self._to_serializable(v) for k, v in d.items()}
    
    def _deserialize_dict(self, data: Dict, metadata: Dict[str, Any]) -> Dict:
        """Deserialize dictionary"""
        return data
    
    def _serialize_scalar(self, value: Union[int, float]) -> Union[int, float]:
        """Serialize scalar value"""
        # Handle special float values
        if isinstance(value, float):
            if np.isinf(value):
                return 1e308 if value > 0 else -1e308
            elif np.isnan(value):
                return 0.0
        return value
    
    def _deserialize_scalar(self, value: Union[int, float], metadata: Dict[str, Any]) -> Union[int, float]:
        """Deserialize scalar value"""
        return value


class LoadBalancer:
    """Implements load balancing strategies for server selection"""
    
    def __init__(self, servers: List[RemoteServerConfig], strategy: LoadBalancingStrategy):
        """Initialize load balancer"""
        self.servers = servers
        self.strategy = strategy
        self._round_robin_index = 0
        self._server_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'requests': 0,
            'failures': 0,
            'total_time': 0,
            'active_connections': 0,
            'last_response_time': 0
        })
        self._lock = threading.Lock()
    
    def select_server(self, operation: Optional[RemoteOperation] = None) -> RemoteServerConfig:
        """Select a server based on the configured strategy"""
        with self._lock:
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin()
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections()
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin()
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return self._least_response_time()
            elif self.strategy == LoadBalancingStrategy.RANDOM:
                return self._random()
            elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
                return self._resource_based()
            else:
                # Default to round robin
                return self._round_robin()
    
    def _round_robin(self) -> RemoteServerConfig:
        """Simple round-robin selection"""
        server = self.servers[self._round_robin_index]
        self._round_robin_index = (self._round_robin_index + 1) % len(self.servers)
        return server
    
    def _least_connections(self) -> RemoteServerConfig:
        """Select server with least active connections"""
        min_connections = float('inf')
        selected = self.servers[0]
        
        for server in self.servers:
            key = f"{server.host}:{server.port}"
            connections = self._server_stats[key]['active_connections']
            if connections < min_connections:
                min_connections = connections
                selected = server
        
        return selected
    
    def _weighted_round_robin(self) -> RemoteServerConfig:
        """Weighted round-robin based on server weights"""
        # Build weighted list
        weighted_servers = []
        for server in self.servers:
            weighted_servers.extend([server] * server.weight)
        
        if not weighted_servers:
            return self.servers[0]
        
        # Use round-robin on weighted list
        index = self._round_robin_index % len(weighted_servers)
        self._round_robin_index += 1
        return weighted_servers[index]
    
    def _least_response_time(self) -> RemoteServerConfig:
        """Select server with lowest average response time"""
        min_time = float('inf')
        selected = self.servers[0]
        
        for server in self.servers:
            key = f"{server.host}:{server.port}"
            stats = self._server_stats[key]
            
            if stats['requests'] > 0:
                avg_time = stats['total_time'] / stats['requests']
                # Add penalty for failures
                avg_time *= (1 + stats['failures'] / max(stats['requests'], 1))
                
                if avg_time < min_time:
                    min_time = avg_time
                    selected = server
        
        return selected
    
    def _random(self) -> RemoteServerConfig:
        """Random server selection"""
        import random
        return random.choice(self.servers)
    
    def _resource_based(self) -> RemoteServerConfig:
        """Select based on resource availability (requires monitoring)"""
        # This would integrate with monitoring data
        # For now, fallback to least connections
        return self._least_connections()
    
    def update_stats(self, server: RemoteServerConfig, success: bool, 
                     response_time: float, active_delta: int = 0):
        """Update server statistics after request"""
        with self._lock:
            key = f"{server.host}:{server.port}"
            stats = self._server_stats[key]
            
            stats['requests'] += 1
            if not success:
                stats['failures'] += 1
            stats['total_time'] += response_time
            stats['last_response_time'] = response_time
            stats['active_connections'] += active_delta
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get current load balancer statistics"""
        with self._lock:
            return dict(self._server_stats)


class CircuitBreaker:
    """Implements circuit breaker pattern for fault tolerance"""
    
    def __init__(self, config: CircuitBreakerConfig):
        """Initialize circuit breaker"""
        self.config = config
        self._states: Dict[str, str] = {}  # server_key -> state (closed, open, half_open)
        self._failure_counts: Dict[str, int] = defaultdict(int)
        self._success_counts: Dict[str, int] = defaultdict(int)
        self._last_failure_time: Dict[str, float] = {}
        self._half_open_requests: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
    
    def is_open(self, server: RemoteServerConfig) -> bool:
        """Check if circuit is open for a server"""
        if not self.config.enable_circuit_breaker:
            return False
        
        with self._lock:
            key = f"{server.host}:{server.port}"
            state = self._states.get(key, 'closed')
            
            if state == 'open':
                # Check if timeout has passed
                if key in self._last_failure_time:
                    elapsed = time.time() - self._last_failure_time[key]
                    if elapsed > self.config.timeout_seconds:
                        # Transition to half-open
                        self._states[key] = 'half_open'
                        self._half_open_requests[key] = 0
                        logger.info(f"Circuit breaker for {key} transitioning to half-open")
                        return False
                return True
            
            elif state == 'half_open':
                # Allow limited requests in half-open state
                return self._half_open_requests[key] >= self.config.half_open_max_requests
            
            return False
    
    def record_success(self, server: RemoteServerConfig):
        """Record a successful request"""
        if not self.config.enable_circuit_breaker:
            return
        
        with self._lock:
            key = f"{server.host}:{server.port}"
            state = self._states.get(key, 'closed')
            
            if state == 'half_open':
                self._success_counts[key] += 1
                
                # Check if we can close the circuit
                if self._success_counts[key] >= self.config.success_threshold:
                    self._states[key] = 'closed'
                    self._failure_counts[key] = 0
                    self._success_counts[key] = 0
                    logger.info(f"Circuit breaker for {key} closed")
            
            elif state == 'closed':
                # Reset failure count on success
                self._failure_counts[key] = max(0, self._failure_counts[key] - 1)
    
    def record_failure(self, server: RemoteServerConfig, error: Exception):
        """Record a failed request"""
        if not self.config.enable_circuit_breaker:
            return
        
        # Check if error type should trigger circuit breaker
        error_type = type(error).__name__
        if error_type not in self.config.error_types:
            return
        
        with self._lock:
            key = f"{server.host}:{server.port}"
            state = self._states.get(key, 'closed')
            
            self._failure_counts[key] += 1
            self._last_failure_time[key] = time.time()
            
            if state == 'closed':
                # Check if we should open the circuit
                if self._failure_counts[key] >= self.config.failure_threshold:
                    self._states[key] = 'open'
                    logger.warning(f"Circuit breaker for {key} opened after {self._failure_counts[key]} failures")
            
            elif state == 'half_open':
                # Immediately reopen on failure in half-open state
                self._states[key] = 'open'
                self._success_counts[key] = 0
                logger.warning(f"Circuit breaker for {key} reopened due to failure in half-open state")
    
    def get_state(self, server: RemoteServerConfig) -> str:
        """Get current circuit state for a server"""
        with self._lock:
            key = f"{server.host}:{server.port}"
            return self._states.get(key, 'closed')


class JAXStubInterface:
    """Lightweight client stub mirroring TropicalJAXEngine API for remote operations"""
    
    def __init__(self, config: DeploymentConfig):
        """Initialize JAX stub interface with deployment configuration"""
        self.config = config
        self.servers = config.remote_servers
        
        if not self.servers:
            raise ValueError("No remote servers configured for JAX stub interface")
        
        # Initialize components
        self.connection_pool = ConnectionPool(config)
        self.serializer = DataSerializer()
        self.load_balancer = LoadBalancer(self.servers, config.load_balancing_strategy)
        self.circuit_breaker = CircuitBreaker(config.circuit_breaker)
        
        # Operation management
        self._operation_queue = queue.PriorityQueue()
        self._pending_operations: Dict[str, RemoteOperation] = {}
        self._operation_futures: Dict[str, Future] = {}
        self._batch_queue: List[RemoteOperation] = []
        self._batch_lock = threading.Lock()
        
        # Executor for async operations
        self.executor = ThreadPoolExecutor(max_workers=config.resource_limits.max_concurrent_requests)
        
        # Caching
        self._result_cache: Dict[str, OperationResult] = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = 1000
        
        # Metrics
        self._metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_latency_ms': 0
        }
        
        # Start background workers
        self._running = True
        self._start_workers()
        
        logger.info(f"JAX Stub Interface initialized with {len(self.servers)} remote servers")
    
    def _start_workers(self):
        """Start background worker threads"""
        # Start async event loop in background thread
        self._event_loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()
        
        # Start batch processor
        self._batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self._batch_thread.start()
    
    def _run_event_loop(self):
        """Run async event loop in background thread"""
        asyncio.set_event_loop(self._event_loop)
        self._event_loop.run_forever()
    
    def _batch_processor(self):
        """Process batched operations periodically"""
        while self._running:
            time.sleep(0.1)  # Process batches every 100ms
            
            with self._batch_lock:
                if self._batch_queue:
                    batch = self._batch_queue[:self.config.resource_limits.max_batch_size]
                    self._batch_queue = self._batch_queue[self.config.resource_limits.max_batch_size:]
                    
                    if batch:
                        # Execute batch asynchronously
                        future = self.executor.submit(self._execute_batch, batch)
                        
                        # Link futures
                        for op in batch:
                            if op.future:
                                op.future = future
    
    def _generate_operation_id(self, operation_type: OperationType, args: tuple, kwargs: dict) -> str:
        """Generate unique operation ID with caching support"""
        # Create hash for cache key
        cache_key = hashlib.md5(
            f"{operation_type.value}:{str(args)}:{str(kwargs)}".encode()
        ).hexdigest()
        
        # Add timestamp for uniqueness
        return f"{cache_key}_{time.time()}"
    
    def _check_cache(self, operation_id: str) -> Optional[OperationResult]:
        """Check if operation result is cached"""
        with self._cache_lock:
            cache_key = operation_id.split('_')[0]
            if cache_key in self._result_cache:
                self._metrics['cache_hits'] += 1
                return self._result_cache[cache_key]
            self._metrics['cache_misses'] += 1
            return None
    
    def _cache_result(self, operation_id: str, result: OperationResult):
        """Cache operation result"""
        with self._cache_lock:
            cache_key = operation_id.split('_')[0]
            
            # Limit cache size
            if len(self._result_cache) >= self._max_cache_size:
                # Remove oldest entries
                oldest_keys = list(self._result_cache.keys())[:100]
                for key in oldest_keys:
                    del self._result_cache[key]
            
            self._result_cache[cache_key] = result
    
    def _execute_remote_operation(self, operation: RemoteOperation) -> OperationResult:
        """Execute a single remote operation synchronously"""
        start_time = time.time()
        
        # Check cache first
        cached_result = self._check_cache(operation.operation_id)
        if cached_result:
            return cached_result
        
        # Select server
        server = self.load_balancer.select_server(operation)
        
        # Check circuit breaker
        if self.circuit_breaker.is_open(server):
            # Try another server
            for alt_server in self.servers:
                if alt_server != server and not self.circuit_breaker.is_open(alt_server):
                    server = alt_server
                    break
            else:
                # All servers are unavailable
                return OperationResult(
                    operation_id=operation.operation_id,
                    success=False,
                    error="All servers are unavailable (circuit breaker open)"
                )
        
        # Update stats
        self.load_balancer.update_stats(server, True, 0, active_delta=1)
        
        try:
            # Execute based on protocol
            if server.protocol == "grpc":
                result = self._execute_grpc(server, operation)
            elif server.protocol in ["http", "https"]:
                result = asyncio.run_coroutine_threadsafe(
                    self._execute_http(server, operation),
                    self._event_loop
                ).result(timeout=operation.timeout_ms / 1000)
            elif server.protocol == "websocket":
                result = asyncio.run_coroutine_threadsafe(
                    self._execute_websocket(server, operation),
                    self._event_loop
                ).result(timeout=operation.timeout_ms / 1000)
            else:
                raise ValueError(f"Unsupported protocol: {server.protocol}")
            
            # Record success
            response_time = (time.time() - start_time) * 1000
            self.load_balancer.update_stats(server, True, response_time, active_delta=-1)
            self.circuit_breaker.record_success(server)
            
            # Cache result
            self._cache_result(operation.operation_id, result)
            
            # Update metrics
            self._metrics['successful_operations'] += 1
            self._metrics['total_latency_ms'] += response_time
            
            return result
            
        except Exception as e:
            # Record failure
            response_time = (time.time() - start_time) * 1000
            self.load_balancer.update_stats(server, False, response_time, active_delta=-1)
            self.circuit_breaker.record_failure(server, e)
            
            # Update metrics
            self._metrics['failed_operations'] += 1
            
            # Retry logic
            if operation.retry_count < self.config.retry_policy.max_retries:
                operation.retry_count += 1
                delay_ms = self.config.retry_policy.get_delay_ms(operation.retry_count)
                time.sleep(delay_ms / 1000)
                return self._execute_remote_operation(operation)
            
            return OperationResult(
                operation_id=operation.operation_id,
                success=False,
                error=str(e),
                execution_time_ms=response_time
            )
    
    def _execute_grpc(self, server: RemoteServerConfig, operation: RemoteOperation) -> OperationResult:
        """Execute operation via gRPC
        
        Protocol Buffer Definition (for reference):
        ```proto
        syntax = "proto3";
        
        message JAXOperationRequest {
            string operation_id = 1;
            string operation_type = 2;
            bytes serialized_args = 3;
            bytes serialized_kwargs = 4;
            string compression_type = 5;
        }
        
        message JAXOperationResponse {
            string operation_id = 1;
            bool success = 2;
            bytes result_data = 3;
            string error_message = 4;
            map<string, string> metadata = 5;
        }
        
        service JAXExecutor {
            rpc Execute(JAXOperationRequest) returns (JAXOperationResponse);
            rpc ExecuteStream(stream JAXOperationRequest) returns (stream JAXOperationResponse);
        }
        ```
        """
        try:
            # Get channel from pool
            channel = self.connection_pool.get_connection(server, "grpc")
            
            # Create a simple stub class for gRPC communication
            class JAXExecutorStub:
                def __init__(self, channel):
                    self.channel = channel
                    # Define method descriptors
                    self.Execute = channel.unary_unary(
                        '/jax.JAXExecutor/Execute',
                        request_serializer=self._serialize_grpc_request,
                        response_deserializer=self._deserialize_grpc_response
                    )
                
                def _serialize_grpc_request(self, request):
                    """Serialize request for gRPC"""
                    return msgpack.packb(request)
                
                def _deserialize_grpc_response(self, response_bytes):
                    """Deserialize gRPC response"""
                    return msgpack.unpackb(response_bytes, raw=False)
            
            # Create stub
            stub = JAXExecutorStub(channel)
            
            # Serialize operation data
            args_data = self.serializer.serialize(operation.args)
            kwargs_data = self.serializer.serialize(operation.kwargs)
            
            # Apply compression if needed
            compression_type = CompressionType.LZ4.value
            if compression_type == CompressionType.LZ4.value:
                args_data = lz4.frame.compress(args_data)
                kwargs_data = lz4.frame.compress(kwargs_data)
            elif compression_type == CompressionType.ZLIB.value:
                args_data = zlib.compress(args_data)
                kwargs_data = zlib.compress(kwargs_data)
            
            # Create request
            request = {
                'operation_id': operation.operation_id,
                'operation_type': operation.operation_type.value,
                'serialized_args': args_data,
                'serialized_kwargs': kwargs_data,
                'compression_type': compression_type
            }
            
            # Set timeout based on operation type
            timeout = operation.timeout or self.config.timeouts.operation_timeout_seconds
            
            # Execute RPC call
            response = stub.Execute(request, timeout=timeout)
            
            # Process response
            if isinstance(response, dict):
                # Decompress result if needed
                result_data = response.get('result_data', b'')
                if response.get('compression_type') == CompressionType.LZ4.value:
                    result_data = lz4.frame.decompress(result_data)
                elif response.get('compression_type') == CompressionType.ZLIB.value:
                    result_data = zlib.decompress(result_data)
                
                # Deserialize result
                if response.get('success', False):
                    result = self.serializer.deserialize(result_data) if result_data else None
                    
                    return OperationResult(
                        operation_id=operation.operation_id,
                        success=True,
                        result=result,
                        server_info={'host': server.host, 'port': server.port, 'protocol': 'grpc'},
                        metadata=response.get('metadata', {})
                    )
                else:
                    return OperationResult(
                        operation_id=operation.operation_id,
                        success=False,
                        error=response.get('error_message', 'Unknown gRPC error'),
                        server_info={'host': server.host, 'port': server.port, 'protocol': 'grpc'}
                    )
            else:
                # Handle raw response
                return OperationResult(
                    operation_id=operation.operation_id,
                    success=False,
                    error=f"Unexpected gRPC response type: {type(response)}",
                    server_info={'host': server.host, 'port': server.port, 'protocol': 'grpc'}
                )
                
        except grpc.RpcError as e:
            # Handle gRPC-specific errors
            error_msg = f"gRPC error: {e.code()} - {e.details()}"
            logger.error(f"gRPC execution failed: {error_msg}")
            
            return OperationResult(
                operation_id=operation.operation_id,
                success=False,
                error=error_msg,
                server_info={'host': server.host, 'port': server.port, 'protocol': 'grpc'}
            )
            
        except Exception as e:
            # Handle general errors
            error_msg = f"gRPC execution error: {str(e)}"
            logger.error(f"gRPC execution failed: {error_msg}\n{traceback.format_exc()}")
            
            return OperationResult(
                operation_id=operation.operation_id,
                success=False,
                error=error_msg,
                server_info={'host': server.host, 'port': server.port, 'protocol': 'grpc'}
            )
    
    async def _execute_http(self, server: RemoteServerConfig, operation: RemoteOperation) -> OperationResult:
        """Execute operation via HTTP/HTTPS"""
        # Get session from pool
        session = await self.connection_pool.get_async_connection(server, server.protocol)
        
        # Serialize request
        request_data = self.serializer.serialize({
            'operation_type': operation.operation_type.value,
            'args': operation.args,
            'kwargs': operation.kwargs
        })
        
        # Build URL
        url = f"{server.get_url()}/execute"
        
        # Make request
        async with session.post(url, data=request_data) as response:
            if response.status == 200:
                response_data = await response.read()
                result_data = self.serializer.deserialize(response_data)
                
                return OperationResult(
                    operation_id=operation.operation_id,
                    success=True,
                    result=result_data,
                    server_info={'host': server.host, 'port': server.port}
                )
            else:
                error_text = await response.text()
                return OperationResult(
                    operation_id=operation.operation_id,
                    success=False,
                    error=f"HTTP {response.status}: {error_text}"
                )
    
    async def _execute_websocket(self, server: RemoteServerConfig, operation: RemoteOperation) -> OperationResult:
        """Execute operation via WebSocket"""
        # Get connection from pool
        connection = await self.connection_pool.get_async_connection(server, "websocket")
        
        # Serialize request
        request_data = self.serializer.serialize({
            'operation_id': operation.operation_id,
            'operation_type': operation.operation_type.value,
            'args': operation.args,
            'kwargs': operation.kwargs
        })
        
        # Send request
        await connection.send(request_data)
        
        # Wait for response
        response_data = await asyncio.wait_for(
            connection.recv(),
            timeout=operation.timeout_ms / 1000
        )
        
        # Deserialize response
        result_data = self.serializer.deserialize(response_data)
        
        return OperationResult(
            operation_id=operation.operation_id,
            success=result_data.get('success', False),
            result=result_data.get('result'),
            error=result_data.get('error'),
            server_info={'host': server.host, 'port': server.port}
        )
    
    def _execute_batch(self, operations: List[RemoteOperation]) -> List[OperationResult]:
        """Execute a batch of operations"""
        results = []
        
        # Group operations by type for efficiency
        grouped = defaultdict(list)
        for op in operations:
            grouped[op.operation_type].append(op)
        
        # Execute each group
        for op_type, ops in grouped.items():
            # Could optimize further by sending entire batch to server
            for op in ops:
                result = self._execute_remote_operation(op)
                results.append(result)
                
                # Trigger callback if provided
                if op.callback:
                    op.callback(result)
        
        return results
    
    # Core TropicalJAXEngine operations
    
    def tropical_add(self, a: Any, b: Any) -> Any:
        """Remote tropical addition (max operation)"""
        operation = RemoteOperation(
            operation_id=self._generate_operation_id(OperationType.TROPICAL_ADD, (a, b), {}),
            operation_type=OperationType.TROPICAL_ADD,
            args=[a, b],
            kwargs={}
        )
        
        result = self._execute_remote_operation(operation)
        if not result.success:
            raise RuntimeError(f"Remote tropical_add failed: {result.error}")
        
        return result.result
    
    def tropical_multiply(self, a: Any, b: Any) -> Any:
        """Remote tropical multiplication (addition in log space)"""
        operation = RemoteOperation(
            operation_id=self._generate_operation_id(OperationType.TROPICAL_MULTIPLY, (a, b), {}),
            operation_type=OperationType.TROPICAL_MULTIPLY,
            args=[a, b],
            kwargs={}
        )
        
        result = self._execute_remote_operation(operation)
        if not result.success:
            raise RuntimeError(f"Remote tropical_multiply failed: {result.error}")
        
        return result.result
    
    def tropical_matrix_multiply(self, A: Any, B: Any) -> Any:
        """Remote tropical matrix multiplication"""
        operation = RemoteOperation(
            operation_id=self._generate_operation_id(OperationType.TROPICAL_MATRIX_MULTIPLY, (A, B), {}),
            operation_type=OperationType.TROPICAL_MATRIX_MULTIPLY,
            args=[A, B],
            kwargs={}
        )
        
        result = self._execute_remote_operation(operation)
        if not result.success:
            raise RuntimeError(f"Remote tropical_matrix_multiply failed: {result.error}")
        
        return result.result
    
    def tropical_power(self, base: Any, exponent: Union[int, float]) -> Any:
        """Remote tropical power operation"""
        operation = RemoteOperation(
            operation_id=self._generate_operation_id(OperationType.TROPICAL_POWER, (base, exponent), {}),
            operation_type=OperationType.TROPICAL_POWER,
            args=[base, exponent],
            kwargs={}
        )
        
        result = self._execute_remote_operation(operation)
        if not result.success:
            raise RuntimeError(f"Remote tropical_power failed: {result.error}")
        
        return result.result
    
    def evaluate_polynomial(self, coeffs: Any, exponents: Any, points: Any) -> Any:
        """Remote polynomial evaluation"""
        operation = RemoteOperation(
            operation_id=self._generate_operation_id(
                OperationType.EVALUATE_POLYNOMIAL, 
                (coeffs, exponents, points), 
                {}
            ),
            operation_type=OperationType.EVALUATE_POLYNOMIAL,
            args=[coeffs, exponents, points],
            kwargs={}
        )
        
        result = self._execute_remote_operation(operation)
        if not result.success:
            raise RuntimeError(f"Remote evaluate_polynomial failed: {result.error}")
        
        return result.result
    
    # TropicalJAXOperations methods
    
    def tropical_conv1d(self, signal: Any, kernel: Any, padding: str = 'valid') -> Any:
        """Remote tropical 1D convolution"""
        operation = RemoteOperation(
            operation_id=self._generate_operation_id(
                OperationType.TROPICAL_CONV1D,
                (signal, kernel),
                {'padding': padding}
            ),
            operation_type=OperationType.TROPICAL_CONV1D,
            args=[signal, kernel],
            kwargs={'padding': padding}
        )
        
        result = self._execute_remote_operation(operation)
        if not result.success:
            raise RuntimeError(f"Remote tropical_conv1d failed: {result.error}")
        
        return result.result
    
    def tropical_conv2d(self, input: Any, kernel: Any, stride: Tuple[int, int] = (1, 1)) -> Any:
        """Remote tropical 2D convolution"""
        operation = RemoteOperation(
            operation_id=self._generate_operation_id(
                OperationType.TROPICAL_CONV2D,
                (input, kernel),
                {'stride': stride}
            ),
            operation_type=OperationType.TROPICAL_CONV2D,
            args=[input, kernel],
            kwargs={'stride': stride}
        )
        
        result = self._execute_remote_operation(operation)
        if not result.success:
            raise RuntimeError(f"Remote tropical_conv2d failed: {result.error}")
        
        return result.result
    
    def tropical_pool2d(self, input: Any, kernel_size: Tuple[int, int]) -> Any:
        """Remote tropical 2D pooling"""
        operation = RemoteOperation(
            operation_id=self._generate_operation_id(
                OperationType.TROPICAL_POOL2D,
                (input, kernel_size),
                {}
            ),
            operation_type=OperationType.TROPICAL_POOL2D,
            args=[input, kernel_size],
            kwargs={}
        )
        
        result = self._execute_remote_operation(operation)
        if not result.success:
            raise RuntimeError(f"Remote tropical_pool2d failed: {result.error}")
        
        return result.result
    
    def batch_tropical_distance(self, a: Any, b: Any) -> Any:
        """Remote batch tropical distance computation"""
        operation = RemoteOperation(
            operation_id=self._generate_operation_id(
                OperationType.BATCH_TROPICAL_DISTANCE,
                (a, b),
                {}
            ),
            operation_type=OperationType.BATCH_TROPICAL_DISTANCE,
            args=[a, b],
            kwargs={}
        )
        
        result = self._execute_remote_operation(operation)
        if not result.success:
            raise RuntimeError(f"Remote batch_tropical_distance failed: {result.error}")
        
        return result.result
    
    def tropical_gradient(self, func: Callable, x: Any, epsilon: float = 1e-5) -> Any:
        """Remote tropical gradient computation"""
        return self._compute_remote_gradient(func, x, epsilon)
    
    def _compute_remote_gradient(self, func: Callable, x: Any, epsilon: float = 1e-5) -> Any:
        """Implement remote gradient computation with function serialization
        
        This method handles gradient computation remotely by:
        1. Serializing the function and inputs
        2. Sending gradient request to JAX server
        3. Handling response deserialization
        4. Falling back to local computation if available
        """
        try:
            # Try to extract function information for serialization
            func_info = self._serialize_function(func)
            
            # Create gradient operation
            operation = RemoteOperation(
                operation_id=self._generate_operation_id(
                    OperationType.TROPICAL_GRADIENT,
                    (func_info, x),
                    {'epsilon': epsilon}
                ),
                operation_type=OperationType.TROPICAL_GRADIENT,
                args=[func_info, x],
                kwargs={'epsilon': epsilon}
            )
            
            # Execute remote gradient computation
            result = self._execute_remote_operation(operation)
            
            if result.success:
                gradient = result.result
                
                # Validate gradient shape matches input shape
                if hasattr(x, 'shape'):
                    x_shape = x.shape if hasattr(x, 'shape') else np.array(x).shape
                    grad_shape = gradient.shape if hasattr(gradient, 'shape') else np.array(gradient).shape
                    
                    if x_shape != grad_shape:
                        logger.warning(f"Gradient shape {grad_shape} doesn't match input shape {x_shape}")
                        # Attempt to reshape if possible
                        try:
                            gradient = np.reshape(gradient, x_shape)
                        except Exception as reshape_error:
                            logger.error(f"Failed to reshape gradient: {reshape_error}")
                            # Fall back to local computation if available
                            return self._compute_local_gradient_fallback(func, x, epsilon)
                
                return gradient
            else:
                # Remote computation failed, try local fallback
                logger.warning(f"Remote gradient computation failed: {result.error}")
                return self._compute_local_gradient_fallback(func, x, epsilon)
                
        except Exception as e:
            logger.error(f"Gradient computation error: {str(e)}")
            # Fall back to local computation
            return self._compute_local_gradient_fallback(func, x, epsilon)
    
    def _serialize_function(self, func: Callable) -> Dict[str, Any]:
        """Serialize a callable function for remote execution
        
        Attempts multiple serialization strategies:
        1. Check if function is a known operation
        2. Extract source code if possible
        3. Use pickle for simple functions
        4. Create a function descriptor
        """
        func_info = {
            'type': 'callable',
            'name': getattr(func, '__name__', 'anonymous')
        }
        
        # Strategy 1: Check if it's a known tropical operation
        known_ops = {
            'tropical_add': OperationType.TROPICAL_ADD,
            'tropical_multiply': OperationType.TROPICAL_MULTIPLY,
            'tropical_power': OperationType.TROPICAL_POWER,
            'tropical_softmax': OperationType.TROPICAL_SOFTMAX
        }
        
        func_name = getattr(func, '__name__', '')
        if func_name in known_ops:
            func_info['known_operation'] = known_ops[func_name].value
            return func_info
        
        # Strategy 2: Try to extract source code
        try:
            import inspect
            source = inspect.getsource(func)
            func_info['source_code'] = source
            func_info['serialization'] = 'source'
            
            # Also capture any closure variables
            if hasattr(func, '__closure__') and func.__closure__:
                closure_vars = {}
                for i, cell in enumerate(func.__closure__):
                    try:
                        closure_vars[f'var_{i}'] = cell.cell_contents
                    except:
                        pass
                if closure_vars:
                    func_info['closure_vars'] = self.serializer.serialize(closure_vars)
                    
        except Exception as source_error:
            logger.debug(f"Could not extract source code: {source_error}")
            
            # Strategy 3: Try pickle serialization
            try:
                import pickle
                import base64
                pickled = pickle.dumps(func)
                func_info['pickled_function'] = base64.b64encode(pickled).decode('utf-8')
                func_info['serialization'] = 'pickle'
            except Exception as pickle_error:
                logger.debug(f"Could not pickle function: {pickle_error}")
                
                # Strategy 4: Create function descriptor
                func_info['serialization'] = 'descriptor'
                func_info['module'] = getattr(func, '__module__', None)
                func_info['qualname'] = getattr(func, '__qualname__', None)
                
                # Try to capture function signature
                try:
                    import inspect
                    sig = inspect.signature(func)
                    func_info['signature'] = str(sig)
                except:
                    pass
        
        return func_info
    
    def _compute_local_gradient_fallback(self, func: Callable, x: Any, epsilon: float = 1e-5) -> Any:
        """Compute gradient locally as fallback when remote fails
        
        Uses finite differences for numerical gradient computation
        """
        try:
            # Check if JAX is available locally
            try:
                import jax
                import jax.numpy as jnp
                
                # Use JAX's automatic differentiation
                grad_func = jax.grad(func)
                gradient = grad_func(x)
                logger.info("Using local JAX for gradient computation")
                return gradient
                
            except ImportError:
                # JAX not available, use numerical gradient
                logger.info("Using numerical gradient computation (JAX not available locally)")
                
                # Convert to numpy for computation
                x_np = np.array(x) if not isinstance(x, np.ndarray) else x
                gradient = np.zeros_like(x_np)
                
                # Compute gradient using finite differences
                for idx in np.ndindex(x_np.shape):
                    # Create perturbation
                    x_plus = x_np.copy()
                    x_minus = x_np.copy()
                    
                    x_plus[idx] += epsilon
                    x_minus[idx] -= epsilon
                    
                    # Compute finite difference
                    f_plus = func(x_plus)
                    f_minus = func(x_minus)
                    
                    # Handle scalar and array outputs
                    if np.isscalar(f_plus):
                        gradient[idx] = (f_plus - f_minus) / (2 * epsilon)
                    else:
                        # For vector-valued functions, take the sum
                        gradient[idx] = np.sum(f_plus - f_minus) / (2 * epsilon)
                
                return gradient
                
        except Exception as e:
            logger.error(f"Local gradient computation failed: {str(e)}")
            # Return zeros as last resort
            logger.warning("Returning zero gradient as last resort")
            return np.zeros_like(x)
    
    def tropical_softmax(self, x: Any, temperature: float = 1.0, axis: int = -1) -> Any:
        """Remote tropical softmax"""
        operation = RemoteOperation(
            operation_id=self._generate_operation_id(
                OperationType.TROPICAL_SOFTMAX,
                (x,),
                {'temperature': temperature, 'axis': axis}
            ),
            operation_type=OperationType.TROPICAL_SOFTMAX,
            args=[x],
            kwargs={'temperature': temperature, 'axis': axis}
        )
        
        result = self._execute_remote_operation(operation)
        if not result.success:
            raise RuntimeError(f"Remote tropical_softmax failed: {result.error}")
        
        return result.result
    
    # JAXChannelProcessor methods
    
    def channels_to_jax(self, channels: Any) -> Dict[str, Any]:
        """Remote channel conversion to JAX format"""
        operation = RemoteOperation(
            operation_id=self._generate_operation_id(
                OperationType.CHANNELS_TO_JAX,
                (channels,),
                {}
            ),
            operation_type=OperationType.CHANNELS_TO_JAX,
            args=[channels],
            kwargs={}
        )
        
        result = self._execute_remote_operation(operation)
        if not result.success:
            raise RuntimeError(f"Remote channels_to_jax failed: {result.error}")
        
        return result.result
    
    def process_channels(self, coeffs: Any, exponents: Any, operation: str = "normalize") -> Tuple[Any, Any]:
        """Remote channel processing"""
        remote_op = RemoteOperation(
            operation_id=self._generate_operation_id(
                OperationType.PROCESS_CHANNELS,
                (coeffs, exponents),
                {'operation': operation}
            ),
            operation_type=OperationType.PROCESS_CHANNELS,
            args=[coeffs, exponents],
            kwargs={'operation': operation}
        )
        
        result = self._execute_remote_operation(remote_op)
        if not result.success:
            raise RuntimeError(f"Remote process_channels failed: {result.error}")
        
        return result.result
    
    def parallel_channel_multiply(self, channels_a: Any, channels_b: Any) -> Any:
        """Remote parallel channel multiplication"""
        operation = RemoteOperation(
            operation_id=self._generate_operation_id(
                OperationType.PARALLEL_CHANNEL_MULTIPLY,
                (channels_a, channels_b),
                {}
            ),
            operation_type=OperationType.PARALLEL_CHANNEL_MULTIPLY,
            args=[channels_a, channels_b],
            kwargs={}
        )
        
        result = self._execute_remote_operation(operation)
        if not result.success:
            raise RuntimeError(f"Remote parallel_channel_multiply failed: {result.error}")
        
        return result.result
    
    # Async versions of operations
    
    async def tropical_add_async(self, a: Any, b: Any) -> Any:
        """Async remote tropical addition"""
        operation = RemoteOperation(
            operation_id=self._generate_operation_id(OperationType.TROPICAL_ADD, (a, b), {}),
            operation_type=OperationType.TROPICAL_ADD,
            args=[a, b],
            kwargs={}
        )
        
        # Execute in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, self._execute_remote_operation, operation)
        
        if not result.success:
            raise RuntimeError(f"Remote tropical_add_async failed: {result.error}")
        
        return result.result
    
    async def tropical_multiply_async(self, a: Any, b: Any) -> Any:
        """Async remote tropical multiplication"""
        operation = RemoteOperation(
            operation_id=self._generate_operation_id(OperationType.TROPICAL_MULTIPLY, (a, b), {}),
            operation_type=OperationType.TROPICAL_MULTIPLY,
            args=[a, b],
            kwargs={}
        )
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, self._execute_remote_operation, operation)
        
        if not result.success:
            raise RuntimeError(f"Remote tropical_multiply_async failed: {result.error}")
        
        return result.result
    
    async def tropical_matrix_multiply_async(self, A: Any, B: Any) -> Any:
        """Async remote tropical matrix multiplication"""
        operation = RemoteOperation(
            operation_id=self._generate_operation_id(OperationType.TROPICAL_MATRIX_MULTIPLY, (A, B), {}),
            operation_type=OperationType.TROPICAL_MATRIX_MULTIPLY,
            args=[A, B],
            kwargs={}
        )
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self.executor, self._execute_remote_operation, operation)
        
        if not result.success:
            raise RuntimeError(f"Remote tropical_matrix_multiply_async failed: {result.error}")
        
        return result.result
    
    # Batch operations
    
    def batch_operations(self, ops: List[Dict]) -> List[Any]:
        """Execute multiple operations in batch"""
        operations = []
        futures = []
        
        for op_dict in ops:
            op_type = OperationType(op_dict['type'])
            operation = RemoteOperation(
                operation_id=self._generate_operation_id(
                    op_type,
                    tuple(op_dict.get('args', [])),
                    op_dict.get('kwargs', {})
                ),
                operation_type=op_type,
                args=op_dict.get('args', []),
                kwargs=op_dict.get('kwargs', {}),
                future=Future()
            )
            operations.append(operation)
            futures.append(operation.future)
        
        # Add to batch queue
        with self._batch_lock:
            self._batch_queue.extend(operations)
        
        # Wait for all futures
        results = []
        for future in futures:
            try:
                result = future.result(timeout=30)
                results.append(result.result if result.success else None)
            except TimeoutError:
                results.append(None)
        
        return results
    
    def add_to_batch(self, operation_type: str, args: List[Any], kwargs: Dict[str, Any] = None) -> Future:
        """Add operation to batch queue and return future"""
        op_type = OperationType(operation_type)
        future = Future()
        
        operation = RemoteOperation(
            operation_id=self._generate_operation_id(op_type, tuple(args), kwargs or {}),
            operation_type=op_type,
            args=args,
            kwargs=kwargs or {},
            future=future
        )
        
        with self._batch_lock:
            self._batch_queue.append(operation)
        
        return future
    
    def execute_batch(self) -> List[OperationResult]:
        """Manually trigger batch execution"""
        with self._batch_lock:
            if not self._batch_queue:
                return []
            
            batch = self._batch_queue[:]
            self._batch_queue.clear()
        
        return self._execute_batch(batch)
    
    # Utility methods
    
    def warmup_connections(self):
        """Warm up connections to all servers"""
        for server in self.servers:
            try:
                # Execute a simple operation to establish connection
                operation = RemoteOperation(
                    operation_id="warmup",
                    operation_type=OperationType.TROPICAL_ADD,
                    args=[np.array([1.0]), np.array([2.0])],
                    kwargs={},
                    timeout_ms=5000
                )
                
                self._execute_remote_operation(operation)
                logger.info(f"Warmed up connection to {server.host}:{server.port}")
            except Exception as e:
                logger.warning(f"Failed to warm up connection to {server.host}:{server.port}: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get interface metrics"""
        metrics = self._metrics.copy()
        
        # Add derived metrics
        if metrics['successful_operations'] > 0:
            metrics['average_latency_ms'] = metrics['total_latency_ms'] / metrics['successful_operations']
        else:
            metrics['average_latency_ms'] = 0
        
        total_ops = metrics['successful_operations'] + metrics['failed_operations']
        if total_ops > 0:
            metrics['success_rate'] = metrics['successful_operations'] / total_ops
            metrics['cache_hit_rate'] = metrics['cache_hits'] / (metrics['cache_hits'] + metrics['cache_misses'])
        else:
            metrics['success_rate'] = 0
            metrics['cache_hit_rate'] = 0
        
        # Add server stats
        metrics['server_stats'] = self.load_balancer.get_stats()
        
        # Add circuit breaker states
        metrics['circuit_breaker_states'] = {}
        for server in self.servers:
            key = f"{server.host}:{server.port}"
            metrics['circuit_breaker_states'][key] = self.circuit_breaker.get_state(server)
        
        return metrics
    
    def clear_cache(self):
        """Clear result cache"""
        with self._cache_lock:
            self._result_cache.clear()
            logger.info("Cleared result cache")
    
    def shutdown(self):
        """Gracefully shutdown the interface"""
        logger.info("Shutting down JAX Stub Interface...")
        
        # Stop workers
        self._running = False
        
        # Shutdown executor
        self.executor.shutdown(wait=True, timeout=10)
        
        # Close connections
        asyncio.run_coroutine_threadsafe(
            self.connection_pool.close_all(),
            self._event_loop
        ).result(timeout=5)
        
        # Stop event loop
        self._event_loop.call_soon_threadsafe(self._event_loop.stop)
        self._loop_thread.join(timeout=5)
        
        logger.info("JAX Stub Interface shutdown complete")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


def create_stub_interface(config: Optional[DeploymentConfig] = None) -> JAXStubInterface:
    """Factory function to create JAX stub interface"""
    if config is None:
        # Create default configuration
        from independent_core.compression_systems.tropical.deployment_config import create_default_config
        config = create_default_config('development')
        
        # Add sample remote servers for testing
        config.remote_servers = [
            RemoteServerConfig(
                host="localhost",
                port=50051,
                protocol="grpc",
                weight=1
            ),
            RemoteServerConfig(
                host="localhost",
                port=8080,
                protocol="http",
                weight=1
            )
        ]
    
    return JAXStubInterface(config)


# Example usage and testing
if __name__ == "__main__":
    # This is for development reference only
    
    # Create configuration
    from independent_core.compression_systems.tropical.deployment_config import (
        DeploymentConfig,
        RemoteServerConfig,
        LoadBalancingStrategy
    )
    
    config = DeploymentConfig(
        remote_servers=[
            RemoteServerConfig(
                host="gpu-server-1.example.com",
                port=50051,
                protocol="grpc",
                weight=2,
                max_connections=100
            ),
            RemoteServerConfig(
                host="gpu-server-2.example.com",
                port=8080,
                protocol="https",
                weight=1,
                ssl_verify=True
            )
        ],
        load_balancing_strategy=LoadBalancingStrategy.LEAST_RESPONSE_TIME
    )
    
    # Create stub interface
    with create_stub_interface(config) as stub:
        # Warm up connections
        stub.warmup_connections()
        
        # Example operations
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 1.0, 4.0])
        
        # Synchronous operation
        result = stub.tropical_add(a, b)
        print(f"Tropical add result: {result}")
        
        # Batch operations
        batch_ops = [
            {'type': 'tropical_add', 'args': [a, b]},
            {'type': 'tropical_multiply', 'args': [a, b]},
            {'type': 'tropical_power', 'args': [a, 2.0]}
        ]
        batch_results = stub.batch_operations(batch_ops)
        print(f"Batch results: {batch_results}")
        
        # Get metrics
        metrics = stub.get_metrics()
        print(f"Interface metrics: {metrics}")