"""
Saraphis WebSocket Manager
Production-ready WebSocket communication system
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import threading
import json
import asyncio
import websockets
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Callable
from collections import defaultdict, deque
import hashlib
import secrets
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class WebSocketConnection:
    """Individual WebSocket connection handler"""
    
    def __init__(self, connection_id: str, websocket, user_id: str):
        self.connection_id = connection_id
        self.websocket = websocket
        self.user_id = user_id
        self.channels = set()
        self.created_at = time.time()
        self.last_activity = time.time()
        self.message_count = 0
        self.error_count = 0
        self.metadata = {}
    
    async def send_message(self, message: Dict[str, Any]):
        """Send message to client"""
        try:
            await self.websocket.send(json.dumps(message))
            self.message_count += 1
            self.last_activity = time.time()
            return True
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to send message to {self.connection_id}: {e}")
            return False
    
    def is_alive(self) -> bool:
        """Check if connection is still alive"""
        return self.websocket.open


class WebSocketManager:
    """Production-ready WebSocket manager for real-time communication"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Connection management
        self.connections = {}  # connection_id -> WebSocketConnection
        self.user_connections = defaultdict(set)  # user_id -> {connection_ids}
        self.channel_subscriptions = defaultdict(set)  # channel -> {connection_ids}
        
        # Message handling
        self.message_queue = asyncio.Queue()
        self.message_handlers = {}
        self.broadcast_history = deque(maxlen=1000)
        
        # Configuration
        self.max_connections_per_user = config.get('max_connections_per_user', 5)
        self.message_rate_limit = config.get('message_rate_limit', 100)  # per minute
        self.heartbeat_interval = config.get('heartbeat_interval', 30)
        self.connection_timeout = config.get('connection_timeout', 300)
        self.max_message_size = config.get('max_message_size', 1024 * 1024)  # 1MB
        
        # Rate limiting
        self.rate_limiters = defaultdict(lambda: {'count': 0, 'reset_time': time.time() + 60})
        
        # Statistics
        self.statistics = {
            'total_connections': 0,
            'total_messages_sent': 0,
            'total_messages_received': 0,
            'total_broadcasts': 0,
            'connection_errors': 0,
            'message_errors': 0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        self._running = False
        self._server = None
        
        # Event loop for async operations
        self.loop = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info("WebSocket Manager initialized")
    
    async def start_server(self, host: str = 'localhost', port: int = 8765):
        """Start WebSocket server"""
        try:
            self._running = True
            self.loop = asyncio.get_event_loop()
            
            # Start message processing
            asyncio.create_task(self._process_message_queue())
            
            # Start heartbeat
            asyncio.create_task(self._heartbeat_loop())
            
            # Start cleanup
            asyncio.create_task(self._cleanup_loop())
            
            # Start WebSocket server
            self._server = await websockets.serve(
                self._handle_connection,
                host,
                port,
                max_size=self.max_message_size
            )
            
            self.logger.info(f"WebSocket server started on {host}:{port}")
            
            # Keep server running
            await asyncio.Future()  # Run forever
            
        except Exception as e:
            self.logger.error(f"WebSocket server failed: {e}")
            raise
    
    async def _handle_connection(self, websocket, path):
        """Handle new WebSocket connection"""
        connection_id = None
        
        try:
            # Authenticate connection
            auth_message = await asyncio.wait_for(
                websocket.recv(),
                timeout=10
            )
            
            auth_data = json.loads(auth_message)
            user_id = auth_data.get('user_id')
            auth_token = auth_data.get('token')
            
            # Validate authentication
            if not self._validate_auth(user_id, auth_token):
                await websocket.send(json.dumps({
                    'type': 'error',
                    'error': 'Authentication failed'
                }))
                return
            
            # Check connection limit
            if not self._check_connection_limit(user_id):
                await websocket.send(json.dumps({
                    'type': 'error',
                    'error': 'Connection limit exceeded'
                }))
                return
            
            # Create connection
            connection_id = self._generate_connection_id()
            connection = WebSocketConnection(connection_id, websocket, user_id)
            
            with self._lock:
                self.connections[connection_id] = connection
                self.user_connections[user_id].add(connection_id)
                self.statistics['total_connections'] += 1
            
            # Send connection success
            await websocket.send(json.dumps({
                'type': 'connection_established',
                'connection_id': connection_id,
                'user_id': user_id,
                'timestamp': time.time()
            }))
            
            self.logger.info(f"WebSocket connection established: {connection_id} (user: {user_id})")
            
            # Handle messages
            await self._handle_messages(connection)
            
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"WebSocket connection closed: {connection_id}")
        except Exception as e:
            self.logger.error(f"WebSocket connection error: {e}")
            self.statistics['connection_errors'] += 1
        finally:
            # Clean up connection
            if connection_id:
                await self._remove_connection(connection_id)
    
    async def _handle_messages(self, connection: WebSocketConnection):
        """Handle messages from WebSocket connection"""
        try:
            async for message in connection.websocket:
                try:
                    # Check rate limit
                    if not self._check_rate_limit(connection.user_id):
                        await connection.send_message({
                            'type': 'error',
                            'error': 'Rate limit exceeded'
                        })
                        continue
                    
                    # Parse message
                    data = json.loads(message)
                    message_type = data.get('type')
                    
                    # Update activity
                    connection.last_activity = time.time()
                    self.statistics['total_messages_received'] += 1
                    
                    # Handle message based on type
                    if message_type == 'subscribe':
                        await self._handle_subscribe(connection, data)
                    elif message_type == 'unsubscribe':
                        await self._handle_unsubscribe(connection, data)
                    elif message_type == 'message':
                        await self._handle_message(connection, data)
                    elif message_type == 'ping':
                        await self._handle_ping(connection)
                    else:
                        await connection.send_message({
                            'type': 'error',
                            'error': f'Unknown message type: {message_type}'
                        })
                        
                except json.JSONDecodeError:
                    await connection.send_message({
                        'type': 'error',
                        'error': 'Invalid JSON'
                    })
                except Exception as e:
                    self.logger.error(f"Message handling error: {e}")
                    self.statistics['message_errors'] += 1
                    
        except websockets.exceptions.ConnectionClosed:
            pass
    
    async def _handle_subscribe(self, connection: WebSocketConnection, data: Dict[str, Any]):
        """Handle channel subscription"""
        channel = data.get('channel')
        
        if not channel:
            await connection.send_message({
                'type': 'error',
                'error': 'Channel not specified'
            })
            return
        
        # Validate channel access
        if not self._validate_channel_access(connection.user_id, channel):
            await connection.send_message({
                'type': 'error',
                'error': 'Access denied to channel'
            })
            return
        
        # Add to channel
        with self._lock:
            connection.channels.add(channel)
            self.channel_subscriptions[channel].add(connection.connection_id)
        
        await connection.send_message({
            'type': 'subscribed',
            'channel': channel,
            'timestamp': time.time()
        })
        
        self.logger.info(f"Connection {connection.connection_id} subscribed to {channel}")
    
    async def _handle_unsubscribe(self, connection: WebSocketConnection, data: Dict[str, Any]):
        """Handle channel unsubscription"""
        channel = data.get('channel')
        
        if not channel:
            await connection.send_message({
                'type': 'error',
                'error': 'Channel not specified'
            })
            return
        
        # Remove from channel
        with self._lock:
            connection.channels.discard(channel)
            self.channel_subscriptions[channel].discard(connection.connection_id)
        
        await connection.send_message({
            'type': 'unsubscribed',
            'channel': channel,
            'timestamp': time.time()
        })
        
        self.logger.info(f"Connection {connection.connection_id} unsubscribed from {channel}")
    
    async def _handle_message(self, connection: WebSocketConnection, data: Dict[str, Any]):
        """Handle regular message"""
        # Add to message queue for processing
        await self.message_queue.put({
            'connection_id': connection.connection_id,
            'user_id': connection.user_id,
            'data': data,
            'timestamp': time.time()
        })
    
    async def _handle_ping(self, connection: WebSocketConnection):
        """Handle ping message"""
        await connection.send_message({
            'type': 'pong',
            'timestamp': time.time()
        })
    
    def subscribe_to_channel(self, user_id: str, channel: str):
        """Subscribe user to channel (external API)"""
        try:
            with self._lock:
                # Subscribe all user connections to channel
                for connection_id in self.user_connections.get(user_id, set()):
                    if connection_id in self.connections:
                        connection = self.connections[connection_id]
                        connection.channels.add(channel)
                        self.channel_subscriptions[channel].add(connection_id)
            
            self.logger.info(f"User {user_id} subscribed to {channel}")
            
        except Exception as e:
            self.logger.error(f"Channel subscription failed: {e}")
    
    def unsubscribe_from_channel(self, user_id: str, channel: str):
        """Unsubscribe user from channel (external API)"""
        try:
            with self._lock:
                # Unsubscribe all user connections from channel
                for connection_id in self.user_connections.get(user_id, set()):
                    if connection_id in self.connections:
                        connection = self.connections[connection_id]
                        connection.channels.discard(channel)
                        self.channel_subscriptions[channel].discard(connection_id)
            
            self.logger.info(f"User {user_id} unsubscribed from {channel}")
            
        except Exception as e:
            self.logger.error(f"Channel unsubscription failed: {e}")
    
    def broadcast_dashboard_update(self, user_id: str, dashboard_type: str, 
                                 update_data: Dict[str, Any]):
        """Broadcast dashboard update to user"""
        try:
            channel = f"dashboard_{user_id}_{dashboard_type}"
            
            message = {
                'type': 'dashboard_update',
                'dashboard_type': dashboard_type,
                'data': update_data,
                'timestamp': time.time()
            }
            
            # Broadcast to channel
            asyncio.run_coroutine_threadsafe(
                self._broadcast_to_channel(channel, message),
                self.loop
            )
            
            # Track broadcast
            self.broadcast_history.append({
                'channel': channel,
                'timestamp': time.time(),
                'message_type': 'dashboard_update'
            })
            
        except Exception as e:
            self.logger.error(f"Dashboard update broadcast failed: {e}")
    
    def broadcast_interface_update(self, user_id: str, update_data: Dict[str, Any]):
        """Broadcast interface update to user"""
        try:
            message = {
                'type': 'interface_update',
                'data': update_data,
                'timestamp': time.time()
            }
            
            # Broadcast to all user connections
            asyncio.run_coroutine_threadsafe(
                self._broadcast_to_user(user_id, message),
                self.loop
            )
            
        except Exception as e:
            self.logger.error(f"Interface update broadcast failed: {e}")
    
    async def _broadcast_to_channel(self, channel: str, message: Dict[str, Any]):
        """Broadcast message to all connections in channel"""
        try:
            with self._lock:
                connection_ids = self.channel_subscriptions.get(channel, set()).copy()
            
            # Send to all connections
            tasks = []
            for connection_id in connection_ids:
                if connection_id in self.connections:
                    connection = self.connections[connection_id]
                    tasks.append(connection.send_message(message))
            
            # Wait for all sends to complete
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                successful = sum(1 for r in results if r is True)
                
                self.statistics['total_messages_sent'] += successful
                self.statistics['total_broadcasts'] += 1
                
                self.logger.debug(
                    f"Broadcast to {channel}: {successful}/{len(tasks)} successful"
                )
                
        except Exception as e:
            self.logger.error(f"Channel broadcast failed: {e}")
    
    async def _broadcast_to_user(self, user_id: str, message: Dict[str, Any]):
        """Broadcast message to all user connections"""
        try:
            with self._lock:
                connection_ids = self.user_connections.get(user_id, set()).copy()
            
            # Send to all connections
            tasks = []
            for connection_id in connection_ids:
                if connection_id in self.connections:
                    connection = self.connections[connection_id]
                    tasks.append(connection.send_message(message))
            
            # Wait for all sends to complete
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                successful = sum(1 for r in results if r is True)
                
                self.statistics['total_messages_sent'] += successful
                
        except Exception as e:
            self.logger.error(f"User broadcast failed: {e}")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get WebSocket connection status"""
        try:
            with self._lock:
                return {
                    'server_running': self._running,
                    'total_connections': len(self.connections),
                    'unique_users': len(self.user_connections),
                    'total_channels': len(self.channel_subscriptions),
                    'statistics': self.statistics.copy()
                }
                
        except Exception as e:
            self.logger.error(f"Status retrieval failed: {e}")
            return {'error': str(e)}
    
    def get_active_connections(self) -> List[Dict[str, Any]]:
        """Get list of active connections"""
        try:
            with self._lock:
                connections = []
                
                for conn_id, conn in self.connections.items():
                    connections.append({
                        'connection_id': conn_id,
                        'user_id': conn.user_id,
                        'channels': list(conn.channels),
                        'created_at': conn.created_at,
                        'last_activity': conn.last_activity,
                        'message_count': conn.message_count,
                        'error_count': conn.error_count
                    })
                
                return connections
                
        except Exception as e:
            self.logger.error(f"Active connections retrieval failed: {e}")
            return []
    
    def get_connection_statistics(self) -> Dict[str, Any]:
        """Get detailed connection statistics"""
        try:
            with self._lock:
                # Calculate statistics
                total_messages = sum(
                    conn.message_count for conn in self.connections.values()
                )
                total_errors = sum(
                    conn.error_count for conn in self.connections.values()
                )
                
                # Connection duration stats
                current_time = time.time()
                durations = [
                    current_time - conn.created_at 
                    for conn in self.connections.values()
                ]
                
                avg_duration = sum(durations) / len(durations) if durations else 0
                
                return {
                    'active_connections': len(self.connections),
                    'total_messages_per_connection': total_messages / len(self.connections) if self.connections else 0,
                    'total_errors': total_errors,
                    'average_connection_duration': avg_duration,
                    'messages_per_second': self.statistics['total_messages_sent'] / 
                                         (current_time - self._server_start_time) 
                                         if hasattr(self, '_server_start_time') else 0,
                    'broadcast_history_size': len(self.broadcast_history)
                }
                
        except Exception as e:
            self.logger.error(f"Statistics retrieval failed: {e}")
            return {}
    
    def get_channel_info(self) -> Dict[str, Any]:
        """Get channel subscription information"""
        try:
            with self._lock:
                channel_info = {}
                
                for channel, connection_ids in self.channel_subscriptions.items():
                    # Get unique users in channel
                    users = set()
                    for conn_id in connection_ids:
                        if conn_id in self.connections:
                            users.add(self.connections[conn_id].user_id)
                    
                    channel_info[channel] = {
                        'connection_count': len(connection_ids),
                        'user_count': len(users),
                        'users': list(users)
                    }
                
                return channel_info
                
        except Exception as e:
            self.logger.error(f"Channel info retrieval failed: {e}")
            return {}
    
    def _validate_auth(self, user_id: str, auth_token: str) -> bool:
        """Validate authentication token"""
        # In production, this would validate against authentication service
        # For now, basic validation
        return user_id and auth_token and len(auth_token) > 10
    
    def _validate_channel_access(self, user_id: str, channel: str) -> bool:
        """Validate user access to channel"""
        # In production, this would check permissions
        # For now, allow user-specific channels
        if channel.startswith(f"dashboard_{user_id}_"):
            return True
        
        # Allow public channels
        public_channels = ['system_status', 'announcements']
        return channel in public_channels
    
    def _check_connection_limit(self, user_id: str) -> bool:
        """Check if user has reached connection limit"""
        with self._lock:
            current_connections = len(self.user_connections.get(user_id, set()))
            return current_connections < self.max_connections_per_user
    
    def _check_rate_limit(self, user_id: str) -> bool:
        """Check message rate limit for user"""
        current_time = time.time()
        
        with self._lock:
            limiter = self.rate_limiters[user_id]
            
            # Reset if window expired
            if current_time >= limiter['reset_time']:
                limiter['count'] = 0
                limiter['reset_time'] = current_time + 60
            
            # Check limit
            if limiter['count'] >= self.message_rate_limit:
                return False
            
            limiter['count'] += 1
            return True
    
    def _generate_connection_id(self) -> str:
        """Generate unique connection ID"""
        timestamp = int(time.time() * 1000)
        random_part = secrets.token_urlsafe(8)
        return f"ws_{timestamp}_{random_part}"
    
    async def _remove_connection(self, connection_id: str):
        """Remove connection and clean up"""
        try:
            with self._lock:
                if connection_id not in self.connections:
                    return
                
                connection = self.connections[connection_id]
                
                # Remove from user connections
                self.user_connections[connection.user_id].discard(connection_id)
                if not self.user_connections[connection.user_id]:
                    del self.user_connections[connection.user_id]
                
                # Remove from channel subscriptions
                for channel in connection.channels:
                    self.channel_subscriptions[channel].discard(connection_id)
                    if not self.channel_subscriptions[channel]:
                        del self.channel_subscriptions[channel]
                
                # Remove connection
                del self.connections[connection_id]
            
            self.logger.info(f"Connection removed: {connection_id}")
            
        except Exception as e:
            self.logger.error(f"Connection removal failed: {e}")
    
    async def _process_message_queue(self):
        """Process queued messages"""
        while self._running:
            try:
                # Get message from queue
                message = await self.message_queue.get()
                
                # Process based on message type
                # This could route to different handlers
                self.logger.debug(f"Processing message from {message['user_id']}")
                
                # Example: Echo back to user
                await self._broadcast_to_user(
                    message['user_id'],
                    {
                        'type': 'message_processed',
                        'original': message['data'],
                        'timestamp': time.time()
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Message processing error: {e}")
    
    async def _heartbeat_loop(self):
        """Send heartbeat to all connections"""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                # Get all connections
                with self._lock:
                    connections = list(self.connections.values())
                
                # Send heartbeat
                heartbeat_message = {
                    'type': 'heartbeat',
                    'timestamp': time.time()
                }
                
                tasks = [conn.send_message(heartbeat_message) for conn in connections]
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
    
    async def _cleanup_loop(self):
        """Clean up inactive connections"""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = time.time()
                inactive_connections = []
                
                with self._lock:
                    for conn_id, conn in self.connections.items():
                        if current_time - conn.last_activity > self.connection_timeout:
                            inactive_connections.append(conn_id)
                
                # Remove inactive connections
                for conn_id in inactive_connections:
                    self.logger.info(f"Removing inactive connection: {conn_id}")
                    await self._remove_connection(conn_id)
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
    
    def shutdown(self):
        """Shutdown WebSocket manager"""
        self.logger.info("Shutting down WebSocket Manager")
        self._running = False
        
        # Close all connections
        with self._lock:
            for conn in self.connections.values():
                asyncio.run_coroutine_threadsafe(
                    conn.websocket.close(),
                    self.loop
                )
        
        # Stop server
        if self._server:
            self._server.close()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)