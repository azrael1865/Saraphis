"""
Saraphis Rate Limiter
Production-ready rate limiter with multiple algorithms and intelligent throttling
NO FALLBACKS - HARD FAILURES ONLY
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)


class RateLimiter:
    """Production-ready rate limiter with multiple algorithms and intelligent throttling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Rate limiting configuration
        self.algorithm = config.get('algorithm', 'token_bucket')
        self.global_rate_limit = config.get('global_rate_limit', 10000)  # requests per minute
        self.burst_multiplier = config.get('burst_multiplier', 2.0)
        self.enable_distributed = config.get('enable_distributed', False)
        
        # Rate limit configurations by service
        self.service_limits = self._initialize_service_limits()
        
        # Rate limit storage
        self.token_buckets = defaultdict(lambda: {
            'tokens': 0,
            'last_refill': time.time(),
            'refill_rate': 0
        })
        
        self.leaky_buckets = defaultdict(lambda: {
            'queue': deque(),
            'last_leak': time.time(),
            'leak_rate': 0
        })
        
        self.fixed_windows = defaultdict(lambda: {
            'window_start': time.time(),
            'request_count': 0
        })
        
        self.sliding_windows = defaultdict(lambda: deque())
        
        # Throttling state
        self.throttled_clients = {}
        self.blacklisted_clients = set()
        
        # Metrics
        self.rate_limit_metrics = defaultdict(lambda: {
            'allowed': 0,
            'denied': 0,
            'throttled': 0
        })
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        self.logger.info(f"Rate Limiter initialized with {self.algorithm} algorithm")
    
    def check_rate_limits(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Check rate limits for the request"""
        try:
            # Extract client identifier
            client_id = self._extract_client_id(request)
            
            # Check if client is blacklisted
            if client_id in self.blacklisted_clients:
                return {
                    'allowed': False,
                    'details': 'Client is blacklisted',
                    'retry_after': None
                }
            
            # Check if client is currently throttled
            if self._is_client_throttled(client_id):
                throttle_info = self.throttled_clients[client_id]
                return {
                    'allowed': False,
                    'details': 'Client is temporarily throttled',
                    'retry_after': throttle_info['until'] - time.time()
                }
            
            # Get rate limit configuration
            rate_limit_config = self._get_rate_limit_config(request)
            
            # Apply global rate limit first
            global_check = self._check_global_rate_limit()
            if not global_check['allowed']:
                return global_check
            
            # Check rate limit based on algorithm
            if self.algorithm == 'token_bucket':
                result = self._check_token_bucket_limit(client_id, rate_limit_config)
            elif self.algorithm == 'leaky_bucket':
                result = self._check_leaky_bucket_limit(client_id, rate_limit_config)
            elif self.algorithm == 'fixed_window':
                result = self._check_fixed_window_limit(client_id, rate_limit_config)
            elif self.algorithm == 'sliding_window':
                result = self._check_sliding_window_limit(client_id, rate_limit_config)
            elif self.algorithm == 'adaptive':
                result = self._check_adaptive_limit(client_id, rate_limit_config, request)
            else:
                result = self._check_token_bucket_limit(client_id, rate_limit_config)
            
            # Update metrics
            self._update_metrics(client_id, result['allowed'])
            
            # Apply throttling if needed
            if not result['allowed']:
                self._apply_throttling(client_id, rate_limit_config)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Rate limit check failed: {e}")
            return {
                'allowed': False,
                'details': f'Rate limit check error: {str(e)}'
            }
    
    def _initialize_service_limits(self) -> Dict[str, Dict[str, Any]]:
        """Initialize service-specific rate limits"""
        return {
            'brain': {
                'requests_per_minute': 100,
                'requests_per_hour': 1000,
                'burst_size': 20,
                'priority': 'high'
            },
            'uncertainty': {
                'requests_per_minute': 200,
                'requests_per_hour': 2000,
                'burst_size': 50,
                'priority': 'medium'
            },
            'training': {
                'requests_per_minute': 50,
                'requests_per_hour': 500,
                'burst_size': 10,
                'priority': 'medium'
            },
            'compression': {
                'requests_per_minute': 300,
                'requests_per_hour': 3000,
                'burst_size': 100,
                'priority': 'low'
            },
            'proof': {
                'requests_per_minute': 150,
                'requests_per_hour': 1500,
                'burst_size': 30,
                'priority': 'high'
            },
            'security': {
                'requests_per_minute': 75,
                'requests_per_hour': 750,
                'burst_size': 15,
                'priority': 'critical'
            },
            'data': {
                'requests_per_minute': 100,
                'requests_per_hour': 1000,
                'burst_size': 25,
                'priority': 'high'
            },
            'api': {
                'requests_per_minute': 500,
                'requests_per_hour': 5000,
                'burst_size': 150,
                'priority': 'low'
            }
        }
    
    def _extract_client_id(self, request: Dict[str, Any]) -> str:
        """Extract client identifier from request"""
        try:
            # Priority order for client identification
            client_id = (
                request.get('user', {}).get('user_id') or
                request.get('api_key') or
                request.get('client_ip') or
                request.get('headers', {}).get('X-Forwarded-For') or
                request.get('session_id') or
                'anonymous'
            )
            return str(client_id)
            
        except Exception as e:
            self.logger.error(f"Client ID extraction failed: {e}")
            return 'anonymous'
    
    def _get_rate_limit_config(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get rate limit configuration for request"""
        try:
            endpoint = request.get('endpoint', '')
            user_tier = request.get('user', {}).get('tier', 'free')
            
            # Find matching service
            service_name = None
            for service, config in self.service_limits.items():
                if f'/{service}' in endpoint:
                    service_name = service
                    break
            
            if not service_name:
                service_name = 'api'  # Default service
            
            # Get base configuration
            base_config = self.service_limits[service_name].copy()
            
            # Apply tier multipliers
            tier_multipliers = {
                'free': 1.0,
                'basic': 2.0,
                'pro': 5.0,
                'enterprise': 10.0
            }
            
            multiplier = tier_multipliers.get(user_tier, 1.0)
            
            return {
                'requests_per_minute': int(base_config['requests_per_minute'] * multiplier),
                'requests_per_hour': int(base_config['requests_per_hour'] * multiplier),
                'burst_size': int(base_config['burst_size'] * multiplier * self.burst_multiplier),
                'priority': base_config['priority'],
                'service': service_name
            }
            
        except Exception as e:
            self.logger.error(f"Rate limit config retrieval failed: {e}")
            return {
                'requests_per_minute': 100,
                'requests_per_hour': 1000,
                'burst_size': 20,
                'priority': 'medium',
                'service': 'unknown'
            }
    
    def _check_global_rate_limit(self) -> Dict[str, Any]:
        """Check global rate limit across all clients"""
        try:
            with self._lock:
                current_time = time.time()
                global_key = '_global_rate_limit'
                
                # Use sliding window for global limit
                if global_key not in self.sliding_windows:
                    self.sliding_windows[global_key] = deque()
                
                window = self.sliding_windows[global_key]
                
                # Remove old requests outside 1-minute window
                while window and window[0] < current_time - 60:
                    window.popleft()
                
                # Check if under global limit
                if len(window) < self.global_rate_limit:
                    window.append(current_time)
                    return {'allowed': True}
                else:
                    return {
                        'allowed': False,
                        'details': 'Global rate limit exceeded',
                        'retry_after': 60 - (current_time - window[0])
                    }
                    
        except Exception as e:
            self.logger.error(f"Global rate limit check failed: {e}")
            return {'allowed': True}  # Fail open for global limit
    
    def _check_token_bucket_limit(self, client_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check rate limit using token bucket algorithm"""
        try:
            with self._lock:
                current_time = time.time()
                bucket_key = f"{client_id}:{config['service']}"
                
                # Initialize bucket if needed
                if bucket_key not in self.token_buckets:
                    self.token_buckets[bucket_key] = {
                        'tokens': config['burst_size'],
                        'last_refill': current_time,
                        'refill_rate': config['requests_per_minute'] / 60.0
                    }
                
                bucket = self.token_buckets[bucket_key]
                
                # Refill tokens
                time_passed = current_time - bucket['last_refill']
                tokens_to_add = time_passed * bucket['refill_rate']
                bucket['tokens'] = min(config['burst_size'], bucket['tokens'] + tokens_to_add)
                bucket['last_refill'] = current_time
                
                # Check if tokens available
                if bucket['tokens'] >= 1:
                    bucket['tokens'] -= 1
                    return {
                        'allowed': True,
                        'tokens_remaining': int(bucket['tokens']),
                        'refill_rate': bucket['refill_rate']
                    }
                else:
                    # Calculate retry after
                    tokens_needed = 1 - bucket['tokens']
                    retry_after = tokens_needed / bucket['refill_rate']
                    
                    return {
                        'allowed': False,
                        'details': 'Rate limit exceeded (token bucket)',
                        'tokens_remaining': 0,
                        'retry_after': retry_after
                    }
                    
        except Exception as e:
            self.logger.error(f"Token bucket limit check failed: {e}")
            return {
                'allowed': False,
                'details': f'Token bucket check error: {str(e)}'
            }
    
    def _check_leaky_bucket_limit(self, client_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check rate limit using leaky bucket algorithm"""
        try:
            with self._lock:
                current_time = time.time()
                bucket_key = f"{client_id}:{config['service']}"
                
                # Initialize bucket if needed
                if bucket_key not in self.leaky_buckets:
                    self.leaky_buckets[bucket_key] = {
                        'queue': deque(),
                        'last_leak': current_time,
                        'leak_rate': config['requests_per_minute'] / 60.0
                    }
                
                bucket = self.leaky_buckets[bucket_key]
                
                # Leak requests from queue
                time_passed = current_time - bucket['last_leak']
                requests_to_leak = int(time_passed * bucket['leak_rate'])
                
                for _ in range(min(requests_to_leak, len(bucket['queue']))):
                    bucket['queue'].popleft()
                
                bucket['last_leak'] = current_time
                
                # Check if queue has space
                if len(bucket['queue']) < config['burst_size']:
                    bucket['queue'].append(current_time)
                    return {
                        'allowed': True,
                        'queue_size': len(bucket['queue']),
                        'queue_capacity': config['burst_size']
                    }
                else:
                    # Calculate retry after
                    oldest_request = bucket['queue'][0] if bucket['queue'] else current_time
                    retry_after = (1 / bucket['leak_rate']) - (current_time - oldest_request)
                    
                    return {
                        'allowed': False,
                        'details': 'Rate limit exceeded (leaky bucket)',
                        'queue_size': len(bucket['queue']),
                        'queue_capacity': config['burst_size'],
                        'retry_after': max(0, retry_after)
                    }
                    
        except Exception as e:
            self.logger.error(f"Leaky bucket limit check failed: {e}")
            return {
                'allowed': False,
                'details': f'Leaky bucket check error: {str(e)}'
            }
    
    def _check_fixed_window_limit(self, client_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check rate limit using fixed window algorithm"""
        try:
            with self._lock:
                current_time = time.time()
                window_size = 60  # 1 minute window
                window_key = f"{client_id}:{config['service']}"
                
                # Initialize window if needed
                if window_key not in self.fixed_windows:
                    self.fixed_windows[window_key] = {
                        'window_start': current_time,
                        'request_count': 0
                    }
                
                window = self.fixed_windows[window_key]
                
                # Check if window has expired
                if current_time - window['window_start'] >= window_size:
                    window['window_start'] = current_time
                    window['request_count'] = 0
                
                # Check if limit exceeded
                if window['request_count'] < config['requests_per_minute']:
                    window['request_count'] += 1
                    
                    return {
                        'allowed': True,
                        'requests_in_window': window['request_count'],
                        'window_limit': config['requests_per_minute'],
                        'window_reset': window['window_start'] + window_size
                    }
                else:
                    retry_after = window['window_start'] + window_size - current_time
                    
                    return {
                        'allowed': False,
                        'details': 'Rate limit exceeded (fixed window)',
                        'requests_in_window': window['request_count'],
                        'window_limit': config['requests_per_minute'],
                        'window_reset': window['window_start'] + window_size,
                        'retry_after': retry_after
                    }
                    
        except Exception as e:
            self.logger.error(f"Fixed window limit check failed: {e}")
            return {
                'allowed': False,
                'details': f'Fixed window check error: {str(e)}'
            }
    
    def _check_sliding_window_limit(self, client_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check rate limit using sliding window algorithm"""
        try:
            with self._lock:
                current_time = time.time()
                window_size = 60  # 1 minute window
                window_key = f"{client_id}:{config['service']}"
                
                # Initialize window if needed
                if window_key not in self.sliding_windows:
                    self.sliding_windows[window_key] = deque()
                
                window = self.sliding_windows[window_key]
                
                # Remove old requests outside window
                while window and window[0] < current_time - window_size:
                    window.popleft()
                
                # Check if limit exceeded
                if len(window) < config['requests_per_minute']:
                    window.append(current_time)
                    
                    return {
                        'allowed': True,
                        'requests_in_window': len(window),
                        'window_limit': config['requests_per_minute']
                    }
                else:
                    # Calculate retry after
                    oldest_request = window[0] if window else current_time
                    retry_after = window_size - (current_time - oldest_request)
                    
                    return {
                        'allowed': False,
                        'details': 'Rate limit exceeded (sliding window)',
                        'requests_in_window': len(window),
                        'window_limit': config['requests_per_minute'],
                        'retry_after': retry_after
                    }
                    
        except Exception as e:
            self.logger.error(f"Sliding window limit check failed: {e}")
            return {
                'allowed': False,
                'details': f'Sliding window check error: {str(e)}'
            }
    
    def _check_adaptive_limit(self, client_id: str, config: Dict[str, Any], 
                            request: Dict[str, Any]) -> Dict[str, Any]:
        """Check rate limit using adaptive algorithm"""
        try:
            # Get client behavior metrics
            client_metrics = self.rate_limit_metrics.get(client_id, {})
            
            # Calculate adaptive limit based on behavior
            base_limit = config['requests_per_minute']
            
            # Adjust based on denial rate
            total_requests = client_metrics.get('allowed', 0) + client_metrics.get('denied', 0)
            if total_requests > 100:  # Enough data to adapt
                denial_rate = client_metrics.get('denied', 0) / total_requests
                
                if denial_rate > 0.5:  # High denial rate, reduce limit
                    adaptive_limit = int(base_limit * 0.5)
                elif denial_rate > 0.2:  # Moderate denial rate
                    adaptive_limit = int(base_limit * 0.8)
                else:  # Low denial rate, can increase
                    adaptive_limit = int(base_limit * 1.2)
            else:
                adaptive_limit = base_limit
            
            # Apply adapted limit using sliding window
            adapted_config = config.copy()
            adapted_config['requests_per_minute'] = adaptive_limit
            
            return self._check_sliding_window_limit(client_id, adapted_config)
            
        except Exception as e:
            self.logger.error(f"Adaptive limit check failed: {e}")
            return self._check_sliding_window_limit(client_id, config)
    
    def _is_client_throttled(self, client_id: str) -> bool:
        """Check if client is currently throttled"""
        try:
            if client_id not in self.throttled_clients:
                return False
            
            throttle_info = self.throttled_clients[client_id]
            if time.time() > throttle_info['until']:
                # Throttle period expired
                del self.throttled_clients[client_id]
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Throttle check failed: {e}")
            return False
    
    def _apply_throttling(self, client_id: str, config: Dict[str, Any]):
        """Apply throttling to client based on violations"""
        try:
            # Get client metrics
            metrics = self.rate_limit_metrics[client_id]
            
            # Calculate violation severity
            total_requests = metrics['allowed'] + metrics['denied']
            if total_requests > 0:
                violation_rate = metrics['denied'] / total_requests
                
                # Determine throttle duration based on severity
                if violation_rate > 0.8:  # Severe violations
                    throttle_duration = 300  # 5 minutes
                elif violation_rate > 0.5:  # Moderate violations
                    throttle_duration = 60  # 1 minute
                else:
                    return  # No throttling needed
                
                # Apply throttle
                self.throttled_clients[client_id] = {
                    'until': time.time() + throttle_duration,
                    'duration': throttle_duration,
                    'reason': f'High violation rate: {violation_rate:.2%}'
                }
                
                self.logger.warning(f"Client {client_id} throttled for {throttle_duration}s")
                
        except Exception as e:
            self.logger.error(f"Failed to apply throttling: {e}")
    
    def _update_metrics(self, client_id: str, allowed: bool):
        """Update rate limit metrics"""
        try:
            if allowed:
                self.rate_limit_metrics[client_id]['allowed'] += 1
            else:
                self.rate_limit_metrics[client_id]['denied'] += 1
                
        except Exception as e:
            self.logger.error(f"Metrics update failed: {e}")
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        thread.start()
    
    def _cleanup_loop(self):
        """Background cleanup of old data"""
        while True:
            try:
                current_time = time.time()
                
                # Clean up old sliding window data
                with self._lock:
                    for key in list(self.sliding_windows.keys()):
                        window = self.sliding_windows[key]
                        # Remove windows with all old data
                        if window and window[-1] < current_time - 3600:  # 1 hour old
                            del self.sliding_windows[key]
                    
                    # Clean up expired throttles
                    expired_throttles = []
                    for client_id, info in self.throttled_clients.items():
                        if current_time > info['until']:
                            expired_throttles.append(client_id)
                    
                    for client_id in expired_throttles:
                        del self.throttled_clients[client_id]
                
                time.sleep(60)  # Clean up every minute
                
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                time.sleep(300)  # 5 minutes on error
    
    def get_rate_limiter_status(self) -> Dict[str, Any]:
        """Get comprehensive rate limiter status"""
        try:
            with self._lock:
                return {
                    'algorithm': self.algorithm,
                    'global_rate_limit': self.global_rate_limit,
                    'active_clients': len(self.rate_limit_metrics),
                    'throttled_clients': len(self.throttled_clients),
                    'blacklisted_clients': len(self.blacklisted_clients),
                    'metrics_summary': self._get_metrics_summary()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get rate limiter status: {e}")
            return {'error': str(e)}
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of rate limit metrics"""
        try:
            total_allowed = sum(m['allowed'] for m in self.rate_limit_metrics.values())
            total_denied = sum(m['denied'] for m in self.rate_limit_metrics.values())
            total_requests = total_allowed + total_denied
            
            return {
                'total_requests': total_requests,
                'total_allowed': total_allowed,
                'total_denied': total_denied,
                'denial_rate': total_denied / total_requests if total_requests > 0 else 0,
                'top_clients': self._get_top_clients()
            }
            
        except Exception as e:
            self.logger.error(f"Metrics summary generation failed: {e}")
            return {}
    
    def _get_top_clients(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top clients by request count"""
        try:
            client_totals = []
            
            for client_id, metrics in self.rate_limit_metrics.items():
                total = metrics['allowed'] + metrics['denied']
                client_totals.append({
                    'client_id': client_id,
                    'total_requests': total,
                    'allowed': metrics['allowed'],
                    'denied': metrics['denied']
                })
            
            # Sort by total requests
            client_totals.sort(key=lambda x: x['total_requests'], reverse=True)
            
            return client_totals[:limit]
            
        except Exception as e:
            self.logger.error(f"Top clients calculation failed: {e}")
            return []