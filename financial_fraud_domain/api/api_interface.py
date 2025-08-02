"""
API Interface for Financial Fraud Detection.
Provides RESTful endpoints for fraud detection services.
"""

import logging
import json
import time
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class APIResponseStatus(Enum):
    """API response status codes"""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    PENDING = "pending"
    TIMEOUT = "timeout"


class APIRequestType(Enum):
    """API request types"""
    SINGLE_TRANSACTION = "single_transaction"
    BATCH_TRANSACTION = "batch_transaction"
    ANALYSIS = "analysis"
    METRICS = "metrics"
    VALIDATION = "validation"


@dataclass
class APIRequest:
    """API request object"""
    request_id: str
    request_type: APIRequestType
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = None
    priority: int = 5  # 1-10, higher is more important


@dataclass
class APIResponse:
    """API response object"""
    request_id: str
    status: APIResponseStatus
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    warnings: List[str] = None
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class APIMetrics:
    """API performance metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    
    average_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    min_response_time_ms: float = float('inf')
    
    requests_by_type: Dict[APIRequestType, int] = None
    hourly_request_rate: Dict[str, int] = None
    error_types: Dict[str, int] = None
    
    def __post_init__(self):
        if self.requests_by_type is None:
            self.requests_by_type = defaultdict(int)
        if self.hourly_request_rate is None:
            self.hourly_request_rate = defaultdict(int)
        if self.error_types is None:
            self.error_types = defaultdict(int)


class FinancialFraudAPI:
    """
    API interface for fraud detection domain.
    
    Provides RESTful endpoints for fraud detection, analysis, and monitoring.
    """
    
    def __init__(self, fraud_core=None, config_manager=None):
        """
        Initialize API interface.
        
        Args:
            fraud_core: FraudDetectionCore instance
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.fraud_core = fraud_core
        self.config_manager = config_manager
        self.config = self._load_config()
        
        # Request tracking
        self.active_requests = {}
        self.request_history = []
        self.max_history_size = self.config.get('max_history_size', 10000)
        
        # Performance metrics
        self.metrics = APIMetrics()
        
        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 8)
        )
        
        # Rate limiting
        self.rate_limiter = self._initialize_rate_limiter()
        
        # Cache
        self._response_cache = {}
        self._cache_ttl = timedelta(minutes=self.config.get('cache_ttl_minutes', 5))
        
        # Storage
        self.storage_path = Path(self.config.get('storage_path', 'api_data'))
        self.storage_path.mkdir(exist_ok=True)
        
        self.logger.info("FinancialFraudAPI initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load API configuration"""
        default_config = {
            'max_workers': 8,
            'timeout_seconds': 30,
            'cache_ttl_minutes': 5,
            'max_history_size': 10000,
            'rate_limit_per_minute': 100,
            'batch_size_limit': 1000,
            'enable_caching': True,
            'enable_async': True,
            'log_requests': True,
            'log_responses': True
        }
        
        if self.config_manager:
            return self.config_manager.get_config('api', default_config)
        return default_config
    
    def _initialize_rate_limiter(self) -> Dict[str, Any]:
        """Initialize rate limiting"""
        return {
            'requests_per_minute': self.config.get('rate_limit_per_minute', 100),
            'request_times': defaultdict(list),
            'blocked_until': {}
        }
    
    # Main API Methods
    
    def detect_fraud(self, 
                    transaction_data: Union[Dict[str, Any], List[Dict[str, Any]]],
                    options: Optional[Dict[str, Any]] = None) -> APIResponse:
        """
        Detect fraud in transaction(s).
        
        Args:
            transaction_data: Single transaction or list of transactions
            options: Detection options (strategy, generate_proof, etc.)
            
        Returns:
            API response with detection results
        """
        request_id = self._generate_request_id()
        start_time = time.time()
        
        try:
            # Create API request
            request = APIRequest(
                request_id=request_id,
                request_type=(APIRequestType.SINGLE_TRANSACTION 
                            if isinstance(transaction_data, dict) 
                            else APIRequestType.BATCH_TRANSACTION),
                timestamp=datetime.now(),
                data={'transactions': transaction_data},
                metadata=options or {}
            )
            
            # Log request
            if self.config.get('log_requests', True):
                self._log_request(request)
            
            # Check rate limit
            if not self._check_rate_limit(request_id):
                return self._create_error_response(
                    request_id, "Rate limit exceeded", 
                    APIResponseStatus.ERROR
                )
            
            # Validate input
            validation_result = self._validate_transaction_input(transaction_data)
            if not validation_result['valid']:
                return self._create_error_response(
                    request_id, validation_result['error']
                )
            
            # Check cache
            if self.config.get('enable_caching', True):
                cached_response = self._get_cached_response(request)
                if cached_response:
                    self.logger.debug(f"Cache hit for request {request_id}")
                    return cached_response
            
            # Track active request
            with self._lock:
                self.active_requests[request_id] = request
            
            # Execute fraud detection
            if self.config.get('enable_async', True) and isinstance(transaction_data, list):
                # Async processing for batch
                result = self._async_detect_fraud(transaction_data, options)
            else:
                # Sync processing
                result = self._sync_detect_fraud(transaction_data, options)
            
            # Create response
            processing_time = (time.time() - start_time) * 1000
            response = APIResponse(
                request_id=request_id,
                status=APIResponseStatus.SUCCESS,
                timestamp=datetime.now(),
                data=result,
                processing_time_ms=processing_time
            )
            
            # Cache response
            if self.config.get('enable_caching', True):
                self._cache_response(request, response)
            
            # Update metrics
            self._update_metrics(response, request)
            
            # Log response
            if self.config.get('log_responses', True):
                self._log_response(response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Fraud detection failed: {e}", exc_info=True)
            processing_time = (time.time() - start_time) * 1000
            
            error_response = APIResponse(
                request_id=request_id,
                status=APIResponseStatus.ERROR,
                timestamp=datetime.now(),
                error=str(e),
                processing_time_ms=processing_time
            )
            
            self._update_metrics(error_response, None)
            return error_response
            
        finally:
            # Clean up active request
            with self._lock:
                self.active_requests.pop(request_id, None)
    
    def analyze_transaction(self, 
                          transaction_id: str,
                          transaction_data: Optional[Dict[str, Any]] = None) -> APIResponse:
        """
        Analyze a specific transaction in detail.
        
        Args:
            transaction_id: Transaction identifier
            transaction_data: Optional transaction data (if not in system)
            
        Returns:
            API response with detailed analysis
        """
        request_id = self._generate_request_id()
        start_time = time.time()
        
        try:
            # Create request
            request = APIRequest(
                request_id=request_id,
                request_type=APIRequestType.ANALYSIS,
                timestamp=datetime.now(),
                data={
                    'transaction_id': transaction_id,
                    'transaction_data': transaction_data
                }
            )
            
            # Validate
            if not transaction_id and not transaction_data:
                return self._create_error_response(
                    request_id, "Transaction ID or data required"
                )
            
            # Get transaction data if needed
            if not transaction_data:
                transaction_data = self._get_transaction_data(transaction_id)
                if not transaction_data:
                    return self._create_error_response(
                        request_id, f"Transaction {transaction_id} not found"
                    )
            
            # Perform analysis
            if self.fraud_core:
                analysis_result = self.fraud_core.analyze_transaction(transaction_data)
            else:
                # Mock analysis for testing
                analysis_result = {
                    'transaction_id': transaction_id,
                    'risk_indicators': {
                        'amount_anomaly': 0.2,
                        'velocity_risk': 0.1,
                        'merchant_risk': 0.3
                    },
                    'recommendations': ['Monitor for patterns'],
                    'detailed_analysis': 'Transaction appears normal'
                }
            
            # Create response
            processing_time = (time.time() - start_time) * 1000
            response = APIResponse(
                request_id=request_id,
                status=APIResponseStatus.SUCCESS,
                timestamp=datetime.now(),
                data=analysis_result,
                processing_time_ms=processing_time
            )
            
            self._update_metrics(response, request)
            return response
            
        except Exception as e:
            self.logger.error(f"Transaction analysis failed: {e}", exc_info=True)
            return self._create_error_response(request_id, str(e))
    
    def get_results(self, 
                   request_ids: Optional[List[str]] = None,
                   filters: Optional[Dict[str, Any]] = None) -> APIResponse:
        """
        Get results for previous requests.
        
        Args:
            request_ids: Specific request IDs to retrieve
            filters: Filter criteria (date range, status, etc.)
            
        Returns:
            API response with results
        """
        request_id = self._generate_request_id()
        
        try:
            results = []
            
            if request_ids:
                # Get specific requests
                for req_id in request_ids:
                    result = self._get_request_result(req_id)
                    if result:
                        results.append(result)
            else:
                # Get filtered results
                results = self._get_filtered_results(filters or {})
            
            response = APIResponse(
                request_id=request_id,
                status=APIResponseStatus.SUCCESS,
                timestamp=datetime.now(),
                data={
                    'count': len(results),
                    'results': results
                }
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Get results failed: {e}", exc_info=True)
            return self._create_error_response(request_id, str(e))
    
    def batch_check(self, 
                   transactions: List[Dict[str, Any]],
                   options: Optional[Dict[str, Any]] = None) -> APIResponse:
        """
        Check multiple transactions for fraud.
        
        Args:
            transactions: List of transactions to check
            options: Batch processing options
            
        Returns:
            API response with batch results
        """
        # Validate batch size
        batch_limit = self.config.get('batch_size_limit', 1000)
        if len(transactions) > batch_limit:
            return self._create_error_response(
                self._generate_request_id(),
                f"Batch size {len(transactions)} exceeds limit {batch_limit}"
            )
        
        # Use detect_fraud with batch
        return self.detect_fraud(transactions, options)
    
    def get_fraud_score(self, transaction_id: str) -> APIResponse:
        """
        Get fraud score for a specific transaction.
        
        Args:
            transaction_id: Transaction identifier
            
        Returns:
            API response with fraud score
        """
        request_id = self._generate_request_id()
        
        try:
            # Look up transaction in history
            score_data = self._get_transaction_score(transaction_id)
            
            if score_data:
                response = APIResponse(
                    request_id=request_id,
                    status=APIResponseStatus.SUCCESS,
                    timestamp=datetime.now(),
                    data=score_data
                )
            else:
                response = APIResponse(
                    request_id=request_id,
                    status=APIResponseStatus.ERROR,
                    timestamp=datetime.now(),
                    error=f"Transaction {transaction_id} not found"
                )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Get fraud score failed: {e}", exc_info=True)
            return self._create_error_response(request_id, str(e))
    
    def get_metrics(self) -> APIResponse:
        """
        Get API performance metrics.
        
        Returns:
            API response with metrics
        """
        request_id = self._generate_request_id()
        
        try:
            metrics_data = self._export_metrics()
            
            response = APIResponse(
                request_id=request_id,
                status=APIResponseStatus.SUCCESS,
                timestamp=datetime.now(),
                data=metrics_data
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Get metrics failed: {e}", exc_info=True)
            return self._create_error_response(request_id, str(e))
    
    def validate_transaction(self, transaction_data: Dict[str, Any]) -> APIResponse:
        """
        Validate transaction data format and content.
        
        Args:
            transaction_data: Transaction to validate
            
        Returns:
            API response with validation results
        """
        request_id = self._generate_request_id()
        
        try:
            validation_result = self._comprehensive_validation(transaction_data)
            
            response = APIResponse(
                request_id=request_id,
                status=(APIResponseStatus.SUCCESS if validation_result['valid'] 
                       else APIResponseStatus.ERROR),
                timestamp=datetime.now(),
                data=validation_result,
                warnings=validation_result.get('warnings', [])
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}", exc_info=True)
            return self._create_error_response(request_id, str(e))
    
    # Async processing methods
    
    async def async_detect_fraud(self, 
                               transaction_data: Union[Dict[str, Any], List[Dict[str, Any]]],
                               options: Optional[Dict[str, Any]] = None) -> APIResponse:
        """
        Async version of detect_fraud.
        
        Args:
            transaction_data: Transaction(s) to check
            options: Detection options
            
        Returns:
            API response with results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.detect_fraud,
            transaction_data,
            options
        )
    
    async def async_batch_check(self, 
                              transactions: List[Dict[str, Any]],
                              options: Optional[Dict[str, Any]] = None) -> APIResponse:
        """
        Async version of batch_check.
        
        Args:
            transactions: Transactions to check
            options: Batch options
            
        Returns:
            API response with results
        """
        return await self.async_detect_fraud(transactions, options)
    
    # Helper methods
    
    def _sync_detect_fraud(self, 
                         transaction_data: Union[Dict[str, Any], List[Dict[str, Any]]],
                         options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synchronous fraud detection"""
        if self.fraud_core:
            # Use actual fraud core
            strategy = options.get('strategy') if options else None
            generate_proof = options.get('generate_proof', True) if options else True
            
            result = self.fraud_core.detect_fraud(
                transaction_data,
                strategy=strategy,
                generate_proof=generate_proof
            )
            
            # Convert result to dict
            if isinstance(result, list):
                return {
                    'batch_results': [self._serialize_detection_result(r) for r in result],
                    'summary': self.fraud_core.aggregate_results(result)
                }
            else:
                return self._serialize_detection_result(result)
        else:
            # Mock result for testing
            if isinstance(transaction_data, list):
                return {
                    'batch_results': [
                        {
                            'transaction_id': t.get('transaction_id', f'tx_{i}'),
                            'fraud_score': np.random.random(),
                            'is_fraud': np.random.random() > 0.7,
                            'risk_level': np.random.choice(['low', 'medium', 'high'])
                        }
                        for i, t in enumerate(transaction_data)
                    ]
                }
            else:
                return {
                    'transaction_id': transaction_data.get('transaction_id', 'unknown'),
                    'fraud_score': np.random.random(),
                    'is_fraud': np.random.random() > 0.7,
                    'risk_level': np.random.choice(['low', 'medium', 'high'])
                }
    
    def _async_detect_fraud(self, 
                          transactions: List[Dict[str, Any]],
                          options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Asynchronous batch fraud detection"""
        # Split into chunks for parallel processing
        chunk_size = 50
        chunks = [transactions[i:i+chunk_size] 
                 for i in range(0, len(transactions), chunk_size)]
        
        # Process chunks in parallel
        futures = []
        for chunk in chunks:
            future = self._executor.submit(
                self._sync_detect_fraud, chunk, options
            )
            futures.append(future)
        
        # Collect results
        all_results = []
        for future in futures:
            try:
                result = future.result(timeout=self.config.get('timeout_seconds', 30))
                if 'batch_results' in result:
                    all_results.extend(result['batch_results'])
            except Exception as e:
                self.logger.error(f"Chunk processing failed: {e}")
        
        return {
            'batch_results': all_results,
            'summary': {
                'total_processed': len(all_results),
                'fraud_detected': sum(1 for r in all_results if r.get('is_fraud', False))
            }
        }
    
    def _validate_transaction_input(self, 
                                  transaction_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Validate transaction input data"""
        if isinstance(transaction_data, list):
            # Validate batch
            if not transaction_data:
                return {'valid': False, 'error': 'Empty transaction list'}
            
            errors = []
            for i, transaction in enumerate(transaction_data):
                result = self._validate_single_transaction(transaction)
                if not result['valid']:
                    errors.append(f"Transaction {i}: {result['error']}")
            
            if errors:
                return {
                    'valid': False,
                    'error': f"Validation errors: {'; '.join(errors[:5])}"
                           + (f" and {len(errors)-5} more" if len(errors) > 5 else "")
                }
            
            return {'valid': True}
        else:
            # Validate single
            return self._validate_single_transaction(transaction_data)
    
    def _validate_single_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Validate single transaction"""
        required_fields = ['amount']
        optional_fields = ['transaction_id', 'timestamp', 'user_id', 'merchant_id']
        
        # Check for empty dict
        if not transaction:
            return {'valid': False, 'error': 'Empty transaction data'}
        
        # Check required fields
        missing_fields = [f for f in required_fields if f not in transaction]
        if missing_fields:
            return {
                'valid': False,
                'error': f"Missing required fields: {missing_fields}"
            }
        
        # Validate amount
        try:
            amount = float(transaction['amount'])
            if amount < 0:
                return {'valid': False, 'error': 'Negative amount not allowed'}
        except (ValueError, TypeError):
            return {'valid': False, 'error': 'Invalid amount format'}
        
        return {'valid': True}
    
    def _comprehensive_validation(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive validation with warnings"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'field_status': {}
        }
        
        # Basic validation
        basic_result = self._validate_single_transaction(transaction)
        if not basic_result['valid']:
            result['valid'] = False
            result['errors'].append(basic_result['error'])
        
        # Field-by-field validation
        fields = {
            'transaction_id': {'type': str, 'required': False},
            'amount': {'type': (int, float), 'required': True, 'min': 0},
            'timestamp': {'type': (str, datetime), 'required': False},
            'user_id': {'type': (str, int), 'required': False},
            'merchant_id': {'type': (str, int), 'required': False},
            'merchant_name': {'type': str, 'required': False},
            'category': {'type': str, 'required': False},
            'location': {'type': str, 'required': False}
        }
        
        for field, spec in fields.items():
            if field in transaction:
                value = transaction[field]
                
                # Type check
                if not isinstance(value, spec['type']):
                    result['warnings'].append(
                        f"Field '{field}' has unexpected type {type(value).__name__}"
                    )
                    result['field_status'][field] = 'warning'
                else:
                    result['field_status'][field] = 'valid'
                
                # Range check
                if 'min' in spec and isinstance(value, (int, float)):
                    if value < spec['min']:
                        result['errors'].append(
                            f"Field '{field}' below minimum value {spec['min']}"
                        )
                        result['field_status'][field] = 'error'
                        result['valid'] = False
            elif spec.get('required', False):
                result['errors'].append(f"Missing required field '{field}'")
                result['field_status'][field] = 'missing'
                result['valid'] = False
        
        # Additional checks
        if 'timestamp' in transaction:
            try:
                ts = pd.to_datetime(transaction['timestamp'])
                if ts > datetime.now():
                    result['warnings'].append("Transaction timestamp is in the future")
            except:
                result['warnings'].append("Unable to parse timestamp")
        
        return result
    
    def _serialize_detection_result(self, result) -> Dict[str, Any]:
        """Serialize fraud detection result"""
        if hasattr(result, '__dict__'):
            # Convert object to dict
            serialized = {
                'transaction_id': result.transaction_id,
                'fraud_score': result.fraud_score,
                'is_fraud': result.is_fraud,
                'risk_level': result.risk_level.value if hasattr(result.risk_level, 'value') else result.risk_level,
                'confidence': result.confidence,
                'explanations': result.explanations,
                'anomaly_indicators': result.anomaly_indicators,
                'processing_time_ms': result.processing_time_ms
            }
            
            # Add optional fields
            if hasattr(result, 'ml_scores') and result.ml_scores:
                serialized['ml_scores'] = result.ml_scores
            if hasattr(result, 'rule_violations') and result.rule_violations:
                serialized['rule_violations'] = result.rule_violations
            
            return serialized
        else:
            # Already a dict
            return result
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        return f"fraud_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    
    def _check_rate_limit(self, request_id: str, client_id: Optional[str] = None) -> bool:
        """Check if request exceeds rate limit"""
        if not client_id:
            client_id = 'default'
        
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        with self._lock:
            # Remove old requests
            self.rate_limiter['request_times'][client_id] = [
                t for t in self.rate_limiter['request_times'][client_id]
                if t > minute_ago
            ]
            
            # Check if blocked
            if client_id in self.rate_limiter['blocked_until']:
                if now < self.rate_limiter['blocked_until'][client_id]:
                    return False
                else:
                    del self.rate_limiter['blocked_until'][client_id]
            
            # Check rate
            request_count = len(self.rate_limiter['request_times'][client_id])
            if request_count >= self.rate_limiter['requests_per_minute']:
                # Block for 1 minute
                self.rate_limiter['blocked_until'][client_id] = now + timedelta(minutes=1)
                return False
            
            # Add request
            self.rate_limiter['request_times'][client_id].append(now)
            return True
    
    def _get_cached_response(self, request: APIRequest) -> Optional[APIResponse]:
        """Get cached response if available"""
        # Create cache key
        cache_key = self._create_cache_key(request)
        
        with self._lock:
            if cache_key in self._response_cache:
                response, timestamp = self._response_cache[cache_key]
                
                # Check if still valid
                if datetime.now() - timestamp < self._cache_ttl:
                    # Update request_id
                    response.request_id = request.request_id
                    return response
                else:
                    del self._response_cache[cache_key]
        
        return None
    
    def _cache_response(self, request: APIRequest, response: APIResponse):
        """Cache API response"""
        cache_key = self._create_cache_key(request)
        
        with self._lock:
            self._response_cache[cache_key] = (response, datetime.now())
            
            # Limit cache size
            max_cache_size = 1000
            if len(self._response_cache) > max_cache_size:
                # Remove oldest entries
                sorted_keys = sorted(
                    self._response_cache.keys(),
                    key=lambda k: self._response_cache[k][1]
                )
                for key in sorted_keys[:len(self._response_cache) - max_cache_size]:
                    del self._response_cache[key]
    
    def _create_cache_key(self, request: APIRequest) -> str:
        """Create cache key for request"""
        # Use request type and data hash
        import hashlib
        data_str = json.dumps(request.data, sort_keys=True)
        data_hash = hashlib.md5(data_str.encode()).hexdigest()
        return f"{request.request_type.value}_{data_hash}"
    
    def _create_error_response(self, 
                             request_id: str, 
                             error_message: str,
                             status: APIResponseStatus = APIResponseStatus.ERROR) -> APIResponse:
        """Create error response"""
        return APIResponse(
            request_id=request_id,
            status=status,
            timestamp=datetime.now(),
            error=error_message
        )
    
    def _log_request(self, request: APIRequest):
        """Log API request"""
        self.logger.info(
            f"API Request: {request.request_id} | "
            f"Type: {request.request_type.value} | "
            f"Data size: {len(json.dumps(request.data))} bytes"
        )
        
        # Store in history
        with self._lock:
            self.request_history.append({
                'request_id': request.request_id,
                'timestamp': request.timestamp,
                'type': request.request_type.value,
                'metadata': request.metadata
            })
            
            # Limit history size
            if len(self.request_history) > self.max_history_size:
                self.request_history = self.request_history[-self.max_history_size:]
    
    def _log_response(self, response: APIResponse):
        """Log API response"""
        self.logger.info(
            f"API Response: {response.request_id} | "
            f"Status: {response.status.value} | "
            f"Time: {response.processing_time_ms:.2f}ms"
        )
    
    def _update_metrics(self, response: APIResponse, request: Optional[APIRequest] = None):
        """Update API metrics"""
        with self._lock:
            self.metrics.total_requests += 1
            
            if response.status == APIResponseStatus.SUCCESS:
                self.metrics.successful_requests += 1
            elif response.status == APIResponseStatus.ERROR:
                self.metrics.failed_requests += 1
                if response.error:
                    error_type = response.error.split(':')[0]
                    self.metrics.error_types[error_type] += 1
            elif response.status == APIResponseStatus.TIMEOUT:
                self.metrics.timeout_requests += 1
            
            # Update timing metrics
            if response.processing_time_ms > 0:
                # Running average
                if self.metrics.average_response_time_ms == 0:
                    self.metrics.average_response_time_ms = response.processing_time_ms
                else:
                    alpha = 0.1
                    self.metrics.average_response_time_ms = (
                        alpha * response.processing_time_ms +
                        (1 - alpha) * self.metrics.average_response_time_ms
                    )
                
                self.metrics.max_response_time_ms = max(
                    self.metrics.max_response_time_ms,
                    response.processing_time_ms
                )
                self.metrics.min_response_time_ms = min(
                    self.metrics.min_response_time_ms,
                    response.processing_time_ms
                )
            
            # Update request type metrics
            if request:
                self.metrics.requests_by_type[request.request_type] += 1
            
            # Update hourly rate
            hour_key = datetime.now().strftime('%Y-%m-%d_%H')
            self.metrics.hourly_request_rate[hour_key] += 1
    
    def _get_transaction_data(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get transaction data by ID"""
        # Look in request history
        for req in reversed(self.request_history):
            if req.get('type') in ['single_transaction', 'batch_transaction']:
                # This would need actual implementation to search transaction data
                pass
        
        return None
    
    def _get_transaction_score(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get fraud score for transaction"""
        # This would search through processed results
        # For now, return None
        return None
    
    def _get_request_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get result for specific request"""
        # Search in history
        for req in self.request_history:
            if req['request_id'] == request_id:
                return req
        return None
    
    def _get_filtered_results(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get filtered results from history"""
        results = []
        
        # Apply filters
        start_date = filters.get('start_date')
        end_date = filters.get('end_date')
        request_type = filters.get('type')
        limit = filters.get('limit', 100)
        
        for req in reversed(self.request_history):
            # Date filter
            if start_date and req['timestamp'] < start_date:
                continue
            if end_date and req['timestamp'] > end_date:
                continue
            
            # Type filter
            if request_type and req['type'] != request_type:
                continue
            
            results.append(req)
            
            if len(results) >= limit:
                break
        
        return results
    
    def _export_metrics(self) -> Dict[str, Any]:
        """Export metrics in JSON-serializable format"""
        metrics = self.metrics
        
        return {
            'summary': {
                'total_requests': metrics.total_requests,
                'successful_requests': metrics.successful_requests,
                'failed_requests': metrics.failed_requests,
                'timeout_requests': metrics.timeout_requests,
                'success_rate': (metrics.successful_requests / metrics.total_requests 
                               if metrics.total_requests > 0 else 0.0)
            },
            'performance': {
                'average_response_time_ms': round(metrics.average_response_time_ms, 2),
                'max_response_time_ms': round(metrics.max_response_time_ms, 2),
                'min_response_time_ms': round(metrics.min_response_time_ms, 2) 
                                       if metrics.min_response_time_ms != float('inf') else 0
            },
            'requests_by_type': {
                k.value: v for k, v in metrics.requests_by_type.items()
            },
            'hourly_request_rate': dict(metrics.hourly_request_rate),
            'error_types': dict(metrics.error_types),
            'timestamp': datetime.now().isoformat()
        }
    
    def persist_metrics(self) -> bool:
        """Persist metrics to storage"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = self.storage_path / f'api_metrics_{timestamp}.json'
            
            metrics_data = self._export_metrics()
            
            with open(filename, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            self.logger.info(f"Persisted metrics to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to persist metrics: {e}")
            return False
    
    def load_metrics(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load metrics from storage"""
        try:
            filepath = self.storage_path / filename
            
            with open(filepath, 'r') as f:
                metrics_data = json.load(f)
            
            self.logger.info(f"Loaded metrics from {filepath}")
            return metrics_data
            
        except Exception as e:
            self.logger.error(f"Failed to load metrics: {e}")
            return None
    
    def reset_metrics(self):
        """Reset API metrics"""
        with self._lock:
            self.metrics = APIMetrics()
            self.logger.info("API metrics reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get API status"""
        return {
            'status': 'operational',
            'active_requests': len(self.active_requests),
            'cache_size': len(self._response_cache),
            'history_size': len(self.request_history),
            'metrics': self._export_metrics(),
            'config': {
                'max_workers': self.config.get('max_workers'),
                'rate_limit': self.rate_limiter['requests_per_minute'],
                'cache_enabled': self.config.get('enable_caching'),
                'async_enabled': self.config.get('enable_async')
            }
        }
    
    def __repr__(self) -> str:
        return (f"FinancialFraudAPI(requests={self.metrics.total_requests}, "
               f"success_rate={self.metrics.successful_requests/self.metrics.total_requests:.2%} "
               f"if self.metrics.total_requests > 0 else 'N/A')")


# Maintain backward compatibility
FraudDetectionAPI = FinancialFraudAPI


if __name__ == "__main__":
    # Example usage and testing
    import asyncio
    
    # Initialize API
    api = FinancialFraudAPI()
    
    print("=== Financial Fraud Detection API ===\n")
    
    # Test 1: Single transaction detection
    print("Test 1: Single Transaction Detection")
    transaction = {
        'transaction_id': 'TX001',
        'amount': 5000,
        'timestamp': datetime.now(),
        'user_id': 'USER123',
        'merchant_id': 'MERCH456',
        'merchant_name': 'Online Store'
    }
    
    response = api.detect_fraud(transaction)
    print(f"Status: {response.status.value}")
    print(f"Processing time: {response.processing_time_ms:.2f}ms")
    if response.data:
        print(f"Fraud score: {response.data.get('fraud_score', 'N/A')}")
        print(f"Is fraud: {response.data.get('is_fraud', 'N/A')}")
    print()
    
    # Test 2: Batch detection
    print("Test 2: Batch Transaction Detection")
    transactions = [
        {
            'transaction_id': f'TX{i:03d}',
            'amount': np.random.uniform(100, 10000),
            'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 24)),
            'user_id': f'USER{np.random.randint(100, 200)}',
            'merchant_id': f'MERCH{np.random.randint(400, 600)}'
        }
        for i in range(10)
    ]
    
    batch_response = api.batch_check(transactions)
    print(f"Status: {batch_response.status.value}")
    print(f"Processing time: {batch_response.processing_time_ms:.2f}ms")
    if batch_response.data and 'summary' in batch_response.data:
        print(f"Total processed: {batch_response.data['summary'].get('total_processed', 0)}")
        print(f"Fraud detected: {batch_response.data['summary'].get('fraud_detected', 0)}")
    print()
    
    # Test 3: Transaction validation
    print("Test 3: Transaction Validation")
    invalid_transaction = {
        'amount': -1000,  # Invalid negative amount
        'timestamp': 'invalid-date'
    }
    
    validation_response = api.validate_transaction(invalid_transaction)
    print(f"Status: {validation_response.status.value}")
    if validation_response.data:
        print(f"Valid: {validation_response.data.get('valid', False)}")
        print(f"Errors: {validation_response.data.get('errors', [])}")
        print(f"Warnings: {validation_response.data.get('warnings', [])}")
    print()
    
    # Test 4: Get metrics
    print("Test 4: API Metrics")
    metrics_response = api.get_metrics()
    if metrics_response.data:
        print(f"Total requests: {metrics_response.data['summary']['total_requests']}")
        print(f"Success rate: {metrics_response.data['summary']['success_rate']:.2%}")
        print(f"Avg response time: {metrics_response.data['performance']['average_response_time_ms']:.2f}ms")
    print()
    
    # Test 5: Async operation
    print("Test 5: Async Operation")
    async def test_async():
        """Test async API calls"""
        tasks = []
        for i in range(5):
            transaction = {
                'transaction_id': f'ASYNC_TX{i:03d}',
                'amount': np.random.uniform(1000, 5000)
            }
            task = api.async_detect_fraud(transaction)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    # Run async test
    async_results = asyncio.run(test_async())
    print(f"Processed {len(async_results)} async requests")
    success_count = sum(1 for r in async_results if r.status == APIResponseStatus.SUCCESS)
    print(f"Successful: {success_count}/{len(async_results)}")
    print()
    
    # Test 6: API Status
    print("Test 6: API Status")
    status = api.get_status()
    print(f"Status: {status['status']}")
    print(f"Active requests: {status['active_requests']}")
    print(f"Cache size: {status['cache_size']}")
    print(f"Config: {json.dumps(status['config'], indent=2)}")
    
    print("\nAPI testing complete!")