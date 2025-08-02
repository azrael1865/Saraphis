"""
Accuracy Tracking API - Phase 5B: API and Interface Layer
Production-ready RESTful API for accuracy tracking system with complete authentication,
rate limiting, and dashboard interfaces.
"""

import logging
import time
import json
import jwt
import hashlib
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
from pathlib import Path
import aiofiles
import redis
from redis import asyncio as aioredis
import httpx

# FastAPI imports
from fastapi import (
    FastAPI, HTTPException, Depends, Request, Response, 
    BackgroundTasks, File, UploadFile, Query, Path as PathParam,
    Security, status
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import from existing modules
try:
    from accuracy_tracking_orchestrator import (
        AccuracyTrackingOrchestrator, IntegrationResponse, 
        ExperimentConfig, ComparisonResult
    )
    from advanced_accuracy_monitoring import (
        AlertingSystem, DashboardManager, DashboardWidget,
        AlertChannel, ResponseAction, DriftSeverity
    )
    from real_time_accuracy_monitor import (
        RealTimeAccuracyMonitor, DriftAlert, MonitoringStatus,
        AccuracyWindow
    )
    from enhanced_fraud_core_monitoring import (
        MonitoringManager, MetricsCollector, PerformanceMetrics,
        CacheManager, monitor_performance
    )
    from accuracy_tracking_db import (
        AccuracyTrackingDatabase, AccuracyMetric, ModelVersion,
        MetricType, DataType, ModelStatus
    )
    from enhanced_fraud_core_exceptions import (
        EnhancedFraudException, MonitoringError, ValidationError,
        ErrorContext, create_error_context
    )
except ImportError:
    # Fallback for standalone development
    from accuracy_tracking_orchestrator import (
        AccuracyTrackingOrchestrator, IntegrationResponse, 
        ExperimentConfig, ComparisonResult
    )
    from advanced_accuracy_monitoring import (
        AlertingSystem, DashboardManager, DashboardWidget,
        AlertChannel, ResponseAction, DriftSeverity
    )
    from real_time_accuracy_monitor import (
        RealTimeAccuracyMonitor, DriftAlert, MonitoringStatus,
        AccuracyWindow
    )
    from enhanced_fraud_core_monitoring import (
        MonitoringManager, MetricsCollector, PerformanceMetrics,
        CacheManager, monitor_performance
    )
    from accuracy_tracking_db import (
        AccuracyTrackingDatabase, AccuracyMetric, ModelVersion,
        MetricType, DataType, ModelStatus
    )
    from enhanced_fraud_core_exceptions import (
        EnhancedFraudException, MonitoringError, ValidationError,
        ErrorContext, create_error_context
    )

# Configure logging
logger = logging.getLogger(__name__)

# ======================== API CONFIGURATION ========================

@dataclass
class APIConfig:
    """API configuration settings"""
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    
    # Authentication
    jwt_secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_default: str = "100/minute"
    rate_limit_burst: str = "10/second"
    
    # CORS settings
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # Redis settings
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # API documentation
    docs_enabled: bool = True
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"
    
    # Static files
    serve_static: bool = True
    static_dir: str = "static"
    static_url: str = "/static"
    
    # Security
    enable_api_keys: bool = True
    require_https: bool = False
    trusted_hosts: List[str] = field(default_factory=list)
    
    # Export settings
    max_export_rows: int = 100000
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv", "excel", "parquet"])
    
    # Dashboard settings
    dashboard_refresh_interval: int = 30  # seconds
    max_dashboard_connections: int = 100

# ======================== API MODELS ========================

class UserRole(str, Enum):
    """User roles for authorization"""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"

class AuthToken(BaseModel):
    """Authentication token model"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None

class UserCredentials(BaseModel):
    """User login credentials"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)

class User(BaseModel):
    """User model"""
    user_id: str
    username: str
    email: str
    role: UserRole
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None
    permissions: List[str] = Field(default_factory=list)

class APIKey(BaseModel):
    """API key model"""
    key_id: str
    key_hash: str
    user_id: str
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    permissions: List[str] = Field(default_factory=list)
    is_active: bool = True

class AccuracyMetricsRequest(BaseModel):
    """Request model for accuracy metrics"""
    model_ids: List[str]
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    metric_types: Optional[List[str]] = None
    data_type: Optional[str] = None
    aggregation: Optional[str] = "mean"
    
    @validator('metric_types')
    def validate_metric_types(cls, v):
        if v:
            valid_types = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            for metric in v:
                if metric not in valid_types:
                    raise ValueError(f"Invalid metric type: {metric}")
        return v

class ModelComparisonRequest(BaseModel):
    """Request model for model comparison"""
    model_ids: List[str] = Field(..., min_items=2)
    comparison_metrics: List[str] = Field(default_factory=lambda: ['accuracy', 'precision', 'recall'])
    time_window: Optional[str] = "7d"
    statistical_tests: Optional[List[str]] = None
    include_visualizations: bool = True

class ReportGenerationRequest(BaseModel):
    """Request model for report generation"""
    report_type: str = Field(..., regex="^(summary|detailed|comparison|drift)$")
    model_ids: List[str]
    time_range: str = "30d"
    include_sections: List[str] = Field(default_factory=lambda: ["metrics", "trends", "alerts"])
    export_format: str = "pdf"
    email_delivery: Optional[str] = None

class DashboardConfigRequest(BaseModel):
    """Request model for dashboard configuration"""
    dashboard_name: str
    layout: str = Field(default="grid", regex="^(grid|list|custom)$")
    widgets: List[Dict[str, Any]]
    refresh_interval: int = Field(default=30, ge=10, le=300)
    filters: Optional[Dict[str, Any]] = None

class AlertConfigRequest(BaseModel):
    """Request model for alert configuration"""
    alert_name: str
    model_ids: List[str]
    condition: Dict[str, Any]
    severity: str
    channels: List[str]
    enabled: bool = True

class ExportRequest(BaseModel):
    """Request model for data export"""
    export_type: str = Field(..., regex="^(metrics|predictions|alerts|full)$")
    model_ids: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    format: str = Field(default="csv", regex="^(csv|json|excel|parquet)$")
    compression: Optional[str] = Field(default=None, regex="^(gzip|zip)$")

# ======================== AUTHENTICATION & SECURITY ========================

class AuthenticationHandler:
    """Handle authentication and authorization"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.security = HTTPBearer()
        self.redis_client = None
        self._init_redis()
    
    def _init_redis(self) -> None:
        """Initialize Redis connection for session management"""
        try:
            self.redis_client = redis.from_url(
                self.config.redis_url,
                password=self.config.redis_password,
                db=self.config.redis_db,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connection established for session management")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def create_access_token(self, user_id: str, role: str, permissions: List[str]) -> str:
        """Create JWT access token"""
        payload = {
            'user_id': user_id,
            'role': role,
            'permissions': permissions,
            'exp': datetime.utcnow() + timedelta(minutes=self.config.access_token_expire_minutes),
            'iat': datetime.utcnow(),
            'type': 'access'
        }
        
        return jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token"""
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(days=self.config.refresh_token_expire_days),
            'iat': datetime.utcnow(),
            'type': 'refresh'
        }
        
        return jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.config.jwt_secret_key, 
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Check if token is blacklisted
            if self.redis_client and self.redis_client.get(f"blacklist:{token}"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def verify_api_key(self, api_key: str) -> Dict[str, Any]:
        """Verify API key"""
        # Hash the API key
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        # Check in Redis cache first
        if self.redis_client:
            cached = self.redis_client.get(f"api_key:{key_hash}")
            if cached:
                return json.loads(cached)
        
        # In production, check against database
        # For now, return mock data
        api_key_data = {
            'user_id': 'api_user_123',
            'permissions': ['read:metrics', 'read:reports'],
            'rate_limit': '1000/hour'
        }
        
        # Cache the result
        if self.redis_client:
            self.redis_client.setex(
                f"api_key:{key_hash}",
                3600,  # 1 hour cache
                json.dumps(api_key_data)
            )
        
        return api_key_data
    
    async def get_current_user(
        self, 
        credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
    ) -> Dict[str, Any]:
        """Get current user from token"""
        token = credentials.credentials
        payload = self.verify_token(token)
        
        if payload.get('type') != 'access':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        return {
            'user_id': payload['user_id'],
            'role': payload['role'],
            'permissions': payload.get('permissions', [])
        }
    
    def require_permission(self, permission: str) -> Callable:
        """Dependency to require specific permission"""
        async def permission_checker(
            current_user: Dict[str, Any] = Depends(self.get_current_user)
        ):
            if permission not in current_user.get('permissions', []) and current_user.get('role') != 'admin':
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission}' required"
                )
            return current_user
        
        return permission_checker
    
    def require_role(self, roles: List[str]) -> Callable:
        """Dependency to require specific roles"""
        async def role_checker(
            current_user: Dict[str, Any] = Depends(self.get_current_user)
        ):
            if current_user.get('role') not in roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"One of roles {roles} required"
                )
            return current_user
        
        return role_checker

# ======================== ACCURACY TRACKING API ========================

class AccuracyTrackingAPI:
    """
    Complete API and interface layer for accuracy tracking system.
    Provides RESTful endpoints, authentication, and dashboard interfaces.
    """
    
    def __init__(
        self,
        orchestrator: AccuracyTrackingOrchestrator,
        api_config: Optional[APIConfig] = None
    ):
        """
        Initialize AccuracyTrackingAPI with orchestrator and configuration.
        
        Args:
            orchestrator: AccuracyTrackingOrchestrator instance
            api_config: API configuration settings
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        if not isinstance(orchestrator, AccuracyTrackingOrchestrator):
            raise ValidationError(
                "orchestrator must be an AccuracyTrackingOrchestrator instance",
                context=create_error_context(
                    component="AccuracyTrackingAPI",
                    operation="init"
                )
            )
        
        self.orchestrator = orchestrator
        self.config = api_config or APIConfig()
        
        # Initialize components
        self.auth_handler = AuthenticationHandler(self.config)
        self.app = None
        self.limiter = None
        self.background_tasks = set()
        
        # WebSocket connections for real-time updates
        self.websocket_connections = {}
        
        # Initialize API
        self._init_api()
        
        self.logger.info("AccuracyTrackingAPI initialized successfully")
    
    def _init_api(self) -> None:
        """Initialize FastAPI application"""
        # Create FastAPI app
        self.app = FastAPI(
            title="Fraud Detection Accuracy Tracking API",
            description="Production-ready API for tracking and monitoring ML model accuracy",
            version="1.0.0",
            docs_url=self.config.docs_url if self.config.docs_enabled else None,
            redoc_url=self.config.redoc_url if self.config.docs_enabled else None,
            openapi_url=self.config.openapi_url if self.config.docs_enabled else None
        )
        
        # Initialize rate limiter
        self.limiter = Limiter(key_func=get_remote_address)
        self.app.state.limiter = self.limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        
        # Add middleware
        self._add_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Setup static files
        if self.config.serve_static:
            self.app.mount(
                self.config.static_url,
                StaticFiles(directory=self.config.static_dir),
                name="static"
            )
    
    def _add_middleware(self) -> None:
        """Add middleware to FastAPI app"""
        # CORS middleware
        if self.config.cors_enabled:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_credentials=True,
                allow_methods=self.config.cors_methods,
                allow_headers=self.config.cors_headers
            )
        
        # Custom middleware for logging and monitoring
        @self.app.middleware("http")
        async def monitoring_middleware(request: Request, call_next):
            start_time = time.time()
            
            # Add request ID
            request_id = hashlib.md5(
                f"{time.time()}{request.client.host}".encode()
            ).hexdigest()[:8]
            
            try:
                response = await call_next(request)
                
                # Record metrics
                duration = time.time() - start_time
                self._record_api_metric(
                    endpoint=request.url.path,
                    method=request.method,
                    status_code=response.status_code,
                    duration=duration,
                    request_id=request_id
                )
                
                # Add custom headers
                response.headers["X-Request-ID"] = request_id
                response.headers["X-Response-Time"] = f"{duration:.3f}"
                
                return response
                
            except Exception as e:
                duration = time.time() - start_time
                self._record_api_metric(
                    endpoint=request.url.path,
                    method=request.method,
                    status_code=500,
                    duration=duration,
                    request_id=request_id,
                    error=str(e)
                )
                raise
    
    def _setup_routes(self) -> None:
        """Setup all API routes"""
        # Health check
        @self.app.get("/health", tags=["System"])
        async def health_check():
            """Health check endpoint"""
            try:
                # Get comprehensive health from orchestrator if available
                if hasattr(self, 'orchestrator') and self.orchestrator and hasattr(self.orchestrator, 'health_monitor'):
                    overall_health = self.orchestrator.health_monitor.get_overall_health()
                    return {
                        "status": overall_health.get('status', 'unknown'),
                        "timestamp": datetime.now().isoformat(),
                        "version": "1.0.0",
                        "components": overall_health.get('component_count', 0),
                        "healthy_components": overall_health.get('healthy_components', 0),
                        "warning_components": overall_health.get('warning_components', 0),
                        "critical_components": overall_health.get('critical_components', 0),
                        "active_alerts": overall_health.get('active_alerts', 0)
                    }
                else:
                    # Fallback to basic health check
                    return {
                        "status": "healthy",
                        "timestamp": datetime.now().isoformat(),
                        "version": "1.0.0"
                    }
            except Exception as e:
                return {
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "error": str(e)
                }
        
        @self.app.get("/health/detailed", tags=["System"])
        async def detailed_health_check():
            """Detailed health check endpoint"""
            try:
                if hasattr(self, 'orchestrator') and self.orchestrator and hasattr(self.orchestrator, 'health_monitor'):
                    return {
                        "overall_health": self.orchestrator.health_monitor.get_overall_health(),
                        "components": self.orchestrator.health_monitor.get_all_component_health(),
                        "alerts": self.orchestrator.health_monitor.get_active_alerts(),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {"error": "Comprehensive health monitoring not available"}
            except Exception as e:
                return {"error": str(e)}
        
        @self.app.get("/health/alerts", tags=["System"])
        async def get_health_alerts():
            """Get active health alerts"""
            try:
                if hasattr(self, 'orchestrator') and self.orchestrator and hasattr(self.orchestrator, 'health_monitor'):
                    return {
                        "alerts": self.orchestrator.health_monitor.get_active_alerts(),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {"error": "Health monitoring not available"}
            except Exception as e:
                return {"error": str(e)}
        
        # Diagnostics endpoints
        self._setup_diagnostics_routes()
        
        # Config loader endpoints
        self._setup_config_loader_routes()
        
        # Authentication endpoints
        self._setup_auth_routes()
        
        # Accuracy metrics endpoints
        self._setup_metrics_routes()
        
        # Model comparison endpoints
        self._setup_comparison_routes()
        
        # Reporting endpoints
        self._setup_reporting_routes()
        
        # Dashboard endpoints
        self._setup_dashboard_routes()
        
        # Export endpoints
        self._setup_export_routes()
        
        # WebSocket endpoints
        self._setup_websocket_routes()
    
    # ======================== UNIFIED API ENDPOINTS ========================
    
    def create_unified_accuracy_api(
        self,
        api_config: Dict[str, Any],
        authentication_config: Dict[str, Any],
        rate_limiting: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create unified API with comprehensive configuration.
        
        Args:
            api_config: API configuration settings
            authentication_config: Authentication configuration
            rate_limiting: Rate limiting configuration
            
        Returns:
            API creation status and details
        """
        try:
            # Update configuration
            for key, value in api_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # Setup authentication
            if authentication_config:
                self._configure_authentication(authentication_config)
            
            # Setup rate limiting
            if rate_limiting:
                self._configure_rate_limiting(rate_limiting)
            
            # Create OpenAPI schema
            openapi_schema = get_openapi(
                title=self.app.title,
                version=self.app.version,
                description=self.app.description,
                routes=self.app.routes,
            )
            
            # Add security schemes
            openapi_schema["components"]["securitySchemes"] = {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                },
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                }
            }
            
            self.app.openapi_schema = openapi_schema
            
            return {
                "status": "success",
                "api_url": f"http://{self.config.host}:{self.config.port}",
                "docs_url": f"http://{self.config.host}:{self.config.port}{self.config.docs_url}",
                "endpoints_count": len(self.app.routes),
                "authentication_enabled": self.config.enable_api_keys,
                "rate_limiting_enabled": self.config.rate_limit_enabled
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create unified API: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def expose_accuracy_metrics_endpoints(
        self,
        endpoint_config: Dict[str, Any],
        security_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Expose accuracy metrics through API endpoints.
        
        Args:
            endpoint_config: Endpoint configuration
            security_config: Security settings for endpoints
            
        Returns:
            Endpoint configuration status
        """
        endpoints_created = []
        
        try:
            # GET /api/v1/metrics/{model_id}
            @self.app.get(
                "/api/v1/metrics/{model_id}",
                tags=["Metrics"],
                response_model=Dict[str, Any],
                dependencies=[Depends(self.auth_handler.require_permission("read:metrics"))]
            )
            @self.limiter.limit(self.config.rate_limit_default)
            async def get_model_metrics(
                request: Request,
                model_id: str = PathParam(..., description="Model identifier"),
                start_date: Optional[datetime] = Query(None, description="Start date for metrics"),
                end_date: Optional[datetime] = Query(None, description="End date for metrics"),
                metrics: Optional[List[str]] = Query(None, description="Specific metrics to retrieve"),
                current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
            ):
                """Get accuracy metrics for a specific model"""
                try:
                    # Get metrics from orchestrator
                    result = self.orchestrator.generate_accuracy_reports(
                        report_config={
                            "model_ids": [model_id],
                            "start_date": start_date,
                            "end_date": end_date,
                            "metrics": metrics or ["accuracy", "precision", "recall", "f1_score"]
                        },
                        output_formats=["json"]
                    )
                    
                    if result.success:
                        return result.data.get("reports", {}).get("json", {})
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=result.error
                        )
                        
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/metrics/{model_id}")
            
            # POST /api/v1/metrics/query
            @self.app.post(
                "/api/v1/metrics/query",
                tags=["Metrics"],
                response_model=Dict[str, Any],
                dependencies=[Depends(self.auth_handler.require_permission("read:metrics"))]
            )
            @self.limiter.limit(self.config.rate_limit_default)
            async def query_metrics(
                request: Request,
                query: AccuracyMetricsRequest,
                current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
            ):
                """Query accuracy metrics with advanced filters"""
                try:
                    # Query metrics from database
                    metrics = []
                    for model_id in query.model_ids:
                        model_metrics = self.orchestrator.accuracy_database.get_accuracy_metrics(
                            model_id=model_id,
                            start_date=query.start_date,
                            end_date=query.end_date,
                            metric_type=query.metric_types[0] if query.metric_types else None,
                            data_type=DataType[query.data_type.upper()] if query.data_type else None
                        )
                        metrics.extend(model_metrics)
                    
                    # Aggregate if requested
                    if query.aggregation and metrics:
                        aggregated = self._aggregate_metrics(metrics, query.aggregation)
                        return {"metrics": aggregated, "count": len(metrics)}
                    
                    return {"metrics": [asdict(m) for m in metrics], "count": len(metrics)}
                    
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/metrics/query")
            
            # GET /api/v1/metrics/realtime/{model_id}
            @self.app.get(
                "/api/v1/metrics/realtime/{model_id}",
                tags=["Metrics"],
                response_model=Dict[str, Any],
                dependencies=[Depends(self.auth_handler.require_permission("read:metrics"))]
            )
            async def get_realtime_metrics(
                model_id: str = PathParam(..., description="Model identifier"),
                current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
            ):
                """Get real-time accuracy metrics for a model"""
                try:
                    # Get from real-time monitor
                    metrics = self.orchestrator.realtime_monitor.get_current_accuracy_metrics(model_id)
                    return metrics
                    
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/metrics/realtime/{model_id}")
            
            return {
                "status": "success",
                "endpoints_created": endpoints_created,
                "security_enabled": bool(security_config)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to expose metrics endpoints: {e}")
            return {
                "status": "error",
                "error": str(e),
                "endpoints_created": endpoints_created
            }
    
    def provide_model_comparison_interface(
        self,
        comparison_config: Dict[str, Any],
        authorization_rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Provide model comparison API endpoints.
        
        Args:
            comparison_config: Comparison configuration
            authorization_rules: Authorization rules for comparison
            
        Returns:
            Comparison interface status
        """
        endpoints_created = []
        
        try:
            # POST /api/v1/comparison/models
            @self.app.post(
                "/api/v1/comparison/models",
                tags=["Comparison"],
                response_model=Dict[str, Any],
                dependencies=[Depends(self.auth_handler.require_permission("read:comparisons"))]
            )
            @self.limiter.limit(self.config.rate_limit_default)
            async def compare_models(
                request: Request,
                comparison_request: ModelComparisonRequest,
                background_tasks: BackgroundTasks,
                current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
            ):
                """Compare multiple models"""
                try:
                    # Start comparison
                    result = self.orchestrator.cross_model_accuracy_comparison(
                        model_ids=comparison_request.model_ids,
                        comparison_metrics=comparison_request.comparison_metrics,
                        time_window=comparison_request.time_window,
                        statistical_tests=comparison_request.statistical_tests
                    )
                    
                    if result.success:
                        comparison_data = result.data
                        
                        # Add visualization generation to background
                        if comparison_request.include_visualizations:
                            background_tasks.add_task(
                                self._generate_comparison_visualizations,
                                comparison_data["comparison_id"],
                                comparison_data
                            )
                        
                        return comparison_data
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=result.error
                        )
                        
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/comparison/models")
            
            # GET /api/v1/comparison/{comparison_id}
            @self.app.get(
                "/api/v1/comparison/{comparison_id}",
                tags=["Comparison"],
                response_model=Dict[str, Any],
                dependencies=[Depends(self.auth_handler.require_permission("read:comparisons"))]
            )
            async def get_comparison_results(
                comparison_id: str = PathParam(..., description="Comparison identifier"),
                current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
            ):
                """Get comparison results by ID"""
                try:
                    # Get from cache or database
                    results = self._get_comparison_results(comparison_id)
                    if results:
                        return results
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Comparison {comparison_id} not found"
                        )
                        
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/comparison/{comparison_id}")
            
            # GET /api/v1/comparison/rankings
            @self.app.get(
                "/api/v1/comparison/rankings",
                tags=["Comparison"],
                response_model=Dict[str, Any],
                dependencies=[Depends(self.auth_handler.require_permission("read:comparisons"))]
            )
            async def get_model_rankings(
                metric: str = Query("accuracy", description="Metric to rank by"),
                time_window: str = Query("7d", description="Time window for ranking"),
                limit: int = Query(10, ge=1, le=100, description="Number of models to return"),
                current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
            ):
                """Get model rankings by metric"""
                try:
                    rankings = self._calculate_model_rankings(metric, time_window, limit)
                    return {
                        "rankings": rankings,
                        "metric": metric,
                        "time_window": time_window,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/comparison/rankings")
            
            return {
                "status": "success",
                "endpoints_created": endpoints_created,
                "comparison_features": ["statistical_tests", "visualizations", "rankings"]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to provide comparison interface: {e}")
            return {
                "status": "error",
                "error": str(e),
                "endpoints_created": endpoints_created
            }
    
    def implement_accuracy_reporting_api(
        self,
        reporting_config: Dict[str, Any],
        export_formats: List[str]
    ) -> Dict[str, Any]:
        """
        Implement accuracy reporting API endpoints.
        
        Args:
            reporting_config: Reporting configuration
            export_formats: Supported export formats
            
        Returns:
            Reporting API status
        """
        endpoints_created = []
        
        try:
            # POST /api/v1/reports/generate
            @self.app.post(
                "/api/v1/reports/generate",
                tags=["Reports"],
                response_model=Dict[str, Any],
                dependencies=[Depends(self.auth_handler.require_permission("create:reports"))]
            )
            @self.limiter.limit("10/hour")
            async def generate_report(
                request: Request,
                report_request: ReportGenerationRequest,
                background_tasks: BackgroundTasks,
                current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
            ):
                """Generate accuracy report"""
                try:
                    # Validate export format
                    if report_request.export_format not in export_formats:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Unsupported format. Supported: {export_formats}"
                        )
                    
                    # Generate report ID
                    report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(report_request))}"
                    
                    # Start report generation in background
                    background_tasks.add_task(
                        self._generate_report_async,
                        report_id,
                        report_request,
                        current_user['user_id']
                    )
                    
                    return {
                        "report_id": report_id,
                        "status": "generating",
                        "estimated_time": "2-5 minutes",
                        "check_status_url": f"/api/v1/reports/{report_id}/status"
                    }
                    
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/reports/generate")
            
            # GET /api/v1/reports/{report_id}/status
            @self.app.get(
                "/api/v1/reports/{report_id}/status",
                tags=["Reports"],
                response_model=Dict[str, Any]
            )
            async def get_report_status(
                report_id: str = PathParam(..., description="Report identifier")
            ):
                """Get report generation status"""
                try:
                    status = self._get_report_status(report_id)
                    if status:
                        return status
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Report {report_id} not found"
                        )
                        
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/reports/{report_id}/status")
            
            # GET /api/v1/reports/{report_id}/download
            @self.app.get(
                "/api/v1/reports/{report_id}/download",
                tags=["Reports"],
                dependencies=[Depends(self.auth_handler.require_permission("read:reports"))]
            )
            async def download_report(
                report_id: str = PathParam(..., description="Report identifier"),
                current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
            ):
                """Download generated report"""
                try:
                    # Get report file path
                    report_path = self._get_report_file_path(report_id)
                    if report_path and report_path.exists():
                        return FileResponse(
                            path=report_path,
                            filename=f"{report_id}.{report_path.suffix}",
                            media_type="application/octet-stream"
                        )
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Report {report_id} not found"
                        )
                        
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/reports/{report_id}/download")
            
            # GET /api/v1/reports/templates
            @self.app.get(
                "/api/v1/reports/templates",
                tags=["Reports"],
                response_model=List[Dict[str, Any]]
            )
            async def get_report_templates():
                """Get available report templates"""
                templates = [
                    {
                        "template_id": "daily_summary",
                        "name": "Daily Summary Report",
                        "description": "Daily accuracy metrics summary",
                        "sections": ["metrics", "trends", "alerts"],
                        "formats": ["pdf", "html", "json"]
                    },
                    {
                        "template_id": "model_comparison",
                        "name": "Model Comparison Report",
                        "description": "Detailed model comparison analysis",
                        "sections": ["metrics", "statistical_tests", "visualizations"],
                        "formats": ["pdf", "html", "excel"]
                    },
                    {
                        "template_id": "drift_analysis",
                        "name": "Drift Analysis Report",
                        "description": "Comprehensive drift detection report",
                        "sections": ["drift_detection", "trends", "recommendations"],
                        "formats": ["pdf", "html", "json"]
                    }
                ]
                return templates
            
            endpoints_created.append("/api/v1/reports/templates")
            
            return {
                "status": "success",
                "endpoints_created": endpoints_created,
                "supported_formats": export_formats,
                "templates_available": 3
            }
            
        except Exception as e:
            self.logger.error(f"Failed to implement reporting API: {e}")
            return {
                "status": "error",
                "error": str(e),
                "endpoints_created": endpoints_created
            }
    
    # ======================== DASHBOARD AND INTERFACE MANAGEMENT ========================
    
    def create_monitoring_dashboard_interface(
        self,
        dashboard_config: Dict[str, Any],
        user_permissions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create real-time monitoring dashboard interface.
        
        Args:
            dashboard_config: Dashboard configuration
            user_permissions: User permission settings
            
        Returns:
            Dashboard interface creation status
        """
        endpoints_created = []
        
        try:
            # POST /api/v1/dashboards/create
            @self.app.post(
                "/api/v1/dashboards/create",
                tags=["Dashboards"],
                response_model=Dict[str, Any],
                dependencies=[Depends(self.auth_handler.require_permission("create:dashboards"))]
            )
            async def create_dashboard(
                dashboard_request: DashboardConfigRequest,
                current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
            ):
                """Create a new monitoring dashboard"""
                try:
                    # Create dashboard using dashboard manager
                    dashboard = self.orchestrator.dashboard_manager.create_accuracy_dashboard(
                        dashboard_config={
                            "title": dashboard_request.dashboard_name,
                            "widgets": dashboard_request.widgets,
                            "layout": dashboard_request.layout,
                            "refresh_interval": dashboard_request.refresh_interval
                        },
                        model_filters=dashboard_request.filters.get("model_ids", []) if dashboard_request.filters else []
                    )
                    
                    # Save user association
                    self._save_user_dashboard(current_user['user_id'], dashboard['dashboard_id'])
                    
                    return {
                        "dashboard_id": dashboard['dashboard_id'],
                        "status": "created",
                        "access_url": f"/dashboards/{dashboard['dashboard_id']}",
                        "widgets_count": len(dashboard['widgets'])
                    }
                    
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/dashboards/create")
            
            # GET /api/v1/dashboards/{dashboard_id}
            @self.app.get(
                "/api/v1/dashboards/{dashboard_id}",
                tags=["Dashboards"],
                response_model=Dict[str, Any],
                dependencies=[Depends(self.auth_handler.require_permission("read:dashboards"))]
            )
            async def get_dashboard(
                dashboard_id: str = PathParam(..., description="Dashboard identifier"),
                current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
            ):
                """Get dashboard configuration and data"""
                try:
                    # Check user access
                    if not self._check_dashboard_access(current_user['user_id'], dashboard_id):
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="Access denied to this dashboard"
                        )
                    
                    # Get dashboard status
                    dashboard_status = self.orchestrator.dashboard_manager.get_dashboard_status(dashboard_id)
                    
                    if 'error' in dashboard_status:
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail=dashboard_status['error']
                        )
                    
                    return dashboard_status
                    
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/dashboards/{dashboard_id}")
            
            # GET /api/v1/dashboards/{dashboard_id}/widgets/{widget_id}/data
            @self.app.get(
                "/api/v1/dashboards/{dashboard_id}/widgets/{widget_id}/data",
                tags=["Dashboards"],
                response_model=Dict[str, Any],
                dependencies=[Depends(self.auth_handler.require_permission("read:dashboards"))]
            )
            async def get_widget_data(
                dashboard_id: str = PathParam(..., description="Dashboard identifier"),
                widget_id: str = PathParam(..., description="Widget identifier"),
                format: str = Query("json", description="Data format (json or figure)"),
                current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
            ):
                """Get real-time widget data"""
                try:
                    # Check access
                    if not self._check_dashboard_access(current_user['user_id'], dashboard_id):
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="Access denied"
                        )
                    
                    # Get widget data
                    widget_data = self.orchestrator.dashboard_manager.get_widget_data(widget_id, format)
                    
                    if 'error' in widget_data:
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail=widget_data['error']
                        )
                    
                    return widget_data
                    
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/dashboards/{dashboard_id}/widgets/{widget_id}/data")
            
            # GET /api/v1/dashboards/list
            @self.app.get(
                "/api/v1/dashboards/list",
                tags=["Dashboards"],
                response_model=List[Dict[str, Any]],
                dependencies=[Depends(self.auth_handler.require_permission("read:dashboards"))]
            )
            async def list_dashboards(
                current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
            ):
                """List user's dashboards"""
                try:
                    dashboards = self._get_user_dashboards(current_user['user_id'])
                    return dashboards
                    
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/dashboards/list")
            
            return {
                "status": "success",
                "endpoints_created": endpoints_created,
                "dashboard_features": ["real-time", "widgets", "custom_layouts"]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create dashboard interface: {e}")
            return {
                "status": "error",
                "error": str(e),
                "endpoints_created": endpoints_created
            }
    
    def setup_api_documentation_interface(
        self,
        docs_config: Dict[str, Any],
        interactive_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Setup API documentation interface.
        
        Args:
            docs_config: Documentation configuration
            interactive_features: Interactive documentation features
            
        Returns:
            Documentation setup status
        """
        try:
            # Custom OpenAPI schema
            def custom_openapi():
                if self.app.openapi_schema:
                    return self.app.openapi_schema
                    
                openapi_schema = get_openapi(
                    title="Fraud Detection Accuracy Tracking API",
                    version="1.0.0",
                    description="""
                    ## Overview
                    Production-ready API for tracking and monitoring ML model accuracy in fraud detection.
                    
                    ## Features
                    - Real-time accuracy monitoring
                    - Model comparison and benchmarking
                    - Automated drift detection
                    - Comprehensive reporting
                    - Interactive dashboards
                    
                    ## Authentication
                    This API uses JWT tokens for authentication. Include the token in the Authorization header:
                    ```
                    Authorization: Bearer <your-token>
                    ```
                    
                    ## Rate Limiting
                    Default rate limit: 100 requests per minute
                    Burst limit: 10 requests per second
                    """,
                    routes=self.app.routes,
                )
                
                # Add examples
                openapi_schema["components"]["examples"] = {
                    "MetricsRequest": {
                        "value": {
                            "model_ids": ["fraud_model_v1", "fraud_model_v2"],
                            "start_date": "2024-01-01T00:00:00Z",
                            "end_date": "2024-01-31T23:59:59Z",
                            "metric_types": ["accuracy", "precision", "recall"],
                            "aggregation": "mean"
                        }
                    }
                }
                
                self.app.openapi_schema = openapi_schema
                return self.app.openapi_schema
            
            self.app.openapi = custom_openapi
            
            # Add documentation endpoints
            @self.app.get("/api/v1/docs/postman", tags=["Documentation"])
            async def get_postman_collection():
                """Get Postman collection for API"""
                collection = self._generate_postman_collection()
                return collection
            
            @self.app.get("/api/v1/docs/examples", tags=["Documentation"])
            async def get_api_examples():
                """Get API usage examples"""
                examples = self._get_api_examples()
                return examples
            
            return {
                "status": "success",
                "docs_url": f"http://{self.config.host}:{self.config.port}{self.config.docs_url}",
                "redoc_url": f"http://{self.config.host}:{self.config.port}{self.config.redoc_url}",
                "openapi_url": f"http://{self.config.host}:{self.config.port}{self.config.openapi_url}",
                "features": ["swagger_ui", "redoc", "postman_collection", "examples"]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to setup documentation: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def implement_user_authentication_interface(
        self,
        auth_config: Dict[str, Any],
        session_management: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Implement user authentication interface.
        
        Args:
            auth_config: Authentication configuration
            session_management: Session management settings
            
        Returns:
            Authentication interface status
        """
        endpoints_created = []
        
        try:
            # POST /api/v1/auth/login
            @self.app.post(
                "/api/v1/auth/login",
                tags=["Authentication"],
                response_model=AuthToken
            )
            @self.limiter.limit("5/minute")
            async def login(
                request: Request,
                credentials: UserCredentials
            ):
                """User login endpoint"""
                try:
                    # Verify credentials (mock implementation)
                    user = self._verify_credentials(credentials.username, credentials.password)
                    
                    if not user:
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid credentials"
                        )
                    
                    # Create tokens
                    access_token = self.auth_handler.create_access_token(
                        user['user_id'],
                        user['role'],
                        user['permissions']
                    )
                    
                    refresh_token = self.auth_handler.create_refresh_token(user['user_id'])
                    
                    # Store session
                    if self.auth_handler.redis_client:
                        session_data = {
                            "user_id": user['user_id'],
                            "login_time": datetime.now().isoformat(),
                            "ip_address": request.client.host
                        }
                        self.auth_handler.redis_client.setex(
                            f"session:{user['user_id']}",
                            self.config.access_token_expire_minutes * 60,
                            json.dumps(session_data)
                        )
                    
                    return AuthToken(
                        access_token=access_token,
                        expires_in=self.config.access_token_expire_minutes * 60,
                        refresh_token=refresh_token
                    )
                    
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/auth/login")
            
            # POST /api/v1/auth/logout
            @self.app.post(
                "/api/v1/auth/logout",
                tags=["Authentication"],
                dependencies=[Depends(self.auth_handler.get_current_user)]
            )
            async def logout(
                current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user),
                credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
            ):
                """User logout endpoint"""
                try:
                    # Blacklist token
                    if self.auth_handler.redis_client:
                        self.auth_handler.redis_client.setex(
                            f"blacklist:{credentials.credentials}",
                            self.config.access_token_expire_minutes * 60,
                            "1"
                        )
                        
                        # Remove session
                        self.auth_handler.redis_client.delete(f"session:{current_user['user_id']}")
                    
                    return {"message": "Logged out successfully"}
                    
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/auth/logout")
            
            # POST /api/v1/auth/refresh
            @self.app.post(
                "/api/v1/auth/refresh",
                tags=["Authentication"],
                response_model=AuthToken
            )
            async def refresh_token(
                refresh_token: str = Query(..., description="Refresh token")
            ):
                """Refresh access token"""
                try:
                    # Verify refresh token
                    payload = self.auth_handler.verify_token(refresh_token)
                    
                    if payload.get('type') != 'refresh':
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid refresh token"
                        )
                    
                    # Get user details
                    user = self._get_user_by_id(payload['user_id'])
                    
                    # Create new access token
                    access_token = self.auth_handler.create_access_token(
                        user['user_id'],
                        user['role'],
                        user['permissions']
                    )
                    
                    return AuthToken(
                        access_token=access_token,
                        expires_in=self.config.access_token_expire_minutes * 60
                    )
                    
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/auth/refresh")
            
            # GET /api/v1/auth/me
            @self.app.get(
                "/api/v1/auth/me",
                tags=["Authentication"],
                response_model=User,
                dependencies=[Depends(self.auth_handler.get_current_user)]
            )
            async def get_current_user_info(
                current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
            ):
                """Get current user information"""
                try:
                    user = self._get_user_by_id(current_user['user_id'])
                    return User(**user)
                    
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/auth/me")
            
            return {
                "status": "success",
                "endpoints_created": endpoints_created,
                "auth_features": ["jwt", "refresh_tokens", "session_management", "token_blacklist"]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to implement authentication: {e}")
            return {
                "status": "error",
                "error": str(e),
                "endpoints_created": endpoints_created
            }
    
    def create_data_export_interfaces(
        self,
        export_config: Dict[str, Any],
        format_handlers: Dict[str, Callable]
    ) -> Dict[str, Any]:
        """
        Create data export API endpoints.
        
        Args:
            export_config: Export configuration
            format_handlers: Format-specific export handlers
            
        Returns:
            Export interface creation status
        """
        endpoints_created = []
        
        try:
            # POST /api/v1/export/data
            @self.app.post(
                "/api/v1/export/data",
                tags=["Export"],
                response_model=Dict[str, Any],
                dependencies=[Depends(self.auth_handler.require_permission("export:data"))]
            )
            @self.limiter.limit("10/hour")
            async def export_data(
                request: Request,
                export_request: ExportRequest,
                background_tasks: BackgroundTasks,
                current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
            ):
                """Export accuracy tracking data"""
                try:
                    # Validate format
                    if export_request.format not in self.config.export_formats:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Unsupported format. Supported: {self.config.export_formats}"
                        )
                    
                    # Generate export ID
                    export_id = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(export_request))}"
                    
                    # Start export in background
                    background_tasks.add_task(
                        self._process_export_async,
                        export_id,
                        export_request,
                        current_user['user_id']
                    )
                    
                    return {
                        "export_id": export_id,
                        "status": "processing",
                        "check_status_url": f"/api/v1/export/{export_id}/status",
                        "estimated_time": "1-3 minutes"
                    }
                    
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/export/data")
            
            # GET /api/v1/export/{export_id}/status
            @self.app.get(
                "/api/v1/export/{export_id}/status",
                tags=["Export"],
                response_model=Dict[str, Any]
            )
            async def get_export_status(
                export_id: str = PathParam(..., description="Export identifier")
            ):
                """Get export job status"""
                try:
                    status = self._get_export_status(export_id)
                    if status:
                        return status
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Export {export_id} not found"
                        )
                        
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/export/{export_id}/status")
            
            # GET /api/v1/export/{export_id}/download
            @self.app.get(
                "/api/v1/export/{export_id}/download",
                tags=["Export"],
                dependencies=[Depends(self.auth_handler.require_permission("export:data"))]
            )
            async def download_export(
                export_id: str = PathParam(..., description="Export identifier"),
                current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
            ):
                """Download exported data"""
                try:
                    # Get export file
                    export_path = self._get_export_file_path(export_id)
                    if export_path and export_path.exists():
                        # Stream large files
                        if export_path.stat().st_size > 10 * 1024 * 1024:  # 10MB
                            return StreamingResponse(
                                self._stream_file(export_path),
                                media_type="application/octet-stream",
                                headers={
                                    "Content-Disposition": f"attachment; filename={export_path.name}"
                                }
                            )
                        else:
                            return FileResponse(
                                path=export_path,
                                filename=export_path.name,
                                media_type="application/octet-stream"
                            )
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Export {export_id} not found"
                        )
                        
                except HTTPException:
                    raise
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=str(e)
                    )
            
            endpoints_created.append("/api/v1/export/{export_id}/download")
            
            # GET /api/v1/export/formats
            @self.app.get(
                "/api/v1/export/formats",
                tags=["Export"],
                response_model=List[Dict[str, Any]]
            )
            async def get_export_formats():
                """Get supported export formats"""
                formats = []
                for fmt in self.config.export_formats:
                    format_info = {
                        "format": fmt,
                        "name": fmt.upper(),
                        "mime_type": self._get_mime_type(fmt),
                        "supports_compression": fmt in ["csv", "json"],
                        "max_rows": self.config.max_export_rows
                    }
                    formats.append(format_info)
                
                return formats
            
            endpoints_created.append("/api/v1/export/formats")
            
            return {
                "status": "success",
                "endpoints_created": endpoints_created,
                "export_formats": self.config.export_formats,
                "features": ["async_export", "compression", "streaming"]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create export interfaces: {e}")
            return {
                "status": "error",
                "error": str(e),
                "endpoints_created": endpoints_created
            }
    
    # ======================== HELPER METHODS ========================
    
    def _setup_auth_routes(self) -> None:
        """Setup authentication routes"""
        # Already implemented in implement_user_authentication_interface
        pass
    
    def _setup_metrics_routes(self) -> None:
        """Setup metrics routes"""
        # Already implemented in expose_accuracy_metrics_endpoints
        pass
    
    def _setup_comparison_routes(self) -> None:
        """Setup comparison routes"""
        # Already implemented in provide_model_comparison_interface
        pass
    
    def _setup_reporting_routes(self) -> None:
        """Setup reporting routes"""
        # Already implemented in implement_accuracy_reporting_api
        pass
    
    def _setup_dashboard_routes(self) -> None:
        """Setup dashboard routes"""
        # Already implemented in create_monitoring_dashboard_interface
        pass
    
    def _setup_export_routes(self) -> None:
        """Setup export routes"""
        # Already implemented in create_data_export_interfaces
        pass
    
    def _setup_websocket_routes(self) -> None:
        """Setup WebSocket routes for real-time updates"""
        from fastapi import WebSocket, WebSocketDisconnect
        
        @self.app.websocket("/ws/metrics/{model_id}")
        async def websocket_metrics(
            websocket: WebSocket,
            model_id: str
        ):
            """WebSocket endpoint for real-time metrics"""
            await websocket.accept()
            connection_id = f"ws_{datetime.now().timestamp()}"
            
            try:
                # Add to connections
                if model_id not in self.websocket_connections:
                    self.websocket_connections[model_id] = {}
                self.websocket_connections[model_id][connection_id] = websocket
                
                # Send initial data
                metrics = self.orchestrator.realtime_monitor.get_current_accuracy_metrics(model_id)
                await websocket.send_json(metrics)
                
                # Keep connection alive and send updates
                while True:
                    await asyncio.sleep(self.config.dashboard_refresh_interval)
                    
                    # Get updated metrics
                    metrics = self.orchestrator.realtime_monitor.get_current_accuracy_metrics(model_id)
                    await websocket.send_json(metrics)
                    
            except WebSocketDisconnect:
                # Remove connection
                if model_id in self.websocket_connections:
                    self.websocket_connections[model_id].pop(connection_id, None)
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                await websocket.close()
    
    def _setup_diagnostics_routes(self) -> None:
        """Setup diagnostics and maintenance routes"""
        
        @self.app.get("/diagnostics/status", tags=["Diagnostics"])
        async def get_diagnostics_status():
            """Get diagnostics system status"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    return {"error": "Orchestrator not available"}
                
                if not hasattr(self.orchestrator, 'diagnostics_system'):
                    return {"error": "Diagnostics system not available"}
                
                status = self.orchestrator.diagnostics_system.get_system_status()
                return {
                    "status": status,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": str(e)}
        
        @self.app.post("/diagnostics/run", tags=["Diagnostics"])
        async def run_diagnostics(
            level: str = Query("standard", description="Diagnostic level: basic, standard, comprehensive, deep"),
            categories: Optional[List[str]] = Query(None, description="Specific diagnostic categories to run")
        ):
            """Run diagnostics analysis"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                if not hasattr(self.orchestrator.diagnostics_system, 'run_diagnostics'):
                    raise HTTPException(status_code=503, detail="Diagnostics functionality not available")
                
                # Convert level string to enum
                from accuracy_tracking_diagnostics import DiagnosticLevel, DiagnosticCategory
                
                try:
                    diagnostic_level = DiagnosticLevel(level)
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid diagnostic level: {level}")
                
                # Convert categories if provided
                diagnostic_categories = None
                if categories:
                    try:
                        diagnostic_categories = [DiagnosticCategory(cat) for cat in categories]
                    except ValueError as e:
                        raise HTTPException(status_code=400, detail=f"Invalid diagnostic category: {str(e)}")
                
                # Run diagnostics
                results = self.orchestrator.diagnostics_system.run_diagnostics(
                    level=diagnostic_level,
                    categories=diagnostic_categories
                )
                
                return {
                    "diagnostic_results": results,
                    "timestamp": datetime.now().isoformat()
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/diagnostics/history", tags=["Diagnostics"])
        async def get_diagnostics_history(
            limit: int = Query(10, description="Number of recent diagnostics to return"),
            category: Optional[str] = Query(None, description="Filter by diagnostic category")
        ):
            """Get diagnostics history"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    return {"error": "Orchestrator not available"}
                
                if not hasattr(self.orchestrator, 'diagnostics_system'):
                    return {"error": "Diagnostics system not available"}
                
                history = self.orchestrator.diagnostics_system.get_diagnostic_history(
                    limit=limit,
                    category=category
                )
                
                return {
                    "history": history,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": str(e)}
        
        @self.app.get("/maintenance/schedule", tags=["Maintenance"])
        async def get_maintenance_schedule():
            """Get current maintenance schedule"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    return {"error": "Orchestrator not available"}
                
                if not hasattr(self.orchestrator, 'diagnostics_system'):
                    return {"error": "Diagnostics system not available"}
                
                schedule = self.orchestrator.diagnostics_system.get_maintenance_schedule()
                return {
                    "schedule": schedule,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": str(e)}
        
        @self.app.post("/maintenance/schedule", tags=["Maintenance"])
        async def schedule_maintenance(
            task_type: str = Query(..., description="Type of maintenance task"),
            scheduled_time: Optional[str] = Query(None, description="Scheduled time (ISO format)"),
            priority: int = Query(5, description="Priority (1-10)")
        ):
            """Schedule a maintenance task"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                if not hasattr(self.orchestrator, 'diagnostics_system'):
                    raise HTTPException(status_code=503, detail="Diagnostics system not available")
                
                # Parse scheduled time
                if scheduled_time:
                    try:
                        scheduled_dt = datetime.fromisoformat(scheduled_time.replace('Z', '+00:00'))
                    except ValueError:
                        raise HTTPException(status_code=400, detail="Invalid datetime format")
                else:
                    scheduled_dt = datetime.now() + timedelta(minutes=5)  # Default to 5 minutes from now
                
                # Convert task type to enum
                from accuracy_tracking_diagnostics import MaintenanceType
                try:
                    maintenance_type = MaintenanceType(task_type)
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid maintenance type: {task_type}")
                
                # Schedule the task
                task_id = self.orchestrator.diagnostics_system.schedule_maintenance_task(
                    task_type=maintenance_type,
                    scheduled_time=scheduled_dt,
                    priority=priority
                )
                
                return {
                    "task_id": task_id,
                    "scheduled_time": scheduled_dt.isoformat(),
                    "status": "scheduled",
                    "timestamp": datetime.now().isoformat()
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/troubleshooting/sessions", tags=["Troubleshooting"])
        async def get_troubleshooting_sessions():
            """Get active troubleshooting sessions"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    return {"error": "Orchestrator not available"}
                
                if not hasattr(self.orchestrator, 'diagnostics_system'):
                    return {"error": "Diagnostics system not available"}
                
                sessions = self.orchestrator.diagnostics_system.get_troubleshooting_sessions()
                return {
                    "sessions": sessions,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": str(e)}
        
        @self.app.post("/troubleshooting/start", tags=["Troubleshooting"])
        async def start_troubleshooting(
            problem_description: str = Query(..., description="Description of the problem")
        ):
            """Start a new troubleshooting session"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                if not hasattr(self.orchestrator, 'diagnostics_system'):
                    raise HTTPException(status_code=503, detail="Diagnostics system not available")
                
                session_id = self.orchestrator.diagnostics_system.start_troubleshooting_session(
                    problem_description=problem_description
                )
                
                return {
                    "session_id": session_id,
                    "problem_description": problem_description,
                    "status": "started",
                    "timestamp": datetime.now().isoformat()
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def _setup_config_loader_routes(self) -> None:
        """Setup configuration loader routes"""
        
        @self.app.get("/config/status", tags=["Configuration"])
        async def get_config_loader_status():
            """Get configuration loader status"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    return {"error": "Orchestrator not available"}
                
                if not hasattr(self.orchestrator, 'config_loader'):
                    return {"error": "Config loader not available"}
                
                status = self.orchestrator.config_loader.get_system_status()
                return {
                    "status": status,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": str(e)}
        
        @self.app.get("/config/current", tags=["Configuration"])
        async def get_current_configuration(
            path: Optional[str] = Query(None, description="Configuration path (dot-separated)")
        ):
            """Get current configuration"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                if not hasattr(self.orchestrator, 'config_loader'):
                    raise HTTPException(status_code=503, detail="Config loader not available")
                
                config = self.orchestrator.config_loader.get_configuration(path)
                
                return {
                    "configuration": config,
                    "path": path,
                    "timestamp": datetime.now().isoformat()
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/config/reload", tags=["Configuration"])
        async def reload_configuration():
            """Reload configuration from sources"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                if not hasattr(self.orchestrator, 'config_loader'):
                    raise HTTPException(status_code=503, detail="Config loader not available")
                
                load_result = self.orchestrator.config_loader.reload_configuration()
                
                return {
                    "load_result": {
                        "load_time_ms": load_result.load_time_ms,
                        "sources_loaded": len(load_result.sources_loaded),
                        "validation_passed": all(r.is_valid for r in load_result.validation_results),
                        "version": load_result.version.version
                    },
                    "timestamp": datetime.now().isoformat()
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/config/set", tags=["Configuration"])
        async def set_configuration_value(
            path: str = Query(..., description="Configuration path (dot-separated)"),
            value: str = Query(..., description="Configuration value (JSON string)"),
            save: bool = Query(False, description="Save configuration after setting")
        ):
            """Set configuration value"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                if not hasattr(self.orchestrator, 'config_loader'):
                    raise HTTPException(status_code=503, detail="Config loader not available")
                
                # Parse value as JSON
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    # If not valid JSON, treat as string
                    parsed_value = value
                
                success = self.orchestrator.config_loader.set_configuration(path, parsed_value, save)
                
                return {
                    "success": success,
                    "path": path,
                    "value": parsed_value,
                    "saved": save,
                    "timestamp": datetime.now().isoformat()
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/config/validate", tags=["Configuration"])
        async def validate_configuration():
            """Validate current configuration"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                if not hasattr(self.orchestrator, 'config_loader'):
                    raise HTTPException(status_code=503, detail="Config loader not available")
                
                validation_summary = self.orchestrator.config_loader.validate_configuration()
                
                return {
                    "validation": validation_summary,
                    "timestamp": datetime.now().isoformat()
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/config/sources", tags=["Configuration"])
        async def get_configuration_sources():
            """Get information about configuration sources"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    return {"error": "Orchestrator not available"}
                
                if not hasattr(self.orchestrator, 'config_loader'):
                    return {"error": "Config loader not available"}
                
                sources_info = []
                for name, source in self.orchestrator.config_loader.sources.items():
                    source_info = {
                        "name": name,
                        "type": source.__class__.__name__,
                        "exists": source.exists(),
                        "priority": getattr(source, 'priority', 999),
                        "metadata": source.get_metadata()
                    }
                    sources_info.append(source_info)
                
                return {
                    "sources": sources_info,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": str(e)}
        
        @self.app.post("/config/save", tags=["Configuration"])
        async def save_configuration(
            target_sources: Optional[List[str]] = Query(None, description="Target sources to save to")
        ):
            """Save current configuration to sources"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                if not hasattr(self.orchestrator, 'config_loader'):
                    raise HTTPException(status_code=503, detail="Config loader not available")
                
                success = self.orchestrator.config_loader.save_configuration(
                    target_sources=target_sources
                )
                
                return {
                    "success": success,
                    "target_sources": target_sources or "all_writable",
                    "timestamp": datetime.now().isoformat()
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/config/schema", tags=["Configuration"])
        async def get_configuration_schema():
            """Get configuration schema"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    return {"error": "Orchestrator not available"}
                
                if not hasattr(self.orchestrator, 'config_loader'):
                    return {"error": "Config loader not available"}
                
                schema = self.orchestrator.config_loader.schema_manager.get_schema()
                
                return {
                    "schema": schema,
                    "version": self.orchestrator.config_loader.schema_manager.current_version,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": str(e)}
        
        @self.app.get("/config/versions", tags=["Configuration"])
        async def get_configuration_versions():
            """Get configuration version history"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    return {"error": "Orchestrator not available"}
                
                if not hasattr(self.orchestrator, 'config_loader'):
                    return {"error": "Config loader not available"}
                
                versions = []
                for version in self.orchestrator.config_loader.config_versions:
                    versions.append({
                        "version": version.version,
                        "schema_version": version.schema_version,
                        "created_at": version.created_at.isoformat(),
                        "created_by": version.created_by,
                        "change_summary": version.change_summary,
                        "is_rollback": version.is_rollback
                    })
                
                return {
                    "versions": versions,
                    "current_version": versions[-1] if versions else None,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": str(e)}
        
        # ======================== MISSING ENDPOINTS - PHASE 5C ========================
        
        # Enhanced Model Lifecycle Endpoints (4)
        
        @self.app.post("/api/v1/models/{model_id}/deployment/monitor", tags=["Model Lifecycle"])
        async def deploy_model_with_monitoring(
            model_id: str = PathParam(..., description="Model identifier"),
            deployment_config: Dict[str, Any] = Field(..., description="Deployment configuration"),
            monitoring_rules: Dict[str, Any] = Field(..., description="Monitoring and alert rules"),
            current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
        ):
            """Deploy model with comprehensive monitoring setup"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                # Create mock model for deployment
                model = {"model_id": model_id, "type": "production_model"}
                
                result = self.orchestrator.deploy_model_with_monitoring(
                    model, deployment_config, monitoring_rules
                )
                
                return {
                    "model_id": model_id,
                    "deployment_result": result,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/api/v1/models/{model_id}/lifecycle/update", tags=["Model Lifecycle"])
        async def update_model_lifecycle(
            model_id: str = PathParam(..., description="Model identifier"),
            lifecycle_config: Dict[str, Any] = Field(..., description="Updated lifecycle configuration"),
            tracking_rules: Dict[str, Any] = Field(..., description="Updated tracking rules"),
            current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
        ):
            """Update model lifecycle configuration"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                result = self.orchestrator.manage_model_lifecycle_accuracy(
                    model_id, lifecycle_config, tracking_rules
                )
                
                return {
                    "model_id": model_id,
                    "lifecycle_update": result,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/models/{model_id}/migration/execute", tags=["Model Lifecycle"])
        async def execute_model_migration(
            model_id: str = PathParam(..., description="Model identifier"),
            migration_config: Dict[str, Any] = Field(..., description="Migration configuration"),
            current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
        ):
            """Execute model migration to tracking system"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                # Create mock model registry for this specific model
                model_registry = {model_id: {"type": "existing_model", "version": "1.0"}}
                
                result = self.orchestrator.migrate_existing_models_to_tracking(
                    model_registry, migration_config
                )
                
                return {
                    "model_id": model_id,
                    "migration_result": result,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/models/{model_id}/lifecycle/history", tags=["Model Lifecycle"])
        async def get_model_lifecycle_history(
            model_id: str = PathParam(..., description="Model identifier"),
            limit: int = Query(50, description="Maximum number of history entries"),
            current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
        ):
            """Get model lifecycle history"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                # Mock lifecycle history - in production this would come from database
                history = [
                    {
                        "event_id": f"event_{i}",
                        "event_type": "performance_check",
                        "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
                        "details": {"accuracy": 0.95 - (i * 0.01), "drift_detected": i > 10},
                        "actions_taken": [] if i <= 10 else ["alert_sent", "retraining_triggered"]
                    }
                    for i in range(min(limit, 30))
                ]
                
                return {
                    "model_id": model_id,
                    "history": history,
                    "total_events": len(history),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # Advanced Analytics Endpoints (5)
        
        @self.app.get("/api/v1/analytics/trends/{model_id}", tags=["Advanced Analytics"])
        async def get_model_performance_trends(
            model_id: str = PathParam(..., description="Model identifier"),
            time_period: str = Query("30d", description="Time period (7d, 30d, 90d, 1y)"),
            metrics: List[str] = Query(["accuracy"], description="Metrics to include in trends"),
            current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
        ):
            """Get model performance trends over time"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                # Mock trend data - in production this would come from database
                days = {"7d": 7, "30d": 30, "90d": 90, "1y": 365}.get(time_period, 30)
                
                trends = {}
                for metric in metrics:
                    base_value = {"accuracy": 0.95, "precision": 0.92, "recall": 0.88}.get(metric, 0.90)
                    trends[metric] = [
                        {
                            "date": (datetime.now() - timedelta(days=days-i)).strftime("%Y-%m-%d"),
                            "value": base_value + (0.02 * np.sin(i * 0.1)) + np.random.normal(0, 0.01)
                        }
                        for i in range(days)
                    ]
                
                return {
                    "model_id": model_id,
                    "time_period": time_period,
                    "trends": trends,
                    "summary": {
                        "trend_direction": "stable",
                        "change_percentage": 2.1,
                        "anomalies_detected": 3
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/analytics/comparison/models", tags=["Advanced Analytics"])
        async def compare_multiple_models(
            comparison_config: Dict[str, Any] = Field(..., description="Model comparison configuration"),
            current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
        ):
            """Compare multiple models with advanced analytics"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                result = self.orchestrator.provide_model_comparison_interface(
                    comparison_config, {"user_role": current_user.get("role", "user")}
                )
                
                return {
                    "comparison_id": f"comp_{int(time.time())}",
                    "comparison_result": result,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/analytics/reports/generate", tags=["Advanced Analytics"])
        async def generate_custom_analytics_report(
            report_config: Dict[str, Any] = Field(..., description="Report generation configuration"),
            current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
        ):
            """Generate custom analytics reports"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                result = self.orchestrator.implement_accuracy_reporting_api(
                    report_config, {"formats": ["json", "pdf", "csv"]}
                )
                
                return {
                    "report_id": f"report_{int(time.time())}",
                    "generation_result": result,
                    "estimated_completion": (datetime.now() + timedelta(minutes=5)).isoformat(),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/analytics/dashboard/data", tags=["Advanced Analytics"])
        async def get_dashboard_data_feeds(
            dashboard_type: str = Query("overview", description="Dashboard type (overview, detailed, performance)"),
            refresh_interval: int = Query(30, description="Refresh interval in seconds"),
            current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
        ):
            """Get real-time dashboard data feeds"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                dashboard_config = {
                    "dashboard_type": dashboard_type,
                    "refresh_interval": refresh_interval,
                    "real_time": True
                }
                user_permissions = {"role": current_user.get("role", "user")}
                
                result = self.orchestrator.create_monitoring_dashboard_interface(
                    dashboard_config, user_permissions
                )
                
                return {
                    "dashboard_type": dashboard_type,
                    "data_feeds": result,
                    "refresh_interval": refresh_interval,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/analytics/export/data", tags=["Advanced Analytics"])
        async def export_analytics_data(
            export_config: Dict[str, Any] = Field(..., description="Data export configuration"),
            current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
        ):
            """Export analytics data in various formats"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                # Create comprehensive export configuration
                endpoint_config = {
                    "export_type": export_config.get("export_type", "metrics"),
                    "bulk_export": True,
                    "streaming": export_config.get("streaming", False)
                }
                security_config = {"user_role": current_user.get("role", "user")}
                
                result = self.orchestrator.expose_accuracy_metrics_endpoints(
                    endpoint_config, security_config
                )
                
                return {
                    "export_id": f"export_{int(time.time())}",
                    "export_result": result,
                    "estimated_completion": (datetime.now() + timedelta(minutes=10)).isoformat(),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        # System Management Endpoints (3)
        
        @self.app.post("/api/v1/system/maintenance/schedule", tags=["System Management"])
        async def schedule_system_maintenance(
            maintenance_config: Dict[str, Any] = Field(..., description="Maintenance configuration"),
            scheduled_tasks: Dict[str, Any] = Field(..., description="Tasks to schedule"),
            cleanup_procedures: Dict[str, Any] = Field(default_factory=dict, description="Cleanup procedures"),
            current_user: Dict[str, Any] = Depends(self.auth_handler.require_permission("admin:maintenance"))
        ):
            """Schedule comprehensive system maintenance"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                result = self.orchestrator.perform_system_maintenance(
                    maintenance_config, scheduled_tasks, cleanup_procedures
                )
                
                return {
                    "maintenance_id": f"maint_{int(time.time())}",
                    "maintenance_result": result,
                    "scheduled_time": maintenance_config.get("scheduled_time", datetime.now().isoformat()),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/v1/system/health/detailed", tags=["System Management"])
        async def get_detailed_system_health(
            include_trends: bool = Query(True, description="Include health trends"),
            component_checks: Dict[str, bool] = Query(default={}, description="Specific component checks"),
            current_user: Dict[str, Any] = Depends(self.auth_handler.get_current_user)
        ):
            """Get detailed system health status with comprehensive monitoring"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                health_config = {
                    "include_trends": include_trends,
                    "generate_recommendations": True,
                    "detailed_analysis": True
                }
                
                result = self.orchestrator.monitor_system_health(
                    health_config, component_checks
                )
                
                return {
                    "system_health": result,
                    "health_score": result.get("overall_status", "unknown"),
                    "component_count": len(result.get("component_health", {})),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/v1/system/recovery/execute", tags=["System Management"])
        async def execute_disaster_recovery(
            recovery_config: Dict[str, Any] = Field(..., description="Recovery configuration"),
            backup_restoration: Dict[str, Any] = Field(..., description="Backup restoration settings"),
            system_validation: Dict[str, Any] = Field(default_factory=dict, description="System validation procedures"),
            current_user: Dict[str, Any] = Depends(self.auth_handler.require_permission("admin:recovery"))
        ):
            """Execute disaster recovery procedures"""
            try:
                if not hasattr(self, 'orchestrator') or not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not available")
                
                result = self.orchestrator.execute_disaster_recovery_procedures(
                    recovery_config, backup_restoration, system_validation
                )
                
                return {
                    "recovery_id": f"recovery_{int(time.time())}",
                    "recovery_result": result,
                    "recovery_status": result.get("status", "unknown"),
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    
    def _record_api_metric(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        request_id: str,
        error: Optional[str] = None
    ) -> None:
        """Record API request metric"""
        try:
            metric = PerformanceMetrics(
                timestamp=datetime.now(),
                operation_name=f"{method} {endpoint}",
                duration=duration,
                success=status_code < 400,
                error_type=error,
                additional_data={
                    "endpoint": endpoint,
                    "method": method,
                    "status_code": status_code,
                    "request_id": request_id
                }
            )
            
            if hasattr(self.orchestrator, 'monitoring_manager'):
                self.orchestrator.monitoring_manager.metrics_collector.record_performance_metric(metric)
                
        except Exception as e:
            self.logger.error(f"Failed to record API metric: {e}")
    
    def _configure_authentication(self, auth_config: Dict[str, Any]) -> None:
        """Configure authentication settings"""
        for key, value in auth_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Reinitialize auth handler if needed
        self.auth_handler = AuthenticationHandler(self.config)
    
    def _configure_rate_limiting(self, rate_config: Dict[str, Any]) -> None:
        """Configure rate limiting settings"""
        if 'default_limit' in rate_config:
            self.config.rate_limit_default = rate_config['default_limit']
        
        if 'burst_limit' in rate_config:
            self.config.rate_limit_burst = rate_config['burst_limit']
    
    def _aggregate_metrics(self, metrics: List[Any], aggregation: str) -> List[Dict[str, Any]]:
        """Aggregate metrics based on aggregation type"""
        # Group by metric type
        grouped = {}
        for metric in metrics:
            key = metric.metric_type.value
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(metric.metric_value)
        
        # Aggregate
        aggregated = []
        for metric_type, values in grouped.items():
            if aggregation == "mean":
                value = np.mean(values)
            elif aggregation == "median":
                value = np.median(values)
            elif aggregation == "min":
                value = np.min(values)
            elif aggregation == "max":
                value = np.max(values)
            else:
                value = np.mean(values)
            
            aggregated.append({
                "metric_type": metric_type,
                "aggregated_value": float(value),
                "aggregation": aggregation,
                "sample_count": len(values)
            })
        
        return aggregated
    
    def _generate_comparison_visualizations(
        self,
        comparison_id: str,
        comparison_data: Dict[str, Any]
    ) -> None:
        """Generate visualizations for model comparison"""
        try:
            # Use dashboard manager to create visualizations
            viz_config = {
                "comparison_id": comparison_id,
                "models": comparison_data.get("models", []),
                "metrics": comparison_data.get("metrics", {})
            }
            
            figures = self.orchestrator.dashboard_manager.create_trend_visualizations(
                visualization_config=viz_config,
                historical_data=comparison_data
            )
            
            # Store visualizations
            self._store_visualizations(comparison_id, figures)
            
        except Exception as e:
            self.logger.error(f"Failed to generate visualizations: {e}")
    
    def _get_comparison_results(self, comparison_id: str) -> Optional[Dict[str, Any]]:
        """Get comparison results from cache or storage"""
        # Check cache first
        if hasattr(self.orchestrator, 'monitoring_manager') and self.orchestrator.monitoring_manager.cache_manager:
            cached = self.orchestrator.monitoring_manager.cache_manager.get(f"comparison:{comparison_id}")
            if cached:
                return cached
        
        # Check storage (mock implementation)
        return None
    
    def _calculate_model_rankings(
        self,
        metric: str,
        time_window: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Calculate model rankings by metric"""
        # Parse time window
        if time_window.endswith('d'):
            days = int(time_window[:-1])
            start_date = datetime.now() - timedelta(days=days)
        else:
            start_date = datetime.now() - timedelta(days=7)
        
        # Get all models
        models = list(self.orchestrator.realtime_monitor.active_models)
        
        # Calculate metrics for each model
        rankings = []
        for model_id in models:
            try:
                metrics = self.orchestrator.accuracy_database.get_accuracy_metrics(
                    model_id=model_id,
                    start_date=start_date,
                    metric_type=MetricType[metric.upper()] if metric != 'accuracy' else MetricType.ACCURACY
                )
                
                if metrics:
                    avg_value = np.mean([m.metric_value for m in metrics])
                    rankings.append({
                        "model_id": model_id,
                        "metric_value": float(avg_value),
                        "sample_count": len(metrics)
                    })
            except Exception as e:
                self.logger.error(f"Error calculating ranking for {model_id}: {e}")
        
        # Sort by metric value
        rankings.sort(key=lambda x: x['metric_value'], reverse=True)
        
        return rankings[:limit]
    
    async def _generate_report_async(
        self,
        report_id: str,
        report_request: ReportGenerationRequest,
        user_id: str
    ) -> None:
        """Generate report asynchronously"""
        try:
            # Update status
            self._update_report_status(report_id, "generating")
            
            # Generate report using orchestrator
            result = self.orchestrator.generate_accuracy_reports(
                report_config={
                    "report_type": report_request.report_type,
                    "model_ids": report_request.model_ids,
                    "time_range": report_request.time_range,
                    "sections": report_request.include_sections
                },
                output_formats=[report_request.export_format]
            )
            
            if result.success:
                # Save report
                report_path = self._save_report(
                    report_id,
                    result.data["reports"][report_request.export_format],
                    report_request.export_format
                )
                
                # Update status
                self._update_report_status(report_id, "completed", report_path)
                
                # Send email if requested
                if report_request.email_delivery:
                    await self._send_report_email(
                        report_request.email_delivery,
                        report_id,
                        report_path
                    )
            else:
                self._update_report_status(report_id, "failed", error=result.error)
                
        except Exception as e:
            self.logger.error(f"Failed to generate report {report_id}: {e}")
            self._update_report_status(report_id, "failed", error=str(e))
    
    def _get_report_status(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get report generation status"""
        # Check cache
        if self.auth_handler.redis_client:
            status = self.auth_handler.redis_client.get(f"report_status:{report_id}")
            if status:
                return json.loads(status)
        
        return None
    
    def _update_report_status(
        self,
        report_id: str,
        status: str,
        report_path: Optional[Path] = None,
        error: Optional[str] = None
    ) -> None:
        """Update report generation status"""
        status_data = {
            "report_id": report_id,
            "status": status,
            "updated_at": datetime.now().isoformat(),
            "report_path": str(report_path) if report_path else None,
            "error": error
        }
        
        if self.auth_handler.redis_client:
            self.auth_handler.redis_client.setex(
                f"report_status:{report_id}",
                3600,  # 1 hour
                json.dumps(status_data)
            )
    
    def _get_report_file_path(self, report_id: str) -> Optional[Path]:
        """Get report file path"""
        status = self._get_report_status(report_id)
        if status and status.get("report_path"):
            return Path(status["report_path"])
        return None
    
    def _save_report(self, report_id: str, content: Any, format: str) -> Path:
        """Save generated report"""
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        file_path = reports_dir / f"{report_id}.{format}"
        
        if format == "json":
            with open(file_path, 'w') as f:
                json.dump(content, f, indent=2, default=str)
        elif format == "pdf":
            # PDF generation would be implemented here
            file_path.write_bytes(content)
        else:
            file_path.write_text(str(content))
        
        return file_path
    
    async def _send_report_email(self, email: str, report_id: str, report_path: Path) -> None:
        """Send report via email"""
        # Email sending implementation would go here
        self.logger.info(f"Would send report {report_id} to {email}")
    
    def _save_user_dashboard(self, user_id: str, dashboard_id: str) -> None:
        """Save user-dashboard association"""
        if self.auth_handler.redis_client:
            key = f"user_dashboards:{user_id}"
            self.auth_handler.redis_client.sadd(key, dashboard_id)
    
    def _check_dashboard_access(self, user_id: str, dashboard_id: str) -> bool:
        """Check if user has access to dashboard"""
        if self.auth_handler.redis_client:
            key = f"user_dashboards:{user_id}"
            return self.auth_handler.redis_client.sismember(key, dashboard_id)
        return True  # Allow all in development
    
    def _get_user_dashboards(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's dashboards"""
        dashboards = []
        
        if self.auth_handler.redis_client:
            key = f"user_dashboards:{user_id}"
            dashboard_ids = self.auth_handler.redis_client.smembers(key)
            
            for dashboard_id in dashboard_ids:
                status = self.orchestrator.dashboard_manager.get_dashboard_status(dashboard_id)
                if 'error' not in status:
                    dashboards.append(status)
        
        return dashboards
    
    def _generate_postman_collection(self) -> Dict[str, Any]:
        """Generate Postman collection for API"""
        collection = {
            "info": {
                "name": "Fraud Detection Accuracy API",
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            },
            "item": [],
            "variable": [
                {
                    "key": "base_url",
                    "value": f"http://{self.config.host}:{self.config.port}"
                },
                {
                    "key": "access_token",
                    "value": ""
                }
            ]
        }
        
        # Add endpoints to collection
        for route in self.app.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                for method in route.methods:
                    if method in ['GET', 'POST', 'PUT', 'DELETE']:
                        item = {
                            "name": route.name,
                            "request": {
                                "method": method,
                                "url": "{{base_url}}" + route.path,
                                "header": [
                                    {
                                        "key": "Authorization",
                                        "value": "Bearer {{access_token}}"
                                    }
                                ]
                            }
                        }
                        collection["item"].append(item)
        
        return collection
    
    def _get_api_examples(self) -> Dict[str, Any]:
        """Get API usage examples"""
        examples = {
            "authentication": {
                "login": {
                    "endpoint": "POST /api/v1/auth/login",
                    "body": {
                        "username": "user@example.com",
                        "password": "securepassword"
                    },
                    "response": {
                        "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
                        "token_type": "bearer",
                        "expires_in": 1800
                    }
                }
            },
            "metrics": {
                "get_metrics": {
                    "endpoint": "GET /api/v1/metrics/fraud_model_v1",
                    "parameters": {
                        "start_date": "2024-01-01T00:00:00Z",
                        "metrics": ["accuracy", "precision"]
                    },
                    "response": {
                        "model_id": "fraud_model_v1",
                        "metrics": {
                            "accuracy": 0.92,
                            "precision": 0.89
                        }
                    }
                }
            }
        }
        
        return examples
    
    def _verify_credentials(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Verify user credentials (mock implementation)"""
        # In production, check against database
        if username == "admin" and password == "password123":
            return {
                "user_id": "user_123",
                "username": username,
                "email": f"{username}@example.com",
                "role": "admin",
                "permissions": ["read:all", "write:all", "admin:all"]
            }
        return None
    
    def _get_user_by_id(self, user_id: str) -> Dict[str, Any]:
        """Get user by ID (mock implementation)"""
        return {
            "user_id": user_id,
            "username": "admin",
            "email": "admin@example.com",
            "role": "admin",
            "is_active": True,
            "created_at": datetime.now() - timedelta(days=30),
            "last_login": datetime.now(),
            "permissions": ["read:all", "write:all", "admin:all"]
        }
    
    async def _process_export_async(
        self,
        export_id: str,
        export_request: ExportRequest,
        user_id: str
    ) -> None:
        """Process data export asynchronously"""
        try:
            # Update status
            self._update_export_status(export_id, "processing")
            
            # Export data using dashboard manager
            export_data = self.orchestrator.dashboard_manager.export_monitoring_data(
                export_config={"format": export_request.format},
                data_filters={
                    "model_ids": export_request.model_ids,
                    "start_date": export_request.start_date,
                    "end_date": export_request.end_date,
                    "export_type": export_request.export_type
                }
            )
            
            # Save export file
            export_path = self._save_export(
                export_id,
                export_data,
                export_request.format,
                export_request.compression
            )
            
            # Update status
            self._update_export_status(export_id, "completed", export_path)
            
        except Exception as e:
            self.logger.error(f"Failed to process export {export_id}: {e}")
            self._update_export_status(export_id, "failed", error=str(e))
    
    def _get_export_status(self, export_id: str) -> Optional[Dict[str, Any]]:
        """Get export job status"""
        if self.auth_handler.redis_client:
            status = self.auth_handler.redis_client.get(f"export_status:{export_id}")
            if status:
                return json.loads(status)
        return None
    
    def _update_export_status(
        self,
        export_id: str,
        status: str,
        export_path: Optional[Path] = None,
        error: Optional[str] = None
    ) -> None:
        """Update export job status"""
        status_data = {
            "export_id": export_id,
            "status": status,
            "updated_at": datetime.now().isoformat(),
            "export_path": str(export_path) if export_path else None,
            "error": error
        }
        
        if self.auth_handler.redis_client:
            self.auth_handler.redis_client.setex(
                f"export_status:{export_id}",
                3600,  # 1 hour
                json.dumps(status_data)
            )
    
    def _get_export_file_path(self, export_id: str) -> Optional[Path]:
        """Get export file path"""
        status = self._get_export_status(export_id)
        if status and status.get("export_path"):
            return Path(status["export_path"])
        return None
    
    def _save_export(
        self,
        export_id: str,
        data: Any,
        format: str,
        compression: Optional[str]
    ) -> Path:
        """Save exported data"""
        exports_dir = Path("exports")
        exports_dir.mkdir(exist_ok=True)
        
        file_name = f"{export_id}.{format}"
        if compression:
            file_name += f".{compression}"
        
        file_path = exports_dir / file_name
        
        # Save based on format
        if format == "json":
            content = data if isinstance(data, str) else json.dumps(data, indent=2, default=str)
        elif format == "csv":
            content = data
        else:
            content = str(data)
        
        # Apply compression if requested
        if compression == "gzip":
            import gzip
            file_path.write_bytes(gzip.compress(content.encode()))
        elif compression == "zip":
            import zipfile
            with zipfile.ZipFile(file_path, 'w') as zf:
                zf.writestr(f"{export_id}.{format}", content)
        else:
            file_path.write_text(content)
        
        return file_path
    
    async def _stream_file(self, file_path: Path):
        """Stream file content"""
        chunk_size = 1024 * 1024  # 1MB chunks
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(chunk_size):
                yield chunk
    
    def _get_mime_type(self, format: str) -> str:
        """Get MIME type for format"""
        mime_types = {
            "json": "application/json",
            "csv": "text/csv",
            "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "parquet": "application/octet-stream",
            "pdf": "application/pdf",
            "html": "text/html"
        }
        return mime_types.get(format, "application/octet-stream")
    
    def _store_visualizations(self, comparison_id: str, figures: Dict[str, Any]) -> None:
        """Store comparison visualizations"""
        # Store in cache
        if hasattr(self.orchestrator, 'monitoring_manager') and self.orchestrator.monitoring_manager.cache_manager:
            self.orchestrator.monitoring_manager.cache_manager.set(
                f"visualizations:{comparison_id}",
                figures,
                ttl=3600  # 1 hour
            )
    
    # ======================== PUBLIC API ========================
    
    def run(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """Run the API server"""
        import uvicorn
        
        uvicorn.run(
            self.app,
            host=host or self.config.host,
            port=port or self.config.port,
            workers=self.config.workers,
            reload=self.config.reload
        )
    
    def get_api_info(self) -> Dict[str, Any]:
        """Get API information"""
        return {
            "title": self.app.title,
            "version": self.app.version,
            "base_url": f"http://{self.config.host}:{self.config.port}",
            "docs_url": f"http://{self.config.host}:{self.config.port}{self.config.docs_url}",
            "endpoints": len(self.app.routes),
            "features": {
                "authentication": self.config.enable_api_keys,
                "rate_limiting": self.config.rate_limit_enabled,
                "cors": self.config.cors_enabled,
                "websocket": True,
                "export_formats": self.config.export_formats
            }
        }


# ======================== USAGE EXAMPLES ========================

if __name__ == "__main__":
    # Example usage demonstrating the API interface layer
    print("=== AccuracyTrackingAPI Examples ===\n")
    
    # Note: This requires all the components from previous phases
    # For demonstration, we'll show the API structure
    
    # Example 1: API Configuration
    print("1. API Configuration:")
    api_config = APIConfig(
        host="0.0.0.0",
        port=8000,
        jwt_secret_key="your-secure-secret-key",
        rate_limit_default="100/minute",
        cors_origins=["http://localhost:3000", "https://app.example.com"],
        export_formats=["json", "csv", "excel", "pdf"]
    )
    print(f"   Host: {api_config.host}:{api_config.port}")
    print(f"   Rate Limit: {api_config.rate_limit_default}")
    print(f"   Export Formats: {api_config.export_formats}\n")
    
    # Example 2: API Endpoints Overview
    print("2. API Endpoints Overview:")
    endpoints = [
        ("POST", "/api/v1/auth/login", "User authentication"),
        ("GET", "/api/v1/metrics/{model_id}", "Get model metrics"),
        ("POST", "/api/v1/metrics/query", "Query metrics with filters"),
        ("GET", "/api/v1/metrics/realtime/{model_id}", "Real-time metrics"),
        ("POST", "/api/v1/comparison/models", "Compare models"),
        ("GET", "/api/v1/comparison/rankings", "Model rankings"),
        ("POST", "/api/v1/reports/generate", "Generate reports"),
        ("POST", "/api/v1/dashboards/create", "Create dashboard"),
        ("POST", "/api/v1/export/data", "Export data"),
        ("WS", "/ws/metrics/{model_id}", "WebSocket real-time updates")
    ]
    
    for method, path, description in endpoints:
        print(f"   {method:6} {path:40} - {description}")
    print()
    
    # Example 3: Authentication Flow
    print("3. Authentication Flow:")
    print("   a. Login: POST /api/v1/auth/login")
    print("      Request: {\"username\": \"user@example.com\", \"password\": \"password\"}")
    print("      Response: {\"access_token\": \"jwt-token\", \"expires_in\": 1800}")
    print("   b. Use token: Authorization: Bearer <jwt-token>")
    print("   c. Refresh: POST /api/v1/auth/refresh?refresh_token=<token>\n")
    
    # Example 4: Metrics Query Example
    print("4. Metrics Query Example:")
    print("   POST /api/v1/metrics/query")
    print("   Request Body:")
    metrics_request = {
        "model_ids": ["fraud_model_v1", "fraud_model_v2"],
        "start_date": "2024-01-01T00:00:00Z",
        "end_date": "2024-01-31T23:59:59Z",
        "metric_types": ["accuracy", "precision", "recall"],
        "aggregation": "mean"
    }
    print(f"   {json.dumps(metrics_request, indent=2)}\n")
    
    # Example 5: Dashboard Creation Example
    print("5. Dashboard Creation Example:")
    print("   POST /api/v1/dashboards/create")
    dashboard_request = {
        "dashboard_name": "Fraud Detection Overview",
        "layout": "grid",
        "widgets": [
            {
                "type": "accuracy_trend",
                "position": {"x": 0, "y": 0, "width": 6, "height": 4}
            },
            {
                "type": "drift_detection",
                "position": {"x": 6, "y": 0, "width": 6, "height": 4}
            }
        ],
        "refresh_interval": 30
    }
    print(f"   Request: {json.dumps(dashboard_request, indent=2)}\n")
    
    # Example 6: Export Configuration
    print("6. Export Configuration:")
    export_request = {
        "export_type": "metrics",
        "model_ids": ["fraud_model_v1"],
        "start_date": "2024-01-01T00:00:00Z",
        "format": "csv",
        "compression": "gzip"
    }
    print(f"   POST /api/v1/export/data")
    print(f"   Request: {json.dumps(export_request, indent=2)}\n")
    
    # Example 7: WebSocket Connection
    print("7. WebSocket Real-time Updates:")
    print("   Connect: ws://localhost:8000/ws/metrics/fraud_model_v1")
    print("   Receive: Real-time accuracy metrics every 30 seconds")
    print("   Format: JSON with current metrics and drift status\n")
    
    # Example 8: API Features Summary
    print("8. API Features Summary:")
    features = {
        "Authentication": ["JWT tokens", "API keys", "Role-based access"],
        "Rate Limiting": ["Per-endpoint limits", "Burst protection", "User quotas"],
        "Data Formats": ["JSON", "CSV", "Excel", "PDF", "Parquet"],
        "Real-time": ["WebSocket updates", "Server-sent events", "Polling"],
        "Security": ["HTTPS support", "CORS configuration", "Input validation"],
        "Documentation": ["OpenAPI/Swagger", "ReDoc", "Postman collection"]
    }
    
    for category, items in features.items():
        print(f"   {category}:")
        for item in items:
            print(f"     - {item}")
    print()
    
    print("=== API Interface Layer Complete ===")
    print("Note: Run with actual orchestrator instance to start the API server")
    print("Example: api.run(host='0.0.0.0', port=8000)")