"""
Enhanced Proof Verifier - Chunk 1: Analysis and Integration Utilities
Comprehensive integration utilities and system analysis for the enhanced proof verifier,
providing seamless integration with existing Saraphis components and advanced analysis capabilities.
"""

import logging
import json
import hashlib
import time
import threading
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Type, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
import traceback
import psutil
import numpy as np
import pandas as pd

# Import existing proof verifier components
try:
    from enhanced_proof_verifier import (
        FinancialProofVerifier as EnhancedFinancialProofVerifier,
        EnhancedProofClaim, EnhancedProofEvidence, EnhancedProofResult,
        SecurityLevel, ProofVerificationException, ProofConfigurationError,
        ProofGenerationError, ProofValidationError, ProofTimeoutError,
        ProofSecurityError, ProofIntegrityError, ClaimValidationError,
        EvidenceValidationError, ProofSystemError, ProofStorageError,
        CryptographicError, ProofExpiredError, ResourceLimitError,
        SecurityValidator, ResourceMonitor
    )
    from proof_verifier import (
        ProofType, ProofStatus, ProofLevel, ProofClaim, ProofEvidence, ProofResult
    )
    PROOF_COMPONENTS = True
except ImportError as e:
    PROOF_COMPONENTS = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Proof verifier components not available: {e}")

# Import existing system components for integration
try:
    from enhanced_data_validator import EnhancedFinancialDataValidator
    from enhanced_transaction_validator import EnhancedTransactionFieldValidator
    from enhanced_data_loader import EnhancedFinancialDataLoader
    from enhanced_preprocessing_integration import PreprocessingIntegrationManager
    from enhanced_ml_integration import MLIntegrationManager
    SYSTEM_COMPONENTS = True
except ImportError as e:
    SYSTEM_COMPONENTS = False
    logger = logging.getLogger(__name__)
    logger.warning(f"System components not available: {e}")

# Configure logging
logger = logging.getLogger(__name__)

# ======================== PROOF INTEGRATION CONFIGURATION ========================

@dataclass
class ProofIntegrationConfig:
    """Configuration for proof verifier integration with system components"""
    
    # Core integration settings
    enable_data_validation: bool = True
    enable_preprocessing: bool = True
    enable_ml_integration: bool = True
    enable_transaction_validation: bool = True
    enable_real_time_verification: bool = True
    
    # Performance settings
    max_worker_threads: int = 4
    max_worker_processes: int = 2
    proof_timeout: int = 60  # seconds
    batch_size: int = 100
    cache_size: int = 1000
    
    # Security settings
    security_level: SecurityLevel = SecurityLevel.HIGH if PROOF_COMPONENTS else None
    enable_cryptographic_proofs: bool = True
    enable_audit_logging: bool = True
    enable_access_control: bool = True
    
    # Monitoring settings
    enable_performance_monitoring: bool = True
    enable_resource_monitoring: bool = True
    enable_health_checks: bool = True
    monitoring_interval: int = 60  # seconds
    
    # Integration settings
    integrate_with_validator: bool = True
    integrate_with_loader: bool = True
    integrate_with_preprocessor: bool = True
    integrate_with_ml_predictor: bool = True
    
    # Proof system settings
    enable_rule_based_proofs: bool = True
    enable_ml_based_proofs: bool = True
    enable_cryptographic_proofs_system: bool = True
    enable_composite_proofs: bool = True
    
    # Fallback settings
    enable_fallback_mode: bool = True
    fallback_to_basic_validation: bool = True
    fallback_timeout: int = 30  # seconds
    
    # Testing settings
    enable_test_mode: bool = False
    test_data_size: int = 100
    mock_external_services: bool = False

class ProofIntegrationError(ProofVerificationException):
    """Raised when proof integration fails"""
    def __init__(self, message: str, component: str = None, **kwargs):
        super().__init__(message, error_code="PROOF_INTEGRATION_ERROR", **kwargs)
        self.component = component
        self.recoverable = True

class ProofSystemAnalysisError(ProofVerificationException):
    """Raised when proof system analysis fails"""
    def __init__(self, message: str, analysis_type: str = None, **kwargs):
        super().__init__(message, error_code="PROOF_ANALYSIS_ERROR", **kwargs)
        self.analysis_type = analysis_type
        self.recoverable = True

# ======================== PROOF SYSTEM ANALYZER ========================

class ProofSystemAnalyzer:
    """Analyzes proof system performance, reliability, and integration health"""
    
    def __init__(self, config: ProofIntegrationConfig):
        self.config = config
        self.analysis_history = []
        self.performance_metrics = {}
        self.health_status = {}
        self.lock = threading.Lock()
        
    def analyze_system_health(self, proof_verifier: Any) -> Dict[str, Any]:
        """Analyze overall system health and performance"""
        try:
            analysis_start = time.time()
            
            health_report = {
                'timestamp': datetime.now().isoformat(),
                'system_status': 'healthy',
                'components': {},
                'performance': {},
                'recommendations': [],
                'issues': []
            }
            
            # Analyze proof verifier health
            if proof_verifier:
                verifier_health = self._analyze_verifier_health(proof_verifier)
                health_report['components']['proof_verifier'] = verifier_health
                
                # Check for issues
                if verifier_health.get('error_rate', 0) > 0.1:
                    health_report['issues'].append({
                        'component': 'proof_verifier',
                        'severity': 'high',
                        'message': f"High error rate: {verifier_health.get('error_rate', 0):.2%}",
                        'recommendation': 'Review error logs and check system resources'
                    })
            
            # Analyze resource usage
            resource_health = self._analyze_resource_health()
            health_report['components']['resources'] = resource_health
            
            # Check resource issues
            if resource_health.get('memory_usage', 0) > 0.8:
                health_report['issues'].append({
                    'component': 'resources',
                    'severity': 'medium',
                    'message': f"High memory usage: {resource_health.get('memory_usage', 0):.1%}",
                    'recommendation': 'Consider increasing memory limits or optimizing proof caching'
                })
            
            # Analyze integration health
            integration_health = self._analyze_integration_health()
            health_report['components']['integration'] = integration_health
            
            # Performance analysis
            performance_analysis = self._analyze_performance_metrics()
            health_report['performance'] = performance_analysis
            
            # Generate recommendations
            recommendations = self._generate_health_recommendations(health_report)
            health_report['recommendations'] = recommendations
            
            # Determine overall system status
            if len(health_report['issues']) == 0:
                health_report['system_status'] = 'healthy'
            elif any(issue['severity'] == 'high' for issue in health_report['issues']):
                health_report['system_status'] = 'degraded'
            else:
                health_report['system_status'] = 'warning'
            
            analysis_duration = time.time() - analysis_start
            health_report['analysis_duration'] = analysis_duration
            
            # Store analysis history
            with self.lock:
                self.analysis_history.append(health_report)
                
                # Keep only last 100 analyses
                if len(self.analysis_history) > 100:
                    self.analysis_history = self.analysis_history[-100:]
            
            logger.info(f"System health analysis completed in {analysis_duration:.2f}s - Status: {health_report['system_status']}")
            return health_report
            
        except Exception as e:
            logger.error(f"System health analysis failed: {e}")
            raise ProofSystemAnalysisError(f"Health analysis failed: {e}", analysis_type="system_health")
    
    def _analyze_verifier_health(self, proof_verifier: Any) -> Dict[str, Any]:
        """Analyze proof verifier specific health"""
        try:
            verifier_health = {
                'status': 'unknown',
                'total_proofs': 0,
                'successful_proofs': 0,
                'failed_proofs': 0,
                'error_rate': 0.0,
                'average_processing_time': 0.0,
                'cache_hit_rate': 0.0,
                'last_activity': None
            }
            
            # Get verifier statistics if available
            if hasattr(proof_verifier, 'get_statistics'):
                stats = proof_verifier.get_statistics()
                verifier_health.update({
                    'total_proofs': stats.get('total_proofs', 0),
                    'successful_proofs': stats.get('successful_proofs', 0),
                    'failed_proofs': stats.get('failed_proofs', 0),
                    'average_processing_time': stats.get('average_processing_time', 0.0),
                    'cache_hit_rate': stats.get('cache_hit_rate', 0.0),
                    'last_activity': stats.get('last_activity')
                })
                
                # Calculate error rate
                total_proofs = verifier_health['total_proofs']
                if total_proofs > 0:
                    verifier_health['error_rate'] = verifier_health['failed_proofs'] / total_proofs
            
            # Determine status
            if verifier_health['error_rate'] > 0.2:
                verifier_health['status'] = 'unhealthy'
            elif verifier_health['error_rate'] > 0.1:
                verifier_health['status'] = 'degraded'
            else:
                verifier_health['status'] = 'healthy'
            
            return verifier_health
            
        except Exception as e:
            logger.error(f"Verifier health analysis failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _analyze_resource_health(self) -> Dict[str, Any]:
        """Analyze system resource health"""
        try:
            # Get system resource information
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            resource_health = {
                'cpu_usage': cpu_percent / 100.0,
                'memory_usage': memory.percent / 100.0,
                'disk_usage': disk.percent / 100.0,
                'available_memory': memory.available,
                'total_memory': memory.total,
                'free_disk_space': disk.free,
                'total_disk_space': disk.total,
                'status': 'healthy'
            }
            
            # Determine resource status
            if (resource_health['cpu_usage'] > 0.9 or 
                resource_health['memory_usage'] > 0.9 or 
                resource_health['disk_usage'] > 0.9):
                resource_health['status'] = 'critical'
            elif (resource_health['cpu_usage'] > 0.7 or 
                  resource_health['memory_usage'] > 0.8 or 
                  resource_health['disk_usage'] > 0.8):
                resource_health['status'] = 'warning'
            
            return resource_health
            
        except Exception as e:
            logger.error(f"Resource health analysis failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _analyze_integration_health(self) -> Dict[str, Any]:
        """Analyze integration component health"""
        try:
            integration_health = {
                'components_available': {
                    'proof_components': PROOF_COMPONENTS,
                    'system_components': SYSTEM_COMPONENTS
                },
                'integration_status': {},
                'connectivity_issues': [],
                'status': 'healthy'
            }
            
            # Check component availability
            if not PROOF_COMPONENTS:
                integration_health['connectivity_issues'].append({
                    'component': 'proof_components',
                    'severity': 'high',
                    'message': 'Proof verifier components not available'
                })
            
            if not SYSTEM_COMPONENTS:
                integration_health['connectivity_issues'].append({
                    'component': 'system_components',
                    'severity': 'medium',
                    'message': 'System integration components not available'
                })
            
            # Determine integration status
            if len(integration_health['connectivity_issues']) > 0:
                high_severity_issues = [i for i in integration_health['connectivity_issues'] 
                                      if i['severity'] == 'high']
                if high_severity_issues:
                    integration_health['status'] = 'degraded'
                else:
                    integration_health['status'] = 'warning'
            
            return integration_health
            
        except Exception as e:
            logger.error(f"Integration health analysis failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _analyze_performance_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics"""
        try:
            performance_analysis = {
                'response_times': {
                    'average': 0.0,
                    'median': 0.0,
                    'p95': 0.0,
                    'p99': 0.0
                },
                'throughput': {
                    'proofs_per_second': 0.0,
                    'proofs_per_minute': 0.0
                },
                'error_rates': {
                    'overall': 0.0,
                    'by_type': {}
                },
                'trends': {
                    'response_time_trend': 'stable',
                    'throughput_trend': 'stable',
                    'error_rate_trend': 'stable'
                }
            }
            
            # Calculate metrics from stored performance data
            if hasattr(self, 'performance_metrics') and self.performance_metrics:
                # Response time analysis
                response_times = [m.get('response_time', 0) for m in self.performance_metrics.values()]
                if response_times:
                    performance_analysis['response_times']['average'] = np.mean(response_times)
                    performance_analysis['response_times']['median'] = np.median(response_times)
                    performance_analysis['response_times']['p95'] = np.percentile(response_times, 95)
                    performance_analysis['response_times']['p99'] = np.percentile(response_times, 99)
                
                # Throughput analysis
                recent_metrics = [m for m in self.performance_metrics.values() 
                                if m.get('timestamp') and 
                                datetime.fromisoformat(m['timestamp']) > datetime.now() - timedelta(minutes=5)]
                
                if recent_metrics:
                    proofs_in_last_5min = sum(m.get('proofs_processed', 0) for m in recent_metrics)
                    performance_analysis['throughput']['proofs_per_minute'] = proofs_in_last_5min
                    performance_analysis['throughput']['proofs_per_second'] = proofs_in_last_5min / 300
            
            return performance_analysis
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {'error': str(e)}
    
    def _generate_health_recommendations(self, health_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate health recommendations based on analysis"""
        recommendations = []
        
        try:
            # Performance recommendations
            if health_report['performance'].get('response_times', {}).get('average', 0) > 5.0:
                recommendations.append({
                    'category': 'performance',
                    'priority': 'high',
                    'title': 'Optimize Response Times',
                    'description': 'Average response time is above 5 seconds',
                    'action': 'Consider enabling proof caching or increasing worker threads'
                })
            
            # Resource recommendations
            resource_status = health_report['components'].get('resources', {})
            if resource_status.get('memory_usage', 0) > 0.8:
                recommendations.append({
                    'category': 'resources',
                    'priority': 'medium',
                    'title': 'Optimize Memory Usage',
                    'description': 'Memory usage is above 80%',
                    'action': 'Consider increasing memory limits or optimizing proof caching'
                })
            
            # Integration recommendations
            integration_issues = health_report['components'].get('integration', {}).get('connectivity_issues', [])
            for issue in integration_issues:
                if issue['severity'] == 'high':
                    recommendations.append({
                        'category': 'integration',
                        'priority': 'high',
                        'title': f'Fix {issue["component"]} Integration',
                        'description': issue['message'],
                        'action': 'Check component availability and configuration'
                    })
            
            # Error rate recommendations
            verifier_health = health_report['components'].get('proof_verifier', {})
            if verifier_health.get('error_rate', 0) > 0.05:
                recommendations.append({
                    'category': 'reliability',
                    'priority': 'high',
                    'title': 'Reduce Error Rate',
                    'description': f'Error rate is {verifier_health.get("error_rate", 0):.2%}',
                    'action': 'Review error logs and implement additional validation'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []
    
    def get_analysis_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analysis history"""
        with self.lock:
            return self.analysis_history[-limit:] if self.analysis_history else []
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_analyses = [
                analysis for analysis in self.analysis_history
                if datetime.fromisoformat(analysis['timestamp']) > cutoff_time
            ]
            
            if not recent_analyses:
                return {'error': 'No recent analysis data available'}
            
            # Extract trend data
            timestamps = [datetime.fromisoformat(a['timestamp']) for a in recent_analyses]
            response_times = [a['performance'].get('response_times', {}).get('average', 0) for a in recent_analyses]
            error_rates = [a['components'].get('proof_verifier', {}).get('error_rate', 0) for a in recent_analyses]
            
            trends = {
                'time_range': {
                    'start': min(timestamps).isoformat(),
                    'end': max(timestamps).isoformat(),
                    'hours': hours
                },
                'response_time_trend': {
                    'values': response_times,
                    'trend': self._calculate_trend(response_times),
                    'average': np.mean(response_times) if response_times else 0
                },
                'error_rate_trend': {
                    'values': error_rates,
                    'trend': self._calculate_trend(error_rates),
                    'average': np.mean(error_rates) if error_rates else 0
                }
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Performance trend analysis failed: {e}")
            return {'error': str(e)}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values"""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine trend
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'

# ======================== PROOF INTEGRATION MANAGER ========================

class ProofIntegrationManager:
    """Manages integration between proof verifier and other system components"""
    
    def __init__(self, config: ProofIntegrationConfig):
        self.config = config
        self.proof_verifier = None
        self.data_validator = None
        self.transaction_validator = None
        self.data_loader = None
        self.preprocessor = None
        self.ml_integration = None
        self.analyzer = ProofSystemAnalyzer(config)
        self.integration_state = {}
        self.lock = threading.Lock()
        self.executor = None
        
        # Initialize components
        self._initialize_components()
        
        # Start monitoring if enabled
        if self.config.enable_performance_monitoring:
            self._start_monitoring()
    
    def _initialize_components(self):
        """Initialize all integration components"""
        try:
            # Initialize proof verifier
            if PROOF_COMPONENTS:
                self.proof_verifier = EnhancedFinancialProofVerifier(
                    security_level=self.config.security_level,
                    enable_cryptographic_proofs=self.config.enable_cryptographic_proofs
                )
                
                # Initialize supporting components
                if SYSTEM_COMPONENTS:
                    if self.config.integrate_with_validator:
                        self.data_validator = EnhancedFinancialDataValidator()
                        self.transaction_validator = EnhancedTransactionFieldValidator()
                    
                    if self.config.integrate_with_loader:
                        self.data_loader = EnhancedFinancialDataLoader()
                    
                    if self.config.integrate_with_preprocessor:
                        from enhanced_preprocessing_integration import get_integrated_preprocessor
                        self.preprocessor = get_integrated_preprocessor("production")
                    
                    if self.config.integrate_with_ml_predictor:
                        from enhanced_ml_integration import get_integrated_ml_system
                        self.ml_integration = get_integrated_ml_system("production")
                
                # Initialize thread pool
                self.executor = ThreadPoolExecutor(max_workers=self.config.max_worker_threads)
                
                logger.info("Proof integration components initialized successfully")
            else:
                logger.warning("Proof components not available")
                
        except Exception as e:
            logger.error(f"Failed to initialize proof integration components: {e}")
            raise ProofIntegrationError(f"Component initialization failed: {e}", component="initialization")
    
    def _start_monitoring(self):
        """Start background monitoring tasks"""
        if self.config.enable_performance_monitoring:
            monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            monitoring_thread.start()
            logger.info("Proof integration monitoring started")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                # Perform system health analysis
                if self.proof_verifier:
                    health_report = self.analyzer.analyze_system_health(self.proof_verifier)
                    
                    # Log health status
                    status = health_report.get('system_status', 'unknown')
                    logger.info(f"System health: {status}")
                    
                    # Handle critical issues
                    if status == 'degraded':
                        logger.warning("System is degraded, consider investigating issues")
                    elif status == 'critical':
                        logger.error("System is in critical state, immediate attention required")
                
                # Sleep until next monitoring cycle
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait before retrying
    
    @contextmanager
    def proof_session(self, session_id: str = None):
        """Context manager for proof processing sessions"""
        session_id = session_id or f"proof_session_{int(time.time())}"
        
        try:
            with self.lock:
                self.integration_state[session_id] = {
                    'start_time': datetime.now(),
                    'status': 'active',
                    'operations': [],
                    'proofs_processed': 0
                }
            
            logger.info(f"Starting proof session: {session_id}")
            yield session_id
            
        except Exception as e:
            logger.error(f"Proof session {session_id} failed: {e}")
            if session_id in self.integration_state:
                self.integration_state[session_id]['status'] = 'failed'
                self.integration_state[session_id]['error'] = str(e)
            raise
            
        finally:
            if session_id in self.integration_state:
                self.integration_state[session_id]['end_time'] = datetime.now()
                self.integration_state[session_id]['status'] = 'completed'
                self._cleanup_session(session_id)
    
    def _cleanup_session(self, session_id: str):
        """Clean up resources for a proof session"""
        try:
            if session_id in self.integration_state:
                session_data = self.integration_state[session_id]
                
                # Log session summary
                duration = (session_data.get('end_time', datetime.now()) - 
                           session_data['start_time']).total_seconds()
                
                proofs_processed = session_data.get('proofs_processed', 0)
                
                logger.info(f"Proof session {session_id} completed in {duration:.2f}s - "
                           f"Processed {proofs_processed} proofs")
                
                # Keep only recent sessions
                if len(self.integration_state) > 50:
                    oldest_sessions = sorted(
                        self.integration_state.keys(),
                        key=lambda x: self.integration_state[x]['start_time']
                    )[:-25]
                    
                    for old_session in oldest_sessions:
                        del self.integration_state[old_session]
                        
        except Exception as e:
            logger.error(f"Session cleanup failed for {session_id}: {e}")
    
    def process_proof_with_full_integration(self, transaction_data: Dict[str, Any], 
                                          proof_type: str = "transaction_fraud") -> Dict[str, Any]:
        """Process proof with full system integration"""
        with self.proof_session() as session_id:
            try:
                # Step 1: Data validation
                validation_result = None
                if self.config.enable_data_validation and self.data_validator:
                    if isinstance(transaction_data, dict):
                        # Convert to DataFrame for validation
                        df_data = pd.DataFrame([transaction_data])
                        validation_result = self.data_validator.validate_transaction_data(df_data)
                        
                        if not validation_result.is_valid:
                            logger.warning(f"Data validation issues: {len(validation_result.issues)}")
                            
                            # Check for critical issues
                            critical_issues = [i for i in validation_result.issues 
                                             if i.severity.value >= 4]
                            if critical_issues:
                                raise ProofIntegrationError(
                                    f"Critical validation issues: {len(critical_issues)}", 
                                    component="data_validation"
                                )
                
                # Step 2: Transaction field validation
                if self.config.enable_transaction_validation and self.transaction_validator:
                    field_validation = self.transaction_validator.validate_transaction_fields(transaction_data)
                    if not field_validation.is_valid:
                        logger.warning(f"Transaction field validation issues: {len(field_validation.issues)}")
                
                # Step 3: Preprocessing if enabled
                processed_data = transaction_data
                if self.config.enable_preprocessing and self.preprocessor:
                    if isinstance(transaction_data, dict):
                        df_data = pd.DataFrame([transaction_data])
                        processed_df = self.preprocessor.process_data_integrated(df_data)
                        processed_data = processed_df.iloc[0].to_dict()
                
                # Step 4: ML integration for enhanced evidence
                ml_evidence = {}
                if self.config.enable_ml_integration and self.ml_integration:
                    try:
                        if isinstance(processed_data, dict):
                            df_data = pd.DataFrame([processed_data])
                            ml_results = self.ml_integration.process_with_full_integration(df_data)
                            
                            if ml_results:
                                ml_result = ml_results[0]
                                ml_evidence = {
                                    'ml_fraud_probability': ml_result.fraud_probability,
                                    'ml_risk_score': ml_result.risk_score,
                                    'ml_confidence': ml_result.confidence,
                                    'ml_model_version': ml_result.model_version,
                                    'ml_risk_factors': ml_result.risk_factors
                                }
                    except Exception as e:
                        logger.warning(f"ML integration failed: {e}")
                        ml_evidence = {'ml_error': str(e)}
                
                # Step 5: Generate proof
                if not self.proof_verifier:
                    raise ProofIntegrationError("Proof verifier not initialized", component="proof_verifier")
                
                # Create enhanced proof claim
                proof_claim = self._create_enhanced_proof_claim(
                    transaction_data=processed_data,
                    proof_type=proof_type,
                    ml_evidence=ml_evidence,
                    validation_result=validation_result
                )
                
                # Verify proof
                proof_result = self.proof_verifier.verify_proof(proof_claim)
                
                # Step 6: Enhance result with integration data
                enhanced_result = self._enhance_proof_result(
                    proof_result=proof_result,
                    validation_result=validation_result,
                    ml_evidence=ml_evidence,
                    session_id=session_id
                )
                
                # Update session state
                self.integration_state[session_id]['proofs_processed'] += 1
                self.integration_state[session_id]['operations'].append({
                    'operation': 'full_integration_proof',
                    'timestamp': datetime.now(),
                    'proof_type': proof_type,
                    'success': True
                })
                
                logger.info(f"Full integration proof processing completed for session {session_id}")
                return enhanced_result
                
            except Exception as e:
                logger.error(f"Full integration proof processing failed: {e}")
                
                # Update session state
                self.integration_state[session_id]['operations'].append({
                    'operation': 'full_integration_proof',
                    'timestamp': datetime.now(),
                    'error': str(e),
                    'success': False
                })
                
                # Return error result
                return {
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat()
                }
    
    def _create_enhanced_proof_claim(self, transaction_data: Dict[str, Any], 
                                   proof_type: str, ml_evidence: Dict[str, Any],
                                   validation_result: Any) -> Any:
        """Create enhanced proof claim with integrated evidence"""
        try:
            # Extract key information
            transaction_id = transaction_data.get('transaction_id', f"TXN_{int(time.time())}")
            timestamp = datetime.now()
            
            # Combine evidence
            evidence = {
                'transaction_data': transaction_data,
                'ml_evidence': ml_evidence,
                'validation_evidence': {
                    'is_valid': validation_result.is_valid if validation_result else True,
                    'issues_count': len(validation_result.issues) if validation_result else 0
                },
                'integration_metadata': {
                    'processing_timestamp': timestamp.isoformat(),
                    'data_validated': validation_result is not None,
                    'ml_processed': bool(ml_evidence),
                    'preprocessing_applied': self.config.enable_preprocessing
                }
            }
            
            # Calculate fraud probability and risk score
            fraud_probability = ml_evidence.get('ml_fraud_probability', 0.0)
            risk_score = ml_evidence.get('ml_risk_score', 0.0)
            
            # Create proof claim
            if PROOF_COMPONENTS:
                proof_claim = EnhancedProofClaim(
                    claim_id=f"CLAIM_{transaction_id}_{int(time.time())}",
                    claim_type=getattr(ProofType, proof_type.upper(), ProofType.TRANSACTION_FRAUD),
                    transaction_id=transaction_id,
                    timestamp=timestamp,
                    fraud_probability=fraud_probability,
                    risk_score=risk_score,
                    evidence=evidence,
                    model_version=ml_evidence.get('ml_model_version', 'unknown'),
                    model_confidence=ml_evidence.get('ml_confidence', 0.0)
                )
            else:
                # Fallback to basic proof claim
                proof_claim = {
                    'claim_id': f"CLAIM_{transaction_id}_{int(time.time())}",
                    'claim_type': proof_type,
                    'transaction_id': transaction_id,
                    'timestamp': timestamp.isoformat(),
                    'fraud_probability': fraud_probability,
                    'risk_score': risk_score,
                    'evidence': evidence
                }
            
            return proof_claim
            
        except Exception as e:
            logger.error(f"Enhanced proof claim creation failed: {e}")
            raise ProofIntegrationError(f"Proof claim creation failed: {e}", component="proof_claim")
    
    def _enhance_proof_result(self, proof_result: Any, validation_result: Any,
                            ml_evidence: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Enhance proof result with integration information"""
        try:
            # Extract base result information
            if hasattr(proof_result, '__dict__'):
                result_dict = asdict(proof_result) if hasattr(proof_result, '__dataclass_fields__') else vars(proof_result)
            else:
                result_dict = proof_result if isinstance(proof_result, dict) else {'result': str(proof_result)}
            
            # Add integration enhancements
            enhanced_result = {
                'success': True,
                'proof_result': result_dict,
                'integration_data': {
                    'session_id': session_id,
                    'processing_timestamp': datetime.now().isoformat(),
                    'validation_summary': {
                        'data_validation_passed': validation_result.is_valid if validation_result else True,
                        'validation_issues': len(validation_result.issues) if validation_result else 0
                    },
                    'ml_summary': {
                        'ml_processed': bool(ml_evidence),
                        'fraud_probability': ml_evidence.get('ml_fraud_probability'),
                        'risk_score': ml_evidence.get('ml_risk_score'),
                        'model_version': ml_evidence.get('ml_model_version')
                    },
                    'system_health': {
                        'components_available': {
                            'proof_verifier': self.proof_verifier is not None,
                            'data_validator': self.data_validator is not None,
                            'ml_integration': self.ml_integration is not None
                        }
                    }
                }
            }
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Proof result enhancement failed: {e}")
            return {
                'success': False,
                'error': f"Result enhancement failed: {e}",
                'proof_result': proof_result,
                'session_id': session_id
            }
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'proof_verifier': self.proof_verifier is not None,
                    'data_validator': self.data_validator is not None,
                    'transaction_validator': self.transaction_validator is not None,
                    'data_loader': self.data_loader is not None,
                    'preprocessor': self.preprocessor is not None,
                    'ml_integration': self.ml_integration is not None
                },
                'config': asdict(self.config),
                'active_sessions': len([s for s in self.integration_state.values() 
                                      if s['status'] == 'active']),
                'total_sessions': len(self.integration_state),
                'system_health': {}
            }
            
            # Add system health if available
            if self.proof_verifier:
                health_report = self.analyzer.analyze_system_health(self.proof_verifier)
                status['system_health'] = health_report
            
            return status
            
        except Exception as e:
            logger.error(f"Integration status retrieval failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def shutdown(self):
        """Shutdown integration manager and cleanup resources"""
        try:
            if self.executor:
                self.executor.shutdown(wait=True)
            
            logger.info("Proof integration manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")

# Export components
__all__ = [
    'ProofIntegrationConfig',
    'ProofIntegrationManager',
    'ProofSystemAnalyzer',
    'ProofIntegrationError',
    'ProofSystemAnalysisError'
]

if __name__ == "__main__":
    print("Enhanced Proof Verifier - Chunk 1: Analysis and integration utilities loaded")
    
    # Basic integration test
    try:
        config = ProofIntegrationConfig()
        manager = ProofIntegrationManager(config)
        print("✓ Proof integration manager created successfully")
        
        # Test integration status
        status = manager.get_integration_status()
        print(f"✓ Integration status retrieved: {len(status['components'])} components")
        
    except Exception as e:
        print(f"✗ Proof integration test failed: {e}")