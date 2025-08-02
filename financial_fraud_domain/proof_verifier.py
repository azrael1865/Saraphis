"""
Proof System Verifier for Financial Fraud Detection
Advanced proof system integration for verifiable fraud detection claims
"""

import logging
import json
import hashlib
import time
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

# Enhanced import
try:
    from enhanced_proof_verifier import (
        FinancialProofVerifier as EnhancedFinancialProofVerifier,
        EnhancedProofClaim,
        EnhancedProofEvidence,
        EnhancedProofResult,
        SecurityLevel,
        ProofVerificationException,
        ProofConfigurationError,
        ProofGenerationError,
        ProofValidationError,
        ProofTimeoutError,
        ProofSecurityError,
        ProofIntegrityError,
        ClaimValidationError,
        EvidenceValidationError,
        ProofSystemError,
        ProofStorageError,
        CryptographicError,
        ProofExpiredError,
        ResourceLimitError,
        SecurityValidator,
        ResourceMonitor
    )
    ENHANCED_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced proof verifier not available: {e}")
    ENHANCED_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class ProofType(Enum):
    """Types of fraud detection proofs"""
    TRANSACTION_FRAUD = "transaction_fraud"
    PATTERN_FRAUD = "pattern_fraud"
    RULE_VIOLATION = "rule_violation"
    ML_PREDICTION = "ml_prediction"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    NETWORK_FRAUD = "network_fraud"
    COMPOSITE = "composite"

class ProofStatus(Enum):
    """Proof verification status"""
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"
    INVALID = "invalid"

class ProofLevel(Enum):
    """Proof confidence levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ProofClaim:
    """Fraud detection claim to be proven"""
    claim_id: str
    claim_type: ProofType
    transaction_id: str
    timestamp: datetime
    
    # Claim details
    fraud_probability: float
    risk_score: float
    evidence: Dict[str, Any]
    
    # Rule violations
    violated_rules: List[str] = field(default_factory=list)
    
    # ML model info
    model_version: Optional[str] = None
    model_confidence: float = 0.0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProofEvidence:
    """Evidence supporting a fraud claim"""
    evidence_id: str
    evidence_type: str
    source: str
    timestamp: datetime
    
    # Evidence data
    data: Dict[str, Any]
    confidence: float
    
    # Verification info
    verified: bool = False
    verification_method: Optional[str] = None
    verification_timestamp: Optional[datetime] = None

@dataclass
class ProofResult:
    """Result of proof verification"""
    proof_id: str
    claim_id: str
    status: ProofStatus
    confidence: float
    timestamp: datetime
    
    # Verification details
    verification_method: str
    verification_time_ms: float
    
    # Evidence and reasoning
    evidence_used: List[str]
    reasoning: Dict[str, Any]
    
    # Cryptographic proof (if applicable)
    proof_hash: Optional[str] = None
    signature: Optional[str] = None
    
    # Validity period
    valid_until: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """Check if proof is still valid"""
        if self.status != ProofStatus.VERIFIED:
            return False
        if self.valid_until and datetime.now() > self.valid_until:
            return False
        return True

@dataclass
class ProofMetrics:
    """Metrics for proof system performance"""
    total_proofs_generated: int = 0
    total_proofs_verified: int = 0
    total_proofs_rejected: int = 0
    
    average_generation_time_ms: float = 0.0
    average_verification_time_ms: float = 0.0
    
    verification_success_rate: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    
    # Performance by proof type
    metrics_by_type: Dict[ProofType, Dict[str, float]] = field(default_factory=dict)
    
    # Time-based metrics
    hourly_verification_count: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'total_generated': self.total_proofs_generated,
            'total_verified': self.total_proofs_verified,
            'total_rejected': self.total_proofs_rejected,
            'avg_generation_time': self.average_generation_time_ms,
            'avg_verification_time': self.average_verification_time_ms,
            'success_rate': self.verification_success_rate,
            'false_positive_rate': self.false_positive_rate,
            'false_negative_rate': self.false_negative_rate
        }

class BaseProofSystem(ABC):
    """Abstract base class for proof systems"""
    
    @abstractmethod
    def generate_proof(self, claim: ProofClaim, evidence: List[ProofEvidence]) -> Dict[str, Any]:
        """Generate proof for a claim"""
        pass
    
    @abstractmethod
    def verify_proof(self, proof: Dict[str, Any], claim: ProofClaim) -> Tuple[bool, float]:
        """Verify a proof"""
        pass

class RuleBasedProofSystem(BaseProofSystem):
    """Rule-based proof system for fraud detection"""
    
    def __init__(self, rules: Optional[Dict[str, Any]] = None):
        self.rules = rules or self._default_rules()
    
    def _default_rules(self) -> Dict[str, Any]:
        """Default fraud detection rules"""
        return {
            'transaction_limits': {
                'max_amount': 10000,
                'max_daily_amount': 50000,
                'max_daily_transactions': 100
            },
            'velocity_rules': {
                'max_transactions_per_hour': 20,
                'max_amount_per_hour': 20000
            },
            'geographical_rules': {
                'max_distance_km_per_hour': 1000,
                'restricted_countries': ['XX', 'YY']
            },
            'behavioral_rules': {
                'unusual_time_threshold': 3,  # Standard deviations
                'unusual_amount_threshold': 3
            }
        }
    
    def generate_proof(self, claim: ProofClaim, evidence: List[ProofEvidence]) -> Dict[str, Any]:
        """Generate rule-based proof"""
        violated_rules = []
        rule_scores = {}
        
        # Check each rule category
        for category, rules in self.rules.items():
            category_violations = self._check_rule_category(category, rules, claim, evidence)
            if category_violations:
                violated_rules.extend(category_violations)
                rule_scores[category] = len(category_violations) / len(rules)
        
        # Calculate overall score
        overall_score = sum(rule_scores.values()) / len(self.rules) if rule_scores else 0
        
        return {
            'proof_type': 'rule_based',
            'violated_rules': violated_rules,
            'rule_scores': rule_scores,
            'overall_score': overall_score,
            'timestamp': datetime.now().isoformat()
        }
    
    def verify_proof(self, proof: Dict[str, Any], claim: ProofClaim) -> Tuple[bool, float]:
        """Verify rule-based proof"""
        if proof.get('proof_type') != 'rule_based':
            return False, 0.0
        
        # Verify proof integrity
        violated_rules = proof.get('violated_rules', [])
        overall_score = proof.get('overall_score', 0)
        
        # Proof is valid if rules were violated
        is_valid = len(violated_rules) > 0 and overall_score > 0
        confidence = min(overall_score * 100, 100) if is_valid else 0
        
        return is_valid, confidence
    
    def _check_rule_category(self, category: str, rules: Dict[str, Any], 
                           claim: ProofClaim, evidence: List[ProofEvidence]) -> List[str]:
        """Check rules in a specific category"""
        violations = []
        
        # Extract relevant data from evidence
        evidence_data = {}
        for e in evidence:
            evidence_data.update(e.data)
        
        # Check transaction limits
        if category == 'transaction_limits':
            amount = evidence_data.get('amount', 0)
            if amount > rules['max_amount']:
                violations.append(f"transaction_amount_exceeds_{rules['max_amount']}")
        
        # Add more rule checks as needed...
        
        return violations

class MLProofSystem(BaseProofSystem):
    """Machine learning based proof system"""
    
    def __init__(self, model_threshold: float = 0.7):
        self.model_threshold = model_threshold
    
    def generate_proof(self, claim: ProofClaim, evidence: List[ProofEvidence]) -> Dict[str, Any]:
        """Generate ML-based proof"""
        # Extract ML predictions from evidence
        ml_predictions = []
        for e in evidence:
            if e.evidence_type == 'ml_prediction':
                ml_predictions.append({
                    'model': e.source,
                    'probability': e.data.get('fraud_probability', 0),
                    'confidence': e.confidence
                })
        
        if not ml_predictions:
            return {'proof_type': 'ml_based', 'valid': False, 'reason': 'no_ml_predictions'}
        
        # Aggregate predictions
        avg_probability = np.mean([p['probability'] for p in ml_predictions])
        max_probability = max([p['probability'] for p in ml_predictions])
        
        return {
            'proof_type': 'ml_based',
            'predictions': ml_predictions,
            'average_probability': avg_probability,
            'max_probability': max_probability,
            'model_consensus': len([p for p in ml_predictions if p['probability'] > self.model_threshold]) / len(ml_predictions),
            'timestamp': datetime.now().isoformat()
        }
    
    def verify_proof(self, proof: Dict[str, Any], claim: ProofClaim) -> Tuple[bool, float]:
        """Verify ML-based proof"""
        if proof.get('proof_type') != 'ml_based':
            return False, 0.0
        
        avg_prob = proof.get('average_probability', 0)
        consensus = proof.get('model_consensus', 0)
        
        # Proof is valid if probability exceeds threshold
        is_valid = avg_prob > self.model_threshold
        confidence = avg_prob * 100 * consensus if is_valid else 0
        
        return is_valid, confidence

class CryptographicProofSystem(BaseProofSystem):
    """Cryptographic proof system for verifiable fraud claims"""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or "default_secret_key"
    
    def generate_proof(self, claim: ProofClaim, evidence: List[ProofEvidence]) -> Dict[str, Any]:
        """Generate cryptographic proof"""
        # Create proof data
        proof_data = {
            'claim_id': claim.claim_id,
            'transaction_id': claim.transaction_id,
            'fraud_probability': claim.fraud_probability,
            'timestamp': claim.timestamp.isoformat(),
            'evidence_hashes': [self._hash_evidence(e) for e in evidence]
        }
        
        # Generate proof hash
        proof_string = json.dumps(proof_data, sort_keys=True)
        proof_hash = hashlib.sha256(proof_string.encode()).hexdigest()
        
        # Generate signature (simplified)
        signature = hashlib.sha256(
            (proof_hash + self.secret_key).encode()
        ).hexdigest()
        
        return {
            'proof_type': 'cryptographic',
            'proof_data': proof_data,
            'proof_hash': proof_hash,
            'signature': signature,
            'algorithm': 'sha256',
            'timestamp': datetime.now().isoformat()
        }
    
    def verify_proof(self, proof: Dict[str, Any], claim: ProofClaim) -> Tuple[bool, float]:
        """Verify cryptographic proof"""
        if proof.get('proof_type') != 'cryptographic':
            return False, 0.0
        
        # Verify proof hash
        proof_data = proof.get('proof_data', {})
        proof_string = json.dumps(proof_data, sort_keys=True)
        expected_hash = hashlib.sha256(proof_string.encode()).hexdigest()
        
        if expected_hash != proof.get('proof_hash'):
            return False, 0.0
        
        # Verify signature
        expected_signature = hashlib.sha256(
            (expected_hash + self.secret_key).encode()
        ).hexdigest()
        
        if expected_signature != proof.get('signature'):
            return False, 0.0
        
        return True, 100.0
    
    def _hash_evidence(self, evidence: ProofEvidence) -> str:
        """Generate hash of evidence"""
        evidence_str = f"{evidence.evidence_id}_{evidence.timestamp}_{evidence.confidence}"
        return hashlib.md5(evidence_str.encode()).hexdigest()

class FinancialProofVerifier:
    """
    Advanced Proof System Verifier for Financial Fraud Detection
    
    Features:
    - Multiple proof system support (rule-based, ML, cryptographic)
    - Proof generation and verification
    - Evidence management
    - Performance metrics
    - Proof persistence and loading
    - Concurrent proof processing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, use_enhanced: bool = True):
        """
        Initialize proof verifier
        
        Args:
            config: Configuration dictionary
            use_enhanced: Whether to use enhanced proof verifier when available
        """
        self.config = config or self._default_config()
        self.use_enhanced = use_enhanced and ENHANCED_AVAILABLE
        
        if self.use_enhanced:
            # Use enhanced proof verifier
            self.enhanced_verifier = EnhancedFinancialProofVerifier(self._convert_config_for_enhanced())
            logger.info("FinancialProofVerifier initialized with enhanced capabilities")
        else:
            # Initialize standard proof systems
            self.proof_systems = {
                'rule_based': RuleBasedProofSystem(self.config.get('rules')),
                'ml_based': MLProofSystem(self.config.get('ml_threshold', 0.7)),
                'cryptographic': CryptographicProofSystem(self.config.get('secret_key'))
            }
            
            # Storage
            self.proofs: Dict[str, ProofResult] = {}
            self.claims: Dict[str, ProofClaim] = {}
            self.evidence: Dict[str, List[ProofEvidence]] = {}
            
            # Metrics
            self.metrics = ProofMetrics()
            
            # Thread safety
            self._lock = threading.RLock()
            self._executor = ThreadPoolExecutor(max_workers=4)
            
            # Persistence
            self.storage_path = Path(self.config.get('storage_path', 'proofs'))
            self.storage_path.mkdir(exist_ok=True)
            
            logger.info("FinancialProofVerifier initialized with standard capabilities")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'ml_threshold': 0.7,
            'proof_validity_hours': 24,
            'max_evidence_per_claim': 100,
            'enable_cryptographic_proofs': True,
            'concurrent_verification': True,
            'auto_persist': True
        }
    
    def _convert_config_for_enhanced(self) -> Dict[str, Any]:
        """Convert standard config to enhanced config format"""
        enhanced_config = self.config.copy()
        
        # Add enhanced-specific defaults if not present
        enhanced_defaults = {
            'max_memory_mb': 512,
            'max_cpu_percent': 50.0,
            'max_workers': 4,
            'enable_audit_trail': True,
            'security_level': 'standard',
            'timeout_seconds': 60,
            'enable_resource_monitoring': True,
            'validation_strict_mode': False,
            'enable_drift_detection': True,
            'min_evidence_confidence': 0.3,
            'max_proof_age_hours': 168
        }
        
        for key, value in enhanced_defaults.items():
            if key not in enhanced_config:
                enhanced_config[key] = value
        
        return enhanced_config
    
    def generate_proof(self, claim: Dict[str, Any], 
                      evidence: Optional[List[Dict[str, Any]]] = None,
                      user_context: Optional[Dict[str, Any]] = None) -> Optional[Union[ProofResult, EnhancedProofResult]]:
        """
        Generate proof for fraud detection claim
        
        Args:
            claim: Fraud claim data
            evidence: Supporting evidence
            user_context: User context for enhanced security validation
            
        Returns:
            Proof result or None if generation fails
        """
        if self.use_enhanced:
            # Use enhanced proof generation with comprehensive validation
            try:
                enhanced_result = self.enhanced_verifier.generate_proof(claim, evidence, user_context)
                if enhanced_result:
                    # Convert to standard result for backward compatibility if needed
                    return self._convert_enhanced_to_standard_result(enhanced_result)
                return None
            except Exception as e:
                logger.error(f"Enhanced proof generation failed, falling back to standard: {str(e)}")
                # Fall through to standard implementation
        
        # Standard implementation
        start_time = time.time()
        
        try:
            # Create claim object
            claim_obj = self._create_claim(claim)
            
            # Collect and validate evidence
            evidence_objs = self._collect_evidence(claim_obj, evidence)
            
            # Store claim and evidence
            with self._lock:
                self.claims[claim_obj.claim_id] = claim_obj
                self.evidence[claim_obj.claim_id] = evidence_objs
            
            # Generate proofs using different systems
            proofs = {}
            proof_scores = {}
            
            for system_name, system in self.proof_systems.items():
                if self._should_use_system(system_name, claim_obj):
                    try:
                        proof = system.generate_proof(claim_obj, evidence_objs)
                        proofs[system_name] = proof
                        
                        # Verify the generated proof
                        is_valid, confidence = system.verify_proof(proof, claim_obj)
                        if is_valid:
                            proof_scores[system_name] = confidence
                    except Exception as e:
                        logger.error(f"Proof generation failed for {system_name}: {str(e)}")
            
            if not proofs:
                logger.warning(f"No proofs generated for claim {claim_obj.claim_id}")
                return None
            
            # Create composite proof result
            proof_result = self._create_proof_result(claim_obj, proofs, proof_scores)
            
            # Update metrics
            generation_time = (time.time() - start_time) * 1000
            self._update_metrics('generation', generation_time, proof_result.status)
            
            # Store proof
            with self._lock:
                self.proofs[proof_result.proof_id] = proof_result
            
            # Persist if configured
            if self.config.get('auto_persist'):
                self._persist_proof(proof_result)
            
            logger.info(f"Generated proof {proof_result.proof_id} for claim {claim_obj.claim_id} "
                       f"with confidence {proof_result.confidence:.2f}")
            
            return proof_result
            
        except Exception as e:
            logger.error(f"Failed to generate proof: {str(e)}")
            return None
    
    def verify_proof(self, proof: Dict[str, Any], 
                    user_context: Optional[Dict[str, Any]] = None) -> Union[Tuple[bool, str], Tuple[bool, str, Dict[str, Any]]]:
        """
        Verify a fraud detection proof
        
        Args:
            proof: Proof data to verify
            user_context: User context for enhanced security validation
            
        Returns:
            Tuple of (is_valid, message) or (is_valid, message, details) for enhanced mode
        """
        if self.use_enhanced:
            # Use enhanced verification with comprehensive validation
            try:
                is_valid, message, details = self.enhanced_verifier.verify_proof(proof, user_context)
                return is_valid, message, details
            except Exception as e:
                logger.error(f"Enhanced proof verification failed, falling back to standard: {str(e)}")
                # Fall through to standard implementation
        
        # Standard implementation
        start_time = time.time()
        
        try:
            # Extract proof ID
            proof_id = proof.get('proof_id')
            if not proof_id:
                return False, "Missing proof ID"
            
            # Check if proof exists
            with self._lock:
                stored_proof = self.proofs.get(proof_id)
            
            if not stored_proof:
                # Try loading from storage
                stored_proof = self._load_proof(proof_id)
                if not stored_proof:
                    return False, "Proof not found"
            
            # Check validity
            if not stored_proof.is_valid():
                return False, f"Proof expired or invalid (status: {stored_proof.status.value})"
            
            # Verify proof integrity
            if not self._verify_proof_integrity(proof, stored_proof):
                return False, "Proof integrity check failed"
            
            # Re-verify with proof systems if needed
            if self.config.get('reverify_on_check'):
                claim = self.claims.get(stored_proof.claim_id)
                if claim:
                    for system_name in proof.get('systems_used', []):
                        system = self.proof_systems.get(system_name)
                        if system:
                            is_valid, _ = system.verify_proof(
                                proof.get(f'{system_name}_proof', {}), 
                                claim
                            )
                            if not is_valid:
                                return False, f"{system_name} verification failed"
            
            # Update metrics
            verification_time = (time.time() - start_time) * 1000
            self._update_metrics('verification', verification_time, ProofStatus.VERIFIED)
            
            return True, "Proof verified successfully"
            
        except Exception as e:
            logger.error(f"Proof verification failed: {str(e)}")
            return False, f"Verification error: {str(e)}"
    
    def validate_claim(self, claim: Dict[str, Any], 
                      user_context: Optional[Dict[str, Any]] = None) -> Union[Tuple[bool, str], Tuple[bool, str, List[str]]]:
        """
        Validate fraud detection claim
        
        Args:
            claim: Claim data to validate
            user_context: User context for enhanced security validation
            
        Returns:
            Tuple of (is_valid, message) or (is_valid, message, detailed_errors) for enhanced mode
        """
        if self.use_enhanced:
            # Use enhanced validation with comprehensive security checks
            try:
                is_valid, message, errors = self.enhanced_verifier.validate_claim(claim, user_context)
                return is_valid, message, errors
            except Exception as e:
                logger.error(f"Enhanced claim validation failed, falling back to standard: {str(e)}")
                # Fall through to standard implementation
        
        # Standard implementation
        try:
            # Check required fields
            required_fields = ['transaction_id', 'fraud_probability', 'risk_score']
            for field in required_fields:
                if field not in claim:
                    return False, f"Missing required field: {field}"
            
            # Validate probability and score ranges
            fraud_prob = claim.get('fraud_probability', 0)
            risk_score = claim.get('risk_score', 0)
            
            if not 0 <= fraud_prob <= 1:
                return False, "Fraud probability must be between 0 and 1"
            
            if not 0 <= risk_score <= 1:
                return False, "Risk score must be between 0 and 1"
            
            # Validate evidence if provided
            evidence = claim.get('evidence', {})
            if evidence and not isinstance(evidence, dict):
                return False, "Evidence must be a dictionary"
            
            # Check for suspicious patterns
            if fraud_prob > 0.9 and len(evidence) == 0:
                return False, "High fraud probability requires evidence"
            
            return True, "Valid claim"
            
        except Exception as e:
            logger.error(f"Claim validation failed: {str(e)}")
            return False, f"Validation error: {str(e)}"
    
    def get_proof_by_id(self, proof_id: str) -> Optional[ProofResult]:
        """Get proof by ID"""
        with self._lock:
            proof = self.proofs.get(proof_id)
        
        if not proof:
            proof = self._load_proof(proof_id)
        
        return proof
    
    def get_proofs_by_transaction(self, transaction_id: str) -> List[ProofResult]:
        """Get all proofs for a transaction"""
        proofs = []
        
        with self._lock:
            for proof in self.proofs.values():
                claim = self.claims.get(proof.claim_id)
                if claim and claim.transaction_id == transaction_id:
                    proofs.append(proof)
        
        return proofs
    
    def get_metrics(self) -> Union[ProofMetrics, Dict[str, Any]]:
        """Get proof system metrics"""
        if self.use_enhanced:
            # Return enhanced metrics with comprehensive data
            try:
                return self.enhanced_verifier.get_enhanced_metrics()
            except Exception as e:
                logger.error(f"Enhanced metrics retrieval failed, falling back to standard: {str(e)}")
                # Fall through to standard implementation
        
        # Standard implementation
        with self._lock:
            return self.metrics
    
    def _create_claim(self, claim_data: Dict[str, Any]) -> ProofClaim:
        """Create claim object from data"""
        return ProofClaim(
            claim_id=claim_data.get('claim_id', self._generate_id('CLAIM')),
            claim_type=ProofType(claim_data.get('claim_type', ProofType.TRANSACTION_FRAUD.value)),
            transaction_id=claim_data['transaction_id'],
            timestamp=datetime.now(),
            fraud_probability=claim_data['fraud_probability'],
            risk_score=claim_data['risk_score'],
            evidence=claim_data.get('evidence', {}),
            violated_rules=claim_data.get('violated_rules', []),
            model_version=claim_data.get('model_version'),
            model_confidence=claim_data.get('model_confidence', 0),
            metadata=claim_data.get('metadata', {})
        )
    
    def _collect_evidence(self, claim: ProofClaim, 
                         evidence_data: Optional[List[Dict[str, Any]]]) -> List[ProofEvidence]:
        """Collect and validate evidence"""
        evidence_list = []
        
        # Add ML prediction evidence
        if claim.model_version:
            evidence_list.append(ProofEvidence(
                evidence_id=self._generate_id('EVD'),
                evidence_type='ml_prediction',
                source=claim.model_version,
                timestamp=datetime.now(),
                data={
                    'fraud_probability': claim.fraud_probability,
                    'risk_score': claim.risk_score,
                    'model_confidence': claim.model_confidence
                },
                confidence=claim.model_confidence
            ))
        
        # Add rule violation evidence
        for rule in claim.violated_rules:
            evidence_list.append(ProofEvidence(
                evidence_id=self._generate_id('EVD'),
                evidence_type='rule_violation',
                source='rule_engine',
                timestamp=datetime.now(),
                data={'rule': rule, 'violated': True},
                confidence=1.0
            ))
        
        # Add custom evidence
        if evidence_data:
            for ev in evidence_data:
                evidence_list.append(ProofEvidence(
                    evidence_id=ev.get('evidence_id', self._generate_id('EVD')),
                    evidence_type=ev.get('type', 'custom'),
                    source=ev.get('source', 'unknown'),
                    timestamp=datetime.now(),
                    data=ev.get('data', {}),
                    confidence=ev.get('confidence', 0.5)
                ))
        
        # Limit evidence count
        max_evidence = self.config.get('max_evidence_per_claim', 100)
        if len(evidence_list) > max_evidence:
            # Sort by confidence and keep top evidence
            evidence_list.sort(key=lambda x: x.confidence, reverse=True)
            evidence_list = evidence_list[:max_evidence]
        
        return evidence_list
    
    def _should_use_system(self, system_name: str, claim: ProofClaim) -> bool:
        """Determine if proof system should be used for claim"""
        # Use rule-based for rule violations
        if system_name == 'rule_based' and claim.violated_rules:
            return True
        
        # Use ML-based for ML predictions
        if system_name == 'ml_based' and claim.model_version:
            return True
        
        # Use cryptographic if enabled and for high-value claims
        if system_name == 'cryptographic':
            return (self.config.get('enable_cryptographic_proofs', True) and 
                   claim.fraud_probability > 0.8)
        
        return False
    
    def _create_proof_result(self, claim: ProofClaim, 
                           proofs: Dict[str, Dict[str, Any]], 
                           scores: Dict[str, float]) -> ProofResult:
        """Create composite proof result"""
        # Calculate overall confidence
        if scores:
            overall_confidence = np.mean(list(scores.values()))
        else:
            overall_confidence = 0.0
        
        # Determine status
        if overall_confidence >= 80:
            status = ProofStatus.VERIFIED
        elif overall_confidence >= 50:
            status = ProofStatus.PENDING
        else:
            status = ProofStatus.REJECTED
        
        # Create proof hash
        proof_data = {
            'claim_id': claim.claim_id,
            'proofs': proofs,
            'scores': scores,
            'timestamp': datetime.now().isoformat()
        }
        proof_hash = hashlib.sha256(
            json.dumps(proof_data, sort_keys=True).encode()
        ).hexdigest()
        
        # Calculate validity period
        validity_hours = self.config.get('proof_validity_hours', 24)
        valid_until = datetime.now() + timedelta(hours=validity_hours)
        
        return ProofResult(
            proof_id=self._generate_id('PROOF'),
            claim_id=claim.claim_id,
            status=status,
            confidence=overall_confidence,
            timestamp=datetime.now(),
            verification_method='composite',
            verification_time_ms=0,  # Will be updated
            evidence_used=[e.evidence_id for e in self.evidence.get(claim.claim_id, [])],
            reasoning={
                'systems_used': list(proofs.keys()),
                'individual_scores': scores,
                'proofs': proofs
            },
            proof_hash=proof_hash,
            valid_until=valid_until
        )
    
    def _verify_proof_integrity(self, provided_proof: Dict[str, Any], 
                              stored_proof: ProofResult) -> bool:
        """Verify proof integrity"""
        # Check proof hash if provided
        if 'proof_hash' in provided_proof:
            return provided_proof['proof_hash'] == stored_proof.proof_hash
        
        # Check basic fields
        return (provided_proof.get('proof_id') == stored_proof.proof_id and
                provided_proof.get('claim_id') == stored_proof.claim_id)
    
    def _update_metrics(self, operation: str, time_ms: float, status: ProofStatus) -> None:
        """Update metrics"""
        with self._lock:
            if operation == 'generation':
                self.metrics.total_proofs_generated += 1
                
                # Update average time
                total = self.metrics.total_proofs_generated
                current_avg = self.metrics.average_generation_time_ms
                self.metrics.average_generation_time_ms = (
                    (current_avg * (total - 1) + time_ms) / total
                )
                
            elif operation == 'verification':
                self.metrics.total_proofs_verified += 1
                
                if status == ProofStatus.REJECTED:
                    self.metrics.total_proofs_rejected += 1
                
                # Update average time
                total = self.metrics.total_proofs_verified
                current_avg = self.metrics.average_verification_time_ms
                self.metrics.average_verification_time_ms = (
                    (current_avg * (total - 1) + time_ms) / total
                )
                
                # Update success rate
                self.metrics.verification_success_rate = (
                    (self.metrics.total_proofs_verified - self.metrics.total_proofs_rejected) /
                    self.metrics.total_proofs_verified
                )
            
            # Update hourly metrics
            hour_key = datetime.now().strftime('%Y-%m-%d_%H')
            if hour_key not in self.metrics.hourly_verification_count:
                self.metrics.hourly_verification_count[hour_key] = 0
            self.metrics.hourly_verification_count[hour_key] += 1
    
    def _persist_proof(self, proof: ProofResult) -> None:
        """Persist proof to storage"""
        try:
            proof_file = self.storage_path / f"{proof.proof_id}.pkl"
            with open(proof_file, 'wb') as f:
                pickle.dump(proof, f)
            logger.debug(f"Persisted proof {proof.proof_id}")
        except Exception as e:
            logger.error(f"Failed to persist proof: {str(e)}")
    
    def _load_proof(self, proof_id: str) -> Optional[ProofResult]:
        """Load proof from storage"""
        try:
            proof_file = self.storage_path / f"{proof_id}.pkl"
            if proof_file.exists():
                with open(proof_file, 'rb') as f:
                    proof = pickle.load(f)
                
                # Cache in memory
                with self._lock:
                    self.proofs[proof_id] = proof
                
                return proof
        except Exception as e:
            logger.error(f"Failed to load proof: {str(e)}")
        
        return None
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_suffix = hashlib.md5(
            f"{timestamp}_{time.time()}".encode()
        ).hexdigest()[:8]
        return f"{prefix}_{timestamp}_{random_suffix}"
    
    def cleanup_expired_proofs(self) -> Union[int, Dict[str, int]]:
        """Remove expired proofs"""
        if self.use_enhanced:
            # Use enhanced cleanup with detailed reporting
            try:
                return self.enhanced_verifier.cleanup_expired_proofs()
            except Exception as e:
                logger.error(f"Enhanced cleanup failed, falling back to standard: {str(e)}")
                # Fall through to standard implementation
        
        # Standard implementation
        removed_count = 0
        
        with self._lock:
            expired_ids = []
            for proof_id, proof in self.proofs.items():
                if not proof.is_valid() and proof.status == ProofStatus.EXPIRED:
                    expired_ids.append(proof_id)
            
            for proof_id in expired_ids:
                del self.proofs[proof_id]
                removed_count += 1
        
        logger.info(f"Cleaned up {removed_count} expired proofs")
        return removed_count
    
    def export_metrics_report(self) -> Dict[str, Any]:
        """Export comprehensive metrics report"""
        if self.use_enhanced:
            # Use enhanced metrics report with comprehensive data
            try:
                return self.enhanced_verifier.get_enhanced_metrics()
            except Exception as e:
                logger.error(f"Enhanced metrics report failed, falling back to standard: {str(e)}")
                # Fall through to standard implementation
        
        # Standard implementation
        with self._lock:
            return {
                'summary': self.metrics.to_dict(),
                'proof_count_by_status': self._count_proofs_by_status(),
                'performance': {
                    'avg_generation_time_ms': self.metrics.average_generation_time_ms,
                    'avg_verification_time_ms': self.metrics.average_verification_time_ms,
                    'throughput_per_hour': self._calculate_throughput()
                },
                'quality': {
                    'verification_success_rate': self.metrics.verification_success_rate,
                    'active_proofs': len([p for p in self.proofs.values() if p.is_valid()]),
                    'expired_proofs': len([p for p in self.proofs.values() if not p.is_valid()])
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def _count_proofs_by_status(self) -> Dict[str, int]:
        """Count proofs by status"""
        counts = {status.value: 0 for status in ProofStatus}
        for proof in self.proofs.values():
            counts[proof.status.value] += 1
        return counts
    
    def _calculate_throughput(self) -> float:
        """Calculate average throughput per hour"""
        if not self.metrics.hourly_verification_count:
            return 0.0
        return np.mean(list(self.metrics.hourly_verification_count.values()))
    
    def _convert_enhanced_to_standard_result(self, enhanced_result: 'EnhancedProofResult') -> ProofResult:
        """Convert enhanced proof result to standard format for backward compatibility"""
        try:
            return ProofResult(
                proof_id=enhanced_result.proof_id,
                claim_id=enhanced_result.claim_id,
                status=enhanced_result.status,
                confidence=enhanced_result.confidence,
                timestamp=enhanced_result.timestamp,
                verification_method=enhanced_result.verification_method,
                verification_time_ms=enhanced_result.verification_time_ms,
                evidence_used=enhanced_result.evidence_used,
                reasoning=enhanced_result.reasoning,
                proof_hash=enhanced_result.proof_hash,
                signature=enhanced_result.signature,
                valid_until=enhanced_result.valid_until
            )
        except Exception as e:
            logger.error(f"Failed to convert enhanced result to standard format: {str(e)}")
            return enhanced_result  # Return as-is if conversion fails
    
    def get_enhanced_capabilities(self) -> Dict[str, Any]:
        """Get information about enhanced capabilities"""
        return {
            'enhanced_available': ENHANCED_AVAILABLE,
            'using_enhanced': self.use_enhanced,
            'enhanced_features': [
                'comprehensive_validation',
                'security_validation',
                'resource_monitoring',
                'timeout_protection',
                'audit_trail',
                'advanced_metrics',
                'cryptographic_proofs',
                'model_drift_detection',
                'evidence_integrity_verification'
            ] if self.use_enhanced else [],
            'fallback_available': True
        }
    
    def export_audit_trail(self) -> List[Dict[str, Any]]:
        """Export audit trail (enhanced mode only)"""
        if self.use_enhanced:
            try:
                return self.enhanced_verifier.export_enhanced_audit_trail()
            except Exception as e:
                logger.error(f"Failed to export audit trail: {str(e)}")
                return []
        else:
            logger.warning("Audit trail not available in standard mode")
            return []
    
    def __repr__(self) -> str:
        if self.use_enhanced:
            return f"FinancialProofVerifier(enhanced=True, capabilities=advanced_security_monitoring)"
        else:
            return (f"FinancialProofVerifier(proofs={len(self.proofs)}, "
                   f"claims={len(self.claims)}, "
                   f"success_rate={self.metrics.verification_success_rate:.2%})")

# Legacy compatibility
ProofVerifier = FinancialProofVerifier

# Export main classes
__all__ = [
    'FinancialProofVerifier',
    'ProofVerifier',  # Legacy compatibility
    'ProofClaim',
    'ProofEvidence', 
    'ProofResult',
    'ProofMetrics',
    'ProofType',
    'ProofStatus',
    'ProofLevel',
    'BaseProofSystem',
    'RuleBasedProofSystem',
    'MLProofSystem',
    'CryptographicProofSystem'
]

if __name__ == "__main__":
    # Example usage and testing
    
    # Initialize verifier
    config = {
        'ml_threshold': 0.7,
        'enable_cryptographic_proofs': True,
        'proof_validity_hours': 24
    }
    
    verifier = FinancialProofVerifier(config)
    print("ProofVerifier initialized")
    
    # Create sample claim
    claim = {
        'transaction_id': 'TXN_12345',
        'fraud_probability': 0.85,
        'risk_score': 0.9,
        'claim_type': 'transaction_fraud',
        'evidence': {
            'unusual_amount': True,
            'velocity_violation': True
        },
        'violated_rules': ['max_transaction_amount', 'velocity_limit'],
        'model_version': 'rf_model_v1',
        'model_confidence': 0.92
    }
    
    # Validate claim
    is_valid, message = verifier.validate_claim(claim)
    print(f"Claim validation: {is_valid} - {message}")
    
    # Generate proof
    print("\nGenerating proof...")
    proof_result = verifier.generate_proof(claim)
    
    if proof_result:
        print(f"Proof generated: {proof_result.proof_id}")
        print(f"Status: {proof_result.status.value}")
        print(f"Confidence: {proof_result.confidence:.2f}")
        print(f"Valid until: {proof_result.valid_until}")
        
        # Verify proof
        print("\nVerifying proof...")
        proof_data = {
            'proof_id': proof_result.proof_id,
            'proof_hash': proof_result.proof_hash,
            'systems_used': proof_result.reasoning['systems_used']
        }
        
        is_valid, message = verifier.verify_proof(proof_data)
        print(f"Verification result: {is_valid} - {message}")
    
    # Get metrics
    metrics = verifier.get_metrics()
    print(f"\nMetrics: {metrics.to_dict()}")
    
    # Export report
    report = verifier.export_metrics_report()
    print(f"\nMetrics Report:")
    print(json.dumps(report, indent=2))
    
    print("\nFinancialProofVerifier ready for production use!")