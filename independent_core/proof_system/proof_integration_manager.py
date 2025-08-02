"""
Proof Integration Manager
Coordinates all proof engines and manages comprehensive proof generation
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .rule_based_engine import RuleBasedProofEngine
from .ml_based_engine import MLBasedProofEngine
from .cryptographic_engine import CryptographicProofEngine
from .confidence_generator import ConfidenceGenerator
from .algebraic_rule_enforcer import AlgebraicRuleEnforcer

logger = logging.getLogger(__name__)


class ProofIntegrationManager:
    """Manages integration and coordination of all proof engines"""
    
    def __init__(self):
        """Initialize proof integration manager"""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize engines
        self.engines = {}
        self.confidence_generator = ConfidenceGenerator()
        self.algebraic_enforcer = AlgebraicRuleEnforcer()
        
        # Event handling
        self.event_handlers = {}
        self.event_log = []
        self.event_lock = threading.Lock()
        
        # Performance monitoring
        self.performance_stats = {
            'total_proofs': 0,
            'generation_times': [],
            'engine_performance': {}
        }
        self.monitoring_enabled = False
        
        # Configuration
        self.timeout = 30  # seconds
        self.max_retries = 3
        
    def register_engine(self, engine_name: str, engine_instance: Any) -> None:
        """Register a proof engine"""
        self.engines[engine_name] = engine_instance
        self.logger.info(f"Registered proof engine: {engine_name}")
        
        # Initialize performance tracking for this engine
        self.performance_stats['engine_performance'][engine_name] = {
            'calls': 0,
            'total_time': 0,
            'avg_time': 0,
            'errors': 0
        }
        
    def get_registered_engines(self) -> List[str]:
        """Get list of registered engine names"""
        return list(self.engines.keys())
        
    def generate_comprehensive_proof(self, transaction: Dict[str, Any], 
                                   model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive proof using all available engines"""
        start_time = time.time()
        
        self._log_event('proof_generation_started', {
            'transaction_id': transaction.get('transaction_id', 'unknown'),
            'engines_available': list(self.engines.keys())
        })
        
        proof_results = {}
        errors = {}
        
        # Generate proofs from each engine
        for engine_name, engine in self.engines.items():
            engine_start = time.time()
            
            try:
                if hasattr(engine, 'generate_proof'):
                    proof = engine.generate_proof(transaction, model_state)
                elif hasattr(engine, 'evaluate_transaction'):
                    # For rule-based engine
                    proof = engine.evaluate_transaction(transaction)
                elif hasattr(engine, 'generate_ml_proof'):
                    # For ML-based engine
                    proof = engine.generate_ml_proof(transaction, model_state)
                elif hasattr(engine, 'generate_proof_for_transaction'):
                    # For cryptographic engine
                    proof = engine.generate_proof_for_transaction(transaction)
                else:
                    raise AttributeError(f"Engine {engine_name} has no recognized proof generation method")
                    
                proof_results[engine_name] = proof
                
                # Update performance stats
                if self.monitoring_enabled:
                    engine_time = time.time() - engine_start
                    self._update_engine_performance(engine_name, engine_time, success=True)
                    
            except Exception as e:
                error_msg = f"Engine {engine_name} failed: {str(e)}"
                self.logger.error(error_msg)
                errors[engine_name] = error_msg
                
                if self.monitoring_enabled:
                    engine_time = time.time() - engine_start
                    self._update_engine_performance(engine_name, engine_time, success=False)
                    
        # Generate confidence score
        try:
            confidence_inputs = self._extract_confidence_inputs(proof_results)
            confidence_result = self.confidence_generator.generate_confidence(**confidence_inputs)
        except Exception as e:
            self.logger.error(f"Confidence generation failed: {str(e)}")
            confidence_result = {'score': 0.0, 'error': str(e)}
            
        # Create comprehensive proof
        comprehensive_proof = {
            'transaction_id': transaction.get('transaction_id', 'unknown'),
            'generation_timestamp': datetime.now().isoformat(),
            'engines_used': list(proof_results.keys()),
            'confidence': confidence_result.get('score', 0.0),
            'confidence_details': confidence_result
        }
        
        # Add individual engine results
        comprehensive_proof.update(proof_results)
        
        # Add errors if any
        if errors:
            comprehensive_proof['errors'] = errors
            
        # Add performance metrics
        generation_time = time.time() - start_time
        comprehensive_proof['generation_time_ms'] = generation_time * 1000
        
        # Update overall performance stats
        if self.monitoring_enabled:
            self.performance_stats['total_proofs'] += 1
            self.performance_stats['generation_times'].append(generation_time)
            
        self._log_event('proof_generation_completed', {
            'transaction_id': transaction.get('transaction_id', 'unknown'),
            'engines_succeeded': list(proof_results.keys()),
            'engines_failed': list(errors.keys()),
            'generation_time_ms': generation_time * 1000,
            'confidence_score': confidence_result.get('score', 0.0)
        })
        
        return comprehensive_proof
        
    def generate_batch_proofs(self, transactions: List[Dict[str, Any]], 
                            model_state: Dict[str, Any],
                            async_mode: bool = False) -> List[Dict[str, Any]]:
        """Generate proofs for batch of transactions"""
        if async_mode:
            return self._generate_batch_async(transactions, model_state)
        else:
            return self._generate_batch_sync(transactions, model_state)
            
    def _generate_batch_sync(self, transactions: List[Dict[str, Any]], 
                           model_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate proofs synchronously"""
        results = []
        for transaction in transactions:
            proof = self.generate_comprehensive_proof(transaction, model_state)
            results.append(proof)
        return results
        
    def _generate_batch_async(self, transactions: List[Dict[str, Any]], 
                            model_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate proofs asynchronously using thread pool"""
        results = [None] * len(transactions)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_index = {
                executor.submit(self.generate_comprehensive_proof, tx, model_state): i
                for i, tx in enumerate(transactions)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    self.logger.error(f"Async proof generation failed for transaction {index}: {str(e)}")
                    results[index] = {
                        'error': str(e),
                        'transaction_index': index
                    }
                    
        return results
        
    def _extract_confidence_inputs(self, proof_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract inputs for confidence generation from proof results"""
        inputs = {}
        
        # Extract rule-based score
        if 'rule_based' in proof_results:
            rule_result = proof_results['rule_based']
            if isinstance(rule_result, dict):
                inputs['rule_score'] = rule_result.get('risk_score', rule_result.get('confidence', 0.0))
            else:
                inputs['rule_score'] = 0.0
                
        # Extract ML probability
        if 'ml_based' in proof_results:
            ml_result = proof_results['ml_based']
            if isinstance(ml_result, dict):
                ml_analysis = ml_result.get('ml_analysis', ml_result)
                inputs['ml_probability'] = ml_analysis.get('model_prediction', ml_analysis.get('confidence_score', 0.0))
            else:
                inputs['ml_probability'] = 0.0
                
        # Extract cryptographic validation
        if 'cryptographic' in proof_results:
            crypto_result = proof_results['cryptographic']
            if isinstance(crypto_result, dict):
                inputs['crypto_valid'] = crypto_result.get('basic_proof', {}).get('hash') is not None
            else:
                inputs['crypto_valid'] = False
                
        return inputs
        
    def aggregate_proofs(self, individual_proofs: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate proofs from multiple engines"""
        # Calculate consensus metrics
        scores = []
        confidences = []
        
        for engine_name, proof in individual_proofs.items():
            if isinstance(proof, dict):
                # Extract score/confidence from proof
                score = proof.get('risk_score', proof.get('confidence', proof.get('score', 0.0)))
                if isinstance(score, (int, float)):
                    scores.append(score)
                    
                confidence = proof.get('confidence', proof.get('confidence_score', 0.0))
                if isinstance(confidence, (int, float)):
                    confidences.append(confidence)
                    
        # Calculate aggregated metrics
        if scores:
            mean_score = sum(scores) / len(scores)
            score_std = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5
            consensus_reached = score_std < 0.2  # Low standard deviation indicates consensus
        else:
            mean_score = 0.0
            score_std = 1.0
            consensus_reached = False
            
        if confidences:
            overall_confidence = sum(confidences) / len(confidences)
        else:
            overall_confidence = 0.0
            
        return {
            'overall_confidence': overall_confidence,
            'mean_score': mean_score,
            'score_standard_deviation': score_std,
            'consensus_reached': consensus_reached,
            'aggregation_method': 'mean_aggregation',
            'engines_contributing': len(individual_proofs),
            'timestamp': datetime.now().isoformat()
        }
        
    def register_event_handler(self, event_type: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register event handler for specific event type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        
    def _log_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log event and notify handlers"""
        event = {
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        
        with self.event_lock:
            self.event_log.append(event)
            
            # Keep only recent events (last 1000)
            if len(self.event_log) > 1000:
                self.event_log = self.event_log[-1000:]
                
        # Notify handlers
        handlers = self.event_handlers.get(event_type, []) + self.event_handlers.get('all', [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(f"Event handler failed: {str(e)}")
                
    def get_event_log(self) -> List[Dict[str, Any]]:
        """Get copy of event log"""
        with self.event_lock:
            return self.event_log.copy()
            
    def clear_event_log(self) -> None:
        """Clear event log"""
        with self.event_lock:
            self.event_log.clear()
            
    def enable_performance_monitoring(self) -> None:
        """Enable performance monitoring"""
        self.monitoring_enabled = True
        self.logger.info("Performance monitoring enabled")
        
    def disable_performance_monitoring(self) -> None:
        """Disable performance monitoring"""
        self.monitoring_enabled = False
        self.logger.info("Performance monitoring disabled")
        
    def _update_engine_performance(self, engine_name: str, execution_time: float, success: bool) -> None:
        """Update performance statistics for an engine"""
        stats = self.performance_stats['engine_performance'][engine_name]
        stats['calls'] += 1
        
        if success:
            stats['total_time'] += execution_time
            stats['avg_time'] = stats['total_time'] / stats['calls']
        else:
            stats['errors'] += 1
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.monitoring_enabled:
            return {'error': 'Performance monitoring not enabled'}
            
        generation_times = self.performance_stats['generation_times']
        
        stats = {
            'total_proofs': self.performance_stats['total_proofs'],
            'monitoring_enabled': self.monitoring_enabled,
            'engine_performance': self.performance_stats['engine_performance'].copy()
        }
        
        if generation_times:
            stats.update({
                'avg_generation_time': sum(generation_times) / len(generation_times),
                'min_generation_time': min(generation_times),
                'max_generation_time': max(generation_times),
                'total_generation_time': sum(generation_times)
            })
            
        return stats
        
    def validate_with_algebraic_rules(self, gradients: Any, learning_rate: float) -> Dict[str, Any]:
        """Validate using algebraic rule enforcer"""
        return self.algebraic_enforcer.validate_gradients(gradients, learning_rate)
        
    def set_timeout(self, timeout_seconds: int) -> None:
        """Set timeout for proof generation operations"""
        self.timeout = timeout_seconds
        self.logger.info(f"Proof generation timeout set to {timeout_seconds} seconds")
        
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all engines"""
        health_status = {
            'overall_status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'engines': {}
        }
        
        failed_engines = []
        
        for engine_name, engine in self.engines.items():
            try:
                # Simple test transaction
                test_transaction = {
                    'transaction_id': 'health_check',
                    'amount': 100,
                    'test': True
                }
                test_model_state = {'iteration': 0, 'test': True}
                
                # Try to generate proof
                if hasattr(engine, 'generate_proof'):
                    engine.generate_proof(test_transaction, test_model_state)
                elif hasattr(engine, 'evaluate_transaction'):
                    engine.evaluate_transaction(test_transaction)
                elif hasattr(engine, 'generate_ml_proof'):
                    engine.generate_ml_proof(test_transaction, test_model_state)
                    
                health_status['engines'][engine_name] = {
                    'status': 'healthy',
                    'last_check': datetime.now().isoformat()
                }
                
            except Exception as e:
                health_status['engines'][engine_name] = {
                    'status': 'unhealthy',
                    'error': str(e),
                    'last_check': datetime.now().isoformat()
                }
                failed_engines.append(engine_name)
                
        if failed_engines:
            health_status['overall_status'] = 'degraded'
            health_status['failed_engines'] = failed_engines
            
        return health_status