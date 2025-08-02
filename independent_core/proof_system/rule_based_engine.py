"""
Rule-Based Proof Engine
Generates proofs based on business rules and domain knowledge
"""

import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class RuleBasedProofEngine:
    """Rule-based proof engine for transaction validation"""
    
    def __init__(self):
        """Initialize rule-based engine with default rules"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rules = self._initialize_default_rules()
        self.custom_rules = []
        
    def _initialize_default_rules(self) -> List[Dict[str, Any]]:
        """Initialize default fraud detection rules"""
        return [
            {
                'name': 'high_amount_new_account',
                'condition': lambda t: (
                    t.get('transaction_amount', 0) > 5000 and
                    t.get('account_age_days', 0) < 30
                ),
                'risk_contribution': 0.8,
                'description': 'High amount transaction on new account'
            },
            {
                'name': 'multiple_fraud_history',
                'condition': lambda t: t.get('previous_fraud_count', 0) >= 3,
                'risk_contribution': 0.9,
                'description': 'Account has multiple fraud history'
            },
            {
                'name': 'velocity_spike',
                'condition': lambda t: (
                    t.get('velocity_spike', False) or
                    t.get('transaction_count_24h', 0) > 10
                ),
                'risk_contribution': 0.6,
                'description': 'Unusual transaction velocity detected'
            },
            {
                'name': 'location_mismatch',
                'condition': lambda t: t.get('location_mismatch', False),
                'risk_contribution': 0.5,
                'description': 'Transaction location differs from normal pattern'
            },
            {
                'name': 'unusual_time',
                'condition': lambda t: t.get('unusual_time', False),
                'risk_contribution': 0.3,
                'description': 'Transaction at unusual time'
            },
            {
                'name': 'high_risk_merchant',
                'condition': lambda t: t.get('merchant_risk_category', 'low') == 'high',
                'risk_contribution': 0.4,
                'description': 'Transaction with high-risk merchant'
            }
        ]
        
    def add_rule(self, rule: Dict[str, Any]) -> None:
        """Add custom rule to the engine"""
        required_fields = ['name', 'condition', 'risk_contribution']
        if not all(field in rule for field in required_fields):
            raise ValueError(f"Rule must contain: {required_fields}")
            
        self.custom_rules.append(rule)
        self.logger.info(f"Added custom rule: {rule['name']}")
        
    def evaluate_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate transaction against all rules"""
        start_time = time.time()
        
        triggered_rules = []
        total_risk_score = 0.0
        
        # Evaluate default rules
        for rule in self.rules:
            try:
                if rule['condition'](transaction):
                    triggered_rules.append({
                        'name': rule['name'],
                        'description': rule.get('description', ''),
                        'risk_contribution': rule['risk_contribution']
                    })
                    total_risk_score += rule['risk_contribution']
            except Exception as e:
                self.logger.warning(f"Error evaluating rule {rule['name']}: {str(e)}")
                
        # Evaluate custom rules
        for rule in self.custom_rules:
            try:
                if rule['condition'](transaction):
                    triggered_rules.append({
                        'name': rule['name'],
                        'description': rule.get('description', ''),
                        'risk_contribution': rule['risk_contribution']
                    })
                    total_risk_score += rule['risk_contribution']
            except Exception as e:
                self.logger.warning(f"Error evaluating custom rule {rule['name']}: {str(e)}")
                
        # Normalize risk score
        normalized_score = min(total_risk_score, 1.0)
        
        # Determine risk level
        if normalized_score >= 0.8:
            risk_level = 'critical'
        elif normalized_score >= 0.6:
            risk_level = 'high'
        elif normalized_score >= 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'low'
            
        evaluation_time = time.time() - start_time
        
        return {
            'triggered_rules': triggered_rules,
            'risk_score': normalized_score,
            'risk_level': risk_level,
            'confidence': self._calculate_confidence(triggered_rules, normalized_score),
            'evaluation_time_ms': evaluation_time * 1000,
            'timestamp': datetime.now().isoformat()
        }
        
    def _calculate_confidence(self, triggered_rules: List[Dict[str, Any]], risk_score: float) -> float:
        """Calculate confidence in the rule evaluation"""
        if not triggered_rules:
            return 0.5  # Low confidence when no rules triggered
            
        # Confidence increases with number of triggered rules and their strength
        rule_confidence = min(len(triggered_rules) * 0.2, 0.8)
        score_confidence = risk_score * 0.3
        
        return min(rule_confidence + score_confidence, 1.0)
        
    def get_rule_summary(self) -> Dict[str, Any]:
        """Get summary of all available rules"""
        return {
            'default_rules': len(self.rules),
            'custom_rules': len(self.custom_rules),
            'total_rules': len(self.rules) + len(self.custom_rules),
            'rule_names': [rule['name'] for rule in self.rules + self.custom_rules]
        }
        
    def generate_proof(self, transaction: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate rule-based proof for transaction"""
        evaluation = self.evaluate_transaction(transaction)
        
        proof = {
            'engine_type': 'rule_based',
            'transaction_id': transaction.get('transaction_id', 'unknown'),
            'evaluation': evaluation,
            'proof_metadata': {
                'rules_evaluated': len(self.rules) + len(self.custom_rules),
                'rules_triggered': len(evaluation['triggered_rules']),
                'engine_version': '1.0.0',
                'generation_timestamp': datetime.now().isoformat()
            }
        }
        
        if context:
            proof['context'] = context
            
        return proof