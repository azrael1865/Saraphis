#!/usr/bin/env python3
"""
Security Knowledge Base Plugin
==============================

This module provides security knowledge base capabilities for the Universal AI Core system.
Adapted from molecular knowledge base patterns, specialized for cybersecurity intelligence,
threat management, and security rule storage.

Features:
- Threat intelligence database management
- Security rule and policy repositories
- Vulnerability and CVE knowledge storage
- Attack pattern and TTP databases
- Security indicator repositories
- Risk assessment knowledge management
- Compliance requirement storage
- Incident response knowledge base
"""

import logging
import json
import time
import hashlib
import pickle
import threading
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
import re

# Import plugin base classes
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from base import (
    KnowledgeBasePlugin, KnowledgeItem, QueryResult, KnowledgeBaseMetadata,
    KnowledgeType, QueryType, KnowledgeFormat, OperationStatus
)

logger = logging.getLogger(__name__)


@dataclass
class SecurityKnowledgeItem(KnowledgeItem):
    """Extended knowledge item for security data"""
    threat_type: str = ""  # malware, phishing, vulnerability, etc.
    severity: str = "medium"  # low, medium, high, critical
    confidence: float = 0.0  # 0.0 to 1.0
    ioc_type: str = ""  # ip, domain, hash, signature, etc.
    ioc_value: str = ""
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    threat_actors: List[str] = field(default_factory=list)
    attack_vectors: List[str] = field(default_factory=list)
    mitre_tactics: List[str] = field(default_factory=list)
    mitre_techniques: List[str] = field(default_factory=list)
    cvss_score: Optional[float] = None
    cve_ids: List[str] = field(default_factory=list)
    affected_platforms: List[str] = field(default_factory=list)
    
    def is_active(self, time_window_hours: int = 72) -> bool:
        """Check if threat indicator is active within time window"""
        if not self.last_seen:
            return True  # Assume active if no last_seen date
        cutoff = datetime.utcnow() - timedelta(hours=time_window_hours)
        return self.last_seen >= cutoff
    
    def calculate_risk_score(self) -> float:
        """Calculate risk score based on severity, confidence, and recency"""
        severity_weights = {
            'low': 0.25,
            'medium': 0.5, 
            'high': 0.75,
            'critical': 1.0
        }
        
        base_score = severity_weights.get(self.severity.lower(), 0.5)
        confidence_factor = self.confidence
        
        # Recency factor (more recent = higher risk)
        recency_factor = 1.0
        if self.last_seen:
            hours_since = (datetime.utcnow() - self.last_seen).total_seconds() / 3600
            if hours_since > 168:  # More than a week old
                recency_factor = 0.5
            elif hours_since > 72:  # More than 3 days old
                recency_factor = 0.7
        
        return min(1.0, base_score * confidence_factor * recency_factor)


class ThreatIntelligenceSearcher:
    """Threat intelligence search engine adapted from molecular similarity searcher"""
    
    def __init__(self):
        self.ioc_db = {}  # ioc_value -> SecurityKnowledgeItem
        self.hash_index = defaultdict(list)  # hash -> [item_ids]
        self.domain_index = defaultdict(list)  # domain -> [item_ids]
        self.ip_index = defaultdict(list)  # ip -> [item_ids]
        self.signature_index = defaultdict(list)  # signature -> [item_ids]
        self.logger = logging.getLogger(f"{__name__}.ThreatIntelligenceSearcher")
    
    def add_threat_indicator(self, item: SecurityKnowledgeItem):
        """Add threat indicator to search index"""
        try:
            if item.ioc_value:
                self.ioc_db[item.ioc_value] = item
                
                # Index by IOC type
                if item.ioc_type == 'hash':
                    self.hash_index[item.ioc_value].append(item.id)
                elif item.ioc_type == 'domain':
                    self.domain_index[item.ioc_value].append(item.id)
                elif item.ioc_type == 'ip':
                    self.ip_index[item.ioc_value].append(item.id)
                elif item.ioc_type == 'signature':
                    self.signature_index[item.ioc_value].append(item.id)
                    
        except Exception as e:
            self.logger.error(f"Error adding threat indicator to search index: {e}")
    
    def find_similar_threats(self, query_ioc: str, ioc_type: str, 
                           similarity_threshold: float = 0.7, 
                           max_results: int = 100) -> List[Tuple[str, float]]:
        """Find threats similar to query IOC"""
        try:
            results = []
            
            # Exact match first
            if query_ioc in self.ioc_db:
                results.append((self.ioc_db[query_ioc].id, 1.0))
            
            # Fuzzy matching for different IOC types
            if ioc_type == 'domain':
                results.extend(self._find_similar_domains(query_ioc, similarity_threshold))
            elif ioc_type == 'ip':
                results.extend(self._find_similar_ips(query_ioc, similarity_threshold))
            elif ioc_type == 'hash':
                results.extend(self._find_similar_hashes(query_ioc, similarity_threshold))
            elif ioc_type == 'signature':
                results.extend(self._find_similar_signatures(query_ioc, similarity_threshold))
            
            # Sort by similarity and limit results
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:max_results]
            
        except Exception as e:
            self.logger.error(f"Error in threat similarity search: {e}")
            return []
    
    def _find_similar_domains(self, query_domain: str, threshold: float) -> List[Tuple[str, float]]:
        """Find domains similar to query domain"""
        results = []
        query_parts = query_domain.split('.')
        
        for domain, item_ids in self.domain_index.items():
            if domain == query_domain:
                continue
                
            domain_parts = domain.split('.')
            
            # Calculate domain similarity
            similarity = self._calculate_domain_similarity(query_parts, domain_parts)
            
            if similarity >= threshold:
                for item_id in item_ids:
                    results.append((item_id, similarity))
        
        return results
    
    def _find_similar_ips(self, query_ip: str, threshold: float) -> List[Tuple[str, float]]:
        """Find IPs similar to query IP (same subnet, etc.)"""
        results = []
        
        try:
            import ipaddress
            query_network = ipaddress.ip_network(f"{query_ip}/24", strict=False)
            
            for ip, item_ids in self.ip_index.items():
                if ip == query_ip:
                    continue
                
                try:
                    ip_addr = ipaddress.ip_address(ip)
                    if ip_addr in query_network:
                        similarity = 0.8  # Same subnet
                        for item_id in item_ids:
                            results.append((item_id, similarity))
                except:
                    continue
                    
        except Exception:
            # Fallback to simple string similarity
            for ip, item_ids in self.ip_index.items():
                if ip == query_ip:
                    continue
                    
                similarity = self._calculate_string_similarity(query_ip, ip)
                if similarity >= threshold:
                    for item_id in item_ids:
                        results.append((item_id, similarity))
        
        return results
    
    def _find_similar_hashes(self, query_hash: str, threshold: float) -> List[Tuple[str, float]]:
        """Find hashes similar to query hash"""
        results = []
        
        # Hash similarity is typically exact match or fuzzy hash comparison
        # For demo purposes, we'll use string similarity
        for hash_val, item_ids in self.hash_index.items():
            if hash_val == query_hash:
                continue
                
            similarity = self._calculate_string_similarity(query_hash, hash_val)
            if similarity >= threshold:
                for item_id in item_ids:
                    results.append((item_id, similarity))
        
        return results
    
    def _find_similar_signatures(self, query_sig: str, threshold: float) -> List[Tuple[str, float]]:
        """Find signatures similar to query signature"""
        results = []
        
        for signature, item_ids in self.signature_index.items():
            if signature == query_sig:
                continue
                
            # Use regex and string similarity for signature matching
            similarity = max(
                self._calculate_string_similarity(query_sig, signature),
                self._calculate_regex_similarity(query_sig, signature)
            )
            
            if similarity >= threshold:
                for item_id in item_ids:
                    results.append((item_id, similarity))
        
        return results
    
    def _calculate_domain_similarity(self, domain1_parts: List[str], domain2_parts: List[str]) -> float:
        """Calculate domain similarity based on parts"""
        if not domain1_parts or not domain2_parts:
            return 0.0
        
        # Check TLD and second-level domain
        tld_match = domain1_parts[-1] == domain2_parts[-1] if len(domain1_parts) > 0 and len(domain2_parts) > 0 else False
        sld_match = (domain1_parts[-2] == domain2_parts[-2] 
                    if len(domain1_parts) > 1 and len(domain2_parts) > 1 else False)
        
        if tld_match and sld_match:
            return 0.9
        elif tld_match:
            return 0.6
        elif sld_match:
            return 0.7
        else:
            return self._calculate_string_similarity('.'.join(domain1_parts), '.'.join(domain2_parts))
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Levenshtein distance"""
        if not str1 or not str2:
            return 0.0
        
        # Simple Levenshtein distance implementation
        len1, len2 = len(str1), len(str2)
        if len1 == 0:
            return 0.0 if len2 > 0 else 1.0
        if len2 == 0:
            return 0.0
        
        # Create distance matrix
        distances = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(len1 + 1):
            distances[i][0] = i
        for j in range(len2 + 1):
            distances[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if str1[i-1] == str2[j-1] else 1
                distances[i][j] = min(
                    distances[i-1][j] + 1,      # deletion
                    distances[i][j-1] + 1,      # insertion
                    distances[i-1][j-1] + cost  # substitution
                )
        
        max_len = max(len1, len2)
        return 1.0 - (distances[len1][len2] / max_len)
    
    def _calculate_regex_similarity(self, pattern1: str, pattern2: str) -> float:
        """Calculate similarity between regex patterns"""
        try:
            # Simple pattern similarity - count common regex elements
            common_elements = 0
            total_elements = 0
            
            regex_elements = [r'\d', r'\w', r'\s', r'\.', r'\*', r'\+', r'\?', r'\[', r'\]']
            
            for element in regex_elements:
                in_pattern1 = element in pattern1
                in_pattern2 = element in pattern2
                total_elements += 1
                
                if in_pattern1 and in_pattern2:
                    common_elements += 1
                elif not in_pattern1 and not in_pattern2:
                    common_elements += 0.5  # Partial credit for both not having it
            
            return common_elements / total_elements if total_elements > 0 else 0.0
            
        except Exception:
            return 0.0


class SecurityRuleDatabase:
    """Security rule knowledge management adapted from chemical rule engine"""
    
    def __init__(self):
        self.rules = self._initialize_security_rules()
        self.custom_rules = {}
        self.rule_applications = defaultdict(list)
        self.logger = logging.getLogger(f"{__name__}.SecurityRuleDatabase")
    
    def _initialize_security_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize security rules database"""
        return {
            "network_security": {
                "suspicious_port_activity": {
                    "description": "Detect suspicious port scanning activity",
                    "conditions": {
                        "unique_ports_contacted": {"operator": ">", "threshold": 100},
                        "connection_attempts": {"operator": ">", "threshold": 1000},
                        "time_window_minutes": {"operator": "<=", "threshold": 60}
                    },
                    "severity": "high",
                    "mitre_techniques": ["T1046"],
                    "indicators": ["port_scan", "reconnaissance"],
                    "response_actions": ["block_ip", "alert_soc"],
                    "confidence": 0.85
                },
                "dns_tunneling": {
                    "description": "Detect DNS tunneling attempts",
                    "conditions": {
                        "dns_query_length": {"operator": ">", "threshold": 255},
                        "dns_query_entropy": {"operator": ">", "threshold": 4.5},
                        "unusual_dns_types": {"operator": ">", "threshold": 0},
                        "dns_frequency": {"operator": ">", "threshold": 100}
                    },
                    "severity": "high",
                    "mitre_techniques": ["T1071.004"],
                    "indicators": ["dns_tunneling", "data_exfiltration"],
                    "response_actions": ["block_domain", "deep_packet_inspection"],
                    "confidence": 0.80
                },
                "beaconing_behavior": {
                    "description": "Detect C2 beaconing behavior",
                    "conditions": {
                        "connection_regularity": {"operator": ">", "threshold": 0.9},
                        "connection_frequency": {"operator": ">", "threshold": 10},
                        "payload_size_consistency": {"operator": ">", "threshold": 0.8},
                        "external_destinations": {"operator": ">", "threshold": 0}
                    },
                    "severity": "critical",
                    "mitre_techniques": ["T1071", "T1573"],
                    "indicators": ["c2_beacon", "malware_communication"],
                    "response_actions": ["isolate_host", "block_domain", "incident_response"],
                    "confidence": 0.90
                }
            },
            
            "endpoint_security": {
                "process_injection": {
                    "description": "Detect process injection techniques",
                    "conditions": {
                        "dll_injection_attempts": {"operator": ">", "threshold": 0},
                        "process_hollowing": {"operator": ">", "threshold": 0},
                        "reflective_dll_loading": {"operator": ">", "threshold": 0},
                        "cross_process_memory_access": {"operator": ">", "threshold": 5}
                    },
                    "severity": "critical",
                    "mitre_techniques": ["T1055"],
                    "indicators": ["process_injection", "malware_execution"],
                    "response_actions": ["terminate_process", "isolate_host", "forensic_analysis"],
                    "confidence": 0.95
                },
                "privilege_escalation": {
                    "description": "Detect privilege escalation attempts",
                    "conditions": {
                        "token_manipulation": {"operator": ">", "threshold": 0},
                        "uac_bypass_attempts": {"operator": ">", "threshold": 0},
                        "service_creation": {"operator": ">", "threshold": 3},
                        "registry_modifications": {"operator": ">", "threshold": 10}
                    },
                    "severity": "high",
                    "mitre_techniques": ["T1548", "T1134"],
                    "indicators": ["privilege_escalation", "local_exploitation"],
                    "response_actions": ["alert_admin", "audit_privileges", "monitor_closely"],
                    "confidence": 0.85
                },
                "persistence_mechanisms": {
                    "description": "Detect persistence establishment",
                    "conditions": {
                        "startup_modifications": {"operator": ">", "threshold": 0},
                        "scheduled_task_creation": {"operator": ">", "threshold": 2},
                        "service_installation": {"operator": ">", "threshold": 1},
                        "registry_autorun_keys": {"operator": ">", "threshold": 0}
                    },
                    "severity": "medium",
                    "mitre_techniques": ["T1547", "T1053"],
                    "indicators": ["persistence", "malware_installation"],
                    "response_actions": ["remove_persistence", "system_restore", "monitor"],
                    "confidence": 0.75
                }
            },
            
            "data_security": {
                "data_exfiltration": {
                    "description": "Detect data exfiltration attempts",
                    "conditions": {
                        "large_outbound_transfers": {"operator": ">", "threshold": 1000000},  # 1MB
                        "unusual_upload_destinations": {"operator": ">", "threshold": 0},
                        "compressed_file_transfers": {"operator": ">", "threshold": 5},
                        "encrypted_channel_usage": {"operator": ">", "threshold": 0}
                    },
                    "severity": "critical",
                    "mitre_techniques": ["T1041", "T1030"],
                    "indicators": ["data_theft", "intellectual_property_theft"],
                    "response_actions": ["block_transfer", "isolate_host", "data_loss_prevention"],
                    "confidence": 0.80
                },
                "credential_theft": {
                    "description": "Detect credential harvesting",
                    "conditions": {
                        "lsass_access_attempts": {"operator": ">", "threshold": 0},
                        "credential_dumping_tools": {"operator": ">", "threshold": 0},
                        "password_cracking_attempts": {"operator": ">", "threshold": 100},
                        "keylogger_behavior": {"operator": ">", "threshold": 0}
                    },
                    "severity": "critical",
                    "mitre_techniques": ["T1003", "T1056"],
                    "indicators": ["credential_theft", "lateral_movement_prep"],
                    "response_actions": ["force_password_reset", "isolate_host", "revoke_sessions"],
                    "confidence": 0.90
                }
            },
            
            "web_security": {
                "sql_injection": {
                    "description": "Detect SQL injection attempts",
                    "conditions": {
                        "sql_keywords_in_params": {"operator": ">", "threshold": 3},
                        "union_select_patterns": {"operator": ">", "threshold": 0},
                        "comment_injection": {"operator": ">", "threshold": 0},
                        "error_based_patterns": {"operator": ">", "threshold": 0}
                    },
                    "severity": "high",
                    "mitre_techniques": ["T1190"],
                    "indicators": ["sql_injection", "web_exploitation"],
                    "response_actions": ["block_request", "waf_rule_update", "security_review"],
                    "confidence": 0.85
                },
                "xss_attacks": {
                    "description": "Detect cross-site scripting attempts",
                    "conditions": {
                        "script_tags_in_params": {"operator": ">", "threshold": 0},
                        "javascript_events": {"operator": ">", "threshold": 0},
                        "encoded_payloads": {"operator": ">", "threshold": 0},
                        "dom_manipulation": {"operator": ">", "threshold": 0}
                    },
                    "severity": "medium",
                    "mitre_techniques": ["T1190"],
                    "indicators": ["xss_attack", "client_side_exploitation"],
                    "response_actions": ["sanitize_input", "csp_enforcement", "user_education"],
                    "confidence": 0.80
                }
            },
            
            "compliance": {
                "pci_dss_violations": {
                    "description": "Detect PCI DSS compliance violations",
                    "conditions": {
                        "unencrypted_card_data": {"operator": ">", "threshold": 0},
                        "weak_authentication": {"operator": ">", "threshold": 0},
                        "insecure_transmission": {"operator": ">", "threshold": 0},
                        "insufficient_logging": {"operator": "==", "threshold": 1}
                    },
                    "severity": "critical",
                    "compliance_framework": "PCI DSS",
                    "indicators": ["compliance_violation", "data_protection_failure"],
                    "response_actions": ["immediate_remediation", "compliance_report", "audit_trail"],
                    "confidence": 0.95
                },
                "gdpr_violations": {
                    "description": "Detect GDPR compliance violations",
                    "conditions": {
                        "personal_data_exposure": {"operator": ">", "threshold": 0},
                        "consent_violations": {"operator": ">", "threshold": 0},
                        "data_retention_violations": {"operator": ">", "threshold": 0},
                        "cross_border_transfer_violations": {"operator": ">", "threshold": 0}
                    },
                    "severity": "critical",
                    "compliance_framework": "GDPR",
                    "indicators": ["privacy_violation", "regulatory_breach"],
                    "response_actions": ["data_protection_measures", "breach_notification", "dpo_alert"],
                    "confidence": 0.90
                }
            }
        }
    
    def apply_rule(self, rule_category: str, rule_name: str, 
                  security_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply security rule to data"""
        try:
            if rule_category not in self.rules or rule_name not in self.rules[rule_category]:
                return {"applicable": False, "error": "Rule not found"}
            
            rule = self.rules[rule_category][rule_name]
            result = {
                "rule_category": rule_category,
                "rule_name": rule_name,
                "description": rule["description"],
                "applicable": True,
                "triggered": False,
                "violations": [],
                "severity": rule.get("severity", "medium"),
                "confidence": rule.get("confidence", 0.5),
                "mitre_techniques": rule.get("mitre_techniques", []),
                "indicators": rule.get("indicators", []),
                "response_actions": rule.get("response_actions", [])
            }
            
            # Check conditions
            if "conditions" in rule:
                violations = 0
                for condition_name, condition in rule["conditions"].items():
                    if condition_name in security_data:
                        value = security_data[condition_name]
                        operator = condition["operator"]
                        threshold = condition["threshold"]
                        
                        if self._evaluate_condition(value, operator, threshold):
                            violations += 1
                            result["violations"].append({
                                "condition": condition_name,
                                "value": value,
                                "operator": operator,
                                "threshold": threshold
                            })
                
                # Rule is triggered if any conditions are met
                result["triggered"] = violations > 0
                result["violation_count"] = violations
            
            # Record rule application
            self.rule_applications[f"{rule_category}.{rule_name}"].append({
                "timestamp": datetime.utcnow(),
                "result": result,
                "data_signature": hashlib.md5(str(security_data).encode()).hexdigest()
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying rule {rule_category}.{rule_name}: {e}")
            return {"applicable": False, "error": str(e)}
    
    def _evaluate_condition(self, value: Any, operator: str, threshold: Any) -> bool:
        """Evaluate rule condition"""
        try:
            if operator == "<=":
                return float(value) <= float(threshold)
            elif operator == ">=":
                return float(value) >= float(threshold)
            elif operator == "<":
                return float(value) < float(threshold)
            elif operator == ">":
                return float(value) > float(threshold)
            elif operator == "==":
                return value == threshold
            elif operator == "!=":
                return value != threshold
            elif operator == "contains":
                return str(threshold) in str(value)
            elif operator == "matches":
                return bool(re.search(str(threshold), str(value)))
        except Exception:
            return False
        return False
    
    def get_applicable_rules(self, security_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get all applicable rules for given security data"""
        applicable_rules = []
        
        for category, rules in self.rules.items():
            for rule_name in rules:
                result = self.apply_rule(category, rule_name, security_data)
                if result.get("applicable", False):
                    applicable_rules.append(result)
        
        return applicable_rules


class SecurityKnowledgeBase(KnowledgeBasePlugin):
    """
    Security knowledge base plugin for cybersecurity intelligence.
    
    Provides specialized storage and retrieval for security knowledge,
    including threat indicators, attack patterns, vulnerabilities, and rules.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the security knowledge base"""
        super().__init__(config)
        
        # Configuration
        self.enable_threat_intelligence = self.config.get('threat_intelligence', True)
        self.threat_retention_days = self.config.get('threat_retention_days', 365)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # Specialized storage
        self.security_storage = {}  # item_id -> SecurityKnowledgeItem
        self.ioc_index = defaultdict(list)  # ioc_value -> [item_ids]
        self.threat_type_index = defaultdict(list)  # threat_type -> [item_ids]
        self.severity_index = defaultdict(list)  # severity -> [item_ids]
        self.mitre_index = defaultdict(list)  # technique -> [item_ids]
        self.cve_index = defaultdict(list)  # cve_id -> [item_ids]
        
        # Security tools
        self.threat_searcher = ThreatIntelligenceSearcher() if self.enable_threat_intelligence else None
        self.rule_database = SecurityRuleDatabase()
        
        # Statistics
        self.security_stats = {
            'threats_stored': 0,
            'iocs_indexed': 0,
            'rules_applied': 0,
            'threat_searches': 0,
            'active_threats': 0
        }
        
        self.logger.info(f"🛡️ Security Knowledge Base initialized (TI: {self.enable_threat_intelligence})")
    
    def _create_metadata(self) -> KnowledgeBaseMetadata:
        """Create metadata for security knowledge base"""
        return KnowledgeBaseMetadata(
            name="SecurityKnowledgeBase",
            version="1.0.0",
            author="Universal AI Core",
            description="Specialized knowledge base for cybersecurity intelligence and threat management",
            supported_knowledge_types=[
                KnowledgeType.FACTUAL,
                KnowledgeType.PROCEDURAL,
                KnowledgeType.PATTERN,
                KnowledgeType.RULE
            ],
            supported_formats=[
                KnowledgeFormat.JSON,
                KnowledgeFormat.GRAPH,
                KnowledgeFormat.VECTOR
            ],
            supported_query_types=[
                QueryType.EXACT_MATCH,
                QueryType.FUZZY_SEARCH,
                QueryType.SIMILARITY_SEARCH,
                QueryType.PATTERN_MATCH
            ],
            storage_backend="security_specialized",
            indexing_method="threat_intelligence",
            vector_dimension=1024,  # Security feature dimension
            capabilities=[
                "threat_intelligence_search",
                "security_rule_application",
                "vulnerability_management",
                "ioc_indexing",
                "mitre_mapping",
                "compliance_tracking"
            ]
        )
    
    def connect(self) -> bool:
        """Connect to security knowledge base"""
        try:
            # Initialize indices
            self._initialize_security_indices()
            self._is_connected = True
            self.logger.info("✅ Connected to security knowledge base")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to security knowledge base: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from security knowledge base"""
        self._is_connected = False
        self.logger.info("🔌 Disconnected from security knowledge base")
    
    def store_knowledge(self, item: KnowledgeItem) -> bool:
        """Store security knowledge item"""
        try:
            # Convert to security knowledge item if needed
            if isinstance(item, KnowledgeItem) and not isinstance(item, SecurityKnowledgeItem):
                sec_item = self._convert_to_security_item(item)
            else:
                sec_item = item
            
            # Validate security data
            if sec_item.ioc_value and not self._validate_ioc(sec_item.ioc_value, sec_item.ioc_type):
                self.logger.warning(f"Invalid IOC: {sec_item.ioc_type}:{sec_item.ioc_value}")
                return False
            
            # Store in main storage
            self.security_storage[sec_item.id] = sec_item
            
            # Update indices
            self._update_security_indices(sec_item)
            
            # Add to threat searcher
            if self.threat_searcher and sec_item.ioc_value:
                self.threat_searcher.add_threat_indicator(sec_item)
            
            self.security_stats['threats_stored'] += 1
            if sec_item.ioc_value:
                self.security_stats['iocs_indexed'] += 1
            if sec_item.is_active():
                self.security_stats['active_threats'] += 1
            
            self._knowledge_count += 1
            
            self.logger.debug(f"🔒 Stored security knowledge: {sec_item.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error storing security knowledge {item.id}: {e}")
            return False
    
    def retrieve_knowledge(self, item_id: str) -> Optional[SecurityKnowledgeItem]:
        """Retrieve security knowledge item by ID"""
        return self.security_storage.get(item_id)
    
    def query_knowledge(self, query: str, query_type: QueryType = QueryType.FUZZY_SEARCH,
                       max_results: int = 10, **kwargs) -> QueryResult:
        """Query security knowledge base"""
        start_time = time.time()
        
        try:
            if query_type == QueryType.EXACT_MATCH:
                results = self._exact_match_search(query, max_results)
            elif query_type == QueryType.FUZZY_SEARCH:
                results = self._fuzzy_search(query, max_results)
            elif query_type == QueryType.SIMILARITY_SEARCH:
                results = self._threat_similarity_search(query, max_results, **kwargs)
            elif query_type == QueryType.PATTERN_MATCH:
                results = self._pattern_match_search(query, max_results, **kwargs)
            else:
                results = []
            
            query_time = time.time() - start_time
            
            return QueryResult(
                items=results,
                query=query,
                query_type=query_type,
                total_results=len(results),
                retrieved_count=len(results),
                query_time=query_time,
                status=OperationStatus.SUCCESS
            )
            
        except Exception as e:
            query_time = time.time() - start_time
            self.logger.error(f"❌ Query error: {e}")
            
            return QueryResult(
                items=[],
                query=query,
                query_type=query_type,
                total_results=0,
                retrieved_count=0,
                query_time=query_time,
                status=OperationStatus.ERROR,
                error_message=str(e)
            )
    
    def _exact_match_search(self, query: str, max_results: int) -> List[SecurityKnowledgeItem]:
        """Exact match search for IOCs, CVEs, etc."""
        results = []
        
        # Search by IOC value
        if query in [item.ioc_value for item in self.security_storage.values()]:
            for item in self.security_storage.values():
                if item.ioc_value == query:
                    results.append(item)
                    if len(results) >= max_results:
                        break
        
        # Search by CVE ID
        for item in self.security_storage.values():
            if query in item.cve_ids and item not in results:
                results.append(item)
                if len(results) >= max_results:
                    break
        
        # Search by ID
        if query in self.security_storage:
            item = self.security_storage[query]
            if item not in results:
                results.append(item)
        
        return results[:max_results]
    
    def _fuzzy_search(self, query: str, max_results: int) -> List[SecurityKnowledgeItem]:
        """Fuzzy search in content and metadata"""
        results = []
        query_lower = query.lower()
        
        for item in self.security_storage.values():
            score = 0
            
            # Search in content
            if hasattr(item, 'content') and query_lower in str(item.content).lower():
                score += 2
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in item.tags):
                score += 3
            
            # Search in threat type
            if query_lower in item.threat_type.lower():
                score += 3
            
            # Search in threat actors
            if any(query_lower in actor.lower() for actor in item.threat_actors):
                score += 2
            
            # Search in MITRE techniques
            if any(query_lower in technique.lower() for technique in item.mitre_techniques):
                score += 2
            
            # Search in IOC value
            if query_lower in item.ioc_value.lower():
                score += 4
            
            if score > 0:
                results.append((item, score))
        
        # Sort by score and return items
        results.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in results[:max_results]]
    
    def _threat_similarity_search(self, query: str, max_results: int, **kwargs) -> List[SecurityKnowledgeItem]:
        """Threat intelligence similarity search"""
        if not self.threat_searcher:
            return []
        
        ioc_type = kwargs.get('ioc_type', 'unknown')
        threshold = kwargs.get('similarity_threshold', self.confidence_threshold)
        
        # Find similar threats
        similar_ids = self.threat_searcher.find_similar_threats(query, ioc_type, threshold, max_results)
        
        results = []
        for item_id, similarity in similar_ids:
            item = self.security_storage.get(item_id)
            if item:
                # Add similarity score to metadata
                item_copy = SecurityKnowledgeItem(**item.__dict__)
                item_copy.metadata = item_copy.metadata.copy()
                item_copy.metadata['similarity_score'] = similarity
                results.append(item_copy)
        
        self.security_stats['threat_searches'] += 1
        return results
    
    def _pattern_match_search(self, query: str, max_results: int, **kwargs) -> List[SecurityKnowledgeItem]:
        """Pattern-based search for security attributes"""
        results = []
        
        # Severity search: "severity:high"
        if query.startswith('severity:'):
            severity = query.split(':', 1)[1].strip()
            results = [item for item in self.security_storage.values() 
                      if item.severity.lower() == severity.lower()]
        
        # Threat type search: "type:malware"
        elif query.startswith('type:'):
            threat_type = query.split(':', 1)[1].strip()
            results = [item for item in self.security_storage.values() 
                      if threat_type.lower() in item.threat_type.lower()]
        
        # MITRE technique search: "mitre:T1055"
        elif query.startswith('mitre:'):
            technique = query.split(':', 1)[1].strip()
            results = [item for item in self.security_storage.values() 
                      if technique in item.mitre_techniques]
        
        # CVE search: "cve:CVE-2021-44228"
        elif query.startswith('cve:'):
            cve_id = query.split(':', 1)[1].strip()
            results = [item for item in self.security_storage.values() 
                      if cve_id in item.cve_ids]
        
        # Active threats: "active:true"
        elif query.startswith('active:'):
            active_filter = query.split(':', 1)[1].strip().lower() == 'true'
            results = [item for item in self.security_storage.values() 
                      if item.is_active() == active_filter]
        
        return results[:max_results]
    
    def _convert_to_security_item(self, item: KnowledgeItem) -> SecurityKnowledgeItem:
        """Convert generic knowledge item to security knowledge item"""
        sec_item = SecurityKnowledgeItem(**item.__dict__)
        
        # Extract security data from content if it's a dictionary
        if isinstance(item.content, dict):
            sec_item.threat_type = item.content.get('threat_type', '')
            sec_item.severity = item.content.get('severity', 'medium')
            sec_item.confidence = item.content.get('confidence', 0.0)
            sec_item.ioc_type = item.content.get('ioc_type', '')
            sec_item.ioc_value = item.content.get('ioc_value', '')
            sec_item.threat_actors = item.content.get('threat_actors', [])
            sec_item.attack_vectors = item.content.get('attack_vectors', [])
            sec_item.mitre_tactics = item.content.get('mitre_tactics', [])
            sec_item.mitre_techniques = item.content.get('mitre_techniques', [])
            sec_item.cvss_score = item.content.get('cvss_score')
            sec_item.cve_ids = item.content.get('cve_ids', [])
            sec_item.affected_platforms = item.content.get('affected_platforms', [])
            
            # Parse timestamps
            if 'first_seen' in item.content:
                try:
                    sec_item.first_seen = datetime.fromisoformat(item.content['first_seen'])
                except:
                    pass
            if 'last_seen' in item.content:
                try:
                    sec_item.last_seen = datetime.fromisoformat(item.content['last_seen'])
                except:
                    pass
        
        return sec_item
    
    def _validate_ioc(self, ioc_value: str, ioc_type: str) -> bool:
        """Validate IOC value format"""
        try:
            if ioc_type == 'ip':
                import ipaddress
                ipaddress.ip_address(ioc_value)
                return True
            elif ioc_type == 'domain':
                # Simple domain validation
                return '.' in ioc_value and len(ioc_value) > 3
            elif ioc_type == 'hash':
                # Validate hash length (MD5, SHA1, SHA256)
                return len(ioc_value) in [32, 40, 64] and all(c in '0123456789abcdefABCDEF' for c in ioc_value)
            elif ioc_type == 'url':
                return ioc_value.startswith(('http://', 'https://'))
            else:
                return True  # Accept other types for now
        except:
            return False
    
    def _initialize_security_indices(self):
        """Initialize security indices"""
        self.ioc_index = defaultdict(list)
        self.threat_type_index = defaultdict(list)
        self.severity_index = defaultdict(list)
        self.mitre_index = defaultdict(list)
        self.cve_index = defaultdict(list)
    
    def _update_security_indices(self, item: SecurityKnowledgeItem):
        """Update all indices for the security item"""
        # IOC index
        if item.ioc_value:
            self.ioc_index[item.ioc_value].append(item.id)
        
        # Threat type index
        if item.threat_type:
            self.threat_type_index[item.threat_type].append(item.id)
        
        # Severity index
        self.severity_index[item.severity].append(item.id)
        
        # MITRE technique index
        for technique in item.mitre_techniques:
            self.mitre_index[technique].append(item.id)
        
        # CVE index
        for cve_id in item.cve_ids:
            self.cve_index[cve_id].append(item.id)
    
    def apply_security_rule(self, item_id: str, rule_category: str, rule_name: str) -> Dict[str, Any]:
        """Apply security rule to knowledge item"""
        item = self.security_storage.get(item_id)
        if not item:
            return {"error": "Item not found"}
        
        # Prepare security data for rule application
        security_data = {
            'threat_type': item.threat_type,
            'severity': item.severity,
            'confidence': item.confidence,
            'cvss_score': item.cvss_score or 0.0,
            'mitre_techniques_count': len(item.mitre_techniques),
            'threat_actors_count': len(item.threat_actors),
            'is_active': 1 if item.is_active() else 0
        }
        
        result = self.rule_database.apply_rule(rule_category, rule_name, security_data)
        self.security_stats['rules_applied'] += 1
        
        return result
    
    def get_threats_by_severity(self, severity: str) -> List[SecurityKnowledgeItem]:
        """Get all threats by severity level"""
        item_ids = self.severity_index.get(severity, [])
        return [self.security_storage[item_id] for item_id in item_ids if item_id in self.security_storage]
    
    def get_threats_by_mitre_technique(self, technique: str) -> List[SecurityKnowledgeItem]:
        """Get all threats associated with MITRE technique"""
        item_ids = self.mitre_index.get(technique, [])
        return [self.security_storage[item_id] for item_id in item_ids if item_id in self.security_storage]
    
    def get_active_threats(self, time_window_hours: int = 72) -> List[SecurityKnowledgeItem]:
        """Get all active threats within time window"""
        return [item for item in self.security_storage.values() if item.is_active(time_window_hours)]
    
    def calculate_threat_landscape(self) -> Dict[str, Any]:
        """Calculate threat landscape metrics"""
        total_threats = len(self.security_storage)
        active_threats = len(self.get_active_threats())
        
        # Severity distribution
        severity_dist = {}
        for severity in ['low', 'medium', 'high', 'critical']:
            severity_dist[severity] = len(self.get_threats_by_severity(severity))
        
        # Threat type distribution
        threat_types = defaultdict(int)
        for item in self.security_storage.values():
            threat_types[item.threat_type] += 1
        
        # Top MITRE techniques
        technique_counts = defaultdict(int)
        for item in self.security_storage.values():
            for technique in item.mitre_techniques:
                technique_counts[technique] += 1
        
        top_techniques = sorted(technique_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_threats': total_threats,
            'active_threats': active_threats,
            'threat_activity_rate': active_threats / total_threats if total_threats > 0 else 0,
            'severity_distribution': dict(severity_dist),
            'threat_type_distribution': dict(threat_types),
            'top_mitre_techniques': dict(top_techniques),
            'average_confidence': np.mean([item.confidence for item in self.security_storage.values()]) if self.security_storage else 0,
            'high_confidence_threats': len([item for item in self.security_storage.values() if item.confidence > 0.8])
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get security knowledge base statistics"""
        base_stats = super().get_statistics()
        base_stats.update(self.security_stats)
        
        base_stats.update({
            "iocs_indexed": len(self.ioc_index),
            "threat_types": len(self.threat_type_index),
            "mitre_techniques": len(self.mitre_index),
            "cves_tracked": len(self.cve_index),
            "threat_searcher_size": len(self.threat_searcher.ioc_db) if self.threat_searcher else 0
        })
        
        return base_stats
    
    def health_check(self) -> bool:
        """Perform health check"""
        try:
            # Test basic functionality
            test_item = SecurityKnowledgeItem(
                id="test_threat",
                content={"test": "data"},
                knowledge_type=KnowledgeType.FACTUAL,
                format=KnowledgeFormat.JSON,
                threat_type="test",
                severity="low",
                confidence=0.8,
                ioc_type="ip",
                ioc_value="192.168.1.1"
            )
            
            # Test validation
            if not self._validate_ioc(test_item.ioc_value, test_item.ioc_type):
                return False
            
            # Test indices
            return len(self.security_storage) == len([item for sublist in self.ioc_index.values() for item in sublist])
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False


# Plugin metadata for discovery
__plugin_metadata__ = {
    "name": "SecurityKnowledgeBase",
    "version": "1.0.0",
    "author": "Universal AI Core",
    "description": "Specialized knowledge base for cybersecurity intelligence and threat management",
    "plugin_type": "knowledge_base",
    "entry_point": f"{__name__}:SecurityKnowledgeBase",
    "dependencies": [],
    "capabilities": [
        "threat_intelligence_search",
        "security_rule_application",
        "vulnerability_management",
        "ioc_indexing",
        "mitre_mapping",
        "compliance_tracking"
    ],
    "hooks": []
}


if __name__ == "__main__":
    # Test the security knowledge base
    print("🛡️ SECURITY KNOWLEDGE BASE TEST")
    print("=" * 50)
    
    # Initialize knowledge base
    config = {
        'threat_intelligence': True,
        'threat_retention_days': 365,
        'confidence_threshold': 0.7
    }
    
    sec_kb = SecurityKnowledgeBase(config)
    
    # Connect
    connected = sec_kb.connect()
    print(f"📡 Connected: {'✅' if connected else '❌'}")
    
    # Test security threats
    test_threats = [
        {
            'id': 'threat1',
            'threat_type': 'malware',
            'severity': 'high',
            'ioc_type': 'hash',
            'ioc_value': 'a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456',
            'content': {'name': 'Evil Trojan', 'family': 'Banking Trojan'},
            'tags': ['trojan', 'banking', 'credential_theft'],
            'confidence': 0.9,
            'mitre_techniques': ['T1003', 'T1056'],
            'threat_actors': ['APT29']
        },
        {
            'id': 'threat2',
            'threat_type': 'phishing',
            'severity': 'medium',
            'ioc_type': 'domain',
            'ioc_value': 'evil-bank.com',
            'content': {'name': 'Phishing Campaign', 'target': 'Financial Sector'},
            'tags': ['phishing', 'social_engineering'],
            'confidence': 0.8,
            'mitre_techniques': ['T1566'],
            'cve_ids': []
        },
        {
            'id': 'threat3',
            'threat_type': 'vulnerability',
            'severity': 'critical',
            'ioc_type': 'cve',
            'ioc_value': 'CVE-2021-44228',
            'content': {'name': 'Log4Shell', 'description': 'Log4j RCE vulnerability'},
            'tags': ['rce', 'log4j', 'zero_day'],
            'confidence': 1.0,
            'cvss_score': 10.0,
            'cve_ids': ['CVE-2021-44228'],
            'affected_platforms': ['java', 'linux', 'windows']
        }
    ]
    
    # Store threats
    print(f"\n🔒 Storing {len(test_threats)} threats...")
    for threat_data in test_threats:
        item = SecurityKnowledgeItem(
            id=threat_data['id'],
            content=threat_data['content'],
            knowledge_type=KnowledgeType.FACTUAL,
            format=KnowledgeFormat.JSON,
            tags=threat_data['tags'],
            threat_type=threat_data['threat_type'],
            severity=threat_data['severity'],
            confidence=threat_data['confidence'],
            ioc_type=threat_data['ioc_type'],
            ioc_value=threat_data['ioc_value'],
            mitre_techniques=threat_data.get('mitre_techniques', []),
            threat_actors=threat_data.get('threat_actors', []),
            cvss_score=threat_data.get('cvss_score'),
            cve_ids=threat_data.get('cve_ids', []),
            affected_platforms=threat_data.get('affected_platforms', []),
            last_seen=datetime.utcnow()
        )
        
        success = sec_kb.store_knowledge(item)
        print(f"  {threat_data['id']}: {'✅' if success else '❌'}")
    
    # Test retrieval
    print(f"\n🔍 Testing retrieval...")
    retrieved = sec_kb.retrieve_knowledge('threat1')
    print(f"Retrieved threat1: {'✅' if retrieved else '❌'}")
    
    if retrieved:
        print(f"  Type: {retrieved.threat_type}")
        print(f"  Severity: {retrieved.severity}")
        print(f"  Confidence: {retrieved.confidence}")
        print(f"  Risk Score: {retrieved.calculate_risk_score():.2f}")
    
    # Test queries
    print(f"\n🔎 Testing queries...")
    
    # Exact match by IOC
    result = sec_kb.query_knowledge('evil-bank.com', QueryType.EXACT_MATCH)
    print(f"Exact match for 'evil-bank.com': {len(result.items)} results")
    
    # Fuzzy search
    result = sec_kb.query_knowledge('phishing', QueryType.FUZZY_SEARCH)
    print(f"Fuzzy search for 'phishing': {len(result.items)} results")
    
    # Pattern search by severity
    result = sec_kb.query_knowledge('severity:critical', QueryType.PATTERN_MATCH)
    print(f"Pattern search for 'severity:critical': {len(result.items)} results")
    
    # Pattern search by MITRE technique
    result = sec_kb.query_knowledge('mitre:T1003', QueryType.PATTERN_MATCH)
    print(f"Pattern search for 'mitre:T1003': {len(result.items)} results")
    
    # Test security rule application
    if sec_kb.security_storage:
        print(f"\n⚖️ Testing security rules...")
        item_id = list(sec_kb.security_storage.keys())[0]
        rule_result = sec_kb.apply_security_rule(item_id, 'endpoint_security', 'process_injection')
        print(f"Process injection rule application: {'✅' if rule_result.get('applicable') else '❌'}")
        if rule_result.get('applicable'):
            print(f"  Triggered: {rule_result.get('triggered', False)}")
            print(f"  Severity: {rule_result.get('severity', 'unknown')}")
            print(f"  Violations: {rule_result.get('violation_count', 0)}")
    
    # Test threat landscape analysis
    print(f"\n🌐 Threat landscape analysis...")
    landscape = sec_kb.calculate_threat_landscape()
    print(f"  Total threats: {landscape['total_threats']}")
    print(f"  Active threats: {landscape['active_threats']}")
    print(f"  High confidence threats: {landscape['high_confidence_threats']}")
    print(f"  Severity distribution: {landscape['severity_distribution']}")
    
    # Test health check
    health = sec_kb.health_check()
    print(f"\n🏥 Health check: {'✅' if health else '❌'}")
    
    # Show statistics
    stats = sec_kb.get_statistics()
    print(f"\n📊 Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Disconnect
    sec_kb.disconnect()
    print("\n✅ Security knowledge base test completed!")