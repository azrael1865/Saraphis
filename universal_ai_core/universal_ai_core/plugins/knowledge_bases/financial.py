#!/usr/bin/env python3
"""
Financial Knowledge Base Plugin
===============================

This module provides financial knowledge base capabilities for the Universal AI Core system.
Adapted from molecular knowledge base patterns, specialized for financial rules, market intelligence,
and regulatory compliance management.

Features:
- Financial rule and regulation repositories
- Market intelligence database management
- Trading strategy knowledge storage
- Risk management rule databases
- Economic indicator repositories
- Regulatory compliance knowledge management
- Financial instrument metadata storage
- Portfolio optimization knowledge base
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
class FinancialKnowledgeItem(KnowledgeItem):
    """Extended knowledge item for financial data"""
    regulation_type: str = ""  # basel_iii, mifid_ii, dodd_frank, etc.
    asset_class: str = ""  # equities, fixed_income, derivatives, commodities
    market_sector: str = ""  # technology, healthcare, finance, etc.
    risk_rating: str = "medium"  # low, medium, high, extreme
    compliance_status: str = "compliant"  # compliant, non_compliant, pending
    effective_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    jurisdiction: List[str] = field(default_factory=list)
    applicable_entities: List[str] = field(default_factory=list)  # banks, brokers, funds
    financial_metrics: Dict[str, float] = field(default_factory=dict)
    currency: str = "USD"
    liquidity_tier: str = ""  # tier1, tier2, tier3
    
    def is_current(self) -> bool:
        """Check if financial rule/data is currently valid"""
        now = datetime.utcnow()
        if self.effective_date and now < self.effective_date:
            return False
        if self.expiry_date and now > self.expiry_date:
            return False
        return True
    
    def calculate_compliance_score(self) -> float:
        """Calculate compliance score based on various factors"""
        base_score = 1.0 if self.compliance_status == "compliant" else 0.0
        
        # Age factor (newer regulations have higher weight)
        if self.effective_date:
            days_since = (datetime.utcnow() - self.effective_date).days
            age_factor = max(0.1, 1.0 - (days_since / 3650))  # Decay over 10 years
        else:
            age_factor = 1.0
        
        # Risk factor
        risk_weights = {
            'low': 1.0,
            'medium': 0.8,
            'high': 0.6,
            'extreme': 0.4
        }
        risk_factor = risk_weights.get(self.risk_rating.lower(), 0.8)
        
        return min(1.0, base_score * age_factor * risk_factor)


class FinancialRuleSearcher:
    """Financial rule search engine adapted from molecular similarity searcher"""
    
    def __init__(self):
        self.regulation_index = defaultdict(set)
        self.sector_index = defaultdict(set)
        self.asset_class_index = defaultdict(set)
        self.jurisdiction_index = defaultdict(set)
        self.text_index = defaultdict(set)
        self._lock = threading.Lock()
    
    def index_item(self, item_id: str, item: FinancialKnowledgeItem):
        """Index financial knowledge item for fast search"""
        with self._lock:
            # Index by regulation type
            if item.regulation_type:
                self.regulation_index[item.regulation_type.lower()].add(item_id)
            
            # Index by sector
            if item.market_sector:
                self.sector_index[item.market_sector.lower()].add(item_id)
            
            # Index by asset class
            if item.asset_class:
                self.asset_class_index[item.asset_class.lower()].add(item_id)
            
            # Index by jurisdictions
            for jurisdiction in item.jurisdiction:
                self.jurisdiction_index[jurisdiction.lower()].add(item_id)
            
            # Index text content
            text_content = f"{item.title} {item.description} {item.content}".lower()
            words = re.findall(r'\b\w+\b', text_content)
            for word in words:
                if len(word) > 2:  # Skip very short words
                    self.text_index[word].add(item_id)
    
    def search_by_regulation(self, regulation_type: str) -> Set[str]:
        """Search by regulation type"""
        return self.regulation_index.get(regulation_type.lower(), set())
    
    def search_by_sector(self, sector: str) -> Set[str]:
        """Search by market sector"""
        return self.sector_index.get(sector.lower(), set())
    
    def search_by_asset_class(self, asset_class: str) -> Set[str]:
        """Search by asset class"""
        return self.asset_class_index.get(asset_class.lower(), set())
    
    def search_by_jurisdiction(self, jurisdiction: str) -> Set[str]:
        """Search by jurisdiction"""
        return self.jurisdiction_index.get(jurisdiction.lower(), set())
    
    def search_text(self, query: str) -> Set[str]:
        """Search in text content"""
        words = re.findall(r'\b\w+\b', query.lower())
        if not words:
            return set()
        
        result = self.text_index.get(words[0], set())
        for word in words[1:]:
            result = result.intersection(self.text_index.get(word, set()))
        
        return result


class FinancialComplianceEngine:
    """Financial compliance engine for regulatory validation"""
    
    def __init__(self):
        self.basel_iii_rules = {}
        self.mifid_ii_rules = {}
        self.dodd_frank_rules = {}
        self.solvency_ii_rules = {}
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default financial compliance rules"""
        # Basel III Capital Requirements
        self.basel_iii_rules = {
            "common_equity_tier1_ratio": {"min_threshold": 0.045, "description": "CET1 ratio minimum 4.5%"},
            "tier1_capital_ratio": {"min_threshold": 0.06, "description": "Tier 1 capital ratio minimum 6%"},
            "total_capital_ratio": {"min_threshold": 0.08, "description": "Total capital ratio minimum 8%"},
            "leverage_ratio": {"min_threshold": 0.03, "description": "Leverage ratio minimum 3%"},
            "liquidity_coverage_ratio": {"min_threshold": 1.0, "description": "LCR minimum 100%"},
            "net_stable_funding_ratio": {"min_threshold": 1.0, "description": "NSFR minimum 100%"}
        }
        
        # MiFID II Rules
        self.mifid_ii_rules = {
            "best_execution": {"required": True, "description": "Best execution obligation"},
            "client_categorization": {"required": True, "description": "Proper client categorization"},
            "position_limits": {"commodities_only": True, "description": "Position limits for commodity derivatives"},
            "transaction_reporting": {"required": True, "description": "Transaction reporting obligation"}
        }
        
        # Dodd-Frank Rules
        self.dodd_frank_rules = {
            "volcker_rule": {"proprietary_trading_limit": 0.03, "description": "Volcker rule compliance"},
            "swap_margin_requirements": {"required": True, "description": "Margin requirements for swaps"},
            "systemically_important": {"assets_threshold": 50e9, "description": "SIFI designation threshold"}
        }
    
    def validate_basel_iii_compliance(self, financial_data: Dict[str, float]) -> Dict[str, Any]:
        """Validate Basel III compliance"""
        results = {}
        
        for rule_name, rule_config in self.basel_iii_rules.items():
            if rule_name in financial_data:
                value = financial_data[rule_name]
                threshold = rule_config["min_threshold"]
                compliant = value >= threshold
                
                results[rule_name] = {
                    "compliant": compliant,
                    "value": value,
                    "threshold": threshold,
                    "description": rule_config["description"],
                    "severity": "high" if not compliant else "low"
                }
        
        return results
    
    def validate_portfolio_constraints(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate portfolio-level constraints"""
        results = {}
        
        # Maximum single position concentration
        if "position_weights" in portfolio_data:
            max_weight = max(portfolio_data["position_weights"])
            max_concentration_limit = 0.10  # 10% maximum single position
            
            results["concentration_risk"] = {
                "compliant": max_weight <= max_concentration_limit,
                "max_weight": max_weight,
                "limit": max_concentration_limit,
                "description": "Maximum single position concentration limit"
            }
        
        # Sector concentration limits
        if "sector_weights" in portfolio_data:
            sector_weights = portfolio_data["sector_weights"]
            max_sector_limit = 0.25  # 25% maximum sector exposure
            
            violations = {sector: weight for sector, weight in sector_weights.items() 
                         if weight > max_sector_limit}
            
            results["sector_concentration"] = {
                "compliant": len(violations) == 0,
                "violations": violations,
                "limit": max_sector_limit,
                "description": "Maximum sector concentration limit"
            }
        
        return results


class FinancialKnowledgeBasePlugin(KnowledgeBasePlugin):
    """
    Financial Knowledge Base Plugin
    
    Provides financial rule storage, regulatory compliance knowledge,
    and market intelligence capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.searcher = FinancialRuleSearcher()
        self.compliance_engine = FinancialComplianceEngine()
        self.items: Dict[str, FinancialKnowledgeItem] = {}
        self._lock = threading.Lock()
        
        # Load default financial knowledge
        self._load_default_knowledge()
    
    def _load_default_knowledge(self):
        """Load default financial knowledge items"""
        default_items = [
            {
                "id": "basel_iii_capital_requirements",
                "title": "Basel III Capital Requirements",
                "description": "International regulatory framework for bank capital adequacy",
                "content": "Basel III capital requirements including CET1, Tier 1, and Total Capital ratios",
                "regulation_type": "basel_iii",
                "asset_class": "banking",
                "jurisdiction": ["global", "eu", "us"],
                "effective_date": datetime(2019, 1, 1),
                "risk_rating": "high"
            },
            {
                "id": "mifid_ii_best_execution",
                "title": "MiFID II Best Execution",
                "description": "Best execution obligations under MiFID II",
                "content": "Requirements for achieving best execution when executing client orders",
                "regulation_type": "mifid_ii",
                "asset_class": "all",
                "jurisdiction": ["eu"],
                "effective_date": datetime(2018, 1, 3),
                "risk_rating": "medium"
            },
            {
                "id": "dodd_frank_volcker_rule",
                "title": "Dodd-Frank Volcker Rule",
                "description": "Restrictions on proprietary trading by banks",
                "content": "Prohibitions and restrictions on proprietary trading activities",
                "regulation_type": "dodd_frank",
                "asset_class": "all",
                "jurisdiction": ["us"],
                "effective_date": datetime(2015, 7, 21),
                "risk_rating": "high"
            }
        ]
        
        for item_data in default_items:
            item = FinancialKnowledgeItem(
                id=item_data["id"],
                title=item_data["title"],
                description=item_data["description"],
                content=item_data["content"],
                regulation_type=item_data["regulation_type"],
                asset_class=item_data["asset_class"],
                jurisdiction=item_data["jurisdiction"],
                effective_date=item_data["effective_date"],
                risk_rating=item_data["risk_rating"]
            )
            self.add_item(item)
    
    def add_item(self, item: FinancialKnowledgeItem) -> OperationStatus:
        """Add financial knowledge item"""
        try:
            with self._lock:
                self.items[item.id] = item
                self.searcher.index_item(item.id, item)
            
            logger.info(f"Added financial knowledge item: {item.id}")
            return OperationStatus.SUCCESS
            
        except Exception as e:
            logger.error(f"Error adding financial knowledge item {item.id}: {str(e)}")
            return OperationStatus.ERROR
    
    def get_item(self, item_id: str) -> Optional[FinancialKnowledgeItem]:
        """Get financial knowledge item by ID"""
        return self.items.get(item_id)
    
    def remove_item(self, item_id: str) -> OperationStatus:
        """Remove financial knowledge item"""
        try:
            with self._lock:
                if item_id in self.items:
                    del self.items[item_id]
                    # Note: For production, should also remove from searcher indexes
                    return OperationStatus.SUCCESS
                else:
                    return OperationStatus.NOT_FOUND
        except Exception as e:
            logger.error(f"Error removing financial knowledge item {item_id}: {str(e)}")
            return OperationStatus.ERROR
    
    def query(self, query_text: str, query_type: QueryType = QueryType.SEMANTIC,
              filters: Optional[Dict[str, Any]] = None, limit: int = 10) -> QueryResult:
        """Query financial knowledge base"""
        try:
            start_time = time.time()
            matching_ids = set()
            
            # Apply filters first if provided
            if filters:
                filtered_ids = set(self.items.keys())
                
                if "regulation_type" in filters:
                    reg_ids = self.searcher.search_by_regulation(filters["regulation_type"])
                    filtered_ids = filtered_ids.intersection(reg_ids) if reg_ids else filtered_ids
                
                if "asset_class" in filters:
                    asset_ids = self.searcher.search_by_asset_class(filters["asset_class"])
                    filtered_ids = filtered_ids.intersection(asset_ids) if asset_ids else filtered_ids
                
                if "jurisdiction" in filters:
                    juris_ids = self.searcher.search_by_jurisdiction(filters["jurisdiction"])
                    filtered_ids = filtered_ids.intersection(juris_ids) if juris_ids else filtered_ids
                
                if "market_sector" in filters:
                    sector_ids = self.searcher.search_by_sector(filters["market_sector"])
                    filtered_ids = filtered_ids.intersection(sector_ids) if sector_ids else filtered_ids
                
                matching_ids = filtered_ids
            else:
                matching_ids = set(self.items.keys())
            
            # Apply text search if query provided
            if query_text.strip():
                text_ids = self.searcher.search_text(query_text)
                matching_ids = matching_ids.intersection(text_ids) if text_ids else set()
            
            # Score and sort results
            scored_results = []
            for item_id in matching_ids:
                item = self.items[item_id]
                
                # Calculate relevance score
                score = self._calculate_relevance_score(item, query_text, filters)
                
                scored_results.append((score, item))
            
            # Sort by score and limit results
            scored_results.sort(key=lambda x: x[0], reverse=True)
            top_results = scored_results[:limit]
            
            # Convert to QueryResult format
            items = [item for _, item in top_results]
            scores = [score for score, _ in top_results]
            
            query_time = time.time() - start_time
            
            return QueryResult(
                items=items,
                scores=scores,
                total_found=len(matching_ids),
                query_time=query_time,
                status=OperationStatus.SUCCESS
            )
            
        except Exception as e:
            logger.error(f"Error querying financial knowledge base: {str(e)}")
            return QueryResult(
                items=[],
                scores=[],
                total_found=0,
                query_time=0.0,
                status=OperationStatus.ERROR,
                error_message=str(e)
            )
    
    def _calculate_relevance_score(self, item: FinancialKnowledgeItem, 
                                 query_text: str, filters: Optional[Dict[str, Any]]) -> float:
        """Calculate relevance score for financial knowledge item"""
        score = 0.0
        
        # Base compliance score
        score += item.calculate_compliance_score() * 0.3
        
        # Text relevance
        if query_text:
            text_content = f"{item.title} {item.description} {item.content}".lower()
            query_words = re.findall(r'\b\w+\b', query_text.lower())
            
            for word in query_words:
                if word in text_content:
                    score += 0.1
        
        # Currency/recency bonus
        if item.is_current():
            score += 0.2
        
        # Risk rating adjustment
        risk_weights = {"low": 0.1, "medium": 0.05, "high": 0.0, "extreme": -0.1}
        score += risk_weights.get(item.risk_rating.lower(), 0.0)
        
        return min(1.0, score)
    
    def validate_compliance(self, entity_data: Dict[str, Any], 
                          regulation_type: str = "basel_iii") -> Dict[str, Any]:
        """Validate compliance against financial regulations"""
        try:
            if regulation_type == "basel_iii":
                return self.compliance_engine.validate_basel_iii_compliance(entity_data)
            elif regulation_type == "portfolio_constraints":
                return self.compliance_engine.validate_portfolio_constraints(entity_data)
            else:
                return {"error": f"Unknown regulation type: {regulation_type}"}
                
        except Exception as e:
            logger.error(f"Error validating compliance: {str(e)}")
            return {"error": str(e)}
    
    def get_regulatory_updates(self, jurisdiction: str = "all", 
                             days_back: int = 30) -> List[FinancialKnowledgeItem]:
        """Get recent regulatory updates"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        updates = []
        for item in self.items.values():
            if item.effective_date and item.effective_date >= cutoff_date:
                if jurisdiction == "all" or jurisdiction.lower() in [j.lower() for j in item.jurisdiction]:
                    updates.append(item)
        
        # Sort by effective date (most recent first)
        updates.sort(key=lambda x: x.effective_date or datetime.min, reverse=True)
        return updates
    
    def get_metadata(self) -> KnowledgeBaseMetadata:
        """Get financial knowledge base metadata"""
        current_items = [item for item in self.items.values() if item.is_current()]
        
        return KnowledgeBaseMetadata(
            total_items=len(self.items),
            knowledge_types=[KnowledgeType.REGULATORY, KnowledgeType.MARKET_DATA, KnowledgeType.RISK_RULES],
            supported_formats=[KnowledgeFormat.JSON, KnowledgeFormat.TEXT],
            last_updated=datetime.utcnow(),
            version="1.0.0",
            description="Financial regulatory and market intelligence knowledge base",
            custom_metadata={
                "regulation_types": list(set(item.regulation_type for item in self.items.values() if item.regulation_type)),
                "asset_classes": list(set(item.asset_class for item in self.items.values() if item.asset_class)),
                "jurisdictions": list(set(j for item in self.items.values() for j in item.jurisdiction)),
                "current_items": len(current_items),
                "compliance_engine_version": "1.0"
            }
        )


# Plugin registration
def create_plugin(config: Optional[Dict[str, Any]] = None) -> FinancialKnowledgeBasePlugin:
    """Create and return the financial knowledge base plugin instance"""
    return FinancialKnowledgeBasePlugin(config)


# Export the plugin class
__all__ = ['FinancialKnowledgeBasePlugin', 'FinancialKnowledgeItem', 'create_plugin']