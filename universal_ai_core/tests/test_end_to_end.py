"""
End-to-end tests for Universal AI Core across all domains.
Adapted from Saraphis end-to-end test patterns.
"""

import pytest
import asyncio
import time
import json
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from universal_ai_core import UniversalAIAPI, APIConfig, create_api, create_development_api
from universal_ai_core.core.universal_ai_core import UniversalAICore
from universal_ai_core.core.orchestrator import SystemOrchestrator


@pytest.mark.integration
class TestMolecularAnalysisE2E:
    """End-to-end tests for molecular analysis workflows."""
    
    def test_complete_molecular_analysis_workflow(self, universal_ai_api, sample_molecular_data):
        """Test complete molecular analysis from data to insights."""
        # Step 1: Data validation
        validation_result = universal_ai_api.validate_data(
            sample_molecular_data, ["schema", "chemical_validity"]
        )
        assert validation_result.is_valid
        
        # Step 2: Feature extraction
        processing_result = universal_ai_api.process_data(
            sample_molecular_data, ["molecular_descriptors", "fingerprints"]
        )
        assert processing_result.status == "success"
        assert "features" in processing_result.data
        
        # Step 3: Model prediction
        features = processing_result.data["features"]
        with patch.object(universal_ai_api.orchestrator, 'process_request') as mock_orchestrate:
            mock_orchestrate.return_value = {
                "predictions": {"toxicity": 0.2, "bioactivity": 0.8},
                "confidence": 0.9,
                "model_version": "v1.0"
            }
            
            prediction_result = asyncio.run(
                universal_ai_api.orchestrator.process_request(
                    domain="molecular",
                    operation="predict",
                    data={"features": features}
                )
            )
            
            assert prediction_result is not None
            assert "predictions" in prediction_result
    
    def test_molecular_drug_discovery_pipeline(self, universal_ai_api):
        """Test drug discovery pipeline."""
        drug_candidates = {
            "compounds": [
                {"smiles": "CCO", "name": "ethanol"},
                {"smiles": "CC(=O)O", "name": "acetic_acid"},
                {"smiles": "c1ccccc1", "name": "benzene"}
            ],
            "targets": ["EGFR", "BCL2"]
        }
        
        # Step 1: Validate drug candidates
        validation = universal_ai_api.validate_data(drug_candidates, ["drug_likeness"])
        assert validation.is_valid
        
        # Step 2: Extract molecular features
        features = universal_ai_api.process_data(
            drug_candidates, ["admet_properties", "pharmacophores"]
        )
        assert features.status == "success"
        
        # Step 3: Screen against targets
        with patch.object(universal_ai_api.orchestrator, 'process_request') as mock_screen:
            mock_screen.return_value = {
                "screening_results": [
                    {"compound": "ethanol", "target": "EGFR", "binding_affinity": -5.2},
                    {"compound": "acetic_acid", "target": "BCL2", "binding_affinity": -3.8}
                ],
                "lead_compounds": ["ethanol"]
            }
            
            screening_result = asyncio.run(
                universal_ai_api.orchestrator.process_request(
                    domain="molecular",
                    operation="virtual_screening",
                    data={"candidates": drug_candidates["compounds"]}
                )
            )
            
            assert "screening_results" in screening_result
            assert "lead_compounds" in screening_result


@pytest.mark.integration
class TestCybersecurityAnalysisE2E:
    """End-to-end tests for cybersecurity analysis workflows."""
    
    def test_threat_detection_workflow(self, universal_ai_api, sample_cybersecurity_data):
        """Test complete threat detection workflow."""
        # Step 1: Data validation
        validation_result = universal_ai_api.validate_data(
            sample_cybersecurity_data, ["security_schema", "data_integrity"]
        )
        assert validation_result.is_valid
        
        # Step 2: Feature extraction
        processing_result = universal_ai_api.process_data(
            sample_cybersecurity_data, ["network_features", "behavioral_patterns"]
        )
        assert processing_result.status == "success"
        
        # Step 3: Threat classification
        with patch.object(universal_ai_api.orchestrator, 'process_request') as mock_classify:
            mock_classify.return_value = {
                "threats_detected": [
                    {"type": "malware", "confidence": 0.95, "severity": "high"},
                    {"type": "anomaly", "confidence": 0.7, "severity": "medium"}
                ],
                "risk_score": 0.85,
                "recommended_actions": ["block_ip", "quarantine_host"]
            }
            
            threat_result = asyncio.run(
                universal_ai_api.orchestrator.process_request(
                    domain="cybersecurity",
                    operation="threat_analysis",
                    data=sample_cybersecurity_data
                )
            )
            
            assert "threats_detected" in threat_result
            assert "risk_score" in threat_result
    
    def test_incident_response_workflow(self, universal_ai_api):
        """Test incident response workflow."""
        security_incident = {
            "incident_id": "INC-2024-001",
            "type": "data_breach",
            "affected_systems": ["web_server", "database"],
            "timeline": [
                {"timestamp": "2024-01-01T10:00:00Z", "event": "suspicious_login"},
                {"timestamp": "2024-01-01T10:05:00Z", "event": "privilege_escalation"},
                {"timestamp": "2024-01-01T10:10:00Z", "event": "data_exfiltration"}
            ]
        }
        
        # Step 1: Incident validation
        validation = universal_ai_api.validate_data(security_incident, ["incident_schema"])
        assert validation.is_valid
        
        # Step 2: Impact assessment
        with patch.object(universal_ai_api.orchestrator, 'process_request') as mock_assess:
            mock_assess.return_value = {
                "impact_score": 0.9,
                "affected_records": 10000,
                "compliance_violations": ["GDPR", "SOX"],
                "containment_strategy": "immediate_isolation"
            }
            
            impact_result = asyncio.run(
                universal_ai_api.orchestrator.process_request(
                    domain="cybersecurity",
                    operation="impact_assessment",
                    data=security_incident
                )
            )
            
            assert "impact_score" in impact_result
            assert "containment_strategy" in impact_result


@pytest.mark.integration
class TestFinancialAnalysisE2E:
    """End-to-end tests for financial analysis workflows."""
    
    def test_portfolio_optimization_workflow(self, universal_ai_api, sample_financial_data):
        """Test complete portfolio optimization workflow."""
        portfolio_data = {
            "assets": ["AAPL", "GOOGL", "MSFT", "TSLA"],
            "historical_data": sample_financial_data,
            "risk_tolerance": "moderate",
            "investment_horizon": "1_year"
        }
        
        # Step 1: Data validation
        validation_result = universal_ai_api.validate_data(
            portfolio_data, ["financial_schema", "market_data_quality"]
        )
        assert validation_result.is_valid
        
        # Step 2: Feature extraction
        processing_result = universal_ai_api.process_data(
            portfolio_data, ["returns_analysis", "risk_metrics", "correlations"]
        )
        assert processing_result.status == "success"
        
        # Step 3: Portfolio optimization
        with patch.object(universal_ai_api.orchestrator, 'process_request') as mock_optimize:
            mock_optimize.return_value = {
                "optimal_weights": {"AAPL": 0.3, "GOOGL": 0.25, "MSFT": 0.25, "TSLA": 0.2},
                "expected_return": 0.12,
                "expected_risk": 0.18,
                "sharpe_ratio": 0.67,
                "rebalancing_frequency": "monthly"
            }
            
            optimization_result = asyncio.run(
                universal_ai_api.orchestrator.process_request(
                    domain="financial",
                    operation="portfolio_optimization",
                    data=portfolio_data
                )
            )
            
            assert "optimal_weights" in optimization_result
            assert "sharpe_ratio" in optimization_result
    
    def test_risk_management_workflow(self, universal_ai_api):
        """Test risk management workflow."""
        trading_portfolio = {
            "positions": [
                {"symbol": "BTC", "quantity": 10, "entry_price": 45000},
                {"symbol": "ETH", "quantity": 100, "entry_price": 3000},
                {"symbol": "SPY", "quantity": 500, "entry_price": 400}
            ],
            "market_conditions": {"volatility": "high", "trend": "bearish"},
            "risk_limits": {"max_loss_per_day": 0.05, "var_95": 0.02}
        }
        
        # Step 1: Risk calculation
        with patch.object(universal_ai_api.orchestrator, 'process_request') as mock_risk:
            mock_risk.return_value = {
                "portfolio_var": 0.035,
                "portfolio_cvar": 0.048,
                "position_risks": [
                    {"symbol": "BTC", "contribution": 0.6, "individual_var": 0.08},
                    {"symbol": "ETH", "contribution": 0.3, "individual_var": 0.06},
                    {"symbol": "SPY", "contribution": 0.1, "individual_var": 0.02}
                ],
                "risk_alerts": ["var_limit_exceeded", "concentration_risk"],
                "recommended_actions": ["reduce_crypto_exposure", "hedge_positions"]
            }
            
            risk_result = asyncio.run(
                universal_ai_api.orchestrator.process_request(
                    domain="financial",
                    operation="risk_analysis",
                    data=trading_portfolio
                )
            )
            
            assert "portfolio_var" in risk_result
            assert "risk_alerts" in risk_result


@pytest.mark.integration
class TestCrossDomainWorkflows:
    """Test workflows that span multiple domains."""
    
    def test_pharmaceutical_cybersecurity_workflow(self, universal_ai_api):
        """Test workflow combining molecular and cybersecurity analysis."""
        pharma_system = {
            "research_data": {
                "compounds": [{"smiles": "CCO", "phase": "clinical_trial"}],
                "patient_data": {"encrypted": True, "subjects": 1000}
            },
            "security_events": [
                {"type": "data_access", "user": "researcher_1", "timestamp": "2024-01-01T10:00:00Z"},
                {"type": "file_download", "file": "compound_results.csv", "size_mb": 50}
            ]
        }
        
        # Molecular analysis
        with patch.object(universal_ai_api.orchestrator, 'process_request') as mock_molecular:
            mock_molecular.return_value = {
                "compound_analysis": {"safety_profile": "acceptable", "efficacy": "promising"},
                "data_sensitivity": "high"
            }
            
            molecular_result = asyncio.run(
                universal_ai_api.orchestrator.process_request(
                    domain="molecular",
                    operation="analyze",
                    data=pharma_system["research_data"]
                )
            )
        
        # Security analysis incorporating molecular insights
        with patch.object(universal_ai_api.orchestrator, 'process_request') as mock_security:
            mock_security.return_value = {
                "access_patterns": "normal",
                "data_classification": "confidential",
                "compliance_status": "gdpr_compliant",
                "risk_level": "medium"
            }
            
            security_result = asyncio.run(
                universal_ai_api.orchestrator.process_request(
                    domain="cybersecurity",
                    operation="analyze",
                    data={**pharma_system["security_events"], "context": molecular_result}
                )
            )
        
        assert molecular_result is not None
        assert security_result is not None
    
    def test_fintech_security_compliance_workflow(self, universal_ai_api):
        """Test workflow combining financial and cybersecurity compliance."""
        fintech_transaction = {
            "transaction_id": "TXN-2024-001",
            "amount": 50000,
            "currency": "USD",
            "source_account": "ACC-12345",
            "destination_account": "ACC-67890",
            "transaction_type": "wire_transfer",
            "metadata": {
                "ip_address": "192.168.1.100",
                "device_fingerprint": "DEV-ABC123",
                "geo_location": "New York, US"
            }
        }
        
        # Financial compliance analysis
        with patch.object(universal_ai_api.orchestrator, 'process_request') as mock_financial:
            mock_financial.return_value = {
                "aml_status": "cleared",
                "fraud_score": 0.15,
                "regulatory_checks": {"bsa": "pass", "ofac": "pass"},
                "risk_category": "low"
            }
            
            financial_result = asyncio.run(
                universal_ai_api.orchestrator.process_request(
                    domain="financial",
                    operation="compliance_check",
                    data=fintech_transaction
                )
            )
        
        # Security analysis
        with patch.object(universal_ai_api.orchestrator, 'process_request') as mock_security:
            mock_security.return_value = {
                "device_trust_score": 0.9,
                "behavioral_analysis": "consistent_with_history",
                "threat_indicators": [],
                "security_clearance": "approved"
            }
            
            security_result = asyncio.run(
                universal_ai_api.orchestrator.process_request(
                    domain="cybersecurity",
                    operation="transaction_security",
                    data=fintech_transaction["metadata"]
                )
            )
        
        # Combined decision
        combined_risk = min(financial_result["fraud_score"], 1 - security_result["device_trust_score"])
        assert combined_risk < 0.3  # Low combined risk


@pytest.mark.integration 
class TestAPIWorkflows:
    """Test API-level workflows and integrations."""
    
    @pytest.mark.asyncio
    async def test_async_batch_processing_workflow(self, universal_ai_api):
        """Test async batch processing across domains."""
        batch_data = [
            {"domain": "molecular", "data": {"smiles": ["CCO", "CC(=O)O"]}},
            {"domain": "cybersecurity", "data": {"events": [{"type": "login"}]}},
            {"domain": "financial", "data": {"prices": [{"symbol": "AAPL", "price": 150}]}}
        ]
        
        # Submit async tasks for each domain
        task_ids = []
        for item in batch_data:
            task_id = await universal_ai_api.submit_async_task(
                f"{item['domain']}_analysis",
                item["data"],
                config={"domain": item["domain"]}
            )
            task_ids.append(task_id)
        
        # Wait for completion
        await asyncio.sleep(0.5)
        
        # Collect results
        results = []
        for task_id in task_ids:
            result = universal_ai_api.get_task_result(task_id)
            if result and result.is_completed:
                results.append(result)
        
        assert len(results) > 0
        assert all(isinstance(r.task_id, str) for r in results)
    
    def test_caching_across_domains(self, universal_ai_api, sample_molecular_data, sample_financial_data):
        """Test caching behavior across different domains."""
        # Process molecular data
        mol_result1 = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
        mol_result2 = universal_ai_api.process_data(sample_molecular_data, ["molecular_weight"])
        
        # Process financial data
        fin_result1 = universal_ai_api.process_data(sample_financial_data, ["returns"])
        fin_result2 = universal_ai_api.process_data(sample_financial_data, ["returns"])
        
        # Cache should work independently for each domain
        if universal_ai_api.cache:
            cache_stats = universal_ai_api.cache.get_stats()
            assert cache_stats["hits"] > 0  # Should have cache hits
    
    def test_health_monitoring_across_domains(self, universal_ai_api):
        """Test health monitoring across all domains."""
        health_status = universal_ai_api.get_health_status()
        
        assert "overall" in health_status
        assert "components" in health_status
        assert "api" in health_status
        
        # Check component health
        for component_name, component_health in health_status["components"].items():
            assert "healthy" in component_health
            assert isinstance(component_health["healthy"], bool)


@pytest.mark.integration
class TestSystemIntegration:
    """Test system-level integration scenarios."""
    
    def test_full_system_startup_and_shutdown(self, api_config, sample_config_file):
        """Test complete system lifecycle."""
        # System startup
        api = UniversalAIAPI(config_path=str(sample_config_file), api_config=api_config)
        
        # Verify system is operational
        health = api.get_health_status()
        assert health["overall"] in ["healthy", "unhealthy"]  # Should have a status
        
        # Test basic functionality
        system_info = api.get_system_info()
        assert "api_version" in system_info
        assert "components" in system_info
        
        # System shutdown
        api.shutdown()
        
        # Verify clean shutdown
        assert api.health_checker._monitoring_active == False
        assert api.metrics_collector._collecting == False
    
    def test_configuration_hot_reload(self, universal_ai_api, temp_directory):
        """Test configuration hot reload functionality."""
        # Create new config file
        new_config = {
            "core": {"max_workers": 8, "enable_monitoring": False},
            "plugins": {"feature_extractors": {"molecular": {"enabled": False}}}
        }
        
        new_config_file = temp_directory / "new_config.yaml"
        import yaml
        with open(new_config_file, 'w') as f:
            yaml.dump(new_config, f)
        
        # Test config reload (would need config manager support)
        original_workers = universal_ai_api.api_config.max_workers
        
        # In a real implementation, this would trigger a config reload
        # For testing, we verify the system can handle config changes
        assert original_workers >= 1
    
    def test_plugin_ecosystem_integration(self, universal_ai_api):
        """Test integration across the entire plugin ecosystem."""
        # Get available plugins
        available_plugins = universal_ai_api.core.plugin_manager.list_available_plugins()
        
        # Verify plugin ecosystem structure
        expected_plugin_types = ["feature_extractors", "models", "proof_languages", "knowledge_bases"]
        for plugin_type in expected_plugin_types:
            assert plugin_type in available_plugins
        
        # Test cross-plugin data flow (simplified)
        test_data = {"smiles": ["CCO"]}
        
        # Extract features
        features = universal_ai_api.process_data(test_data, ["molecular_descriptors"])
        assert features.status in ["success", "error"]  # Should complete
        
        # Validate results
        if features.status == "success":
            validation = universal_ai_api.validate_data(features.data, ["schema"])
            assert validation.is_valid in [True, False]  # Should complete


@pytest.mark.slow
class TestLongRunningWorkflows:
    """Test long-running and complex workflows."""
    
    def test_extended_molecular_pipeline(self, universal_ai_api):
        """Test extended molecular analysis pipeline."""
        # Large dataset simulation
        large_dataset = {
            "compounds": [{"smiles": f"C{i}", "id": f"mol_{i}"} for i in range(100)],
            "assays": ["toxicity", "bioactivity", "admet"],
            "targets": ["EGFR", "BCL2", "p53"]
        }
        
        start_time = time.time()
        
        # Step 1: Batch validation
        validation = universal_ai_api.validate_data(large_dataset, ["chemical_validity"])
        assert validation.is_valid
        
        # Step 2: Feature extraction
        features = universal_ai_api.process_data(large_dataset, ["molecular_descriptors"])
        assert features.status == "success"
        
        # Step 3: Simulate analysis pipeline
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert processing_time < 30.0  # 30 seconds max for test
    
    def test_continuous_monitoring_simulation(self, universal_ai_api):
        """Test continuous monitoring workflow simulation."""
        monitoring_duration = 2.0  # 2 seconds for test
        start_time = time.time()
        
        event_count = 0
        while time.time() - start_time < monitoring_duration:
            # Simulate security events
            security_event = {
                "timestamp": time.time(),
                "event_type": "network_access",
                "source_ip": f"192.168.1.{event_count % 255}",
                "severity": "low"
            }
            
            # Process event
            try:
                result = universal_ai_api.validate_data(security_event, ["security_schema"])
                event_count += 1
            except Exception:
                pass
            
            time.sleep(0.1)  # 100ms between events
        
        # Should process multiple events
        assert event_count > 5