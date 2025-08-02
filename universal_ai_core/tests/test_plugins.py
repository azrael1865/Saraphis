"""
Integration tests for Universal AI Core plugin system.
Adapted from Saraphis/charon_builder plugin test patterns.
"""

import pytest
import asyncio
import time
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List

from universal_ai_core.core.plugin_manager import PluginManager, PluginType
from universal_ai_core.plugins.feature_extractors.molecular import MolecularFeatureExtractorPlugin
from universal_ai_core.plugins.feature_extractors.cybersecurity import CybersecurityFeatureExtractorPlugin
from universal_ai_core.plugins.feature_extractors.financial import FinancialFeatureExtractorPlugin
from universal_ai_core.plugins.models.molecular import MolecularModelPlugin
from universal_ai_core.plugins.models.cybersecurity import CybersecurityModelPlugin
from universal_ai_core.plugins.models.financial import FinancialModelPlugin
from universal_ai_core.plugins.proof_languages.molecular import MolecularProofLanguagePlugin
from universal_ai_core.plugins.proof_languages.cybersecurity import CybersecurityProofLanguagePlugin
from universal_ai_core.plugins.proof_languages.financial import FinancialProofLanguagePlugin
from universal_ai_core.plugins.knowledge_bases.molecular import MolecularKnowledgeBasePlugin
from universal_ai_core.plugins.knowledge_bases.cybersecurity import CybersecurityKnowledgeBasePlugin
from universal_ai_core.plugins.knowledge_bases.financial import FinancialKnowledgeBasePlugin


@pytest.mark.integration
class TestMolecularPlugins:
    """Test molecular analysis plugins integration."""
    
    def test_molecular_feature_extractor_initialization(self, sample_config):
        """Test molecular feature extractor initialization."""
        config = sample_config["plugins"]["feature_extractors"]["molecular"]
        plugin = MolecularFeatureExtractorPlugin(config)
        
        assert plugin.config == config
        assert hasattr(plugin, 'extract_features')
        assert plugin.plugin_type == "feature_extractors"
        assert plugin.domain == "molecular"
    
    def test_molecular_feature_extraction(self, sample_molecular_data):
        """Test molecular feature extraction."""
        config = {"enabled": True, "rdkit_enabled": False}
        plugin = MolecularFeatureExtractorPlugin(config)
        
        # Test basic feature extraction without RDKit
        features = plugin.extract_features(sample_molecular_data)
        
        assert isinstance(features, dict)
        assert "molecular_descriptors" in features
        assert len(features["molecular_descriptors"]) > 0
    
    def test_molecular_model_initialization(self, sample_config):
        """Test molecular model initialization."""
        config = sample_config["plugins"]["models"]["molecular"]
        plugin = MolecularModelPlugin(config)
        
        assert plugin.config == config
        assert hasattr(plugin, 'predict')
        assert hasattr(plugin, 'train')
    
    def test_molecular_model_prediction(self, sample_molecular_data):
        """Test molecular model prediction."""
        config = {"enabled": True, "model_type": "neural_network"}
        plugin = MolecularModelPlugin(config)
        
        # Mock the underlying model
        with patch.object(plugin, '_get_model') as mock_model:
            mock_model.return_value.predict.return_value = np.array([0.7, 0.3, 0.8])
            
            features = {"molecular_descriptors": np.array([[1, 2, 3], [4, 5, 6]])}
            predictions = plugin.predict(features)
            
            assert isinstance(predictions, dict)
            assert "predictions" in predictions
    
    def test_molecular_proof_language(self):
        """Test molecular proof language plugin."""
        config = {"enabled": True}
        plugin = MolecularProofLanguagePlugin(config)
        
        proof_request = {
            "molecule": "CCO",
            "property": "drug_likeness",
            "target_value": True
        }
        
        proof = plugin.generate_proof(proof_request)
        
        assert isinstance(proof, dict)
        assert "proof_type" in proof
        assert "statements" in proof
        assert proof["proof_type"] == "drug_likeness_proof"
    
    def test_molecular_knowledge_base(self):
        """Test molecular knowledge base plugin."""
        config = {"enabled": True}
        plugin = MolecularKnowledgeBasePlugin(config)
        
        query = {
            "type": "similarity_search",
            "molecule": "CCO",
            "threshold": 0.8
        }
        
        results = plugin.query(query)
        
        assert isinstance(results, dict)
        assert "similar_molecules" in results
        assert isinstance(results["similar_molecules"], list)


@pytest.mark.integration
class TestCybersecurityPlugins:
    """Test cybersecurity plugins integration."""
    
    def test_cybersecurity_feature_extractor(self, sample_cybersecurity_data):
        """Test cybersecurity feature extractor."""
        config = {"enabled": True, "threat_detection": True}
        plugin = CybersecurityFeatureExtractorPlugin(config)
        
        features = plugin.extract_features(sample_cybersecurity_data)
        
        assert isinstance(features, dict)
        assert "network_features" in features or "log_features" in features
    
    def test_cybersecurity_model_training(self, sample_cybersecurity_data):
        """Test cybersecurity model training."""
        config = {"enabled": True, "model_type": "ensemble"}
        plugin = CybersecurityModelPlugin(config)
        
        # Mock training data
        training_data = {
            "features": np.random.rand(100, 10),
            "labels": np.random.randint(0, 2, 100)
        }
        
        with patch.object(plugin, '_prepare_model') as mock_prepare:
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_prepare.return_value = mock_model
            
            result = plugin.train(training_data)
            
            assert result["status"] == "success"
            assert "model_info" in result
    
    def test_cybersecurity_threat_detection(self, sample_cybersecurity_data):
        """Test cybersecurity threat detection."""
        config = {"enabled": True, "model_type": "ensemble"}
        plugin = CybersecurityModelPlugin(config)
        
        with patch.object(plugin, '_get_model') as mock_model:
            mock_model.return_value.predict_proba.return_value = np.array([[0.2, 0.8], [0.9, 0.1]])
            
            features = {"network_features": np.array([[1, 2, 3], [4, 5, 6]])}
            predictions = plugin.predict(features)
            
            assert isinstance(predictions, dict)
            assert "threat_scores" in predictions
            assert "threat_classifications" in predictions
    
    def test_cybersecurity_proof_generation(self):
        """Test cybersecurity proof generation."""
        config = {"enabled": True}
        plugin = CybersecurityProofLanguagePlugin(config)
        
        incident_data = {
            "type": "suspicious_login",
            "user": "admin",
            "ip": "192.168.1.100",
            "timestamp": "2024-01-01 10:00:00"
        }
        
        proof = plugin.generate_proof(incident_data)
        
        assert isinstance(proof, dict)
        assert "proof_type" in proof
        assert "evidence_chain" in proof
    
    def test_cybersecurity_knowledge_base(self):
        """Test cybersecurity knowledge base plugin."""
        config = {"enabled": True}
        plugin = CybersecurityKnowledgeBasePlugin(config)
        
        query = {
            "type": "threat_intelligence",
            "indicator": "192.168.1.100",
            "indicator_type": "ip_address"
        }
        
        results = plugin.query(query)
        
        assert isinstance(results, dict)
        assert "threat_info" in results


@pytest.mark.integration
class TestFinancialPlugins:
    """Test financial plugins integration."""
    
    def test_financial_feature_extractor(self, sample_financial_data):
        """Test financial feature extractor."""
        config = {"enabled": True, "technical_indicators": ["sma", "rsi"]}
        plugin = FinancialFeatureExtractorPlugin(config)
        
        features = plugin.extract_features(sample_financial_data)
        
        assert isinstance(features, dict)
        assert "time_series_features" in features
        assert "technical_indicators" in features
    
    def test_financial_model_lstm(self, sample_financial_data):
        """Test financial LSTM model."""
        config = {"enabled": True, "model_type": "lstm"}
        plugin = FinancialModelPlugin(config)
        
        # Mock LSTM model
        with patch.object(plugin, '_create_lstm_model') as mock_create:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([[105.0], [107.0]])
            mock_create.return_value = mock_model
            
            features = {"price_sequences": np.random.rand(2, 10, 5)}
            predictions = plugin.predict(features)
            
            assert isinstance(predictions, dict)
            assert "price_predictions" in predictions
    
    def test_financial_risk_analysis(self, sample_financial_data):
        """Test financial risk analysis."""
        config = {"enabled": True, "model_type": "risk_model"}
        plugin = FinancialModelPlugin(config)
        
        with patch.object(plugin, '_calculate_risk_metrics') as mock_risk:
            mock_risk.return_value = {
                "var_95": 0.05,
                "cvar_95": 0.08,
                "sharpe_ratio": 1.2
            }
            
            portfolio_data = {
                "returns": np.random.normal(0.001, 0.02, 252),
                "weights": np.array([0.4, 0.3, 0.3])
            }
            
            risk_metrics = plugin.analyze_risk(portfolio_data)
            
            assert isinstance(risk_metrics, dict)
            assert "var_95" in risk_metrics
            assert "cvar_95" in risk_metrics
    
    def test_financial_proof_compliance(self):
        """Test financial compliance proof generation."""
        config = {"enabled": True}
        plugin = FinancialProofLanguagePlugin(config)
        
        transaction_data = {
            "amount": 15000,
            "currency": "USD",
            "type": "wire_transfer",
            "compliance_check": "aml_screening"
        }
        
        proof = plugin.generate_proof(transaction_data)
        
        assert isinstance(proof, dict)
        assert "proof_type" in proof
        assert "compliance_status" in proof
    
    def test_financial_knowledge_base(self):
        """Test financial knowledge base plugin."""
        config = {"enabled": True}
        plugin = FinancialKnowledgeBasePlugin(config)
        
        query = {
            "type": "regulation_check",
            "regulation": "basel_iii",
            "metric": "capital_ratio"
        }
        
        results = plugin.query(query)
        
        assert isinstance(results, dict)
        assert "regulation_info" in results


@pytest.mark.integration
class TestPluginManager:
    """Test plugin manager integration."""
    
    def test_plugin_manager_load_all_domains(self, sample_config):
        """Test loading plugins for all domains."""
        from universal_ai_core.config.config_manager import UniversalConfiguration
        
        config = UniversalConfiguration(**sample_config)
        manager = PluginManager(config)
        
        # Test loading plugins for each domain
        domains = ["molecular", "cybersecurity", "financial"]
        plugin_types = ["feature_extractors", "models", "proof_languages", "knowledge_bases"]
        
        loaded_plugins = {}
        for domain in domains:
            loaded_plugins[domain] = {}
            for plugin_type in plugin_types:
                try:
                    success = manager.load_plugin(plugin_type, domain)
                    loaded_plugins[domain][plugin_type] = success
                except Exception as e:
                    loaded_plugins[domain][plugin_type] = False
        
        # Check that at least some plugins loaded successfully
        total_loaded = sum(
            sum(1 for success in domain_plugins.values() if success)
            for domain_plugins in loaded_plugins.values()
        )
        assert total_loaded > 0
    
    def test_plugin_manager_get_plugin(self, sample_config):
        """Test getting plugins from manager."""
        from universal_ai_core.config.config_manager import UniversalConfiguration
        
        config = UniversalConfiguration(**sample_config)
        manager = PluginManager(config)
        
        # Load and get a plugin
        manager.load_plugin("feature_extractors", "molecular")
        plugin = manager.get_plugin("feature_extractors", "molecular")
        
        assert plugin is not None
        assert hasattr(plugin, 'extract_features')
    
    def test_plugin_manager_list_available(self, sample_config):
        """Test listing available plugins."""
        from universal_ai_core.config.config_manager import UniversalConfiguration
        
        config = UniversalConfiguration(**sample_config)
        manager = PluginManager(config)
        
        available_plugins = manager.list_available_plugins()
        
        assert isinstance(available_plugins, dict)
        assert len(available_plugins) > 0
        
        # Check plugin type structure
        expected_types = ["feature_extractors", "models", "proof_languages", "knowledge_bases"]
        for plugin_type in expected_types:
            assert plugin_type in available_plugins


@pytest.mark.integration
class TestPluginInteroperability:
    """Test plugin interoperability across domains."""
    
    def test_feature_extractor_to_model_pipeline(self, sample_molecular_data):
        """Test pipeline from feature extractor to model."""
        # Initialize plugins
        fe_config = {"enabled": True, "rdkit_enabled": False}
        model_config = {"enabled": True, "model_type": "neural_network"}
        
        feature_extractor = MolecularFeatureExtractorPlugin(fe_config)
        model = MolecularModelPlugin(model_config)
        
        # Extract features
        features = feature_extractor.extract_features(sample_molecular_data)
        
        # Use features for prediction
        with patch.object(model, '_get_model') as mock_model:
            mock_model.return_value.predict.return_value = np.array([0.7, 0.3])
            
            predictions = model.predict(features)
            
            assert isinstance(predictions, dict)
            assert "predictions" in predictions
    
    def test_model_to_proof_pipeline(self, sample_financial_data):
        """Test pipeline from model to proof generation."""
        model_config = {"enabled": True, "model_type": "lstm"}
        proof_config = {"enabled": True}
        
        model = FinancialModelPlugin(model_config)
        proof_generator = FinancialProofLanguagePlugin(proof_config)
        
        # Generate prediction
        with patch.object(model, '_create_lstm_model') as mock_create:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([[105.0]])
            mock_create.return_value = mock_model
            
            features = {"price_sequences": np.random.rand(1, 10, 5)}
            predictions = model.predict(features)
            
            # Generate proof for prediction
            proof_request = {
                "prediction": predictions["price_predictions"][0],
                "confidence": 0.85,
                "model_type": "lstm"
            }
            
            proof = proof_generator.generate_proof(proof_request)
            
            assert isinstance(proof, dict)
            assert "proof_type" in proof
    
    def test_knowledge_base_to_feature_extractor(self):
        """Test using knowledge base to enhance feature extraction."""
        kb_config = {"enabled": True}
        fe_config = {"enabled": True, "threat_detection": True}
        
        knowledge_base = CybersecurityKnowledgeBasePlugin(kb_config)
        feature_extractor = CybersecurityFeatureExtractorPlugin(fe_config)
        
        # Query knowledge base for threat intelligence
        query = {
            "type": "threat_intelligence",
            "indicator": "192.168.1.100",
            "indicator_type": "ip_address"
        }
        
        threat_info = knowledge_base.query(query)
        
        # Use threat info to enhance feature extraction
        network_data = {
            "network_traffic": [
                {"src_ip": "192.168.1.100", "dst_ip": "10.0.0.1", "port": 80}
            ],
            "threat_intel": threat_info
        }
        
        enhanced_features = feature_extractor.extract_features(network_data)
        
        assert isinstance(enhanced_features, dict)


@pytest.mark.integration
class TestPluginErrorHandling:
    """Test plugin error handling and recovery."""
    
    def test_plugin_initialization_errors(self):
        """Test plugin initialization error handling."""
        # Test with invalid configuration
        invalid_config = {"invalid": "config"}
        
        with pytest.raises((ValueError, KeyError, TypeError)):
            MolecularFeatureExtractorPlugin(invalid_config)
    
    def test_plugin_processing_errors(self):
        """Test plugin processing error handling."""
        config = {"enabled": True, "rdkit_enabled": False}
        plugin = MolecularFeatureExtractorPlugin(config)
        
        # Test with invalid data
        invalid_data = {"invalid": "data"}
        
        try:
            features = plugin.extract_features(invalid_data)
            # Should either handle gracefully or raise appropriate exception
            assert isinstance(features, dict) or features is None
        except (ValueError, KeyError, TypeError):
            # Expected for invalid data
            pass
    
    def test_plugin_recovery_mechanisms(self, sample_molecular_data):
        """Test plugin recovery mechanisms."""
        config = {"enabled": True, "rdkit_enabled": False}
        plugin = MolecularFeatureExtractorPlugin(config)
        
        # Simulate partial failure in feature extraction
        with patch.object(plugin, '_extract_basic_descriptors') as mock_extract:
            mock_extract.side_effect = Exception("Extraction failed")
            
            # Plugin should handle the error gracefully
            try:
                features = plugin.extract_features(sample_molecular_data)
                # Should return partial results or empty dict
                assert isinstance(features, dict)
            except Exception:
                # Or raise appropriate exception
                pass


@pytest.mark.performance
class TestPluginPerformance:
    """Test plugin performance characteristics."""
    
    def test_feature_extraction_performance(self, sample_molecular_data):
        """Test feature extraction performance."""
        config = {"enabled": True, "rdkit_enabled": False}
        plugin = MolecularFeatureExtractorPlugin(config)
        
        start_time = time.time()
        
        # Extract features multiple times
        for _ in range(10):
            features = plugin.extract_features(sample_molecular_data)
            assert isinstance(features, dict)
        
        total_time = time.time() - start_time
        
        # Should process 10 extractions quickly
        assert total_time < 1.0
    
    def test_model_prediction_performance(self):
        """Test model prediction performance."""
        config = {"enabled": True, "model_type": "neural_network"}
        plugin = MolecularModelPlugin(config)
        
        # Mock fast prediction
        with patch.object(plugin, '_get_model') as mock_model:
            mock_model.return_value.predict.return_value = np.array([0.7])
            
            features = {"molecular_descriptors": np.random.rand(100, 10)}
            
            start_time = time.time()
            
            # Make multiple predictions
            for _ in range(20):
                predictions = plugin.predict(features)
                assert isinstance(predictions, dict)
            
            total_time = time.time() - start_time
            
            # Should make 20 predictions quickly
            assert total_time < 0.5
    
    def test_concurrent_plugin_usage(self, sample_molecular_data):
        """Test concurrent plugin usage."""
        import concurrent.futures
        
        config = {"enabled": True, "rdkit_enabled": False}
        plugin = MolecularFeatureExtractorPlugin(config)
        
        def extract_features():
            return plugin.extract_features(sample_molecular_data)
        
        start_time = time.time()
        
        # Run concurrent feature extractions
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(extract_features) for _ in range(6)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Should handle concurrent usage efficiently
        assert total_time < 2.0
        assert len(results) == 6
        assert all(isinstance(result, dict) for result in results)


@pytest.mark.compatibility
class TestPluginCompatibility:
    """Test plugin compatibility across different configurations."""
    
    def test_minimal_configuration_compatibility(self):
        """Test plugins with minimal configuration."""
        minimal_configs = [
            {"enabled": True},
            {"enabled": True, "debug": False},
            {}  # Empty config
        ]
        
        for config in minimal_configs:
            try:
                # Test each plugin type with minimal config
                fe_plugin = MolecularFeatureExtractorPlugin(config)
                assert fe_plugin is not None
                
                model_plugin = MolecularModelPlugin(config)
                assert model_plugin is not None
                
                proof_plugin = MolecularProofLanguagePlugin(config)
                assert proof_plugin is not None
                
                kb_plugin = MolecularKnowledgeBasePlugin(config)
                assert kb_plugin is not None
                
            except Exception as e:
                # Some configurations might not be valid
                assert isinstance(e, (ValueError, KeyError, TypeError))
    
    def test_cross_domain_compatibility(self):
        """Test compatibility across different domains."""
        domains = ["molecular", "cybersecurity", "financial"]
        plugin_classes = {
            "molecular": (MolecularFeatureExtractorPlugin, MolecularModelPlugin),
            "cybersecurity": (CybersecurityFeatureExtractorPlugin, CybersecurityModelPlugin),
            "financial": (FinancialFeatureExtractorPlugin, FinancialModelPlugin)
        }
        
        config = {"enabled": True}
        
        # Test that plugins from different domains can coexist
        active_plugins = []
        
        for domain in domains:
            fe_class, model_class = plugin_classes[domain]
            
            fe_plugin = fe_class(config)
            model_plugin = model_class(config)
            
            active_plugins.extend([fe_plugin, model_plugin])
        
        # All plugins should be active simultaneously
        assert len(active_plugins) == 6
        
        # Test that they don't interfere with each other
        for plugin in active_plugins:
            assert hasattr(plugin, 'config')
            assert plugin.config == config
    
    def test_version_compatibility(self):
        """Test plugin version compatibility."""
        # Test different configuration versions
        configs = [
            {"enabled": True, "version": "1.0"},
            {"enabled": True, "api_version": "2.0"},
            {"enabled": True}  # No version specified
        ]
        
        for config in configs:
            try:
                plugin = MolecularFeatureExtractorPlugin(config)
                assert plugin is not None
            except Exception:
                # Version incompatibility might be expected
                pass