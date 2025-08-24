"""
Comprehensive test suite for ProofIntegrationManager
Tests all 22 methods and various edge cases
"""

import pytest
import time
import threading
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, call
from concurrent.futures import Future

from proof_system.proof_integration_manager import ProofIntegrationManager


class TestProofIntegrationManagerInitialization:
    """Test ProofIntegrationManager initialization"""
    
    def test_default_initialization(self):
        """Test default initialization of ProofIntegrationManager"""
        manager = ProofIntegrationManager()
        
        assert manager.engines == {}
        assert manager.confidence_generator is not None
        assert manager.algebraic_enforcer is not None
        assert manager.event_handlers == {}
        assert manager.event_log == []
        assert manager.performance_stats['total_proofs'] == 0
        assert manager.performance_stats['generation_times'] == []
        assert manager.performance_stats['engine_performance'] == {}
        assert manager.monitoring_enabled == False
        assert manager.timeout == 30
        assert manager.max_retries == 3
    
    def test_logger_initialization(self):
        """Test logger is properly initialized"""
        manager = ProofIntegrationManager()
        
        assert manager.logger is not None
        assert manager.logger.name == 'ProofIntegrationManager'


class TestEngineRegistration:
    """Test engine registration functionality"""
    
    def test_register_engine(self):
        """Test registering a new engine"""
        manager = ProofIntegrationManager()
        mock_engine = Mock()
        
        manager.register_engine('test_engine', mock_engine)
        
        assert 'test_engine' in manager.engines
        assert manager.engines['test_engine'] is mock_engine
        assert 'test_engine' in manager.performance_stats['engine_performance']
        
        perf_stats = manager.performance_stats['engine_performance']['test_engine']
        assert perf_stats['calls'] == 0
        assert perf_stats['total_time'] == 0
        assert perf_stats['avg_time'] == 0
        assert perf_stats['errors'] == 0
    
    def test_register_multiple_engines(self):
        """Test registering multiple engines"""
        manager = ProofIntegrationManager()
        
        engine1 = Mock()
        engine2 = Mock()
        engine3 = Mock()
        
        manager.register_engine('engine1', engine1)
        manager.register_engine('engine2', engine2)
        manager.register_engine('engine3', engine3)
        
        assert len(manager.engines) == 3
        assert manager.engines['engine1'] is engine1
        assert manager.engines['engine2'] is engine2
        assert manager.engines['engine3'] is engine3
    
    def test_get_registered_engines(self):
        """Test getting list of registered engines"""
        manager = ProofIntegrationManager()
        
        manager.register_engine('engine_a', Mock())
        manager.register_engine('engine_b', Mock())
        
        engines = manager.get_registered_engines()
        
        assert len(engines) == 2
        assert 'engine_a' in engines
        assert 'engine_b' in engines


class TestComprehensiveProofGeneration:
    """Test comprehensive proof generation"""
    
    def setup_method(self):
        """Setup for each test"""
        self.manager = ProofIntegrationManager()
        
        # Create mock engines
        self.rule_engine = Mock()
        self.rule_engine.evaluate_transaction.return_value = {
            'risk_score': 0.7,
            'confidence': 0.8
        }
        
        self.ml_engine = Mock()
        self.ml_engine.generate_ml_proof.return_value = {
            'ml_analysis': {
                'model_prediction': 0.6,
                'confidence_score': 0.75
            }
        }
        
        self.crypto_engine = Mock()
        self.crypto_engine.generate_proof_for_transaction.return_value = {
            'basic_proof': {
                'hash': 'test_hash_123'
            }
        }
        
        # Register engines
        self.manager.register_engine('rule_based', self.rule_engine)
        self.manager.register_engine('ml_based', self.ml_engine)
        self.manager.register_engine('cryptographic', self.crypto_engine)
    
    def test_generate_comprehensive_proof_all_engines(self):
        """Test generating comprehensive proof with all engines"""
        transaction = {'transaction_id': 'tx_123', 'amount': 1000}
        model_state = {'iteration': 10}
        
        proof = self.manager.generate_comprehensive_proof(transaction, model_state)
        
        assert proof['transaction_id'] == 'tx_123'
        assert 'generation_timestamp' in proof
        assert proof['engines_used'] == ['rule_based', 'ml_based', 'cryptographic']
        assert 'confidence' in proof
        assert 'confidence_details' in proof
        assert 'generation_time_ms' in proof
        
        # Check engines were called
        self.rule_engine.evaluate_transaction.assert_called_once_with(transaction)
        self.ml_engine.generate_ml_proof.assert_called_once_with(transaction, model_state)
        self.crypto_engine.generate_proof_for_transaction.assert_called_once_with(transaction)
    
    def test_generate_comprehensive_proof_with_engine_failure(self):
        """Test proof generation when one engine fails"""
        self.ml_engine.generate_ml_proof.side_effect = Exception("ML engine error")
        
        transaction = {'transaction_id': 'tx_456'}
        model_state = {}
        
        proof = self.manager.generate_comprehensive_proof(transaction, model_state)
        
        assert 'errors' in proof
        assert 'ml_based' in proof['errors']
        assert 'ML engine error' in proof['errors']['ml_based']
        assert 'rule_based' in proof['engines_used']
        assert 'cryptographic' in proof['engines_used']
        assert 'ml_based' not in proof['engines_used']
    
    def test_generate_comprehensive_proof_with_generic_engine(self):
        """Test with engine having generate_proof method"""
        generic_engine = Mock()
        generic_engine.generate_proof.return_value = {'generic_result': True}
        
        self.manager.register_engine('generic', generic_engine)
        
        transaction = {'transaction_id': 'tx_789'}
        model_state = {}
        
        proof = self.manager.generate_comprehensive_proof(transaction, model_state)
        
        assert 'generic' in proof['engines_used']
        generic_engine.generate_proof.assert_called_once_with(transaction, model_state)
    
    def test_generate_comprehensive_proof_no_recognized_method(self):
        """Test engine with no recognized proof generation method"""
        bad_engine = Mock(spec=[])  # No methods
        self.manager.register_engine('bad_engine', bad_engine)
        
        transaction = {'transaction_id': 'tx_bad'}
        model_state = {}
        
        proof = self.manager.generate_comprehensive_proof(transaction, model_state)
        
        assert 'errors' in proof
        assert 'bad_engine' in proof['errors']
        assert 'no recognized proof generation method' in proof['errors']['bad_engine']


class TestBatchProofGeneration:
    """Test batch proof generation"""
    
    def setup_method(self):
        """Setup for each test"""
        self.manager = ProofIntegrationManager()
        
        # Create simple mock engine
        mock_engine = Mock()
        mock_engine.generate_proof.return_value = {'proof': True}
        self.manager.register_engine('test_engine', mock_engine)
    
    def test_generate_batch_proofs_sync(self):
        """Test synchronous batch proof generation"""
        transactions = [
            {'transaction_id': 'tx_1'},
            {'transaction_id': 'tx_2'},
            {'transaction_id': 'tx_3'}
        ]
        model_state = {}
        
        results = self.manager.generate_batch_proofs(transactions, model_state, async_mode=False)
        
        assert len(results) == 3
        assert all('transaction_id' in r for r in results)
        assert results[0]['transaction_id'] == 'tx_1'
        assert results[1]['transaction_id'] == 'tx_2'
        assert results[2]['transaction_id'] == 'tx_3'
    
    @patch('proof_system.proof_integration_manager.ThreadPoolExecutor')
    def test_generate_batch_proofs_async(self, mock_executor_class):
        """Test asynchronous batch proof generation"""
        # Setup mock executor
        mock_executor = MagicMock()
        mock_executor_class.return_value.__enter__.return_value = mock_executor
        
        # Create mock futures
        futures = []
        for i in range(3):
            future = Mock(spec=Future)
            future.result.return_value = {'transaction_id': f'tx_{i+1}', 'async': True}
            futures.append(future)
        
        mock_executor.submit.side_effect = futures
        
        # Mock as_completed
        with patch('proof_system.proof_integration_manager.as_completed') as mock_as_completed:
            mock_as_completed.return_value = futures
            
            transactions = [
                {'transaction_id': 'tx_1'},
                {'transaction_id': 'tx_2'},
                {'transaction_id': 'tx_3'}
            ]
            model_state = {}
            
            results = self.manager.generate_batch_proofs(transactions, model_state, async_mode=True)
            
            assert len(results) == 3
            assert mock_executor.submit.call_count == 3
    
    def test_generate_batch_proofs_async_with_failure(self):
        """Test async batch generation with failure"""
        with patch('proof_system.proof_integration_manager.ThreadPoolExecutor') as mock_executor_class:
            mock_executor = MagicMock()
            mock_executor_class.return_value.__enter__.return_value = mock_executor
            
            # Create future that raises exception
            future = Mock(spec=Future)
            future.result.side_effect = Exception("Async error")
            mock_executor.submit.return_value = future
            
            with patch('proof_system.proof_integration_manager.as_completed') as mock_as_completed:
                mock_as_completed.return_value = [future]
                
                transactions = [{'transaction_id': 'tx_1'}]
                model_state = {}
                
                results = self.manager.generate_batch_proofs(transactions, model_state, async_mode=True)
                
                assert len(results) == 1
                assert 'error' in results[0]
                assert results[0]['error'] == 'Async error'


class TestConfidenceInputExtraction:
    """Test confidence input extraction"""
    
    def setup_method(self):
        """Setup for each test"""
        self.manager = ProofIntegrationManager()
    
    def test_extract_confidence_inputs_all_present(self):
        """Test extracting confidence inputs when all results present"""
        proof_results = {
            'rule_based': {'risk_score': 0.8},
            'ml_based': {'ml_analysis': {'model_prediction': 0.7}},
            'cryptographic': {'basic_proof': {'hash': 'abc123'}}
        }
        
        inputs = self.manager._extract_confidence_inputs(proof_results)
        
        assert inputs['rule_score'] == 0.8
        assert inputs['ml_probability'] == 0.7
        assert inputs['crypto_valid'] is True
    
    def test_extract_confidence_inputs_alternative_fields(self):
        """Test extraction with alternative field names"""
        proof_results = {
            'rule_based': {'confidence': 0.9},
            'ml_based': {'ml_analysis': {'confidence_score': 0.85}},
            'cryptographic': {'basic_proof': {}}
        }
        
        inputs = self.manager._extract_confidence_inputs(proof_results)
        
        assert inputs['rule_score'] == 0.9
        assert inputs['ml_probability'] == 0.85
        assert inputs['crypto_valid'] is False
    
    def test_extract_confidence_inputs_missing_engines(self):
        """Test extraction with missing engines"""
        proof_results = {
            'rule_based': {'risk_score': 0.5}
        }
        
        inputs = self.manager._extract_confidence_inputs(proof_results)
        
        assert inputs['rule_score'] == 0.5
        assert 'ml_probability' not in inputs
        assert 'crypto_valid' not in inputs
    
    def test_extract_confidence_inputs_invalid_format(self):
        """Test extraction with invalid result format"""
        proof_results = {
            'rule_based': 'invalid',
            'ml_based': None,
            'cryptographic': []
        }
        
        inputs = self.manager._extract_confidence_inputs(proof_results)
        
        assert inputs.get('rule_score', 0.0) == 0.0
        assert inputs.get('ml_probability', 0.0) == 0.0
        assert inputs.get('crypto_valid', False) is False


class TestProofAggregation:
    """Test proof aggregation functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.manager = ProofIntegrationManager()
    
    def test_aggregate_proofs_with_scores(self):
        """Test aggregating proofs with various score fields"""
        individual_proofs = {
            'engine1': {'risk_score': 0.8, 'confidence': 0.9},
            'engine2': {'score': 0.7, 'confidence_score': 0.85},
            'engine3': {'confidence': 0.75}
        }
        
        result = self.manager.aggregate_proofs(individual_proofs)
        
        assert 'overall_confidence' in result
        assert 'mean_score' in result
        assert 'score_standard_deviation' in result
        assert 'consensus_reached' in result
        assert result['aggregation_method'] == 'mean_aggregation'
        assert result['engines_contributing'] == 3
        assert 'timestamp' in result
    
    def test_aggregate_proofs_consensus_reached(self):
        """Test consensus detection with low variance"""
        individual_proofs = {
            'engine1': {'risk_score': 0.75},
            'engine2': {'risk_score': 0.78},
            'engine3': {'risk_score': 0.76}
        }
        
        result = self.manager.aggregate_proofs(individual_proofs)
        
        assert result['consensus_reached'] is True
        assert result['score_standard_deviation'] < 0.2
    
    def test_aggregate_proofs_no_consensus(self):
        """Test consensus detection with high variance"""
        individual_proofs = {
            'engine1': {'risk_score': 0.2},
            'engine2': {'risk_score': 0.9},
            'engine3': {'risk_score': 0.5}
        }
        
        result = self.manager.aggregate_proofs(individual_proofs)
        
        assert result['consensus_reached'] is False
        assert result['score_standard_deviation'] >= 0.2
    
    def test_aggregate_proofs_empty(self):
        """Test aggregating empty proofs"""
        result = self.manager.aggregate_proofs({})
        
        assert result['overall_confidence'] == 0.0
        assert result['mean_score'] == 0.0
        assert result['consensus_reached'] is False
        assert result['engines_contributing'] == 0
    
    def test_aggregate_proofs_invalid_values(self):
        """Test aggregation with invalid values"""
        individual_proofs = {
            'engine1': {'risk_score': 'invalid'},
            'engine2': None,
            'engine3': 'not_a_dict'
        }
        
        result = self.manager.aggregate_proofs(individual_proofs)
        
        assert result['overall_confidence'] == 0.0
        assert result['mean_score'] == 0.0


class TestEventHandling:
    """Test event handling functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.manager = ProofIntegrationManager()
    
    def test_register_event_handler(self):
        """Test registering event handlers"""
        handler1 = Mock()
        handler2 = Mock()
        
        self.manager.register_event_handler('test_event', handler1)
        self.manager.register_event_handler('test_event', handler2)
        
        assert 'test_event' in self.manager.event_handlers
        assert len(self.manager.event_handlers['test_event']) == 2
        assert handler1 in self.manager.event_handlers['test_event']
        assert handler2 in self.manager.event_handlers['test_event']
    
    def test_log_event(self):
        """Test event logging"""
        self.manager._log_event('test_event', {'key': 'value'})
        
        events = self.manager.get_event_log()
        assert len(events) == 1
        assert events[0]['event_type'] == 'test_event'
        assert events[0]['details']['key'] == 'value'
        assert 'timestamp' in events[0]
    
    def test_log_event_with_handler(self):
        """Test event logging with registered handler"""
        handler = Mock()
        self.manager.register_event_handler('test_event', handler)
        
        self.manager._log_event('test_event', {'data': 'test'})
        
        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert event['event_type'] == 'test_event'
        assert event['details']['data'] == 'test'
    
    def test_log_event_with_all_handler(self):
        """Test event logging with 'all' handler"""
        all_handler = Mock()
        specific_handler = Mock()
        
        self.manager.register_event_handler('all', all_handler)
        self.manager.register_event_handler('specific_event', specific_handler)
        
        self.manager._log_event('specific_event', {})
        
        all_handler.assert_called_once()
        specific_handler.assert_called_once()
    
    def test_log_event_handler_exception(self):
        """Test event handler that raises exception"""
        bad_handler = Mock(side_effect=Exception("Handler error"))
        good_handler = Mock()
        
        self.manager.register_event_handler('test_event', bad_handler)
        self.manager.register_event_handler('test_event', good_handler)
        
        # Should not raise, just log error
        self.manager._log_event('test_event', {})
        
        bad_handler.assert_called_once()
        good_handler.assert_called_once()
    
    def test_event_log_limit(self):
        """Test event log size limit"""
        for i in range(1100):
            self.manager._log_event(f'event_{i}', {})
        
        events = self.manager.get_event_log()
        assert len(events) == 1000  # Should keep only last 1000
        assert events[0]['event_type'] == 'event_100'  # First 100 should be removed
    
    def test_clear_event_log(self):
        """Test clearing event log"""
        self.manager._log_event('event1', {})
        self.manager._log_event('event2', {})
        
        assert len(self.manager.get_event_log()) == 2
        
        self.manager.clear_event_log()
        
        assert len(self.manager.get_event_log()) == 0
    
    def test_event_log_thread_safety(self):
        """Test thread safety of event logging"""
        def log_events(start, count):
            for i in range(count):
                self.manager._log_event(f'thread_event_{start}_{i}', {})
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=log_events, args=(i, 20))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        events = self.manager.get_event_log()
        assert len(events) == 100  # 5 threads * 20 events each


class TestPerformanceMonitoring:
    """Test performance monitoring functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.manager = ProofIntegrationManager()
        
        # Register a mock engine
        mock_engine = Mock()
        mock_engine.generate_proof.return_value = {'proof': True}
        self.manager.register_engine('test_engine', mock_engine)
    
    def test_enable_disable_monitoring(self):
        """Test enabling and disabling performance monitoring"""
        assert self.manager.monitoring_enabled is False
        
        self.manager.enable_performance_monitoring()
        assert self.manager.monitoring_enabled is True
        
        self.manager.disable_performance_monitoring()
        assert self.manager.monitoring_enabled is False
    
    def test_update_engine_performance_success(self):
        """Test updating engine performance on success"""
        self.manager.enable_performance_monitoring()
        self.manager._update_engine_performance('test_engine', 0.5, success=True)
        
        stats = self.manager.performance_stats['engine_performance']['test_engine']
        assert stats['calls'] == 1
        assert stats['total_time'] == 0.5
        assert stats['avg_time'] == 0.5
        assert stats['errors'] == 0
        
        # Second call
        self.manager._update_engine_performance('test_engine', 0.3, success=True)
        
        stats = self.manager.performance_stats['engine_performance']['test_engine']
        assert stats['calls'] == 2
        assert stats['total_time'] == 0.8
        assert stats['avg_time'] == 0.4
        assert stats['errors'] == 0
    
    def test_update_engine_performance_failure(self):
        """Test updating engine performance on failure"""
        self.manager.enable_performance_monitoring()
        self.manager._update_engine_performance('test_engine', 0.5, success=False)
        
        stats = self.manager.performance_stats['engine_performance']['test_engine']
        assert stats['calls'] == 1
        assert stats['total_time'] == 0  # No time added on failure
        assert stats['avg_time'] == 0
        assert stats['errors'] == 1
    
    def test_get_performance_stats_monitoring_disabled(self):
        """Test getting performance stats when monitoring disabled"""
        stats = self.manager.get_performance_stats()
        
        assert stats == {'error': 'Performance monitoring not enabled'}
    
    def test_get_performance_stats_monitoring_enabled(self):
        """Test getting performance stats when monitoring enabled"""
        self.manager.enable_performance_monitoring()
        
        # Generate some proofs to populate stats
        transaction = {'transaction_id': 'tx_1'}
        model_state = {}
        
        self.manager.generate_comprehensive_proof(transaction, model_state)
        self.manager.generate_comprehensive_proof(transaction, model_state)
        
        stats = self.manager.get_performance_stats()
        
        assert stats['total_proofs'] == 2
        assert stats['monitoring_enabled'] is True
        assert 'engine_performance' in stats
        assert 'avg_generation_time' in stats
        assert 'min_generation_time' in stats
        assert 'max_generation_time' in stats
        assert 'total_generation_time' in stats
    
    def test_get_performance_stats_empty(self):
        """Test getting performance stats with no data"""
        self.manager.enable_performance_monitoring()
        
        stats = self.manager.get_performance_stats()
        
        assert stats['total_proofs'] == 0
        assert stats['monitoring_enabled'] is True
        assert 'avg_generation_time' not in stats  # No generation times yet


class TestAlgebraicValidation:
    """Test algebraic validation functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.manager = ProofIntegrationManager()
    
    def test_validate_with_algebraic_rules(self):
        """Test validation using algebraic rule enforcer"""
        with patch.object(self.manager.algebraic_enforcer, 'validate_gradients') as mock_validate:
            mock_validate.return_value = {'valid': True, 'violations': []}
            
            gradients = [0.1, 0.2, 0.3]
            learning_rate = 0.01
            
            result = self.manager.validate_with_algebraic_rules(gradients, learning_rate)
            
            mock_validate.assert_called_once_with(gradients, learning_rate)
            assert result['valid'] is True


class TestTimeoutConfiguration:
    """Test timeout configuration"""
    
    def test_set_timeout(self):
        """Test setting timeout value"""
        manager = ProofIntegrationManager()
        
        assert manager.timeout == 30  # Default
        
        manager.set_timeout(60)
        
        assert manager.timeout == 60


class TestHealthCheck:
    """Test health check functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.manager = ProofIntegrationManager()
    
    def test_health_check_all_healthy(self):
        """Test health check with all engines healthy"""
        # Create mock engines
        engine1 = Mock()
        engine1.generate_proof.return_value = {'proof': True}
        
        engine2 = Mock()
        engine2.evaluate_transaction.return_value = {'result': True}
        
        engine3 = Mock()
        engine3.generate_ml_proof.return_value = {'ml_result': True}
        
        self.manager.register_engine('engine1', engine1)
        self.manager.register_engine('engine2', engine2)
        self.manager.register_engine('engine3', engine3)
        
        health = self.manager.health_check()
        
        assert health['overall_status'] == 'healthy'
        assert 'timestamp' in health
        assert len(health['engines']) == 3
        assert all(e['status'] == 'healthy' for e in health['engines'].values())
    
    def test_health_check_with_unhealthy_engine(self):
        """Test health check with unhealthy engine"""
        healthy_engine = Mock()
        healthy_engine.generate_proof.return_value = {'proof': True}
        
        unhealthy_engine = Mock()
        unhealthy_engine.generate_proof.side_effect = Exception("Engine failure")
        
        self.manager.register_engine('healthy', healthy_engine)
        self.manager.register_engine('unhealthy', unhealthy_engine)
        
        health = self.manager.health_check()
        
        assert health['overall_status'] == 'degraded'
        assert health['engines']['healthy']['status'] == 'healthy'
        assert health['engines']['unhealthy']['status'] == 'unhealthy'
        assert 'error' in health['engines']['unhealthy']
        assert health['failed_engines'] == ['unhealthy']
    
    def test_health_check_no_engines(self):
        """Test health check with no engines registered"""
        health = self.manager.health_check()
        
        assert health['overall_status'] == 'healthy'
        assert health['engines'] == {}


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def setup_method(self):
        """Setup for each test"""
        self.manager = ProofIntegrationManager()
    
    def test_comprehensive_proof_no_engines(self):
        """Test generating proof with no engines registered"""
        transaction = {'transaction_id': 'tx_1'}
        model_state = {}
        
        proof = self.manager.generate_comprehensive_proof(transaction, model_state)
        
        assert proof['engines_used'] == []
        assert proof['confidence'] == 0.0
    
    def test_comprehensive_proof_confidence_generation_failure(self):
        """Test proof generation when confidence generation fails"""
        mock_engine = Mock()
        mock_engine.generate_proof.return_value = {'proof': True}
        self.manager.register_engine('test_engine', mock_engine)
        
        with patch.object(self.manager.confidence_generator, 'generate_confidence') as mock_confidence:
            mock_confidence.side_effect = Exception("Confidence error")
            
            transaction = {'transaction_id': 'tx_1'}
            model_state = {}
            
            proof = self.manager.generate_comprehensive_proof(transaction, model_state)
            
            assert proof['confidence'] == 0.0
            assert 'error' in proof['confidence_details']
    
    def test_batch_proofs_empty_list(self):
        """Test batch proof generation with empty transaction list"""
        results = self.manager.generate_batch_proofs([], {})
        
        assert results == []
    
    def test_aggregate_proofs_non_numeric_values(self):
        """Test aggregation with non-numeric score values"""
        individual_proofs = {
            'engine1': {'risk_score': [1, 2, 3]},  # List instead of number
            'engine2': {'risk_score': {'nested': 0.5}},  # Dict instead of number
            'engine3': {'risk_score': None}  # None value
        }
        
        result = self.manager.aggregate_proofs(individual_proofs)
        
        assert result['mean_score'] == 0.0
        assert result['overall_confidence'] == 0.0


class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    def setup_method(self):
        """Setup for realistic test scenario"""
        self.manager = ProofIntegrationManager()
        
        # Setup realistic engines
        rule_engine = Mock()
        rule_engine.evaluate_transaction.return_value = {
            'risk_score': 0.65,
            'confidence': 0.8,
            'rules_triggered': ['high_amount', 'new_account']
        }
        
        ml_engine = Mock()
        ml_engine.generate_ml_proof.return_value = {
            'ml_analysis': {
                'model_prediction': 0.72,
                'confidence_score': 0.85,
                'features_used': 10
            }
        }
        
        crypto_engine = Mock()
        crypto_engine.generate_proof_for_transaction.return_value = {
            'basic_proof': {
                'hash': 'abc123def456',
                'signature': 'sig_xyz789'
            }
        }
        
        self.manager.register_engine('rule_based', rule_engine)
        self.manager.register_engine('ml_based', ml_engine)
        self.manager.register_engine('cryptographic', crypto_engine)
    
    def test_full_workflow_with_monitoring(self):
        """Test complete workflow with monitoring enabled"""
        # Enable monitoring
        self.manager.enable_performance_monitoring()
        
        # Register event handler
        event_handler = Mock()
        self.manager.register_event_handler('all', event_handler)
        
        # Generate proof
        transaction = {
            'transaction_id': 'tx_workflow_1',
            'amount': 10000,
            'account_age_days': 5
        }
        model_state = {
            'iteration': 100,
            'loss': 0.05
        }
        
        proof = self.manager.generate_comprehensive_proof(transaction, model_state)
        
        # Verify proof structure
        assert proof['transaction_id'] == 'tx_workflow_1'
        assert len(proof['engines_used']) == 3
        assert proof['confidence'] > 0
        
        # Verify events were logged
        assert event_handler.call_count >= 2  # At least start and complete events
        
        # Verify performance stats
        stats = self.manager.get_performance_stats()
        assert stats['total_proofs'] == 1
        assert 'test_engine' not in stats['engine_performance']  # Only registered engines
        
        # Perform health check
        health = self.manager.health_check()
        assert health['overall_status'] == 'healthy'
    
    def test_batch_processing_with_failures(self):
        """Test batch processing with some failures"""
        # Make ml_engine fail for specific transactions
        def ml_side_effect(tx, state):
            if tx.get('fail_ml', False):
                raise Exception("ML processing failed")
            return {'ml_analysis': {'model_prediction': 0.5}}
        
        ml_engine = self.manager.engines['ml_based']
        ml_engine.generate_ml_proof.side_effect = ml_side_effect
        
        transactions = [
            {'transaction_id': 'tx_1', 'amount': 100},
            {'transaction_id': 'tx_2', 'amount': 200, 'fail_ml': True},
            {'transaction_id': 'tx_3', 'amount': 300}
        ]
        model_state = {}
        
        results = self.manager.generate_batch_proofs(transactions, model_state)
        
        assert len(results) == 3
        assert 'errors' not in results[0]  # First should succeed
        assert 'errors' in results[1]  # Second should have ML error
        assert 'ml_based' in results[1]['errors']
        assert 'errors' not in results[2]  # Third should succeed
    
    def test_concurrent_event_logging(self):
        """Test concurrent event logging and handler execution"""
        handler_calls = []
        lock = threading.Lock()
        
        def thread_safe_handler(event):
            with lock:
                handler_calls.append(event['event_type'])
        
        self.manager.register_event_handler('all', thread_safe_handler)
        
        def generate_proofs():
            for i in range(5):
                transaction = {'transaction_id': f'tx_{threading.current_thread().name}_{i}'}
                self.manager.generate_comprehensive_proof(transaction, {})
        
        threads = []
        for i in range(3):
            t = threading.Thread(target=generate_proofs, name=f'thread_{i}')
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Should have logged events for all proof generations
        assert len(handler_calls) >= 30  # At least 2 events per proof * 15 proofs
        
        # Event log should be intact
        events = self.manager.get_event_log()
        assert len(events) >= 30


class TestRobustness:
    """Test robustness and error recovery"""
    
    def setup_method(self):
        """Setup for robustness tests"""
        self.manager = ProofIntegrationManager()
    
    def test_engine_registration_overwrites(self):
        """Test that re-registering an engine overwrites the old one"""
        engine1 = Mock()
        engine2 = Mock()
        
        self.manager.register_engine('test', engine1)
        assert self.manager.engines['test'] is engine1
        
        self.manager.register_engine('test', engine2)
        assert self.manager.engines['test'] is engine2
    
    def test_transaction_without_id(self):
        """Test handling transaction without transaction_id"""
        mock_engine = Mock()
        mock_engine.generate_proof.return_value = {'proof': True}
        self.manager.register_engine('test', mock_engine)
        
        transaction = {'amount': 100}  # No transaction_id
        proof = self.manager.generate_comprehensive_proof(transaction, {})
        
        assert proof['transaction_id'] == 'unknown'
    
    def test_large_event_log_performance(self):
        """Test performance with large event log"""
        start_time = time.time()
        
        # Generate many events
        for i in range(2000):
            self.manager._log_event(f'event_{i}', {'index': i})
        
        # Should complete quickly despite large number of events
        elapsed = time.time() - start_time
        assert elapsed < 5  # Should complete in less than 5 seconds
        
        # Log should be trimmed to 1000
        assert len(self.manager.get_event_log()) == 1000
    
    def test_confidence_generator_missing(self):
        """Test behavior when confidence generator is None"""
        self.manager.confidence_generator = None
        
        mock_engine = Mock()
        mock_engine.generate_proof.return_value = {'proof': True}
        self.manager.register_engine('test', mock_engine)
        
        # Should handle gracefully
        with patch.object(self.manager, 'logger') as mock_logger:
            proof = self.manager.generate_comprehensive_proof({'transaction_id': 'tx_1'}, {})
            
            # Should log an error but not crash
            assert mock_logger.error.called
            assert proof['confidence'] == 0.0


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])