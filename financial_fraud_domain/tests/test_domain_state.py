"""
Comprehensive tests for Financial Fraud Domain State Management
Tests all data structures, state management, persistence, and validation functionality
"""

import json
import logging
import os
import pickle
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

# Import state management components
from domain_state import (
    Transaction, TransactionStatus,
    FraudIndicator, FraudSeverity, IndicatorType,
    FraudResult, FraudDecision,
    ValidationResult, ProcessingMetadata,
    FinancialFraudState, FraudDomainStateManager,
    StateValidationError, StatePersistenceError, StateRecoveryError
)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestDataStructures(unittest.TestCase):
    """Test data structure classes and serialization"""
    
    def test_transaction_creation_and_serialization(self):
        """Test Transaction creation and serialization"""
        transaction = Transaction(
            transaction_id="TXN-001",
            amount=100.50,
            currency="USD",
            merchant="Test Store",
            timestamp=datetime.now(),
            user_id="USER-001",
            account_id="ACC-001",
            payment_method="credit_card",
            status=TransactionStatus.PENDING
        )
        
        # Test basic properties
        self.assertEqual(transaction.transaction_id, "TXN-001")
        self.assertEqual(transaction.amount, 100.50)
        self.assertEqual(transaction.status, TransactionStatus.PENDING)
        
        # Test serialization
        data = transaction.to_dict()
        self.assertIsInstance(data['timestamp'], str)
        self.assertEqual(data['status'], 'pending')
        
        # Test deserialization
        transaction2 = Transaction.from_dict(data)
        self.assertEqual(transaction2.transaction_id, transaction.transaction_id)
        self.assertEqual(transaction2.amount, transaction.amount)
        self.assertEqual(transaction2.status, transaction.status)
    
    def test_fraud_indicator_creation_and_serialization(self):
        """Test FraudIndicator creation and serialization"""
        indicator = FraudIndicator(
            indicator_id="IND-001",
            indicator_type=IndicatorType.VELOCITY,
            severity=FraudSeverity.HIGH,
            confidence=0.95,
            description="High velocity detected",
            timestamp=datetime.now(),
            source="velocity_engine"
        )
        
        # Test basic properties
        self.assertEqual(indicator.confidence, 0.95)
        self.assertEqual(indicator.severity, FraudSeverity.HIGH)
        self.assertEqual(indicator.indicator_type, IndicatorType.VELOCITY)
        
        # Test serialization
        data = indicator.to_dict()
        self.assertEqual(data['indicator_type'], 'velocity')
        self.assertEqual(data['severity'], 'high')
        
        # Test deserialization
        indicator2 = FraudIndicator.from_dict(data)
        self.assertEqual(indicator2.indicator_id, indicator.indicator_id)
        self.assertEqual(indicator2.confidence, indicator.confidence)
    
    def test_fraud_result_creation_and_serialization(self):
        """Test FraudResult creation and serialization"""
        result = FraudResult(
            result_id="RES-001",
            transaction_id="TXN-001",
            risk_score=0.75,
            fraud_probability=0.8,
            decision=FraudDecision.REVIEW
        )
        
        # Test basic properties
        self.assertEqual(result.risk_score, 0.75)
        self.assertEqual(result.decision, FraudDecision.REVIEW)
        
        # Test serialization with indicators
        indicator = FraudIndicator(
            indicator_id="IND-001",
            indicator_type=IndicatorType.AMOUNT,
            severity=FraudSeverity.MEDIUM,
            confidence=0.7,
            description="Medium risk amount",
            timestamp=datetime.now(),
            source="amount_checker"
        )
        
        result.indicators.append(indicator)
        data = result.to_dict()
        self.assertEqual(data['decision'], 'review')
        self.assertEqual(len(data['indicators']), 1)
        
        # Test deserialization
        result2 = FraudResult.from_dict(data)
        self.assertEqual(result2.result_id, result.result_id)
        self.assertEqual(len(result2.indicators), 1)
    
    def test_processing_metadata(self):
        """Test ProcessingMetadata functionality"""
        metadata = ProcessingMetadata(
            process_id="PROC-001",
            start_time=datetime.now()
        )
        
        # Test duration calculation
        self.assertIsNone(metadata.duration())
        
        metadata.end_time = metadata.start_time + timedelta(seconds=10)
        duration = metadata.duration()
        self.assertIsNotNone(duration)
        self.assertEqual(duration.total_seconds(), 10.0)
        
        # Test serialization
        data = metadata.to_dict()
        self.assertIn('start_time', data)
        self.assertIn('end_time', data)


class TestFinancialFraudState(unittest.TestCase):
    """Test FinancialFraudState functionality"""
    
    def setUp(self):
        """Set up test state"""
        self.state = FinancialFraudState(
            domain_name="test-domain",
            config={"persistence_enabled": False}
        )
    
    def test_transaction_management(self):
        """Test transaction add/get operations"""
        # Create test transaction
        transaction = Transaction(
            transaction_id="TXN-001",
            amount=100.0,
            currency="USD",
            merchant="Test Merchant",
            timestamp=datetime.now(),
            user_id="USER-001",
            account_id="ACC-001",
            payment_method="card"
        )
        
        # Add transaction
        result = self.state.add_transaction(transaction)
        self.assertTrue(result)
        
        # Retrieve transaction
        retrieved = self.state.get_transaction("TXN-001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.amount, 100.0)
        
        # Test duplicate transaction (should fail)
        result = self.state.add_transaction(transaction)
        self.assertFalse(result)
        
        # Test user transactions
        user_txns = self.state.get_user_transactions("USER-001")
        self.assertEqual(len(user_txns), 1)
        
        # Test recent transactions
        recent = self.state.get_recent_transactions(hours=1)
        self.assertEqual(len(recent), 1)
    
    def test_transaction_validation(self):
        """Test transaction validation"""
        # Invalid amount
        invalid_txn = Transaction(
            transaction_id="TXN-002",
            amount=-100.0,  # Invalid negative amount
            currency="USD",
            merchant="Test",
            timestamp=datetime.now(),
            user_id="USER-001",
            account_id="ACC-001",
            payment_method="card"
        )
        
        result = self.state.add_transaction(invalid_txn)
        self.assertFalse(result)
        
        # Missing user_id
        invalid_txn2 = Transaction(
            transaction_id="TXN-003",
            amount=100.0,
            currency="USD",
            merchant="Test",
            timestamp=datetime.now(),
            user_id="",  # Empty user_id
            account_id="ACC-001",
            payment_method="card"
        )
        
        result = self.state.add_transaction(invalid_txn2)
        self.assertFalse(result)
    
    def test_fraud_indicator_management(self):
        """Test fraud indicator operations"""
        indicator = FraudIndicator(
            indicator_id="IND-001",
            indicator_type=IndicatorType.PATTERN,
            severity=FraudSeverity.HIGH,
            confidence=0.9,
            description="Suspicious pattern detected",
            timestamp=datetime.now(),
            source="pattern_engine",
            transaction_id="TXN-001"
        )
        
        # Add indicator
        result = self.state.add_fraud_indicator(indicator)
        self.assertTrue(result)
        
        # Retrieve indicator
        retrieved = self.state.get_fraud_indicator("IND-001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.confidence, 0.9)
        
        # Test invalid confidence
        invalid_indicator = FraudIndicator(
            indicator_id="IND-002",
            indicator_type=IndicatorType.AMOUNT,
            severity=FraudSeverity.LOW,
            confidence=1.5,  # Invalid confidence > 1
            description="Test",
            timestamp=datetime.now(),
            source="test"
        )
        
        result = self.state.add_fraud_indicator(invalid_indicator)
        self.assertFalse(result)
    
    def test_fraud_result_management(self):
        """Test fraud result operations"""
        # Add a transaction first
        transaction = Transaction(
            transaction_id="TXN-100",
            amount=500.0,
            currency="USD",
            merchant="Test Store",
            timestamp=datetime.now(),
            user_id="USER-100",
            account_id="ACC-100",
            payment_method="card"
        )
        self.state.add_transaction(transaction)
        
        # Create fraud result
        result = FraudResult(
            result_id="RES-001",
            transaction_id="TXN-100",
            risk_score=0.85,
            fraud_probability=0.9,
            decision=FraudDecision.BLOCK
        )
        
        # Add result
        success = self.state.add_fraud_result(result)
        self.assertTrue(success)
        
        # Check transaction status was updated
        updated_txn = self.state.get_transaction("TXN-100")
        self.assertEqual(updated_txn.status, TransactionStatus.BLOCKED)
        self.assertEqual(updated_txn.risk_score, 0.85)
        
        # Retrieve result
        retrieved = self.state.get_fraud_result("RES-001")
        self.assertIsNotNone(retrieved)
        
        # Get result by transaction
        txn_result = self.state.get_transaction_result("TXN-100")
        self.assertIsNotNone(txn_result)
        self.assertEqual(txn_result.result_id, "RES-001")
    
    def test_state_validation(self):
        """Test state validation functionality"""
        # Add valid data
        transaction = Transaction(
            transaction_id="TXN-200",
            amount=100.0,
            currency="USD",
            merchant="Valid Merchant",
            timestamp=datetime.now(),
            user_id="USER-200",
            account_id="ACC-200",
            payment_method="card"
        )
        self.state.add_transaction(transaction)
        
        # Validate state
        validation = self.state.validate_state()
        self.assertTrue(validation.is_valid)
        self.assertEqual(len(validation.errors), 0)
        
        # Add indicator with invalid transaction reference
        indicator = FraudIndicator(
            indicator_id="IND-100",
            indicator_type=IndicatorType.BEHAVIORAL,
            severity=FraudSeverity.MEDIUM,
            confidence=0.7,
            description="Test indicator",
            timestamp=datetime.now(),
            source="test",
            transaction_id="INVALID-TXN"  # Non-existent transaction
        )
        self.state.add_fraud_indicator(indicator)
        
        # Validate again
        validation2 = self.state.validate_state()
        self.assertTrue(validation2.is_valid)  # Warnings don't make it invalid
        self.assertGreater(len(validation2.warnings), 0)
    
    def test_processing_metadata_management(self):
        """Test processing metadata functionality"""
        # Start processing
        process_id = "PROC-001"
        metadata = self.state.start_processing(process_id)
        
        self.assertEqual(metadata.process_id, process_id)
        self.assertIsNotNone(metadata.start_time)
        
        # Update processing metadata
        proc_meta = self.state.processing_metadata[process_id]
        proc_meta.transaction_count = 10
        proc_meta.fraud_detected_count = 2
        
        # End processing
        time.sleep(0.1)  # Ensure some time passes
        result = self.state.end_processing(process_id)
        self.assertTrue(result)
        
        # Check metadata was updated
        final_meta = self.state.processing_metadata[process_id]
        self.assertIsNotNone(final_meta.end_time)
        self.assertEqual(final_meta.status, "completed")
        self.assertIn('throughput', final_meta.performance_metrics)
    
    def test_state_indexes(self):
        """Test state indexing functionality"""
        # Add multiple transactions
        for i in range(5):
            transaction = Transaction(
                transaction_id=f"TXN-{i}",
                amount=100.0 * (i + 1),
                currency="USD",
                merchant=f"Merchant-{i % 2}",
                timestamp=datetime.now() - timedelta(hours=i),
                user_id="USER-001",
                account_id=f"ACC-{i % 3}",
                payment_method="card"
            )
            self.state.add_transaction(transaction)
        
        # Test user index
        user_txns = self.state.get_user_transactions("USER-001")
        self.assertEqual(len(user_txns), 5)
        
        # Test limit
        limited_txns = self.state.get_user_transactions("USER-001", limit=3)
        self.assertEqual(len(limited_txns), 3)
        
        # Test merchant index
        merchant_0_txns = self.state.get_merchant_transactions("Merchant-0")
        merchant_1_txns = self.state.get_merchant_transactions("Merchant-1")
        self.assertEqual(len(merchant_0_txns), 3)  # 0, 2, 4
        self.assertEqual(len(merchant_1_txns), 2)  # 1, 3
    
    def test_state_summary(self):
        """Test state summary generation"""
        # Add some data
        for i in range(3):
            transaction = Transaction(
                transaction_id=f"TXN-S{i}",
                amount=100.0,
                currency="USD",
                merchant=f"Merchant-{i}",
                timestamp=datetime.now(),
                user_id=f"USER-{i}",
                account_id=f"ACC-{i}",
                payment_method="card"
            )
            self.state.add_transaction(transaction)
        
        # Get summary
        summary = self.state.get_state_summary()
        
        self.assertEqual(summary["counts"]["transactions"], 3)
        self.assertEqual(summary["indexes"]["unique_users"], 3)
        self.assertEqual(summary["indexes"]["unique_merchants"], 3)
        self.assertIn("performance_metrics", summary)
    
    def test_cleanup_old_data(self):
        """Test cleanup of old data"""
        # Add old transactions
        old_date = datetime.now() - timedelta(days=40)
        for i in range(3):
            transaction = Transaction(
                transaction_id=f"OLD-TXN-{i}",
                amount=100.0,
                currency="USD",
                merchant="Old Merchant",
                timestamp=old_date,
                user_id="USER-OLD",
                account_id="ACC-OLD",
                payment_method="card"
            )
            self.state.add_transaction(transaction)
        
        # Add recent transaction
        recent_txn = Transaction(
            transaction_id="RECENT-TXN",
            amount=200.0,
            currency="USD",
            merchant="Recent Merchant",
            timestamp=datetime.now(),
            user_id="USER-NEW",
            account_id="ACC-NEW",
            payment_method="card"
        )
        self.state.add_transaction(recent_txn)
        
        # Run cleanup
        removed = self.state.cleanup_old_data(days=30)
        self.assertEqual(removed, 3)
        
        # Check recent transaction still exists
        self.assertIsNotNone(self.state.get_transaction("RECENT-TXN"))
        self.assertIsNone(self.state.get_transaction("OLD-TXN-0"))


class TestStatePersistence(unittest.TestCase):
    """Test state persistence functionality"""
    
    def setUp(self):
        """Set up test with temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        self.state = FinancialFraudState(
            domain_name="test-domain",
            config={
                "persistence_enabled": True,
                "persistence_path": self.temp_dir
            }
        )
    
    def tearDown(self):
        """Clean up temporary directory"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_state(self):
        """Test saving and loading state"""
        # Add test data
        transaction = Transaction(
            transaction_id="PERSIST-TXN-001",
            amount=250.0,
            currency="EUR",
            merchant="Persistent Store",
            timestamp=datetime.now(),
            user_id="USER-PERSIST",
            account_id="ACC-PERSIST",
            payment_method="debit_card"
        )
        self.state.add_transaction(transaction)
        
        indicator = FraudIndicator(
            indicator_id="PERSIST-IND-001",
            indicator_type=IndicatorType.LOCATION,
            severity=FraudSeverity.LOW,
            confidence=0.6,
            description="Location check",
            timestamp=datetime.now(),
            source="geo_engine"
        )
        self.state.add_fraud_indicator(indicator)
        
        # Save state
        filepath = Path(self.temp_dir) / "test_state.pkl"
        result = self.state.save_state(filepath)
        self.assertTrue(result)
        self.assertTrue(filepath.exists())
        
        # Create new state and load
        new_state = FinancialFraudState(
            domain_name="test-domain",
            config={"persistence_enabled": True}
        )
        
        result = new_state.load_state(filepath)
        self.assertTrue(result)
        
        # Verify data was loaded
        loaded_txn = new_state.get_transaction("PERSIST-TXN-001")
        self.assertIsNotNone(loaded_txn)
        self.assertEqual(loaded_txn.amount, 250.0)
        
        loaded_ind = new_state.get_fraud_indicator("PERSIST-IND-001")
        self.assertIsNotNone(loaded_ind)
        self.assertEqual(loaded_ind.confidence, 0.6)
    
    def test_state_recovery(self):
        """Test state recovery from latest checkpoint"""
        # Add data and save
        for i in range(3):
            transaction = Transaction(
                transaction_id=f"RECOVER-TXN-{i}",
                amount=100.0 * i,
                currency="USD",
                merchant="Recovery Test",
                timestamp=datetime.now(),
                user_id="USER-RECOVER",
                account_id="ACC-RECOVER",
                payment_method="card"
            )
            self.state.add_transaction(transaction)
            
            # Save state
            time.sleep(0.1)  # Ensure different timestamps
            self.state.save_state()
        
        # Clear state
        self.state.clear_state()
        self.assertEqual(len(self.state.transactions), 0)
        
        # Recover state
        result = self.state.recover_state()
        self.assertTrue(result)
        
        # Verify recovery
        self.assertEqual(len(self.state.transactions), 3)
        for i in range(3):
            self.assertIsNotNone(self.state.get_transaction(f"RECOVER-TXN-{i}"))
    
    def test_checkpoint_creation(self):
        """Test checkpoint creation"""
        # Add data
        transaction = Transaction(
            transaction_id="CHECKPOINT-TXN",
            amount=500.0,
            currency="GBP",
            merchant="Checkpoint Store",
            timestamp=datetime.now(),
            user_id="USER-CHECK",
            account_id="ACC-CHECK",
            payment_method="card"
        )
        self.state.add_transaction(transaction)
        
        # Create checkpoint
        checkpoint_id = self.state.create_checkpoint()
        self.assertIsNotNone(checkpoint_id)
        
        # Verify checkpoint in history
        self.assertGreater(len(self.state.state_history), 0)
        latest_checkpoint = self.state.state_history[-1]
        self.assertEqual(latest_checkpoint['checkpoint_id'], checkpoint_id)


class TestFraudDomainStateManager(unittest.TestCase):
    """Test FraudDomainStateManager functionality"""
    
    def setUp(self):
        """Set up test manager"""
        self.manager = FraudDomainStateManager()
    
    def test_create_state(self):
        """Test state creation"""
        state = self.manager.create_state("domain1", {"persistence_enabled": False})
        self.assertIsNotNone(state)
        self.assertEqual(state.domain_name, "domain1")
        
        # Test duplicate creation
        state2 = self.manager.create_state("domain1")
        self.assertEqual(state, state2)  # Should return same instance
    
    def test_get_state(self):
        """Test state retrieval"""
        # Create state
        self.manager.create_state("domain2")
        
        # Get state
        state = self.manager.get_state("domain2")
        self.assertIsNotNone(state)
        self.assertEqual(state.domain_name, "domain2")
        
        # Get non-existent state
        state = self.manager.get_state("non-existent")
        self.assertIsNone(state)
    
    def test_remove_state(self):
        """Test state removal"""
        # Create state
        self.manager.create_state("domain3")
        
        # Remove state
        result = self.manager.remove_state("domain3")
        self.assertTrue(result)
        
        # Verify removal
        state = self.manager.get_state("domain3")
        self.assertIsNone(state)
        
        # Remove non-existent
        result = self.manager.remove_state("non-existent")
        self.assertFalse(result)
    
    def test_get_all_states(self):
        """Test getting all states"""
        # Create multiple states
        for i in range(3):
            self.manager.create_state(f"domain{i}")
        
        # Get all states
        all_states = self.manager.get_all_states()
        self.assertEqual(len(all_states), 3)
        self.assertIn("domain0", all_states)
        self.assertIn("domain1", all_states)
        self.assertIn("domain2", all_states)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics functionality"""
    
    def setUp(self):
        """Set up test state"""
        self.state = FinancialFraudState(
            domain_name="perf-test",
            config={"persistence_enabled": False}
        )
    
    def test_performance_tracking(self):
        """Test performance metrics tracking"""
        # Add transactions and results with processing times
        for i in range(5):
            transaction = Transaction(
                transaction_id=f"PERF-TXN-{i}",
                amount=100.0,
                currency="USD",
                merchant="Perf Test",
                timestamp=datetime.now(),
                user_id="USER-PERF",
                account_id="ACC-PERF",
                payment_method="card"
            )
            self.state.add_transaction(transaction)
            
            result = FraudResult(
                result_id=f"PERF-RES-{i}",
                transaction_id=f"PERF-TXN-{i}",
                risk_score=0.5 + (i * 0.1),
                fraud_probability=0.6 + (i * 0.05),
                decision=FraudDecision.ALLOW if i < 3 else FraudDecision.BLOCK,
                processing_time=0.1 + (i * 0.02)
            )
            self.state.add_fraud_result(result)
        
        # Get performance metrics
        metrics = self.state.get_performance_metrics()
        
        self.assertEqual(metrics["total_transactions"], 5)
        self.assertGreater(metrics["fraud_detected"], 0)
        self.assertGreater(metrics["processing_time_avg"], 0)
        self.assertIn("processing_time_min", metrics)
        self.assertIn("processing_time_max", metrics)
        
        # Verify processing time calculations
        self.assertAlmostEqual(metrics["processing_time_min"], 0.1, places=2)
        self.assertAlmostEqual(metrics["processing_time_max"], 0.18, places=2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        """Set up test state"""
        self.state = FinancialFraudState(
            domain_name="edge-test",
            config={"persistence_enabled": False}
        )
    
    def test_empty_state_operations(self):
        """Test operations on empty state"""
        # Get from empty state
        self.assertIsNone(self.state.get_transaction("NON-EXISTENT"))
        self.assertEqual(len(self.state.get_user_transactions("USER")), 0)
        self.assertEqual(len(self.state.get_recent_transactions()), 0)
        
        # Validate empty state
        validation = self.state.validate_state()
        self.assertTrue(validation.is_valid)
        
        # Cleanup empty state
        removed = self.state.cleanup_old_data()
        self.assertEqual(removed, 0)
    
    def test_large_state_handling(self):
        """Test handling of large state"""
        # Add many transactions
        batch_size = 100
        for i in range(batch_size):
            transaction = Transaction(
                transaction_id=f"LARGE-TXN-{i}",
                amount=float(i),
                currency="USD",
                merchant=f"Merchant-{i % 10}",
                timestamp=datetime.now() - timedelta(minutes=i),
                user_id=f"USER-{i % 20}",
                account_id=f"ACC-{i % 15}",
                payment_method="card"
            )
            self.state.add_transaction(transaction)
        
        # Verify state size
        self.assertEqual(len(self.state.transactions), batch_size)
        
        # Test performance with large state
        start_time = time.time()
        user_txns = self.state.get_user_transactions("USER-0")
        elapsed = time.time() - start_time
        
        # Should be fast even with many transactions
        self.assertLess(elapsed, 0.1)
        self.assertEqual(len(user_txns), 5)  # USER-0, USER-20, USER-40, etc.


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run all tests
    unittest.main(verbosity=2)