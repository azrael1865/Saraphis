"""
Basic tests for fraud detection functionality.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class TestFraudDetection(unittest.TestCase):
    """Test fraud detection components."""
    
    def test_import_core(self):
        """Test that core module can be imported."""
        from core.fraud_core import FraudDetectionCore
        detector = FraudDetectionCore()
        self.assertIsNotNone(detector)
    
    def test_import_data_loader(self):
        """Test that data loader can be imported."""
        from data.data_loader import TransactionDataLoader
        loader = TransactionDataLoader()
        self.assertIsNotNone(loader)
    
    def test_import_api(self):
        """Test that API can be imported."""
        from api.api_interface import FraudDetectionAPI
        api = FraudDetectionAPI()
        self.assertIsNotNone(api)
    
    def test_basic_fraud_detection(self):
        """Test basic fraud detection."""
        from core.fraud_core import FraudDetectionCore
        
        detector = FraudDetectionCore()
        
        # Test transaction
        transaction = {
            'transaction_id': 'TEST001',
            'amount': 5000.00,
            'timestamp': '2024-01-15 03:30:00',
            'user_id': 'USER123'
        }
        
        result = detector.detect_fraud(transaction)
        
        # Check result structure
        self.assertIn('fraud_score', result)
        self.assertIn('is_fraud', result)
        self.assertIn('risk_level', result)
        self.assertIn('transaction_id', result)
        
        # Check data types
        self.assertIsInstance(result['fraud_score'], float)
        self.assertIsInstance(result['is_fraud'], bool)
        self.assertIsInstance(result['risk_level'], str)


if __name__ == '__main__':
    unittest.main()
