"""
Test Data Generator for Financial Fraud Detection
Generates test data for fraud detection testing
"""

import logging
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class TestDataGenerator:
    """Generates test data for fraud detection"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize test data generator"""
        self.config = config or {}
        logger.info("TestDataGenerator initialized")
    
    def generate_transaction(self, transaction_type: str = "normal") -> Dict[str, Any]:
        """Generate a test transaction"""
        transaction = {
            "transaction_id": f"TXN-{random.randint(100000, 999999)}",
            "amount": round(random.uniform(10, 10000), 2),
            "currency": random.choice(["USD", "EUR", "GBP"]),
            "merchant": f"Merchant-{random.randint(1, 100)}",
            "timestamp": datetime.now().isoformat(),
            "user_id": f"USER-{random.randint(1000, 9999)}",
            "account_id": f"ACC-{random.randint(1000, 9999)}",
            "payment_method": random.choice(["card", "bank_transfer", "digital_wallet"])
        }
        
        if transaction_type == "suspicious":
            transaction["amount"] = random.uniform(50000, 100000)
            transaction["merchant"] = "Suspicious-Merchant"
        
        return transaction
    
    def generate_batch_transactions(self, count: int, fraud_ratio: float = 0.1) -> List[Dict[str, Any]]:
        """Generate batch of test transactions"""
        transactions = []
        fraud_count = int(count * fraud_ratio)
        
        for i in range(count):
            if i < fraud_count:
                transactions.append(self.generate_transaction("suspicious"))
            else:
                transactions.append(self.generate_transaction("normal"))
        
        random.shuffle(transactions)
        return transactions

if __name__ == "__main__":
    generator = TestDataGenerator()
    transaction = generator.generate_transaction()
    print(f"Generated transaction: {transaction}")
    
    batch = generator.generate_batch_transactions(10)
    print(f"Generated batch of {len(batch)} transactions")