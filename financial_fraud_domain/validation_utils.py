"""
Validation Utilities for Financial Fraud Detection
Validation utilities and helpers
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class ValidationUtils:
    """Validation utilities"""
    
    def __init__(self):
        """Initialize validation utilities"""
        logger.info("ValidationUtils initialized")
    
    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def validate_amount(self, amount: float) -> Tuple[bool, str]:
        """Validate transaction amount"""
        if amount < 0:
            return False, "Amount cannot be negative"
        if amount > 1000000:
            return False, "Amount exceeds maximum limit"
        return True, "Valid amount"
    
    def validate_currency(self, currency: str) -> bool:
        """Validate currency code"""
        valid_currencies = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]
        return currency.upper() in valid_currencies
    
    def validate_required_fields(self, data: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
        """Validate required fields are present"""
        missing_fields = [field for field in required_fields if field not in data or data[field] is None]
        return len(missing_fields) == 0, missing_fields

if __name__ == "__main__":
    utils = ValidationUtils()
    print("Validation utilities initialized")