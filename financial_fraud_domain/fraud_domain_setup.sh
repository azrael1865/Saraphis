#!/bin/bash

# Financial Fraud Detection Domain Setup Script
# This script creates the complete directory structure and skeleton files

echo "Creating Financial Fraud Detection Domain Structure..."

# Set base directory
BASE_DIR="/home/will-casterlin/Desktop/Saraphis/financial_fraud_domain"
cd "$BASE_DIR" || exit 1

# Create main directory structure
echo "Creating directories..."
mkdir -p core
mkdir -p data
mkdir -p models
mkdir -p proof_system
mkdir -p symbolic
mkdir -p api
mkdir -p config
mkdir -p utils
mkdir -p tests
mkdir -p docs
mkdir -p integration
mkdir -p monitoring
mkdir -p deployment

# Create __init__.py files in all directories
echo "Creating __init__.py files..."
for dir in . core data models proof_system symbolic api config utils tests docs integration monitoring deployment; do
    touch "$dir/__init__.py"
done

# Create domain registration file
cat > domain_registration.py << 'EOF'
"""
Domain Registration for Financial Fraud Detection.
Registers the fraud detection domain with the Brain system.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import sys

# Add parent directory to path to import Brain components
sys.path.append(str(Path(__file__).parent.parent / "independent_core"))

from domain_registry import DomainConfig, DomainType
from brain import Brain


class FraudDomainRegistration:
    """Handles registration of the Financial Fraud Detection domain."""
    
    def __init__(self, brain: Optional[Brain] = None):
        """
        Initialize domain registration.
        
        Args:
            brain: Brain instance to register with
        """
        self.brain = brain
        self.logger = logging.getLogger(__name__)
        self.domain_name = "financial_fraud"
        self.domain_config = self._create_domain_config()
    
    def _create_domain_config(self) -> DomainConfig:
        """Create configuration for the fraud detection domain."""
        return DomainConfig(
            domain_type=DomainType.SPECIALIZED,
            description="Financial fraud detection and risk assessment",
            version="1.0.0",
            max_memory_mb=2048,
            max_cpu_percent=40.0,
            priority=9,  # High priority for fraud detection
            hidden_layers=[512, 256, 128, 64],
            activation_function="relu",
            dropout_rate=0.3,
            learning_rate=0.001,
            enable_caching=True,
            cache_size=500,
            enable_logging=True,
            log_level="INFO",
            shared_foundation_layers=3,
            allow_cross_domain_access=True,
            dependencies=["mathematics", "general"],
            author="Financial Fraud Detection Team",
            tags=["finance", "fraud", "security", "ml", "risk"]
        )
    
    def register(self) -> bool:
        """
        Register the fraud detection domain with the Brain system.
        
        Returns:
            True if registration successful, False otherwise
        """
        try:
            if not self.brain:
                self.logger.error("No Brain instance provided")
                return False
            
            # Register domain
            result = self.brain.register_domain(
                self.domain_name,
                self.domain_config
            )
            
            if result.get('success'):
                self.logger.info(f"Successfully registered {self.domain_name} domain")
                return True
            else:
                self.logger.error(f"Failed to register domain: {result.get('error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Registration error: {e}")
            return False
    
    def unregister(self) -> bool:
        """Unregister the domain from the Brain system."""
        try:
            if not self.brain:
                return False
            
            result = self.brain.deregister_domain(self.domain_name)
            return result.get('success', False)
        except Exception as e:
            self.logger.error(f"Unregistration error: {e}")
            return False


def register_fraud_domain(brain: Brain) -> bool:
    """
    Convenience function to register the fraud domain.
    
    Args:
        brain: Brain instance to register with
        
    Returns:
        True if successful, False otherwise
    """
    registration = FraudDomainRegistration(brain)
    return registration.register()
EOF

# Create core fraud detection file
cat > core/fraud_core.py << 'EOF'
"""
Core Fraud Detection Logic.
Main orchestrator for the fraud detection domain.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import json


class FraudDetectionCore:
    """Core fraud detection orchestrator."""
    
    def __init__(self, config_manager=None):
        """
        Initialize fraud detection core.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.logger = logging.getLogger(__name__)
        self.config_manager = config_manager
        
        # Initialize components
        self.data_loader = None
        self.preprocessor = None
        self.ml_predictor = None
        self.symbolic_reasoner = None
        self.proof_verifier = None
        
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all fraud detection components."""
        try:
            # Import components dynamically to avoid circular imports
            self.logger.info("Initializing fraud detection components...")
            
            # For now, use placeholder implementations
            self.data_loader = "DataLoader_Placeholder"
            self.preprocessor = "Preprocessor_Placeholder"
            self.ml_predictor = "MLPredictor_Placeholder"
            self.symbolic_reasoner = "SymbolicReasoner_Placeholder"
            self.proof_verifier = "ProofVerifier_Placeholder"
            
            self.logger.info("Fraud detection components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def detect_fraud(self, 
                    transaction: Union[Dict[str, Any], pd.DataFrame],
                    generate_proof: bool = True) -> Dict[str, Any]:
        """
        Detect fraud in transaction(s).
        
        Args:
            transaction: Single transaction dict or DataFrame of transactions
            generate_proof: Whether to generate verifiable proof
            
        Returns:
            Fraud detection results
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(transaction, dict):
                df = pd.DataFrame([transaction])
                single_transaction = True
            else:
                df = transaction
                single_transaction = False
            
            # Basic fraud detection logic (placeholder)
            results = []
            for _, row in df.iterrows():
                # Simple rule-based detection for now
                amount = row.get('amount', 0)
                hour = pd.to_datetime(row.get('timestamp', datetime.now())).hour
                
                # High amount at unusual hours = higher fraud score
                fraud_score = 0.0
                
                if amount > 5000:
                    fraud_score += 0.4
                if hour < 6 or hour > 22:
                    fraud_score += 0.3
                if amount > 10000:
                    fraud_score += 0.3
                
                result = {
                    'transaction_id': row.get('transaction_id', 'unknown'),
                    'fraud_score': min(fraud_score, 1.0),
                    'is_fraud': fraud_score >= 0.7,
                    'risk_level': self._get_risk_level(fraud_score),
                    'explanation': self._generate_explanation(fraud_score, amount, hour),
                    'timestamp': datetime.now().isoformat()
                }
                
                results.append(result)
            
            return results[0] if single_transaction else results
            
        except Exception as e:
            self.logger.error(f"Fraud detection failed: {e}")
            raise
    
    def _get_risk_level(self, score: float) -> str:
        """Convert fraud score to risk level."""
        if score >= 0.9:
            return 'critical'
        elif score >= 0.7:
            return 'high'
        elif score >= 0.5:
            return 'medium'
        elif score >= 0.3:
            return 'low'
        else:
            return 'very_low'
    
    def _generate_explanation(self, score: float, amount: float, hour: int) -> str:
        """Generate human-readable explanation."""
        explanations = []
        
        if amount > 10000:
            explanations.append("Very high transaction amount")
        elif amount > 5000:
            explanations.append("High transaction amount")
        
        if hour < 6 or hour > 22:
            explanations.append("Transaction at unusual hours")
        
        if not explanations:
            explanations.append("Normal transaction pattern")
        
        return "; ".join(explanations)
EOF

# Create data loader
cat > data/data_loader.py << 'EOF'
"""
Data Loader for Financial Fraud Detection.
Handles loading and preprocessing of transaction data.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import json


class TransactionDataLoader:
    """Loads and manages transaction data for fraud detection."""
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to data directory
        """
        self.data_path = data_path or Path(__file__).parent
        self.logger = logging.getLogger(__name__)
        self.transaction_cache = {}
        self.feature_columns = []
    
    def load_transactions(self, 
                         source: Union[str, Path, pd.DataFrame],
                         date_range: Optional[Tuple[datetime, datetime]] = None) -> pd.DataFrame:
        """
        Load transaction data from various sources.
        
        Args:
            source: Data source (file path, DataFrame, or data type string)
            date_range: Optional date range filter
            
        Returns:
            DataFrame with transaction data
        """
        try:
            # Load based on source type
            if isinstance(source, pd.DataFrame):
                df = source
            elif isinstance(source, (str, Path)):
                df = self._load_from_file(source)
            else:
                raise ValueError(f"Unsupported source type: {type(source)}")
            
            # Apply date filter if provided
            if date_range:
                df = self._filter_by_date(df, date_range)
            
            # Validate and clean data
            df = self._validate_transaction_data(df)
            
            self.logger.info(f"Loaded {len(df)} transactions")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load transactions: {e}")
            raise
    
    def _load_from_file(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load data from file based on extension."""
        filepath = Path(filepath)
        
        if filepath.suffix == '.csv':
            return pd.read_csv(filepath)
        elif filepath.suffix == '.parquet':
            return pd.read_parquet(filepath)
        elif filepath.suffix == '.json':
            return pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file type: {filepath.suffix}")
    
    def _validate_transaction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean transaction data."""
        required_columns = ['transaction_id', 'amount', 'timestamp', 'user_id']
        
        # Check required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean data
        df = df.dropna(subset=required_columns)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        return df
    
    def _filter_by_date(self, df: pd.DataFrame, 
                       date_range: Tuple[datetime, datetime]) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        start_date, end_date = date_range
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        return df[mask]
    
    def generate_sample_data(self, num_transactions: int = 1000) -> pd.DataFrame:
        """Generate sample transaction data for testing."""
        np.random.seed(42)
        
        data = {
            'transaction_id': [f'TX{i:06d}' for i in range(num_transactions)],
            'amount': np.random.lognormal(mean=5, sigma=1.5, size=num_transactions),
            'timestamp': pd.date_range(
                start='2024-01-01', 
                periods=num_transactions, 
                freq='5min'
            ),
            'user_id': [f'USER{np.random.randint(1, 1000):04d}' for _ in range(num_transactions)],
            'merchant_id': [f'MERCH{np.random.randint(1, 100):03d}' for _ in range(num_transactions)],
            'transaction_type': np.random.choice(
                ['purchase', 'withdrawal', 'transfer', 'deposit'], 
                size=num_transactions
            )
        }
        
        # Add some fraudulent transactions
        fraud_indices = np.random.choice(num_transactions, size=int(num_transactions * 0.02))
        for idx in fraud_indices:
            data['amount'][idx] *= 10  # Make fraudulent transactions larger
        
        return pd.DataFrame(data)
EOF

# Create API interface
cat > api/api_interface.py << 'EOF'
"""
API Interface for Financial Fraud Detection.
Provides endpoints for fraud detection services.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json


class FraudDetectionAPI:
    """API interface for fraud detection domain."""
    
    def __init__(self, fraud_core=None):
        """
        Initialize API interface.
        
        Args:
            fraud_core: FraudDetectionCore instance
        """
        self.fraud_core = fraud_core
        self.logger = logging.getLogger(__name__)
        self.request_count = 0
        
    def check_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check single transaction for fraud.
        
        Args:
            transaction_data: Transaction details
            
        Returns:
            Fraud check results
        """
        try:
            self.request_count += 1
            
            # Validate input
            validation_result = self._validate_transaction_input(transaction_data)
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'request_id': self._generate_request_id()
                }
            
            # Detect fraud
            result = self.fraud_core.detect_fraud(transaction_data)
            
            return {
                'success': True,
                'request_id': self._generate_request_id(),
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"API error: {e}")
            return {
                'success': False,
                'error': str(e),
                'request_id': self._generate_request_id()
            }
    
    def batch_check(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check multiple transactions for fraud."""
        try:
            results = []
            for transaction in transactions:
                result = self.check_transaction(transaction)
                results.append(result)
            
            return {
                'success': True,
                'batch_size': len(transactions),
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Batch check error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_fraud_score(self, transaction_id: str) -> Dict[str, Any]:
        """Get fraud score for a specific transaction."""
        # Implementation placeholder
        return {
            'transaction_id': transaction_id,
            'fraud_score': 0.0,
            'status': 'not_implemented'
        }
    
    def _validate_transaction_input(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Validate transaction input data."""
        required_fields = ['transaction_id', 'amount', 'timestamp', 'user_id']
        
        missing_fields = [f for f in required_fields if f not in transaction]
        if missing_fields:
            return {
                'valid': False,
                'error': f"Missing required fields: {missing_fields}"
            }
        
        return {'valid': True}
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return f"fraud_api_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.request_count}"
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Data processing
pyarrow>=6.0.0

# ML utilities
joblib>=1.1.0

# Validation and testing
pytest>=7.0.0

# API and web
fastapi>=0.75.0
uvicorn>=0.17.0

# Utilities
python-dotenv>=0.19.0
click>=8.0.0

# Development
black>=22.1.0
flake8>=4.0.0
EOF

# Create README.md
cat > README.md << 'EOF'
# Financial Fraud Detection Domain

An advanced fraud detection system that extends the Universal AI Core Brain system with specialized capabilities for detecting financial fraud in real-time.

## Overview

This domain provides:
- Real-time transaction fraud detection
- Machine learning ensemble models
- Rule-based detection patterns
- API endpoints for integration
- Comprehensive monitoring

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Register with Brain system:
```python
import sys
sys.path.append('../independent_core')

from brain import Brain
from domain_registration import register_fraud_domain

# Initialize Brain and register fraud domain
brain = Brain()
success = register_fraud_domain(brain)
print(f"Registration successful: {success}")
```

3. Use fraud detection:
```python
from core.fraud_core import FraudDetectionCore

# Initialize fraud detector
detector = FraudDetectionCore()

# Test with sample transaction
transaction = {
    'transaction_id': 'TEST001',
    'amount': 5000.00,
    'timestamp': '2024-01-15 03:30:00',
    'user_id': 'USER123',
    'merchant_id': 'MERCHANT456'
}

result = detector.detect_fraud(transaction)
print(f"Fraud Score: {result['fraud_score']:.2%}")
print(f"Risk Level: {result['risk_level']}")
```

## Architecture

- `core/` - Main fraud detection logic
- `data/` - Data loading and processing
- `api/` - API interfaces
- `domain_registration.py` - Brain system integration

## Integration with Brain System

This domain extends the Brain system by:
- Registering as DomainType.SPECIALIZED
- Using Brain's domain management
- Integrating with routing and state management
- Leveraging Brain's monitoring capabilities

## Development Status

This is the foundational structure. Core components are implemented with basic functionality and ready for enhancement.
EOF

# Create basic test file
cat > tests/test_fraud_detection.py << 'EOF'
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
EOF

echo "Financial Fraud Detection Domain structure created successfully!"
echo ""
echo "Files created:"
echo "├── domain_registration.py"
echo "├── core/fraud_core.py"
echo "├── data/data_loader.py"
echo "├── api/api_interface.py"
echo "├── tests/test_fraud_detection.py"
echo "├── requirements.txt"
echo "└── README.md"
echo ""
echo "Next steps:"
echo "1. Install dependencies: pip install -r requirements.txt"
echo "2. Run tests: python -m pytest tests/"
echo "3. Register with Brain system (see README.md)"
echo ""
echo "Ready for development and integration with the Brain system!"