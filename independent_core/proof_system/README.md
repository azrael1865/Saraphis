# Proof System Integration

This directory contains the proof system components integrated into the Saraphis Brain system.

## Components

### Core Engines
- **`rule_based_engine.py`** - Business rule-based proof generation
- **`ml_based_engine.py`** - Machine learning model proof validation  
- **`cryptographic_engine.py`** - Cryptographic integrity proofs
- **`confidence_generator.py`** - Confidence score aggregation
- **`algebraic_rule_enforcer.py`** - Mathematical constraint validation
- **`proof_integration_manager.py`** - Coordinates all proof engines

## Usage

### Basic Integration
The proof system is automatically integrated when the Brain is initialized with `enable_proof_system=True` (default).

```python
from independent_core.brain import Brain

# Proof system enabled by default
brain = Brain(enable_proof_system=True)

# Generate proof for a transaction
transaction = {
    'transaction_id': 'tx_001',
    'amount': 1000,
    'risk_score': 0.7
}

proof = brain.generate_proof(transaction)
```

### Individual Engine Usage
```python
from independent_core.proof_system import (
    RuleBasedProofEngine,
    MLBasedProofEngine, 
    CryptographicProofEngine
)

# Rule-based proof
rule_engine = RuleBasedProofEngine()
rule_result = rule_engine.evaluate_transaction(transaction)

# ML-based proof
ml_engine = MLBasedProofEngine()
ml_result = ml_engine.generate_ml_proof(transaction, model_state)

# Cryptographic proof
crypto_engine = CryptographicProofEngine()
crypto_result = crypto_engine.generate_proof(transaction)
```

### Gradient Validation
```python
import numpy as np

# Validate gradients during training
gradients = np.random.randn(10, 10)
validation_result = brain.validate_gradients(gradients, learning_rate=0.001)

if validation_result['valid']:
    print("Gradients are valid")
else:
    print(f"Gradient validation failed: {validation_result.get('error', 'Unknown error')}")
```

## Testing

Run the proof system tests:

```bash
# Run all proof system tests
python run_proof_tests.py

# Run only unit tests
python run_proof_tests.py --type unit

# Run with verbose output
python run_proof_tests.py --verbose
```

## Configuration

The proof system can be configured through the Brain configuration:

```python
brain_config = BrainSystemConfig(
    enable_proof_system=True,
    proof_system_config={
        'enable_rule_based_proofs': True,
        'enable_ml_based_proofs': True, 
        'enable_cryptographic_proofs': True,
        'fraud_detection_rules': True,
        'gradient_verification': True,
        'confidence_tracking': True,
        'algebraic_enforcement': True
    }
)

brain = Brain(config=brain_config)
```

## Integration with Financial Fraud Domain

The proof system integrates automatically with the financial fraud detection domain to provide:

- Rule-based fraud detection
- ML model validation 
- Transaction integrity verification
- Confidence scoring for fraud predictions
- Gradient validation during model training

## Performance

The proof system is designed for minimal overhead:
- Target: <10% performance impact
- Parallel proof generation
- Configurable engine selection
- Automatic fallback mechanisms

## Error Handling

The proof system includes comprehensive error handling:
- Graceful degradation when engines fail
- Automatic fallback to available engines
- Detailed error logging and reporting
- Recovery mechanisms for transient failures