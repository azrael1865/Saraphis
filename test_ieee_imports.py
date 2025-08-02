#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

print('=== TESTING IEEE IMPORTS ===')

try:
    from financial_fraud_domain.training_integrator import IEEEFraudTrainingIntegrator
    print('SUCCESS: IEEEFraudTrainingIntegrator imported')
except Exception as e:
    print(f'FAILED: {e}')
    import traceback
    traceback.print_exc()

try:
    from financial_fraud_domain.enhanced_ml_models import ModelFactory, ModelType
    print('SUCCESS: ModelFactory, ModelType imported')
except Exception as e:
    print(f'FAILED: {e}')
    import traceback
    traceback.print_exc()

try:
    from financial_fraud_domain.enhanced_ml_framework import ModelConfig, BaseModel
    print('SUCCESS: ModelConfig, BaseModel imported')
except Exception as e:
    print(f'FAILED: {e}')
    import traceback
    traceback.print_exc() 