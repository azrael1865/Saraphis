#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

print('=== TESTING DATETIME ===')

try:
    from datetime import datetime
    print('datetime imported successfully')
    
    now = datetime.now()
    print(f'datetime.now() returned: {now}')
    print(f'type of datetime.now(): {type(now)}')
    
    iso_str = now.isoformat()
    print(f'isoformat() returned: {iso_str}')
    
except Exception as e:
    print(f'FAILED: {e}')
    import traceback
    traceback.print_exc()

print('\n=== TESTING DATA_VALIDATOR IMPORT ===')

try:
    from financial_fraud_domain.data_validator import ValidationIssue
    print('ValidationIssue imported successfully')
except Exception as e:
    print(f'FAILED: {e}')
    import traceback
    traceback.print_exc() 