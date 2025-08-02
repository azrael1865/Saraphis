#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

print('=== TESTING MINIMAL IMPORT CHAIN ===')

try:
    print('1. Importing datetime directly...')
    from datetime import datetime
    print('   datetime imported successfully')
    
    print('2. Testing datetime.now()...')
    now = datetime.now()
    print(f'   datetime.now() returned: {now}')
    
    print('3. Testing datetime.now().isoformat()...')
    iso_str = now.isoformat()
    print(f'   isoformat() returned: {iso_str}')
    
    print('4. Importing data_validator...')
    import financial_fraud_domain.data_validator
    print('   data_validator imported successfully')
    
except Exception as e:
    print(f'FAILED: {e}')
    import traceback
    traceback.print_exc() 