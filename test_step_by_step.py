#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

print('=== STEP BY STEP IMPORT TEST ===')

try:
    print('1. Importing datetime...')
    from datetime import datetime
    print('   datetime imported successfully')
    
    print('2. Testing datetime.now()...')
    now = datetime.now()
    print(f'   datetime.now() returned: {now}')
    
    print('3. Testing datetime.now().isoformat()...')
    iso_str = now.isoformat()
    print(f'   isoformat() returned: {iso_str}')
    
    print('4. Importing financial_fraud_domain...')
    import financial_fraud_domain
    print('   financial_fraud_domain imported successfully')
    
    print('5. Checking if datetime is still available...')
    now2 = datetime.now()
    print(f'   datetime.now() still works: {now2}')
    
    print('6. Importing data_validator...')
    import financial_fraud_domain.data_validator
    print('   data_validator imported successfully')
    
except Exception as e:
    print(f'FAILED at step: {e}')
    import traceback
    traceback.print_exc() 