#!/usr/bin/env python3

import sys
sys.path.insert(0, '.')

print('=== DEBUGGING DATETIME ISSUE ===')

try:
    print('1. Importing datetime...')
    from datetime import datetime
    print(f'   datetime module: {datetime}')
    print(f'   datetime type: {type(datetime)}')
    
    print('2. Testing datetime.now...')
    now_func = datetime.now
    print(f'   datetime.now: {now_func}')
    print(f'   datetime.now type: {type(now_func)}')
    
    print('3. Testing datetime.now()...')
    now = now_func()
    print(f'   datetime.now() returned: {now}')
    print(f'   type of result: {type(now)}')
    
    print('4. Testing isoformat...')
    iso_func = now.isoformat
    print(f'   isoformat function: {iso_func}')
    print(f'   isoformat type: {type(iso_func)}')
    
    print('5. Testing isoformat()...')
    iso_str = iso_func()
    print(f'   isoformat() returned: {iso_str}')
    
    print('6. Testing lambda...')
    test_lambda = lambda: datetime.now().isoformat()
    print(f'   lambda: {test_lambda}')
    print(f'   lambda type: {type(test_lambda)}')
    
    print('7. Testing lambda execution...')
    result = test_lambda()
    print(f'   lambda result: {result}')
    
except Exception as e:
    print(f'FAILED: {e}')
    import traceback
    traceback.print_exc() 