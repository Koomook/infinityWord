import os
os.environ['KEY'] = 'VALUE'

event = {'category':'Russian', 'start_letters':'RUS'}

from lambda_function import lambda_handler
response = lambda_handler(event=event, context=None)
print('response', response)