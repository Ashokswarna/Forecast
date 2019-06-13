# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
a= 10
b =30
c = a + \
b

string_my = 'Sample string'
print (string_my)

import time
time()
ticks = time.localtime()
    
ticks

localtime = time.localtime(time.time())
print ("Local current time :", localtime)

def callme(strong):
    print('the passed string was :') , strong
    print (strong)
    return;

callme('the time for lunch')

from datetime import datetime, timedelta
def convert_to_date(date_string, date_format='%Y%W%w'):
    _ds = date_string + '1'
    dt = datetime.strptime(_ds, date_format)
    return dt

convert_to_date('201901', '20203102')

import add_sub_mul

#called a module and used it here
add_sub_mul.add_sub_mul(120, 10)

from fib import fibonacci
import os
os.getcwd()

my = input('Enter your input :')

print ('received input is'), my




