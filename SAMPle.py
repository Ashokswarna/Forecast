# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 17:29:13 2019

@author: ashok.swarna
"""

a = [1,2,3,4,5]
b = [6,7,8]

i = len(a)
j = len(b)

k = i+j
k = k - 1


y = 0
z = 0
x = 0
c=[]

while x <= k :
    if z < j:
        c.append(a[y])
        c.append(b[z])
        x += 2
        y += 1
        z += 1
    else:
        c.append(a[y])
        x += 1
        y += 1
        

def series_test (a):
    x = len(a)
    if a == 1:
        print('Is not stable')
    elif a == 2:
        print('Is not stable')
        
    all(number == a[0] for number in a)
    print ('is stable')
    
    d = a[1] - a[0]
    
    while i <= x:
        if (a[i] - a[i+1] != d):
            print('is not stable')
        else:
            print('is stable')
    
    
    
    
    
    