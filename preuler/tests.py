#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:17:33 2020

@author: pioters
"""

varA=10
varB=2

if (type(varA)==str or type(varB)==str):
    print("string involved")
elif varA>varB:
    print("bigger")
elif varA==varB:
    print("equal")
elif varA<varB:
    print("smaller")
    
print ("Hello!")
n=10
while (n>0):
    print(n)
    n -= 2

ret=0
while (end>0):
    ret += end
    end -= 1

print(ret)

print ("Hello!")
for n in range(10,1,-2):
    print(n)
    
ret=0
for i in range(0,end+1):
    ret += i
print(ret)
    
