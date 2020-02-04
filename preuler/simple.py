# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

s='uuincwftpzkukjxrss'

prev=''
out=''
tmp=''
for i in s: 
    if i >= prev: 
        tmp=tmp+i
    else:
        if len(tmp)>len(out):
            out=tmp
        tmp=i
    prev=i

if len(tmp)>len(out):
    out=tmp
print('Longest substring in alphabetical order is: '+ out)