#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 13:32:08 2021

@author: thomas_yang
"""
import numpy as np

#%%
tmp1 = []
with open('./data_oid_htc(times)_coco.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        tmp1.append(line)

#%%
tmp2 = []
tmp3 = []
for i in tmp1:
    if np.random.rand() < 0.0030:
        if i not in tmp2:
            tmp2.append(i)
    else:
        tmp3.append(i)
        
#%%        
tmp4=[]
for i in tmp3:
    if i not in tmp2:
        tmp4.append(i)

print(len(tmp2)%16)
print(len(tmp4)%16)

#%%
with open('./new_train.txt', 'w') as fw:     
    for i in tmp4:
        fw.write(i + '\n')  

with open('./new_val.txt', 'w') as fw:     
    for i in tmp2:
        fw.write(i + '\n')          
#%%