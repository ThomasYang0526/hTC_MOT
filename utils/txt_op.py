#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 12:53:18 2021

@author: thomas_yang
"""

def get_txt_combine(txt_file_path, txt_item, training = None):
    train_line = []
    for file in txt_item:
        with open(txt_file_path + file + '.txt', 'r') as f:
            for line in f.readlines():
                line = line.strip()
                train_line.append(line)
                
    train_flag={False:'data_val.txt', True:'data_train.txt'}
    with open(txt_file_path + train_flag[training], 'w') as fw:     
        for i in train_line:
            fw.write(i + '\n')      