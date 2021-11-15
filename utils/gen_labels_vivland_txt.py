#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:11:35 2021

@author: thomas_yang
"""

total_info = []
save_txt_dir = '/home/thomas_yang/ML/hTC_MOT/txt_file/pose_detection/vive_land_autolabel_vote_7_tid.txt'
load_txt_dir = '/home/thomas_yang/ML/hTC_MOT/txt_file/pose_detection/vive_land_autolabel_vote_7_.txt'
f = open(load_txt_dir)
for line in f.readlines():
    line = line.strip()
    line = line.split()
    info = line[0:3]    
    for i in range(len(line[3:])//5):
        xmin = line[3+ i*5+ 0]
        ymin = line[3+ i*5+ 1]
        xmax = line[3+ i*5+ 2]
        ymax = line[3+ i*5+ 3]
        classes = line[3+ i*5+ 4]
        if classes == '0':
            info += [xmin, ymin, xmax, ymax, classes, str(-1)]
    info = ' '.join(info)+ '\n'
    total_info.append(info)
    print(info)

with open(save_txt_dir, 'w') as f:
    for label_str in total_info:
        f.write(label_str)    