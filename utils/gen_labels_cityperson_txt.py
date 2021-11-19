#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 18:09:25 2021

@author: thomas_yang
"""

import os
import cv2

save_txt_dir = '/home/thomas_yang/ML/hTC_MOT/txt_file/pose_detection/citypersons_val.txt'
load_txt_dirs = '/home/thomas_yang/ML/datasets/Citypersons/labels_with_ids/val/'
load_txt_dirs = [load_txt_dirs + i for i in os.listdir(load_txt_dirs)]
load_txt_dirs.sort()

txt_files = []
for load_txt_dir in load_txt_dirs:
    txt_file = [os.path.join(load_txt_dir, i) for i in os.listdir(load_txt_dir)]
    txt_file.sort()
    txt_files += txt_file

imgs_info = []
for idx, txt_fname in enumerate(txt_files):
    print('{:d}------{:d}'.format(idx+1, len(txt_files)))
    img_fname = txt_fname.replace('.txt', '.jpg').replace('labels_with_ids', 'images')
    if not os.path.isfile(img_fname):
        img_fname = txt_fname.replace('.txt', '.png').replace('labels_with_ids', 'images')
        
    img = cv2.imread(img_fname)
    img_height, img_width = img.shape[:2]    
    
    with open(txt_fname, 'r') as f:
        img_info = [img_fname, str(img_height), str(img_width)]
        for line_idx, line in enumerate(f.readlines()):
            line = line.strip().split()
            tid_curr = int(line[1])
            xcen = float(line[2]) * img_width
            ycen = float(line[3]) * img_height
            w = float(line[4]) * img_width
            h = float(line[5]) * img_height
            xmin = xcen - w/2
            ymin = ycen - h/2
            xmax = xcen + w/2
            ymax = ycen + h/2
            # label_str = '{:.2f} {:.2f} {:.2f} {:.2f} 0 {:d}'.format(xmin, ymin, xmax, ymax, tid_curr)
            label_str = '{:.2f} {:.2f} {:.2f} {:.2f} 0 -1'.format(xmin, ymin, xmax, ymax)
            img_info.append(label_str)
    img_info = ' '.join(img_info)+'\n'
    imgs_info.append(img_info)
    

with open(save_txt_dir, 'w') as f:
    for label_str in imgs_info:
        f.write(label_str)