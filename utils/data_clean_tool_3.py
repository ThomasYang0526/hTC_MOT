#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 20:32:00 2021

@author: thomas_yang

針對openpose-movenet給出的auto-label結果
進行frame之間的比對
刪除左右邊身體瞬間反轉的異常frame
 
"""

import cv2
import numpy as np
import sys
import collections

txt_path = '/home/thomas_yang/ML/CenterNet_TensorFlow2/txt_file/pose_detection/'
txt_file = 'vive_land_autolabel_vote_7.txt'
circle_color = [(0, 0, 255), (0, 255, 0)]
joint_color = [(255,0,255), (0,255,255)]
circle_radius = [5, 7]
text_list = ['R', 'L']

total_list = []
filter_list = []
with open(txt_path + txt_file, 'r') as f:
    for line in f.readlines():
        line = line.strip()
        total_list.append(line)

filter_list = []
for j in range(1, len(total_list)):
    pre_line = total_list[j-1]
    cur_line = total_list[j]
    pre_line_split = pre_line.split(' ')
    cur_line_split = cur_line.split(' ')

    pre_data = collections.defaultdict(list)
    for i in range(0, len(pre_line_split[3:])//5):
        xmin = int(float(pre_line_split[3 + i*5 + 0]))
        ymin = int(float(pre_line_split[3 + i*5 + 1]))
        xmax = int(float(pre_line_split[3 + i*5 + 2]))
        ymax = int(float(pre_line_split[3 + i*5 + 3]))
        class_id = int(float(pre_line_split[3 + i*5 + 4]))
        if class_id == 0:
            c_x =(xmin+xmax)//2
            c_y =(xmin+xmax)//2
            key = str(c_x) +'_'+ str(c_y)
            pre_data[key]=np.zeros((12, 2))
        else :
            class_id -= 6
            joint_x = (xmin + xmax)//2
            joint_y = (ymin + ymax)//2
            pre_data[key][class_id, 0] = joint_x
            pre_data[key][class_id, 1] = joint_y

    cur_data = collections.defaultdict(list)
    for i in range(0, len(cur_line_split[3:])//5):
        xmin = int(float(cur_line_split[3 + i*5 + 0]))
        ymin = int(float(cur_line_split[3 + i*5 + 1]))
        xmax = int(float(cur_line_split[3 + i*5 + 2]))
        ymax = int(float(cur_line_split[3 + i*5 + 3]))
        class_id = int(float(cur_line_split[3 + i*5 + 4]))
        if class_id == 0:
            c_x =(xmin+xmax)//2
            c_y =(xmin+xmax)//2
            key = str(c_x) +'_'+ str(c_y)
            cur_data[key]=np.zeros((12, 2))
        else :
            class_id -= 6
            joint_x = (xmin + xmax)//2
            joint_y = (ymin + ymax)//2
            cur_data[key][class_id, 0] = joint_x
            cur_data[key][class_id, 1] = joint_y
    
    flag = False
    for key_cur in cur_data:
        for key_pre in pre_data:
            cur_x = int(key_cur.split('_')[0])
            cur_y = int(key_cur.split('_')[1])
            pre_x = int(key_pre.split('_')[0])
            pre_y = int(key_pre.split('_')[1])
            if np.sqrt((cur_x-pre_x)**2 + (cur_y-pre_y)**2)<15:
                max_v = np.max(np.abs(cur_data[key_cur] - pre_data[key_pre]))
                if max_v > 15:
                    flag = True
                # print(max_v)
                # print(cur_data[key_cur] - pre_data[key_pre])
                # print('got')
    
    if flag or len(pre_data) != len(cur_data):
        continue
    if len(pre_data) != len(cur_data):
        print(len(pre_data), len(cur_data))
    filter_list.append(cur_line)
        
print(len(total_list), len(filter_list), len(total_list) - len(filter_list))
        
with open('/home/thomas_yang/ML/CenterNet_TensorFlow2/txt_file/pose_detection/' + txt_file, 'w') as fw:     
    for i in filter_list:
        fw.write(i + '\n')  