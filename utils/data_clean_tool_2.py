#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 14:49:33 2021

@author: thomas_yang

刪除有異常座標的bbox 
比方xmax<xmin 物體太小被裁切異常
等等異常bbox
"""

import cv2
import numpy as np
import sys

txt_path = '/home/thomas_yang/ML/CenterNet_TensorFlow2/txt_file/pose_detection/'
txt_file = 'haggling_a3.txt'
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
        line_split = line.split()
        img_name = line_split[0]
        
        check = False
        for i in range(len(line_split[3:])//5):
            xmin = int(float(line_split[3 + i*5 + 0]))
            ymin = int(float(line_split[3 + i*5 + 1]))
            xmax = int(float(line_split[3 + i*5 + 2]))
            ymax = int(float(line_split[3 + i*5 + 3]))
            class_id = int(float(line_split[3 + i*5 + 4]))
            h = ymax - ymin
            w = xmax - xmin
            if class_id == 0:
                if ymax < ymin or xmax<xmin or h < 20 or w < 20\
                    or xmax > 1920 or xmin > 1920\
                    or xmax < 0 or xmin < 0\
                    or ymax > 1080 or ymin > 1080\
                    or ymax < 0 or ymin < 0\
                    or h < 450:
                    check = True
                    img = cv2.imread(img_name)
                    cv2.rectangle(img=img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(250, 206, 135), thickness=2)
                    print('got', xmin, xmax, ymin, ymax)
                    img = cv2.resize(img, (960, 540))
                    cv2.imshow('img', img)
                    if cv2.waitKey(1) == ord('q'):
                        sys.exit()   
        if check:
            continue
        filter_list.append(line)

print(len(total_list), len(filter_list), len(total_list) - len(filter_list))
        
with open('/home/thomas_yang/ML/CenterNet_TensorFlow2/txt_file/pose_detection/haggling_a3_.txt', 'w') as fw:     
    for i in filter_list:
        fw.write(i + '\n')                       

#%%
# min_area =  np.Inf
# max_area = -np.Inf

# with open(txt_path + txt_file, 'r') as f:
#     for line in f.readlines():
#         line_split = line.split()
#         img_name = line_split[0]
        
#         for i in range(len(line_split[3:])//5):
#             xmin = int(float(line_split[3 + i*5 + 0]))
#             ymin = int(float(line_split[3 + i*5 + 1]))
#             xmax = int(float(line_split[3 + i*5 + 2]))
#             ymax = int(float(line_split[3 + i*5 + 3]))
#             class_id = int(float(line_split[3 + i*5 + 4]))
#             if class_id == 0:
#                 h = ymax - ymin
#                 w = xmax - xmin
#                 if h*w < min_area:
#                     img = cv2.imread(img_name)                    
#                     min_h = h
#                     min_w = w
#                     min_area = w*h
#                     # print(xmin, xmax, ymin, ymax)
#                     # print(min_h, min_w)
#                     # cv2.rectangle(img=img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(250, 206, 135), thickness=2)
#                     # img = cv2.resize(img, (416, 416))
#                     # cv2.imshow('img', img)
#                     # if cv2.waitKey(0) == ord('q'):
#                     #     sys.exit()   
#                 elif h*w > max_area:
#                     img = cv2.imread(img_name)
#                     max_h = h
#                     max_w = w                
#                     max_area = w*h
#                     # print('max', xmin, xmax, ymin, ymax)
#                     # print('max', max_h, max_w)
#                     # print(line)
#                     # cv2.rectangle(img=img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(250, 206, 135), thickness=2)
#                     # img = cv2.resize(img, (416, 416))
#                     # cv2.imshow('img', img)
#                     # if cv2.waitKey(0) == ord('q'):
#                     #     sys.exit()                     
# print(min_h, min_w, max_h, max_w)                    
     

# # cv2.destroyAllWindows() 