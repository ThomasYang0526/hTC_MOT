#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 17:55:38 2021

@author: thomas_yang

MPII dataset to coco txt

"""



from scipy import io
import cv2

"""
16    0 - r ankle, 
14    1 - r knee, 
12    2 - r hip, 
11    3 - l hip, 
13    4 - l knee, 
15    5 - l ankle, 
    6 - pelvis, 
    7 - thorax, 
    8 - upper neck, 
    9 - head top, 
10    10 - r wrist, 
 8    11 - r elbow, 
 6   12 - r shoulder, 
 5   13 - l shoulder, 
 7   14 - l elbow, 
 9   15 - l wrist
"""
import json
from collections import defaultdict
 
f = open('mpii_trainval.json',)
data = json.load(f)
f.close()
#%%
img_dict = defaultdict(list)
for info in data:
    print(info['image'])    
    image_name = info['image']
    img_dict[image_name].append(info)

#%%
mpii2coco = {0:16, 
             1:14,
             2:12,
             3:11,
             4:13,
             5:15,
             10:10,
             11:8,
             12:6,
             13:5,
             14:7,
             15:9}
f = open('MPII_data_list.txt', 'w')
for key_idx, key in enumerate(img_dict):
    print(key_idx, '/', len(img_dict))
    img_path = '/home/thomas_yang/ML/hTC_BodyPose/mpii_images/' + key
    img = cv2.imread(img_path)
    img_info = img_dict[key]
    
    bboxes = []
    joints_bbox = []
    for obj in img_info:
        xmin,ymin,xmax,ymax = img.shape[1], img.shape[0], 0, 0
        
        for point_idx, point in enumerate(obj['joints']):
            x = point[0]
            y = point[1]
            if x == -1 or y == -1:
                continue
            
            if point_idx in mpii2coco:
                w = 24
                coco_label = mpii2coco[point_idx]
                # cv2.circle(img, (int(x), int(y)), 5, (0 , 0, 255), -1)
                cv2.putText(img, '%d' %coco_label, (int(x), int(y)), 1, 1, (0,255,0))
                joints_bbox.append(str(max(x-w,0)))
                joints_bbox.append(str(max(y-w,0)))
                joints_bbox.append(str(min(x+w,img.shape[1])))
                joints_bbox.append(str(min(y+w,img.shape[0])))
                joints_bbox.append(str(coco_label + 1))
                # cv2.rectangle(img, (int(x-w), int(y-w)), (int(x+w), int(y+w)), (0 , 0, 255), 2)
                
            xmin = int(min(xmin, x))
            ymin = int(min(ymin, y))
            xmax = int(max(xmax, x))
            ymax = int(max(ymax, y))
        
        buffer = 0.1
        xminf = max(int(xmin - (xmax - xmin)*buffer), 0)
        yminf = max(int(ymin - (ymax - ymin)*buffer), 0)
        xmaxf = min(int(xmax + (xmax - xmin)*buffer), img.shape[1])
        ymaxf = min(int(ymax + (ymax - ymin)*buffer), img.shape[0])        
        # cv2.rectangle(img, (xminf, yminf), (xmaxf, ymaxf), (0 , 0, 255), 2)
        bboxes.append(str(xminf))
        bboxes.append(str(yminf))
        bboxes.append(str(xmaxf))
        bboxes.append(str(ymaxf))
        bboxes.append(str(0))
    # print(bboxes)  
    # print(joints_bbox)
    
    record = img_path + ' ' + str(img.shape[0]) + ' ' + str(img.shape[1]) + ' ' + ' '.join(bboxes)+ ' ' + ' '.join(joints_bbox)
    record += '\n'
    # print(record)
    f.write(record)
    # cv2.imshow('img', img)                
    # key = cv2.waitKey(0)
    # if  key == ord('q') : # 跳出
    #     flag = True
    #     cv2.destroyAllWindows()
    #     break             
f.close() 
    
    