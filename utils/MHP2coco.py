#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 12:00:22 2021

@author: thomas_yang
"""

import scipy.io
import os
import cv2
import numpy as np

anno_path = '/home/thomas_yang/ML/CenterNet_TensorFlow2/data/datasets/MHP/val/pose_annos/'
image_path = '/home/thomas_yang/ML/CenterNet_TensorFlow2/data/datasets/MHP/val/images/'
save_name = 'MHP_val.txt'
image_format = '.jpg'
joint_target = [0,1,2,3,4,5,10,11,12,13,14,15]
mhp2coco={0:16,1:14,2:12,3:11,4:13,5:15,10:10,11:8,12:6,13:5,14:7,15:9}
circle_color = [(0, 0, 255), (0, 255, 0)]
joint_color = [(255,0,255), (0,255,255)]
circle_radius = [5, 7]
text_list = ['R', 'L']
w = 24
train_anno_list = os.listdir(anno_path)
train_anno_list.sort()

f = open(save_name, 'w')
for idx, anno in enumerate(train_anno_list):
    print(idx, len(train_anno_list))
    img_file = image_path + anno.split('.')[0] + image_format
    img = cv2.imread(img_file)
    
    mat = scipy.io.loadmat(anno_path + anno) 
    bboxes = []
    joints_bbox = []      
    for key in mat.keys():      
        if 'person' in key:           
            joint_int = mat[key].astype(np.int)
            for i in range(joint_int.shape[0]-2):
                if i in joint_target:                    
                    joint_x = joint_int[i, 0]
                    joint_y = joint_int[i, 1]
                    if joint_x == -1 or joint_y == -1:
                        continue
                    
                    class_id = mhp2coco[i]
                    joints_bbox.append(str(max(joint_x-w,0)))
                    joints_bbox.append(str(max(joint_y-w,0)))
                    joints_bbox.append(str(min(joint_x+w,img.shape[1])))
                    joints_bbox.append(str(min(joint_y+w,img.shape[0])))
                    joints_bbox.append(str(class_id + 1))                   
                    # cv2.circle(img=img, center=(joint_x, joint_y), 
                    #            radius=circle_radius[class_id%2], 
                    #            color=circle_color[class_id%2], thickness=-1)
                    # cv2.putText(img=img, text=text_list[(class_id)%2], org=(joint_x, joint_y), 
                    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                    #             color=joint_color[(class_id)%2], thickness=1)
            xmin = joint_int[18, 0]
            ymin = joint_int[18, 1]
            xmax = joint_int[19, 0]
            ymax = joint_int[19, 1]
            # cv2.rectangle(img=img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(250, 206, 135), thickness=2)
            
            bboxes.append(str(xmin))
            bboxes.append(str(ymin))
            bboxes.append(str(xmax))
            bboxes.append(str(ymax))
            bboxes.append(str(0))                
    
    record = img_file + ' ' + str(img.shape[0]) + ' ' + str(img.shape[1]) + ' ' + ' '.join(bboxes)+ ' ' + ' '.join(joints_bbox)
    record += '\n'
    # print(record)
    f.write(record)
    # cv2.imshow('img', img)
    # if cv2.waitKey(0) == ord('q'):
    #     break
f.close() 

