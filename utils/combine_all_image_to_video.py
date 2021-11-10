#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 12:05:41 2021

@author: thomas_yang
"""

import os
import cv2


img_dir = '/home/thomas_yang/ML/CenterNet_TensorFlow2/data/datasets/CrowPose/images/'
img_save_dir = '/home/thomas_yang/ML/CenterNet_TensorFlow2/data/datasets/CrowPose/512x512/'
video_save_dir = '/home/thomas_yang/ML/CenterNet_TensorFlow2/data/datasets/CrowPose/crowpose_512x512.avi'

img_list = os.listdir(img_dir)
img_list.sort()
img_read_list = [img_dir+i for i in img_list]
img_save_list = [img_save_dir+i for i in img_list]
target_size = 512

# cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_save_dir, fourcc, 20.0, (target_size,  target_size))
for idx, (i, j) in enumerate(zip(img_read_list,img_save_list)):
    print('{}/{}'.format(idx, len(img_list)))
    img = cv2.imread(i)
    
    img = cv2.resize(img, (target_size, target_size))
    cv2.imwrite(j, img)    
    out.write(img)
    
    # cv2.imshow('img', img)
    # if cv2.waitKey(1) == ord('q'):
    #     break

# cap.release()
out.release()  
cv2.destroyAllWindows()  

