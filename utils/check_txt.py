#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 13:40:33 2021

@author: thomas_yang
"""

import cv2

txt_path = '/home/thomas_yang/ML/hTG_ObjectDetection/txt_file/pose_detection/'
txt_file = 'MOT_16.txt'
circle_color = [(0, 0, 255), (0, 255, 0)]
joint_color = [(255,0,255), (0,255,255)]
circle_radius = [5, 7]
text_list = ['R', 'L']

with open(txt_path + txt_file, 'r') as f:
    for line_idx, line in enumerate(f.readlines()):
        line_split = line.split()
        img_name = line_split[0]

        img = cv2.imread(img_name)
        for i in range(len(line_split[3:])//5):
            xmin = int(float(line_split[3 + i*5 + 0]))
            ymin = int(float(line_split[3 + i*5 + 1]))
            xmax = int(float(line_split[3 + i*5 + 2]))
            ymax = int(float(line_split[3 + i*5 + 3]))
            class_id = int(float(line_split[3 + i*5 + 4]))
            if class_id == 0:
                cv2.rectangle(img=img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(250, 206, 135), thickness=2)
            else :
                class_id -= 1
                joint_x = (xmin + xmax)//2
                joint_y = (ymin + ymax)//2
                cv2.circle(img=img, center=(joint_x, joint_y), 
                            radius=circle_radius[class_id%2], 
                            color=circle_color[class_id%2], thickness=-1)               
                cv2.putText(img=img, text=text_list[(class_id)%2], org=(joint_x, joint_y), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                            color=joint_color[(class_id)%2], thickness=1)            
                cv2.putText(img=img, text=str(class_id), org=(joint_x+10, joint_y), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                            color=(200,200,0), thickness=1)         
        
        img = cv2.resize(img, (512, 512))
        cv2.imshow('img', img)
        print(line_idx)
        if cv2.waitKey(1) == ord('q'):
            break        

cv2.destroyAllWindows()        
        