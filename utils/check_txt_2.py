#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 17:19:45 2021

@author: thomas_yang

確認 joint-heatmap 以及 location 是否正確
"""

import cv2
import numpy as np

txt_path = '/home/thomas_yang/ML/CenterNet_TensorFlow2/txt_file/pose_detection/'
txt_file = 'vive_land_autolabel_vote_7.txt'
save_txt_file = 'vive_land_autolabel_vote_7_.txt'
circle_color = [(0, 0, 255), (0, 255, 0)]
joint_color = [(255,0,255), (0,255,255)]
circle_radius = [5, 7]
text_list = ['R', 'L']

new_line_list = []
count = 0
with open(txt_path + txt_file, 'r') as f:
    for line in f.readlines():
        line_split = line.split()
        if np.sum('0' == np.array(line_split)) < 2:
            continue
        
        img_name = line_split[0]
        new_line = line_split[0:3]
        for i in range(0, len(line_split[3:])//5):
            xmin = ((line_split[3 + i*5 + 0]))
            ymin = ((line_split[3 + i*5 + 1]))
            xmax = ((line_split[3 + i*5 + 2]))
            ymax = ((line_split[3 + i*5 + 3]))
            class_id = ((line_split[3 + i*5 + 4]))
            if xmin == '0' and ymin == '0' and xmax == '24' and ymax == '24':
                count+=1
                continue
                print(count)
            new_line.append(xmin)
            new_line.append(ymin)
            new_line.append(xmax)
            new_line.append(ymax)
            new_line.append(class_id)
        # print(new_line) 
        if len(new_line) > 3:
            new_line_list.append(' '.join(new_line))

with open(txt_path + save_txt_file, 'w') as fw:     
    for i in new_line_list:
        fw.write(i + '\n')  


for line_idx, line in enumerate(new_line_list):
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
            cen_x = (xmin + xmax)//2
            cen_y = (ymin + ymax)//2
            cv2.circle(img=img, center=(cen_x, cen_y), 
                        radius=circle_radius[class_id%2], 
                        color=(255, 0, 255), thickness=-1)              
        else :
            class_id -= 1
            joint_x = (xmin + xmax)//2
            joint_y = (ymin + ymax)//2
            # cv2.circle(img=img, center=(joint_x, joint_y), 
            #             radius=circle_radius[class_id%2], 
            #             color=circle_color[class_id%2], thickness=-1) 
            cv2.line(img, (cen_x, cen_y), (joint_x, joint_y), circle_color[(class_id)%2], 2)
            cv2.putText(img=img, text=text_list[(class_id)%2], org=(joint_x, joint_y), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                        color=joint_color[(class_id)%2], thickness=1)            
            cv2.putText(img=img, text=str(class_id), org=(joint_x+10, joint_y), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                        color=(200,200,0), thickness=1)         
    
    img = cv2.resize(img, (960, 540))
    print(line_idx)
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break        

cv2.destroyAllWindows()        
        