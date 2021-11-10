#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 15:36:56 2021

@author: thomas_yang
"""


import json
import os

crowpose2coco={0:5,1:6,2:7,3:8,4:9,5:10,6:11,7:12,8:13,9:14,10:15,11:16}

with open('/home/thomas_yang/ML/CenterNet_TensorFlow2/data/datasets/CrowPose/drive-download-20210929T071327Z-001/crowdpose_test.json', newline='') as jsonfile:
    data = json.load(jsonfile)
    # 或者這樣
    # data = json.loads(jsonfile.read())
    # print(data)
#%% save image path, height, width   
id_dict = {}
count = 0
for img in data['images']:
    img_path = '/home/thomas_yang/ML/CenterNet_TensorFlow2/data/datasets/CrowPose/images'
    img_info = [os.path.join(img_path,img['file_name']), str(img['height']), str(img['width'])]
    id_dict[img['id']] = ' '.join(img_info)

#%% add bbox
count = 0
for annotation in data['annotations']:
    bbox = annotation['bbox']    
    xmin = int(bbox[0])
    ymin = int(bbox[1])
    xmax = int(xmin + bbox[2])
    ymax = int(ymin + bbox[3])    
    new_bbox = [str(xmin), str(ymin), str(xmax), str(ymax), '0']
    
#%% add joint
# for annotation in data['annotations']:
    if annotation['image_id'] in id_dict:
        keypoint = annotation['keypoints']
        # print(len(keypoint)//3)
        
        x_c = keypoint[0::3]
        y_c = keypoint[1::3]
        visible = keypoint[2::3]
        joints = [i for i in range(0,14)]
        
        w = 24
        new_key = []        
        for x, y, v, j in zip(x_c, y_c, visible, joints):
            if x != 0 and y != 0 and j in crowpose2coco:
                height = int(id_dict[annotation['image_id']].split(' ')[1])
                width = int(id_dict[annotation['image_id']].split(' ')[2])
                xmin = int(max(x-w, 0))
                ymin = int(max(y-w, 0))
                xmax = int(min(x+w, width))
                ymax = int(min(y+w, height))
                new_key+=[str(xmin), str(ymin), str(xmax), str(ymax), str(crowpose2coco[j] + 1)]
        if len(' '.join(new_key)) > 0:
            id_dict[annotation['image_id']] += ' ' + ' '.join(new_bbox)
            id_dict[annotation['image_id']] += ' ' + ' '.join(new_key)
            

#%% save txt    
with open('/home/thomas_yang/ML/CenterNet_TensorFlow2/txt_file/pose_detection/crowpose_test.txt', 'w') as fw:     
    for key in id_dict.keys():
        info = id_dict[key]
        fw.write(info + '\n')   
        
        
        