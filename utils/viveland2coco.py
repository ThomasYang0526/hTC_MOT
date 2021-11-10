#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 20:08:42 2021

@author: thomas_yang

Viveland manual 500 image to coco txt

"""

import json
import os

with open('viveland_datasets-29_val2017.json', newline='') as jsonfile:
    data = json.load(jsonfile)
    # 或者這樣
    # data = json.loads(jsonfile.read())
    # print(data)
#%% save image path, height, width   
id_dict = {}
count = 0
for img in data['images']:
    if '_' in img['file_name']:
        img_path = '/home/thomas_yang/coco-annotator/datasets/viveland_datasets'
    # else:
    #     img_path = '/home/thomas_yang/coco-annotator/datasets/val2017'
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
    
    if annotation['image_id'] in id_dict:
        id_dict[annotation['image_id']] += ' ' + ' '.join(new_bbox)

#%% add joint
for annotation in data['annotations']:
    if annotation['image_id'] in id_dict:
        keypoint = annotation['keypoints']
        
        x_c = keypoint[0::3]
        y_c = keypoint[1::3]
        visible = keypoint[2::3]
        joints = [i for i in range(1,18)]
        
        w = 24
        new_key = []        
        for x, y, v, j in zip(x_c, y_c, visible, joints):
            if v == 2:
                xmin = int(max(x-w, 0))
                ymin = int(max(y-w, 0))
                xmax = int(min(x+w, 1920))
                ymax = int(min(y+w, 1080))
                new_key+=[str(xmin), str(ymin), str(xmax), str(ymax), str(j)]
        id_dict[annotation['image_id']] += ' ' + ' '.join(new_key)
            

#%% save txt    
with open('./data_vive_land.txt', 'w') as fw:     
    for key in id_dict.keys():
        info = id_dict[key]
        fw.write(info + '\n')   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        