#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:34:56 2021

@author: thomas_yang
"""

## Initialize
# from ochumanApi.ochuman import OCHuman
# # <Filter>: 
# #      None(default): load all. each has a bbox. some instances have keypoint and some have mask annotations.
# #            images: 5081, instances: 13360
# #     'kpt&segm' or 'segm&kpt': only load instances contained both keypoint and mask annotations (and bbox)
# #            images: 4731, instances: 8110
# #     'kpt|segm' or 'segm|kpt': load instances contained either keypoint or mask annotations (and bbox)
# #            images: 5081, instances: 10375
# #     'kpt' or 'segm': load instances contained particular kind of annotations (and bbox)
# #            images: 5081/4731, instances: 10375/8110
# ochuman = OCHuman(AnnoFile='/home/thomas_yang/ML/CenterNet_TensorFlow2/data/datasets/OCHuman/ochuman.json', Filter='kpt&segm')
# image_ids = ochuman.getImgIds()
# print ('Total images: %d'%len(image_ids))


# ochuman.toCocoFormart(subset='all', maxIouRange=(0.0, 1.0), save_dir='./')

# import json
# from collections import defaultdict
 
# f = open('/home/thomas_yang/ML/CenterNet_TensorFlow2/data/datasets/OCHuman/ochuman_coco_format_all_range_0.00_1.00.json',)
# data = json.load(f)
# f.close()

import json
import os

with open('/home/thomas_yang/ML/CenterNet_TensorFlow2/data/datasets/OCHuman/ochuman_coco_format_all_range_0.00_1.00.json', newline='') as jsonfile:
    data = json.load(jsonfile)
    # 或者這樣
    # data = json.loads(jsonfile.read())
    # print(data)
#%% save image path, height, width   
id_dict = {}
count = 0
for img in data['images']:
    img_path = '/home/thomas_yang/ML/CenterNet_TensorFlow2/data/datasets/OCHuman/images'
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
            if x != 0 and y != 0:
                height = int(id_dict[annotation['image_id']].split(' ')[1])
                width = int(id_dict[annotation['image_id']].split(' ')[2])
                xmin = int(max(x-w, 0))
                ymin = int(max(y-w, 0))
                xmax = int(min(x+w, width))
                ymax = int(min(y+w, height))
                new_key+=[str(xmin), str(ymin), str(xmax), str(ymax), str(j)]
        id_dict[annotation['image_id']] += ' ' + ' '.join(new_key)
            

#%% save txt    
with open('/home/thomas_yang/ML/CenterNet_TensorFlow2/txt_file/pose_detection/ochuman.txt', 'w') as fw:     
    for key in id_dict.keys():
        info = id_dict[key]
        fw.write(info + '\n')   