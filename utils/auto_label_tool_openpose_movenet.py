#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 12:54:56 2021

@author: thomas_yang
"""

import cv2
import numpy as np
import sys
import os

open2coco = {2:6, 
             3:8,
             4:10,
             5:5,
             6:7,
             7:9,
             9:12,
             10:14,
             11:16,
             12:11,
             13:13,
             14:15}

video_path = '/home/thomas_yang/Downloads/Viveland-records-20210422'
video_1 = 'vlc-record-2021-04-22-15h48m40s-rtsp___192.168.102.3_8554_fhd-.mp4'
video_2 = 'vlc-record-2021-04-22-16h00m58s-rtsp___192.168.102.3_8554_fhd-.mp4'
video_3 = 'vlc-record-2021-04-22-16h05m31s-rtsp___192.168.102.3_8554_fhd-.mp4'
video_4 = 'vlc-record-2021-04-22-16h12m47s-rtsp___192.168.102.3_8554_fhd-.mp4'
video_5 = 'vlc-record-2021-04-22-16h23m28s-rtsp___192.168.102.3_8554_fhd-.mp4'
video_6 = 'vlc-record-2021-04-22-16h31m33s-rtsp___192.168.102.3_8554_fhd-.mp4'
video_7 = 'vlc-record-2021-04-22-17h02m58s-rtsp___192.168.102.3_8554_fhd-.mp4'
#%%
# cap = cv2.VideoCapture(os.path.join(video_path, video_7))
# length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# count = 0
# while cap.isOpened():
#     print(count, length)
#     ret, frame = cap.read()
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break       
#     cv2.imwrite('/home/thomas_yang/ML/CenterNet_TensorFlow2/data/datasets/Viveland/video_7/%08d.jpg' %count, frame)
#     count+=1
       
# cap.release()     

#%%
import tensorflow as tf
model_name = "movenet_thunder_f16.tflite"

input_size = 256
# Initialize the TFLite interpreter
interpreter = tf.lite.Interpreter(model_path="/home/thomas_yang/ML/CenterNet_TensorFlow2/model.tflite")
interpreter.allocate_tensors()

def movenet(input_image):
  # TF Lite format expects tensor type of uint8.
  input_image = tf.cast(input_image, dtype=tf.uint8)
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
  # Invoke inference.
  interpreter.invoke()
  # Get the model prediction.
  keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
  return keypoints_with_scores

#%%

circle_color = [(0, 0, 255), (0, 255, 0)]
joint_color = [(255,0,255), (0,255,255)]
circle_radius = [5, 7]
text_list = ['R', 'L']

import pickle5 as pickle
import os
# pickle_file = '/home/thomas_yang/ML/CenterNet_TensorFlow2/data/datasets/Viveland/output_/' + video_7.replace('mp4', 'pickle')
txt_path = '/home/thomas_yang/ML/CenterNet_TensorFlow2/txt_file/pose_detection/mhp_autolabel_vote.txt'
pickle_file = '/home/thomas_yang/ML/CenterNet_TensorFlow2/data/datasets/MHP/mhp_512x512.pickle'
img_dir = '/home/thomas_yang/ML/CenterNet_TensorFlow2/data/datasets/MHP/512x512/'
img_list = os.listdir(img_dir)
img_list.sort()
img_list = [img_dir + i for i in img_list]

objects = []
with (open(pickle_file, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
objects = objects[0] 
info_list = []       

got = 0
for idx, persons in enumerate(objects): 
    if type(persons) == type(None):
        continue
    img_path = img_list[idx]
    img = cv2.imread(img_path)
    img_copy = np.copy(img)
    info = ' '.join([img_path, str(img.shape[0]), str(img.shape[1])]) + ' '
    for person in persons:
        # if np.sum(person[:,0]==0) > 3:
        if np.sum(person[[2,3,4,5,6,7,9,10,11,12,13,14],0] == 0) > 3: 
            continue
        open_array = np.zeros((12, 2))     
        xmin, ymin, _ = np.min(person[person[:,0] > 0].astype(np.int), 0) - [25, 35, 0]
        xmax, ymax, _ = np.max(person[person[:,0] > 0].astype(np.int), 0) + [25, 25, 0]
        for joint in open2coco: 
            class_id = open2coco[joint]
            joint_x = int(person[joint, 0])
            joint_y = int(person[joint, 1])
            open_array[class_id-5] = [person[joint, 0], person[joint, 1]]
            cv2.circle(img=img_copy, center=(joint_x, joint_y), 
                        radius=circle_radius[class_id%2], 
                        color=circle_color[class_id%2], thickness=-1)
        xmin = int(max(0, xmin))
        ymin = int(max(0, ymin))
        xmax = int(min(img.shape[1], xmax))
        ymax = int(min(img.shape[0], ymax))
        h, w = ymax-ymin, xmax-xmin
        
        cv2.rectangle(img=img_copy, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(250, 206, 135), thickness=2)
        crop = np.copy(img[ymin:ymax, xmin:xmax, :])

        input_image = tf.expand_dims(crop[...,::-1], axis=0)
        input_image = tf.image.resize(input_image, (input_size, input_size))
        input_image_np = np.array(input_image.numpy().astype(np.uint8)[...,::-1][0])
        move_array = movenet(input_image)
        move_array_nor = (np.squeeze(move_array)[:,:2] * [h, w] + [ymin, xmin])[5:, :]
        move_array_nor = move_array_nor[...,::-1]
        for joint in range(move_array_nor.shape[0]): 
            class_id = joint+5
            joint_x = int(move_array_nor[joint, 0])
            joint_y = int(move_array_nor[joint, 1])
            cv2.circle(img=img_copy, center=(joint_x, joint_y), 
                        radius=15, 
                        color=circle_color[class_id%2], thickness= 2)
        
        dis = np.sqrt(np.sum((move_array_nor - open_array)**2, 1))
        dis_flag = dis < 15
        if np.sum(dis_flag) < 10:
            continue
        
        info += ' '.join([str(xmin), str(ymin), str(xmax), str(ymax), str(0)]) + ' '
        for flag_idx, flag in enumerate(dis_flag):
            w = 24
            if flag:                
                joint_x = int((move_array_nor[flag_idx, 0] + open_array[flag_idx, 0])/2)
                joint_y = int((move_array_nor[flag_idx, 1] + open_array[flag_idx, 1])/2)
            else :                
                joint_x = int((open_array[flag_idx, 0]))
                joint_y = int((open_array[flag_idx, 1]))                
            cxmin = int(max(joint_x-w, 0))
            cymin = int(max(joint_y-w, 0))
            cxmax = int(min(joint_x+w, img.shape[1]))
            cymax = int(min(joint_y+w, img.shape[0]))
            cv2.circle(img=img_copy, center=(joint_x, joint_y), 
                        radius=3, 
                        color=(0, 255, 255), thickness= -1)
            if cxmin==0 and cymin==0 and cxmax==w and cymax==w:
                continue
            info += ' '.join([str(cxmin), str(cymin), str(cxmax), str(cymax), str(flag_idx+6)]) + ' '        
    
    if len(info.split(' ')) > 4:
        info_list.append(info)
        got += 1
        
    img_copy = cv2.resize(img_copy, (512, 512))
    cv2.imshow('img_copy', img_copy)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break
    print(len(objects), idx, got)
                             
with open(txt_path, 'w') as fw:     
    for i in info_list:
        fw.write(i + '\n')   