#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:07:36 2021

@author: thomas_yang

找出並且刪除loss過大的影像
挑出並呈現 label有問題的影像

"""

import tensorflow as tf
# GPU settings
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


import numpy as np
from core.centernet import PostProcessing
from utils.visualize import visualize_training_results, visualize_training_results_step
import datetime
from configuration import Config
from data.dataloader import DetectionDataset, DataLoader
from core.models.mobilenet import MobileNetV2
from utils.show_traing_val_image import show_traing_val_image

import cv2

if __name__ == '__main__':

    # train/validation dataset
    val_dataset = DetectionDataset()
    val_data, val_size = val_dataset.generate_val_datatset()
    
    data_loader = DataLoader()

    # model
    centernet = MobileNetV2(training=True)
    load_weights_from_epoch = Config.load_weights_from_epoch
    if Config.load_weights_before_training:
        centernet.load_weights(filepath=Config.save_model_dir+"epoch-{}".format(load_weights_from_epoch))
        # centernet.load_weights(filepath=Config.save_model_dir+"epoch-{}.h5".format(load_weights_from_epoch), by_name=True)
        print("Successfully load weights!")
    else:
        load_weights_from_epoch = -1
    
    # metrics
    post_process = PostProcessing() 
    
    def val_step(model, batch_images, batch_labels):  
        pred = model(batch_images, False)
        total_loss, heatmap_loss, wh_loss, offset_loss, joint_loss, gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, gt_joint \
        = post_process.training_procedure(batch_labels=batch_labels, pred=pred)
        
        return pred, gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, gt_joint, total_loss, heatmap_loss, wh_loss, offset_loss, joint_loss
    
    #%%
    loss_dict = {}
    # validation part           
    for idx, val_batch_data in enumerate(val_data):
        print(idx, val_size)
        val_images, val_labels = data_loader.read_batch_data(val_batch_data, augment=False)  
        
        val_pred, val_gt_heatmap, _, _, _, _, val_gt_joint, \
        total_loss, heatmap_loss, wh_loss, offset_loss, joint_loss = val_step(centernet, val_images, val_labels)
        
        file_name = val_batch_data.numpy()[0].decode("utf-8").split(' ')[0]
        loss_dict[file_name] = total_loss.numpy()
        
        # show gt_heatmap and pre_heatpmap 
        # show_traing_val_image(val_images, val_gt_heatmap, val_gt_joint, val_pred, training = False)

#%%
import numpy as np
import cv2 
# np.save('loss_dict.npy', loss_dict) 
loss_dict2 = np.load('loss_dict.npy',allow_pickle='TRUE').item()  

mean = 0
for key in loss_dict2:
    mean += loss_dict2[key]
mean /= len(loss_dict2)

std = 0
for key in loss_dict2:
    std += (loss_dict2[key] - mean)**2    
std = np.sqrt(std/(len(loss_dict2)-1))
print(mean, std)


#%%
tmp1 = {}
with open('/home/thomas_yang/ML/CenterNet_TensorFlow2/txt_file/pose_detection/data_val.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        tmp1[line.split(' ')[0]] = line

filter_img = []
good_img = []
for key in loss_dict2:
    if loss_dict2[key] > (mean + std*0.3):
        filter_img.append(tmp1[key])
    else:
        good_img.append(tmp1[key])
print(len(filter_img))
print(len(good_img))

#%%
# circle_color = [(0, 0, 255), (0, 255, 0)]
# joint_color = [(255,0,255), (0,255,255)]
# circle_radius = [5, 7]
# text_list = ['R', 'L']
# for line in filter_img:
#     line_split = line.split()
#     img_name = line_split[0]
    # print(line)
    
#     img = cv2.imread(img_name)
#     for i in range(len(line_split[3:])//5):
#         xmin = int(float(line_split[3 + i*5 + 0]))
#         ymin = int(float(line_split[3 + i*5 + 1]))
#         xmax = int(float(line_split[3 + i*5 + 2]))
#         ymax = int(float(line_split[3 + i*5 + 3]))
#         class_id = int(float(line_split[3 + i*5 + 4]))
#         if class_id == 0:
#             cv2.rectangle(img=img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(250, 206, 135), thickness=2)
#         else :
#             class_id -= 1
#             joint_x = (xmin + xmax)//2
#             joint_y = (ymin + ymax)//2
#             cv2.circle(img=img, center=(joint_x, joint_y), 
#                         radius=circle_radius[class_id%2], 
#                         color=circle_color[class_id%2], thickness=-1)               
#             cv2.putText(img=img, text=text_list[(class_id)%2], org=(joint_x, joint_y), 
#                         fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
#                         color=joint_color[(class_id)%2], thickness=1)            
#             cv2.putText(img=img, text=str(class_id), org=(joint_x+10, joint_y), 
#                         fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
#                         color=(200,200,0), thickness=1)         

#     cv2.imshow('img', img)
#     if cv2.waitKey(0) == ord('q'):
#         break        

# cv2.destroyAllWindows()

#%%
count = 0    
for line in filter_img:
    if 'mpii' in line:
        count += 1
print(count)

count = 0    
for line in filter_img:
    if 'Crow' in line:
        count += 1
print(count)
 
count = 0    
for line in filter_img:
    if 'MHP' in line:
        count += 1
print(count)

count = 0    
for line in filter_img:
    if 'vive' in line:
        count += 1
print(count)

print(len(filter_img))
print(len(good_img))

#%%
circle_color = [(0, 0, 255), (0, 255, 0)]
joint_color = [(255,0,255), (0,255,255)]
circle_radius = [5, 7]
text_list = ['R', 'L']
count = 0
data_train_filter = []
for line in good_img:    
    line_split = line.split()
    check_negtive = False
    check_size = False
    for i in line_split[1:]:
        if '-' in i:
            check_negtive = True
   
    img_height = int(float(line_split[1]))
    img_width =  int(float(line_split[2]))   
    for i in range(len(line_split[3:])//5):
        xmin = int(float(line_split[3 + i*5 + 0]))
        ymin = int(float(line_split[3 + i*5 + 1]))
        xmax = int(float(line_split[3 + i*5 + 2]))
        ymax = int(float(line_split[3 + i*5 + 3]))
        class_id = int(float(line_split[3 + i*5 + 4]))
        if class_id == 0 and ((xmax- xmin) * (ymax-ymin)) / (img_height * img_width) > 0.5:            
            check_size = True
    if check_negtive :
        count += 1
        continue
        
    if  check_size :   
        count += 1
        # img_name = line_split[0]        
        # img = cv2.imread(img_name)                                
        # for i in range(len(line_split[3:])//5):
        #     xmin = int(float(line_split[3 + i*5 + 0]))
        #     ymin = int(float(line_split[3 + i*5 + 1]))
        #     xmax = int(float(line_split[3 + i*5 + 2]))
        #     ymax = int(float(line_split[3 + i*5 + 3]))
        #     class_id = int(float(line_split[3 + i*5 + 4]))
        #     if class_id == 0:
        #         cv2.rectangle(img=img, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(250, 206, 135), thickness=2)
        #     else :
        #         class_id -= 1
        #         joint_x = (xmin + xmax)//2
        #         joint_y = (ymin + ymax)//2
        #         cv2.circle(img=img, center=(joint_x, joint_y), 
        #                     radius=circle_radius[class_id%2], 
        #                     color=circle_color[class_id%2], thickness=-1)               
        #         cv2.putText(img=img, text=text_list[(class_id)%2], org=(joint_x, joint_y), 
        #                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
        #                     color=joint_color[(class_id)%2], thickness=1)            
        #         cv2.putText(img=img, text=str(class_id), org=(joint_x+10, joint_y), 
        #                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
        #                     color=(200,200,0), thickness=1)         

        # cv2.imshow('img', img)
        # print('got ', line)
        # if cv2.waitKey(0) == ord('q'):
        #     cv2.destroyAllWindows()
        #     break
        continue
    data_train_filter.append(line)        
print(count)
print('new', len(data_train_filter))
#%%
with open('/home/thomas_yang/ML/CenterNet_TensorFlow2/txt_file/pose_detection/data_train_filter_large_size.txt', 'w') as fw:     
    for i in data_train_filter:
        fw.write(i + '\n') 

print('old', len(tmp1))
print('new', len(data_train_filter))
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
    
 
