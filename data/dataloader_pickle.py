#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 18:48:45 2021

@author: thomas_yang
"""

import tensorflow as tf
import numpy as np

from configuration import Config
from utils.gaussian import gaussian_radius, draw_umich_gaussian

from PIL import Image
import cv2

# import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import sys
import pickle
import time

class DetectionDataset:
    def __init__(self):
        self.txt_file = '/home/thomas_yang/ML/CenterNet_TensorFlow2/txt_file/pose_detection_pickle/train.txt'
        self.val_txt_file = '/home/thomas_yang/ML/CenterNet_TensorFlow2/txt_file/pose_detection_pickle/val.txt'        
        self.batch_size = Config.batch_size
        self.val_batch_size = Config.batch_size

    @staticmethod
    def __get_length_of_dataset(dataset):
        length = 0
        for _ in dataset:
            length += 1
        return length

    def generate_datatset(self):
        dataset = tf.data.TextLineDataset(filenames=self.txt_file)
        length_of_dataset = DetectionDataset.__get_length_of_dataset(dataset)
        dataset = dataset.shuffle(length_of_dataset) # shuffle
        train_dataset = dataset.batch(batch_size=self.batch_size)
        return train_dataset, length_of_dataset

    def generate_val_datatset(self):
        dataset = tf.data.TextLineDataset(filenames=self.val_txt_file)
        length_of_dataset = DetectionDataset.__get_length_of_dataset(dataset)
        val_dataset = dataset.batch(batch_size=self.val_batch_size)
        return val_dataset, length_of_dataset

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class DataLoader:

    input_image_height = Config.get_image_size()[0]
    input_image_width = Config.get_image_size()[1]
    input_image_channels = Config.image_channels

    def __init__(self):
        self.max_boxes_per_image = Config.max_boxes_per_image
        self.seq = iaa.Sequential([
            iaa.Multiply((0.8, 1.2)),
            # iaa.GaussianBlur(sigma=(0, 1)),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255)),
            iaa.Affine(translate_px={"x": (-15, 15), "y": (-15, 15)}, scale=(0.6, 1.2)),
            iaa.Affine(rotate=(-10, 10)),
            iaa.AddToHueAndSaturation((-10, 10), per_channel=True),
            iaa.GammaContrast((0.75, 1.25)),
            # iaa.MultiplyHueAndSaturation(),
            iaa.MotionBlur(k=3),
            # iaa.Crop(percent=(0, 0.1)),
        ], random_order=True)  
        
        self.flip = iaa.Sequential([iaa.Fliplr(1.0)])


    def read_batch_data(self, batch_data, augment):
        batch_size = batch_data.shape[0]
        image_file_list = []
        boxes_list = []
        
        for n in range(batch_size):
            image_file, boxes = self.__get_image_information(single_line=batch_data[n])
            image_file_list.append(image_file)
            boxes_list.append(boxes)        
        boxes = np.stack(boxes_list, axis=0)
        
        image_tensor_list = []
        for idx, image in enumerate(image_file_list):
            if augment == False:
                image_tensor = image
            image_tensor_list.append(image_tensor)        
        images = tf.stack(values=image_tensor_list, axis=0)    
        return images, boxes

    def __get_image_information(self, single_line):                
        line_string = bytes.decode(single_line.numpy(), encoding="utf-8")
        image_name = line_string.replace('.pickle', '_image_file.pickle')
        with open(image_name, 'rb') as handle:
            gt_info = pickle.load(handle) 
        image_file = gt_info['image_value']
        return image_file, line_string

    @classmethod
    def image_preprocess(cls, is_training, image_dir):
        image_raw = tf.io.read_file(filename=image_dir)
        decoded_image = tf.io.decode_image(contents=image_raw, channels=DataLoader.input_image_channels, dtype=tf.dtypes.float32)
        decoded_image = tf.image.resize(images=decoded_image, size=(DataLoader.input_image_height, DataLoader.input_image_width))
        return decoded_image

    def show_train_data(self, images, labels, gt_heatmap, gt_wh):
        for i in range(images.shape[0]):
            img = (images[i].numpy()*255).astype(np.uint8)
            for j in range(labels[i].shape[0]):
                if labels[i, j, 4] != -1:
                    xmin, ymin, xmax, ymax, class_id = labels[i, j, :].astype(np.int)
                    x_c = (xmin + xmax)//2
                    y_c = (ymin + ymax)//2
                    w = gt_wh[i, j, 0]//2
                    h = gt_wh[i, j, 1]//2
                    xmin = int(x_c - w*Config.downsampling_ratio)
                    xmax = int(x_c + w*Config.downsampling_ratio)
                    ymin = int(y_c - h*Config.downsampling_ratio)
                    ymax = int(y_c + h*Config.downsampling_ratio)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    
                    text_list = ['R', 'L']
                    joint_color = [(255,0,0), (0,255,0)]
                    if class_id != 0 and (class_id-1) > 4:
                        cv2.putText(img=img, text=text_list[(class_id-1)%2], org=(xmin, ymin), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=joint_color[(class_id-1)%2], thickness=1)
                    elif class_id == 0:
                        cv2.putText(img=img, text='TL', org=(xmin, ymin), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 255), thickness=1)
                        cv2.putText(img=img, text='BR', org=(xmax, ymax), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 255), thickness=1)
            
            mix_img_hm = np.copy(img[:, :, ::-1])
            gt_hm = (cv2.resize((gt_heatmap[i]), (img.shape[0], img.shape[1])))
            mix_img_hm[gt_hm>0.1, :] = [255, 128, 64]
        
            cv2.imshow('train', img[:, :, ::-1])
            cv2.imshow('mix_img_hm', mix_img_hm)
            cv2.imshow('gt_heatmap', gt_heatmap[i]) 
            if cv2.waitKey(0) == ord('q'):
                cv2.destroyAllWindows()
                sys.exit()            

class GT:
    def __init__(self, batch_labels):
        self.downsampling_ratio = Config.downsampling_ratio # efficientdet: 8, others: 4
        self.features_shape = np.array(Config.get_image_size(), dtype=np.int32) // self.downsampling_ratio # efficientnet: 64*64
        self.batch_labels = batch_labels
        self.batch_size = batch_labels.shape[0]

    def get_gt_values(self):
       
        gt_heatmap = np.zeros(shape=(self.batch_size, self.features_shape[0], self.features_shape[1], Config.num_classes), dtype=np.float32)
        gt_reg = np.zeros(shape=(self.batch_size, Config.max_boxes_per_image, 2), dtype=np.float32)
        gt_wh = np.zeros(shape=(self.batch_size, Config.max_boxes_per_image, 2), dtype=np.float32)
        gt_reg_mask = np.zeros(shape=(self.batch_size, Config.max_boxes_per_image), dtype=np.float32)
        gt_indices  = np.zeros(shape=(self.batch_size, Config.max_boxes_per_image), dtype=np.float32)
        
        gt_joint = np.zeros(shape=(self.batch_size, self.features_shape[0], self.features_shape[1], Config.num_joints), dtype=np.float32)
        gt_joint_loc = np.zeros(shape=(self.batch_size, Config.max_boxes_per_image, 2*Config.num_joints), dtype=np.float32)
        gt_joint_reg_mask = np.zeros(shape=(self.batch_size, Config.max_boxes_per_image), dtype=np.float32)
        gt_joint_indices  = np.zeros(shape=(self.batch_size, Config.max_boxes_per_image), dtype=np.float32)        
        
        for i, pickle_ in enumerate(self.batch_labels):
            with open(pickle_, 'rb') as handle:
                gt_info = pickle.load(handle) 
            
            hm = gt_info['gt_heatmap']
            reg = gt_info['gt_reg']
            wh = gt_info['gt_wh']
            reg_mask = gt_info['gt_reg_mask']
            ind = gt_info['gt_indices']
            hm_joint = gt_info['gt_joint']
            joint_loc = gt_info['gt_joint_loc']
            joint_reg_mask = gt_info['gt_joint_reg_mask']
            joint_ind = gt_info['gt_joint_indices']           
            
            gt_heatmap[i, :, :, :] = hm
            gt_reg[i, :, :] = reg
            gt_wh[i, :, :] = wh
            gt_reg_mask[i, :] = reg_mask
            gt_indices[i, :] = ind
            gt_joint[i, :, :, :] = hm_joint
            gt_joint_loc[i, :, :] = joint_loc
            gt_joint_reg_mask[i, :] = joint_reg_mask
            gt_joint_indices[i, :] = joint_ind
            
        return gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, gt_joint, gt_joint_loc, gt_joint_reg_mask, gt_joint_indices

    
if __name__ == '__main__':
    train_dataset = DetectionDataset()
    train_data, train_size = train_dataset.generate_datatset()    
    data_loader = DataLoader()
    steps_per_epoch = tf.math.ceil(train_size / Config.batch_size)
    
    for step, batch_data in enumerate(train_data): 
        print(step, '------------------------', train_size)
        # print(batch_data)
        
        # load data 
        step_start_time = time.time()        
        images, labels = data_loader.read_batch_data(batch_data, augment = False)
        step_end_time = time.time()
        print('load', step_end_time - step_start_time)        
        
        img_rgb = np.array((images.numpy()*255).astype(np.uint8)[0][...,::-1])
        
        step_start_time = time.time()
        gt = GT(labels)
        gt_heatmap, gt_reg, gt_wh, gt_reg_mask, gt_indices, gt_joint, gt_joint_loc, gt_joint_reg_mask, gt_joint_indices = gt.get_gt_values()
        step_end_time = time.time()
        print('GT', step_end_time - step_start_time)
             
        gt_joint_max = (np.expand_dims(np.max(gt_joint[0], axis=-1), 2)*255).astype(np.uint8)
        gt_joint_max = cv2.resize(gt_joint_max, (Config.image_size["mobilenetv2"]))
 
        B, H, W, C = gt_joint.shape
        indice = gt_joint_indices[gt_joint_reg_mask==1]
        mask_gt_joint_loc = gt_joint_loc[gt_joint_reg_mask==1]

        topk_xs = (indice % W).astype(np.int)
        topk_ys = (indice // W).astype(np.int)
        
        for i in range(topk_xs.shape[0]):            
            cv2.circle(img_rgb, (topk_xs[i]*4, topk_ys[i]*4), 4, (0, 255, 0), -1)
            for j in range(Config.num_joints):
                x = int(mask_gt_joint_loc[i,j*2+0]*4+topk_xs[i]*4)
                y = int(mask_gt_joint_loc[i,j*2+1]*4+topk_ys[i]*4)
                cv2.line(img_rgb, (x, y), (topk_xs[i]*4, topk_ys[i]*4), (0, 255, 255), 1)
        
        
        cv2.imshow('img_ori', img_rgb)
        img_rgb[gt_joint_max>0,-1] = gt_joint_max[gt_joint_max>0]
        cv2.imshow('img', img_rgb)
        if cv2.waitKey(1) == ord('q'):
            break        

    cv2.destroyAllWindows()             